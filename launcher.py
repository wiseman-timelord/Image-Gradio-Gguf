#!/usr/bin/env python3
"""
launcher.py - Startup, shutdown, and main loop.
Activated from the venv by the batch menu (option 1).

The Gradio UI is served on a local-only loopback port as before, but
instead of opening it in the user's default web browser, this launcher
hosts it inside a PyQt6 window (QWebEngineView). To the user this looks
and behaves like a normal desktop application window — title bar, resize,
minimize, taskbar icon — even though the content is rendered by an
embedded Chromium browser engine pointed at our own local server.

Shutdown is unified: both the window's native "X" close button and the
in-page "Exit Program" button (clicked inside Gradio, which runs on a
server worker thread, not the Qt GUI thread) end up calling the exact
same _shutdown() sequence, exactly once, on the Qt main thread. That
sequence saves window geometry, closes the Gradio server, closes the Qt
window, and returns control to the calling batch script.
"""

from __future__ import annotations

import platform
import socket
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    import gradio as gr
except ImportError:
    print("ERROR: Gradio is not installed.")
    print("Run option 2 (Installation) from the batch menu first.")
    sys.exit(1)

_major = int(gr.__version__.split(".")[0])
if _major < 6:
    print(f"WARNING: Gradio {gr.__version__} detected. This app targets Gradio 6.x.")

try:
    from PyQt6.QtCore import QUrl, QObject, pyqtSignal
    from PyQt6.QtGui import QIcon
    from PyQt6.QtWidgets import QApplication, QMainWindow
    from PyQt6.QtWebEngineWidgets import QWebEngineView
except ImportError:
    print("ERROR: PyQt6 (and PyQt6-WebEngine) are not installed.")
    print("Run option 2 (Installation) from the batch menu first.")
    sys.exit(1)

import scripts.configure as configure
import scripts.utilities as utilities
import scripts.display as display

APP_TITLE = "Image-Gradio-Gguf"
SERVER_NAME = "127.0.0.1"
SERVER_PORT = 7860


def _print_banner() -> None:
    cpu = configure.get_cpu_info()
    vk  = configure.get_vulkan_info()
    bs  = utilities.get_build_status()
    cfg = configure.load_configuration()

    print(f"  Versioning: Python {platform.python_version()}; Gradio {gr.__version__}")
    print(f"\n  CPU     : {cpu['brand']}")
    print(f"  Threads : {cpu['default_threads']} (85% of {cpu['cores_logical']} logical cores)")
    print(f"  Vulkan  : {vk['available']}  ({vk['version']})")
    for d in vk["devices"]:
        print(f"    GPU{d['index']}: {d['name']}")

    if not bs["llama_built"] or not bs["sd_built"]:
        print("\n  NOTE: Backends not yet built.")
        print("        Run option 2 (Installation) from the batch menu.")
    else:
        print(f"\n  llama-cli : {bs['llama_path']}")
        print(f"  sd        : {bs['sd_path']}")

    enc  = cfg.get("encoder_model_path", "")
    diff = cfg.get("imagegen_model_path", "")
    vae  = cfg.get("vae_model_path", "")
    print(f"\n  Encoder  : {'OK — ' + enc   if enc  and Path(enc).exists()  else 'NOT SET'}")
    print(f"  Diffusion: {'OK — ' + diff  if diff and Path(diff).exists() else 'NOT SET'}")
    print(f"  VAE      : {'OK — ' + vae   if vae  and Path(vae).exists()  else 'NOT SET'}")
    print()


def _wait_for_server(host: str, port: int, timeout: float = 20.0) -> bool:
    """Poll the loopback socket until the Gradio server is accepting
    connections (or timeout). Avoids a race where the QWebEngineView loads
    the URL before uvicorn has finished binding in its background thread."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.1)
    return False


class _ExitBridge(QObject):
    """Thread-safe bridge so the in-page 'Exit Program' button (which runs
    on a Gradio/Starlette worker thread) can ask the Qt GUI thread to close
    the window. Qt signals emitted from a non-GUI thread are automatically
    queued and delivered on the receiving object's thread, so this is the
    correct way to reach into Qt from another thread — calling window
    methods directly from a worker thread is not safe."""
    close_requested = pyqtSignal()


class AppWindow(QMainWindow):
    """A QMainWindow hosting a QWebEngineView, dressed up as a standalone
    desktop app rather than a browser tab. Keeps the native title bar
    (resizable / minimizable / taskbar-friendly) but hides browser chrome
    (no address bar, no tabs, no bookmarks) since we only ever load our own
    local server's URL — there's nothing else for the user to navigate to.

    Both the native title-bar "X" button and the in-page "Exit Program"
    button funnel into closeEvent() exactly once, which is where geometry
    gets saved and the rest of the shutdown sequence runs."""

    def __init__(self, url: str, geometry: dict, on_close) -> None:
        super().__init__()
        self._on_close = on_close
        self._closed_once = False

        self.setWindowTitle(APP_TITLE)
        self._apply_saved_geometry(geometry)

        icon_path = configure.get_media_dir() / "program_icon.ico"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))

        self.view = QWebEngineView(self)
        self.setCentralWidget(self.view)
        self.view.load(QUrl(url))

        # Bridge for cross-thread close requests (see _ExitBridge docstring).
        self.exit_bridge = _ExitBridge()
        self.exit_bridge.close_requested.connect(self.close)

    def _apply_saved_geometry(self, geometry: dict) -> None:
        self.resize(geometry["width"], geometry["height"])
        if geometry["x"] != configure.WINDOW_GEOMETRY_UNSET and \
           geometry["y"] != configure.WINDOW_GEOMETRY_UNSET:
            self.move(geometry["x"], geometry["y"])
        if geometry["maximized"]:
            self.showMaximized()

    def current_geometry(self) -> dict:
        """Read back the window's current position/size/maximized state for
        saving. When maximized, normalGeometry() gives the pre-maximize
        rect — that's what we want, so un-maximizing later restores to a
        sane size/position instead of (0, 0)."""
        maximized = self.isMaximized()
        rect = self.normalGeometry() if maximized else self.geometry()
        return {
            "x": rect.x(), "y": rect.y(),
            "width": rect.width(), "height": rect.height(),
            "maximized": maximized,
        }

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        if self._closed_once:
            event.accept()
            return
        self._closed_once = True
        event.accept()
        self._on_close(self.current_geometry())


def _shutdown(window_geometry: dict, gradio_app) -> None:
    """Single shutdown sequence used by both the window's 'X' button and
    the in-page 'Exit Program' button. Saves window geometry, closes the
    Gradio server, then exits the process so the calling batch script
    regains control and returns to its menu (matching prior behavior)."""
    print("\n  Shutting down...")
    try:
        configure.save_window_geometry(
            x=window_geometry["x"], y=window_geometry["y"],
            width=window_geometry["width"], height=window_geometry["height"],
            maximized=window_geometry["maximized"],
        )
        print("  Window geometry saved.")
    except Exception as e:
        print(f"  WARNING: failed to save window geometry: {e}")

    try:
        gradio_app.close(verbose=False)
        print("  Gradio server closed.")
    except Exception as e:
        print(f"  WARNING: error closing Gradio server: {e}")

    print("  Goodbye.")
    import os
    os._exit(0)


def main() -> None:
    configure.ensure_data_dirs()
    _print_banner()
    blocks_app, _css = display.build_app()

    # Suppress the Starlette deprecation warning from Gradio internals
    import warnings
    warnings.filterwarnings(
        "ignore",
        message=".*HTTP_422_UNPROCESSABLE_ENTITY.*"
    )

    # prevent_thread_lock=True starts the Gradio/uvicorn server on a
    # background thread and returns immediately instead of blocking, so we
    # can hand control to the Qt event loop afterwards. inbrowser is left
    # False — the local URL is never opened in the system web browser; only
    # the embedded QWebEngineView ever points at it.
    _server_app, local_url, _share_url = blocks_app.launch(
        server_name=SERVER_NAME,
        server_port=SERVER_PORT,
        share=False,
        inbrowser=False,
        prevent_thread_lock=True,
        show_error=True,
        theme=gr.themes.Soft(),
        css=_css,
    )

    if not _wait_for_server(SERVER_NAME, SERVER_PORT):
        print("ERROR: Gradio server did not come up in time.")
        sys.exit(1)

    qt_app = QApplication(sys.argv)
    qt_app.setApplicationName(APP_TITLE)

    saved_geometry = configure.get_window_geometry()
    window = AppWindow(
        local_url,
        saved_geometry,
        on_close=lambda geom: _shutdown(geom, blocks_app),
    )

    # The in-page "Exit Program" button (display.py) runs on a Gradio
    # worker thread. It calls this function, which only does the
    # thread-safe thing: emit a signal asking the GUI thread to close the
    # window. The actual save-and-shutdown work happens once, in
    # closeEvent, regardless of which path triggered it.
    def _request_exit_from_gradio_thread() -> None:
        window.exit_bridge.close_requested.emit()

    display.set_exit_handler(_request_exit_from_gradio_thread)

    window.show()
    sys.exit(qt_app.exec())


if __name__ == "__main__":
    main()