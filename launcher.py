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
    from PyQt6.QtWebEngineCore import QWebEnginePage
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


# Injected into the page <head>. Gradio renders its textareas with spellcheck
# turned off, so even with a dictionary loaded Chromium would underline
# nothing. This turns it back on for the two prompt boxes ONLY -- not for the
# read-only status bar, the model-path boxes (full of filenames and drive
# letters that are not words) or anything else.
#
# A MutationObserver is needed rather than a one-shot pass: Gradio mounts the
# textareas after the initial document load, and re-mounts them whenever the
# component re-renders, which would drop a statically-applied attribute. The
# callback is idempotent and only touches nodes whose attribute is not already
# correct, so the observer costs a cheap query per mutation batch and nothing
# else.
_SPELLCHECK_HEAD = """
<script>
(function () {
  var SEL = '#prompt-positive textarea, #prompt-negative textarea';
  var queued = false;
  function apply() {
    queued = false;
    document.querySelectorAll(SEL).forEach(function (t) {
      if (t.getAttribute('spellcheck') !== 'true') {
        t.setAttribute('spellcheck', 'true');
        t.setAttribute('lang', 'en-US');
      }
    });
  }
  function schedule() {
    if (queued) return;
    queued = true;
    requestAnimationFrame(apply);
  }
  document.addEventListener('DOMContentLoaded', schedule);
  new MutationObserver(schedule).observe(document.documentElement, {
    childList: true, subtree: true
  });
  schedule();
})();
</script>
"""


def _configure_spellcheck_env() -> bool:
    """Point QtWebEngine at our local dictionary folder. Returns whether a
    dictionary is actually present.

    MUST run before the QApplication (and therefore before any QWebEngineProfile)
    is created -- QtWebEngine reads this environment variable once, during engine
    initialisation, and ignores later changes.

    Everything here is local: a .bdic file on disk, read in-process by
    Chromium's Hunspell. No network, no service, nothing leaves the machine.
    Returns False when the installer could not build the dictionary, in which
    case spellcheck stays off and the app is otherwise unchanged.
    """
    import os
    dict_dir = configure.get_dictionaries_dir()
    # Glob, not an exact filename: Chromium's dictionaries carry a version
    # suffix (en-US-10-1.bdic), the installer keeps whatever name upstream
    # published, and Qt itself matches by language-code PREFIX. Requiring an
    # exact "en-US.bdic" would find the file only by luck.
    if not any(dict_dir.glob(f"{configure.SPELLCHECK_LANGUAGE}*.bdic")):
        return False
    os.environ["QTWEBENGINE_DICTIONARIES_PATH"] = str(dict_dir)
    return True


def _enable_spellcheck(view) -> None:
    """Switch on the profile's spellchecker. Off by default in QtWebEngine.

    Wrapped because it is a nicety: a PyQt6 build without the spellcheck
    feature compiled in raises on these calls, and that must cost the user a
    squiggle, not their program."""
    try:
        profile = view.page().profile()
        profile.setSpellCheckEnabled(True)
        profile.setSpellCheckLanguages([configure.SPELLCHECK_LANGUAGE])
    except Exception as e:
        print(f"  NOTE: spellcheck unavailable ({e})")


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


class _QuietPage(QWebEnginePage):
    """QWebEnginePage that swallows one specific benign console message.

    As the Gradio frontend boots inside the embedded Chromium, QtWebEngine
    forwards a couple of 'Method not implemented.' console lines — a browser
    API the embedded engine does not implement, harmless to us since it only
    affects features we don't use. They are pure noise on the batch console,
    so we drop exactly those and pass everything else through unchanged (real
    errors still surface)."""

    def javaScriptConsoleMessage(self, level, message, line, source):  # noqa: N802
        if "Method not implemented" in str(message):
            return
        super().javaScriptConsoleMessage(level, message, line, source)


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

        icon_path = configure.get_images_dir() / "banner_llama.ico"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))

        self.view = QWebEngineView(self)
        self.view.setPage(_QuietPage(self.view))
        _enable_spellcheck(self.view)
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


def _set_windows_app_id() -> None:
    """Give this process its own AppUserModelID on Windows.

    Without this, Windows groups the taskbar entry under the host
    interpreter (python.exe) and shows *its* icon instead of ours,
    regardless of what QIcon we set on the window. Must run before the
    QApplication / any window is created."""
    if platform.system() != "Windows":
        return
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            f"WiseManTimeLord.{APP_TITLE}"
        )
    except Exception:
        pass


def main() -> None:
    _set_windows_app_id()
    configure.ensure_data_dirs()
    # Load the persisted bits of session state into configure.APP_STATE once,
    # here, before any UI exists — currently the "Add Image" picker's starting
    # folder (configuration.json's last_image_browse_dir). Doing it at startup
    # rather than on first use keeps the file reads off the click path, and
    # resets the session-only entries (reference-image list, Use All /
    # Chain All mode) to their launch defaults.
    configure.init_session_state()
    # Before the banner so its line reports the real state, and well before the
    # QApplication -- QtWebEngine reads the dictionaries path exactly once, at
    # engine start, and ignores it afterwards.
    _spell_ok = _configure_spellcheck_env()
    _print_banner()
    if _spell_ok:
        print("  Spellcheck: en-US dictionary loaded (prompt boxes underline typos)")
    else:
        # Actionable, not just a status. The dictionary is built by the
        # installer, so a user upgrading from a build that predates spellcheck
        # has everything EXCEPT the .bdic and no reason to guess that re-running
        # option 2 is what fixes it.
        print("  Spellcheck: OFF - no dictionary at "
              f"{configure.get_dictionaries_dir()}")
        print("              Run option 2 (Installation) from the batch menu "
              "to download it.")
    print()
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
        # Re-enables the browser's own offline spellchecker on the two prompt
        # boxes; see _SPELLCHECK_HEAD.
        head=_SPELLCHECK_HEAD,
    )

    if not _wait_for_server(SERVER_NAME, SERVER_PORT):
        print("ERROR: Gradio server did not come up in time.")
        sys.exit(1)

    qt_app = QApplication(sys.argv)
    qt_app.setApplicationName(APP_TITLE)

    app_icon_path = configure.get_images_dir() / "banner_llama.ico"
    if app_icon_path.exists():
        qt_app.setWindowIcon(QIcon(str(app_icon_path)))

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