#!/usr/bin/env python3
"""
display.py - Gradio 6 UI for Image-Gradio-Gguf.
Three tabs: Generate | Configuration | Debug / Info
Build/install functionality lives in installer.py only.

Backend dropdowns are populated from configure.get_backend_choices(), which
reflects whatever GPUs ggml actually enumerated on THIS machine at install
time. Nothing here assumes a GPU count or a particular device index.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
_GRADIO_MAJOR = int(gr.__version__.split(".")[0])

import scripts.configure as configure
import scripts.inference as inference


# ---------------------------------------------------------------------------
# Exit handling
# ---------------------------------------------------------------------------
# The "Exit Program" button click runs on a Gradio/Starlette worker thread,
# not the Qt GUI thread launcher.py creates. launcher.py registers a
# thread-safe handler here (via set_exit_handler) that signals the Qt
# window to close, so window-geometry saving and shutdown all happen in
# one place (launcher.py's _shutdown()). If nothing registers a handler
# (e.g. display.py is ever driven without launcher.py), fall back to a
# plain process exit so the button still works.
_exit_handler: Optional[Any] = None


def set_exit_handler(handler) -> None:
    """Register the function to call when 'Exit Program' is clicked.
    Called once by launcher.py during startup."""
    global _exit_handler
    _exit_handler = handler


def _handle_exit_click() -> None:
    if _exit_handler is not None:
        _exit_handler()
    else:
        os._exit(0)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _cfg() -> Dict[str, Any]:
    """Fresh load from persistent.json."""
    return configure.load_persistent()


_FILETYPES_MODEL = [
    ("Model files", "*.gguf *.safetensors"),
    ("GGUF",        "*.gguf"),
    ("Safetensors", "*.safetensors"),
    ("All files",   "*.*"),
]

# The VAE is always a safetensors (ae.safetensors, ~335MB), never a gguf, so
# its picker leads with that rather than making the user wade past .gguf files.
_FILETYPES_VAE = [
    ("Safetensors", "*.safetensors"),
    ("All files",   "*.*"),
]


def _browse_file(file_types: Optional[List[Tuple[str, str]]] = None) -> str:
    """Open a native file dialog and return the chosen path."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        
        # Determine the initial directory for the file dialog
        cfg = configure.load_persistent()
        last_dir = cfg.get("last_model_browse_dir", "")
        initial_dir = str(configure.get_models_dir())
        
        if last_dir:
            # Resolve the path (handles relative paths like ".\models")
            p = Path(last_dir)
            if not p.is_absolute():
                p = configure._get_project_root() / p
            if p.exists() and p.is_dir():
                initial_dir = str(p)
                
        path = filedialog.askopenfilename(
            initialdir=initial_dir,
            filetypes=file_types or _FILETYPES_MODEL,
        )
        root.destroy()
        
        # If a file was selected, save its directory for next time
        if path:
            selected_dir = str(Path(path).parent)
            configure.update_persistent({"last_model_browse_dir": selected_dir})
            return path
            
        return ""
    except Exception:
        return ""


def _backend_choices() -> List[str]:
    return configure.get_backend_choices()["all_choices"]


def _default_backend_value(key: str) -> str:
    """
    Load the saved backend string; if it's no longer in the current
    choices list (e.g. GPU was removed, or this is a fresh install with
    a newly-branded CPU label), fall back to the CPU entry.
    """
    saved = _cfg().get(key, "")
    choices = _backend_choices()
    if saved in choices:
        return saved
    # First entry is always the CPU label
    return choices[0] if choices else "CPU"


def _thread_choices() -> List[int]:
    return configure.get_thread_choices()


def _detect_vae(diff_path_str: str) -> Tuple[str, str]:
    r"""
    Given a diffusion model path, locate ae.safetensors and return
    (vae_full_path, vae_stem), or ("", "") if not found.

    Search order: the model's own folder first, then each folder above it up
    to and including .\models, then a full sweep of .\models.

    The walk-up matters. Z-Image-Turbo finetunes ship nested — BigDannyPt's
    collection lays out as DarkBeast\DBZiT9-DIMRClaw\darkBeastMar2126Latest_
    dbzit9DIMRclaw-Q8_0.gguf, two folders deep — and none of those repos
    include a VAE, so the single ae.safetensors from the stock Z-Image-Turbo
    repo lives at the .\models root and is shared by every diffusion model.
    Looking only in the model's own folder found it for a flat .\models and
    for nothing else.
    """
    if not diff_path_str:
        return "", ""
    p = Path(diff_path_str).expanduser()
    if not p.exists():
        return "", ""

    def _scan(folder: Path) -> Optional[Path]:
        try:
            for f in folder.iterdir():
                if f.is_file() and f.name.lower() == "ae.safetensors":
                    return f
        except OSError:
            pass
        return None

    models_dir = configure.get_models_dir().resolve()

    # The model's folder, then upwards. Bounded: stop once we pass the models
    # dir, and never climb more than 4 levels, so a model parked somewhere odd
    # cannot send us walking up to C:\.
    current = p.parent.resolve()
    for _ in range(5):
        hit = _scan(current)
        if hit:
            return str(hit), hit.stem
        if current == models_dir or current == current.parent:
            break
        current = current.parent

    # Models root, then anywhere beneath it.
    hit = _scan(models_dir)
    if hit:
        return str(hit), hit.stem
    try:
        for f in models_dir.rglob("*.safetensors"):
            if f.is_file() and f.name.lower() == "ae.safetensors":
                return str(f), f.stem
    except OSError:
        pass
    return "", ""


def _resolve_vae(diff_path_str: str, current_vae_path: str,
                 current_vae_name: str) -> Tuple[Any, Any]:
    """
    Decide what the VAE boxes should hold after the diffusion model changes.
    Returns (path_update, name_update) for (vae_path_tb, vae_name_tb).

    Three rules, in order:
      1. ae.safetensors found for the new model  -> use it.
      2. Not found, but the box already holds a VAE that still exists on disk
         -> keep it. This is the normal case for the community finetunes: they
         ship a DiT gguf and nothing else, so switching from vanilla to e.g.
         zImageTurboNSFW must not throw away the ae.safetensors that is
         already configured and still perfectly valid for it.
      3. Not found and nothing usable held -> blank, and wait for the user.
         A stale path to a deleted file is cleared rather than kept, so the
         box never claims a VAE that generation would then fail on.
    """
    vae_path, vae_name = _detect_vae(diff_path_str)
    if vae_path:
        return vae_path, vae_name
    if current_vae_path and Path(current_vae_path).expanduser().exists():
        return gr.update(), gr.update()          # rule 2 — leave as-is
    return "", ""


# ---------------------------------------------------------------------------
# Recent images helpers
# ---------------------------------------------------------------------------


# Module-level cache for recent images – stores the full list, not limited
_gallery_cache: Dict[str, Any] = {
    "full_list": None,      # list of all image paths (unlimited)
    "mtime": 0.0,
}

def _get_recent_images(max_images: int = 50) -> List[str]:
    """Return paths of images in the output folder, newest first.
       Caches the full list of images and slices it according to max_images.
       Prints a concise start/end summary only when a rescan occurs.
    """
    out_dir = configure.get_output_dir()
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    try:
        current_mtime = out_dir.stat().st_mtime if out_dir.exists() else 0.0
        # If the directory hasn't changed, reuse the cached full list
        if (_gallery_cache["full_list"] is not None and
            abs(current_mtime - _gallery_cache["mtime"]) < 0.1):
            full = _gallery_cache["full_list"]
            # Return the first max_images entries (sliced)
            return full[:max_images]

        # Directory changed (or first run) – perform a fresh scan
        print("[gallery] Scanning for Thumbnails....", flush=True)
        files = [
            f for f in out_dir.iterdir()
            if f.is_file() and f.suffix.lower() in exts
        ]
        files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        full_list = [str(f) for f in files]   # store all, no limit
        count = len(full_list)
        print(f"[gallery] Rescanned {out_dir}: {count} image{'s' if count != 1 else ''}", flush=True)
        print()
       
        # Update cache
        _gallery_cache["full_list"] = full_list
        _gallery_cache["mtime"] = current_mtime
        return full_list[:max_images]
    except Exception as e:
        print(f"[gallery] rescan FAILED: {e}", flush=True)
        return []

# ---------------------------------------------------------------------------
# Preview box status images (.\media\program_*.jpg)
# ---------------------------------------------------------------------------

def _status_image(name: str) -> Optional[str]:
    """Return str path to a media/program_<name>.jpg, or None if missing."""
    p = configure.get_media_dir() / f"program_{name}.jpg"
    return str(p) if p.exists() else None


def _idle_preview_image() -> Optional[str]:
    """Preview box image when no generation is running: the newest output
    image if one exists, otherwise the 'no media' placeholder."""
    recent = _get_recent_images(max_images=1)
    if recent:
        return recent[0]
    return _status_image("no_media")


# ---------------------------------------------------------------------------
# Tab 1 — Generate  (UI widgets + event wiring split for shared status)
# ---------------------------------------------------------------------------

# Module-level refs so _wire_generate_events() can access them
_gen: Dict[str, Any] = {}


def _missing_models() -> List[str]:
    """Labels of the model files that are not set, or set but gone from disk.

    All three are mandatory for every run — a Z-Image diffusion gguf is the
    DiT alone, so it cannot condition without the Qwen3 encoder (--llm) nor
    reach pixels without the VAE (--vae). Checked as one group so the user is
    told everything that is missing at once, rather than fixing the diffusion
    model only to be sent back for the VAE.
    """
    c = configure.load_persistent()
    labels = [("encoder_model_path", "Encoder"),
              ("imagegen_model_path", "Diffusion"),
              ("vae_model_path", "VAE")]
    return [label for key, label in labels
            if not c.get(key) or not Path(c[key]).exists()]


def _models_configured() -> bool:
    """Return True if all three model paths are set and exist on disk."""
    return not _missing_models()


def _build_generate_tab_inner() -> None:
    """Build Generate tab widgets; store refs in _gen for later wiring."""
    cfg = _cfg()
    presets = configure.get_generation_presets()
    configured = _models_configured()

    prompt_ph  = ("Describe the image you want to generate..."
                  if configured else "Set locations of models on Configuration page first...")
    neg_ph     = ("Things to exclude..."
                  if configured else "Set locations of models on Configuration page first...")

    with gr.Row():
        # ── Left column: settings ────────────────────────────────────────────
        with gr.Column(scale=3):
            gr.Markdown("### Settings")
            _gen["prompt_tb"] = gr.Textbox(
                label="Positive Prompt",
                placeholder=prompt_ph,
                lines=2, max_lines=10,
                value=cfg.get("last_prompt", ""),
            )
            _gen["negative_tb"] = gr.Textbox(
                label="Negative Prompt",
                placeholder=neg_ph,
                lines=2, max_lines=10,
                value=cfg.get("negative_prompt", ""),
            )
            with gr.Row():
                _gen["preset_dd"] = gr.Dropdown(
                    label="Quality Preset", choices=list(presets.keys()),
                    value=cfg.get("imagegen_quality_preset", "Fast (Turbo)"),
                )
                _gen["sampler_dd"] = gr.Dropdown(
                    label="Sampler Type", choices=list(configure.SAMPLER_MAP.keys()),
                    value=cfg.get("imagegen_sampling", "euler_a"),
                )

            with gr.Row():
                _gen["batch_dd"] = gr.Dropdown(label="Batch Count",
                                               choices=configure.BATCH_COUNT_CHOICES,
                                               value=cfg.get("imagegen_batch_count", 1)
                )
                _gen["width_dd"]  = gr.Dropdown(label="Image Width",  choices=configure.IMAGE_SIZES,
                                                value=cfg.get("imagegen_width", 512))
                _gen["height_dd"] = gr.Dropdown(label="Image Height", choices=configure.IMAGE_SIZES,
                                                value=cfg.get("imagegen_height", 512))
                _gen["output_fmt_dd"] = gr.Dropdown(
                    label="Output Format", choices=configure.OUTPUT_FORMATS,
                    value=cfg.get("output_format", "png"),
                )

            with gr.Row():
                _gen["steps_dd"] = gr.Dropdown(
                    label="Diffuse Steps", choices=configure.STEP_CHOICES,
                    value=cfg.get("imagegen_steps", 4),
                )
                _gen["cfg_scale_sld"] = gr.Slider(
                    label="CFG Scale", minimum=0.5, maximum=20.0, step=0.5,
                    value=cfg.get("imagegen_cfg_scale", 1.0),
                )
                _gen["seed_num"] = gr.Number(label="Gen Seed (-1 = random)",
                                             value=cfg.get("imagegen_seed", -1), precision=0)

            # NOTE: no manual "Save as Default" button — successful
            # generations auto-save their settings panel values (see
            # on_generate_click / do_generate's success branch), so the
            # next launch picks up the last settings that actually worked.

        # ── Right column: image preview only ────────────────────────────────
        with gr.Column(scale=2):
            gr.Markdown("### Output")
            with gr.Row(visible=configured) as _gen["generate_row"]:
                # Single dynamic button: shows "Generate" (primary) when idle
                # and switches to "..Please Wait.." (disabled) while a
                # generation is running. do_generate() flips it to Please Wait
                # on entry and back to Generate on its final yield.
                _gen["generate_btn"] = gr.Button("Generate Image", variant="primary", size="lg")

            # Single currently-selected/in-progress image — the ONLY image
            # shown here is either the most recent generation, the live
            # encoding/diffusion phase status image, or whatever the user
            # has clicked in the gallery below. Never shows Gradio's
            # built-in progress bar.
            # Gradio 6 replaced show_download_button/show_share_button with a
            # single buttons=[...] list, so build the kwarg for whichever
            # major version is actually installed.
            _img_button_kwargs = (
                {"buttons": ["download"]} if _GRADIO_MAJOR >= 6
                else {"show_download_button": True, "show_share_button": False}
            )
            _gen["preview_img"] = gr.Image(
                label="Image Preview",
                type="filepath",
                value=_idle_preview_image(),
                height=configure.PREVIEW_IMAGE_HEIGHT,
                interactive=False,
                show_label=True,
                container=True,
                elem_id="preview-img",
                **_img_button_kwargs,
            )

    # ── Gallery: full width, underneath BOTH columns ─────────────────────────
    # Its only job is to show thumbnails of everything in .\output, populated
    # solely via full rescans (_get_recent_images), never a per-call image
    # list. Clicking a thumbnail here updates the preview box above — it does
    # not show generation progress, and it is not itself the preview.
    gr.Markdown("### Thumbnails Gallery")
    _gen["output_gallery"] = gr.Gallery(
        label="Generated Images",
        value=_get_recent_images(),
        columns=16, rows=1,
        height=123,
        object_fit="contain",
        allow_preview=False,
        show_label=False,
        fit_columns=False,
        elem_id="output-gallery",
    )


    # Preset change wired immediately
    presets_map = presets

    def apply_preset(name: str):
        # Custom preset carries no values — leave all widgets unchanged.
        if name == "Custom":
            return (gr.update(), gr.update(), gr.update(), gr.update(), gr.update())
        p = presets_map.get(name, {})
        return (p.get("imagegen_width", 512), p.get("imagegen_height", 512),
                p.get("imagegen_steps", 4), p.get("imagegen_sampling", "euler_a"),
                p.get("imagegen_cfg_scale", 1.0))

    _gen["preset_dd"].change(
        apply_preset, inputs=_gen["preset_dd"],
        outputs=[_gen["width_dd"], _gen["height_dd"], _gen["steps_dd"],
                 _gen["sampler_dd"], _gen["cfg_scale_sld"]],
    )

    # When the user manually changes any individual setting widget, the
    # Quality Preset automatically switches to "Custom" — signalling that
    # settings are user-defined rather than tied to a named preset.
    def _set_custom(_ignored):
        return gr.update(value="Custom")

    for _widget_key in ("width_dd", "height_dd", "steps_dd", "sampler_dd",
                        "cfg_scale_sld", "batch_dd", "output_fmt_dd", "seed_num"):
        _gen[_widget_key].change(
            _set_custom,
            inputs=_gen[_widget_key],
            outputs=_gen["preset_dd"],
        )


def _wire_generate_events(status_box: gr.Textbox) -> None:
    """Register Generate tab event handlers that output to shared status_box."""
    import threading

    # ── Inactivity timeout (20 min after last generation finishes) ──
    _INACTIVITY_SECONDS = 20 * 60
    _timeout_timer: Dict[str, Any] = {"handle": None}

    def _reset_inactivity_timer():
        if _timeout_timer["handle"] is not None:
            _timeout_timer["handle"].cancel()
        t = threading.Timer(_INACTIVITY_SECONDS, inference.unload_models)
        t.daemon = True
        t.start()
        _timeout_timer["handle"] = t

    def _cancel_inactivity_timer():
        if _timeout_timer["handle"] is not None:
            _timeout_timer["handle"].cancel()
            _timeout_timer["handle"] = None

    def do_generate(prompt, negative, width, height, steps, sampler,
    cfg_scale, seed, batch, output_format, quality_preset):
        """
        Generator: yields (preview_img, gallery, status, btn_update) tuples
        so the preview box can switch between program_encoding.jpg /
        program_diffusion.jpg while generation runs, WITHOUT using Gradio's
        built-in progress bar anywhere (gallery, preview, or otherwise).
        btn_update flips the single Generate/Please Wait button to its
        "..Please Wait.." appearance for the duration of the run and back
        to "Generate" on the final yield (including early-exit validation
        failures, which never started a run and so should leave the button
        as Generate). Final yield swaps the preview to the finished image
        and rescans .\\output into the gallery — the gallery never receives
        a per-call image list, only full folder rescans.
        """
        gallery_now  = _get_recent_images()
        preview_now  = _idle_preview_image()
        _btn_generate = gr.update(value="Generate Image", variant="primary", interactive=True)
        _btn_wait     = gr.update(value="..Please Wait..", variant="primary", interactive=False)

        if not prompt or not prompt.strip():
            yield preview_now, gallery_now, "Please enter a prompt.", _btn_generate
            return
        missing = _missing_models()
        if missing:
            yield (preview_now, gallery_now,
                   "Complete model details on Configuration page first! "
                   f"(missing: {', '.join(missing)})", _btn_generate)
            return
        c = _cfg()

        # Cancel any pending unload while we are generating
        _cancel_inactivity_timer()

        gen_cfg = dict(c)
        gen_cfg.update(
            imagegen_width=int(width), imagegen_height=int(height),
            imagegen_steps=int(steps), imagegen_sampling=sampler,
            imagegen_cfg_scale=float(cfg_scale),
            imagegen_seed=int(seed), imagegen_batch_count=int(batch),
            negative_prompt=negative,
            output_format=output_format,
        )

        configure.APP_STATE["cancel_requested"] = False

        # ── Run generation on a worker thread; main thread yields preview
        #    + status updates based on phase, polled from a shared mutable
        #    holder ("_phase"). The status string shows which of the two
        #    phases is active (1/2 Encoding, 2/2 Diffusing) plus a live
        #    timer for that phase. Once configure.TIMING_STATS has data from
        #    a prior generation this session, the timer becomes an ETA
        #    countdown-style display ("~Ns left"); until then it just counts
        #    up, since there's nothing to estimate against yet. ──
        _batch_count = int(batch)
        _phase: Dict[str, Any] = {
            "name": "encoding", "result": None, "done": False,
            "phase_start": time.time(), "step": 0, "total_steps": 0,
            "batch_current": 1, "batch_total": _batch_count,
            "last_step_seen": 0,
        }

        def prog_cb(msg: str, pct: float, info: Dict[str, Any] = None):
            info = info or {}
            phase = info.get("phase")
            if phase in ("encoding", "diffusion"):
                _phase["name"] = phase
            else:
                # Fallback for any caller that didn't pass phase info.
                m = msg.lower()
                if "enhanc" in m or "encod" in m:
                    _phase["name"] = "encoding"
                elif "generat" in m or "step" in m or "%" in m:
                    _phase["name"] = "diffusion"
            if "phase_start" in info:
                _phase["phase_start"] = info["phase_start"]
            if "step" in info:
                new_step = info["step"]
                # Detect when the step counter resets (new image in batch):
                # a step value of 1 arriving after we've already seen a
                # higher step means sd.cpp has moved on to the next image.
                if (new_step == 1 and _phase["last_step_seen"] > 1
                        and _phase["batch_current"] < _phase["batch_total"]):
                    _phase["batch_current"] += 1
                _phase["last_step_seen"] = new_step
                _phase["step"] = new_step
            if "total_steps" in info:
                _phase["total_steps"] = info["total_steps"]

        def _format_status() -> str:
            """Build the 'Batch Number X/Y; Generate Stage N/2; ... Phase
            {step}/{total}...###s (prev_batch_Ns)' status string. Seconds are
            always whole numbers (no split seconds) per the fixed status-bar
            format — never decimals."""
            name = _phase["name"]
            elapsed_s = int(time.time() - _phase["phase_start"])
            batch_cur = _phase["batch_current"]
            batch_tot = _phase["batch_total"]
            batch_prefix = f"Batch Number {batch_cur}/{batch_tot}; "

            # Previous batch elapsed suffix — only shown when we have a
            # recorded time from a completed batch earlier this session.
            prev_elapsed = configure.APP_STATE.get("last_batch_elapsed_seconds", 0)
            prev_suffix = f" ({int(prev_elapsed)}s)" if prev_elapsed else ""

            if name == "encoding":
                # enhance_prompt() (inference.py) is a single-shot llama-cli
                # call with no per-token step/total reported back through
                # progress_callback, so step/total are only ever populated
                # once a step-aware encoder backend supplies them. Until
                # then this degrades to a plain running timer rather than
                # showing a fabricated "0/0".
                step = _phase.get("step", 0)
                total = _phase.get("total_steps", 0)
                step_part = f" {step}/{total}" if total else ""
                return (f"{batch_prefix}Generate Stage 1/2; Encoding Phase{step_part}..."
                        f"{elapsed_s}s{prev_suffix}")

            # diffusion
            step = _phase.get("step", 0)
            total = _phase.get("total_steps", 0) or int(gen_cfg.get("imagegen_steps", 0))
            step_part = f" {step}/{total}" if total else ""
            return (f"{batch_prefix}Generate Stage 2/2; Diffusing Phase{step_part}..."
                    f"{elapsed_s}s{prev_suffix}")

        def worker():
            try:
                _phase["result"] = inference.generate_image(
                    prompt.strip(), gen_cfg, progress_callback=prog_cb)
            except Exception as e:
                _phase["result"] = {"success": False, "output_path": "",
                                    "message": f"Error: {e}"}
            finally:
                _phase["done"] = True

        t = threading.Thread(target=worker, daemon=True)
        t.start()

        last_shown_img = None
        last_shown_second = -1
        first_tick = True
        # Button switches to "..Please Wait.." the moment the worker thread starts.
        # Only send that update on the FIRST poll tick — re-sending the same
        # gr.update() on every 0.15s tick forces Gradio to re-render the
        # button node repeatedly, which was cascading into a layout
        # recalculation of sibling nodes in the same column (incl. the
        # preview image) and intermittently knocking out its object-fit
        # CSS override. Once the button is already showing "..Please Wait..", later
        # ticks pass a true no-op (gr.update()) for it.
        #
        # The status string itself is also throttled to once per whole
        # second (last_shown_second), independent of the 0.15s poll
        # cadence — the timer display only ever shows whole seconds, so
        # there is no reason to push a new status string more than once a
        # second even though we keep polling faster for image/button
        # responsiveness.
        while not _phase["done"]:
            img = _status_image(_phase["name"])
            current_second = int(time.time() - _phase["phase_start"])
            btn_update = _btn_wait if first_tick else gr.update()
            first_tick = False

            status_update = gr.update()
            if current_second != last_shown_second:
                last_shown_second = current_second
                status_update = _format_status()

            if img and img != last_shown_img:
                last_shown_img = img
                yield img, gr.update(), status_update, btn_update
            else:
                yield gr.update(), gr.update(), status_update, btn_update
            time.sleep(0.15)
        t.join()

        # Start inactivity timer now that generation is finished
        _reset_inactivity_timer()

        result = _phase["result"] or {"success": False, "message": "Unknown error"}

        if result.get("success") and result.get("output_path"):
            out_path = Path(result["output_path"])
            try:
                sz = out_path.stat().st_size
                print(f"[generate] output file: {out_path}  ({sz} bytes)", flush=True)
            except Exception as e:
                print(f"[generate] output file STAT FAILED: {out_path}  {e}", flush=True)
            # Record the total batch elapsed time so the next generation can
            # display it as a reference in the status bar (previous batch time).
            batch_elapsed = result.get("elapsed_seconds", 0.0)
            configure.APP_STATE["last_batch_elapsed_seconds"] = int(round(batch_elapsed))
            msg = (f"{result['message']} | Seed: {result['seed_used']} "
                   f"| Time: {int(round(result['elapsed_seconds']))}s")
            new_gallery = _get_recent_images()
            yield str(out_path), new_gallery, msg, _btn_generate
            # Auto-save settings only when the Quality Preset is "Custom".
            # Named presets (Fast, Balanced, Quality, etc.) are fixed — no
            # need to persist them since they are always reconstructed from
            # configure.get_generation_presets(). Custom captures any
            # user-modified combination that deviates from the named presets,
            # and must be saved so it survives the next launch.
            if quality_preset == "Custom":
                configure.update_persistent({
                    "imagegen_quality_preset": "Custom",
                    "imagegen_width": int(width), "imagegen_height": int(height),
                    "imagegen_steps": int(steps), "imagegen_sampling": sampler,
                    "imagegen_cfg_scale": float(cfg_scale),
                    "imagegen_seed": int(seed), "imagegen_batch_count": int(batch),
                    "negative_prompt": negative,
                    "output_format": output_format,
                })
            else:
                # Still persist the currently-active named preset so that
                # it is restored correctly on next launch.
                configure.update_persistent({
                    "imagegen_quality_preset": quality_preset,
                })
        else:
            new_gallery = _get_recent_images()
            yield _idle_preview_image(), new_gallery, result.get("message", "Unknown error"), _btn_generate


    def on_generate_click(prompt, negative, width, height, steps, sampler,
    cfg_scale, seed, batch, output_format, quality_preset):
        """Dispatch a click on the single dynamic button. The button reads
        "Generate" when idle and starts a run (delegating to the do_generate
        generator, which yields its own button-state updates). While a run
        is in progress the button is disabled and shows "..Please Wait..",
        preventing concurrent runs."""
        yield from do_generate(prompt, negative, width, height, steps, sampler,
    cfg_scale, seed, batch, output_format, quality_preset)

    _gen["generate_btn"].click(
        on_generate_click,
        inputs=[_gen["prompt_tb"], _gen["negative_tb"],
        _gen["width_dd"], _gen["height_dd"], _gen["steps_dd"],
        _gen["sampler_dd"], _gen["cfg_scale_sld"],
        _gen["seed_num"], _gen["batch_dd"], _gen["output_fmt_dd"],
        _gen["preset_dd"]],
        outputs=[_gen["preview_img"], _gen["output_gallery"], status_box,
        _gen["generate_btn"]],
    )

    def on_gallery_select(evt: gr.SelectData):
        """When the user clicks a thumbnail, display it in the preview box."""
        if evt.value and isinstance(evt.value, dict):
            path = evt.value.get("image", {}).get("path") or evt.value.get("path")
            if path:
                return path
        elif evt.value and isinstance(evt.value, str):
            return evt.value
        return gr.update()

    _gen["output_gallery"].select(
        on_gallery_select,
        inputs=None,
        outputs=_gen["preview_img"],
    )

    # ── Refresh generate_row visibility when prompt changes ──
    def _refresh_generate_row(prompt_val):
        configured = _models_configured()
        prompt_ph = ("Describe the image you want to generate..."
                     if configured else "Set locations of models on Configuration page first...")
        neg_ph    = ("Things to exclude..."
                     if configured else "Set locations of models on Configuration page first...")
        return (
            gr.update(visible=configured),
            gr.update(placeholder=prompt_ph),
            gr.update(placeholder=neg_ph),
        )

    _gen["prompt_tb"].focus(
        _refresh_generate_row,
        inputs=_gen["prompt_tb"],
        outputs=[_gen["generate_row"], _gen["prompt_tb"], _gen["negative_tb"]],
    )


# ---------------------------------------------------------------------------
# Tab 2 — Configuration  (UI widgets + event wiring split for shared status)
# ---------------------------------------------------------------------------

_cfg_w: Dict[str, Any] = {}  # widget refs for wiring
_dbg:   Dict[str, Any] = {}  # debug tab widget refs


def _build_config_tab_inner() -> None:
    """Build Configuration tab widgets; store refs in _cfg_w for later wiring."""
    cfg     = _cfg()
    choices = _backend_choices()
    threads = _thread_choices()
    dt      = configure.get_default_threads()

    # ── Model paths ──
    # Left column: encoder. Right column: the image-generation pair —
    # diffusion model on the first row, its VAE on the second, because the two
    # are always chosen together. Every Z-Image diffusion gguf is the DiT
    # component ONLY (no VAE tensors, no text encoder), so a run needs all
    # three files and the VAE cannot stay an invisible auto-detected extra:
    # the community finetunes ship no ae.safetensors at all, so detection
    # legitimately misses and the user has to be able to point at the one they
    # already have. Full paths live in the hidden *_path_tb boxes; the visible
    # boxes carry display names only.
    gr.Markdown("### Model Paths")
    _cfg_w["enc_path_tb"]  = gr.Textbox(value=cfg.get("encoder_model_path", ""),  visible=False)
    _cfg_w["diff_path_tb"] = gr.Textbox(value=cfg.get("imagegen_model_path", ""), visible=False)
    _cfg_w["vae_path_tb"]  = gr.Textbox(value=cfg.get("vae_model_path", ""),      visible=False)

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Row():
                _cfg_w["enc_name_tb"] = gr.Textbox(
                    label="Encoder Name",
                    value=cfg.get("encoder_model_name", ""),
                    placeholder="Qwen3-4b-Z-Image-Turbo",
                    interactive=True,
                    scale=8,
                )
                _cfg_w["enc_browse_btn"] = gr.Button("Browse...", size="sm", scale=1, min_width=90)

        with gr.Column(scale=1):
            with gr.Row():
                _cfg_w["diff_name_tb"] = gr.Textbox(
                    label="Diffusion Name",
                    value=cfg.get("imagegen_model_name", ""),
                    placeholder="z_image_turbo",
                    interactive=True,
                    scale=8,
                )
                _cfg_w["diff_browse_btn"] = gr.Button("Browse...", size="sm", scale=1, min_width=90)

            with gr.Row():
                _cfg_w["vae_name_tb"] = gr.Textbox(
                    label="VAE Name",
                    value=cfg.get("vae_model_name", ""),
                    placeholder="ae.safetensors",
                    info="Auto-filled when found next to the diffusion model.",
                    interactive=True,
                    scale=8,
                )
                _cfg_w["vae_browse_btn"] = gr.Button("Browse...", size="sm", scale=1, min_width=90)

    # ── Backend selection + CPU threads (consolidated) ──
    is_cpu_only = configure.get_install_type() == "cpu_only"

    gr.Markdown("### Backend Selection")
    if is_cpu_only:
        gr.Markdown(
            "(Cpu-only install). "
        )
    else:
        gr.Markdown(
            "(Vulkan install). "
        )
    with gr.Row():
        with gr.Column(scale=2):
            _cfg_w["enc_backend_dd"] = gr.Dropdown(
                label="Encoder Backend",
                choices=choices,
                value=_default_backend_value("backend_encoder"),
                interactive=not is_cpu_only,
            )

        with gr.Column(scale=1):
            _cfg_w["threads_dd"] = gr.Dropdown(
                label="CPU Threads",
                choices=threads,
                value=cfg.get("encoder_threads", dt),
            )

        with gr.Column(scale=2):
            _cfg_w["img_backend_dd"] = gr.Dropdown(
                label="ImageGen Backend",
                choices=choices,
                value=_default_backend_value("backend_imagegen"),
                interactive=not is_cpu_only,
            )

    # Initial interactive/value state for the two GPU-dependent controls
    # below is driven by the ACTUAL selected backend, not just install type.
    # A Vulkan-capable install with "CPU" currently selected must still show
    # GPU Layers / Diffuser Placement as locked at their CPU-forced values —
    # otherwise the controls look interactive but silently have no effect,
    # which was the root of the original bug report.
    enc_backend_val = _default_backend_value("backend_encoder")
    enc_is_vulkan   = (not is_cpu_only) and ("Vulkan" in enc_backend_val)
    img_backend_val = _default_backend_value("backend_imagegen")
    img_is_vulkan   = (not is_cpu_only) and ("Vulkan" in img_backend_val)

    with gr.Row():
        # ── Encoder (LLM) settings ──
        with gr.Column(scale=2):
            gr.Markdown("### Encoder (LLM) Settings")
            with gr.Row():
                _cfg_w["enc_batch_dd"] = gr.Dropdown(label="Batch Size",
                                           choices=configure.BATCH_SIZE_CHOICES,
                                           value=cfg.get("encoder_batch_size", 512))
                _cfg_w["enc_ctx_dd"] = gr.Dropdown(label="Context Size",
                                         choices=configure.CTX_SIZE_CHOICES,
                                         value=cfg.get("encoder_ctx_size", 4096))

            with gr.Row():
                _cfg_w["enc_ngl_dd"] = gr.Dropdown(
                    label="GPU Layers",
                    choices=configure.GPU_LAYER_CHOICES,
                    value=cfg.get("encoder_gpu_layers", -1) if enc_is_vulkan else 0,
                    info=("(-1 = all layers)." if enc_is_vulkan
                          else "Encoder Backend is CPU — all layers run on CPU."),
                    interactive=enc_is_vulkan,
                )
                _cfg_w["enc_flash_chk"] = gr.Checkbox(
                    label="Flash Attention",
                    value=cfg.get("encoder_flash_attn", True),
                    info="Reduces VRAM usage for long contexts.",
                )

        # ── ImageGen settings ──
        with gr.Column(scale=2):
            gr.Markdown("### Image Generation Settings")
            with gr.Row():
                _cfg_w["img_clip_dd"] = gr.Dropdown(label="CLIP Skip",
                                          choices=configure.CLIP_SKIP_CHOICES,
                                          value=cfg.get("imagegen_clip_skip", 2))
            with gr.Row():
                # sd.cpp has no per-layer GPU offload for the diffuser (no
                # -ngl equivalent) — only whole-component placement, so this
                # is a 3-way choice rather than a layer-count dropdown. See
                # configure.DIFFUSER_PLACEMENT_CHOICES / parse_diffuser_placement().
                _cfg_w["img_placement_dd"] = gr.Dropdown(
                    label="Diffuser Placement",
                    choices=configure.DIFFUSER_PLACEMENT_CHOICES,
                    value=(cfg.get("imagegen_placement", configure.DIFFUSER_PLACEMENT_FULL_GPU)
                          if img_is_vulkan else configure.DIFFUSER_PLACEMENT_FULL_CPU),
                    info=("Split keeps the encoder+VAE on CPU, diffusion model on GPU."
                          if img_is_vulkan else
                          "ImageGen Backend is CPU — sd.cpp will not touch the GPU at all."),
                    interactive=img_is_vulkan,
                )

    # ── Prompt template ──
    gr.Markdown("### Advanced")
    _cfg_w["prompt_template_tb"] = gr.Textbox(
        label="Prompt Template",
        value=cfg.get("prompt_template",
                      "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"),
        lines=2,
    )

    # ── Save ──
    with gr.Row():
        _cfg_w["save_all_btn"]    = gr.Button("Save All Configuration", variant="primary", size="lg")


    # ── Events: browse & scan — no status output, wire immediately ──
    def _browse_encoder():
        p = _browse_file()
        return (p, Path(p).stem) if p else (gr.update(), gr.update())

    def _browse_diffusion(current_vae_path: str, current_vae_name: str):
        p = _browse_file()
        if not p:
            return gr.update(), gr.update(), gr.update(), gr.update()
        vae_path, vae_name = _resolve_vae(p, current_vae_path, current_vae_name)
        return p, Path(p).stem, vae_path, vae_name

    def _browse_vae():
        # No _resolve_vae here: an explicit pick by the user is final and is
        # never second-guessed by auto-detection.
        p = _browse_file(_FILETYPES_VAE)
        return (p, Path(p).stem) if p else (gr.update(), gr.update())

    _cfg_w["enc_browse_btn"].click(
        _browse_encoder,
        outputs=[_cfg_w["enc_path_tb"], _cfg_w["enc_name_tb"]]
    )
    _cfg_w["diff_browse_btn"].click(
        _browse_diffusion,
        inputs=[_cfg_w["vae_path_tb"], _cfg_w["vae_name_tb"]],
        outputs=[_cfg_w["diff_path_tb"], _cfg_w["diff_name_tb"],
                 _cfg_w["vae_path_tb"], _cfg_w["vae_name_tb"]]
    )
    _cfg_w["vae_browse_btn"].click(
        _browse_vae,
        outputs=[_cfg_w["vae_path_tb"], _cfg_w["vae_name_tb"]]
    )

    # When the diffusion path changes — by Browse, or by hand — re-resolve the
    # VAE. Same _resolve_vae() as the browse handler, so whichever of the two
    # fires (and .change fires for programmatic updates too, so both often do)
    # the outcome is identical and re-running is harmless.
    def _on_diff_path_change(path: str, current_vae_path: str,
                             current_vae_name: str):
        return _resolve_vae(path, current_vae_path, current_vae_name)

    _cfg_w["diff_path_tb"].change(
        _on_diff_path_change,
        inputs=[_cfg_w["diff_path_tb"], _cfg_w["vae_path_tb"],
                _cfg_w["vae_name_tb"]],
        outputs=[_cfg_w["vae_path_tb"], _cfg_w["vae_name_tb"]]
    )

    # ── Keep GPU Layers / Diffuser Placement in sync with their backend
    # dropdowns. Switching a backend to CPU must force the dependent
    # control to its CPU value and lock it (0 layers / Full CPU placement);
    # switching back to Vulkan must restore the last value the user had
    # and re-enable editing. Without this, the dropdowns stayed interactive
    # and showed stale numbers no matter what backend was actually selected
    # — which is what made it look like the CPU selection "did nothing".
    _last_ngl_value: Dict[str, int] = {"v": cfg.get("encoder_gpu_layers", -1)}
    _last_placement_value: Dict[str, str] = {
        "v": cfg.get("imagegen_placement", configure.DIFFUSER_PLACEMENT_FULL_GPU)
    }

    def _on_enc_backend_change(backend_choice: str, current_ngl):
        if "Vulkan" in backend_choice:
            restore = _last_ngl_value["v"]
            return gr.update(value=restore, interactive=True,
                             info="(-1 = all layers).")
        # Remember the value the user had before forcing it to 0, so
        # switching back to Vulkan restores it instead of resetting to -1.
        if current_ngl is not None:
            try:
                _last_ngl_value["v"] = int(current_ngl)
            except (TypeError, ValueError):
                pass
        return gr.update(value=0, interactive=False,
                         info="Encoder Backend is CPU — all layers run on CPU.")

    _cfg_w["enc_backend_dd"].change(
        _on_enc_backend_change,
        inputs=[_cfg_w["enc_backend_dd"], _cfg_w["enc_ngl_dd"]],
        outputs=_cfg_w["enc_ngl_dd"],
    )

    def _on_img_backend_change(backend_choice: str, current_placement):
        if "Vulkan" in backend_choice:
            restore = _last_placement_value["v"]
            return gr.update(value=restore, interactive=True,
                             info="Split keeps the encoder+VAE on CPU, diffusion model on GPU.")
        if current_placement and current_placement != configure.DIFFUSER_PLACEMENT_FULL_CPU:
            _last_placement_value["v"] = current_placement
        return gr.update(value=configure.DIFFUSER_PLACEMENT_FULL_CPU, interactive=False,
                         info="ImageGen Backend is CPU — sd.cpp will not touch the GPU at all.")

    _cfg_w["img_backend_dd"].change(
        _on_img_backend_change,
        inputs=[_cfg_w["img_backend_dd"], _cfg_w["img_placement_dd"]],
        outputs=_cfg_w["img_placement_dd"],
    )


def _wire_config_events(status_box: gr.Textbox) -> None:
    """Register Configuration tab save event that outputs to shared status_box."""
    w = _cfg_w

    def save_all(ep, en, dp, dn, vp, vn,
                 enc_back, img_back, threads,
                 eb, ec, engl, ef,
                 ic, img_placement, pt):
        enc_parsed = configure.parse_backend_choice(enc_back)
        img_parsed = configure.parse_backend_choice(img_back)
        configure.update_persistent({
            "encoder_model_path":  ep,  "encoder_model_name":  en,
            "imagegen_model_path": dp,  "imagegen_model_name": dn,
            "vae_model_path":      vp,  "vae_model_name":      vn,
            "backend_encoder":     enc_back,
            "backend_imagegen":    img_back,
            # Per-side, and READ per-side by inference.py. The old code also
            # wrote a shared "vulkan_device" from the ImageGen dropdown only,
            # while enhance_prompt() read that same key for the ENCODER -- so
            # picking "CPU" for ImageGen silently set the encoder's device to
            # -1. The two keys below were already being written and never read.
            "encoder_vulkan_device": enc_parsed["vulkan_device"],
            "imagegen_vulkan_device": img_parsed["vulkan_device"],
            "encoder_threads":     int(threads),
            "imagegen_threads":    int(threads),
            "encoder_batch_size":  int(eb),
            "encoder_ctx_size":    int(ec),
            "encoder_gpu_layers":  int(engl),
            "encoder_flash_attn":  bool(ef),
            "imagegen_clip_skip":  int(ic),
            "imagegen_placement":  img_placement,
            "prompt_template":     pt,
            "first_run":           False,
        })
        return "All settings saved!"

    w["save_all_btn"].click(
        save_all,
        inputs=[
            w["enc_path_tb"], w["enc_name_tb"], w["diff_path_tb"], w["diff_name_tb"],
            w["vae_path_tb"], w["vae_name_tb"],
            w["enc_backend_dd"], w["img_backend_dd"], w["threads_dd"],
            w["enc_batch_dd"], w["enc_ctx_dd"],
            w["enc_ngl_dd"], w["enc_flash_chk"],
            w["img_clip_dd"], w["img_placement_dd"], w["prompt_template_tb"],
        ],
        outputs=status_box,
    )

# ---------------------------------------------------------------------------
# Tab 3 — Debug / Info
# ---------------------------------------------------------------------------

def _collect_debug() -> str:
    """
    Hardware and build report, plus the raw constants.ini.

    This exists so hardware problems are visible instead of inferred. It is
    also the home for utilities.get_memory_info / check_prerequisites /
    get_relevant_env / get_build_status, which previously had no caller at all
    -- a diagnostics toolkit with nothing to diagnose. The GPU section is the
    important one: it shows the exact device indices the program will pass to
    `-dev Vulkan<N>` and `--backend vulkan<N>`, as reported by ggml itself.
    """
    try:
        import scripts.utilities as utilities
        W = 60
        L: List[str] = []

        def rule(title: str) -> None:
            L.append("=" * W)
            L.append(f"  {title}")
            L.append("=" * W)

        # --- CPU ---
        cpu = configure.get_cpu_info()
        rule("CPU")
        L.append(f"  Brand        : {cpu.get('brand')}")
        L.append(f"  Vendor       : {cpu.get('vendor')}")
        L.append(f"  Cores        : {cpu.get('cores_logical')} logical / "
                 f"{cpu.get('cores_physical')} physical")
        L.append(f"  Threads      : {cpu.get('default_threads')} default (85% of logical)")
        feats = [f["name"] for f in configure.CPU_FEATURES if cpu.get(f["key"])]
        L.append(f"  Features     : {', '.join(feats) if feats else 'none reported'}")
        L.append(f"  Build arch   : {cpu.get('arch_selection')}")
        L.append(f"  AOCL present : {cpu.get('has_aocl')}  (detected only; not wired into the build)")
        L.append("")

        # --- Memory ---
        mem = utilities.get_memory_info()
        if mem:
            rule("MEMORY")
            L.append(f"  RAM          : {mem.get('ram_used_mb')} / "
                     f"{mem.get('ram_total_mb')} MB used ({mem.get('ram_percent')}%)")
            L.append("")

        # --- GPU ---
        vk = configure.get_vulkan_info()
        rule("GPU / VULKAN")
        L.append(f"  Install type : {configure.get_install_type()}")
        L.append(f"  Vulkan       : {vk.get('available')}  (version {vk.get('version')})")
        L.append(f"  SDK          : {vk.get('sdk') or 'not set'}")
        L.append(f"  Enumerated by: {vk.get('enumerated_by')}")
        if vk["devices"]:
            L.append("  Devices ggml will accept:")
            for d in vk["devices"]:
                L.append(f"    {d['backend']}{d['index']}: {d['name']}")
                L.append(f"        {d['vram_total_mb']} MiB total, "
                         f"{d['vram_free_mb']} MiB free at install time")
            L.append("")
            L.append("  The index above is what is passed to -dev Vulkan<N>")
            L.append("  and --backend vulkan<N>. It is ggml's own numbering.")
        else:
            L.append("  Devices      : none")
            L.append("  (CPU-only install, or ggml found no usable GPU.)")
        L.append("")

        # --- Backends / build tools ---
        bs = utilities.get_build_status()
        rule("BACKEND BINARIES")
        L.append(f"  llama-completion : {bs['llama_path'] or 'NOT BUILT'}")
        L.append(f"  sd-cli           : {bs['sd_path'] or 'NOT BUILT'}")
        pre = utilities.check_prerequisites()
        L.append(f"  cmake            : {pre['cmake_path'] or 'not found'}")
        L.append(f"  git              : {pre['git_path'] or 'not found'}")
        L.append("")

        # --- Models ---
        c = configure.load_persistent()
        rule("MODELS")
        for label, key in (("Encoder  ", "encoder_model_path"),
                           ("Diffusion", "imagegen_model_path"),
                           ("VAE      ", "vae_model_path")):
            p = c.get(key, "")
            state = "OK" if p and Path(p).exists() else "NOT SET / MISSING"
            L.append(f"  {label}: {state}")
            if p:
                L.append(f"             {p}")
        L.append("")

        # --- Env ---
        rule("RELEVANT ENVIRONMENT")
        env = utilities.get_relevant_env()
        if env:
            for k, v in env.items():
                L.append(f"  {k} = {v}")
        else:
            L.append("  (none set)")
        L.append("")

        # --- constants.ini verbatim ---
        rule("CONSTANTS.INI")
        constants_path = configure.get_constants_path()
        if constants_path.exists():
            try:
                with open(constants_path, "r", encoding="utf-8") as _f:
                    L.append(_f.read())
            except Exception as _e:
                L.append(f"  (error reading constants.ini: {_e})")
        else:
            L.append(f"  (constants.ini not found at: {constants_path})")
        L.append("=" * W)
        return "\n".join(L)
    except Exception:
        return f"Error collecting debug info:\n{traceback.format_exc()}"


def _copy_to_clipboard(text: str) -> str:
    if not text:
        return "Nothing to copy."
    try:
        if sys.platform == "win32":
            proc = subprocess.Popen(["clip.exe"],
                                    stdin=subprocess.PIPE, text=True)
            proc.communicate(text, timeout=10)
            return "Copied to clipboard!"
        return "Clipboard copy only supported on Windows."
    except Exception as e:
        return f"Copy failed: {e}"


def _build_debug_tab_inner() -> gr.Textbox:
    """Build Debug tab widgets; info section above, debug info below."""
    with gr.Group():
        gr.Markdown("### Image-Gradio-Gguf")
        gr.HTML(
            "<p>A Windows local image generator using Gradio, llama.cpp and stable-diffusion.cpp, by "
            "<a href=\"mailto:wiseman-timelord@mail.com\">WiseMan-Time-Lord</a> at "
            "<a href=\"http://wisetime.rf.gd/\">WiseTime.Rf.Gd</a></p>"
            "<p><strong>Where you may find this and my other programming projects on </strong>"
            "<a href=\"https://github.com/wiseman-timelord\">GitHub</a></p>"
            "<p><strong>Support/Donate to assist in the continuation of my projects at, </strong>"
            "<a href=\"https://patreon.com/WiseManTimeLord\">Patreon</a>, "
            "<a href=\"https://ko-fi.com/WiseManTimeLord\">Ko-Fi</a></p>",
            elem_classes=["info-textbox-match"],
        )
        with gr.Row():
            _dbg["refresh_btn"] = gr.Button("Refresh", variant="primary")
            _dbg["copy_btn"]    = gr.Button("Copy to Clipboard")

        # Pre-populate with debug info (no need for app.load)
        _dbg["info_text"] = gr.Textbox(
            label="Debug Info",
            interactive=False,
            lines=14, max_lines=30,
            autoscroll=False,
            value=_collect_debug(),   # Call directly to show info on load
        )

        _dbg["refresh_btn"].click(_collect_debug, outputs=_dbg["info_text"])

    return _dbg["info_text"]


def _wire_debug_events(status_box: gr.Textbox) -> None:
    """Register Debug tab copy event that outputs to shared status_box."""
    _dbg["copy_btn"].click(
        _copy_to_clipboard, inputs=_dbg["info_text"], outputs=status_box
    )


# ---------------------------------------------------------------------------
# App assembly
# ---------------------------------------------------------------------------

def build_app():
    """Assemble and return (app, css) for Gradio 6+.
    css must be passed to launch() rather than the Blocks constructor."""
    configure.ensure_data_dirs()

    _css = """
#exit-btn {
min-height: 3.5rem !important;
background: #a93226 !important;
border-color: #922b21 !important;
color: #fff !important;
font-weight: 700 !important;
font-size: 1rem !important;
}
#exit-btn:hover { background: #c0392b !important; }

/* ── Preview box: the box height is driven by configure.PREVIEW_IMAGE_HEIGHT
(see the __PREVIEW_IMG_HEIGHT__px placeholder below, substituted once at the
end of this string) — the SAME value passed to the gr.Image(height=...)
kwarg above, so the two can never disagree. gr.Image in Gradio 6.19.0 has
NO object_fit kwarg (only gr.Gallery does), so fit-to-box behavior must be
driven entirely by CSS here. Gradio's own native rule is `.image-frame img
{ width:100%; height:100%; object-fit:scale-down }` — scale-down only ever
shrinks an oversized image, it never enlarges one smaller than the box
(e.g. a 256x256 generation inside this box), which is why small images
rendered tiny. We replicate the same width/height:100% sizing (required for
object-fit to have any box to fit against) but swap in object-fit:contain
so it scales BOTH directions — shrinking large images and enlarging small
ones, always preserving aspect ratio. The Svelte scope hash on Gradio's own
rule can out-rank generic selectors depending on load order, so target the
structural classes directly with #id + class stacking to win the cascade
regardless of load order. ─────────────────────────────────────────────── */
#preview-img.gradio-container,
#preview-img,
#preview-img .image-container,
#preview-img .image-container.svelte-12vrxzd,
#preview-img .image-frame,
#preview-img .image-frame.svelte-12vrxzd {
height: __PREVIEW_IMG_HEIGHT__px !important;
max-height: __PREVIEW_IMG_HEIGHT__px !important;
}
#preview-img img,
#preview-img .image-frame img,
#preview-img .image-frame.svelte-12vrxzd img {
width: 100% !important;
height: 100% !important;
object-fit: contain !important;
}

/* ── Gallery Thumbnails: force contain to prevent clipping ───────────────
Gradio 6's grid gallery renders each cell as <div class="thumbnail-lg ...">
  <img> inside, with object-fit driven by a `--object-fit` CSS variable that
  Gradio sets inline on .grid-container. That variable mechanism is fragile
  (relies on the object_fit= prop reaching the right node), and the rule
  that consumes it carries a Svelte scope hash (e.g. .svelte-7anmrz) which
  changes between Gradio versions and can out-rank generic selectors. So we
  target the real structural class directly with #id + class stacking and
  !important, the same approach used for #preview-img above — this wins
  the cascade regardless of the scope hash or load order, and no longer
  depends on the --object-fit variable being set correctly at all. We also
  keep the old .gallery-item/.grid-wrap/.thumbnail-item selectors as a
  harmless fallback in case a future Gradio version reintroduces them. ── */
#output-gallery .thumbnail-lg > img,
#output-gallery .thumbnail-lg img,
#output-gallery .grid-wrap img,
#output-gallery .gallery-item img,
#output-gallery .thumbnail-item img,
#output-gallery img {
    object-fit: contain !important;
    width: 100% !important;
    height: 100% !important;
    max-width: 100% !important;
    max-height: 100% !important;
}
#output-gallery .thumbnail-lg,
#output-gallery .gallery-item,
#output-gallery .grid-wrap .gallery-item {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    overflow: hidden !important;
    background: var(--background-fill-secondary) !important;
}

/* ── Gallery: auto horizontal scrollbar, no vertical scroll ──────────────
overflow-x:auto lets the browser decide — scrollbar appears only when the
thumbnails actually overflow the container width, and disappears when they
all fit. This avoids a dead non-interactive track when there are few images.
overflow-x:scroll would always show the track regardless of content width. ── */
#output-gallery .grid-wrap {
    overflow-x: auto !important;
    overflow-y: hidden !important;
    scrollbar-width: thin !important;      /* Firefox: thin bar when visible */
}
"""
    # Substitute the preview-box height placeholder with the single shared
    # constant (configure.PREVIEW_IMAGE_HEIGHT) — same value used for the
    # gr.Image(height=...) kwarg, so the two can never drift apart again.
    _css = _css.replace("__PREVIEW_IMG_HEIGHT__", str(configure.PREVIEW_IMAGE_HEIGHT))

    with gr.Blocks(title="Image-Gradio-Gguf") as app:
        gr.Markdown("# Image-Gradio-Gguf")

        # ── Tabs ──────────────────────────────────────────────────────────────
        with gr.Tabs():
            with gr.TabItem("Generate"):
                _build_generate_tab_inner()

            with gr.TabItem("Configuration"):
                _build_config_tab_inner()

            with gr.TabItem("Debug / Info"):
                info_text = _build_debug_tab_inner()

        # ── Unified bottom bar (spans below all tabs) ─────────────────────────
        with gr.Row(elem_id="bottom-bar"):
            shared_status = gr.Textbox(
                value="Ready.",
                label=None,
                show_label=False,
                interactive=False,
                container=False,
                placeholder="Ready.",
                elem_id=configure.STATUS_BAR_KEY,
                scale=9,
            )
            exit_btn = gr.Button(
                "Exit Program",
                variant="stop",
                scale=1,
                elem_id="exit-btn",
                min_width=140,
            )

        exit_btn.click(_handle_exit_click, inputs=[], outputs=[])

        # Wire all per-tab events to the shared status box.
        _wire_generate_events(shared_status)
        _wire_config_events(shared_status)
        _wire_debug_events(shared_status)

    return app, _css