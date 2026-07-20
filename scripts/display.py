#!/usr/bin/env python3
"""
display.py - Gradio 6 UI for Image-Gradio-Gguf.
Four tabs: Generate | Configuration | Preferences | Debug / Info
Build/install functionality lives in installer.py only.

Two settings files, one per page, and no key appears in both:
  data/configuration.json - Configuration page (models, backends, threads),
                            plus the Generate page's auto-saved generation
                            settings and the Qt window geometry.
  data/preferences.json   - Preferences page (prompt template, max thumbnails).

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
    """Fresh load of data/configuration.json — the Configuration page's file."""
    return configure.load_configuration()


def _prefs() -> Dict[str, Any]:
    """Fresh load of data/preferences.json — the Preferences page's file.

    Kept separate from _cfg() on purpose: the two pages own two files, and
    nothing should be able to write a preference through a configuration save
    or vice versa. inference.py still receives a single merged dict (see
    do_generate), because a generation needs values from both.
    """
    return configure.load_preferences()


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
        cfg = configure.load_configuration()
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
            configure.update_configuration({"last_model_browse_dir": selected_dir})
            return path
            
        return ""
    except Exception:
        return ""


def _open_output_folder() -> str:
    """Open a native Windows Explorer window on .\\output.

    explorer.exe routinely exits with a non-zero code even on a fully
    successful launch (e.g. if a window for that folder was already open),
    so success/failure is judged by whether Popen could start the process
    at all, not by its return code — checking the return code would flag
    normal explorer behaviour as an error.
    """
    out_dir = configure.get_output_dir()
    try:
        subprocess.Popen(["explorer", str(out_dir)])
        return f"Opened folder: {out_dir}"
    except Exception as e:
        return f"ERROR: could not open output folder: {e}"


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


def _vae_name_matches_family(filename: str, family: Optional[str]) -> bool:
    """True when a safetensors filename is the VAE the given diffuser family
    wants. For Z-Image that is ae.safetensors; for Flux.2 it is flux2_ae (which
    frequently is NOT name-detectable, so this legitimately returns False for
    the generic diffusion_pytorch_model.safetensors and the user picks by hand).
    When family is None, fall back to the historic ae.safetensors match so a
    flat models folder still behaves as before.
    """
    low = filename.lower()
    if family == configure.DIFFUSER_FAMILY_FLUX2:
        return configure.vae_family(low) == configure.DIFFUSER_FAMILY_FLUX2
    # z-image or unknown -> the exact ae.safetensors filename
    return low == "ae.safetensors"


def _detect_vae(diff_path_str: str) -> Tuple[str, str]:
    r"""
    Given a diffusion model path, locate the VAE that matches its family and
    return (vae_full_path, vae_stem), or ("", "") if not found.

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

    # Family of the NEW diffuser decides which VAE filename we hunt for.
    family = configure.diffuser_family(diff_path_str)

    def _scan(folder: Path) -> Optional[Path]:
        try:
            for f in folder.iterdir():
                if f.is_file() and f.name.lower().endswith(".safetensors") \
                        and _vae_name_matches_family(f.name, family):
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
            if f.is_file() and _vae_name_matches_family(f.name, family):
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
    # Rule 0 (new): cross-family switch. If the VAE currently held belongs to
    # the OTHER family (z-image ae vs flux2 vae), it can never be valid for the
    # new diffuser, so blank it unconditionally and make the user set the right
    # one. This is the flux.2 ⇄ z-image switch the user asked to be handled.
    new_family = configure.diffuser_family(diff_path_str) if diff_path_str else None
    cur_vae_family = configure.vae_family(current_vae_name or "")
    if new_family and cur_vae_family and new_family != cur_vae_family:
        return "", ""

    vae_path, vae_name = _detect_vae(diff_path_str)
    if vae_path:
        return vae_path, vae_name
    if current_vae_path and Path(current_vae_path).expanduser().exists():
        return gr.update(), gr.update()          # rule 2 — leave as-is
    return "", ""


def _diff_change_message(diff_path_str: str, current_vae_name: str) -> str:
    """Status-bar line for a diffusion-model change: announces the family the
    interface has switched to, and prompts for a VAE path on a cross-family
    switch (where _resolve_vae just blanked the box).

    Examples:
      "Handling/interface set to Z-Image-Turbo."
      "Switched from Z-Image-Turbo to Flux.2-Klein — set VAE (safetensors) path."
    """
    if not diff_path_str:
        return "Handling/interface set to no model selected."
    new_family = configure.diffuser_family(diff_path_str)
    new_label = configure.diffuser_family_label(diff_path_str)
    cur_vae_family = configure.vae_family(current_vae_name or "")
    if new_family and cur_vae_family and new_family != cur_vae_family:
        old_label = configure.DIFFUSER_FAMILY_LABELS.get(cur_vae_family, "the previous model")
        return f"Switched from {old_label} to {new_label} — set VAE (safetensors) path."
    return f"Handling/interface set to {new_label}."


def _vae_hint_update(diff_path_str: str) -> Any:
    """gr.update(info=...) naming the VAE the current diffuser family expects."""
    fam = configure.diffuser_family(diff_path_str) if diff_path_str else None
    hint = configure.DIFFUSER_VAE_HINTS.get(
        fam, "Auto-filled when found next to the diffusion model.")
    return gr.update(info=hint)


def _on_diff_path_change_full(path: str, current_vae_path: str,
                              current_vae_name: str) -> Tuple[Any, Any, Any]:
    """Combined diffusion-path change handler used in _wire_config_events so it
    can also write the shared status bar. Returns updates for
    (vae_path_tb, vae_name_tb, status_box). The family hint is folded INTO the
    vae_name update so a single widget is not targeted twice in one event."""
    vae_p, vae_n = _resolve_vae(path, current_vae_path, current_vae_name)
    fam = configure.diffuser_family(path) if path else None
    hint = configure.DIFFUSER_VAE_HINTS.get(
        fam, "Auto-filled when found next to the diffusion model.")
    # vae_n is one of: "" (blank), a stem string, or gr.update() (keep as-is).
    if isinstance(vae_n, str):
        vae_n = gr.update(value=vae_n, info=hint)
    else:
        vae_n = gr.update(info=hint)   # keep value, just retune the hint
    msg = _diff_change_message(path, current_vae_name)
    return vae_p, vae_n, msg


# ---------------------------------------------------------------------------
# Recent images helpers
# ---------------------------------------------------------------------------


# Module-level cache for recent images – stores the full list, not limited
_gallery_cache: Dict[str, Any] = {
    "full_list": None,      # list of all image paths (unlimited)
    "mtime": 0.0,
}

def _get_recent_images(max_images: Optional[int] = None) -> List[str]:
    """Return paths of images in the output folder, newest first.
       Caches the full list of images and slices it according to max_images.
       Prints a concise start/end summary only when a rescan occurs.

       max_images=None means "however many the Preferences page says"
       (Max Thumbnails Displayed). The cache holds the unsliced list, so
       changing that preference re-slices what is already in memory rather
       than forcing another disk sweep. Callers that want a specific count
       — _idle_preview_image() only needs the single newest file — still
       pass one explicitly.
    """
    if max_images is None:
        max_images = configure.get_max_thumbnails()
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


def _model_path_ok(value: Any) -> bool:
    """True when a saved model-path value points at a file that is on disk now.

    Deliberately the SAME test inference.py makes before it runs (a plain
    Path(...).exists() on the raw saved string — no expanduser, no
    resolve_model_path fallback). If this were the more forgiving of the two,
    the Generate button would appear for a path inference then refuses to load,
    and the run would fail with nothing on screen to explain why. Equal tests
    mean the button is visible exactly when a generation can actually start.

    A non-string (a null from a hand-edited configuration.json) and an
    unopenable path (embedded null, bad drive letter — OSError, not False)
    both count as not-ok rather than raising out of a UI refresh.
    """
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        return Path(value).exists()
    except OSError:
        return False


def _missing_models() -> List[str]:
    """Labels of the model files that are not set, or set but gone from disk.

    All three are mandatory for every run — a Z-Image diffusion gguf is the
    DiT alone, so it cannot condition without the Qwen3 encoder (--llm) nor
    reach pixels without the VAE (--vae). Checked as one group so the user is
    told everything that is missing at once, rather than fixing the diffusion
    model only to be sent back for the VAE.

    Read from configuration.json, NOT from the Configuration page's textboxes:
    a path typed but not saved is not yet a path this program will run with.
    """
    c = configure.load_configuration()
    labels = [("encoder_model_path", "Encoder"),
              ("imagegen_model_path", "Diffusion"),
              ("vae_model_path", "VAE")]
    return [label for key, label in labels if not _model_path_ok(c.get(key))]


def _models_configured() -> bool:
    """Return True if all three model paths are set and exist on disk."""
    return not _missing_models()


# The Generate button lives in a Row whose visibility is this gate. The gate is
# re-evaluated from disk, never remembered: the widgets are built ONCE per
# launch, so a `configured` value captured at build time is a snapshot of how
# things looked before the user had configured anything, and it is wrong the
# moment they save a model path. Every event that can change the answer calls
# _generate_gate_updates() and pushes the result at the same three widgets.
_PROMPT_PH_READY   = "Describe the image you want to generate..."
_PROMPT_PH_NOMODEL = "Set locations of models on Configuration page first..."
_NEG_PH_READY      = "Things to exclude..."
_NEG_PH_NOMODEL    = _PROMPT_PH_NOMODEL


def _generate_gate_updates() -> Tuple[Any, Any, Any]:
    """(generate_row, prompt_tb, negative_tb) updates for the current state.

    Returned in the order the outputs= lists below expect. Placeholders move
    with the button: if the row is hidden there has to be something on screen
    saying why, and the prompt box is where the user is looking.
    """
    configured = _models_configured()
    return (
        gr.update(visible=configured),
        gr.update(placeholder=_PROMPT_PH_READY if configured else _PROMPT_PH_NOMODEL),
        gr.update(placeholder=_NEG_PH_READY if configured else _NEG_PH_NOMODEL),
    )


def _generate_family_updates(cur_steps: Any = None, cur_cfg: Any = None,
                             cur_width: Any = None, cur_height: Any = None
                             ) -> Tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
    """(settings_header, ref_row, status, sampler_dd, steps_dd, cfg_scale_sld,
    width_dd, height_dd) for the current diffuser.

    Keeps the "Settings (Flux 2 / Z-Image-Turbo / no model selected)" header and
    the Flux.2-only controls in step with whatever model is set, flashes any
    encoder/VAE compatibility problem on arrival, leaves the Sampler untouched
    (euler_a default suits both families), and swaps the Diffuse Steps, CFG-Scale
    and Width/Height choices to the set that fits the family: distilled klein →
    steps 4-8, cfg ~1.0, sizes 512-1536; klein-base → ~20 steps, cfg ~4; Z-Image
    → 8 steps, cfg ~1-2, sizes 256-1536. (Flux.2's 512 floor is deliberate — it
    degrades below that.)

    Current steps/cfg/width/height are passed in so a value already valid for the
    new family is KEPT (no clobbering a deliberate choice); only an out-of-range
    value is snapped to the family default. Returned in the generate_tab.select
    outputs order.
    """
    c = configure.load_configuration()
    diff = c.get("imagegen_model_path", "")
    fam_label = configure.diffuser_family_label(diff)
    is_flux2 = (configure.diffuser_family(diff)
                == configure.DIFFUSER_FAMILY_FLUX2)
    ok, msg = inference.check_model_compatibility(c)
    status = gr.update() if ok else gr.update(value="⚠ " + msg)
    # Sampler is NOT touched by family selection — euler_a is the default for
    # both families and empirically the better choice for Flux.2 on this build,
    # so leave whatever the user has set (no reset-to-euler).
    sampler_upd = gr.update()

    spec = configure.family_step_cfg(diff)
    step_choices, step_default = spec["steps"]
    cfg_min, cfg_max, cfg_step, cfg_default = spec["cfg"]

    try:
        cs = int(cur_steps)
    except (TypeError, ValueError):
        cs = None
    step_val = cs if cs in step_choices else step_default
    steps_upd = gr.update(choices=step_choices, value=step_val)

    try:
        cc = float(cur_cfg)
    except (TypeError, ValueError):
        cc = None
    cfg_val = cc if (cc is not None and cfg_min <= cc <= cfg_max) else cfg_default
    cfg_upd = gr.update(minimum=cfg_min, maximum=cfg_max, step=cfg_step, value=cfg_val)

    # Width / height: swap to the family's allowed sizes, keeping a valid current
    # value, else snapping to 768 (a safe native-ish default for both families).
    sizes = configure.family_image_sizes(diff)
    def _size_upd(cur: Any) -> Any:
        try:
            cv = int(cur)
        except (TypeError, ValueError):
            cv = None
        return gr.update(choices=sizes, value=(cv if cv in sizes else 768))
    width_upd = _size_upd(cur_width)
    height_upd = _size_upd(cur_height)

    return (
        gr.update(value=f"### Settings ({fam_label})"),
        gr.update(visible=is_flux2),
        status,
        sampler_upd,
        steps_upd,
        cfg_upd,
        width_upd,
        height_upd,
    )


def _build_generate_tab_inner() -> None:
    """Build Generate tab widgets; store refs in _gen for later wiring."""
    cfg = _cfg()
    presets = configure.get_generation_presets()
    configured = _models_configured()

    prompt_ph  = _PROMPT_PH_READY if configured else _PROMPT_PH_NOMODEL
    neg_ph     = _NEG_PH_READY    if configured else _NEG_PH_NOMODEL

    # Settings header names the active diffuser family, per the requested
    # "Settings (Flux 2)" / "Settings (Z-Image-Turbo)" / "Settings (no model
    # selected)". Rebuilt once at launch; refreshed on tab-select by
    # _generate_family_updates() so it tracks a model chosen after launch.
    _initial_family_label = configure.diffuser_family_label(
        cfg.get("imagegen_model_path", ""))

    with gr.Row():
        # ── Column 1: settings only ─────────────────────────────────────────
        # Prompts and the Generate button moved to column 2 so the three
        # sections read left-to-right as: tune → describe/input/go → view.
        with gr.Column(scale=1):
            _gen["settings_header"] = gr.Markdown(f"### Settings ({_initial_family_label})")
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
                _gen["width_dd"]  = gr.Dropdown(label="Image Width",  choices=configure.IMAGE_SIZES,
                                                value=cfg.get("imagegen_width", 512))
                _gen["height_dd"] = gr.Dropdown(label="Image Height", choices=configure.IMAGE_SIZES,
                                                value=cfg.get("imagegen_height", 512))
            with gr.Row():
                _gen["steps_dd"] = gr.Dropdown(
                    label="Diffuse Steps", choices=configure.STEP_CHOICES,
                    value=cfg.get("imagegen_steps", 4),
                )
                _gen["cfg_scale_sld"] = gr.Slider(
                    label="CFG Scale", minimum=0.5, maximum=20.0, step=0.5,
                    value=cfg.get("imagegen_cfg_scale", 1.0),
                )
            with gr.Row():
                _gen["batch_dd"] = gr.Dropdown(label="Batch Count",
                                               choices=configure.BATCH_COUNT_CHOICES,
                                               value=cfg.get("imagegen_batch_count", 1)
                )
                _gen["output_fmt_dd"] = gr.Dropdown(
                    label="Output Format", choices=configure.OUTPUT_FORMATS,
                    value=cfg.get("output_format", "png"),
                )
            with gr.Row():
                _gen["seed_num"] = gr.Number(label="Gen Seed (-1 = random)",
                                             value=cfg.get("imagegen_seed", -1), precision=0)
            # NOTE: no manual "Save as Default" button — successful generations
            # auto-save their settings panel values (see do_generate's success
            # branch), so the next launch picks up the last settings that worked.

        # ── Column 2: prompts, dynamic image input, Generate button ──────────
        with gr.Column(scale=1):
            gr.Markdown("### Prompts")
            # ── Positive Prompt, with a "(history)" popout ──────────────────
            # The label itself is the toggle: a gr.Button stripped of button
            # chrome by #positive-history-toggle CSS, reading as plain label
            # text until clicked. Clicking opens/closes the recent-prompts
            # panel (same handler both directions). History is pre-loaded at
            # build time and re-fetched on every toggle so a just-made
            # generation shows without a reload.
            _gen["positive_history_toggle"] = gr.Button(
                "Positive Prompt (click for history)",
                elem_id="positive-history-toggle",
            )
            _gen["prompt_tb"] = gr.Textbox(
                show_label=False,
                placeholder=prompt_ph,
                lines=2, max_lines=10,
                value=cfg.get("last_prompt", ""),
                elem_id="prompt-positive",
            )
            _positive_history = configure.get_prompt_history("positive")
            with gr.Column(visible=False, elem_id="positive-history-panel") as _gen["positive_history_panel"]:
                _gen["positive_history_btns"] = []
                for _hist_text in _positive_history:
                    _gen["positive_history_btns"].append(
                        gr.Button(_hist_text, visible=False,
                                 elem_classes=["prompt-history-item"])
                    )
            _gen["positive_history_state"] = gr.State(False)

            # ── Negative Prompt, same "(history)" popout pattern ─────────────
            _gen["negative_history_toggle"] = gr.Button(
                "Negative Prompt (click for history)",
                elem_id="negative-history-toggle",
            )
            _gen["negative_tb"] = gr.Textbox(
                show_label=False,
                placeholder=neg_ph,
                lines=2, max_lines=10,
                value=cfg.get("negative_prompt", ""),
                elem_id="prompt-negative",
            )
            _negative_history = configure.get_prompt_history("negative")
            with gr.Column(visible=False, elem_id="negative-history-panel") as _gen["negative_history_panel"]:
                _gen["negative_history_btns"] = []
                for _hist_text in _negative_history:
                    _gen["negative_history_btns"].append(
                        gr.Button(_hist_text, visible=False,
                                 elem_classes=["prompt-history-item"])
                    )
            _gen["negative_history_state"] = gr.State(False)

            # ── Flux.2 controls: flash-attn toggle + reference images ───────
            # Whole column shown ONLY when the diffuser is Flux.2.
            _flux2_now = (configure.diffuser_family(cfg.get("imagegen_model_path", ""))
                          == configure.DIFFUSER_FAMILY_FLUX2)
            with gr.Column(visible=_flux2_now) as _gen["ref_row"]:
                gr.Markdown("#### Image Edit (img+txt to img)")
                # Reference images for image-to-image / editing (sd.cpp -r,
                # repeatable). "Add Image" opens a file picker and APPENDS to
                # the list (add, then add again); "Clear Images" empties it. No
                # drop-zone. The chosen files are listed one per line below and
                # are cleared automatically when a generation completes.
                with gr.Row():
                    _gen["ref_add_btn"] = gr.UploadButton(
                        "Add Image", file_count="multiple",
                        file_types=["image"], type="filepath", size="sm",
                    )
                    _gen["ref_clear_btn"] = gr.Button("Clear Images", size="sm")
                _gen["ref_list_tb"] = gr.Textbox(
                    show_label=False, interactive=False, visible=False,
                    lines=1, max_lines=8, elem_id="ref-image-list",
                )
                # Accumulated list of reference-image paths (the real input to
                # generation); the textbox above is just its visible form.
                _gen["ref_images_state"] = gr.State([])

            gr.Markdown("#### Submitting Input (check settings)")
            with gr.Row(visible=configured) as _gen["generate_row"]:
                # Single dynamic button: "Generate" (primary) when idle,
                # "..Please Wait.." (disabled) while running. do_generate()
                # flips it on entry and back on its final yield.
                _gen["generate_btn"] = gr.Button("Generate Image", variant="primary", size="lg")

        # ── Column 3: image preview only ────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### Output")
            # Single currently-selected/in-progress image — most recent
            # generation, the live phase status image, or a clicked gallery
            # thumbnail. Never shows Gradio's built-in progress bar.
            # Gradio 6 replaced show_download_button/show_share_button with a
            # single buttons=[...] list, so build the kwarg for whichever
            # major version is installed.
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
    # Styled as a gr.Button rather than gr.Markdown so it is clickable: a
    # click opens a native Windows Explorer window on .\output (see
    # _open_output_folder / the click wiring in _wire_generate_events).
    # CSS in build_app() (#thumbnails-gallery-link) strips all button chrome
    # so it still reads as a plain "### Thumbnails Gallery" heading.
    _gen["thumbnails_link"] = gr.Button(
        "Thumbnails Gallery (click to open)",
        elem_id="thumbnails-gallery-link",
    )
    # One row, always, holding ALL of Max Thumbnails Displayed (>=50) images,
    # with a horizontal scrollbar to reach the ones past the window edge. The
    # actual single-row / fixed-thumbnail-width / horizontal-overflow layout
    # is enforced by the #output-gallery CSS in build_app() (grid-auto-flow:
    # column), NOT by these columns/rows props — Gradio's native grid would
    # otherwise reflow the images into multiple width-sharing rows that shrink
    # with the window, which is exactly the bug this replaces. columns/rows
    # here are left only as Gradio's own non-authoritative defaults; the CSS
    # overrides them with !important, so their values are cosmetic. height is
    # the one number that sizes the row and is shared with the CSS via
    # configure.THUMBNAIL_GALLERY_HEIGHT so the two can never disagree; the
    # per-thumbnail width is derived from it in the CSS, keeping cells square.
    _gen["output_gallery"] = gr.Gallery(
        label="Generated Images",
        value=_get_recent_images(),
        columns=16, rows=1,
        height=configure.THUMBNAIL_GALLERY_HEIGHT,
        object_fit="contain",
        allow_preview=False,
        show_label=False,
        fit_columns=False,
        elem_id="output-gallery",
    )


    # Preset change wired immediately
    presets_map = presets

    def apply_preset(name: str):
        # Resolve the preset against the currently-loaded diffuser family, so
        # steps/cfg/sampler come out correct for Z-Image vs Flux.2 (distilled
        # or base) rather than a one-size-fits-all set. Custom/unknown leaves
        # every widget unchanged.
        model = configure.load_configuration().get("imagegen_model_path", "")
        p = configure.resolve_preset(name, model)
        if not p:
            return (gr.update(), gr.update(), gr.update(), gr.update(), gr.update())
        return (p["imagegen_width"], p["imagegen_height"],
                p["imagegen_steps"], p["imagegen_sampling"],
                p["imagegen_cfg_scale"])

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
    cfg_scale, seed, batch, output_format, quality_preset, ref_images=None):
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

        # Record this submission into prompt_cache.json's "Positive/Negative
        # Prompt (history)" log. Each field is recorded independently and is
        # a no-op if it matches the most-recent saved entry for that field,
        # so e.g. re-using the same negative prompt across several positive
        # prompts adds no duplicate negative-history rows (see
        # configure.record_prompt_history). Recorded on every submission,
        # not only on success, since the user did type it either way.
        configure.record_prompt_history("positive", prompt)
        configure.record_prompt_history("negative", negative)

        # Cancel any pending unload while we are generating
        _cancel_inactivity_timer()

        # inference.py takes ONE dict and reads keys from both files out of it
        # — enhance_prompt() wants prompt_template, which now lives in
        # preferences.json, while everything else comes from configuration.json.
        # Merged here, at the one place a generation is launched, so neither
        # inference.py nor the save handlers need to know about the split.
        gen_cfg = dict(c)
        gen_cfg["prompt_template"] = _prefs().get(
            "prompt_template", configure.DEFAULT_PROMPT_TEMPLATE)
        # Reference images (Flux.2 -r). gr.File(file_count="multiple") returns
        # a list of paths, a single path, or None; normalise to a list of
        # existing paths. inference.generate_image() ignores this entirely for
        # Z-Image and only uses it when the diffuser is Flux.2.
        if ref_images is None:
            _refs = []
        elif isinstance(ref_images, (list, tuple)):
            _refs = [str(r) for r in ref_images if r]
        else:
            _refs = [str(ref_images)]

        gen_cfg.update(
            imagegen_width=int(width), imagegen_height=int(height),
            imagegen_steps=int(steps), imagegen_sampling=sampler,
            imagegen_cfg_scale=float(cfg_scale),
            imagegen_seed=int(seed), imagegen_batch_count=int(batch),
            negative_prompt=negative,
            output_format=output_format,
            ref_images=_refs,
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
            """Build the 'Generate Stage N/2; [Batch Number X/Y; ]... Phase
            {step}/{total}...###s (prev_batch_Ns)' status string. Seconds are
            always whole numbers (no split seconds) per the fixed status-bar
            format — never decimals.

            Ordering: Generate Stage leads, because Stage 1 (encoding) runs ONCE
            up front and Stage 2 (diffusion) is what actually iterates per image
            — so Batch Number is a Stage-2 concept and is shown only then, after
            the stage, never during encoding."""
            name = _phase["name"]
            elapsed_s = int(time.time() - _phase["phase_start"])
            batch_cur = _phase["batch_current"]
            batch_tot = _phase["batch_total"]

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
                # showing a fabricated "0/0". No Batch Number here: encoding
                # is a one-time Stage-1 step, not per-image.
                step = _phase.get("step", 0)
                total = _phase.get("total_steps", 0)
                step_part = f" {step}/{total}" if total else ""
                return (f"Generate Stage 1/2; Encoding Phase{step_part}..."
                        f"{elapsed_s}s{prev_suffix}")

            # diffusion (Stage 2) — Batch Number belongs here, after the stage.
            batch_prefix = f"Batch Number {batch_cur}/{batch_tot}; "
            step = _phase.get("step", 0)
            total = _phase.get("total_steps", 0) or int(gen_cfg.get("imagegen_steps", 0))
            step_part = f" {step}/{total}" if total else ""
            return (f"Generate Stage 2/2; {batch_prefix}Diffusing Phase{step_part}..."
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
                configure.update_configuration({
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
                configure.update_configuration({
                    "imagegen_quality_preset": quality_preset,
                })
        else:
            new_gallery = _get_recent_images()
            yield _idle_preview_image(), new_gallery, result.get("message", "Unknown error"), _btn_generate


    def on_generate_click(prompt, negative, width, height, steps, sampler,
    cfg_scale, seed, batch, output_format, quality_preset, ref_images=None):
        """Dispatch a click on the single dynamic button. The button reads
        "Generate" when idle and starts a run (delegating to the do_generate
        generator, which yields its own button-state updates). While a run
        is in progress the button is disabled and shows "..Please Wait..",
        preventing concurrent runs. ref_images is the Flux.2 input-image list
        (from ref_images_state); it is ignored for Z-Image runs inside
        inference.generate_image(). Flash attention is decided automatically
        from the selected GPU's fp16 capability — no input here."""
        yield from do_generate(prompt, negative, width, height, steps, sampler,
    cfg_scale, seed, batch, output_format, quality_preset, ref_images)

    # ── Reference-image Add / Clear handlers ────────────────────────────────
    def _render_ref_list(paths: List[str]) -> Any:
        """Show accumulated reference filenames one per line; hide when empty."""
        if not paths:
            return gr.update(value="", visible=False)
        names = "\n".join(Path(p).name for p in paths)
        return gr.update(value=names, visible=True)

    def _add_ref_images(new_files, current):
        """Append the just-picked file(s) to the accumulated list. gr.UploadButton
        hands us a path, a list of paths, or None."""
        acc = list(current or [])
        if new_files:
            if isinstance(new_files, (list, tuple)):
                acc.extend(str(f) for f in new_files if f)
            else:
                acc.append(str(new_files))
        return acc, _render_ref_list(acc)

    def _clear_ref_images():
        return [], _render_ref_list([])

    _gen["ref_add_btn"].upload(
        _add_ref_images,
        inputs=[_gen["ref_add_btn"], _gen["ref_images_state"]],
        outputs=[_gen["ref_images_state"], _gen["ref_list_tb"]],
    )
    _gen["ref_clear_btn"].click(
        _clear_ref_images,
        inputs=None,
        outputs=[_gen["ref_images_state"], _gen["ref_list_tb"]],
    )

    _gen_evt = _gen["generate_btn"].click(
        on_generate_click,
        inputs=[_gen["prompt_tb"], _gen["negative_tb"],
        _gen["width_dd"], _gen["height_dd"], _gen["steps_dd"],
        _gen["sampler_dd"], _gen["cfg_scale_sld"],
        _gen["seed_num"], _gen["batch_dd"], _gen["output_fmt_dd"],
        _gen["preset_dd"], _gen["ref_images_state"]],
        outputs=[_gen["preview_img"], _gen["output_gallery"], status_box,
        _gen["generate_btn"]],
    )
    # Reference images are consumed at generation time, then cleared once the
    # run finishes (the requested "disappear on complete response"). Done via
    # .then so do_generate's yield contract is untouched.
    _gen_evt.then(
        _clear_ref_images,
        inputs=None,
        outputs=[_gen["ref_images_state"], _gen["ref_list_tb"]],
    )

    _gen["thumbnails_link"].click(
        _open_output_folder,
        inputs=[],
        outputs=status_box,
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

    # ── Re-check the gate whenever the user arrives at, or interacts with,
    # the Generate page. Three triggers, because each covers a hole in the
    # others:
    #   * tab select — the normal path. Configure models, save, click Generate:
    #     the button is there. This is the one that was missing, which is why
    #     the button stayed hidden for the whole session in which the models
    #     were first set up (it was only ever built once, at launch, when
    #     nothing was configured yet).
    #   * prompt focus — covers a model file appearing/vanishing on disk while
    #     the user sits on the Generate page, without a tab switch.
    #   * (see _wire_config_events) saving on the Configuration page pushes the
    #     same updates, so the gate is already correct before the tab is even
    #     clicked.
    # All three call the same _generate_gate_updates(), so they cannot drift.
    _gen["generate_tab"].select(
        _generate_gate_updates,
        inputs=None,
        outputs=[_gen["generate_row"], _gen["prompt_tb"], _gen["negative_tb"]],
    )

    # Second select handler: refresh the family-dependent bits — the Settings
    # header label, the Flux.2-only input-image row's visibility, and an
    # encoder/VAE compatibility warning — whenever the user lands on the page.
    # Separate from the gate above so neither has to know the other's outputs.
    _gen["generate_tab"].select(
        _generate_family_updates,
        inputs=[_gen["steps_dd"], _gen["cfg_scale_sld"],
                _gen["width_dd"], _gen["height_dd"]],
        outputs=[_gen["settings_header"], _gen["ref_row"], status_box,
                 _gen["sampler_dd"], _gen["steps_dd"], _gen["cfg_scale_sld"],
                 _gen["width_dd"], _gen["height_dd"]],
    )

    _gen["prompt_tb"].focus(
        _generate_gate_updates,
        inputs=None,
        outputs=[_gen["generate_row"], _gen["prompt_tb"], _gen["negative_tb"]],
    )

    _wire_prompt_history_events()


def _prompt_history_toggle_updates(kind: str, is_open: bool) -> Tuple[Any, ...]:
    """(panel_visible, row1_update, ..., row5_update, new_state) for a click
    on the "<Positive|Negative> Prompt (history)" label.

    Toggles closed -> open (fetching the current 5 most-recent entries fresh
    from prompt_cache.json and revealing whichever of them are non-empty) or
    open -> closed (hides the panel again, main edit box untouched) --
    whichever the current state calls for. A fresh fetch on every open means
    a generation submitted moments ago already shows up without a page
    reload.
    """
    new_open = not is_open
    history = configure.get_prompt_history(kind)
    updates: List[Any] = [gr.update(visible=new_open)]
    for text in history:
        updates.append(gr.update(value=text, visible=bool(new_open and text)))
    updates.append(new_open)
    return tuple(updates)


def _select_prompt_history_entry(text: str) -> Tuple[Any, Any, bool]:
    """Clicking one of the 5 history rows: load its text into the prompt box
    and close the panel, returning focus to the single edit box (same as
    clicking the label toggle a second time would)."""
    return gr.update(value=text), gr.update(visible=False), False


def _wire_prompt_history_events() -> None:
    """Wire the Positive/Negative Prompt "(history)" toggles and their 5 row
    buttons each. Split out from _wire_generate_events only for readability
    -- both fields follow the exact same open/close/select pattern, just
    against a different (kind, toggle, panel, buttons, state) tuple."""
    for kind, toggle_key, panel_key, btns_key, state_key in (
        ("positive", "positive_history_toggle", "positive_history_panel",
         "positive_history_btns", "positive_history_state"),
        ("negative", "negative_history_toggle", "negative_history_panel",
         "negative_history_btns", "negative_history_state"),
    ):
        toggle_fn = (lambda is_open, _k=kind: _prompt_history_toggle_updates(_k, is_open))
        _gen[toggle_key].click(
            toggle_fn,
            inputs=_gen[state_key],
            outputs=[_gen[panel_key], *_gen[btns_key], _gen[state_key]],
        )

    target_tb = {"positive": _gen["prompt_tb"], "negative": _gen["negative_tb"]}
    for kind, panel_key, btns_key, state_key in (
        ("positive", "positive_history_panel", "positive_history_btns", "positive_history_state"),
        ("negative", "negative_history_panel", "negative_history_btns", "negative_history_state"),
    ):
        for btn in _gen[btns_key]:
            btn.click(
                _select_prompt_history_entry,
                inputs=btn,
                outputs=[target_tb[kind], _gen[panel_key], _gen[state_key]],
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

    # ── Backend selection + CPU threads (consolidated) ──
    # First on the page: which device each side runs on decides what several
    # of the controls below are even allowed to say (GPU Layers, Diffuser
    # Placement), so it reads top-down in the order the choices actually
    # depend on each other.
    #
    # The install type ("Vulkan install" / "Cpu-only install") is deliberately
    # NOT printed here. It is a property of the install, not a setting, and it
    # is already reported on the Debug / Info page (see _collect_debug's
    # "GPU / VULKAN" section, "Install type" line). It still governs this
    # section's behaviour — a cpu_only install has no GPU entries in `choices`
    # and locks both dropdowns — it just does not announce itself.
    is_cpu_only = configure.get_install_type() == "cpu_only"

    gr.Markdown("### Backend Selection (model loading is one-shot)")
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
                # Encoder flash attention is automatic and needs no toggle:
                # inference.py passes llama.cpp --flash-attn "auto". Unlike
                # sd.cpp's Vulkan diffusion FA, llama.cpp FA on a GPU without
                # fp16/coopmat2 just falls back to CPU for the attention math
                # (correct, only slower), so it is safe on any card including a
                # no-fp16 RX 470 — no fp16 gating, no checkbox.

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

    # NOTE: Prompt Template used to sit here under an "Advanced" heading. It
    # is now on the Preferences page (_build_preferences_tab_inner) and is
    # stored in data/preferences.json, not data/configuration.json — so the
    # Save All Configuration button below neither reads nor writes it.

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
    # NOTE: the diffusion-path .change registration moved to
    # _wire_config_events so it can also write the shared status bar (the
    # "Switched to Flux.2-Klein — set VAE path" message) and update the VAE
    # box's family hint. Same _resolve_vae logic, now via
    # _on_diff_path_change_full.

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

    # Diffusion-path change: re-resolve the VAE (blanking it on a cross-family
    # z-image ⇄ flux.2 switch), flash the family/switch message, and retune the
    # VAE box's hint. Fires for Browse (which sets diff_path_tb programmatically)
    # and for hand-edits alike.
    w["diff_path_tb"].change(
        _on_diff_path_change_full,
        inputs=[w["diff_path_tb"], w["vae_path_tb"], w["vae_name_tb"]],
        outputs=[w["vae_path_tb"], w["vae_name_tb"], status_box],
    )

    def save_all(ep, en, dp, dn, vp, vn,
                 enc_back, img_back, threads,
                 eb, ec, engl,
                 ic, img_placement):
        enc_parsed = configure.parse_backend_choice(enc_back)
        img_parsed = configure.parse_backend_choice(img_back)
        configure.update_configuration({
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
            "imagegen_clip_skip":  int(ic),
            "imagegen_placement":  img_placement,
            "first_run":           False,
        })
        # Saving is the moment the model paths become real (the Generate page
        # reads configuration.json, not these textboxes), so refresh that
        # page's gate right here rather than leaving it stale until the user
        # goes looking for a button that is not there. Same helper the tab
        # select and prompt focus use.
        row_u, prompt_u, neg_u = _generate_gate_updates()
        missing = _missing_models()
        if missing:
            msg = ("Configuration saved — but Generate stays hidden until every "
                   f"model is set and on disk (missing: {', '.join(missing)}).")
        else:
            msg = "All configuration saved! Generate page is ready."
        return msg, row_u, prompt_u, neg_u

    w["save_all_btn"].click(
        save_all,
        inputs=[
            w["enc_path_tb"], w["enc_name_tb"], w["diff_path_tb"], w["diff_name_tb"],
            w["vae_path_tb"], w["vae_name_tb"],
            w["enc_backend_dd"], w["img_backend_dd"], w["threads_dd"],
            w["enc_batch_dd"], w["enc_ctx_dd"],
            w["enc_ngl_dd"],
            w["img_clip_dd"], w["img_placement_dd"],
        ],
        outputs=[status_box, _gen["generate_row"], _gen["prompt_tb"],
                 _gen["negative_tb"]],
    )

# ---------------------------------------------------------------------------
# Tab 3 — Preferences  (UI widgets + event wiring split for shared status)
# ---------------------------------------------------------------------------
# Everything here is written to data/preferences.json and NOTHING here is
# written to data/configuration.json. The split is by page, one file each:
# Configuration owns this machine's model/backend wiring (and is reseeded by
# the installer on a clean install), Preferences owns the user's standing
# taste, which survives that.
# ---------------------------------------------------------------------------

_prf: Dict[str, Any] = {}  # preferences tab widget refs


def _build_preferences_tab_inner() -> None:
    """Build Preferences tab widgets; store refs in _prf for later wiring."""
    prefs = _prefs()

    gr.Markdown("### Preferences")
    _prf["prompt_template_tb"] = gr.Textbox(
        label="Prompt Template",
        value=prefs.get("prompt_template", configure.DEFAULT_PROMPT_TEMPLATE),
        info="{prompt} is replaced with the request sent to the encoder.",
        lines=2,
    )

    with gr.Row():
        with gr.Column(scale=1):
            _prf["max_thumbs_dd"] = gr.Dropdown(
                label="Max Thumbnails Displayed",
                choices=configure.MAX_THUMBNAIL_CHOICES,
                value=configure.get_max_thumbnails(),
                info="How many images the Thumbnails Gallery shows, newest first.",
            )
        with gr.Column(scale=1):
            _prf["encoder_debug_chk"] = gr.Checkbox(
                label="Encoder Model Debug",
                value=bool(prefs.get("encoder_model_debug", False)),
                info="Print encoder output to terminal.",
            )
        # Spacer so the two controls keep a sane width instead of stretching.
        with gr.Column(scale=1):
            pass

    with gr.Row():
        _prf["save_prefs_btn"] = gr.Button("Save All Preferences",
                                           variant="primary", size="lg")


def _wire_preferences_events(status_box: gr.Textbox) -> None:
    """Register the Preferences save event.

    Saving also re-slices the Generate page's gallery, so a new Max Thumbnails
    value takes effect immediately rather than at the next launch. The cached
    output listing is unsliced (see _get_recent_images), so this costs a
    re-render, not a rescan.
    """
    def save_prefs(prompt_template, max_thumbs, encoder_debug):
        try:
            thumbs = int(max_thumbs)
        except (TypeError, ValueError):
            thumbs = configure.DEFAULT_MAX_THUMBNAILS
        if thumbs not in configure.MAX_THUMBNAIL_CHOICES:
            thumbs = configure.DEFAULT_MAX_THUMBNAILS
        configure.update_preferences({
            "prompt_template": prompt_template,
            "max_thumbnails":  thumbs,
            "encoder_model_debug": bool(encoder_debug),
        })
        return "All preferences saved!", _get_recent_images(thumbs)

    _prf["save_prefs_btn"].click(
        save_prefs,
        inputs=[_prf["prompt_template_tb"], _prf["max_thumbs_dd"],
                _prf["encoder_debug_chk"]],
        outputs=[status_box, _gen["output_gallery"]],
    )


# ---------------------------------------------------------------------------
# Tab 4 — Debug / Info
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
        # AOCL is AMD's own math library — there is no such thing as AOCL on an
        # Intel or other non-AMD part, so reporting "AOCL present : False" there
        # states a tautology and invites the user to go looking for something
        # that could never apply to their machine. Shown for AMD only.
        if configure.is_amd_cpu():
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
        c = configure.load_configuration()
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

/* ── Thumbnails Gallery heading, made clickable ───────────────────────────
This is a gr.Button standing in for what used to be a plain
gr.Markdown("### Thumbnails Gallery") heading, so it can open an Explorer
window on .\\output when clicked (see _open_output_folder / the click
wiring in _wire_generate_events). Every rule below exists purely to strip
Gradio's button chrome back off so it still looks like inert heading text
until the user's cursor says otherwise. !important is needed throughout
because the theme's own button rules are otherwise more specific. ────────── */
#thumbnails-gallery-link {
all: unset !important;
display: inline-block !important;
font-size: 1.25rem !important;
font-weight: 600 !important;
line-height: 1.6 !important;
color: var(--body-text-color) !important;
cursor: pointer !important;
margin: 0 !important;
padding: 0 !important;
}
#thumbnails-gallery-link:hover { text-decoration: underline !important; }

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

/* ── Prompt history toggles: "Positive/Negative Prompt (history)" ────────
Same trick as #thumbnails-gallery-link above: a gr.Button standing in for a
plain field label, stripped back to inert-looking text so it reads as a
label until the cursor says otherwise, but clickable to open/close the
5-row history panel beneath the box (see _wire_prompt_history_events). Sized
to match a normal Textbox label rather than the larger heading-style link,
since this sits directly above a form field, not above a page section. ── */
#positive-history-toggle,
#negative-history-toggle {
all: unset !important;
display: inline-block !important;
font-size: 0.875rem !important;
font-weight: 600 !important;
line-height: 1.4 !important;
color: var(--body-text-color) !important;
cursor: pointer !important;
margin: 0 0 2px 0 !important;
padding: 0 !important;
}
#positive-history-toggle:hover,
#negative-history-toggle:hover { text-decoration: underline !important; }

/* ── Prompt history rows: up to 5 recent prompts, one per button ─────────
Plain-looking, left-aligned, single-line rows so a long saved prompt reads
as a list entry rather than a normal centered button. Hidden rows (an empty
history slot) are handled in Python via visible=False, not CSS, so no extra
selector is needed here for that. ───────────────────────────────────────── */
.prompt-history-item {
display: block !important;
width: 100% !important;
text-align: left !important;
justify-content: flex-start !important;
white-space: nowrap !important;
overflow: hidden !important;
text-overflow: ellipsis !important;
margin-bottom: 2px !important;
}

/* ── Prompt boxes: height follows the text, at ANY window width ──────────
Gradio's own textarea auto-grow (Textbox's resize(), which writes an inline
`height: NNNpx` onto the textarea) is driven by the `input` event and nothing
else — no window resize listener, no ResizeObserver. So the height is only
ever correct for the layout width in force at the moment the last keystroke
landed. Drag the app to half a display and the same text re-wraps onto more
lines, but that stale inline pixel height stays exactly as it was: the box
still shows 2 lines and the rest of the prompt is pushed out of view. Gradio
also refuses to grow past its max_lines cap.

The fix is to stop pinning a pixel height at all and let the engine size the
box from its own content. `field-sizing: content` makes a textarea size to
the text it holds, recomputed by the browser on EVERY reflow — including the
reflow caused by the window changing width, which is precisely the event
Gradio's JS misses. `height: auto !important` is what allows that: an
!important rule in a stylesheet outranks the plain (non-important) inline
height Gradio's resize() sets, so its pixel value never applies and there is
no need to patch or fight Gradio's JS. Same reasoning for overflow-y, which
Gradio also sets inline ("scroll") once it thinks the text has outgrown the
box — with the box now always fitting the text, that scrollbar is never
wanted.

min-height restates the 2-line floor that gr.Textbox(lines=2) asks for,
because field-sizing:content sizes from content and ignores the rows
attribute the `lines` kwarg produces. 2lh is exactly two line boxes at
whatever line-height ends up computed, plus the theme's own input padding and
border width, so the arithmetic survives a theme change. There is deliberately
NO max-height: fitting all of the text is the entire point.

The whole block is wrapped in @supports so an engine without field-sizing
(Chromium < 123) gets none of it and keeps Gradio's stock behavior instead of
a broken half-fix. installer.py pins PyQt6-WebEngine 6.9, whose embedded
Chromium is 130, so the supported path is the one that actually runs here.

Scoped strictly to the two elem_ids set on the prompt boxes above: the
Thumbnails Gallery below is untouched by these rules, keeps its own fixed
123px height and one-row horizontal scroller, and its item count still comes
only from Preferences -> Max Thumbnails Displayed. ─────────────────────── */
@supports (field-sizing: content) {
#prompt-positive textarea,
#prompt-negative textarea {
field-sizing: content;
height: auto !important;
min-height: calc(2lh + (var(--input-padding, 10px) * 2) + (var(--input-border-width, 0px) * 2)) !important;
max-height: none !important;
overflow-y: hidden !important;
}
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

/* ── Gallery: ALWAYS one row of ALL thumbnails, scrolling sideways ───────
The requirement: the Thumbnails Gallery is exactly ONE row, it holds every
image Max Thumbnails Displayed asks for (>=50), and when those are wider than
the window a horizontal scrollbar appears at the bottom of the row to reach
the rest. It must NOT reflow, shrink, or drop thumbnails to fit the window —
its size is fixed and independent of the display size.

Gradio's native grid works against all three points. Its .grid-container is
`grid-template-columns: repeat(var(--grid-cols), minmax(100px,1fr))` with
`grid-auto-rows: minmax(100px,1fr)`, driven by the columns=/rows= props. With
columns=16 and 50 images that is 16 columns x 4 rows: the `1fr` makes every
column share (and shrink with) the window width, and the surplus images wrap
onto extra rows. Full-screen it happens to show ~16 in the top row so it
passes for "one row of 16"; half-screen the same images re-share the smaller
width, wrap differently, and the extra rows get clipped by overflow-y. That
is the whole bug.

The fix discards Gradio's column template entirely and rebuilds the grid as a
single row that flows horizontally:
  * grid-auto-flow: column      -> new items extend the row rightward, they
                                   never start a second row.
  * grid-template-columns: none -> throw away repeat(16, 1fr); no 1fr means
                                   nothing shares/shrinks with window width.
  * grid-template-rows: 1fr     -> exactly one explicit row track.
  * grid-auto-columns: <fixed>  -> every thumbnail cell is a FIXED width, so
                                   the row's total width = image_count * cell,
                                   growing past the window when there are many
                                   images and triggering the scrollbar below.
The fixed cell width is derived from the row height (minus the .grid-wrap
8px top+bottom padding, hence -16px) so cells stay roughly square, and it is
the SAME configure.THUMBNAIL_GALLERY_HEIGHT interpolated for the height pin —
one constant sizes the whole thing. No 1fr anywhere means window width no
longer affects the row at all: fixed size, display-independent, as required.
grid-auto-flow: column is CSS Grid Level 1 (every engine since Chrome 57), so
unlike the prompt-box field-sizing rule this needs no @supports fallback.

The horizontal scrollbar itself is the .grid-wrap rule: overflow-x:auto shows
the bar only when the row actually overflows (many images) and hides it when
everything fits (few images), avoiding a dead track; overflow-y:hidden means
the row can never grow a second line or a vertical scroll. Pinning .grid-wrap
height to the same constant keeps the row exactly one thumbnail tall.

None of this reads or caps the image COUNT — that stays entirely with
_get_recent_images()/get_max_thumbnails() (Max Thumbnails Displayed). However
many paths that returns, they all land in this one scrolling row. ────────── */
#output-gallery .grid-container {
    display: grid !important;
    grid-auto-flow: column !important;
    grid-template-columns: none !important;
    grid-template-rows: 1fr !important;
    grid-auto-columns: calc(__THUMB_GALLERY_HEIGHT__px - 16px) !important;
    height: 100% !important;
}
#output-gallery .grid-wrap {
    height: __THUMB_GALLERY_HEIGHT__px !important;
    overflow-x: auto !important;
    overflow-y: hidden !important;
    scrollbar-width: thin !important;      /* Firefox: thin bar when visible */
}
"""
    # Substitute the preview-box height placeholder with the single shared
    # constant (configure.PREVIEW_IMAGE_HEIGHT) — same value used for the
    # gr.Image(height=...) kwarg, so the two can never drift apart again.
    _css = _css.replace("__PREVIEW_IMG_HEIGHT__", str(configure.PREVIEW_IMAGE_HEIGHT))
    # Same single-source pattern for the Thumbnails Gallery row height: this
    # one constant feeds BOTH the gr.Gallery(height=...) kwarg above AND the
    # #output-gallery height/thumbnail-width CSS here, so the row height and
    # the derived square cell size can never drift apart.
    _css = _css.replace("__THUMB_GALLERY_HEIGHT__", str(configure.THUMBNAIL_GALLERY_HEIGHT))

    with gr.Blocks(title="Image-Gradio-Gguf") as app:
        gr.Markdown("# Image-Gradio-Gguf")

        # ── Tabs ──────────────────────────────────────────────────────────────
        with gr.Tabs():
            with gr.TabItem("Generation") as _gen["generate_tab"]:
                _build_generate_tab_inner()

            with gr.TabItem("Configuration"):
                _build_config_tab_inner()

            with gr.TabItem("Preferences"):
                _build_preferences_tab_inner()

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
        _wire_preferences_events(shared_status)
        _wire_debug_events(shared_status)

    return app, _css