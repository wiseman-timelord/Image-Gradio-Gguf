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
import re
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


def _gcfg() -> Dict[str, Any]:
    """Fresh load of data/generation.json — the Generation page's own file.

    The third of the three settings files, and read the same deliberate way as
    the other two: this page's widgets are seeded from here and nowhere else,
    so a Generation setting can never be silently written by a Configuration
    or Preferences save. inference.py still receives ONE merged dict — see
    configure.generation_config(), which do_generate() uses.
    """
    return configure.load_generation()


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


# Reference-image picker filters. The first (default) entry is built from
# configure.SUPPORTED_REF_IMAGE_EXTS rather than a second hand-written list,
# so the picker can never offer a format stb_image cannot decode — or hide one
# it can. PNG/JPEG get their own entries because they are what people actually
# have.
_FILETYPES_IMAGE = [
    ("Image files", " ".join(f"*{ext}"
                             for ext in sorted(configure.SUPPORTED_REF_IMAGE_EXTS))),
    ("PNG",         "*.png"),
    ("JPEG",        "*.jpg *.jpeg"),
    ("All files",   "*.*"),
]


def _browse_images() -> List[str]:
    """Open a native multi-select file dialog and return the chosen paths.

    Same tkinter approach as _browse_file() above, for the same reason: this
    runs on a Gradio worker thread on the SAME machine as the UI (the server
    is loopback-only inside the Qt window), so a native dialog is available
    and is the only way to control where the picker opens. A browser
    <input type=file> — which is what gr.UploadButton renders — cannot be
    told a starting directory by any web API, and it copies the chosen files
    into Gradio's temp folder rather than handing over the real paths.

    Opens in the last folder an image was added from (configuration.json's
    last_image_browse_dir, cached in configure.APP_STATE), defaulting to the
    user's Pictures folder on a fresh install. The folder of the LAST file in
    the selection is written back — with a multi-select they are all from the
    same folder anyway, and it is the one the dialog was sitting in.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        raw = filedialog.askopenfilenames(
            title="Add Reference Image(s)",
            initialdir=configure.get_last_image_dir(),
            filetypes=_FILETYPES_IMAGE,
        )
        # askopenfilenames normally returns a tuple, but some Tk builds hand
        # back a single brace-quoted string instead; splitlist parses both.
        if isinstance(raw, str):
            raw = root.tk.splitlist(raw)
        root.destroy()

        picked = [str(p) for p in (raw or ()) if p]
        if picked:
            configure.set_last_image_dir(str(Path(picked[-1]).parent))
        return picked
    except Exception:
        return []


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
    Load the saved Processing Method string; if it's no longer in the current
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
        return configure.DIFFUSER_FAMILY_FLUX2 in configure.vae_families(low)
    if family == configure.DIFFUSER_FAMILY_SDXL:
        # SDXL has no single canonical VAE filename the way Z-Image has
        # ae.safetensors: sdxl_vae-fp16-fix, xlVAEC_c91 and various finetune
        # VAEs are all legitimate, so membership of the SDXL set is the test.
        return configure.DIFFUSER_FAMILY_SDXL in configure.vae_families(low)
    # z-image, flux1, or unknown -> the exact ae.safetensors filename. FLUX.1
    # takes the same file, verified working on this build, so it needs no
    # branch of its own here.
    return low == "ae.safetensors"


def _sibling_scan(model_path: Path, exts: Tuple[str, ...],
                  match) -> Optional[Path]:
    r"""Return the first file in the DIFFUSION MODEL'S OWN FOLDER that has one
    of `exts` and satisfies `match`, or None.

    Deliberately no walk-up and no sweep of .\models. Both used to be here,
    and the reason they are gone is worth keeping written down, because the
    case that motivated them is real: Z-Image-Turbo finetunes ship nested, none
    of those repos include a VAE, and a single shared ae.safetensors at the
    .\models root used to be found for all of them.

    The cost was worse than the convenience. A per-model folder that is missing
    one sub-model would silently borrow a same-named file from somewhere else
    in the tree -- and every one of these files is generically named. Half a
    dozen unrelated repos each ship "ae.safetensors", "clip_l.safetensors" or a
    t5 encoder, so "found a file with the right name" is nowhere near "found
    the right file". A FLUX.1 model in models\flux1-schnell-GGUF\ picked up a
    T5 from models\ root purely because the name matched, and nothing on
    screen said the file came from a different folder.

    Silently wrong is the worst outcome here: a mismatched VAE or encoder does
    not raise, it produces a black or mangled image. An empty box that asks the
    user to Browse is strictly better, so the rule is now sibling-or-nothing.
    A shared sub-model in a parent folder is still perfectly usable -- it just
    has to be chosen once by hand, and the choice then persists until the
    diffusion model changes again."""
    try:
        folder = model_path.expanduser().parent
        for f in sorted(folder.iterdir()):
            if f.is_file() and f.name.lower().endswith(exts) and match(f):
                return f
    except OSError:
        pass
    return None


def _detect_vae(diff_path_str: str) -> Tuple[str, str]:
    r"""
    Given a diffusion model path, locate the VAE that matches its family and
    return (vae_full_path, vae_stem), or ("", "") if not found.

    SEARCHES THE MODEL'S OWN FOLDER AND NOTHING ELSE. See _sibling_scan().
    """
    if not diff_path_str:
        return "", ""
    p = Path(diff_path_str).expanduser()
    if not p.exists():
        return "", ""

    # Family of the NEW diffuser decides which VAE filename we hunt for.
    family = configure.diffuser_family(diff_path_str)
    hit = _sibling_scan(
        p, (".safetensors",),
        lambda f: _vae_name_matches_family(f.name, family))
    return (str(hit), hit.stem) if hit else ("", "")


def _detect_clip(diff_path_str: str, which: str) -> Tuple[str, str]:
    """Find clip_l / clip_g next to the diffusion model, returning
    (path, name) or ("", ""). `which` is "l" or "g".

    Same sibling-only search as _detect_vae -- the model's own folder and
    nothing else. NOTE this does mean hum-ma's layout, which puts the CLIPs in
    a clip/ subfolder beside the UNet rather than next to it, no longer
    auto-fills; those two files have to be picked once by hand. That is the
    accepted cost of never borrowing a same-named file from another model's
    folder. See _sibling_scan() for why.

    Matching is by filename because these carry no metadata to key off. The
    patterns are deliberately narrow -- an underscore/dash/dot separated
    "clip" followed by the single letter -- so that clip_g never matches a
    request for clip_l, which a looser substring test would do.
    """
    if not diff_path_str:
        return "", ""
    letter = which.lower()
    pat = re.compile(rf"(^|[^a-z0-9])clip[-_. ]?{letter}([^a-z0-9]|$)", re.I)
    hit = _sibling_scan(Path(diff_path_str), (".safetensors",),
                        lambda f: pat.search(f.stem))
    return (str(hit), hit.name) if hit else ("", "")


def _detect_t5xxl(diff_path_str: str) -> Tuple[str, str]:
    r"""Find a T5-XXL encoder near the diffusion model, returning (path, name).

    Separate from _detect_clip() for two reasons rather than one shared
    function with a wider pattern. First the EXTENSION: clip_l/clip_g ship as
    .safetensors, but T5-XXL is usually a gguf quant here -- fp16 is 9.8GB of
    system RAM for a single conditioning pass, so city96's
    t5-v1_1-xxl-encoder-Q4_K_M.gguf (2.7GB) is the realistic file and .gguf
    must be searched. Second the NAMING: the two common spellings share no
    literal substring ("t5xxl_fp16" versus "t5-v1_1-xxl-encoder"), so the
    pattern is an alternation rather than one stem with separators.

    Sibling folder only, like the other two -- this is the detector that made
    the case for that rule, having filled the T5 box from the .\models root
    for a model sitting in models\flux1-schnell-GGUF\."""
    if not diff_path_str:
        return "", ""
    pat = re.compile(r"(t5[-_. ]?xxl|t5[-_. ]?v1[-_. ]?1[-_. ]?xxl)", re.I)
    hit = _sibling_scan(Path(diff_path_str), (".gguf", ".safetensors"),
                        lambda f: pat.search(f.stem))
    return (str(hit), hit.name) if hit else ("", "")


def _resolve_clips(diff_path_str: str, cur_l_path: str, cur_g_path: str) -> tuple:
    """Decide what the CLIP-L/CLIP-G boxes hold after the diffuser changes.

    Returns (l_path, l_name, g_path, g_name) as values or gr.update()s.

    Split SDXL uses both; FLUX.1 uses clip_l alone (its second encoder is
    T5-XXL, resolved by _resolve_t5xxl). Switching to a family that wants
    neither blanks them -- they would otherwise sit there stale and be passed
    to a diffuser that has no --clip_l flag. Within a family that does want
    them: found beside the new model -> use it, otherwise blank. Same two rules
    as _resolve_vae, and for the same reason -- nothing is inherited across a
    model change. cur_l_path / cur_g_path are kept in the signature so the
    wiring does not have to change.
    """
    fam = configure.diffuser_family(diff_path_str) if diff_path_str else None
    if fam == configure.DIFFUSER_FAMILY_FLUX1:
        # clip_l only, and an explicitly blank clip_g: a CLIP-G left over from
        # an SDXL session must not survive the switch, or sd-cli would be
        # handed --clip_g for a model whose spec never lists that slot.
        path, name = _detect_clip(diff_path_str, "l")
        return path, name, "", ""
    if fam != configure.DIFFUSER_FAMILY_SDXL:
        return "", "", "", ""

    out = []
    for which in ("l", "g"):
        path, name = _detect_clip(diff_path_str, which)
        out.extend([path, name] if path else ["", ""])
    return tuple(out)


def _resolve_t5xxl(diff_path_str: str, cur_path: str) -> Tuple[Any, Any]:
    """Decide what the T5-XXL boxes hold after the diffuser changes.

    FLUX.1 is currently the only family with a t5xxl slot, so every other
    family blanks it. Same two rules as the CLIP and VAE resolvers: found
    beside the new model, or blank. cur_path is kept in the signature so the
    wiring does not have to change."""
    fam = configure.diffuser_family(diff_path_str) if diff_path_str else None
    if fam != configure.DIFFUSER_FAMILY_FLUX1:
        return "", ""
    path, name = _detect_t5xxl(diff_path_str)
    return (path, name) if path else ("", "")


def _resolve_vae(diff_path_str: str, current_vae_path: str,
                 current_vae_name: str) -> Tuple[Any, Any]:
    """
    Decide what the VAE boxes should hold after the diffusion model changes.
    Returns (path_update, name_update) for (vae_path_tb, vae_name_tb).

    Two rules now, not three:
      1. A matching VAE sits in the new model's own folder -> use it.
      2. Anything else -> blank, and wait for the user.

    Rule 2 used to be "keep whatever is already in the box if it still exists
    on disk", which read as helpful and was not. Combined with the old
    tree-wide search it meant a box could hold a file the user never chose, for
    a model that never shipped one, from a folder they were never shown --
    and a wrong VAE does not raise, it produces a black or mangled image.
    Picking a diffusion model now clears its companions and refills only from
    beside it, so what is on screen is always either "found next to this model"
    or "you chose this by hand". `current_vae_path` is retained in the
    signature because the wiring passes it and the cross-family check below
    still needs `current_vae_name`.
    """
    # There used to be a "rule 0" here that short-circuited on a cross-family
    # switch: if the VAE already in the box belonged to a different family, it
    # returned blank immediately. That existed only to stop a stale VAE being
    # INHERITED, and nothing is inherited any more -- so all it did was return
    # early and skip the detection below. Switching from an SDXL model to a
    # FLUX.1 one therefore blanked the box and never noticed the correct
    # ae.safetensors sitting in the new model's own folder. Detection first,
    # unconditionally, is both simpler and right: whatever is beside the new
    # model wins, and if there is nothing beside it the box goes empty.
    vae_path, vae_name = _detect_vae(diff_path_str)
    return (vae_path, vae_name) if vae_path else ("", "")


def _diff_change_message(diff_path_str: str, current_vae_name: str) -> str:
    """Status-bar line for a diffusion-model change: announces the family the
    interface has switched to, and asks for a VAE when the box has been left
    empty because none was found beside the new model.

    Keyed on the OUTCOME (did detection find one?) rather than on comparing the
    old and new families, which is what it used to do. The old test asked
    whether the previously-held VAE belonged to a different family -- but it
    could say "set a VAE path" while _resolve_vae had just filled the box
    perfectly well from the new model's folder, and stay silent when the box
    was genuinely empty. Asking what actually happened cannot drift from it.

    Examples:
      "Handling/interface set to Z-Image-Turbo."
      "Handling/interface set to Flux.2-Klein — no VAE beside it, set one."
    """
    if not diff_path_str:
        return "Handling/interface set to no model selected."
    new_label = configure.diffuser_family_label(diff_path_str)
    vae_path, _ = _detect_vae(diff_path_str)
    if not vae_path:
        return (f"Handling/interface set to {new_label} — no VAE found beside "
                f"the model, set one.")
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


def _on_diff_path_change_slots(path: str, cur_l_path: str, cur_g_path: str,
                               cur_t5_path: str) -> tuple:
    """Re-resolve the text-encoder rows whenever the diffusion path changes.

    Registered as a SECOND listener on diff_path_tb.change, alongside the VAE
    handler above. It exists because the encoder rows were previously refreshed
    ONLY inside _browse_diffusion, so any other route to a new diffusion model
    -- typing the path by hand, or the page being rebuilt from a saved config
    -- left them holding whatever was in them before. Harmless while the only
    family with external encoders was SDXL, since a stale CLIP-L is at least
    still a CLIP-L. Not harmless with FLUX.1 in the picture: the T5-XXL row
    would happily sit there displaying a leftover VAE filename, which sd-cli
    would then be handed as --t5xxl and which produces a black image rather
    than an error.

    Browse still does its own resolving, and setting diff_path_tb
    programmatically fires this too, so both run for a Browse. That is
    deliberate and safe -- they compute the identical answer from the identical
    inputs, exactly as the VAE handler above already double-fires.

    It ALSO resets the Model Family override to Auto, which is the fix for a
    trap that cost a debugging session. The override is one global setting
    persisted in configuration.json, not a per-model note: set it to Flux.1 to
    force one stubborn file, pick a Klein model afterwards, and the Klein is
    still driven as Flux.1. Both symptoms of that are silent and neither
    mentions the override -- Denoise Strength appears on a model that has no
    --strength, and the Use All / Chain All switch vanishes on the one family
    that can actually fuse references, because both are read off img2img in
    the spec the override chose. An override is an answer to "what is THIS
    file", so a different file must re-ask the question.

    Returns 15 updates: three path/name pairs, the encoder relabel, three
    browse/clear button pairs, the packaging status line, and the family
    dropdown."""
    configure.set_family_override(configure.FAMILY_OVERRIDE_AUTO)
    slots = _needed_slots(path)
    if slots["clip_l"] or slots["clip_g"]:
        l_path, l_name, g_path, g_name = _resolve_clips(path, cur_l_path, cur_g_path)
    else:
        l_path, l_name, g_path, g_name = "", "", "", ""
    t5_path, t5_name = _resolve_t5xxl(path, cur_t5_path)
    (enc_u, l_vis, _lb, g_vis, _gb, t5_vis, _tb, pack) = _encoder_slot_updates(path)

    def _merge(value, vis):
        u = dict(vis)
        if not isinstance(value, dict):     # a real string, not gr.update()
            u["value"] = value
        return gr.update(**u)

    return (l_path, _merge(l_name, l_vis),
            g_path, _merge(g_name, g_vis),
            t5_path, _merge(t5_name, t5_vis),
            enc_u, l_vis, l_vis, g_vis, g_vis, t5_vis, t5_vis, pack,
            gr.update(value=configure.FAMILY_OVERRIDE_AUTO))


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
# Preview box status images (.\images\program_*.jpg)
# ---------------------------------------------------------------------------

def _status_image(name: str) -> Optional[str]:
    """Return str path to a images/program_<name>.jpg, or None if missing."""
    p = configure.get_images_dir() / f"program_{name}.jpg"
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

# The gr.Blocks object build_app() creates, kept so _allow_local_files() can
# extend its allowed_paths list at runtime. See that function for why.
_blocks_app: Optional[Any] = None


def _allow_local_files(paths: List[str]) -> None:
    r"""Permit Gradio to serve specific files from outside the project folder.

    THE PROBLEM. Gradio refuses to serve any file that is not under the current
    working directory, the system temp directory, its own cache, or an explicit
    allowed_paths entry — it raises InvalidPathError rather than silently
    leaking. That is the right default. But the Input thumbnail gallery shows
    the user's OWN reference images, which live wherever they keep their
    pictures: G:\Pictures\..., a NAS share, an external drive. Handing the
    gallery one of those paths therefore blew up the whole Add Image handler:

        InvalidPathError: Cannot move G:\Pictures\foo.jpg to the gradio cache
        dir because it was not created by the application ...

    This did not happen before the gallery existed, because the reference list
    was a Textbox showing FILENAMES — text, never served as a file. The moment
    an actual image component points at those paths, they have to be allowed.

    WHY NOT allowed_paths AT LAUNCH. launch() overwrites blocks.allowed_paths
    with whatever it is handed, and it is handed it exactly once at startup —
    long before the user has browsed anywhere. There is no set of folders we
    could name up front that would cover "wherever they pick images from next".

    WHY THIS WORKS. Both of Gradio's checks — the postprocess one in
    processing_utils._check_allowed and the request-time one in
    route_utils — read blocks.allowed_paths LIVE on each call rather than
    capturing it at launch. Appending to that list in place therefore takes
    effect immediately, for both the outbound update and the browser's
    subsequent fetch of the image.

    EXACT FILES, NOT FOLDERS. Each individual file is registered, never its
    parent directory. Gradio's is_allowed_file() treats an allowed_paths entry
    that IS the file as a match, so this is sufficient — and it means picking
    one image out of a folder does not quietly make every other file in that
    folder fetchable over the loopback port. The list grows by one entry per
    image the user deliberately chose, and by nothing else.

    Silent on failure: a Gradio internals change that renames or retypes
    allowed_paths must degrade to the old "that image will not display"
    behaviour, not take down the Add Image click that triggered it.
    """
    app = _blocks_app
    if app is None:
        return
    try:
        allowed = app.allowed_paths
        if not isinstance(allowed, list):
            return
        known = {str(p) for p in allowed}
        for p in paths or []:
            if not p:
                continue
            try:
                resolved = str(Path(str(p)).resolve())
            except OSError:
                continue
            if resolved not in known:
                # append() rather than reassignment: other parts of Gradio
                # hold a reference to this same list object.
                allowed.append(resolved)
                known.add(resolved)
    except Exception:
        pass


# SDXL's text encoders are FIXED by architecture: CLIP-L (OpenCLIP ViT-L/14,
# 768-dim) and CLIP-G (ViT-bigG/14, 1280-dim), concatenated to the 2048-dim
# vector the UNet's cross-attention was trained against. A Qwen3 or T5 gguf
# cannot stand in for them -- wrong architecture, wrong dimensions. So for SDXL
# the left-hand Encoder slot has NO conditioning role at all; it is only an
# optional prompt enhancer (a Qwen3 run through llama.cpp that rewrites the
# prompt text before sd-cli ever sees it). Z-Image and Flux.2 are the opposite:
# there the Encoder IS the conditioner and is mandatory.
_ENC_LABEL_CONDITIONER = "Encoder Name"
_ENC_LABEL_ENHANCER    = "Encoder Name (optional)"
_ENC_INFO_CONDITIONER  = "Qwen3 text encoder — required; conditions the diffuser."
_ENC_INFO_ENHANCER     = ("Optional — used only to expand the prompt text. "
                          "This diffuser conditions through its own encoders "
                          "instead, so nothing here reaches sd-cli.")


def _clips_needed(diff_path: str) -> bool:
    """Whether CLIP-L/CLIP-G must be supplied for this diffuser.

    Depends on the DIFFUSION FILE's packaging, and on nothing else. In
    particular it has no connection to the Encoder slot: that runs a Qwen3
    through llama.cpp to rewrite the prompt text and is never passed to
    sd-cli, so setting or clearing it changes nothing here. What matters is
    whether the model file carries its own text encoders -- an sd.cpp-native
    full checkpoint does, a city96 UNet-only quant does not."""
    if not diff_path:
        return False
    if configure.diffuser_family(diff_path) != configure.DIFFUSER_FAMILY_SDXL:
        return False
    return configure.sdxl_is_unet_only(diff_path)


def _needed_slots(diff_path: str) -> Dict[str, bool]:
    """Which external text-encoder rows this diffuser needs on screen.

    Built family-first rather than from diffuser_spec()["text_encoders"],
    because the SDXL spec is itself chosen by whether split CLIPs happen to be
    configured -- asking the spec which boxes to show, in order to decide
    whether to let the user fill those boxes, is circular. _clips_needed()
    already resolves SDXL packaging from the FILE, which is the non-circular
    signal, so SDXL keeps using it and FLUX.1 is stated outright.

    FLUX.1 shows clip_l but NOT clip_g: it is the first family to want one
    without the other, which is why this returns a per-slot mapping instead of
    the single needs_clips boolean it replaces."""
    if not diff_path:
        return {"clip_l": False, "clip_g": False, "t5xxl": False}
    if configure.diffuser_family(diff_path) == configure.DIFFUSER_FAMILY_FLUX1:
        return {"clip_l": True, "clip_g": False, "t5xxl": True}
    need = _clips_needed(diff_path)
    return {"clip_l": need, "clip_g": need, "t5xxl": False}


def _encoder_is_conditioner(diff_path: str) -> bool:
    """True when the Encoder slot is the diffuser's ACTUAL text conditioner
    (passed to sd-cli as --llm) rather than an optional prompt enhancer.

    The one box has always had two jobs. Z-Image and Flux.2 condition through a
    Qwen3 gguf, so the slot is mandatory and its contents are handed straight
    to sd-cli. SDXL and FLUX.1 condition through their own encoders -- CLIP-L +
    CLIP-G, and CLIP-L + T5-XXL respectively -- so for them the Qwen3 is run in
    a separate llama.cpp process purely to rewrite the prompt text, never
    reaches sd-cli, and is entirely optional.

    Asks the SPEC rather than testing for SDXL by name, which is what the two
    call sites used to do. That test was correct while SDXL was the only
    non-LLM family; with FLUX.1 added it labelled a slot "required; conditions
    the diffuser" for a family that ignores it completely. `text_encoders`
    already records exactly this, so it is the thing to read."""
    if not diff_path:
        return True          # nothing chosen yet -- keep the stricter wording
    spec = configure.diffuser_spec(diff_path, "", _has_split_clips())
    return bool(spec and "llm" in spec["text_encoders"])


def _encoder_slot_updates(diff_path: str) -> tuple:
    """UI updates for the encoder/CLIP/T5 slots when the diffuser changes.

    Returns (encoder box, clip_l box, clip_l button, clip_g box, clip_g button,
    t5xxl box, t5xxl button, packaging status). Each row appears only when the
    chosen model actually needs it, so a self-contained checkpoint does not
    present boxes that would be ignored, and FLUX.1 does not present a CLIP-G
    it has no flag for. The Encoder box is relabelled rather than hidden for
    SDXL, since it stays useful there as a prompt enhancer."""
    _is_cond = _encoder_is_conditioner(diff_path)
    slots = _needed_slots(diff_path)
    enc = gr.update(
        label=_ENC_LABEL_CONDITIONER if _is_cond else _ENC_LABEL_ENHANCER,
        info=_ENC_INFO_CONDITIONER if _is_cond else _ENC_INFO_ENHANCER,
    )
    l_vis  = gr.update(visible=slots["clip_l"])
    g_vis  = gr.update(visible=slots["clip_g"])
    t5_vis = gr.update(visible=slots["t5xxl"])
    label = configure.sdxl_packaging_label(diff_path)
    status = gr.update(value=label, visible=bool(label))
    return (enc, l_vis, l_vis, g_vis, g_vis, t5_vis, t5_vis, status)


def _has_split_clips(c: Optional[Dict[str, Any]] = None) -> bool:
    """True when BOTH split-SDXL text encoders are configured and on disk.
    This is what selects SDXL packaging -- see configure.diffuser_spec."""
    c = c if c is not None else configure.load_configuration()
    return (_model_path_ok(c.get("clip_l_model_path"))
            and _model_path_ok(c.get("clip_g_model_path")))


def _family_takes_input_image(diff_path: str) -> bool:
    """True when the diffuser can accept an input image at all, by either
    route (Flux.2's -r references or SDXL's -i init image). Drives whether the
    whole input-image column is on screen."""
    spec = (configure.diffuser_spec(diff_path, "", _has_split_clips())
            if diff_path else None)
    return bool(spec and spec.get("img2img"))


def _family_fuses_refs(diff_path: str) -> bool:
    """True when the diffuser can consume SEVERAL input images in one run.

    Only the "ref" path can: Flux.2 repeats -r once per reference and
    conditions on all of them together. The "init" path (SDXL, FLUX.1)
    denoises from exactly one -i, so several images can only ever be processed
    one after another.

    This is the question the Use All / Chain All radio actually asks, which is
    why it is its own predicate rather than a reading of _family_uses_init_image
    -- a family could in principle be added that fuses references without being
    on either of today's two paths."""
    spec = (configure.diffuser_spec(diff_path, "", _has_split_clips())
            if diff_path else None)
    return bool(spec and spec.get("img2img") == "ref")


def _family_uses_init_image(diff_path: str) -> bool:
    """True when the diffuser uses the standard -i + --strength img2img path
    (SDXL) rather than -r reference conditioning (Flux.2). Drives the Denoise
    Strength slider, which has no meaning on the -r path."""
    spec = (configure.diffuser_spec(diff_path, "", _has_split_clips())
            if diff_path else None)
    return bool(spec and spec.get("img2img") == "init")


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
    diff = c.get("imagegen_model_path", "")
    diff_ok = _model_path_ok(diff)

    # Which files this run needs depends on the diffuser. Until one is chosen
    # (or when it is unrecognised) fall back to the Z-Image spec, which is the
    # three-file set this program required before families existed.
    spec = ((configure.diffuser_spec(diff, "", _has_split_clips(c)) if diff_ok else None)
            or configure.DIFFUSER_FAMILY_SPECS[configure.DIFFUSER_FAMILY_ZIMAGE])

    missing: List[str] = []
    if not diff_ok:
        missing.append("Diffusion")
    if "llm" in spec["text_encoders"] and not _model_path_ok(c.get("encoder_model_path")):
        missing.append("Encoder")
    # Split-SDXL only: a full SDXL .safetensors bundles these and declares no
    # encoder slots, so it never asks for them.
    if "clip_l" in spec["text_encoders"] and not _model_path_ok(c.get("clip_l_model_path")):
        missing.append("CLIP-L")
    if "clip_g" in spec["text_encoders"] and not _model_path_ok(c.get("clip_g_model_path")):
        missing.append("CLIP-G")
    # FLUX.1 only. Without it sd-cli loads, samples, and writes a black image
    # rather than erroring, so a missing T5 has to be caught here.
    if "t5xxl" in spec["text_encoders"] and not _model_path_ok(c.get("t5xxl_model_path")):
        missing.append("T5-XXL")
    # Not universal: a full SDXL checkpoint carries its own VAE, so demanding
    # one there would hide the Generate button on a perfectly valid setup.
    if spec["vae_required"] and not _model_path_ok(c.get("vae_model_path")):
        missing.append("VAE")
    return missing


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
    width_dd, height_dd, strength_sld, ref_mode_radio, batch_dd,
    output_fmt_dd, seed_num, preset_dd) for the current diffuser.

    Keeps the Settings header, the input-image row and every generation control
    in step with whatever model is set, and flashes any encoder/VAE
    compatibility problem on arrival.

    Two distinct jobs, and the distinction is the whole point:

      WITHIN a family, only the CHOICE LISTS are retuned (steps dropdown
      contents, cfg slider range, width/height options). A value the user has
      deliberately set is kept as long as it is still valid, because nothing
      about the model changed.

      ON a family CHANGE, the entire settings panel is snapped to that model's
      defaults -- steps, cfg, sampler, batch count, strength, seed, output
      format, quality preset and resolution -- from
      configure.family_generation_defaults(). This is deliberate and not a
      convenience: the same number means different things to different
      families. cfg 1.5 is correct for a distilled model and badly wrong for
      SDXL base (which wants ~7), yet it sits inside SDXL's 1.0-12.0 range and
      would survive a pure range test as a silently bad setting. 8 steps is a
      full run for Z-Image and a quarter of one for FLUX.1 dev. euler_a is
      right for Flux.2 and wrong for SDXL base.

    The two PROMPTS are never touched by either path. They are the user's own
    text, they are not model-specific, and losing a carefully written prompt
    because the model was swapped would be indefensible.

    Resolution always lands on 512x512 on a family change (see
    DEFAULT_GENERATION_WIDTH) regardless of what the family is natively good
    at -- the compute buffer scales with pixel count, so on an 8GB card 1024 is
    the difference between running and OOMing. The user raises it deliberately
    once they know the model loads.
    """
    c = configure.load_configuration()
    # imagegen_last_family lives in generation.json alongside the steps/cfg/
    # size values it describes; the model path it is compared against lives in
    # configuration.json. Hence two loads rather than one.
    g = configure.load_generation()
    diff = c.get("imagegen_model_path", "")
    fam_label = configure.diffuser_family_label(diff)
    # The input-image column is shown for any family that can take one (Flux.2
    # -r, SDXL -i); the strength slider only for the -i path.
    takes_image = _family_takes_input_image(diff)
    uses_init = _family_uses_init_image(diff)
    # Only "ref" families (Flux.2) can consume more than one input image.
    _spec_now = configure.diffuser_spec(diff) if diff else None
    takes_multi = bool(_spec_now and _spec_now.get("img2img") == "ref")
    ok, msg = inference.check_model_compatibility(c)
    status = gr.update() if ok else gr.update(value="⚠ " + msg)

    spec = configure.family_step_cfg(diff)
    # The full "optimal settings for this model" set -- the same one the
    # Restore To Defaults button applies, so a model change and an explicit
    # restore can never land on different values.
    defaults = configure.family_generation_defaults(diff)
    step_choices, step_default = spec["steps"]
    cfg_min, cfg_max, cfg_step, cfg_default = spec["cfg"]

    # Has the family changed since these values were last saved? If so, the
    # numbers came from a different model class and are not evidence of intent.
    fam_key = configure.family_step_cfg_key(diff)
    family_changed = fam_key != g.get("imagegen_last_family", "")
    if family_changed:
        try:
            configure.update_generation({"imagegen_last_family": fam_key})
        except Exception:
            pass      # a failed write only costs one extra snap next time

    try:
        cs = int(cur_steps)
    except (TypeError, ValueError):
        cs = None
    step_val = step_default if family_changed else (cs if cs in step_choices else step_default)
    steps_upd = gr.update(choices=step_choices, value=step_val)

    try:
        cc = float(cur_cfg)
    except (TypeError, ValueError):
        cc = None
    cfg_val = (cfg_default if family_changed
               else (cc if (cc is not None and cfg_min <= cc <= cfg_max) else cfg_default))
    cfg_upd = gr.update(minimum=cfg_min, maximum=cfg_max, step=cfg_step, value=cfg_val)

    # Width / height: swap to the family's allowed sizes, keeping a valid
    # current value, else snapping to 768.
    sizes = configure.family_image_sizes(diff)
    def _size_upd(cur: Any, fallback: int) -> Any:
        try:
            cv = int(cur)
        except (TypeError, ValueError):
            cv = None
        if family_changed:
            return gr.update(choices=sizes, value=fallback)
        return gr.update(choices=sizes, value=(cv if cv in sizes else 768))
    width_upd  = _size_upd(cur_width,  defaults["imagegen_width"])
    height_upd = _size_upd(cur_height, defaults["imagegen_height"])

    # On a family change every remaining setting is snapped to the new model's
    # defaults too; within a family they are left exactly as the user has them.
    # Sampler used to be excluded from this on the grounds that euler_a suited
    # every family -- which stopped being true once SDXL base (dpm++2m) and
    # FLUX.1 (euler) were supported, and left whichever sampler the previous
    # model wanted silently applied to the new one.
    _snap = (lambda key: gr.update(value=defaults[key]) if family_changed
             else gr.update())
    sampler_upd = _snap("imagegen_sampling")
    batch_upd   = _snap("imagegen_batch_count")
    fmt_upd     = _snap("output_format")
    seed_upd    = _snap("imagegen_seed")
    preset_upd  = _snap("imagegen_quality_preset")

    # Strength carries a value as well as its visibility: it is meaningful only
    # for -i families with an image loaded, but when the family changes the
    # value underneath still has to become the new family's.
    _strength_visible = bool(uses_init and configure.get_ref_images())
    strength_upd = (gr.update(visible=_strength_visible,
                              value=defaults["imagegen_strength"])
                    if family_changed else gr.update(visible=_strength_visible))

    return (
        gr.update(value=f"### Settings ({fam_label})"),
        gr.update(visible=takes_image),
        status,
        sampler_upd,
        steps_upd,
        cfg_upd,
        width_upd,
        height_upd,
        strength_upd,
        # The mode radio depends on the FAMILY as well as the image count, so
        # it is re-synced here: switching from Flux.2 to SDXL on the
        # Configuration page must retract it even though no image changed.
        gr.update(visible=bool(takes_multi and len(configure.get_ref_images()) > 1)),
        batch_upd,
        fmt_upd,
        seed_upd,
        preset_upd,
    )


def _build_generate_tab_inner() -> None:
    """Build Generate tab widgets; store refs in _gen for later wiring."""
    cfg = _cfg()
    # Every widget on this page is seeded from generation.json (gen), NOT from
    # configuration.json (cfg). cfg is still needed here, but only for the two
    # things this page reads rather than owns: the diffusion model path (to
    # label the Settings header and decide whether the input-image column
    # appears) and the "is everything configured yet" gate.
    gen = _gcfg()
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

    # Build the family-dependent dropdowns with the CURRENT family's choice
    # lists, not the generic supersets. _generate_family_updates() swaps them
    # to exactly these lists on tab-select anyway, but that fires only when the
    # user CLICKS the tab — and Generation is the first tab, so on launch it
    # has usually not fired at all.
    #
    # This was survivable while the saved values were always in the generic
    # lists, and stopped being so once generation.json started faithfully
    # restoring family-tuned ones: a saved 20 steps (flux1_dev, flux2_base) or
    # 30 (sdxl_base) is not in STEP_CHOICES, and a Dropdown handed a value
    # outside its choices warns on stdout and does not reliably show it. The
    # user would open the app to a Diffuse Steps box that had silently lost the
    # setting their last run used.
    #
    # Same three lists, same helper, so this can never disagree with what the
    # tab-select handler applies a moment later.
    _initial_diff = cfg.get("imagegen_model_path", "")
    _initial_steps = configure.family_step_cfg(_initial_diff)["steps"][0]
    _initial_cfg_rng = configure.family_step_cfg(_initial_diff)["cfg"]
    _initial_sizes = configure.family_image_sizes(_initial_diff)

    def _seed(value: Any, choices: List[Any], fallback: Any) -> Any:
        """A saved value if the current family still allows it, else the
        family's own fallback. Never returns something outside `choices`."""
        return value if value in choices else fallback

    with gr.Row():
        # ── Column 1: settings only ─────────────────────────────────────────
        # Prompts and the Generate button moved to column 2 so the three
        # sections read left-to-right as: tune → describe/input/go → view.
        with gr.Column(scale=1):
            _gen["settings_header"] = gr.Markdown(f"### Settings ({_initial_family_label})")
            with gr.Row():
                _gen["preset_dd"] = gr.Dropdown(
                    label="Quality Preset", choices=list(presets.keys()),
                    value=gen.get("imagegen_quality_preset", "Fast (Turbo)"),
                )
                _gen["sampler_dd"] = gr.Dropdown(
                    label="Sampler Type", choices=list(configure.SAMPLER_MAP.keys()),
                    value=gen.get("imagegen_sampling", "euler_a"),
                )
            with gr.Row():
                _gen["width_dd"]  = gr.Dropdown(
                    label="Image Width", choices=_initial_sizes,
                    value=_seed(gen.get("imagegen_width", 512), _initial_sizes, 512),
                )
                _gen["height_dd"] = gr.Dropdown(
                    label="Image Height", choices=_initial_sizes,
                    value=_seed(gen.get("imagegen_height", 512), _initial_sizes, 512),
                )
            with gr.Row():
                _gen["steps_dd"] = gr.Dropdown(
                    label="Diffuse Steps", choices=_initial_steps,
                    value=_seed(gen.get("imagegen_steps", 4), _initial_steps,
                                configure.family_step_cfg(_initial_diff)["steps"][1]),
                )
                _gen["cfg_scale_sld"] = gr.Slider(
                    label="CFG Scale",
                    minimum=_initial_cfg_rng[0], maximum=_initial_cfg_rng[1],
                    step=_initial_cfg_rng[2],
                    value=gen.get("imagegen_cfg_scale", _initial_cfg_rng[3]),
                )
            with gr.Row():
                _gen["batch_dd"] = gr.Dropdown(label="Batch Count",
                                               choices=configure.BATCH_COUNT_CHOICES,
                                               value=gen.get("imagegen_batch_count", 1)
                )
                _gen["output_fmt_dd"] = gr.Dropdown(
                    label="Output Format", choices=configure.OUTPUT_FORMATS,
                    value=gen.get("output_format", "png"),
                )
            with gr.Row():
                _gen["seed_num"] = gr.Number(label="Gen Seed (-1 = random)",
                                             value=gen.get("imagegen_seed", -1), precision=0)
            # ── Restore To Defaults ──────────────────────────────────────────
            # Repaints this whole settings panel to the loaded model's optimal
            # values (configure.family_generation_defaults). Deliberately NOT
            # paired with a Save button, unlike the Configuration and
            # Preferences pages: this page has no Save, because it persists
            # itself to generation.json on every submission. So the restored
            # values become permanent the next time Generate is clicked, and
            # clicking away without generating leaves the previous saved set
            # untouched — which is the useful behaviour for a button whose
            # whole purpose is "let me try again from a known-good baseline".
            #
            # variant="stop" (red) matches the revert buttons on the other two
            # pages, so the destructive-ish action reads the same everywhere.
            _gen["restore_defaults_btn"] = gr.Button(
                "Restore To Defaults", variant="stop", size="sm",
            )
            # NOTE: no manual "Save as Default" button — every submission
            # auto-saves this panel to generation.json (see do_generate), so
            # the next launch picks up the last settings that were used.

        # ── Column 2: image input, prompts, Generate button ──────────────────
        # Order is deliberate, top to bottom: Image Edit, then the prompts,
        # then Generate. The image input sits FIRST so that attaching images
        # grows the list downward from the top of the column — pushing the
        # negative prompt (the least interesting box, usually a set-and-forget
        # boilerplate line) toward the bottom of the screen instead of pushing
        # the image list off it. Generate stays pinned last, so the column
        # still reads as input → describe → go.
        with gr.Column(scale=1):
            # ── Flux.2 controls: flash-attn toggle + reference images ───────
            # Whole column shown ONLY when the diffuser is Flux.2. When hidden
            # it occupies no height, so on a Z-Image model column 2 opens on
            # "### Prompts" exactly as it did before this block moved up.
            # Shown for every family that can take an input image at all:
            # Flux.2 (multi-reference editing, -r) and SDXL (standard img2img,
            # -i + --strength). Hidden for Z-Image, which is text-to-image only.
            _img_in_now = _family_takes_input_image(cfg.get("imagegen_model_path", ""))
            _init_now = _family_uses_init_image(cfg.get("imagegen_model_path", ""))
            with gr.Column(visible=_img_in_now) as _gen["ref_row"]:
                gr.Markdown("#### Input Image (img+txt to img)")
                # Reference images for image-to-image / editing (sd.cpp -r,
                # repeatable). "Add Image" opens a native file picker and
                # APPENDS to the list (add, then add again); "Clear Images"
                # empties it. No drop-zone. The chosen files are listed one
                # per line below.
                #
                # The list PERSISTS across generations: finishing a run leaves
                # it exactly as it was, so the normal "same images, tweak the
                # prompt, generate again" loop needs no re-picking. Clear
                # Images is the only thing that empties it. (It used to
                # self-clear on every completed run, which meant re-browsing
                # for the same files on every iteration.)
                #
                # A native picker rather than gr.UploadButton, because only a
                # native dialog can be told where to open — see _browse_images.
                with gr.Row():
                    _gen["ref_add_btn"] = gr.Button("Add Image", size="sm")
                    _gen["ref_clear_btn"] = gr.Button("Clear Images", size="sm")
                # Built visible=True (mounted) on purpose. A Textbox created
                # visible=False is not placed in the DOM by Gradio 6, so the
                # first Add — which sends value + visible=True together — paints
                # nothing until a second update mounts it (the "first image
                # doesn't appear, second makes both appear" bug). Keeping it
                # mounted and never toggling visibility fixes that; the empty
                # state is conveyed by the placeholder instead of by hiding.
                # (See gradio issues #11768 / #12511.)
                _gen["ref_list_tb"] = gr.Textbox(
                    show_label=False, interactive=False, visible=True,
                    lines=1, max_lines=8, elem_id="ref-image-list",
                    placeholder="No reference images added yet \u2014 use \u201cAdd Image\u201d above.",
                )
                # Accumulated list of reference-image paths (the real input to
                # generation); the textbox above is just its visible form.
                # Seeded from, and mirrored into, configure.APP_STATE so the
                # accumulated list is readable outside this event graph — see
                # configure.get_ref_images / set_ref_images. Session-only:
                # nothing about the current pile of images belongs in
                # configuration.json.
                _gen["ref_images_state"] = gr.State(configure.get_ref_images())
                # Only meaningful once 2+ images are accumulated (nothing to
                # choose between with 0 or 1), so built hidden and toggled by
                # _add_ref_images / _clear_ref_images alongside the list itself.
                # "Use All" matches the only prior behaviour (every
                # accumulated image handed to sd.cpp as one multi-reference
                # edit); "Chain All" instead runs each image through its own
                # generation in sequence (see do_generate's chain_mode branch)
                # and is the STARTUP default, since Use All holds every
                # reference in VRAM at once and the OOM risk grows with each
                # added image.
                #
                # Startup default only. Once the user picks Use All it stays
                # picked for the rest of the session — through generations,
                # through Clear Images, through adding a fresh batch of
                # images later. Only relaunching the program (or the user
                # selecting Chain All again) brings the default back. That is
                # why the value comes from configure.get_ref_mode(), which is
                # REF_MODE_DEFAULT until the radio's change handler says
                # otherwise, and why NOTHING below ever writes a value back
                # into this radio.
                _gen["ref_mode_radio"] = gr.Radio(
                    label="Multiple Reference Images",
                    choices=configure.REF_MODE_CHOICES,
                    value=configure.get_ref_mode(),
                    visible=False, elem_id="ref-image-mode",
                )
                # img2img denoise strength. Only meaningful for families that
                # use -i (SDXL); Flux.2 conditions through -r instead and has
                # no strength control, so this is hidden there. 0.0 returns the
                # input untouched, 1.0 ignores it; sd.cpp's default is 0.75.
                # 0.3-0.5 keeps composition and restyles, 0.7-0.9 reinvents.
                _gen["strength_sld"] = gr.Slider(
                    label="Denoise Strength",
                    minimum=0.0, maximum=1.0, step=0.05,
                    value=float(gen.get("imagegen_strength", 0.65)),
                    # Needs BOTH an -i-path family and at least one image
                    # already in the list; an empty list leaves nothing for a
                    # denoise fraction to apply to.
                    visible=bool(_init_now and configure.get_ref_images()),
                    info="Lower keeps more of the input image; higher redraws it.",
                )

            # No "### Prompts" heading here: the two textboxes below are
            # already labelled Positive Prompt and Negative Prompt, so a
            # heading only repeats them and costs vertical space in the
            # centre column, which is the most crowded of the three.
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
                value=gen.get("last_prompt", ""),
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
                value=gen.get("negative_prompt", ""),
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

            gr.Markdown("#### Submitting Input")
            with gr.Row(visible=configured) as _gen["generate_row"]:
                # Single dynamic button: "Generate" (primary) when idle,
                # "..Please Wait.." (disabled) while running. do_generate()
                # flips it on entry and back on its final yield.
                _gen["generate_btn"] = gr.Button("Generate Image", variant="primary", size="lg")

        # ── Column 3: input thumbnails, then image preview ───────────────────
        with gr.Column(scale=1):
            # ── Input: thumbnails of the centre column's reference images ────
            # One row per the requirement, mirroring the centre column's list:
            # every image added there gets a thumbnail here, so the pile can be
            # identified visually rather than by filename alone.
            #
            # The whole block is hidden while the list is empty, so on a
            # text-to-image model (or before the first Add) this column opens
            # straight onto "### Output" exactly as it did before — no reserved
            # blank space, no shifted preview.
            #
            # "Input" is deliberately a #### heading against Output's ###: this
            # is the subordinate, supporting row of the two, and the subtler
            # weight keeps the eye on the preview underneath.
            #
            # SIZE is a Preferences setting (Input Thumbnail Size), because it
            # is a straight trade against the preview below — see
            # configure.INPUT_THUMBNAIL_CHOICES. Read once here, at build time,
            # for the same reason build_app() reads it once for the CSS: the
            # two must agree, and Gradio has no runtime stylesheet hook.
            _input_thumb = configure.get_input_thumbnail_size()
            _refs_now = configure.get_ref_images()
            # The COLUMN toggles; the gallery inside it is built visible=True
            # and stays that way. That is the same shape as the ref_row block
            # in column 2, and it matters for the reason spelled out on
            # ref_list_tb: Gradio 6 does not always paint a component whose
            # value and visibility arrive in the same update, so the first Add
            # can drop its content (gradio issues #11768 / #12511). Two things
            # guard against it here — the gallery itself never toggles, and the
            # ref_images_state .change handler pushes the same updates a second
            # time from the list state, independently of the button click. If
            # the first pass ever misses, the second lands.
            with gr.Column(visible=bool(_refs_now)) as _gen["input_row"]:
                gr.Markdown("#### Input")
                # Same one-row horizontal scroller as the Thumbnails Gallery at
                # the bottom of the page (see the #input-gallery CSS in
                # build_app), for the same reason: a wrapping grid would grow a
                # second and third row as images are added and push the Output
                # section down the page, which is precisely what this row must
                # not do. allow_preview=False keeps a click from opening
                # Gradio's full-screen lightbox — clicking a thumbnail here
                # sends the image to the preview box below instead.
                _gen["input_gallery"] = gr.Gallery(
                    value=_refs_now,
                    columns=16, rows=1,
                    height=_input_thumb,
                    object_fit="contain",
                    allow_preview=False,
                    show_label=False,
                    fit_columns=False,
                    elem_id="input-gallery",
                )
            gr.Markdown("### Output")
            # Single currently-selected/in-progress image — most recent
            # generation, the live phase status image, or a clicked gallery
            # thumbnail. Never shows Gradio's built-in progress bar.
            # No corner buttons on the preview at all. The little download
            # icon was removed on purpose: it did not reliably work, and the
            # intended way to keep/open an image is the Thumbnails Gallery
            # below (click a thumbnail), so the icon was only added complexity.
            # Gradio 6 replaced show_download_button/show_share_button with a
            # single buttons=[...] list — an empty list hides every corner
            # button; the pre-6 branch turns the download button off directly.
            _img_button_kwargs = (
                {"buttons": []} if _GRADIO_MAJOR >= 6
                else {"show_download_button": False, "show_share_button": False}
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
    cfg_scale, seed, batch, output_format, quality_preset, ref_images=None,
    ref_mode=None, strength=None):
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

        ref_mode (configure.REF_MODE_USE_ALL / REF_MODE_CHAIN_ALL) only
        matters when 2+ reference images are accumulated:
          - REF_MODE_CHAIN_ALL (default): each image is run through its OWN
            generation, one after another (see chain_mode / ref_batches
            below); with N images and a batch count of B this produces
            N*B images total, and the status line gets a "Chain Position
            i/N;" prefix identifying which run in the sequence is active. Only
            one reference is ever resident at a time, which is the safer
            choice memory-wise.
          - REF_MODE_USE_ALL: every accumulated image is instead handed to
            ONE generation as multiple -r references — a deliberate opt-in,
            since sd.cpp holds all of them in VRAM at once for that single
            run and the OOM risk grows with each added image.
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

        # inference.py takes ONE dict and reads keys from all THREE settings
        # files out of it — model paths and devices from configuration.json,
        # prompt_template from preferences.json, and everything on this page
        # from generation.json. configure.generation_config() does the merge in
        # the one place that knows about all three, so neither inference.py nor
        # any save handler has to know the split exists.
        # ref_images is added per-run below (see ref_batches) rather than
        # here, since Chain All needs a different list on each run.
        base_gen_cfg = configure.generation_config()
        base_gen_cfg.update(
            imagegen_width=int(width), imagegen_height=int(height),
            imagegen_steps=int(steps), imagegen_sampling=sampler,
            imagegen_cfg_scale=float(cfg_scale),
            imagegen_seed=int(seed), imagegen_batch_count=int(batch),
            negative_prompt=negative,
            output_format=output_format,
        )
        # img2img denoise strength, from the Generate page slider. Falls back
        # to the saved value (then sd.cpp's own 0.75) when the slider is
        # hidden, which is every family that does not use the -i path.
        if strength is not None:
            base_gen_cfg["imagegen_strength"] = float(strength)

        # ── Save the whole Generation page to generation.json ───────────────
        # ON SUBMISSION, before a single pixel is generated — not on success,
        # and not conditionally on the Quality Preset being "Custom", which is
        # what this used to do. Both of those old conditions lost work the user
        # had actually done:
        #   * success-only meant a run that OOM'd, or hit a bad model pairing,
        #     threw away the settings that were on screen when it started —
        #     which are exactly the settings the user wants back to adjust and
        #     retry.
        #   * Custom-only meant that with a named preset selected, the prompt,
        #     negative prompt, seed, batch count and output format were never
        #     saved at all. Those are not part of any preset, so nothing was
        #     reconstructing them; they were simply lost at exit.
        # Saving here, unconditionally, means the page always reopens exactly
        # as it was last left. The preset NAME is saved alongside the values so
        # the dropdown reads correctly on the next launch either way.
        #
        # Reference images are the one thing deliberately not saved — see
        # configure.GENERATION_KEYS for why (the paths may not exist next
        # launch, so restoring them repopulates the input column with dead
        # entries that only fail at generation time).
        #
        # A failed write must never block a generation: the settings are worth
        # remembering, but not at the cost of the run the user just asked for.
        try:
            configure.update_generation({
                "imagegen_quality_preset": quality_preset,
                "imagegen_width": int(width), "imagegen_height": int(height),
                "imagegen_steps": int(steps), "imagegen_sampling": sampler,
                "imagegen_cfg_scale": float(cfg_scale),
                "imagegen_seed": int(seed), "imagegen_batch_count": int(batch),
                "imagegen_strength": float(base_gen_cfg.get("imagegen_strength", 0.65)),
                "imagegen_ref_mode": (ref_mode if ref_mode in configure.REF_MODE_CHOICES
                                      else configure.get_ref_mode()),
                "output_format": output_format,
                "last_prompt": prompt,
                "negative_prompt": negative,
            })
        except Exception as e:
            print(f"[generate] WARNING: could not save generation.json: {e}",
                  flush=True)

        # Reference images (Flux.2 -r). The accumulated list arrives from
        # ref_images_state as a list of real on-disk paths (the native picker
        # hands over the user's own files, not Gradio temp copies); a lone
        # string or None are still tolerated, so normalise to a list.
        # READ ONLY — the list is not emptied here or anywhere downstream; it
        # survives the run and is cleared only by the "Clear Images" button.
        # inference.generate_image() ignores this entirely for Z-Image and
        # only uses it when the diffuser is Flux.2.
        if ref_images is None:
            _refs = []
        elif isinstance(ref_images, (list, tuple)):
            _refs = [str(r) for r in ref_images if r]
        else:
            _refs = [str(ref_images)]

        # Chain All only kicks in with 2+ images -- with 0 or 1 there is
        # nothing to split, so it collapses to the same single-run behaviour
        # as Use All. Each chain "link" is a single-image list, run through
        # its own call to inference.generate_image(); Use All is one run
        # carrying the whole list.
        #
        # WHO GETS A CHOICE, AND WHO DOES NOT
        # Only the "ref" families can consume several images in ONE run:
        # Flux.2 repeats -r per reference. The "init" families -- SDXL and
        # FLUX.1 -- denoise from a single -i and sd-cli has no way to accept a
        # second, so "Use All" would promise something it cannot do and the
        # radio stays hidden for them (_render_ref_mode_visibility).
        #
        # Hidden choice does NOT mean no chaining. This used to drop the extra
        # images with a console note and run only the first, which is why a
        # nine-image FLUX.1 job stopped after image one and never showed a
        # chain position. Every family can process a pile of images one at a
        # time; only the ability to fuse them into a single run is special. So
        # the init families are now FORCED to chain rather than truncated, and
        # the radio's absence just means the one available behaviour is not
        # presented as a decision.
        #
        # Note the interaction with Batch Count, which is deliberate and not a
        # bug: each link is a full generation and still honours it, so nine
        # images at a batch of 2 produce eighteen outputs. Batch Count is
        # "how many variations per input", not "how many images in total".
        _supports_use_all = _family_fuses_refs(c.get("imagegen_model_path", ""))
        chain_mode = len(_refs) > 1 and (
            not _supports_use_all or ref_mode == configure.REF_MODE_CHAIN_ALL)
        ref_batches: List[List[str]] = ([[r] for r in _refs] if chain_mode
                                        else [_refs])
        chain_total = len(ref_batches)

        configure.APP_STATE["cancel_requested"] = False

        last_preview = preview_now
        last_gallery = gallery_now
        last_result: Dict[str, Any] = {"success": False, "message": "Unknown error"}
        chain_successes = 0

        for chain_idx, ref_batch in enumerate(ref_batches, start=1):
            gen_cfg = dict(base_gen_cfg)
            gen_cfg["ref_images"] = ref_batch

            # ── Run generation on a worker thread; main thread yields preview
            #    + status updates based on phase, polled from a shared mutable
            #    holder ("_phase"). The status string shows which of the two
            #    phases is active (1/2 Encoding, 2/2 Diffusing) plus a live
            #    timer for that phase. Once configure.TIMING_STATS has data
            #    from a prior generation this session, the timer becomes an
            #    ETA countdown-style display ("~Ns left"); until then it just
            #    counts up, since there's nothing to estimate against yet. ──
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
                """Build the '[Chain I/N; ]Generate Stage N/2; [Batch Number
                X/Y; ]... Phase {step}/{total}...###s (prev_batch_Ns)' status
                string. Seconds are always whole numbers (no split seconds)
                per the fixed status-bar format — never decimals.

                Ordering: Chain leads (it is the outermost loop, one full
                Generate Stage 1-2 run per link), then Generate Stage, because
                Stage 1 (encoding) runs ONCE up front per link and Stage 2
                (diffusion) is what actually iterates per image — so Batch
                Number is a Stage-2 concept and is shown only then, after the
                stage, never during encoding."""
                name = _phase["name"]
                elapsed_s = int(time.time() - _phase["phase_start"])
                batch_cur = _phase["batch_current"]
                batch_tot = _phase["batch_total"]

                chain_prefix = (f"Chain Position {chain_idx}/{chain_total}; "
                                if chain_total > 1 else "")

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
                    return (f"{chain_prefix}Generate Stage 1/2; Encoding Phase"
                            f"{step_part}...{elapsed_s}s{prev_suffix}")

                # diffusion (Stage 2) — Batch Number belongs here, after the stage.
                batch_prefix = f"Batch Number {batch_cur}/{batch_tot}; "
                step = _phase.get("step", 0)
                total = _phase.get("total_steps", 0) or int(gen_cfg.get("imagegen_steps", 0))
                step_part = f" {step}/{total}" if total else ""
                return (f"{chain_prefix}Generate Stage 2/2; {batch_prefix}"
                        f"Diffusing Phase{step_part}...{elapsed_s}s{prev_suffix}")

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
            # Button switches to "..Please Wait.." the moment the worker thread
            # starts. Only send that update on the FIRST poll tick of the FIRST
            # chain link — re-sending the same gr.update() on every 0.15s tick
            # forces Gradio to re-render the button node repeatedly, which was
            # cascading into a layout recalculation of sibling nodes in the
            # same column (incl. the preview image) and intermittently
            # knocking out its object-fit CSS override. Once the button is
            # already showing "..Please Wait..", later ticks (and later chain
            # links) pass a true no-op (gr.update()) for it.
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
                btn_update = _btn_wait if (first_tick and chain_idx == 1) else gr.update()
                first_tick = False

                status_update = gr.update()
                if current_second != last_shown_second:
                    last_shown_second = current_second
                    status_update = _format_status()

                if img and img != last_shown_img:
                    last_shown_img = img
                    last_preview = img
                    yield img, gr.update(), status_update, btn_update
                else:
                    yield gr.update(), gr.update(), status_update, btn_update
                time.sleep(0.15)
            t.join()

            result = _phase["result"] or {"success": False, "message": "Unknown error"}
            last_result = result
            is_last_link = (chain_idx == chain_total)

            # Button only flips back to idle on the FINAL link's terminal
            # yield — mid-chain it stays "..Please Wait.." since the next
            # link starts immediately. On that final link, when chaining,
            # the per-link message is replaced by the aggregate summary
            # below rather than shown here.
            btn_final = _btn_generate if is_last_link else gr.update()

            if result.get("success") and result.get("output_path"):
                chain_successes += 1
                out_path = Path(result["output_path"])
                try:
                    sz = out_path.stat().st_size
                    print(f"[generate] output file: {out_path}  ({sz} bytes)", flush=True)
                except Exception as e:
                    print(f"[generate] output file STAT FAILED: {out_path}  {e}", flush=True)
                # Record the total batch elapsed time so the next generation
                # can display it as a reference in the status bar (previous
                # batch time). Recorded per link so a mid-chain glance at the
                # status bar still reflects the most recently finished run.
                batch_elapsed = result.get("elapsed_seconds", 0.0)
                configure.APP_STATE["last_batch_elapsed_seconds"] = int(round(batch_elapsed))
                last_gallery = _get_recent_images()
                last_preview = str(out_path)
            else:
                last_gallery = _get_recent_images()

            if is_last_link and chain_total > 1:
                # Chain finished: report the aggregate result rather than
                # just the last link's own message.
                msg = f"Chain complete: {chain_successes}/{chain_total} succeeded."
            elif result.get("success") and result.get("output_path"):
                chain_prefix = f"Chain Position {chain_idx}/{chain_total}; " if chain_total > 1 else ""
                msg = (f"{chain_prefix}{result['message']} | Seed: {result['seed_used']} "
                       f"| Time: {int(round(result['elapsed_seconds']))}s")
            else:
                chain_prefix = f"Chain Position {chain_idx}/{chain_total}; " if chain_total > 1 else ""
                msg = f"{chain_prefix}{result.get('message', 'Unknown error')}"

            if result.get("success") and result.get("output_path"):
                preview_update = last_preview
            else:
                preview_update = _idle_preview_image()
            yield preview_update, last_gallery, msg, btn_final
            # A failed link does not abort the rest of the chain — the
            # remaining reference images still get their own attempt,
            # matching "images dont go missing while processing".

        # Start inactivity timer now that ALL links have finished
        _reset_inactivity_timer()

        # NOTE: nothing is saved here any more. The whole Generation page is
        # written to generation.json on SUBMISSION, up at the top of this
        # function, so a run that fails still keeps the settings that produced
        # it. See the block there for why the old save — which fired only on
        # success, and only when the Quality Preset was "Custom" — was losing
        # the prompt, seed, batch count and output format on every preset run.


    def on_generate_click(prompt, negative, width, height, steps, sampler,
    cfg_scale, seed, batch, output_format, quality_preset, ref_images=None,
    ref_mode=None, strength=None):
        """Dispatch a click on the single dynamic button. The button reads
        "Generate" when idle and starts a run (delegating to the do_generate
        generator, which yields its own button-state updates). While a run
        is in progress the button is disabled and shows "..Please Wait..",
        preventing concurrent runs. ref_images is the Flux.2 input-image list
        (from ref_images_state); it is ignored for Z-Image runs inside
        inference.generate_image(). ref_mode (from ref_mode_radio) selects
        Use All vs Chain All when 2+ reference images are present — see
        do_generate's chain_mode handling. Flash attention is decided
        automatically from the selected GPU's fp16 capability — no input
        here."""
        yield from do_generate(prompt, negative, width, height, steps, sampler,
    cfg_scale, seed, batch, output_format, quality_preset, ref_images, ref_mode,
    strength)

    # ── Reference-image Add / Clear handlers ────────────────────────────────
    def _render_input_gallery(paths: List[str]) -> Tuple[Any, Any]:
        """(gallery value, row visibility) for the right column's Input row.

        Returns BOTH because the row must collapse entirely when the list is
        empty — a visible-but-empty gallery still reserves its row height plus
        Gradio's empty-state placeholder, which would push the Output preview
        down the page for no benefit. Two outputs, one source of truth.

        Only real, still-present files are passed to the gallery. A path can go
        stale between being added and being rendered (the user deletes or
        renames the file, or unplugs the drive), and gr.Gallery given a missing
        path throws rather than skipping it, which would take the whole event
        handler down with it. The reference LIST itself is left untouched by
        this filtering — inference.generate_image() does its own existence
        check and reports missing files properly at generation time, which is
        where the user can actually act on the message."""
        alive = [p for p in (paths or []) if p and Path(str(p)).exists()]
        # Registered HERE, not at the call sites, because this is the one
        # function every gallery update passes through — the Add button, the
        # Clear button, the ref_images_state .change handler and the tab-select
        # resync all land on it. Doing it at each call site instead would mean
        # any future path that forgets is a crash rather than a cosmetic miss.
        # It must also happen before the update is RETURNED, since Gradio
        # validates outgoing paths while postprocessing this handler's result.
        _allow_local_files(alive)
        return gr.update(value=alive), gr.update(visible=bool(alive))

    def _render_ref_list(paths: List[str]) -> Any:
        """Show accumulated reference filenames one per line.

        Only the value is updated here — visibility is intentionally left
        alone. The textbox is built visible=True (see its construction), so it
        is already in the DOM and its value renders on the very first Add. When
        the list is empty the value is cleared and the placeholder shows,
        giving the same "nothing listed" read the old hide-when-empty gave,
        but without the Gradio first-reveal drop.
        """
        names = "\n".join(Path(p).name for p in (paths or []))
        return gr.update(value=names)

    def _render_strength_visibility(count: int) -> Any:
        """Denoise Strength is meaningful only when there is actually an input
        image to denoise FROM, and only on the -i img2img path.

        Flux.2 is excluded on purpose even though it accepts input images: it
        conditions through -r reference images (Kontext-style editing), which
        sd-cli drives with no strength parameter at all, so the slider would
        imply a control that does nothing."""
        c = configure.load_configuration()
        uses_init = _family_uses_init_image(c.get("imagegen_model_path", ""))
        return gr.update(visible=bool(uses_init and count > 0))

    def _render_ref_mode_visibility(count: int) -> Any:
        """Show the Use All / Chain All switch only when there is a real choice.

        TWO conditions, not one. There must be 2+ accumulated images, AND the
        diffuser must be able to FUSE them into a single run -- which only the
        "ref" families can. Flux.2 conditions on every image at once through
        repeatable -r flags; SDXL and FLUX.1 take a single -i init image and
        have no way to accept a second, so offering "Use All" there promises
        something sd-cli cannot do.

        Hiding the radio does not disable chaining. Those families always
        chain when given several images -- see do_generate. All the hidden
        radio means is that there is one possible behaviour rather than two,
        so there is nothing to ask about."""
        c = configure.load_configuration()
        return gr.update(visible=bool(
            _family_fuses_refs(c.get("imagegen_model_path", "")) and count > 1))

    def _add_ref_images(current):
        """Open the native picker and APPEND whatever was chosen to the
        accumulated list. Cancelling the dialog picks nothing, which leaves
        the existing list untouched rather than emptying it.

        Only the mode radio's VISIBILITY is touched here — never its value.
        A user who selected Use All and then adds a third image must still be
        on Use All (see the radio's construction comment).
        """
        acc = list(current or [])
        acc.extend(_browse_images())
        configure.set_ref_images(acc)
        # _render_input_gallery registers these paths with Gradio before
        # returning them; see _allow_local_files.
        _gal, _row = _render_input_gallery(acc)
        return (acc, _render_ref_list(acc), _render_ref_mode_visibility(len(acc)),
                _render_strength_visibility(len(acc)), _gal, _row)

    def _clear_ref_images():
        """The ONE thing that empties the reference-image list.

        The mode radio is hidden again (there is nothing left to choose
        between) but its VALUE is deliberately left alone: a Use All the user
        selected earlier survives a Clear, so the next batch of images runs
        the way they last asked for rather than silently reverting.
        """
        configure.set_ref_images([])
        return ([], _render_ref_list([]), gr.update(visible=False),
                gr.update(visible=False), gr.update(value=[]),
                gr.update(visible=False))

    def _resync_ref_widgets() -> tuple:
        """Recompute the reference-image widgets from the authoritative list,
        on arrival at the Generate tab.

        Returns (ref list textbox, input gallery, input row) — and deliberately
        NOT the mode radio or the strength slider, even though both also depend
        on the list. Those two are already re-synced on this same tab-select by
        _generate_family_updates(), which has to touch them anyway because
        their visibility depends on the FAMILY as well as the image count.
        Two handlers writing the same component on one trigger is a race with
        no upside, so the outputs are split cleanly between them: that one owns
        everything family-dependent, this one owns everything that is purely a
        function of the list.

        This function existed before but was never wired to anything, which is
        why the reference list and (now) the input thumbnails could go stale
        after a round trip through another tab.

        Denoise Strength IS included, and is the one overlap with the other
        handler. That is deliberate belt-and-braces: it is the widget whose
        staleness is most visible and most misleading (a strength slider on
        screen for Flux.2 promises a control sd-cli will never be given, since
        Flux.2 conditions through -r references and has no strength parameter).
        Both handlers compute it from the same _render_strength_visibility, so
        they cannot disagree -- one just guarantees the other. The cost of the
        duplicate is one redundant update per tab click; the cost of missing it
        is a control that lies about what the model will do.
        """
        imgs = configure.get_ref_images()
        gal, row = _render_input_gallery(imgs)
        return (_render_ref_list(imgs), gal, row,
                _render_strength_visibility(len(imgs)))

    def _on_ref_mode_change(mode):
        """Record the user's Use All / Chain All choice in session state, so
        it is the value everything else reads for the rest of the session."""
        configure.set_ref_mode(mode)

    _gen["ref_add_btn"].click(
        _add_ref_images,
        inputs=[_gen["ref_images_state"]],
        outputs=[_gen["ref_images_state"], _gen["ref_list_tb"], _gen["ref_mode_radio"],
                 _gen["strength_sld"], _gen["input_gallery"], _gen["input_row"]],
    )
    _gen["ref_clear_btn"].click(
        _clear_ref_images,
        inputs=None,
        outputs=[_gen["ref_images_state"], _gen["ref_list_tb"], _gen["ref_mode_radio"],
                 _gen["strength_sld"], _gen["input_gallery"], _gen["input_row"]],
    )
    # ── Belt-and-braces sync ─────────────────────────────────────────────
    # The two widgets whose visibility depends on the image list are also
    # driven off the list STATE itself, not only off the Add/Clear buttons.
    #
    # Hanging them on the buttons alone means every future code path that
    # touches the list has to remember to return their updates too, and a
    # path that forgets leaves a stale slider on screen until something else
    # happens to redraw it (switching tabs, for instance). Deriving from the
    # state makes the list the single source of truth: whatever changes it,
    # for whatever reason, these follow.
    def _on_ref_images_changed(imgs):
        n = len(imgs or [])
        gal, row = _render_input_gallery(imgs)
        return (_render_ref_mode_visibility(n), _render_strength_visibility(n),
                gal, row)

    _gen["ref_images_state"].change(
        _on_ref_images_changed,
        inputs=[_gen["ref_images_state"]],
        outputs=[_gen["ref_mode_radio"], _gen["strength_sld"],
                 _gen["input_gallery"], _gen["input_row"]],
    )

    _gen["ref_mode_radio"].change(
        _on_ref_mode_change,
        inputs=[_gen["ref_mode_radio"]],
        outputs=None,
    )

    # No handle kept: the only thing that ever chained off this event was the
    # post-run reference-image clear, which is gone (see the note below).
    _gen["generate_btn"].click(
        on_generate_click,
        inputs=[_gen["prompt_tb"], _gen["negative_tb"],
        _gen["width_dd"], _gen["height_dd"], _gen["steps_dd"],
        _gen["sampler_dd"], _gen["cfg_scale_sld"],
        _gen["seed_num"], _gen["batch_dd"], _gen["output_fmt_dd"],
        _gen["preset_dd"], _gen["ref_images_state"], _gen["ref_mode_radio"],
        _gen["strength_sld"]],
        outputs=[_gen["preview_img"], _gen["output_gallery"], status_box,
        _gen["generate_btn"]],
    )
    # NOTE: no .then(_clear_ref_images) here, on purpose. Reference images are
    # READ at generation time, never consumed — a finished run leaves the list
    # (and the mode radio) exactly as the user left it, so iterating on the
    # same images costs nothing but another click on Generate. "Clear Images"
    # is the only way to empty the list.

    # ── Restore To Defaults (Generation page) ───────────────────────────────
    def _restore_generation_defaults() -> tuple:
        """Repaint the settings panel to the loaded model's optimal values.

        Reads the SAME configure.family_generation_defaults() that a model
        change applies, so "restore" and "switch models" can never disagree
        about what optimal means for a given family. Choice lists are pushed
        alongside the values, because a stale dropdown could otherwise reject
        the very value being restored (steps 20 is not in Z-Image's list).

        The POSITIVE prompt is not touched — it is the user's own writing and
        a settings button has no business deleting it. The NEGATIVE prompt IS
        reset, because it is factory boilerplate with a real default
        (configure.DEFAULT_NEGATIVE_PROMPT) rather than authored content, and
        someone clicking "Restore To Defaults" plainly means to include it.

        Nothing is written to disk here. The restored values persist on the
        next Generate click like any other change to this page.
        """
        diff = configure.load_configuration().get("imagegen_model_path", "")
        d = configure.family_generation_defaults(diff)
        spec = configure.family_step_cfg(diff)
        step_choices, _ = spec["steps"]
        cfg_min, cfg_max, cfg_step, _ = spec["cfg"]
        sizes = configure.family_image_sizes(diff)
        fam_label = configure.diffuser_family_label(diff)
        try:
            gr.Info(f"Generation settings restored to the {fam_label} defaults.")
        except Exception:
            pass
        return (
            gr.update(choices=sizes, value=d["imagegen_width"]),
            gr.update(choices=sizes, value=d["imagegen_height"]),
            gr.update(choices=step_choices, value=d["imagegen_steps"]),
            gr.update(minimum=cfg_min, maximum=cfg_max, step=cfg_step,
                      value=d["imagegen_cfg_scale"]),
            gr.update(value=d["imagegen_sampling"]),
            gr.update(value=d["imagegen_batch_count"]),
            gr.update(value=d["output_format"]),
            gr.update(value=d["imagegen_seed"]),
            gr.update(value=d["imagegen_strength"]),
            gr.update(value=configure.DEFAULT_NEGATIVE_PROMPT),
            # The preset name. Be aware this one may not stick: every widget
            # above carries a .change handler that forces the preset to
            # "Custom" (see _set_custom), and those fire on the frontend after
            # this handler's outputs land, regardless of the order they are
            # listed in here. So the dropdown will often settle on "Custom"
            # a moment after the restore.
            #
            # Left as-is rather than worked around, because it is cosmetic and
            # the honest fix is disproportionate: _set_custom receives only the
            # one widget's value, so teaching it "these values ARE a named
            # preset, do not say Custom" means re-plumbing it to take the whole
            # panel as input. The VALUES restored are correct either way, and
            # "Custom" over a hand-restored set is not a lie.
            gr.update(value=d["imagegen_quality_preset"]),
        )

    _gen["restore_defaults_btn"].click(
        _restore_generation_defaults,
        inputs=None,
        outputs=[_gen["width_dd"], _gen["height_dd"], _gen["steps_dd"],
                 _gen["cfg_scale_sld"], _gen["sampler_dd"], _gen["batch_dd"],
                 _gen["output_fmt_dd"], _gen["seed_num"], _gen["strength_sld"],
                 _gen["negative_tb"], _gen["preset_dd"]],
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
                 _gen["width_dd"], _gen["height_dd"], _gen["strength_sld"],
                 _gen["ref_mode_radio"], _gen["batch_dd"],
                 _gen["output_fmt_dd"], _gen["seed_num"], _gen["preset_dd"]],
    )

    _gen["prompt_tb"].focus(
        _generate_gate_updates,
        inputs=None,
        outputs=[_gen["generate_row"], _gen["prompt_tb"], _gen["negative_tb"]],
    )

    # Third select handler: the widgets that depend ONLY on the reference-image
    # list (the filename list and the input thumbnail strip), which the two
    # handlers above deliberately do not touch. See _resync_ref_widgets for why
    # the outputs are split three ways rather than merged.
    _gen["generate_tab"].select(
        _resync_ref_widgets,
        inputs=None,
        outputs=[_gen["ref_list_tb"], _gen["input_gallery"], _gen["input_row"],
                 _gen["strength_sld"]],
    )

    # Clicking an input thumbnail puts that image in the preview box, so a
    # reference can be inspected full-size without leaving the page. Same
    # handler and same target as the output gallery below it — the preview box
    # is simply "whatever image is currently being looked at", whether that is
    # a generated result or an input about to be used.
    _gen["input_gallery"].select(
        on_gallery_select,
        inputs=None,
        outputs=_gen["preview_img"],
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

    gr.Markdown("### Backend Selection")
    with gr.Row():

        with gr.Column(scale=1):
            _cfg_w["threads_dd"] = gr.Dropdown(
                label="CPU Threads",
                choices=threads,
                value=cfg.get("encoder_threads", dt),
            )

        # ONE control for both halves of a run, where there used to be two
        # ("ImageGen Backend" and "Encoder Backend"). They were only ever set
        # to the same value: on a one-GPU machine there is no second device to
        # split across, and the decision that actually matters -- keeping the
        # encoder and VAE off the card so the diffusion model gets all of it --
        # is made per FAMILY by DIFFUSER_FAMILY_SPECS (encoder_to_cpu /
        # vae_to_cpu) and by Diffuser Placement, not here. Two dropdowns that
        # always agreed were two chances to disagree by accident.
        #
        # scale=2 against CPU Threads' scale=1, so the wider control (which
        # carries a full GPU name and VRAM figure) gets the room it needs and
        # the two sit neatly on one row.
        with gr.Column(scale=2):
            _cfg_w["proc_backend_dd"] = gr.Dropdown(
                label="Processing Method",
                choices=choices,
                value=_default_backend_value("backend_processing"),
                interactive=not is_cpu_only,
                info="Where both the encoder and the image generator run.",
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
    gr.Markdown("### Generation Model Files")
    _cfg_w["enc_path_tb"]  = gr.Textbox(value=cfg.get("encoder_model_path", ""),  visible=False)
    _cfg_w["diff_path_tb"] = gr.Textbox(value=cfg.get("imagegen_model_path", ""), visible=False)
    _cfg_w["vae_path_tb"]  = gr.Textbox(value=cfg.get("vae_model_path", ""),      visible=False)
    # Split-SDXL text encoders. Every SDXL gguf in circulation is UNet-only and
    # needs these supplied separately; Z-Image, Flux.2 and full SDXL
    # .safetensors checkpoints all leave them blank.
    _cfg_w["clip_l_path_tb"] = gr.Textbox(value=cfg.get("clip_l_model_path", ""), visible=False)
    _cfg_w["clip_g_path_tb"] = gr.Textbox(value=cfg.get("clip_g_model_path", ""), visible=False)
    # FLUX.1's second text encoder. Shares the clip_l row's machinery but not
    # its visibility -- FLUX.1 wants clip_l + t5xxl, split SDXL wants
    # clip_l + clip_g, so the three rows are shown independently.
    _cfg_w["t5xxl_path_tb"] = gr.Textbox(value=cfg.get("t5xxl_model_path", ""), visible=False)

    # Whether the currently-saved diffuser is SDXL decides two things on this
    # page: whether the CLIP-L/CLIP-G rows are shown at all, and whether the
    # Encoder box is presented as a required conditioner or an optional prompt
    # enhancer. Recomputed on every diffusion-model pick, below.
    # Whether the Encoder slot is this family's real conditioner or just a
    # prompt enhancer -- see _encoder_is_conditioner(). This replaced an
    # is-it-SDXL test, which stopped being the same question once FLUX.1
    # arrived: it also conditions without the Qwen3.
    _cfg_enc_cond = _encoder_is_conditioner(cfg.get("imagegen_model_path", ""))
    # CLIP boxes appear only when the model file actually lacks its own text
    # encoders -- see _clips_needed(). Nothing to do with the Encoder slot.
    _cfg_slots = _needed_slots(cfg.get("imagegen_model_path", ""))
    _cfg_needs_clips = _cfg_slots["clip_l"]
    _cfg_needs_clip_g = _cfg_slots["clip_g"]
    _cfg_needs_t5 = _cfg_slots["t5xxl"]
    _cfg_pack_label = configure.sdxl_packaging_label(cfg.get("imagegen_model_path", ""))

    # Initial interactive/value state for the two GPU-dependent controls
    # below is driven by the ACTUAL selected backend, not just install type.
    # A Vulkan-capable install with "CPU" currently selected must still show
    # GPU Layers / Diffuser Placement as locked at their CPU-forced values —
    # otherwise the controls look interactive but silently have no effect,
    # which was the root of the original bug report.
    # One Processing Method now drives both, so the two flags are the same
    # boolean. They are kept as separate names because the widgets they gate
    # remain distinct -- GPU Layers belongs to the encoder, Diffuser Placement
    # to the image generator -- and reading "enc_is_vulkan" beside the encoder
    # widget stays clearer than one shared name used in two contexts.
    proc_backend_val = _default_backend_value("backend_processing")
    enc_is_vulkan   = (not is_cpu_only) and ("Vulkan" in proc_backend_val)
    img_is_vulkan   = enc_is_vulkan

    with gr.Row():
        # ── Left column: the IMAGE GENERATOR's files ──────────────────────
        # Everything sd-cli loads for a run, largest first: diffusion
        # (1.49GB at Q4_0), CLIP-G (1.39GB), VAE (~335MB), CLIP-L (247MB).
        # These belong together because they are loaded together, in one
        # process, for one image. The CLIP pair are the diffuser's own text
        # encoders (sd-cli --clip_l / --clip_g), not a user-swappable choice.
        with gr.Column(scale=1):
            # Model Family sits ABOVE the diffusion model it describes, and is
            # the first thing on the page for a reason: it decides which of
            # the boxes below are even shown. Reading it after the boxes it
            # governs made the page look like it was rearranging itself for no
            # visible reason.
            with gr.Row():
                # Escape hatch for models auto-detection cannot place. sd.cpp-
                # native full checkpoints carry no architecture metadata, and
                # many SDXL finetunes are named with no "xl" token at all
                # (artiwaifu-diffusion-v1 is an SDXL 1.0 finetune), so both
                # detection inputs come up empty and the model would silently
                # fall back to the Z-Image command line.
                _cfg_w["img_family_dd"] = gr.Dropdown(
                    label="Model Family",
                    choices=configure.FAMILY_OVERRIDE_CHOICES,
                    value=cfg.get("imagegen_family_override",
                                  configure.FAMILY_OVERRIDE_AUTO),
                    info="Set this if a model is not detected correctly.",
                )
            with gr.Row():
                _cfg_w["diff_name_tb"] = gr.Textbox(
                    label="Diffusion Name",
                    value=cfg.get("imagegen_model_name", ""),
                    placeholder="z_image_turbo",
                    interactive=True,
                    scale=8,
                )
                # Browse and Clear stack vertically beside the textbox: one
                # narrow column of two buttons rather than a wide row of
                # three controls, which keeps the name field readable.
                with gr.Column(scale=1, min_width=90):
                    _cfg_w["diff_browse_btn"] = gr.Button("Browse...", size="sm")
                    _cfg_w["diff_clear_btn"] = gr.Button("Clear", size="sm")

            with gr.Row():
                _cfg_w["clip_g_name_tb"] = gr.Textbox(
                    label="CLIP-G Name",
                    value=cfg.get("clip_g_model_name", ""),
                    placeholder="clip_g.safetensors",
                    info="Auto-filled when found near the diffusion model.",
                    interactive=True,
                    visible=_cfg_needs_clip_g,
                    scale=8,
                )
                with gr.Column(scale=1, min_width=90):
                    _cfg_w["clip_g_browse_btn"] = gr.Button(
                        "Browse...", size="sm", visible=_cfg_needs_clip_g)
                    _cfg_w["clip_g_clear_btn"] = gr.Button(
                        "Clear", size="sm", visible=_cfg_needs_clip_g)

            with gr.Row():
                _cfg_w["vae_name_tb"] = gr.Textbox(
                    label="VAE Name",
                    value=cfg.get("vae_model_name", ""),
                    placeholder="ae.safetensors",
                    info="Auto-filled when found next to the diffusion model.",
                    interactive=True,
                    scale=8,
                )
                with gr.Column(scale=1, min_width=90):
                    _cfg_w["vae_browse_btn"] = gr.Button("Browse...", size="sm")
                    _cfg_w["vae_clear_btn"] = gr.Button("Clear", size="sm")

            with gr.Row():
                _cfg_w["clip_l_name_tb"] = gr.Textbox(
                    label="CLIP-L Name",
                    value=cfg.get("clip_l_model_name", ""),
                    placeholder="clip_l.safetensors",
                    info="Auto-filled when found near the diffusion model.",
                    interactive=True,
                    visible=_cfg_needs_clips,
                    scale=8,
                )
                with gr.Column(scale=1, min_width=90):
                    _cfg_w["clip_l_browse_btn"] = gr.Button(
                        "Browse...", size="sm", visible=_cfg_needs_clips)
                    _cfg_w["clip_l_clear_btn"] = gr.Button(
                        "Clear", size="sm", visible=_cfg_needs_clips)

            with gr.Row():
                _cfg_w["t5xxl_name_tb"] = gr.Textbox(
                    label="T5-XXL Name",
                    value=cfg.get("t5xxl_model_name", ""),
                    placeholder="t5-v1_1-xxl-encoder-Q4_K_M.gguf",
                    info=("FLUX.1's second encoder. A gguf quant is preferred: "
                          "it runs on the CPU, and fp16 costs 9.8GB of RAM."),
                    interactive=True,
                    visible=_cfg_needs_t5,
                    scale=8,
                )
                with gr.Column(scale=1, min_width=90):
                    _cfg_w["t5xxl_browse_btn"] = gr.Button(
                        "Browse...", size="sm", visible=_cfg_needs_t5)
                    _cfg_w["t5xxl_clear_btn"] = gr.Button(
                        "Clear", size="sm", visible=_cfg_needs_t5)
            # Says which packaging was detected and therefore why the CLIP
            # boxes are or are not on screen -- otherwise their appearing and
            # disappearing looks arbitrary.
            _cfg_w["pack_status_md"] = gr.Markdown(
                value=_cfg_pack_label, visible=bool(_cfg_pack_label))

        with gr.Column(scale=1):
            with gr.Row():
                _cfg_w["img_clip_dd"] = gr.Dropdown(label="CLIP Skip",
                                          choices=configure.CLIP_SKIP_CHOICES,
                                          value=cfg.get("imagegen_clip_skip", 2))
                # v-prediction override for SDXL finetunes. The gguf conversion
                # drops the flag that marks a checkpoint as v-pred, so sd.cpp
                # assumes eps and the output comes out washed out. Auto infers
                # it from the filename (noobai vpred, wai-*-vpred*); force it
                # here when a v-pred model is not named as one.
                _cfg_w["img_pred_dd"] = gr.Dropdown(
                    label="Prediction (SDXL)",
                    choices=configure.PREDICTION_CHOICES,
                    value=cfg.get("imagegen_prediction", configure.PREDICTION_AUTO),
                    info="Auto detects v-pred from the filename.",
                )
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
                          "Processing Method is CPU — sd.cpp will not touch the GPU at all."),
                    interactive=img_is_vulkan,
                )


        # ── Right column: the separate LLM encoder ────────────────────────
        # A different program entirely: a Qwen3 gguf run through llama.cpp,
        # in its own process, before sd-cli is invoked. For Z-Image and
        # Flux.2 it is the conditioner and is mandatory; for SDXL it cannot
        # condition anything (see _encoder_slot_updates) and is used only to
        # expand the prompt text. Kept apart from the left column so the two
        # roles are not read as interchangeable.
    # ── Encoder row ──────────────────────────────────────────────────────
    # The page is now two rows of one subject each, rather than two rows that
    # each mix both. Row 1 is everything sd-cli loads and every setting that
    # shapes the image; row 2 is the Qwen3 LLM and its settings. The encoder
    # path used to sit beside the diffusion files, which read as though it
    # were another file sd-cli loads -- it is not. It runs in a separate
    # llama.cpp process before sd-cli is invoked, and for SDXL and FLUX.1 it
    # never reaches sd-cli at all.
    gr.Markdown("### Encoder Model")
    with gr.Row():
        # ── Encoder model path (LEFT) ──
        with gr.Column(scale=1):
            with gr.Row():
                _cfg_w["enc_name_tb"] = gr.Textbox(
                    label=_ENC_LABEL_CONDITIONER if _cfg_enc_cond else _ENC_LABEL_ENHANCER,
                    value=cfg.get("encoder_model_name", ""),
                    placeholder="Qwen3-4b-Z-Image-Turbo",
                    info=_ENC_INFO_CONDITIONER if _cfg_enc_cond else _ENC_INFO_ENHANCER,
                    interactive=True,
                    scale=8,
                )
                with gr.Column(scale=1, min_width=90):
                    _cfg_w["enc_browse_btn"] = gr.Button("Browse...", size="sm")
                    _cfg_w["enc_clear_btn"] = gr.Button("Clear", size="sm")

        # ── Encoder (LLM) settings (RIGHT) ──
        with gr.Column(scale=1):
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
                          else "Processing Method is CPU — all layers run on CPU."),
                    interactive=enc_is_vulkan,
                )
                # Encoder flash attention is automatic and needs no toggle:
                # inference.py passes llama.cpp --flash-attn "auto". Unlike
                # sd.cpp's Vulkan diffusion FA, llama.cpp FA on a GPU without
                # fp16/coopmat2 just falls back to CPU for the attention math
                # (correct, only slower), so it is safe on any card including a
                # no-fp16 RX 470 — no fp16 gating, no checkbox.
    # NOTE: Prompt Template used to sit here under an "Advanced" heading. It
    # is now on the Preferences page (_build_preferences_tab_inner) and is
    # stored in data/preferences.json, not data/configuration.json — so the
    # Save All Configuration button below neither reads nor writes it.

    # ── Save / Revert ──
    # "Revert To Defaults" sits to the right of Save; variant="stop" gives it
    # the red styling (same as the Exit Program button). It only repaints the
    # widgets to their factory defaults — nothing is written to
    # configuration.json until the user clicks Save All Configuration.
    with gr.Row():
        _cfg_w["save_all_btn"] = gr.Button("Save All Configuration", variant="primary", size="lg")
        _cfg_w["revert_btn"]   = gr.Button("Revert To Defaults", variant="stop", size="lg")


    # ── Events: browse & scan — no status output, wire immediately ──
    def _browse_encoder():
        p = _browse_file()
        if not p:
            return gr.update(), gr.update()
        # The mmproj vision projector lives beside the VL model in the same
        # folder, so it shows up in this dialog. It is not a text encoder;
        # refuse it here and leave the current selection untouched.
        if inference.is_mmproj(p):
            try:
                gr.Warning("That file is an mmproj vision projector, not a text "
                           "encoder — not loaded. Pick the main model .gguf.")
            except Exception:
                pass
            return gr.update(), gr.update()
        return p, Path(p).stem

    def _browse_diffusion(current_vae_path: str, current_vae_name: str,
                          cur_l_path: str, cur_g_path: str, cur_t5_path: str):
        p = _browse_file()
        if not p:
            # 18 = the outputs list below: 5 path/name pairs, the encoder
            # relabel, 3 browse/clear button pairs, and the packaging line.
            return (gr.update(),) * 18
        vae_path, vae_name = _resolve_vae(p, current_vae_path, current_vae_name)
        # CLIP-L/CLIP-G are auto-detected the same way the VAE is: the
        # quantizers ship them beside the UNet, so requiring the user to hunt
        # for two more files by hand after picking a model is needless.
        # Only fill the CLIP slots when this model actually needs them; a
        # self-contained checkpoint gets them blanked so nothing stale is
        # passed to sd-cli as a redundant override.
        _slots = _needed_slots(p)
        if _slots["clip_l"] or _slots["clip_g"]:
            l_path, l_name, g_path, g_name = _resolve_clips(p, cur_l_path, cur_g_path)
        else:
            l_path, l_name, g_path, g_name = "", "", "", ""
        t5_path, t5_name = _resolve_t5xxl(p, cur_t5_path)
        # Picking the diffuser also determines whether the CLIP rows are shown
        # at all and what the Encoder slot means, so both refresh here rather
        # than making the user save and revisit the page.
        (enc_u, l_vis, _l_btn, g_vis, _g_btn,
         t5_vis, _t5_btn, pack_status) = _encoder_slot_updates(p)

        # Each name box carries a VALUE and a VISIBILITY change, so the two are
        # merged into one update apiece -- listing a component twice in
        # `outputs` would silently drop the first update. The visibility now
        # differs per slot (FLUX.1 shows clip_l and t5xxl but not clip_g), so
        # the row's own update has to be passed in rather than a shared one.
        def _name_update(value, vis):
            u = dict(vis)
            if not isinstance(value, dict):     # a real string, not gr.update()
                u["value"] = value
            return gr.update(**u)

        return (p, Path(p).stem, vae_path, vae_name,
                l_path, _name_update(l_name, l_vis),
                g_path, _name_update(g_name, g_vis),
                t5_path, _name_update(t5_name, t5_vis),
                enc_u, l_vis, l_vis, g_vis, g_vis, t5_vis, t5_vis, pack_status)

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
        inputs=[_cfg_w["vae_path_tb"], _cfg_w["vae_name_tb"],
                _cfg_w["clip_l_path_tb"], _cfg_w["clip_g_path_tb"],
                _cfg_w["t5xxl_path_tb"]],
        outputs=[_cfg_w["diff_path_tb"], _cfg_w["diff_name_tb"],
                 _cfg_w["vae_path_tb"], _cfg_w["vae_name_tb"],
                 _cfg_w["clip_l_path_tb"], _cfg_w["clip_l_name_tb"],
                 _cfg_w["clip_g_path_tb"], _cfg_w["clip_g_name_tb"],
                 _cfg_w["t5xxl_path_tb"], _cfg_w["t5xxl_name_tb"],
                 _cfg_w["enc_name_tb"],
                 _cfg_w["clip_l_browse_btn"], _cfg_w["clip_l_clear_btn"],
                 _cfg_w["clip_g_browse_btn"], _cfg_w["clip_g_clear_btn"],
                 _cfg_w["t5xxl_browse_btn"], _cfg_w["t5xxl_clear_btn"],
                 _cfg_w["pack_status_md"]]
    )
    # ── Clear buttons ────────────────────────────────────────────────────
    # Each slot needs a way to become EMPTY again, not just to point somewhere
    # else. Browse can only ever replace one path with another -- there was no
    # way to deselect an encoder, or to drop CLIP files in order to test a
    # self-contained checkpoint without them. Blanking both the visible name
    # and the hidden path keeps the pair consistent; nothing is written to
    # configuration.json until Save All Configuration, same as Browse.
    def _clear_slot():
        return "", ""

    def _clear_diffusion():
        """Clearing the diffuser also refreshes everything derived from it --
        the CLIP rows' visibility, the Encoder slot's label, and the packaging
        status line -- because with no model chosen none of those have a
        meaning to display."""
        (enc_u, l_vis, l_btn, g_vis, g_btn,
         t5_vis, t5_btn, pack) = _encoder_slot_updates("")
        return ("", "", enc_u, l_vis, l_btn, l_btn, g_vis, g_btn, g_btn,
                t5_vis, t5_btn, t5_btn, pack)

    def _browse_clip_l():
        p = _browse_file(_FILETYPES_VAE)
        return (p, Path(p).name) if p else (gr.update(), gr.update())

    def _browse_clip_g():
        p = _browse_file(_FILETYPES_VAE)
        return (p, Path(p).name) if p else (gr.update(), gr.update())

    def _browse_t5xxl():
        # No filetype filter: T5-XXL is normally a .gguf quant here but a
        # .safetensors is equally valid, and sd-cli takes either.
        p = _browse_file()
        return (p, Path(p).name) if p else (gr.update(), gr.update())

    _cfg_w["diff_clear_btn"].click(
        _clear_diffusion, inputs=None,
        outputs=[_cfg_w["diff_path_tb"], _cfg_w["diff_name_tb"],
                 _cfg_w["enc_name_tb"],
                 _cfg_w["clip_l_name_tb"], _cfg_w["clip_l_browse_btn"],
                 _cfg_w["clip_l_clear_btn"],
                 _cfg_w["clip_g_name_tb"], _cfg_w["clip_g_browse_btn"],
                 _cfg_w["clip_g_clear_btn"],
                 _cfg_w["t5xxl_name_tb"], _cfg_w["t5xxl_browse_btn"],
                 _cfg_w["t5xxl_clear_btn"],
                 _cfg_w["pack_status_md"]],
    )
    _cfg_w["vae_clear_btn"].click(
        _clear_slot, inputs=None,
        outputs=[_cfg_w["vae_path_tb"], _cfg_w["vae_name_tb"]],
    )
    _cfg_w["enc_clear_btn"].click(
        _clear_slot, inputs=None,
        outputs=[_cfg_w["enc_path_tb"], _cfg_w["enc_name_tb"]],
    )
    _cfg_w["clip_l_clear_btn"].click(
        _clear_slot, inputs=None,
        outputs=[_cfg_w["clip_l_path_tb"], _cfg_w["clip_l_name_tb"]],
    )
    _cfg_w["clip_g_clear_btn"].click(
        _clear_slot, inputs=None,
        outputs=[_cfg_w["clip_g_path_tb"], _cfg_w["clip_g_name_tb"]],
    )
    _cfg_w["t5xxl_clear_btn"].click(
        _clear_slot, inputs=None,
        outputs=[_cfg_w["t5xxl_path_tb"], _cfg_w["t5xxl_name_tb"]],
    )

    _cfg_w["clip_l_browse_btn"].click(
        _browse_clip_l, inputs=None,
        outputs=[_cfg_w["clip_l_path_tb"], _cfg_w["clip_l_name_tb"]],
    )
    _cfg_w["clip_g_browse_btn"].click(
        _browse_clip_g, inputs=None,
        outputs=[_cfg_w["clip_g_path_tb"], _cfg_w["clip_g_name_tb"]],
    )
    _cfg_w["t5xxl_browse_btn"].click(
        _browse_t5xxl, inputs=None,
        outputs=[_cfg_w["t5xxl_path_tb"], _cfg_w["t5xxl_name_tb"]],
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
                         info="Processing Method is CPU — all layers run on CPU.")

    # Both handlers hang off the SAME dropdown now. Two listeners on one event
    # rather than one listener returning both, because they own disjoint
    # outputs (GPU Layers vs Diffuser Placement) and each keeps its own
    # remembered-value dict; merging them would only tangle that.
    _cfg_w["proc_backend_dd"].change(
        _on_enc_backend_change,
        inputs=[_cfg_w["proc_backend_dd"], _cfg_w["enc_ngl_dd"]],
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
                         info="Processing Method is CPU — sd.cpp will not touch the GPU at all.")

    _cfg_w["proc_backend_dd"].change(
        _on_img_backend_change,
        inputs=[_cfg_w["proc_backend_dd"], _cfg_w["img_placement_dd"]],
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

    # Second listener on the same event, for the text-encoder rows. Separate
    # from the handler above rather than merged into it so the status-bar
    # message keeps a single writer -- see _on_diff_path_change_slots.
    w["diff_path_tb"].change(
        _on_diff_path_change_slots,
        inputs=[w["diff_path_tb"], w["clip_l_path_tb"], w["clip_g_path_tb"],
                w["t5xxl_path_tb"]],
        outputs=[w["clip_l_path_tb"], w["clip_l_name_tb"],
                 w["clip_g_path_tb"], w["clip_g_name_tb"],
                 w["t5xxl_path_tb"], w["t5xxl_name_tb"],
                 w["enc_name_tb"],
                 w["clip_l_browse_btn"], w["clip_l_clear_btn"],
                 w["clip_g_browse_btn"], w["clip_g_clear_btn"],
                 w["t5xxl_browse_btn"], w["t5xxl_clear_btn"],
                 w["pack_status_md"], w["img_family_dd"]],
    )

    def save_all(ep, en, dp, dn, vp, vn,
                 clp, cln, cgp, cgn, t5p, t5n,
                 proc_back, threads,
                 eb, ec, engl,
                 ic, img_pred, img_family, img_placement):
        # One Processing Method choice, parsed once, written to BOTH per-side
        # device keys. inference.py still reads encoder_vulkan_device and
        # imagegen_vulkan_device separately -- that split is real (the two run
        # in different processes and each needs its own -dev / --backend
        # argument), it is only the USER-FACING choice that has been merged.
        # So the two keys stay and simply always receive the same value.
        proc_parsed = configure.parse_backend_choice(proc_back)
        # Read BEFORE update_configuration() overwrites it, so the comparison
        # below is against what was on disk rather than what we just wrote.
        _prev_family_override = configure.load_configuration().get(
            "imagegen_family_override", configure.FAMILY_OVERRIDE_AUTO)
        # Never persist a vision projector as the encoder (covers a path carried
        # over from a prior session or a hand-edited config file). Clear it so
        # it can't reach the backend, and flag it in the saved-status message.
        mmproj_note = ""
        if ep and inference.is_mmproj(ep):
            ep, en = "", ""
            mmproj_note = ("The selected encoder was an mmproj vision projector "
                           "and has been cleared — pick the main model .gguf. ")
        configure.update_configuration({
            "encoder_model_path":  ep,  "encoder_model_name":  en,
            "imagegen_model_path": dp,  "imagegen_model_name": dn,
            "vae_model_path":      vp,  "vae_model_name":      vn,
            "clip_l_model_path":   clp, "clip_l_model_name":   cln,
            "clip_g_model_path":   cgp, "clip_g_model_name":   cgn,
            "t5xxl_model_path":    t5p, "t5xxl_model_name":    t5n,
            "backend_processing":  proc_back,
            # Per-side, and READ per-side by inference.py. The old code also
            # wrote a shared "vulkan_device" from the ImageGen dropdown only,
            # while enhance_prompt() read that same key for the ENCODER -- so
            # picking "CPU" for ImageGen silently set the encoder's device to
            # -1. The two keys below were already being written and never read.
            "encoder_vulkan_device": proc_parsed["vulkan_device"],
            "imagegen_vulkan_device": proc_parsed["vulkan_device"],
            "encoder_threads":     int(threads),
            "imagegen_threads":    int(threads),
            "encoder_batch_size":  int(eb),
            "encoder_ctx_size":    int(ec),
            "encoder_gpu_layers":  int(engl),
            "imagegen_clip_skip":  int(ic),
            "imagegen_prediction": img_pred,
            "imagegen_family_override": img_family,
            "imagegen_placement":  img_placement,
            "first_run":           False,
        })
        # Clearing imagegen_last_family forces the Generation page to re-snap
        # steps/cfg/size/sampler to the family defaults next time it is
        # opened -- correct when the user has just told us the model is a
        # different family than we thought, and destructive otherwise.
        #
        # This used to fire on EVERY save. Saving the Configuration page for
        # any reason at all -- correcting a VAE path, changing thread count,
        # switching Processing Method -- therefore silently discarded whatever
        # the user had tuned on the Generation page the moment they navigated
        # back to it. That was survivable while those values were transient;
        # now that generation.json restores them faithfully it is a real loss,
        # and it is exactly the "something is reverting my settings" behaviour
        # worth hunting down.
        #
        # So it is conditional: only when the Model Family override actually
        # differs from what was stored does the marker get cleared.
        if img_family != _prev_family_override:
            try:
                configure.update_generation({"imagegen_last_family": ""})
            except Exception:
                pass
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
        return mmproj_note + msg, row_u, prompt_u, neg_u

    w["save_all_btn"].click(
        save_all,
        inputs=[
            w["enc_path_tb"], w["enc_name_tb"], w["diff_path_tb"], w["diff_name_tb"],
            w["vae_path_tb"], w["vae_name_tb"],
            w["clip_l_path_tb"], w["clip_l_name_tb"],
            w["clip_g_path_tb"], w["clip_g_name_tb"],
            w["t5xxl_path_tb"], w["t5xxl_name_tb"],
            w["proc_backend_dd"], w["threads_dd"],
            w["enc_batch_dd"], w["enc_ctx_dd"],
            w["enc_ngl_dd"],
            w["img_clip_dd"], w["img_pred_dd"], w["img_family_dd"],
            w["img_placement_dd"],
        ],
        outputs=[status_box, _gen["generate_row"], _gen["prompt_tb"],
                 _gen["negative_tb"]],
    ).then(
        # ── Refresh the Generation page AT THE MOMENT THE MODEL CHANGES ─────
        # Saving here is the only way the diffusion model can change, and every
        # family-dependent control on the Generation page is now stale: the
        # Settings header names the old family, the steps/cfg/size choice lists
        # are the old family's, and — the visible symptom that prompted this —
        # Denoise Strength is still on screen after switching from FLUX.1 (which
        # denoises from an -i init image and has a strength) to Flux.2 (which
        # conditions through -r references and has no strength parameter at
        # all, so the slider promises a control sd-cli will never receive).
        #
        # All of that WAS handled, but only by the generate_tab.select handler
        # — so the correction landed whenever the user next clicked the tab,
        # not when the change happened. Anything that got them back to the page
        # without a fresh tab-select left the stale widgets showing. Driving the
        # same helper from the save closes that window: by the time the user
        # looks at the Generation page it is already correct, whatever route
        # they took to get there.
        #
        # Identical function and identical outputs to the tab-select wiring, so
        # the two paths cannot drift. Running it twice (save, then a tab click)
        # is harmless: it is idempotent, and the second pass sees the family
        # marker already updated and so does nothing.
        _generate_family_updates,
        inputs=[_gen["steps_dd"], _gen["cfg_scale_sld"],
                _gen["width_dd"], _gen["height_dd"]],
        outputs=[_gen["settings_header"], _gen["ref_row"], status_box,
                 _gen["sampler_dd"], _gen["steps_dd"], _gen["cfg_scale_sld"],
                 _gen["width_dd"], _gen["height_dd"], _gen["strength_sld"],
                 _gen["ref_mode_radio"], _gen["batch_dd"],
                 _gen["output_fmt_dd"], _gen["seed_num"], _gen["preset_dd"]],
    )

    # ── Revert To Defaults ──
    # Repaints every Configuration widget from the single canonical source
    # (configure.default_configuration()), so this button, a fresh install and
    # the load-time backfill can never disagree. Confirmation is a gr.Info
    # toast rather than the shared status bar: clearing the diffusion path
    # below re-fires the diff-path .change handler, which writes its own line
    # to the status bar, so a toast avoids two writers fighting over it.
    #
    # Backends revert to CPU (first choice), which is why GPU Layers and
    # Diffuser Placement are returned disabled at their CPU-forced values
    # (0 layers / Full CPU) — matching how the page locks them whenever a
    # backend is CPU. Model name/path boxes are emptied. Nothing is persisted
    # until Save All Configuration is clicked; the Generate gate therefore
    # stays as-is (it reads configuration.json, not these textboxes).
    def revert_config():
        d = configure.default_configuration()
        choices = _backend_choices()
        cpu_backend = choices[0] if choices else "CPU"
        try:
            gr.Info("Configuration reverted to defaults — click "
                    "'Save All Configuration' to apply.")
        except Exception:
            pass
        return (
            gr.update(value=cpu_backend),                       # proc_backend_dd
            gr.update(value=d["encoder_threads"]),              # threads_dd
            gr.update(value=""),                                # enc_name_tb
            gr.update(value=""),                                # enc_path_tb
            gr.update(value=""),                                # diff_name_tb
            gr.update(value=""),                                # diff_path_tb
            gr.update(value=""),                                # vae_name_tb
            gr.update(value=""),                                # vae_path_tb
            gr.update(value=""),                                # clip_l_name_tb
            gr.update(value=""),                                # clip_l_path_tb
            gr.update(value=""),                                # clip_g_name_tb
            gr.update(value=""),                                # clip_g_path_tb
            gr.update(value=""),                                # t5xxl_name_tb
            gr.update(value=""),                                # t5xxl_path_tb
            gr.update(value=d["encoder_batch_size"]),           # enc_batch_dd
            gr.update(value=d["encoder_ctx_size"]),             # enc_ctx_dd
            gr.update(value=0, interactive=False,               # enc_ngl_dd
                      info="Processing Method is CPU — all layers run on CPU."),
            gr.update(value=d["imagegen_clip_skip"]),           # img_clip_dd
            gr.update(value=d["imagegen_prediction"]),          # img_pred_dd
            gr.update(value=d["imagegen_family_override"]),     # img_family_dd
            gr.update(value=configure.DIFFUSER_PLACEMENT_FULL_CPU,  # img_placement_dd
                      interactive=False,
                      info="Processing Method is CPU — sd.cpp will not touch the GPU at all."),
        )

    w["revert_btn"].click(
        revert_config,
        inputs=[],
        outputs=[
            w["proc_backend_dd"], w["threads_dd"],
            w["enc_name_tb"], w["enc_path_tb"],
            w["diff_name_tb"], w["diff_path_tb"],
            w["vae_name_tb"], w["vae_path_tb"],
            w["clip_l_name_tb"], w["clip_l_path_tb"],
            w["clip_g_name_tb"], w["clip_g_path_tb"],
            w["t5xxl_name_tb"], w["t5xxl_path_tb"],
            w["enc_batch_dd"], w["enc_ctx_dd"], w["enc_ngl_dd"],
            w["img_clip_dd"], w["img_pred_dd"], w["img_family_dd"],
            w["img_placement_dd"],
        ],
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
            # Sizes the Input thumbnail strip in the Generation page's right
            # column. It shares that column with the Output preview, so this
            # is a direct trade: larger thumbnails are easier to identify,
            # but every pixel comes out of the preview below.
            #
            # The "next launch" note is not boilerplate — it is accurate. The
            # value is baked into the stylesheet when the app assembles it at
            # startup (see build_app), because Gradio offers no way to swap a
            # running app's CSS. Saying so here is better than the user
            # clicking Save and wondering why nothing moved.
            _prf["input_thumb_dd"] = gr.Dropdown(
                label="Input Thumbnail Size",
                choices=configure.INPUT_THUMBNAIL_CHOICES,
                value=configure.get_input_thumbnail_size(),
                info=("Pixel size of the Generation page's Input thumbnails. "
                      "Applies on next launch."),
            )
        with gr.Column(scale=1):
            _prf["encoder_debug_chk"] = gr.Checkbox(
                label="Encoder Model Debug",
                value=bool(prefs.get("encoder_model_debug", False)),
                info="Print encoder output to terminal.",
            )

    # "Revert To Defaults" sits to the right of Save; variant="stop" gives it
    # the red styling. It only repaints the widgets — preferences.json is not
    # written until the user clicks Save All Preferences.
    with gr.Row():
        _prf["save_prefs_btn"]   = gr.Button("Save All Preferences",
                                             variant="primary", size="lg")
        _prf["revert_prefs_btn"] = gr.Button("Revert To Defaults",
                                             variant="stop", size="lg")


def _wire_preferences_events(status_box: gr.Textbox) -> None:
    """Register the Preferences save event.

    Saving also re-slices the Generate page's gallery, so a new Max Thumbnails
    value takes effect immediately rather than at the next launch. The cached
    output listing is unsliced (see _get_recent_images), so this costs a
    re-render, not a rescan.
    """
    def save_prefs(prompt_template, max_thumbs, input_thumb, encoder_debug):
        try:
            thumbs = int(max_thumbs)
        except (TypeError, ValueError):
            thumbs = configure.DEFAULT_MAX_THUMBNAILS
        if thumbs not in configure.MAX_THUMBNAIL_CHOICES:
            thumbs = configure.DEFAULT_MAX_THUMBNAILS
        try:
            in_thumb = int(input_thumb)
        except (TypeError, ValueError):
            in_thumb = configure.DEFAULT_INPUT_THUMBNAIL
        if in_thumb not in configure.INPUT_THUMBNAIL_CHOICES:
            in_thumb = configure.DEFAULT_INPUT_THUMBNAIL
        configure.update_preferences({
            "prompt_template": prompt_template,
            "max_thumbnails":  thumbs,
            "input_thumbnail_size": in_thumb,
            "encoder_model_debug": bool(encoder_debug),
        })
        # Max Thumbnails re-slices the gallery immediately (the cached listing
        # is unsliced, so this costs a re-render, not a rescan). Input
        # Thumbnail Size cannot do the same — it lives in the stylesheet, which
        # is fixed for the life of the process — so the message says so rather
        # than letting the user wonder.
        return ("All preferences saved! Input Thumbnail Size applies on next "
                "launch."), _get_recent_images(thumbs)

    _prf["save_prefs_btn"].click(
        save_prefs,
        inputs=[_prf["prompt_template_tb"], _prf["max_thumbs_dd"],
                _prf["input_thumb_dd"], _prf["encoder_debug_chk"]],
        outputs=[status_box, _gen["output_gallery"]],
    )

    # ── Revert To Defaults ──
    # Repaints the three Preferences widgets to the same values that seed a
    # fresh preferences.json (configure._default_preferences()). Confirmation
    # is a gr.Info toast, matching the Configuration page's revert button, and
    # nothing is written to disk until Save All Preferences is clicked.
    def revert_prefs():
        try:
            gr.Info("Preferences reverted to defaults — click "
                    "'Save All Preferences' to apply.")
        except Exception:
            pass
        return (
            gr.update(value=configure.DEFAULT_PROMPT_TEMPLATE),  # prompt_template_tb
            gr.update(value=configure.DEFAULT_MAX_THUMBNAILS),   # max_thumbs_dd
            gr.update(value=configure.DEFAULT_INPUT_THUMBNAIL),  # input_thumb_dd
            gr.update(value=False),                              # encoder_debug_chk
        )

    _prf["revert_prefs_btn"].click(
        revert_prefs,
        inputs=[],
        outputs=[_prf["prompt_template_tb"], _prf["max_thumbs_dd"],
                 _prf["input_thumb_dd"], _prf["encoder_debug_chk"]],
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
        # Prompt spellcheck. Reported here because it is invisible when it
        # fails: the prompt boxes simply do not underline anything, which looks
        # identical to "there are no typos". The dictionary is built by the
        # installer, so the fix is always the same and is worth stating rather
        # than leaving the user to work out.
        _bdic = configure.get_dictionaries_dir() / f"{configure.SPELLCHECK_LANGUAGE}.bdic"
        if _bdic.exists():
            L.append(f"  spellcheck dict  : {_bdic}")
        else:
            L.append(f"  spellcheck dict  : NOT BUILT — prompt spellcheck is off")
            L.append(f"                     expected at {_bdic}")
            L.append(f"                     run option 2 (Installation) to build it")
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

/* ── Input gallery: the same one-row horizontal scroller, in column 3 ─────
Identical technique to #output-gallery above and for identical reasons — see
that block for the full explanation of why Gradio's native grid has to be
replaced rather than configured. Only two things differ:

  * the height comes from a PREFERENCE (Input Thumbnail Size) rather than a
    constant, interpolated below from configure.get_input_thumbnail_size().
    Every pixel this row occupies is a pixel the Output preview beneath it
    loses, which is exactly why it is the user's choice and not a fixed value.
  * the cell width subtracts configure.INPUT_GALLERY_PADDING rather than a
    literal 16, though the number is the same — Gradio's .grid-wrap adds 8px
    above and below, and the cells have to stay square.

The horizontal scroller is what stops this row from ever growing taller: no
matter how many reference images are added, they extend rightward behind a
scrollbar instead of wrapping onto a second line and shoving Output down the
page. That is the whole requirement — the input thumbnails must not push the
Input/Output sections around. ─────────────────────────────────────────────── */
#input-gallery .thumbnail-lg > img,
#input-gallery .thumbnail-lg img,
#input-gallery .grid-wrap img,
#input-gallery .gallery-item img,
#input-gallery .thumbnail-item img,
#input-gallery img {
    object-fit: contain !important;
    width: 100% !important;
    height: 100% !important;
    max-width: 100% !important;
    max-height: 100% !important;
}
#input-gallery .thumbnail-lg,
#input-gallery .gallery-item,
#input-gallery .grid-wrap .gallery-item {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    overflow: hidden !important;
    background: var(--background-fill-secondary) !important;
}
#input-gallery .grid-container {
    display: grid !important;
    grid-auto-flow: column !important;
    grid-template-columns: none !important;
    grid-template-rows: 1fr !important;
    grid-auto-columns: calc(__INPUT_THUMB__px - __INPUT_PAD__px) !important;
    height: 100% !important;
}
#input-gallery .grid-wrap {
    height: __INPUT_THUMB__px !important;
    overflow-x: auto !important;
    overflow-y: hidden !important;
    scrollbar-width: thin !important;
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
    # Input gallery row height. Unlike the two above this is a PREFERENCE, not
    # a constant, so it is read here rather than hardcoded — the same accessor
    # _build_generate_tab_inner() uses for the gr.Gallery(height=...) kwarg, so
    # the CSS (which carries !important and would otherwise win) and the kwarg
    # always agree.
    #
    # Read ONCE, when the stylesheet is assembled at launch. Changing Input
    # Thumbnail Size therefore takes effect on the next launch, not the moment
    # Save All Preferences is clicked — Gradio has no hook for swapping the
    # stylesheet on a running Blocks app, and the alternatives (injecting a
    # <style> tag through gr.HTML, or swapping elem_classes at runtime) are
    # either at the mercy of Gradio's HTML sanitiser or unreliable across
    # versions. The Preferences info text says so, so the behaviour is stated
    # rather than surprising.
    _css = _css.replace("__INPUT_THUMB__", str(configure.get_input_thumbnail_size()))
    _css = _css.replace("__INPUT_PAD__", str(configure.INPUT_GALLERY_PADDING))

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

    # Kept so _allow_local_files() can extend this app's allowed_paths at
    # runtime, which is what lets the Input gallery display reference images
    # from outside the project folder. Set here rather than passed around,
    # because the handlers that need it are closures wired above.
    global _blocks_app
    _blocks_app = app

    return app, _css