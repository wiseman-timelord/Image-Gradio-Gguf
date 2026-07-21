"""
inference.py - Image generation, model handling, and text generation.
Model discovery, probing, prompt enhancement, and image generation.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import shutil
import struct
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import scripts.configure as configure
import scripts.utilities as utilities


# ---------------------------------------------------------------------------
# GGUF constants
# ---------------------------------------------------------------------------

GGUF_MAGIC = b"GGUF"


class _GGUFArraySkipped(Exception):
    """Raised when a KV array is too large to parse and too expensive to skip.

    The file cursor is left mid-array, so nothing after it can be read; the
    parse must stop, not continue. See _read_gguf_value()'s ARRAY branch.
    """
GGUF_TYPES = {
    0: "UINT8", 1: "INT8", 2: "UINT16", 3: "INT16",
    4: "UINT32", 5: "INT32", 6: "FLOAT32", 7: "BOOL",
    8: "STRING", 9: "ARRAY", 10: "UINT64", 11: "INT64",
    12: "FLOAT64",
}


# ---------------------------------------------------------------------------
# Backend discovery
# ---------------------------------------------------------------------------

def find_sd_cpp() -> Optional[Path]:
    """
    Find the stable-diffusion.cpp CLI binary.  Search order:
      1. ./data/stable_diffusion_binaries/sd-cli.exe  (current installer output)
      2. ./data/stable_diffusion_binaries/sd.exe       (legacy fallback)
      3. PATH
    """
    bin_dir = configure.get_sd_bin_dir()
    for name in ("sd-cli.exe", "sd-cli", "sd.exe", "sd"):
        p = bin_dir / name
        if p.exists():
            return p
    for name in ("sd-cli", "sd"):
        found = shutil.which(name)
        if found:
            return Path(found)
    return None


def find_llama_completion() -> Optional[Path]:
    """
    Find llama-completion.exe -- the one-shot completion binary.

    Deliberately NOT llama-cli: upstream reorganised its tools, tools/main/ is
    gone, and `llama-cli` is now an interactive chat REPL built on the server
    stack (it prints an ASCII logo and /exit, /regen, /clear commands, then
    waits on stdin). Feeding it -p and reading stdout cannot work. The
    successor to the old one-shot main.cpp is tools/completion/ ->
    llama-completion. Legacy names are kept only as a fallback for an old
    install directory.

    Search order:
      1. ./data/llama_cpp_binaries/llama-completion.exe  (compiled by installer)
      2. PATH
    """
    bin_dir = configure.get_llama_bin_dir()
    for name in ("llama-completion.exe", "llama-completion"):
        p = bin_dir / name
        if p.exists():
            return p
    found = shutil.which("llama-completion")
    return Path(found) if found else None


# ---------------------------------------------------------------------------
# Model probing
# ---------------------------------------------------------------------------

def scan_directory(dir_path: str,
                   extensions: Tuple[str, ...] = (".gguf", ".safetensors")
                   ) -> List[Dict[str, Any]]:
    path = Path(dir_path).expanduser()
    if not path.is_dir():
        return []
    results: List[Dict[str, Any]] = []
    for ext in extensions:
        for fp in path.rglob(f"*{ext}"):
            # Cheap name reject before the probe so a projector sitting beside
            # a model is never enumerated (and so never auto-selected).
            if is_mmproj_name(fp.name):
                continue
            meta = probe_model(str(fp))
            if meta and not is_mmproj(meta):
                results.append(meta)
    return results


def probe_model(file_path: str) -> Optional[Dict[str, Any]]:
    p = Path(file_path).expanduser()
    if not p.exists():
        return None
    ext = p.suffix.lower()
    if ext == ".gguf":
        return _probe_gguf(p)
    if ext == ".safetensors":
        return _probe_safetensors(p)
    return None


def find_model_by_name(substring: str, search_dirs: List[str]) -> Optional[str]:
    for d in search_dirs:
        dp = Path(d).expanduser()
        if not dp.is_dir():
            continue
        for fp in dp.rglob("*"):
            if (substring.lower() in fp.name.lower()
                    and fp.suffix.lower() in (".gguf", ".safetensors")
                    and not is_mmproj_name(fp.name)):
                return str(fp)
    return None


def get_quantization_label(filename: str) -> str:
    """
    Read the quantization out of a filename.

    Matched with separator boundaries rather than a bare substring test, and
    longest-token-first, because real filenames carry decoy digits+letters:
      perfeczion_10BF16-Q4_K_M.gguf     -> Q4_K_M   (not F16, from "10BF16")
      zImageTurboNSFW_60BF16Diffusion-Q6_K.gguf -> Q6_K
      darkBeastMar2126Latest_dbzit9DIMRclaw-Q8_0.gguf -> Q8_0
    A leading/trailing alphanumeric disqualifies a match, which is also what
    stops "q4_0" being found inside "iq4_0" and "q4_k" inside "q4_k_m".
    """
    n = filename.lower()
    for tok in sorted(configure.QUANT_TOKENS, key=len, reverse=True):
        if re.search(rf"(?<![a-z0-9]){re.escape(tok)}(?![a-z0-9_])", n):
            return tok.upper()
    return "Unknown"


def _name_haystacks(meta: Dict[str, Any]) -> List[str]:
    """Filename plus up to two containing folder names, lowercased.

    The folders matter: BigDannyPt's collection nests as
    DarkBeast/DBZiT9-DIMRClaw/darkBeastMar2126Latest_...gguf, and users keep
    that layout when they download, so the parent folders frequently identify
    the family better than the file itself does.
    """
    out: List[str] = [str(meta.get("filename", "")).lower()]
    fp = str(meta.get("file", ""))
    if fp:
        try:
            for parent in Path(fp).parents[:2]:
                out.append(parent.name.lower())
        except Exception:
            pass
    return [h for h in out if h]


def _matches_any(patterns: List[str], haystacks: List[str]) -> bool:
    for pat in patterns:
        for h in haystacks:
            if re.search(pat, h):
                return True
    return False


def _arch_of(meta: Dict[str, Any]) -> str:
    """general.architecture from GGUF metadata, lowercased ('' if absent)."""
    return str(meta.get("architecture", "")).strip().lower()


# ---------------------------------------------------------------------------
# mmproj / vision-projector detection  --  these must NEVER load
# ---------------------------------------------------------------------------
# A VL model (Qwen3-VL-4B etc.) ships its language backbone in the main *.gguf
# and its VISION tower in a SEPARATE companion file, conventionally named
# mmproj-*.gguf. This program uses the language backbone as a TEXT encoder only
# (sd.cpp --llm / llama-completion) and has no use for image input, so the
# mmproj must never be loaded. Neither sd.cpp nor llama-completion auto-loads a
# sibling mmproj without an explicit --mmproj flag (which we never pass), but a
# projector can still slip in one way: the user pointing the Encoder selection
# straight at the mmproj file, because it sits in the same folder as the model.
# is_mmproj() is the single gate every selection path funnels through so that
# can't happen -- whichever folder or drive the model was picked from.
#
# Two signals, either one is decisive:
#   1. Filename -- the ecosystem names these mmproj-*.gguf (also mm-proj / mm_proj
#      / *-mmproj-* and, rarely, projector-*). This is the primary signal and is
#      exactly what the user sees in the folder.
#   2. GGUF metadata -- a projector reports general.architecture "clip" AND
#      carries a vision marker (clip.has_vision_encoder true, or a projector_type).
#      A PLAIN CLIP-L/-G TEXT encoder also reports arch "clip" but has NO vision
#      encoder, so the vision marker -- not the arch alone -- is what separates a
#      projector (reject) from a legitimate CLIP text encoder (keep).
_MMPROJ_NAME_RE = re.compile(
    r"(?:^|[-_. ])mm[-_. ]?proj(?:ector)?(?![a-z])"
    r"|(?:^|[-_. ])projector(?![a-z])",
    re.IGNORECASE,
)


def is_mmproj_name(name: str) -> bool:
    """True if the bare filename looks like a vision projector (mmproj)."""
    return bool(_MMPROJ_NAME_RE.search(str(name)))


def is_mmproj(model: Any, meta: Optional[Dict[str, Any]] = None) -> bool:
    """True if `model` is a multimodal vision projector rather than a usable
    text encoder / diffuser / VAE. Accepts a path (str/Path) or an already
    probed metadata dict. Filename decides first (cheap, universal); GGUF
    metadata is the fallback for an oddly-named projector.
    """
    if isinstance(model, dict):
        meta = model
        name = str(meta.get("filename") or Path(str(meta.get("file", ""))).name)
    else:
        name = Path(str(model)).name

    if name and is_mmproj_name(name):
        return True

    # Only probe if we were handed a path and no metadata, and it is a gguf on
    # disk. A missing/unreadable file simply falls through to False here; the
    # existence checks live at the call sites.
    if meta is None and not isinstance(model, dict):
        p = Path(str(model)).expanduser()
        if p.suffix.lower() == ".gguf" and p.exists():
            meta = _probe_gguf(p) or {}

    if meta:
        arch = str(meta.get("architecture", "")).lower()
        has_vision = bool(meta.get("has_vision_encoder"))
        projector_type = str(meta.get("projector_type", "")).strip()
        # Explicit vision encoder is decisive on its own; combined with a
        # clip-family arch it is doubly so. Never flag on arch alone.
        if has_vision:
            return True
        if arch in ("clip", "clip_vision", "mmproj") and projector_type:
            return True
    return False


def classify_model(meta: Dict[str, Any]) -> str:
    """
    Return "encoder", "diffusion", "vae" or "unknown" for one probed model.

    Architecture metadata wins when present — every Z-Image-Turbo diffusion
    gguf reports `lumina2`, whether it is the stock z_image_turbo-Q#.gguf or
    one of the community finetunes (DarkBeast / EventHorizon / PerfecZion /
    SmoothMix Ultimate / ZiT Anime / ZiT NSFW), so they all land in
    "diffusion" regardless of how creatively they are named. Filename and
    folder patterns are the fallback for unreadable metadata and safetensors.
    See configure.py's "Model family identification" block for the lists.
    """
    # A vision projector must never be routed to any usable bin. Catch it
    # BEFORE the arch check: a projector's general.architecture is "clip",
    # which is in ENCODER_ARCHITECTURES, so arch routing would otherwise misfile
    # it as an encoder and offer it as one. "unknown" keeps it out of every
    # encoder list, dropdown, and auto-detection path -- present and future.
    if is_mmproj(meta):
        return "unknown"

    arch = _arch_of(meta)
    if arch:
        if arch in configure.ENCODER_ARCHITECTURES:
            return "encoder"
        if arch in configure.DIFFUSION_ARCHITECTURES:
            return "diffusion"
        if arch in configure.VAE_ARCHITECTURES:
            return "vae"

    hay = _name_haystacks(meta)
    fmt = str(meta.get("format", ""))

    if fmt.startswith("Safetensors"):
        # ae.safetensors is the only safetensors this program consumes.
        return "vae" if _matches_any(configure.VAE_NAME_PATTERNS, hay) else "unknown"

    # Encoder before diffusion: the encoder gguf is named after the diffuser
    # it serves ("Qwen3-4b-Z-Image-Turbo-AbliteratedV1"), so testing the
    # diffusion patterns first would misfile it.
    if _matches_any(configure.ENCODER_NAME_PATTERNS, hay):
        return "encoder"
    if _matches_any(configure.DIFFUSION_NAME_PATTERNS, hay):
        return "diffusion"
    if _matches_any(configure.VAE_NAME_PATTERNS, hay):
        return "vae"
    return "unknown"


def is_sd_classic_model(file_path: str, meta: Optional[Dict[str, Any]] = None) -> bool:
    """
    True only if the checkpoint positively looks like an SD1.x / SD2.x / SDXL
    class model, i.e. one with a CLIP text encoder that --clip-skip applies to.

    Unknown means False. Z-Image conditions through the Qwen3 LLM and has no
    CLIP, so "no clip-skip" is the correct default for everything this program
    is built to run.
    """
    if meta is None:
        meta = probe_model(file_path) or {}
    arch = _arch_of(meta)
    if arch:
        if arch in configure.SD_CLASSIC_ARCHITECTURES:
            return True
        if arch in configure.DIFFUSION_ARCHITECTURES:
            return False  # known diffuser, known not to be CLIP-conditioned
    hay = _name_haystacks(meta) or [Path(str(file_path)).name.lower()]
    return _matches_any(configure.SD_CLASSIC_NAME_PATTERNS, hay)


def categorize_models(models: List[Dict[str, Any]]
                      ) -> Dict[str, List[Dict[str, Any]]]:
    cats: Dict[str, List[Dict[str, Any]]] = {
        "encoder": [], "diffusion": [], "vae": [], "unknown": [],
    }
    for m in models:
        cats[classify_model(m)].append(m)
    return cats


# ---------------------------------------------------------------------------
# Cross-model compatibility (encoder size / VAE family vs the diffuser)
# ---------------------------------------------------------------------------

def _encoder_hidden_dim(enc_path: str) -> Optional[int]:
    """Encoder hidden dimension: GGUF `embedding_length` metadata first
    (2560 for Qwen3-4B, 4096 for Qwen3-8B), filename family as a fallback."""
    meta = probe_model(enc_path) or {}
    emb = meta.get("embedding_length")
    try:
        if emb is not None and int(emb) > 0:
            return int(emb)
    except (TypeError, ValueError):
        pass
    return configure.encoder_dim_from_spec(Path(str(enc_path)).name)


def check_model_compatibility(cfg: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate that the configured encoder + VAE actually match the configured
    diffusion model, BEFORE spending a model load on a run that sd.cpp would
    only reject with a cryptic tensor-shape error.

    Returns (ok, message). ok=True with message="" means "no detectable
    problem" -- which includes the "cannot tell" cases (unrecognised family,
    unreadable metadata): unknowns are allowed through, never blocked, so an
    oddly-named-but-valid pairing still runs. A False is only ever returned for
    a POSITIVE mismatch (a dim we know against a dim we know, or a VAE whose
    family we recognise as the wrong one).

    The rules come straight from configure.py's family matrix:
      * Z-Image-Turbo / Flux.2-klein-4B  need a 2560-dim (Qwen3-4B) encoder.
      * Flux.2-klein-9B                  needs a 4096-dim (Qwen3-8B) encoder.
      * Z-Image uses ae.safetensors; Flux.2 uses flux2_ae -- never swapped.
    """
    diff_path = cfg.get("imagegen_model_path", "")
    if not diff_path or not Path(diff_path).exists():
        return True, ""

    diff_meta = probe_model(diff_path) or {}
    diff_arch = _arch_of(diff_meta)
    fam = configure.diffuser_family(diff_path, diff_arch)
    fam_label = configure.diffuser_family_label(diff_path, diff_arch)

    # ── Encoder is a vision projector, not a text encoder ─────────────────
    # A positive, always-checked reject (independent of the diffuser family, so
    # it fires even when the diffuser is unrecognised). Catches the mmproj that
    # lives beside the VL model on disk being picked as the Encoder.
    enc_path = cfg.get("encoder_model_path", "")
    if enc_path and Path(enc_path).exists() and is_mmproj(enc_path):
        return False, ("The selected encoder is a multimodal vision projector "
                       "(mmproj), not a text encoder. Select the main model "
                       ".gguf instead — the mmproj file adds image input and is "
                       "not used by this program.")

    # ── Encoder size ↔ diffuser size ──────────────────────────────────────
    need_dim = configure.required_encoder_dim(diff_path, diff_arch)
    if enc_path and Path(enc_path).exists() and need_dim:
        have_dim = _encoder_hidden_dim(enc_path)
        if have_dim and have_dim != need_dim:
            need_lbl = "Qwen3-8B (4096-dim)" if need_dim == configure.ENCODER_DIM_QWEN3_8B \
                       else "Qwen3-4B (2560-dim)"
            have_lbl = "Qwen3-8B (4096-dim)" if have_dim == configure.ENCODER_DIM_QWEN3_8B \
                       else (f"Qwen3-4B (2560-dim)" if have_dim == configure.ENCODER_DIM_QWEN3_4B
                             else f"{have_dim}-dim")
            return False, (f"Encoder mismatch: {fam_label} needs a {need_lbl} encoder; "
                           f"the selected encoder is {have_lbl}.")

    # ── VAE family ↔ diffuser family ──────────────────────────────────────
    vae_path = cfg.get("vae_model_path", "")
    if vae_path and Path(vae_path).exists() and fam:
        vfam = configure.vae_family(Path(vae_path).name)
        if vfam and vfam != fam:
            want = "flux2_ae.safetensors" if fam == configure.DIFFUSER_FAMILY_FLUX2 \
                   else "ae.safetensors"
            have = configure.DIFFUSER_FAMILY_LABELS.get(vfam, vfam)
            return False, (f"VAE mismatch: {fam_label} needs {want}; "
                           f"the selected VAE looks like the {have} VAE.")

    return True, ""


# ---------------------------------------------------------------------------
# GGUF parser
# ---------------------------------------------------------------------------

def _probe_gguf(p: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(p, "rb") as f:
            if f.read(4) != GGUF_MAGIC:
                return None
            version = struct.unpack("<I", f.read(4))[0]
            tensor_count = struct.unpack("<Q", f.read(8))[0]
            kv_count = struct.unpack("<Q", f.read(8))[0]
            meta: Dict[str, Any] = {
                "format": "GGUF", "file": str(p), "filename": p.name,
                "version": version, "tensor_count": tensor_count,
                "kv_count": kv_count,
                "size_bytes": p.stat().st_size,
                "size_human": utilities.human_size(p.stat().st_size),
                "quantization": get_quantization_label(p.name),
                "hash_prefix": utilities.quick_hash(str(p)),
            }
            # `*.block_count` is what _resolve_gpu_layers() wants: the real
            # number of offloadable transformer blocks, straight from the file
            # rather than guessed from its name. It is written as
            # "<arch>.block_count" (qwen3.block_count), so the existing
            # last-segment key trick picks it up unchanged.
            #
            # These all sit in the header ahead of the tokenizer arrays, which
            # matters -- see the oversized-array note in _read_gguf_value():
            # the parse cannot safely continue past one of those, so anything
            # written after them is unreachable by design.
            wanted = ("architecture", "name", "description",
                      "block_count", "embedding_length", "context_length")
            # Vision-projector markers, captured if present so is_mmproj() can
            # recognise a projector by metadata as well as by name. These are
            # deliberately NOT part of the early-stop gate below: ordinary
            # encoders/diffusers/VAEs never carry them, so gating on them would
            # defeat the early stop for every file. mmproj files, in turn, lack
            # the block_count/embedding_length keys the gate needs, so they
            # never early-stop and these markers are reached and recorded.
            opportunistic = ("has_vision_encoder", "projector_type")
            gate = wanted[:1] + wanted[3:]
            for _ in range(min(kv_count, 128)):
                try:
                    kv = _read_gguf_kv(f)
                    if kv is None:
                        break
                    key, value = kv
                    short = key.split(".")[-1]
                    if short in wanted or short in opportunistic:
                        meta[short] = value
                    # Everything we care about is present: stop early rather
                    # than parse tokenizer arrays we will never look at.
                    if all(k in meta for k in gate):
                        break
                except Exception:
                    break
            return meta
    except Exception:
        return {
            "format": "GGUF (unreadable)", "file": str(p),
            "filename": p.name,
            "size_bytes": p.stat().st_size,
            "size_human": utilities.human_size(p.stat().st_size),
            "quantization": get_quantization_label(p.name),
        }


def _read_gguf_kv(f) -> Optional[Tuple[str, Any]]:
    try:
        key_len = struct.unpack("<Q", f.read(8))[0]
        if key_len > 65536:
            return None
        key = f.read(key_len).decode("utf-8", errors="replace")
        type_idx = struct.unpack("<I", f.read(4))[0]
        value = _read_gguf_value(f, type_idx)
        return key, value
    except Exception:
        return None


def _read_gguf_value(f, type_idx: int) -> Any:
    gt = GGUF_TYPES.get(type_idx, "UNKNOWN")
    if gt == "UINT8":
        return struct.unpack("<B", f.read(1))[0]
    if gt == "INT8":
        return struct.unpack("<b", f.read(1))[0]
    if gt == "UINT16":
        return struct.unpack("<H", f.read(2))[0]
    if gt == "INT16":
        return struct.unpack("<h", f.read(2))[0]
    if gt == "UINT32":
        return struct.unpack("<I", f.read(4))[0]
    if gt == "INT32":
        return struct.unpack("<i", f.read(4))[0]
    if gt == "FLOAT32":
        return struct.unpack("<f", f.read(4))[0]
    if gt == "UINT64":
        return struct.unpack("<Q", f.read(8))[0]
    if gt == "INT64":
        return struct.unpack("<q", f.read(8))[0]
    if gt == "FLOAT64":
        return struct.unpack("<d", f.read(8))[0]
    if gt == "BOOL":
        return struct.unpack("<B", f.read(1))[0] != 0
    if gt == "STRING":
        slen = struct.unpack("<Q", f.read(8))[0]
        if slen > 1_000_000:
            return ""
        return f.read(slen).decode("utf-8", errors="replace")
    if gt == "ARRAY":
        arr_type = struct.unpack("<I", f.read(4))[0]
        arr_len = struct.unpack("<Q", f.read(8))[0]
        if arr_len > 10000:
            # Bailing out here used to `return []` WITHOUT consuming the
            # array's bytes, leaving the file cursor parked in the middle of
            # it. Every KV read after that point was misaligned garbage, and
            # since the caller records whatever it decodes, a junk key could
            # land in meta. In practice the damage was capped only by luck:
            # the one oversized array in a Qwen3 encoder is
            # tokenizer.ggml.tokens (151936 entries), and everything this
            # program reads is written before it.
            #
            # Raising instead makes the desync explicit: _read_gguf_kv()
            # catches it, returns None, and _probe_gguf() stops parsing. We
            # keep the metadata gathered so far and read nothing we cannot
            # trust. Skipping the array properly is not worth it -- a
            # variable-width STRING array can only be skipped by walking all
            # 151936 length prefixes, and there is nothing after it we want.
            raise _GGUFArraySkipped(
                f"array of {arr_len} exceeds the 10000 parse cap")
        vals = []
        for _ in range(arr_len):
            vals.append(_read_gguf_value(f, arr_type))
        return vals
    return None


# ---------------------------------------------------------------------------
# Safetensors parser
# ---------------------------------------------------------------------------

def _probe_safetensors(p: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(p, "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            if header_len > 100_000_000:
                return None
            header = json.loads(f.read(header_len))
        dtype_counts: Dict[str, int] = {}
        total_params = 0
        for info in header.values():
            if isinstance(info, dict) and "dtype" in info:
                dt = info["dtype"]
                dtype_counts[dt] = dtype_counts.get(dt, 0) + 1
                for x in info.get("shape", []):
                    total_params = total_params * x if total_params else x
        return {
            "format": "Safetensors", "file": str(p), "filename": p.name,
            "size_bytes": p.stat().st_size,
            "size_human": utilities.human_size(p.stat().st_size),
            "tensor_count": len(header), "dtypes": dtype_counts,
            "total_params": total_params,
            "hash_prefix": utilities.quick_hash(str(p)),
        }
    except Exception:
        return {
            "format": "Safetensors (unreadable)", "file": str(p),
            "filename": p.name,
            "size_bytes": p.stat().st_size,
            "size_human": utilities.human_size(p.stat().st_size),
        }


# ---------------------------------------------------------------------------
# Prompt enhancement
# ---------------------------------------------------------------------------

# Used to resolve the -1 sentinel ("all layers") into a concrete count
# before retry dampening arithmetic. Sourced from configure.py to keep
# model constraints centralized. Both supported encoder sizes (Qwen3-4B and
# Qwen3-8B) have 36 blocks; see configure.ENCODER_FAMILIES.
_ENCODER_LAYERS_FALLBACK: int = configure.ENCODER_MAX_LAYERS

# OOM fingerprints that appear in llama-cli stdout/stderr output.
_OOM_MARKERS: Tuple[str, ...] = (
    "out of memory",
    "erroroutofdevicememory",
    "alloc.*failed",
    "failed to allocate",
    "ggml_vulkan: device memory allocation",
    "cudamalloc failed",
)


def _is_oom_output(text: str) -> bool:
    """Return True if the combined process output contains an OOM indicator."""
    lower = text.lower().replace(" ", "")
    for marker in _OOM_MARKERS:
        # markers without regex wildcards: simple substring match
        if "*" not in marker:
            if marker.replace(" ", "") in lower:
                return True
        else:
            if re.search(marker, text, re.IGNORECASE):
                return True
    return False


def _resolve_gpu_layers(raw: int, model_path: str) -> int:
    """
    Resolve the -1 sentinel to a concrete layer count.

    -1 means "offload all layers". Resolving it to a REAL count matters more
    than it looks: enhance_prompt()'s OOM ladder subtracts 1/2/4 from the
    previous attempt, so if this returns an inflated sentinel every rung of
    the ladder is still above the true block count and every retry re-runs the
    identical failing command. That is exactly what the old implementation did
    for anything that was not name-matched as Qwen3-4B -- including both of
    the Qwen3-8B encoders, whose only published quant (Q8_0, 8.71GB) is the
    one most likely to OOM on an 8GB card in the first place.

    Three oracles, in order:
      1. GGUF `*.block_count` metadata, cached by _probe_gguf(). Authoritative
         and independent of filename, matching classify_model()'s policy of
         trusting metadata over naming.
      2. configure.ENCODER_FAMILIES, matched against the filename and then the
         containing folder names -- for files whose metadata could not be read.
      3. configure.ENCODER_MAX_LAYERS (36), correct for every Qwen3 encoder
         this program supports, both 4B and 8B.

    configure.ENCODER_UNKNOWN_LAYERS (99) is deliberately NOT the fallback:
    overshooting is only "safe" for a single attempt, and it is precisely what
    broke the retries.
    """
    if raw >= 0:
        return raw

    meta = probe_model(model_path) or {}

    # 1. Metadata: qwen3.block_count / llama.block_count / etc.
    blocks = meta.get("block_count")
    try:
        if blocks is not None and int(blocks) > 0:
            return int(blocks)
    except (TypeError, ValueError):
        pass

    # 2. Filename, then up to two containing folders (same haystack policy as
    #    classify_model(); users keep the repo's folder layout on download).
    for hay in _name_haystacks(meta) or [Path(str(model_path)).name.lower()]:
        spec = configure.get_encoder_family_spec(hay)
        if spec:
            return int(spec["layers"])

    # 3. Every supported encoder is 36 blocks.
    return _ENCODER_LAYERS_FALLBACK


def enhance_prompt(prompt: str, cfg: Dict[str, Any],
                   progress_callback: Optional[Callable] = None,
                   _oom_status_callback: Optional[Callable[[str], None]] = None,
                   ) -> str:
    """
    Run llama-cli to expand the user prompt into a rich image-gen prompt.

    If the encoder backend is Vulkan and the process exits with an OOM
    indication, up to 3 retries are attempted with progressively fewer GPU
    layers (reductions of 1, then 2, then 4 from the previous attempt).
    On exhausting all retries the original prompt is returned and
    _oom_status_callback (if supplied) is called with a user-facing message.

    Retry schedule (example, Qwen3-4B at -1 → 36 layers):
        Attempt 1 : ngl = 36   (or configured value)
        Attempt 2 : ngl = 35   (-1 from previous)
        Attempt 3 : ngl = 33   (-2 from previous)
        Attempt 4 : ngl = 29   (-4 from previous)
    """
    model_path = cfg.get("encoder_model_path", "")
    if not model_path or not Path(model_path).exists():
        return prompt
    # Never load a vision projector as a text encoder. If the configured
    # encoder is an mmproj, skip enhancement entirely rather than feed it to
    # llama-completion (the run itself is separately blocked by
    # check_model_compatibility). Returning the prompt unchanged keeps a
    # mis-selection from silently loading a projector here.
    if is_mmproj(model_path):
        return prompt
    cli = find_llama_completion()
    if not cli:
        return prompt

    template = cfg.get("prompt_template", "{prompt}")
    system_msg = (
        "You are an expert image prompt engineer. Convert the user's "
        "request into a highly detailed, vivid image generation prompt. "
        "Be descriptive about style, lighting, composition, colors, and "
        "atmosphere. Respond ONLY with the prompt text."
    )
    full_prompt = template.replace(
        "{prompt}", f"{system_msg}\n\nUser request: {prompt}")

    # ------------------------------------------------------------------
    # argv notes -- each of these was independently fatal before:
    #
    # -no-cnv          llama.cpp auto-enables conversation mode when the model
    #                  ships a chat template ("default: auto enabled if chat
    #                  template is available"). Qwen3 does. Without this the
    #                  process goes interactive and waits on stdin until our
    #                  120s timeout kills it.
    # --flash-attn on  -fa takes a VALUE ([on|off|auto]). The parser consumes
    #                  the next argv unconditionally, so a bare --flash-attn
    #                  swallowed the following "-ngl" and threw
    #                  "unknown value for --flash-attn: '-ngl'".
    # no --log-disable it pauses common_log_main(), which is exactly where the
    #                  generated tokens are written (LOG("%s", token_str)), so
    #                  it silenced the output we are here to read.
    # --no-display-prompt
    #                  stops the prompt being echoed back into the text we parse.
    # -dev VulkanN     the real device selector. GGML_VULKAN_DEVICE does not
    #                  exist in ggml (the only device env var is
    #                  GGML_VK_VISIBLE_DEVICES, a CUDA_VISIBLE_DEVICES-style
    #                  filter that RENUMBERS devices, which is a different and
    #                  more confusing thing). Setting a nonexistent variable
    #                  silently left the encoder on ggml's default device.
    # ------------------------------------------------------------------
    base_args = [
        str(cli), "-m", model_path, "-p", full_prompt,
        "-c", str(cfg.get("encoder_ctx_size", 4096)),
        "-t", str(cfg.get("encoder_threads", configure.get_default_threads())),
        "-b", str(cfg.get("encoder_batch_size", 512)),
        "-n", "256", "--temp", "0.7",
        "-no-cnv",
        "--no-display-prompt",
        # Encoder flash attention: pass "auto" and let llama.cpp choose. Unlike
        # sd.cpp's Vulkan diffusion FA (which garbles on a no-fp16 GPU), llama.cpp
        # FA on a GPU without coopmat2/fp16 simply falls back to CPU for the
        # attention math — correct results, only slower — so there is nothing to
        # gate on fp16 here. "auto" also picks the fast path where the hardware
        # supports it, with no user toggle and no head-banging.
        "--flash-attn", "auto",
    ]

    backend = cfg.get("backend_encoder", "CPU")
    use_vulkan = "Vulkan" in backend
    env = os.environ.copy()

    if use_vulkan:
        dev = int(cfg.get("encoder_vulkan_device", -1))
        if dev >= 0:
            base_args.extend(["-dev", f"Vulkan{dev}"])
        raw_ngl = int(cfg.get("encoder_gpu_layers", -1))
        current_ngl = _resolve_gpu_layers(raw_ngl, model_path)
    else:
        base_args.extend(["-dev", "none"])
        current_ngl = 0

    # Retry dampening: reductions applied cumulatively each attempt.
    # Attempt 0 (first run) uses current_ngl unchanged.
    # Attempt 1 : subtract 1  (total -1)
    # Attempt 2 : subtract 2  (total -3)
    # Attempt 3 : subtract 4  (total -7)
    _retry_reductions: Tuple[int, ...] = (0, 1, 2, 4)
    max_attempts = len(_retry_reductions)  # 4

    phase_t0 = time.time()
    if progress_callback:
        progress_callback("Enhancing prompt with LLM...", 0.1,
                          {"phase": "encoding", "phase_start": phase_t0})

    for attempt in range(max_attempts):
        # Cumulative layer reduction across retries: 0, -1, -3, -7
        cumulative_cut = sum(_retry_reductions[1:attempt + 1])
        ngl = max(0, current_ngl - cumulative_cut)

        args = base_args + ["-ngl", str(ngl)]

        if attempt > 0:
            print(
                f"[enhance_prompt] OOM retry {attempt}/{max_attempts - 1}: "
                f"ngl={ngl} (was {ngl + sum(_retry_reductions[attempt:attempt + 1])})",
                flush=True,
            )
            if progress_callback:
                progress_callback(
                    f"Enhancing prompt (OOM retry {attempt}, ngl={ngl})...", 0.1,
                    {"phase": "encoding", "phase_start": phase_t0})

        try:
            result = subprocess.run(
                args, capture_output=True, text=True,
                timeout=120, encoding="utf-8", errors="replace", env=env,
            )
            combined_output = result.stdout + result.stderr

            # Encoder Model Debug (Preferences page, off by default): dump what
            # the encoder actually produced to the background console, so garbled
            # or empty conditioning can be inspected. Prints the raw stdout — the
            # model's own text output — between fixed markers.
            try:
                if configure.load_preferences().get("encoder_model_debug", False):
                    print("\n*** ENCODER DEBUG BEGIN ***\n", flush=True)
                    print(result.stdout.rstrip("\n"), flush=True)
                    print("\n*** ENCODER DEBUG END ***\n", flush=True)
            except Exception:
                pass

            # Check for OOM before inspecting the text output
            if result.returncode != 0 and _is_oom_output(combined_output):
                print(
                    f"[enhance_prompt] OOM detected (attempt {attempt + 1}, "
                    f"ngl={ngl}, exit={result.returncode})",
                    flush=True,
                )
                if attempt < max_attempts - 1:
                    continue  # retry with fewer layers
                # All retries exhausted
                oom_msg = (
                    "Out of memory — reduce encoder GPU layers in the "
                    "Configuration page."
                )
                print(f"[enhance_prompt] {oom_msg}", flush=True)
                if _oom_status_callback:
                    _oom_status_callback(oom_msg)
                return prompt

            output = result.stdout.strip()
            if output:
                for marker in ("<|im_start|>assistant", "<|start_header_id|>assistant"):
                    if marker in output:
                        output = output.split(marker)[-1]
                        break
                for marker in ("<|im_end|>", "<|im_start|>", "<|eot_id|>"):
                    output = output.replace(marker, "")
                output = output.strip()
            if output and len(output) > 10:
                encoder_elapsed = time.time() - phase_t0
                configure.update_timing_stat("encoder_seconds", round(encoder_elapsed, 3))
                if progress_callback:
                    progress_callback(
                        "Prompt enhanced.", 0.2,
                        {"phase": "encoding", "phase_done": True,
                         "phase_elapsed": encoder_elapsed})
                return output[:2000]

            # Non-OOM failure or empty output — no point retrying
            break

        except subprocess.TimeoutExpired:
            break
        except Exception:
            break

    return prompt


# ---------------------------------------------------------------------------
# sd.cpp step-progress parsing
# ---------------------------------------------------------------------------
# stable-diffusion.cpp prints per-step progress lines while sampling, in
# forms such as:
#   "  |==========>      | 2/4 - 1.23s/it"
#   "[sample] 3/4"
#   "sampling, step 4/4"
# This regex tolerates all of the above: it just looks for "N/M" where M
# matches --steps. It deliberately does NOT try to parse the progress-bar
# glyphs or the "s/it" suffix — those vary across builds and aren't needed.
_STEP_LINE_RE = re.compile(r"(\d+)\s*/\s*(\d+)")


def _parse_step_line(line: str, total_steps: int) -> Optional[Tuple[int, int]]:
    """Return (current_step, total_steps) if line looks like step progress
    for THIS run (denominator matches total_steps), else None."""
    for m in _STEP_LINE_RE.finditer(line):
        cur, tot = int(m.group(1)), int(m.group(2))
        if tot == total_steps and 0 <= cur <= tot:
            return cur, tot
    return None


# ---------------------------------------------------------------------------
# Image generation
# ---------------------------------------------------------------------------

def _flux2_needs_offload(diff_path: str, width: int, height: int,
                         vk_dev: int, is_edit: bool) -> bool:
    """Decide whether a Flux.2 run must stream the DiT from CPU RAM
    (--offload-to-cpu) to fit, automatically, so we only pay its heavy speed
    penalty when the DiT plus its compute buffer would actually overflow the
    target GPU's VRAM.

    With the encoder/VAE already forced to CPU for flux.2, only the DiT and its
    compute buffer sit in VRAM. The compute buffer scales with pixel count
    (~3 GB at 1024x1024, measured), and edit mode enlarges it (extra reference
    tokens). If the estimate fits under a VRAM headroom margin, keep the DiT
    resident (GPU speed); otherwise offload. Unknown VRAM -> don't force the
    slow path.
    """
    try:
        dit_mb = Path(diff_path).stat().st_size / (1024.0 * 1024.0)
    except OSError:
        return False
    vram_mb = 0
    for d in configure.get_vulkan_info().get("devices", []):
        if d.get("index") == vk_dev:
            vram_mb = d.get("vram_total_mb", 0) or 0
            break
    if not vram_mb:
        return False
    buf_mb = 3000.0 * (max(width, 1) * max(height, 1)) / (1024.0 * 1024.0)
    if is_edit:
        buf_mb *= 1.5
    return (dit_mb + buf_mb) > vram_mb * 0.92


def generate_image(prompt: str, cfg: Dict[str, Any],
                   progress_callback: Optional[Callable] = None
                   ) -> Dict[str, Any]:
    t0 = time.time()
    result: Dict[str, Any] = {
        "success": False, "output_path": "", "message": "",
        "seed_used": -1, "elapsed_seconds": 0.0,
    }

    diff_path = cfg.get("imagegen_model_path", "")
    if not diff_path or not Path(diff_path).exists():
        result["message"] = "Image generation model not found. Configure in Configuration tab."
        return result

    vae_path = cfg.get("vae_model_path", "")
    if not vae_path or not Path(vae_path).exists():
        result["message"] = "VAE model (ae.safetensors) not found. Configure in Configuration tab."
        return result

    sd_cli = find_sd_cpp()
    if not sd_cli:
        result["message"] = ("stable-diffusion.cpp not found. "
                             "Run installer or build from Configuration tab.")
        return result

    # Refuse a run whose encoder/VAE cannot match the diffuser, with a plain
    # message, rather than letting sd.cpp die on a tensor-shape error the user
    # cannot read. Unknown pairings are allowed through (see the function).
    compat_ok, compat_msg = check_model_compatibility(cfg)
    if not compat_ok:
        result["message"] = compat_msg
        return result

    # Which family are we driving? Everything flux.2-specific below hangs off
    # this; when it is Z-Image the code path is byte-for-byte what it always
    # was. Reference images (flux.2 image-to-image / editing, sd.cpp -r) come
    # in as a list of paths on the cfg; empty/absent means plain text-to-image.
    diff_meta = probe_model(diff_path) or {}
    family = configure.diffuser_family(diff_path, _arch_of(diff_meta))
    is_flux2 = family == configure.DIFFUSER_FAMILY_FLUX2
    ref_images = [p for p in (cfg.get("ref_images") or [])
                  if p and Path(str(p)).exists()]
    is_flux2_edit = is_flux2 and bool(ref_images)

    # Refuse formats sd.cpp's stb_image loader cannot decode (WEBP/AVIF/HEIC/
    # TIFF/JP2), with a plain message, rather than letting it fail mid-encode.
    # PNG/JPG/BMP and friends pass straight through -- no converter needed.
    bad_fmt = [Path(str(r)).name for r in ref_images
               if Path(str(r)).suffix.lower() not in configure.SUPPORTED_REF_IMAGE_EXTS]
    if bad_fmt:
        result["message"] = (
            "Unsupported input-image format: " + ", ".join(bad_fmt) +
            ". Use PNG, JPG or BMP — WEBP, AVIF, HEIC and TIFF cannot be read.")
        return result

    # Enhance prompt — wire OOM status so the message surfaces in the UI.
    # Skipped for flux.2 EDIT runs: the reference image already reaches the
    # conditioner via -r, and a flowery expansion of a literal edit instruction
    # ("change the background to a park") fights it rather than helping — and it
    # would cost a whole extra model load per generation for no gain.
    enhanced = prompt
    _oom_msg: List[str] = []  # populated by callback if OOM exhausted all retries
    if (not is_flux2_edit
            and cfg.get("encoder_model_path")
            and Path(cfg["encoder_model_path"]).exists()):
        try:
            enhanced = enhance_prompt(
                prompt, cfg, progress_callback,
                _oom_status_callback=lambda msg: _oom_msg.append(msg),
            )
        except Exception:
            enhanced = prompt
    # Surface OOM message to caller if enhance_prompt reported it.
    if _oom_msg:
        result["message"] = _oom_msg[-1]
        return result

    # Seed
    seed = cfg.get("imagegen_seed", -1)
    if seed is None or int(seed) < 0:
        seed = random.randint(1, 2147483647)
    else:
        seed = int(seed)

    # Output path
    output_dir = configure.get_output_dir()
    fmt = cfg.get("output_format", "png")
    output_name = f"img_{int(time.time())}_s{seed}.{fmt}"
    output_path = output_dir / output_name

    # Build sd command.
    # --diffusion-model for standalone gguf diffusion weights (not -m).
    # Z-Image-Turbo is S3-DiT architecture: it requires --llm pointing at
    # the Qwen3 encoder gguf. Without it sd.cpp cannot find the conditioner
    # tensors (text_encoders.llm.*) and fails at model metadata validation.
    enc_path = cfg.get("encoder_model_path", "")
    args = [
        str(sd_cli),
        "--diffusion-model", str(diff_path),
        "--vae", str(vae_path),
        "-p", enhanced, "-o", str(output_path),
        "-H", str(int(cfg.get("imagegen_height", 512))),
        "-W", str(int(cfg.get("imagegen_width", 512))),
        "--steps", str(int(cfg.get("imagegen_steps", 4))),
        "--cfg-scale", str(float(cfg.get("imagegen_cfg_scale", 1.0))),
        "--seed", str(seed),
        "--sampling-method",
        configure.SAMPLER_MAP.get(cfg.get("imagegen_sampling", "euler_a"),
                                  "euler_a"),
        "-t", str(int(cfg.get("imagegen_threads", configure.get_default_threads()))),
    ]

    # Pass the Qwen3 encoder as the LLM text encoder for Z-Image-Turbo.
    # This is mandatory — the diffusion gguf has no bundled text encoder.
    # Last-resort guard: never hand sd.cpp a vision projector as --llm, and
    # never emit --mmproj at all, so a projector cannot load by any route even
    # if one slipped past the earlier checks.
    if enc_path and is_mmproj(enc_path):
        raise ValueError(
            "Refusing to run: the configured encoder is an mmproj vision "
            "projector, not a text encoder. Reselect the main model .gguf in "
            "Configuration.")
    if enc_path and Path(enc_path).exists():
        args.extend(["--llm", enc_path])

    neg = cfg.get("negative_prompt", "")
    if neg:
        args.extend(["-n", neg])

    # ------------------------------------------------------------------
    # Device / component placement.
    #
    # sd.cpp has TWO backend assignments and you need both:
    #   --backend         where graphs EXECUTE
    #   --params-backend  where model WEIGHTS are ALLOCATED
    # VRAM pressure is weights, so --params-backend is the one that decides
    # whether Z-Image fits on an 8GB card. It was never passed before.
    #
    # The old code passed `--backend vulkanN` together with --clip-on-cpu /
    # --vae-on-cpu. Upstream docs/backend.md is explicit that those legacy
    # flags "affect runtime backend assignment ONLY when --backend is not set".
    # So they were discarded, everything landed on the GPU, and Vulkan runs
    # OOM'd or thrashed -- while CPU-only runs (which set no --backend, so the
    # legacy flags DID apply) kept working. That asymmetry is exactly the
    # "Vulkan broken, CPU fine" behaviour.
    #
    # Module names come from docs/backend.md: `diffusion` (the DiT), `te` (text
    # encoders / conditioners, which is where the Qwen3 --llm goes), `vae`.
    # ------------------------------------------------------------------
    placement_label = cfg.get("imagegen_placement", configure.DIFFUSER_PLACEMENT_FULL_GPU)
    placement = configure.parse_diffuser_placement(placement_label)

    env = os.environ.copy()
    vk_dev = int(cfg.get("imagegen_vulkan_device", -1))

    if placement["use_vulkan_backend"] and vk_dev >= 0:
        dev = f"vulkan{vk_dev}"
        if is_flux2:
            # Flux.2's Qwen3 text encoder (4-8B, several GB) is far too large to
            # sit on the GPU beside the DiT on consumer VRAM, and conditioning
            # is a one-shot pass, so it ALWAYS goes to CPU for flux.2 — the DiT
            # stays resident on the GPU. This automatic encoder/DiT split (not
            # the placement dropdown) is what lets the 9B fit AND run at GPU
            # speed instead of streaming every step. (Answers "automate the
            # encoder like we did flash attention".)
            assign = f"diffusion={dev},te=cpu,vae=cpu"
        elif placement["split_to_cpu"]:
            # Diffusion on the GPU; Qwen3 conditioner (~3.6GB) and VAE stay in
            # system RAM, both for execution and for weight allocation.
            assign = f"diffusion={dev},te=cpu,vae=cpu"
        else:
            assign = dev
        args.extend(["--backend", assign, "--params-backend", assign])
        sdk = configure.get_vulkan_info().get("sdk", "") or os.environ.get("VULKAN_SDK", "")
        if sdk:
            env["PATH"] = str(Path(sdk) / "Bin") + ";" + env.get("PATH", "")
    else:
        # Explicit, rather than omitting --backend: says what we mean, and
        # stops ggml's default "prefer GPU" preference picking a card.
        args.extend(["--backend", "cpu", "--params-backend", "cpu"])

    batch = int(cfg.get("imagegen_batch_count", 1))
    if batch > 1:
        args.extend(["-b", str(batch)])

    # --clip-skip is only valid for checkpoints that HAVE a CLIP text encoder
    # (SD1.x / SD2.x / SDXL). Z-Image-Turbo and its finetunes condition through
    # the Qwen3 LLM passed via --llm and have no CLIP, so the flag is omitted.
    #
    # The test is now a positive one (see inference.is_sd_classic_model and
    # configure.SD_CLASSIC_*). The previous inverse test — SD-classic unless
    # the name contained "flux/z_image/sd3/wan/ltx" — silently failed for every
    # model this program targets whose name does not spell z_image with an
    # underscore: "zImageTurboAnime_v10", "smoothmixUltimate_zimageTurboV10",
    # "eventHorizon_zitV10", "perfeczion_10BF16" and both darkBeast files all
    # got --clip-skip 2 appended to a model that has no CLIP.
    is_sd_classic = is_sd_classic_model(str(diff_path))
    clip_skip = int(cfg.get("imagegen_clip_skip", 2))
    if clip_skip > 1 and is_sd_classic:
        args.extend(["--clip-skip", str(clip_skip)])

    # ------------------------------------------------------------------
    # Flux.2-Klein specifics. Z-Image needs none of this, so it is all gated
    # on the family and the Z-Image command line is unchanged.
    #
    #   --offload-to-cpu  stages the DiT through system RAM. HEAVY speed penalty
    #                     (streams weights every step), so applied ONLY when the
    #                     DiT + compute buffer would overflow VRAM — decided
    #                     automatically by _flux2_needs_offload(). The encoder is
    #                     already forced to CPU (see placement above), so the DiT
    #                     stays resident on the GPU whenever it fits.
    #   --diffusion-fa    flash attention on the DiT; auto-gated on GPU fp16.
    #   -r <ref.png>      one per reference image, for image-to-image / editing.
    #                     Repeatable (flux.2 is a multi-reference editor).
    #   sampler           left as the user set it (default euler_a); euler_a
    #                     empirically beats plain euler on this build/hardware.
    # See stable-diffusion.cpp docs/flux2.md.
    # ------------------------------------------------------------------
    if is_flux2:
        # --offload-to-cpu only when the DiT + its compute buffer would not fit
        # the GPU (auto, resolution-aware). Forcing it unconditionally streamed
        # ~5.6GB every step for the 9B and left the GPU idle at ~0% — glacial.
        # With the encoder already on CPU, the DiT stays resident on the GPU
        # whenever it fits, at full GPU speed.
        _w = int(cfg.get("imagegen_width", 512))
        _h = int(cfg.get("imagegen_height", 512))
        if _flux2_needs_offload(str(diff_path), _w, _h, vk_dev, is_flux2_edit):
            args.append("--offload-to-cpu")
        args.append("--vae-tiling")
        # Flash attention keeps the compute buffer small and the flux.2 docs
        # recommend it, BUT the Vulkan flash-attention path needs fp16. Some
        # older GPUs (Polaris / RX 470 report "fp16: 0") do not expose it, and
        # there --diffusion-fa can silently produce a garbled image rather than
        # error. The decision is now automatic per the SELECTED diffuser GPU's
        # fp16 capability, detected by the installer and stored in constants.ini
        # (configure.flux2_flash_attn_for / device_supports_fp16); no user
        # toggle. Turning it off raises VRAM use, fine on the 4B but tight on 9B.
        if configure.flux2_flash_attn_for(cfg):
            args.append("--diffusion-fa")
        for ref in ref_images:
            args.extend(["-r", str(ref)])
        # Sampler is left as the user set it (default euler_a). Although Flux.2
        # is a rectified-flow model, in practice on this sd.cpp build euler_a
        # gives clean results and plain euler comes out semi-garbled on the
        # target hardware, so we do NOT override the user's choice here. sd.cpp
        # still applies the Flux2 scheduler internally regardless of sampler.

    total_steps = int(cfg.get("imagegen_steps", 4))
    diffusion_t0 = time.time()
    if progress_callback:
        progress_callback("Generating image...", 0.3,
                          {"phase": "diffusion", "phase_start": diffusion_t0,
                           "step": 0, "total_steps": total_steps})

    try:
        # Echo the resolved command so the ACTUAL sampler / flags in effect are
        # visible on the console — no guessing whether the dropdown took effect.
        try:
            _sm = args[args.index("--sampling-method") + 1] if "--sampling-method" in args else "?"
        except (ValueError, IndexError):
            _sm = "?"
        print(f"[generate] sd-cli: sampler={_sm}  "
              f"steps={cfg.get('imagegen_steps')}  cfg={cfg.get('imagegen_cfg_scale')}  "
              f"flux2={is_flux2}  fa={'--diffusion-fa' in args}  "
              f"offload={'--offload-to-cpu' in args}")
        print("[generate] full command: " + " ".join(str(a) for a in args))
        process = subprocess.Popen(
            args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", env=env,
        )
        output_lines: List[str] = []
        last_step = 0
        for line in process.stdout:
            output_lines.append(line)
            if progress_callback and ("step" in line.lower() or "%" in line or "/" in line):
                step_info = _parse_step_line(line, total_steps)
                info: Dict[str, Any] = {"phase": "diffusion", "phase_start": diffusion_t0}
                if step_info:
                    last_step = step_info[0]
                    info["step"] = step_info[0]
                    info["total_steps"] = step_info[1]
                try:
                    progress_callback(line.strip()[:100], 0.3, info)
                except Exception:
                    pass

        process.wait(timeout=600)
        elapsed = time.time() - t0
        diffusion_elapsed = time.time() - diffusion_t0

        # sd.cpp saves batch outputs with a _0/_1/... index suffix:
        #   img_TIMESTAMP_sSEED_0.jpg, img_TIMESTAMP_sSEED_1.jpg, …
        # For batch=1 it still appends _0, so we check for the _0 file
        # first; the bare path is kept as a fallback for any sd.cpp build
        # that omits the suffix on single-image runs.
        batch_count = int(cfg.get("imagegen_batch_count", 1))
        first_batch_path = output_path.with_name(
            output_path.stem + "_0" + output_path.suffix
        )
        actual_output_path = (
            first_batch_path if first_batch_path.exists() else output_path
        )

        if process.returncode == 0 and (actual_output_path.exists() or output_path.exists()):
            # Record diffusion timing for next-generation ETA, using
            # whichever step count we actually observed (falls back to the
            # configured total if no step lines matched the parser).
            steps_for_avg = last_step if last_step > 0 else total_steps
            configure.update_timing_stat("diffusion_total_seconds", round(diffusion_elapsed, 3))
            configure.update_timing_stat("diffusion_steps", steps_for_avg)
            if steps_for_avg > 0:
                configure.update_timing_stat(
                    "diffusion_per_step_seconds",
                    round(diffusion_elapsed / steps_for_avg, 3))
            # Report the first saved file as output_path for the preview;
            # the gallery rescan will pick up all batch images automatically.
            saved_path = actual_output_path if actual_output_path.exists() else output_path
            batch_label = (f"{batch_count} images" if batch_count > 1
                           else output_name)
            result.update(
                success=True, output_path=str(saved_path),
                message=f"Saved {batch_label} ({int(round(elapsed))}s)",
                seed_used=seed,
                elapsed_seconds=round(elapsed, 2),
            )
            if progress_callback:
                progress_callback(f"Done: {batch_label}", 1.0,
                                  {"phase": "done", "phase_elapsed": diffusion_elapsed})
        else:
            # Full subprocess output -> terminal (Windows console) only.
            full_output = "".join(output_lines)
            print(
                f"\n--- sd.cpp output (exit {process.returncode}) ---\n"
                f"{full_output}"
                f"--- end sd.cpp output ---\n", flush=True)
            result["message"] = (
                f"Generation failed (exit {process.returncode}). "
                f"See terminal for details."
            )
            result["elapsed_seconds"] = round(time.time() - t0, 2)

    except subprocess.TimeoutExpired:
        process.kill()
        print("\n--- sd.cpp timed out after 600 s ---\n", flush=True)
        result["message"] = "Generation timed out (600 s). See terminal."
        result["elapsed_seconds"] = round(time.time() - t0, 2)
    except Exception as e:
        print(f"\n--- generate_image exception ---\n{traceback.format_exc()}"
              f"--- end traceback ---\n", flush=True)
        result["message"] = f"Error: {e}. See terminal for traceback."
        result["elapsed_seconds"] = round(time.time() - t0, 2)

    return result


# ---------------------------------------------------------------------------
# Model lifecycle
# ---------------------------------------------------------------------------

def unload_models() -> None:
    """
    Unload / release models.
    The current backends (llama-cli, sd-cli) are launched as one-shot
    subprocesses, so there is no persistent process to kill.  This function
    resets APP_STATE so that a subsequent generation re-validates paths and
    starts fresh.  If persistent model-server processes are added later,
    terminate them here.
    """
    configure.APP_STATE["last_image_path"]  = ""
    configure.APP_STATE["models_unloaded"]  = True