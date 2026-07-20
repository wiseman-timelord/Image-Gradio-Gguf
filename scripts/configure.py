"""
configure.py - Configuration, global variables, constants, maps, and lists.
All shared constants and config file I/O live here.
Reads hardware info from data/constants.ini (written by installer.py).
"""
from __future__ import annotations
import configparser
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Encoder model constraints (Qwen3 text-encoder family)
# Sourced from GGUF metadata to ensure UI sliders/dropdowns and logic
# respect the actual architecture limits.
# ---------------------------------------------------------------------------
# TWO sizes are supported, and the difference between them is smaller than it
# looks -- which is exactly why it bit us. Both Qwen3-4B and Qwen3-8B declare
# num_hidden_layers = 36 and max_position_embeddings = 40960 and vocab 151936;
# they differ only in hidden_size (2560 vs 4096) and intermediate_size. So
# ENCODER_MAX_LAYERS, ENCODER_MAX_CONTEXT, CTX_SIZE_CHOICES and
# GPU_LAYER_CHOICES are all correct for BOTH sizes as-is, and adding the 8B
# encoders required no change to any of them.
#
# What it DID require: inference._resolve_gpu_layers() used to name-test for
# "4b" and fall back to the 99 sentinel for everything else. 99 is fine as
# "offload all layers", but the OOM retry ladder subtracts 1, then 2, then 4
# from the PREVIOUS attempt -- 99 -> 98 -> 96 -> 92 are all still >= 36, i.e.
# still "every layer", so all three retries were identical to the attempt that
# just OOM'd. The ladder silently did nothing for any non-4B encoder. Hence
# ENCODER_FAMILIES below and the resolver rewrite.
#
#   Qwen3-4B   hidden 2560, 36 layers, 40960 ctx
#     Qwen3-4b-Z-Image-Turbo-AbliteratedV1.Q#.gguf
#         mradermacher static quants of BennyDaBall's abliteration.
#         Q2_K 1.67GB / Q4_K_M 2.5GB / Q6_K 3.31GB / Q8_0 4.28GB / F16 8.05GB,
#         plus IQ4_XS. NOTE the separator: mradermacher writes the quant after
#         a DOT ("...AbliteratedV1.Q4_K_M.gguf"), not the hyphen every other
#         file in this program uses. get_quantization_label() already handles
#         it -- its match is separator-boundaried, not hyphen-specific.
#     Qwen3-4b-Uncensored-Z-Image-Engineer-V4-Q8_0.gguf   (LuffyTheFox, 4.28GB)
#
#   Qwen3-8B   hidden 4096, 36 layers, 40960 ctx
#     Qwen3-8b-erotic-heretic-Q8_0.gguf                  (LuffyTheFox, 8.71GB)
#     Qwen3-8B-Gemini-2.5-Flash-Uncensored-Q8_0.gguf     (LuffyTheFox, 8.71GB)
#         Both are Heretic-abliterated Qwen3-8B derivatives republished as
#         text encoders for Z-Image-Turbo and FLUX Klein. "heretic" is the
#         abliteration TOOL (p-e-w/heretic), not a content descriptor.
#
# VRAM, the practical part: both 8B files exist upstream ONLY as Q8_0, 8.71GB.
# That does not fit the 8GB RX 470 even alone, so a Vulkan encoder run with
# either of them MUST use a partial offload (GPU Layers well below 36) or the
# CPU backend -- which is also the configuration this program already prefers
# when the diffuser is on the GPU. The 4B files (2.5GB at Q4_K_M, 4.28GB at
# Q8_0) are the ones that fit comfortably beside a diffusion model.
# ---------------------------------------------------------------------------
ENCODER_MAX_LAYERS = 36        # num_hidden_layers: 36 for Qwen3-4B AND 8B
ENCODER_MAX_CONTEXT = 40960    # max_position_embeddings: same for both
ENCODER_VOCAB_SIZE = 151936    # same for both

# Per-size facts, keyed by family id. `pattern` is re.search()ed against the
# lowercased filename (and then folder names) by encoder_family(). Keep the
# patterns mutually exclusive: first match wins, dict order decides.
#
# embedding_length is recorded for reporting/diagnostics only -- nothing in
# the program conditions on it -- but it is the one field that actually
# differs, so it is the field that tells you which family you are looking at
# if you ever dump GGUF metadata by hand.
ENCODER_FAMILIES: Dict[str, Dict[str, Any]] = {
    # Qwen3-VL variants used as TEXT encoders. sd.cpp's --llm (and llama.cpp's
    # llama-completion) load only the language backbone of a VL gguf; the vision
    # tower lives in a SEPARATE mmproj file, so passing the base *.gguf WITHOUT
    # an mmproj means vision never activates and no mmproj is loaded or
    # requested -- the model behaves as a plain Qwen3 text encoder. Its text
    # backbone IS the same Qwen3 architecture (and hidden dim) as the dense
    # model of the same size, so klein-9B pairs with a VL-8B backbone (4096) and
    # Z-Image / klein-4B with a VL-4B backbone (2560). These entries exist so
    # the enhance step and -ngl resolution know the specs; the compatibility
    # check still trusts the gguf's own embedding_length over this table.
    # (Confirmed by sd.cpp's Krea2 guide, which drives --llm Qwen3-VL-4B with no
    # mmproj.) These MUST precede the plain entries: a VL filename such as
    # "qwen3-vl-flux2-8b" carries "flux2" but is an ENCODER, and the "vl" token
    # is what distinguishes it.
    "qwen3-vl-8b": {
        "label": "Qwen3-VL-8B (text backbone)",
        "pattern": r"qwen[-_. ]?3[-_. ]?vl.*?8[-_. ]?b(?![a-z0-9])",
        "layers": 36,
        "embedding_length": 4096,
        "max_context": 40960,
    },
    "qwen3-vl-4b": {
        "label": "Qwen3-VL-4B (text backbone)",
        "pattern": r"qwen[-_. ]?3[-_. ]?vl.*?4[-_. ]?b(?![a-z0-9])",
        "layers": 36,
        "embedding_length": 2560,
        "max_context": 40960,
    },
    "qwen3-8b": {
        "label": "Qwen3-8B",
        "pattern": r"qwen[-_. ]?3[-_. ]?8[-_. ]?b(?![a-z0-9])",
        "layers": 36,
        "embedding_length": 4096,
        "max_context": 40960,
    },
    "qwen3-4b": {
        "label": "Qwen3-4B",
        "pattern": r"qwen[-_. ]?3[-_. ]?4[-_. ]?b(?![a-z0-9])",
        "layers": 36,
        "embedding_length": 2560,
        "max_context": 40960,
    },
}

# llama.cpp clamps any -ngl larger than the real block count to "all layers".
# Used only when neither GGUF metadata nor the filename identifies a family;
# see inference._resolve_gpu_layers() for why this is a last resort and not a
# default.
ENCODER_UNKNOWN_LAYERS = 99


def encoder_family(name: str) -> Optional[str]:
    """Return the ENCODER_FAMILIES key matching `name`, or None.

    `name` is any filename or bare model name; matching is case-insensitive.
    Used as the FALLBACK oracle only -- GGUF `*.block_count` metadata is
    authoritative and inference.py consults it first.
    """
    low = str(name).lower()
    for fam, spec in ENCODER_FAMILIES.items():
        if re.search(spec["pattern"], low):
            return fam
    return None


def get_encoder_family_spec(name: str) -> Optional[Dict[str, Any]]:
    """The ENCODER_FAMILIES entry matching `name`, or None if unrecognised."""
    fam = encoder_family(name)
    return ENCODER_FAMILIES[fam] if fam else None

# ---------------------------------------------------------------------------
# Diffuser model constraints (Z-Image-Turbo / Lumina2-style DiT)
# Sourced from GGUF metadata (layers(30), layers.0 .. layers.29) supplied by
# the user. context_refiner / final_layer / embedders are single fixed
# blocks, not part of the offloadable repeating-layer stack.
# ---------------------------------------------------------------------------
DIFFUSER_MAX_LAYERS = 30

# ---------------------------------------------------------------------------
# sd.cpp has NO per-layer GPU offload for the diffusion model. Unlike llama.cpp
# (-ngl <N>), placement is whole-module only. Modern sd.cpp expresses this with
# two assignments -- --backend (where graphs execute) and --params-backend
# (where weights live) -- addressing modules named `diffusion`, `te` and `vae`.
# So we expose a 3-way placement choice rather than a meaningless layer count.
# See parse_diffuser_placement() below and inference.generate_image().
#
# History worth keeping: the older whole-component flags (--clip-on-cpu,
# --vae-on-cpu) are still accepted by sd.cpp but are IGNORED whenever --backend
# is set. Passing both -- which this program used to do -- meant the Split
# option silently behaved as Full GPU, so the Qwen3 conditioner loaded onto VRAM
# beside the diffusion model and 8GB cards OOM'd. CPU-only runs set no --backend
# and so kept working, which is why the failure looked GPU-specific.
# ---------------------------------------------------------------------------
DIFFUSER_PLACEMENT_FULL_GPU = "Full GPU"
DIFFUSER_PLACEMENT_SPLIT    = "Split (Diffusion GPU, Encoder/VAE CPU)"
DIFFUSER_PLACEMENT_FULL_CPU = "Full CPU"

DIFFUSER_PLACEMENT_CHOICES: List[str] = [
    DIFFUSER_PLACEMENT_FULL_GPU,
    DIFFUSER_PLACEMENT_SPLIT,
    DIFFUSER_PLACEMENT_FULL_CPU,
]

# ---------------------------------------------------------------------------
# Diffuser FAMILY  (Z-Image-Turbo vs Flux.2-Klein)  --  the whole flux.2 story
# ---------------------------------------------------------------------------
# This program drives exactly two diffusion families through sd.cpp, and every
# per-family difference (which VAE, which encoder size, which cfg/step defaults,
# which extra CLI flags, which Settings panel to show) hangs off this one
# discriminator. Adding a third family later means adding one dict entry here
# plus the matching branch in inference.generate_image(), nothing more.
#
# BOTH families condition through a Qwen3 LLM passed as sd.cpp's --llm and reach
# pixels through a --vae; neither uses CLIP or T5. What differs:
#
#   Z-Image-Turbo (Tongyi-MAI, arch `lumina2`, 6B S3-DiT)
#       encoder : Qwen3-4B  (hidden 2560) -- the ONLY size that fits; the DiT's
#                 text projection is built for 2560-dim input, so a Qwen3-8B
#                 (4096) mismatches exactly the way Flux.2-klein-4B rejects an
#                 8B encoder. (The old note in this file claiming 8B works for
#                 Z-Image was wrong; Tongyi-MAI themselves say use the coupled
#                 Qwen3-4B.)
#       vae     : ae.safetensors  (the Flux.1 VAE Z-Image reuses)
#
#   Flux.2-Klein (Black-Forest-Labs, arch `flux`/`flux2`)
#       klein-4B / klein-base-4B : encoder Qwen3-4B (2560), vae flux2_ae
#       klein-9B / klein-base-9B : encoder Qwen3-8B (4096), vae flux2_ae
#       image-to-image / editing : native, via sd.cpp -r <ref.png> (repeatable),
#                                  NO vision model needed -- the reference image
#                                  goes straight into the pipeline.
#       distilled  (klein / klein-9b)      : cfg 1.0, ~4 steps
#       base       (klein-base / *-base-*) : cfg 4.0, ~20 steps
#
# So a single Qwen3-4B encoder serves BOTH Z-Image and klein-4B; only klein-9B
# needs the 8B. There are, in effect, just two encoder "slots".
# ---------------------------------------------------------------------------
DIFFUSER_FAMILY_ZIMAGE = "z-image"
DIFFUSER_FAMILY_FLUX2  = "flux2"

# Reference-image input formats sd.cpp can decode. sd.cpp loads -r images via
# stb_image, which handles PNG, JPEG (baseline+progressive), BMP (non-RLE),
# TGA, GIF, PSD, HDR and PNM/PPM/PGM. Every common photo format the user is
# likely to hand us (png/jpg/bmp) is in that set, so NO external converter
# (nconvert etc.) is warranted -- we simply refuse, with a clear message,
# anything stb_image cannot read (WEBP, AVIF, HEIC/HEIF, TIFF, JPEG-2000).
# Note WebP is an OUTPUT-only format in sd.cpp (libwebp encode), not decodable
# as an input here. Extensions are compared lowercased, leading dot included.
SUPPORTED_REF_IMAGE_EXTS = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tga", ".gif",
    ".psd", ".hdr", ".pic", ".pnm", ".ppm", ".pgm",
}

# Encoder hidden_size per Qwen3 size, and the size each diffuser demands. The
# check in inference.check_model_compatibility() is a dimension comparison, not
# a name guess: GGUF `*.embedding_length` (already read by _probe_gguf) gives
# 2560 for Qwen3-4B and 4096 for Qwen3-8B directly.
ENCODER_DIM_QWEN3_4B = 2560
ENCODER_DIM_QWEN3_8B = 4096

# general.architecture (lowercased, exact) -> family. Klein GGUFs report `flux`
# or `flux2`; Z-Image and its finetunes report `lumina2`.
DIFFUSER_FLUX2_ARCH:  List[str] = ["flux2", "flux"]
DIFFUSER_ZIMAGE_ARCH: List[str] = ["lumina2"]

# Filename/folder patterns (re.search, lowercased) for when arch is unreadable.
# Flux2 is checked FIRST: a "flux-2-klein" name must not fall through to the
# z-image bucket. These are a strict subset of DIFFUSION_NAME_PATTERNS split by
# family -- the "flux"/"klein" tokens go to flux2, everything Z-Image to zimage.
DIFFUSER_FLUX2_NAME_PATTERNS: List[str] = [
    r"flux[-_. ]?2",
    r"flux[-_. ]?klein",
    r"\bklein\b",
]
DIFFUSER_ZIMAGE_NAME_PATTERNS: List[str] = [
    r"z[-_. ]?image",
    r"zit[-_. ]?v?\d",
    r"\bzit\b",
    r"dark[-_. ]?beast",
    r"event[-_. ]?horizon",
    r"perfeczion",
    r"smooth[-_. ]?mix",
    r"turbo",
]

# VAE name patterns per family. flux2's VAE frequently downloads from the BFL
# repo as the generic "diffusion_pytorch_model.safetensors" (identifies
# nothing), so flux2 auto-detection legitimately misses and the user points at
# it by hand -- display.py blanks the VAE box on a cross-family switch for
# exactly this reason. Z-Image's ae.safetensors is reliably named.
#
# NOTE the z-image pattern is boundaried so it matches "ae.safetensors" but NOT
# "flux2_ae" (underscore is a word char, so a bare \bae\b would never have fired
# on flux2_ae anyway -- this is also the fix for that latent miss).
VAE_ZIMAGE_NAME_PATTERNS: List[str] = [
    r"(?<![a-z0-9])ae(?![a-z0-9_])",
]
VAE_FLUX2_NAME_PATTERNS: List[str] = [
    r"flux[-_. ]?2[-_. ]?ae",
    r"flux[-_. ]?2[-_. ]?vae",
    r"flux2ae",
]

DIFFUSER_FAMILY_LABELS: Dict[str, str] = {
    DIFFUSER_FAMILY_ZIMAGE: "Z-Image-Turbo",
    DIFFUSER_FAMILY_FLUX2:  "Flux.2-Klein",
}

# What each family's VAE box should say it wants, shown as the box's info hint.
DIFFUSER_VAE_HINTS: Dict[str, str] = {
    DIFFUSER_FAMILY_ZIMAGE: "Z-Image expects ae.safetensors (the Flux.1 VAE).",
    DIFFUSER_FAMILY_FLUX2:  ("Flux.2 expects flux2_ae "
                             "(often downloaded as diffusion_pytorch_model.safetensors)."),
}

# Per-family generation defaults for the distilled (fast) and base variants.
# Used by display.py to seed the correct Settings panel and by inference.py as
# fall-backs. Z-Image-Turbo keeps its existing turbo defaults elsewhere.
FLUX2_DISTILLED_DEFAULTS: Dict[str, Any] = {
    "imagegen_cfg_scale": 1.0, "imagegen_steps": 4, "imagegen_sampling": "euler",
}
FLUX2_BASE_DEFAULTS: Dict[str, Any] = {
    "imagegen_cfg_scale": 4.0, "imagegen_steps": 20, "imagegen_sampling": "euler",
}


def diffuser_family(name: str, arch: str = "") -> Optional[str]:
    """Return DIFFUSER_FAMILY_FLUX2 / DIFFUSER_FAMILY_ZIMAGE / None.

    Architecture metadata wins when present (same policy as classify_model());
    filename+folder patterns are the fallback. Flux2 is tested before Z-Image.
    `name` may be a bare filename or a full path; matching is case-insensitive.
    """
    a = str(arch).strip().lower()
    if a:
        if a in DIFFUSER_FLUX2_ARCH:
            return DIFFUSER_FAMILY_FLUX2
        if a in DIFFUSER_ZIMAGE_ARCH:
            return DIFFUSER_FAMILY_ZIMAGE
    low = str(name).lower()
    for pat in DIFFUSER_FLUX2_NAME_PATTERNS:
        if re.search(pat, low):
            return DIFFUSER_FAMILY_FLUX2
    for pat in DIFFUSER_ZIMAGE_NAME_PATTERNS:
        if re.search(pat, low):
            return DIFFUSER_FAMILY_ZIMAGE
    return None


def diffuser_family_label(name: str, arch: str = "") -> str:
    """Human label for a diffuser, or 'no model selected' when unrecognised."""
    fam = diffuser_family(name, arch)
    return DIFFUSER_FAMILY_LABELS.get(fam, "no model selected")


def is_flux2_base_variant(name: str) -> bool:
    """True for klein-base checkpoints (higher cfg / more steps), False for the
    distilled klein. `base` appears in the filename of every base variant."""
    return bool(re.search(r"\bbase\b|[-_.]base[-_.]", str(name).lower()))


def flux2_size_dim(name: str) -> Optional[int]:
    """Required encoder hidden dim for a Flux.2-klein diffuser, from its size
    token: klein-4B -> 2560 (Qwen3-4B), klein-9B -> 4096 (Qwen3-8B).
    Returns None if no clean 4b/9b token is present."""
    m = re.search(r"(?<![a-z0-9])(\d{1,2})b(?![a-z0-9])", str(name).lower())
    if not m:
        return None
    n = int(m.group(1))
    if n == 4:
        return ENCODER_DIM_QWEN3_4B
    if n == 9:
        return ENCODER_DIM_QWEN3_8B
    return None


def required_encoder_dim(diffuser_name: str, diffuser_arch: str = "") -> Optional[int]:
    """The encoder hidden dim a given diffuser demands, or None if unknown.

    Z-Image-Turbo -> 2560 (Qwen3-4B). Flux.2-klein -> 2560 for 4B, 4096 for 9B.
    None means "cannot tell" (unrecognised family, or a flux2 file with no
    parseable size) -- callers treat None as "skip the check", never as a
    mismatch, so an oddly-named file is allowed through rather than blocked.
    """
    fam = diffuser_family(diffuser_name, diffuser_arch)
    if fam == DIFFUSER_FAMILY_ZIMAGE:
        return ENCODER_DIM_QWEN3_4B
    if fam == DIFFUSER_FAMILY_FLUX2:
        return flux2_size_dim(diffuser_name)
    return None


def encoder_dim_from_spec(name: str) -> Optional[int]:
    """Encoder hidden dim from the filename's Qwen3 family, or None.

    Fallback for when GGUF `embedding_length` metadata is unavailable;
    inference.py reads the metadata first and only calls this if it is missing.
    """
    spec = get_encoder_family_spec(name)
    if spec:
        try:
            return int(spec["embedding_length"])
        except (KeyError, TypeError, ValueError):
            return None
    return None


def vae_family(name: str) -> Optional[str]:
    """Which family a VAE filename belongs to, or None if it identifies nothing
    (e.g. flux2's generic diffusion_pytorch_model.safetensors)."""
    low = str(name).lower()
    for pat in VAE_FLUX2_NAME_PATTERNS:
        if re.search(pat, low):
            return DIFFUSER_FAMILY_FLUX2
    for pat in VAE_ZIMAGE_NAME_PATTERNS:
        if re.search(pat, low):
            return DIFFUSER_FAMILY_ZIMAGE
    return None


def flux2_flash_attn_for(cfg: Dict[str, Any]) -> bool:
    """Decide whether a Flux.2 run should pass --diffusion-fa, automatically,
    from the selected diffuser GPU's fp16 capability — no user toggle.

    sd.cpp's Vulkan flash attention needs device fp16; on a GPU that reports
    fp16:0 (e.g. RX 470 / Polaris) it silently garbles output, which is the bug
    this replaces the old checkbox for. Rules:
      * Diffuser on CPU (no Vulkan backend): fp16 is a GPU concern, so keep FA
        on (it still trims memory and does not garble on CPU).
      * Diffuser on a Vulkan device: FA on only if that device supports fp16.
      * fp16 unknown (detection not re-run since this was added): default OFF,
        the safe choice — an fp16-capable card merely loses a little VRAM
        headroom until the installer's detect pass records gpu<N>_fp16 in
        constants.ini (see installer.probe_ggml_devices).
    """
    placement = parse_diffuser_placement(
        cfg.get("imagegen_placement", DIFFUSER_PLACEMENT_FULL_GPU))
    if not placement.get("use_vulkan_backend"):
        return True
    try:
        dev = int(cfg.get("imagegen_vulkan_device", -1))
    except (TypeError, ValueError):
        dev = -1
    return bool(device_supports_fp16(dev))


def device_supports_fp16(dev: int) -> Optional[bool]:
    """fp16 capability for a ggml Vulkan device index, from what the installer
    detected and recorded per device in constants.ini (gpu<N>_fp16), read back
    via get_vulkan_info(). Returns None when the device was not recorded (e.g.
    detection has not been re-run since this feature was added); callers treat
    None as 'no'. Re-run the installer's detect-only pass to populate it."""
    for d in get_vulkan_info().get("devices", []):
        if d.get("index") == dev and d.get("fp16") is not None:
            return bool(d["fp16"])
    return None

# ---------------------------------------------------------------------------
# Model family identification
# ---------------------------------------------------------------------------
# Single source of truth for "what kind of model is this file?". Consumed by
# inference.categorize_models() and inference.is_sd_classic_model(); nothing
# else in the program should keep its own private keyword list.
#
# TWO oracles, used in priority order by inference.py:
#
#   1. GGUF metadata `general.architecture` (already read by _probe_gguf).
#      Authoritative, and independent of whatever the quantizer felt like
#      naming the file. Every Z-Image-Turbo diffusion gguf reports `lumina2`
#      -- the stock Tongyi-MAI z_image_turbo-Q#.gguf and equally every
#      community finetune of it (BigDannyPt/Z-Image-Turbo-GGUF-Collection:
#      DarkBeast, EventHorizon, PerfecZion, SmoothMix Ultimate, ZiT Anime,
#      ZiT NSFW Photorealistic; 6B params, Q3_K_M ~4.1GB .. Q8_0 ~7.2GB).
#      Those finetunes are the same S3-DiT weights with different training,
#      so they need exactly the same handling as the stock checkpoint:
#      --diffusion-model + --llm <qwen3 encoder> + --vae ae.safetensors,
#      and NO --clip-skip.
#
#   2. Filename patterns, for files whose metadata could not be read and for
#      safetensors (which carry no architecture field at all). The containing
#      folder names are matched too, because the collection ships nested --
#      e.g. DarkBeast/DBZiT9-DIMRClaw/darkBeastMar2126Latest_dbzit9DIMRclaw-Q8_0.gguf
#      -- so the folder often names the family more clearly than the file does.
#
# All *_NAME_PATTERNS entries are REGEX, applied with re.search() against the
# lowercased name. Short/ambiguous tokens are \b-anchored so `vl` matches
# "qwen2-vl-7b" but not "swivl-mix". The Z-Image entries have to survive every
# separator style in the wild (z_image_turbo, Z-Image-Turbo, zImageTurboAnime)
# plus the "ZiT" abbreviation the finetunes use instead of the full name
# (dbzit8, dbzit9, zitV10) -- without those, half the collection matched
# nothing and fell through to "unknown".
# ---------------------------------------------------------------------------

# Exact-match (not substring) against general.architecture, lowercased.
# Exact, because "qwen_image" is a DIFFUSION arch while "qwen3" is an ENCODER
# arch -- a substring test for "qwen" would put the diffuser in the wrong bin.
ENCODER_ARCHITECTURES: List[str] = [
    "qwen3", "qwen3moe", "qwen3vl", "qwen3vlmoe", "qwen2", "qwen2vl",
    "llama", "gemma3", "mistral", "phi3", "t5", "t5encoder", "umt5", "clip",
]

DIFFUSION_ARCHITECTURES: List[str] = [
    "lumina2",      # Z-Image / Z-Image-Turbo and all its finetunes
    "flux", "chroma", "sd1", "sd2", "sd3", "sdxl",
    "qwen_image", "wan", "ltxv", "hunyuan_video",
]

VAE_ARCHITECTURES: List[str] = ["vae", "autoencoder"]

# Checked BEFORE the diffusion patterns: the encoder gguf is itself named
# after the diffuser it serves ("Qwen3-4b-Z-Image-Turbo-AbliteratedV1"), so a
# diffusion-first order would misfile the encoder as a diffuser.
ENCODER_NAME_PATTERNS: List[str] = [
    r"qwen(?!.{0,3}image)",   # qwen3-4b-... but not qwen_image / qwen-image
    r"engineer",
    r"encoder",
    r"\bllava\b",
    r"\bvision\b",
    r"\bvl\b",
    r"\bu?mt5(?:xxl)?\b",
    r"\bclip[-_]?[lg]\b",
]

DIFFUSION_NAME_PATTERNS: List[str] = [
    r"z[-_. ]?image",         # z_image_turbo, Z-Image-Turbo, zImageTurboAnime
    r"zit[-_. ]?v?\d",        # dbzit8, dbzit9DIMRclaw, zitV10
    r"dark[-_. ]?beast",      # BigDannyPt collection, family names
    r"event[-_. ]?horizon",
    r"perfeczion",
    r"smooth[-_. ]?mix",
    r"turbo",
    r"diffusion",
    r"\bunet\b",
    r"\bdit\b",
    r"sd[-_. ]?xl",
    r"flux",
    r"chroma",
    r"illustrious",
    r"\bsd3\b",
    r"\bwan[-_. ]?\d",
    r"\bltx",
]

VAE_NAME_PATTERNS: List[str] = [
    r"\bae\b",                # ae.safetensors — the Z-Image VAE
    r"vae",
    r"autoencoder",
]

# --clip-skip is meaningful ONLY for checkpoints with a CLIP text encoder
# (SD1.x / SD2.x / SDXL and its derivatives). Z-Image conditions through the
# Qwen3 LLM instead and has no CLIP at all, so passing --clip-skip to it is at
# best ignored and at worst a hard error.
#
# This is a POSITIVE allow-list on purpose. The old test was the inverse --
# "SD-classic unless the name contains flux/z_image/sd3/wan/ltx" -- which is
# wrong by default for anything unrecognised, and did not even catch the
# models this program actually ships against: "zImageTurboAnime_v10-Q4_K_M"
# has no underscore in "z_image", so it was treated as SD-classic and had
# --clip-skip 2 bolted onto its command line. Unknown now means "no CLIP",
# which is the safe direction for a Z-Image-first program.
SD_CLASSIC_NAME_PATTERNS: List[str] = [
    r"\bsd[-_. ]?1[-_. ]?[45]\b",
    r"\bsd[-_. ]?15\b",
    r"\bv1[-_. ]?5[-_. ]?pruned",
    r"\bsd[-_. ]?2[-_. ]?[01]\b",
    r"\bsd[-_. ]?21\b",
    r"sd[-_. ]?xl",
    r"illustrious",
    r"\bpony\b",
    r"noobai",
]

SD_CLASSIC_ARCHITECTURES: List[str] = ["sd1", "sd2", "sdxl"]

# Quantization tokens recognised in filenames by
# inference.get_quantization_label(), longest-first at match time so that
# "q4_k_m" wins over "q4_k". BF16 is listed ahead of F16 for the same reason:
# the collection's source-precision names ("perfeczion_10BF16-Q4_K_M") contain
# both, and a bare "f16" substring test used to label BF16-sourced files F16.
QUANT_TOKENS: List[str] = [
    "iq1_s", "iq1_m",
    "iq2_xxs", "iq2_xs", "iq2_s", "iq2_m",
    "iq3_xxs", "iq3_xs", "iq3_s", "iq3_m",
    "iq4_xs", "iq4_nl",
    "q2_k_s", "q2_k",
    "q3_k_s", "q3_k_m", "q3_k_l", "q3_k",
    "q4_0", "q4_1", "q4_k_s", "q4_k_m", "q4_k",
    "q5_0", "q5_1", "q5_k_s", "q5_k_m", "q5_k",
    "q6_k", "q8_0", "q8_k",
    "tq1_0", "tq2_0",
    "bf16", "fp16", "f16", "fp32", "f32", "f64",
]

# ---------------------------------------------------------------------------
# Shared constants / maps / lists
# ---------------------------------------------------------------------------
SAMPLER_MAP: Dict[str, str] = {
    "euler_a": "euler_a", "euler": "euler", "heun": "heun",
    "dpm2": "dpm2", "dpm++2s_a": "dpm++2s_a", "dpm++2m": "dpm++2m",
    "dpm++2mv2": "dpm++2mv2", "lcm": "lcm", "ddim": "ddim", "plms": "plms",
}
# sd.cpp silently rounds width/height up to the nearest multiple of 64
# internally (latent-space alignment requirement of the VAE/DiT). Any value
# here that is not itself a multiple of 64 will be "corrected" by sd.cpp
# without telling the UI, so the dropdown would show one value while the
# actual output used another. Restricting the choices to confirmed-safe,
# already-64-aligned values eliminates that mismatch entirely.
IMAGE_SIZES       = [256, 512, 768, 1024, 1280, 1536, 1792, 2048]

# Per-family width/height choices (both axes share a list). All multiples of
# 256 within each family's usable band:
#   * Flux.2 klein is ~1024-native and visibly degrades below 512 (confirmed in
#     testing — sub-512 output goes mushy/semi-garbled), so its floor is 512.
#     Black Forest Labs' documented ceiling is 4MP (2048x2048); they recommend
#     staying at or below ~2MP (~1536) for best quality/speed, but the hard max
#     is 2048, so a big-VRAM card can use it.
#   * Z-Image-Turbo is also 1024-native, tolerates smaller sizes (256 floor),
#     and Tongyi-MAI report anything up to 2048 is fine — same 2048 ceiling.
# _generate_family_updates() swaps the width/height dropdowns to the matching
# list when a model is selected, keeping a currently-valid value and only
# snapping an out-of-band one to a sensible default. VRAM, not this list, is the
# practical limit at the top end (2048 forces heavy CPU-offload on 8GB).
FLUX2_IMAGE_SIZES  = [512, 768, 1024, 1280, 1536, 1792, 2048]
ZIMAGE_IMAGE_SIZES = [256, 512, 768, 1024, 1280, 1536, 1792, 2048]

FAMILY_IMAGE_SIZES: Dict[str, List[int]] = {
    "flux2_distilled": FLUX2_IMAGE_SIZES,
    "flux2_base":      FLUX2_IMAGE_SIZES,
    "z-image":         ZIMAGE_IMAGE_SIZES,
    "none":            IMAGE_SIZES,
}


def family_image_sizes(diffuser_name: str) -> List[int]:
    """Allowed width/height values for the loaded diffuser's family."""
    return FAMILY_IMAGE_SIZES.get(family_step_cfg_key(diffuser_name), IMAGE_SIZES)

# Z-Image-Turbo is a distilled few-step model; step counts are conventionally
# chosen as doubling powers of two (2/4/6/8/10/12) to match the distillation
# schedule the turbo checkpoint was trained against. Restricting the choices
# here keeps the UI from offering values that don't correspond to a step the
# turbo schedule was actually trained on.
STEP_CHOICES      = [2, 4, 6, 8, 10, 12]
BATCH_SIZE_CHOICES = [128, 256, 512, 1024, 2048]

# ---------------------------------------------------------------------------
# Per-family step / CFG ranges  (the two setting sets the Generation page
# swaps between when the diffuser family changes)
# ---------------------------------------------------------------------------
# Sources: BFL/ComfyUI/community testing for Flux.2-klein (Jan 2026 onward) and
# the Z-Image-Turbo distillation schedule. Key facts encoded here:
#   * Klein DISTILLED: cfg is fixed at 1.0 (guidance-distilled; other values
#     degrade), and it is a 4-step model — 4-6 is ideal, 8 is the practical
#     ceiling, higher REDUCES quality. Editing wants the same, not higher cfg.
#   * Klein BASE (undistilled): behaves like a normal flow model — cfg ~4 and
#     ~20 steps.
#   * Z-Image-Turbo: its own few-step schedule; the app's existing 8-step /
#     cfg-1.0 default is kept (it was already working), with the full slider.
#
# Each entry: steps -> (choices, default); cfg -> (min, max, step, default).
FAMILY_STEP_CFG = {
    "flux2_distilled": {
        "steps":  ([4, 6, 8], 4),
        "cfg":    (1.0, 2.0, 0.5, 1.0),
    },
    "flux2_base": {
        "steps":  ([16, 20, 24, 28], 20),
        "cfg":    (1.0, 8.0, 0.5, 4.0),
    },
    # Z-Image-Turbo (distilled S3-DiT): 8-step model — 8 is the sweet spot,
    # 10-12 rarely helps (try a new seed instead). cfg stays low (~1.0-2.0);
    # 4+ "fries"/over-saturates a distilled model, so the range caps at 4.0.
    # Sampler is NOT forced (euler AND euler_a both work well for Z-Image),
    # unlike Flux.2 which must be euler.
    "z-image": {
        "steps":  ([4, 6, 8, 10, 12], 8),
        "cfg":    (1.0, 4.0, 0.5, 1.0),
    },
    # No diffuser chosen yet: keep the permissive superset.
    "none": {
        "steps":  (STEP_CHOICES, 8),
        "cfg":    (0.5, 20.0, 0.5, 1.0),
    },
}


def family_step_cfg_key(diff_path: str, diff_arch: str = "") -> str:
    """Map a diffuser to its FAMILY_STEP_CFG key: 'flux2_distilled',
    'flux2_base', 'z-image', or 'none'."""
    fam = diffuser_family(diff_path, diff_arch) if diff_path else None
    if fam == DIFFUSER_FAMILY_FLUX2:
        return "flux2_base" if is_flux2_base_variant(diff_path) else "flux2_distilled"
    if fam == DIFFUSER_FAMILY_ZIMAGE:
        return "z-image"
    return "none"


def family_step_cfg(diff_path: str, diff_arch: str = "") -> Dict[str, Any]:
    """The step choices + cfg range set for a diffuser's family (see
    FAMILY_STEP_CFG). Always returns a valid entry (falls back to 'none')."""
    return FAMILY_STEP_CFG[family_step_cfg_key(diff_path, diff_arch)]

# Context size maxes out at the model's trained context length (40960)
CTX_SIZE_CHOICES  = [2048, 4096, 8192, 16384, 32768, 40960]

# GPU layers maxes out at the model's block count (36).
# -1 means "offload all layers to GPU" (resolves to 36 in inference.py's
# _resolve_gpu_layers()). llama.cpp loads exactly that many transformer
# blocks onto the GPU; anything not offloaded (i.e. any layer count below
# the full 36, including the implicit remainder when an OOM retry reduces
# the count) stays resident in system RAM and is computed on CPU for that
# portion. -1 does NOT mean "spill VRAM overflow into RAM mid-layer" — it
# simply offloads every layer; if that doesn't fit in VRAM, llama.cpp/sd.cpp
# will fail to allocate rather than silently falling back, which is why the
# OOM-retry logic in inference.py progressively lowers ngl on failure.
GPU_LAYER_CHOICES = [-1, 0, 4, 8, 12, 16, 20, 24, 28, 32, 36]

CLIP_SKIP_CHOICES = [1, 2, 3]
BATCH_COUNT_CHOICES = [1, 2, 3, 4]
OUTPUT_FORMATS    = ["png", "jpg", "bmp"]

# ---------------------------------------------------------------------------
# Preferences (data/preferences.json — the Preferences page)
# ---------------------------------------------------------------------------
# Settings that are the user's standing taste rather than this machine's
# hardware/model wiring. They live in their own file so that wiping or
# regenerating the Configuration page's file (data/configuration.json, which
# the installer reseeds on a clean install) never takes the user's Prompt
# Template with it.
#
# DEFAULT_PROMPT_TEMPLATE is the Qwen3 ChatML frame the encoder expects;
# {prompt} is substituted by inference.enhance_prompt(). Single source of
# truth — installer.py's write_default_preferences() seeds the same literal,
# and _default_preferences() below backfills it.
DEFAULT_PROMPT_TEMPLATE: str = (
    "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
)

# How many thumbnails the Generate page's Thumbnails Gallery shows. The
# gallery caches the FULL output-folder listing and only slices it (see
# display._get_recent_images), so this is a display cap, not a scan cap —
# raising it costs render time, not another disk sweep.
MAX_THUMBNAIL_CHOICES: List[int] = [50, 100, 200]
DEFAULT_MAX_THUMBNAILS: int = 50

# Key used by the unified bottom status bar shared across all pages.
STATUS_BAR_KEY: str = "status"

# Generate tab preview box height (pixels). Single source of truth — used
# both for the gr.Image(height=...) kwarg in display.py AND for the
# #preview-img CSS rule in the same file. The CSS carries !important and
# would silently override the Python kwarg if the two ever disagreed, so
# changing this one constant is the only thing that should ever be needed
# to resize the preview box; display.py interpolates it into the CSS string
# rather than hardcoding the pixel value a second time.
PREVIEW_IMAGE_HEIGHT: int = 500

# Thumbnails Gallery row height (pixels). Single source of truth, same
# pattern as PREVIEW_IMAGE_HEIGHT above: used BOTH for the gr.Gallery(
# height=...) kwarg in display.py AND interpolated into the #output-gallery
# CSS in the same file (which carries !important and would otherwise silently
# override the Python kwarg). The per-thumbnail cell WIDTH is derived from
# this value in the CSS via calc(), so the gallery stays exactly one row of
# roughly-square thumbnails and this one number is all that ever needs
# changing to resize it. Note this is the row HEIGHT only — it does NOT cap
# how many thumbnails appear; that is Max Thumbnails Displayed (see
# MAX_THUMBNAIL_CHOICES / get_max_thumbnails above), and the row scrolls
# horizontally to reach the ones past the window edge.
THUMBNAIL_GALLERY_HEIGHT: int = 123

# ---------------------------------------------------------------------------
# Qt app-window geometry (persisted across sessions by launcher.py)
# ---------------------------------------------------------------------------
WINDOW_GEOMETRY_UNSET: int = -1   # sentinel for "no saved x/y yet"
WINDOW_DEFAULT_WIDTH:  int = 1280
WINDOW_DEFAULT_HEIGHT: int = 860


# ---------------------------------------------------------------------------
# Runtime state  (transient — not persisted)
# ---------------------------------------------------------------------------

APP_STATE: Dict[str, Any] = {
    "last_image_path": "",
    "last_prompt": "",
    "is_building": False,
    "last_batch_elapsed_seconds": 0,  # elapsed seconds of the last completed batch job
}

# ---------------------------------------------------------------------------
# Generation phase timing  (transient — not persisted to disk)
# ---------------------------------------------------------------------------
# Learned from the previous generation(s) *within this session* so the status
# bar can show an accurate ETA-style timer instead of a flat "Generating...".
# Updated by inference.py as each phase completes:
#   encoder_seconds            - how long the last prompt-encoding pass took
#   diffusion_total_seconds    - how long the last full diffusion pass took
#   diffusion_steps            - how many steps that diffusion pass used
#   diffusion_per_step_seconds - diffusion_total_seconds / diffusion_steps
# All start at 0.0 / 0 until the first generation in the session completes
# the relevant phase at least once. display.py falls back to a plain
# "Encoding..." / "Diffusing..." message (no ETA) until a value is available.
TIMING_STATS: Dict[str, float] = {
    "encoder_seconds": 0.0,
    "diffusion_total_seconds": 0.0,
    "diffusion_steps": 0,
    "diffusion_per_step_seconds": 0.0,
}


def update_timing_stat(key: str, value: float) -> None:
    """Single-key update for TIMING_STATS. Dict item assignment is atomic
    under the GIL, so no lock is needed for this simple case."""
    if key in TIMING_STATS:
        TIMING_STATS[key] = value


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT: Optional[Path] = None


def _get_project_root() -> Path:
    global _PROJECT_ROOT
    if _PROJECT_ROOT is not None:
        return _PROJECT_ROOT
    current = Path(__file__).resolve().parent
    _PROJECT_ROOT = current.parent if current.name == "scripts" else current
    return _PROJECT_ROOT


def get_constants_path() -> Path:
    return _get_project_root() / "data" / "constants.ini"


def get_configuration_path() -> Path:
    """data/configuration.json — everything set on the Configuration page,
    plus the Generate page's auto-saved generation settings and the Qt window
    geometry. Formerly data/persistent.json; see _migrate_legacy_persistent()."""
    return _get_project_root() / "data" / "configuration.json"


def get_preferences_path() -> Path:
    """data/preferences.json — everything set on the Preferences page."""
    return _get_project_root() / "data" / "preferences.json"


def _legacy_persistent_path() -> Path:
    """The pre-split filename. Only ever read, never written."""
    return _get_project_root() / "data" / "persistent.json"


def _migrate_legacy_persistent() -> None:
    """Rename data/persistent.json -> data/configuration.json, once.

    Only fires when the old file exists and the new one does not, so it can
    never clobber a real configuration.json, and re-running it is a no-op.
    A failed rename is not fatal: the caller falls through to defaults, which
    is the same outcome as a first run.
    """
    old = _legacy_persistent_path()
    new = get_configuration_path()
    if old.exists() and not new.exists():
        try:
            old.replace(new)
        except OSError:
            pass


def get_output_dir() -> Path:
    out = _get_project_root() / "output"
    out.mkdir(parents=True, exist_ok=True)
    return out


def get_models_dir() -> Path:
    return _get_project_root() / "models"


def get_media_dir() -> Path:
    """
    Static UI status images (not user data, not generated output):
      media/program_no_media.jpg  - shown when output/ is empty / idle
      media/program_encoding.jpg  - shown while the prompt-encoder LLM runs
      media/program_diffusion.jpg - shown while sd.cpp diffusion runs
    """
    return _get_project_root() / "media"


def get_build_dir() -> Path:
    return _get_project_root() / "data" / "build"


def get_llama_bin_dir() -> Path:
    return _get_project_root() / "data" / "llama_cpp_binaries"


def get_sd_bin_dir() -> Path:
    return _get_project_root() / "data" / "stable_diffusion_binaries"


def ensure_data_dirs() -> None:
    root = _get_project_root()
    for d in ("data", "output", "models", "media"):
        (root / d).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Hardware constants — read from constants.ini (written by installer)
# ---------------------------------------------------------------------------
#
# CPU_FEATURES mirrors installer.py's list. Both are REPORTING ONLY: the build
# uses GGML_NATIVE=ON, so ggml probes the build machine itself and ignores any
# -DGGML_*=ON we might pass. That is what makes the binaries optimal on any AMD
# or Intel CPU without a lookup table -- and it is why nothing here steers the
# build. These keys exist so the UI can tell the user what their CPU has.
CPU_FEATURES: List[Dict[str, str]] = [
    {"key": "has_sse4_2", "name": "SSE4.2"},
    {"key": "has_avx",    "name": "AVX"},
    {"key": "has_avx2",   "name": "AVX2"},
    {"key": "has_f16c",   "name": "F16C"},
    {"key": "has_fma",    "name": "FMA"},
    {"key": "has_avx512", "name": "AVX512"},
]


def _read_constants() -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    p = get_constants_path()
    if p.exists():
        cfg.read(p, encoding="utf-8")
    return cfg


def get_default_threads() -> int:
    """Return 85% of logical cores, sourced from constants.ini if available."""
    cfg = _read_constants()
    try:
        return int(cfg["cpu"]["default_threads"])
    except (KeyError, ValueError):
        return max(1, math.ceil((os.cpu_count() or 4) * 0.85))


def get_cpu_info() -> Dict[str, Any]:
    """Return CPU details dict sourced from constants.ini."""
    cfg = _read_constants()
    sec = cfg["cpu"] if cfg.has_section("cpu") else {}
    cores = int(sec.get("cores_logical", str(os.cpu_count() or 4)))
    phys  = int(sec.get("cores_physical", str(max(1, cores // 2))))
    dt    = int(sec.get("default_threads", str(max(1, math.ceil(cores * 0.85)))))
    info: Dict[str, Any] = {
        "brand":           sec.get("brand", "unknown"),
        "vendor":          sec.get("vendor", "unknown"),
        "arch":            sec.get("arch", "x86_64"),
        "cores_logical":   cores,
        "cores_physical":  phys,
        "default_threads": dt,
        "build_jobs":      int(sec.get("build_jobs", str(dt))),
        "has_aocl":        sec.get("has_aocl", "False") == "True",
        "cmake_flags":     sec.get("cmake_flags", "").split(),
        "arch_selection":  sec.get("arch_selection", "unknown"),
    }
    for feat in CPU_FEATURES:
        info[feat["key"]] = sec.get(feat["key"], "False") == "True"
    return info


def is_amd_cpu() -> bool:
    """True when this machine's CPU is AMD.

    installer.py's _cpu_vendor() writes exactly "AMD", "Intel" or "unknown"
    into constants.ini, so the vendor test is a plain equality. The brand
    fallback covers the "unknown" case -- a constants.ini written on a machine
    where the registry read failed still has a brand string like
    "AMD Ryzen 9 3900X 12-Core Processor" to go on.

    Used to decide whether AMD-only facts (e.g. AOCL presence) are worth
    reporting at all; on an Intel box they are noise, not information.
    """
    info = get_cpu_info()
    vendor = str(info.get("vendor", "")).strip().lower()
    if vendor == "amd":
        return True
    if vendor == "intel":
        return False
    brand = str(info.get("brand", "")).lower()
    return any(k in brand for k in ("amd", "ryzen", "epyc", "threadripper"))


def get_vulkan_info() -> Dict[str, Any]:
    """Return the GPU list ggml itself reported at install time.

    Devices are read from the per-device gpu<N>_* keys written by the installer
    after it ran `llama-completion --list-devices`. Those indices are exactly
    what `-dev Vulkan<N>` and `--backend vulkan<N>` expect -- they are ggml's
    own enumeration, which filters to supported discrete/integrated GPUs and
    deduplicates multi-driver duplicates, and therefore does NOT match a
    vulkaninfo ordering.

    gpu_numbers is the index list; gpu_names is a legacy convenience mirror and
    is deliberately not parsed for names (a GPU name may contain a comma).
    """
    cfg = _read_constants()
    sec = cfg["vulkan"] if cfg.has_section("vulkan") else {}

    available   = sec.get("available", "False") == "True"
    gpu_numbers = sec.get("gpu_numbers", "")      # e.g. "0,1"

    indices = [int(x.strip()) for x in gpu_numbers.split(",")
               if x.strip().lstrip("-").isdigit()]

    devices: List[Dict[str, Any]] = []
    for idx in indices:
        raw_fp16 = sec.get(f"gpu{idx}_fp16", "")
        fp16_val: Optional[bool]
        if raw_fp16 == "True":
            fp16_val = True
        elif raw_fp16 == "False":
            fp16_val = False
        else:
            fp16_val = None  # unknown (not recorded by this install)
        devices.append({
            "index":         idx,
            "name":          sec.get(f"gpu{idx}_name", f"GPU{idx}"),
            "backend":       sec.get(f"gpu{idx}_backend", "Vulkan"),
            "vram_total_mb": int(sec.get(f"gpu{idx}_vram_mb", "0") or 0),
            "vram_free_mb":  int(sec.get(f"gpu{idx}_free_mb", "0") or 0),
            "fp16":          fp16_val,
        })

    return {
        "available":     available,
        "version":       sec.get("version", "unknown"),
        "sdk":           sec.get("sdk", ""),
        "gpu_count":     len(devices),
        "gpu_numbers":   gpu_numbers,
        "gpu_names":     sec.get("gpu_names", ""),
        "enumerated_by": sec.get("enumerated_by", "not probed"),
        "devices":       devices,
    }


def get_install_type() -> str:
    """Return 'vulkan' or 'cpu_only' from constants.ini [general] section.
    Falls back to 'vulkan' when the key is absent (pre-existing installs that
    have Vulkan available should keep full GPU options).
    """
    cfg = _read_constants()
    if cfg.has_section("general"):
        return cfg["general"].get("install_type", "vulkan").strip().lower()
    # Legacy installs without [general]: infer from vulkan availability.
    vk = cfg["vulkan"] if cfg.has_section("vulkan") else {}
    return "vulkan" if vk.get("available", "False") == "True" else "cpu_only"


def get_backend_choices() -> Dict[str, List[str]]:
    """
    Build the encoder/imagegen backend dropdown choices for THIS machine.

    Entirely driven by what ggml enumerated at install time -- no assumption
    about how many GPUs exist or which one is "the" inference card. A laptop
    with one iGPU, a desktop with a monitor card plus a passive compute card,
    and a CPU-only box all produce a correct list.

    CPU entry  : "<CPU brand>"
    GPU entries: "Vulkan GPU 1 - Radeon RX 470 (8192 MiB)"
    GPU entries are omitted entirely for a cpu_only install.
    """
    cpu_info = get_cpu_info()
    cpu_label = cpu_info.get("brand", "CPU") or "CPU"
    cpu_choices = [cpu_label]
    gpu_choices: List[str] = []

    if get_install_type() != "cpu_only":
        vk = get_vulkan_info()
        for d in vk["devices"]:
            vram = f" ({d['vram_total_mb']} MiB)" if d.get("vram_total_mb") else ""
            gpu_choices.append(f"Vulkan GPU {d['index']} - {d['name']}{vram}")

    return {
        "cpu_choices": cpu_choices,
        "gpu_choices": gpu_choices,
        "all_choices": cpu_choices + gpu_choices,
    }


def get_thread_choices() -> List[int]:
    """Thread choices for this machine, capped at its actual logical cores.

    Always includes:
      * default_threads (85% of logical) - the installed default
      * cores_physical  - usually the fastest option for ggml compute, since
                          two threads sharing one core's FPU contend rather
                          than help. Offered so it can be A/B tested.
      * cores_logical   - the ceiling
    The old fixed list offered 48 and 64 on a 12-thread machine.
    """
    info = get_cpu_info()
    logical  = max(1, int(info.get("cores_logical", 4)))
    physical = max(1, int(info.get("cores_physical", max(1, logical // 2))))
    dt = get_default_threads()

    base = [1, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32, 48, 64]
    choices = {c for c in base if c <= logical}
    choices.update({physical, dt, logical})
    return sorted(c for c in choices if 1 <= c <= logical)


def parse_backend_choice(choice: str) -> Dict[str, Any]:
    """
    Convert a backend dropdown string back to the values inference needs.

    Returns {"use_vulkan": bool, "vulkan_device": int}
    e.g. "Vulkan GPU 1 — RX 470" → {"use_vulkan": True, "vulkan_device": 1}
         "AMD Ryzen 9 3900X ..."  → {"use_vulkan": False, "vulkan_device": -1}
    """
    m = re.match(r"^\s*Vulkan GPU\s+(\d+)\b", choice or "")
    if m:
        return {"use_vulkan": True, "vulkan_device": int(m.group(1))}
    return {"use_vulkan": False, "vulkan_device": -1}


def parse_diffuser_placement(placement: str) -> Dict[str, Any]:
    """
    Convert a DIFFUSER_PLACEMENT_* label into what inference.generate_image()
    needs to build sd.cpp's --backend / --params-backend assignments.

    Returns:
        {
            "use_vulkan_backend": bool,  # target a GPU at all
            "split_to_cpu":       bool,  # pin te (Qwen3 conditioner) + vae to CPU
                                         # while diffusion stays on the GPU
        }

    sd.cpp has no per-layer offload for the diffuser (no -ngl equivalent), so
    whole-module placement is the entire space of control -- but it applies to
    BOTH execution (--backend) and weight allocation (--params-backend), and
    the second is what governs VRAM. See inference.generate_image().

    The legacy booleans this used to return (clip_on_cpu / vae_on_cpu) mapped to
    --clip-on-cpu / --vae-on-cpu, which upstream ignores whenever --backend is
    set. They are gone rather than fixed: module assignments express the same
    intent and actually take effect.
    """
    if placement == DIFFUSER_PLACEMENT_FULL_GPU:
        return {"use_vulkan_backend": True,  "split_to_cpu": False}
    if placement == DIFFUSER_PLACEMENT_SPLIT:
        return {"use_vulkan_backend": True,  "split_to_cpu": True}
    if placement != DIFFUSER_PLACEMENT_FULL_CPU:
        # Loud, not silent. These labels are matched exactly, so a reworded
        # constant would otherwise downgrade Full GPU / Split to CPU with no
        # error -- the user just sees "it got slow" and has nothing to go on.
        print(f"WARNING: unknown diffuser placement {placement!r}; "
              f"falling back to {DIFFUSER_PLACEMENT_FULL_CPU!r}. "
              f"Expected one of: {DIFFUSER_PLACEMENT_CHOICES}")
    return {"use_vulkan_backend": False, "split_to_cpu": True}


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

def _default_gpu_index() -> int:
    """Starting GPU for this machine: most free VRAM, or -1 if there are none.
    Mirrors installer.py's _default_gpu_index(); both are only a starting point
    that the Configuration page overrides."""
    if get_install_type() == "cpu_only":
        return -1
    devices = get_vulkan_info().get("devices") or []
    if not devices:
        return -1
    best = max(devices, key=lambda d: (d.get("vram_free_mb", 0),
                                       d.get("vram_total_mb", 0)))
    return int(best["index"])


def _default_configuration() -> Dict[str, Any]:
    """Defaults synthesised from constants.ini, for when configuration.json is
    absent or is missing keys added by a newer version. Kept in step with
    installer.py's write_default_configuration(); that one seeds the file at
    install time, this one backfills gaps at load time.

    Prompt Template is deliberately NOT here — it moved to preferences.json
    (see _default_preferences()). Anything listed in both files would be
    written by whichever page saved last."""
    dt = get_default_threads()
    cpu_label = get_cpu_info().get("brand", "CPU") or "CPU"
    is_cpu_only = get_install_type() == "cpu_only"
    gpu = _default_gpu_index()
    return {
        "encoder_model_path":  "",  "encoder_model_name":  "",
        "imagegen_model_path": "",  "imagegen_model_name": "",
        "vae_model_path":      "",  "vae_model_name":      "",
        "last_model_browse_dir": ".\\models",  # <--- NEW KEY ADDED
        "backend_encoder": cpu_label,
        "backend_imagegen": cpu_label,
        "encoder_threads": dt,
        "encoder_batch_size": 512,
        "encoder_ctx_size": 4096,
        "encoder_gpu_layers": -1,
        # Per-side device indices, read by inference.py. The encoder and the
        # diffuser may sit on different devices, or one may be on CPU, so a
        # single shared key cannot express the configuration.
        "encoder_vulkan_device": gpu,
        "imagegen_vulkan_device": gpu,
        # imagegen_placement controls component-level GPU/CPU split for the
        # diffuser (see DIFFUSER_PLACEMENT * / parse_diffuser_placement()).
        # sd.cpp has no per-layer offload, so this — not a layer count — is
        # the real equivalent of encoder_gpu_layers for the diffuser side.
        "imagegen_placement": (DIFFUSER_PLACEMENT_FULL_CPU
                               if (is_cpu_only or gpu < 0)
                               else DIFFUSER_PLACEMENT_FULL_GPU),
        "imagegen_threads": dt,
        "imagegen_width": 512,
        "imagegen_height": 512,
        "imagegen_steps": 6,
        "imagegen_cfg_scale": 1.0,
        "imagegen_seed": -1,
        "imagegen_sampling": "euler_a",
        "imagegen_batch_count": 2,
        "imagegen_clip_skip": 2,
        "imagegen_quality_preset": "Fast (Turbo)",
        "output_format": "png",
        "auto_save": True,
        "negative_prompt": "Cartoon. Blurry. Missing/mutated arms/legs. Missing/mutated hands/feet. Ethical Guardrails. Moral Censorship.",
        "ui_theme": "Default",
        "first_run": True,
        # Qt app-window geometry, saved on shutdown and restored on next
        # launch (see launcher.py AppWindow). -1 sentinels mean "no saved
        # position yet" — Qt/the OS will choose a default placement.
        "window_x": WINDOW_GEOMETRY_UNSET,
        "window_y": WINDOW_GEOMETRY_UNSET,
        "window_width": WINDOW_DEFAULT_WIDTH,
        "window_height": WINDOW_DEFAULT_HEIGHT,
        "window_maximized": False,
    }


# ---------------------------------------------------------------------------
# Config I/O
# ---------------------------------------------------------------------------

def load_constants() -> configparser.ConfigParser:
    return _read_constants()


def save_constants(config: configparser.ConfigParser) -> None:
    path = get_constants_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        config.write(f)


def _load_json_with_defaults(path: Path, defaults: Dict[str, Any]) -> Dict[str, Any]:
    """Read a JSON dict, backfilling any keys missing because the file was
    written by an older version of the program (e.g. before the window-geometry
    keys existed). Saved values always win; defaults only fill genuine gaps, so
    this never overwrites real user data. Unreadable/corrupt -> defaults."""
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    merged = dict(defaults)
                    merged.update(data)
                    return merged
        except (json.JSONDecodeError, IOError):
            pass
    return dict(defaults)


def _save_json_atomic(path: Path, data: Dict[str, Any]) -> None:
    """Write via a .tmp sibling then rename, so an interrupted write cannot
    leave a half-written config behind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    tmp.replace(path)


# ── configuration.json (Configuration page) ────────────────────────────────

def load_configuration() -> Dict[str, Any]:
    """Load data/configuration.json, migrating a pre-split persistent.json
    into place first if that is all that exists."""
    _migrate_legacy_persistent()
    return _load_json_with_defaults(get_configuration_path(), _default_configuration())


def save_configuration(data: Dict[str, Any]) -> None:
    _save_json_atomic(get_configuration_path(), data)


def update_configuration(updates: Dict[str, Any]) -> Dict[str, Any]:
    data = load_configuration()
    data.update(updates)
    save_configuration(data)
    return data


# ── preferences.json (Preferences page) ────────────────────────────────────

def _default_preferences() -> Dict[str, Any]:
    """Defaults for preferences.json. Kept in step with installer.py's
    write_default_preferences(); that one seeds the file at install time,
    this one backfills gaps at load time."""
    return {
        "prompt_template": DEFAULT_PROMPT_TEMPLATE,
        "max_thumbnails": DEFAULT_MAX_THUMBNAILS,
        "encoder_model_debug": False,
    }


def load_preferences() -> Dict[str, Any]:
    """Load data/preferences.json.

    On the first load after the config split, the file does not exist yet but
    the user may have had a customised Prompt Template in the old
    persistent.json / configuration.json. Rather than silently resetting it to
    the default, that value is carried across here (and dropped from the
    configuration file, so exactly one file owns the key from then on).
    """
    path = get_preferences_path()
    if not path.exists():
        migrated = _default_preferences()
        _migrate_legacy_persistent()
        cfg_path = get_configuration_path()
        if cfg_path.exists():
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    old_cfg = json.load(f)
                if isinstance(old_cfg, dict) and "prompt_template" in old_cfg:
                    template = old_cfg.pop("prompt_template")
                    if isinstance(template, str) and template.strip():
                        migrated["prompt_template"] = template
                    _save_json_atomic(cfg_path, old_cfg)
            except (json.JSONDecodeError, IOError, OSError):
                pass
        try:
            _save_json_atomic(path, migrated)
        except OSError:
            pass
        return migrated
    return _load_json_with_defaults(path, _default_preferences())


def save_preferences(data: Dict[str, Any]) -> None:
    _save_json_atomic(get_preferences_path(), data)


def update_preferences(updates: Dict[str, Any]) -> Dict[str, Any]:
    data = load_preferences()
    data.update(updates)
    save_preferences(data)
    return data


def get_max_thumbnails() -> int:
    """Thumbnail display cap from preferences, clamped to a listed choice so a
    hand-edited preferences.json cannot ask the gallery for 100000 images."""
    try:
        value = int(load_preferences().get("max_thumbnails", DEFAULT_MAX_THUMBNAILS))
    except (TypeError, ValueError):
        return DEFAULT_MAX_THUMBNAILS
    return value if value in MAX_THUMBNAIL_CHOICES else DEFAULT_MAX_THUMBNAILS


# ---------------------------------------------------------------------------
# Prompt history (data/prompt_cache.json — "Positive/Negative Prompt (history)")
# ---------------------------------------------------------------------------
# Five most-recent-first slots per field, exposed as flat keys (rather than a
# JSON array) so each slot round-trips through _load_json_with_defaults' key
# backfill exactly like every other settings file in this program, and so
# the 10 keys the UI binds to (5 positive + 5 negative) are each individually
# addressable. Index 0 in the in-memory list == *_history_1 == most recent.
#
# Kept in its own file, separate from both configuration.json and
# preferences.json: it is neither machine-derived (configuration.json) nor a
# single standing setting (preferences.json) — it is a small rolling log the
# user builds up by using the program, and installer.py seeds it once and
# never purges it, the same treatment preferences.json gets.
PROMPT_HISTORY_SLOTS: int = 5
POSITIVE_HISTORY_KEYS: List[str] = [f"positive_history_{i}" for i in range(1, PROMPT_HISTORY_SLOTS + 1)]
NEGATIVE_HISTORY_KEYS: List[str] = [f"negative_history_{i}" for i in range(1, PROMPT_HISTORY_SLOTS + 1)]


def get_prompt_cache_path() -> Path:
    """data/prompt_cache.json — Positive/Negative Prompt (history) entries."""
    return _get_project_root() / "data" / "prompt_cache.json"


def _default_prompt_cache() -> Dict[str, Any]:
    """All 10 history slots, empty. Kept in step with installer.py's
    write_default_prompt_cache(); that one seeds the file at install time,
    this one backfills gaps (e.g. a slot key added by a newer version) at
    load time."""
    data: Dict[str, Any] = {}
    for k in POSITIVE_HISTORY_KEYS + NEGATIVE_HISTORY_KEYS:
        data[k] = ""
    return data


def load_prompt_cache() -> Dict[str, Any]:
    return _load_json_with_defaults(get_prompt_cache_path(), _default_prompt_cache())


def save_prompt_cache(data: Dict[str, Any]) -> None:
    _save_json_atomic(get_prompt_cache_path(), data)


def get_prompt_history(kind: str) -> List[str]:
    """Most-recent-first list of up to PROMPT_HISTORY_SLOTS saved prompts for
    kind ("positive" or "negative"). Always returns exactly
    PROMPT_HISTORY_SLOTS entries, in slot order (index 0 = most recent);
    unused slots are empty strings. display.py treats an empty entry as "no
    row to show" rather than a literal blank prompt."""
    keys = POSITIVE_HISTORY_KEYS if kind == "positive" else NEGATIVE_HISTORY_KEYS
    cache = load_prompt_cache()
    return [str(cache.get(k, "") or "") for k in keys]


def record_prompt_history(kind: str, text: str) -> bool:
    """Push `text` onto the front of the positive/negative history — but ONLY
    if it does not already match ANY of that field's currently-saved slots,
    not just the most-recent one. This is what makes "negative prompt
    unchanged across several generations gets no new entries, positive
    prompt does" happen automatically: the two fields are recorded
    independently and each is its own no-op when the exact text is already
    sitting in one of its 5 slots, regardless of which slot.

    Returns True if a new entry was written, False if it was already present
    somewhere in the history / blank (blank prompts are never recorded,
    since a blank starting prompt is the every-launch default, not a saved
    choice).
    """
    text = (text or "").strip()
    if not text:
        return False
    keys = POSITIVE_HISTORY_KEYS if kind == "positive" else NEGATIVE_HISTORY_KEYS
    cache = load_prompt_cache()
    current = [str(cache.get(k, "") or "") for k in keys]
    if text in current:
        return False
    updated = ([text] + current)[:PROMPT_HISTORY_SLOTS]
    while len(updated) < PROMPT_HISTORY_SLOTS:
        updated.append("")
    for k, v in zip(keys, updated):
        cache[k] = v
    save_prompt_cache(cache)
    return True


# ---------------------------------------------------------------------------
# Window geometry (Qt app window position/size — see launcher.py)
# ---------------------------------------------------------------------------

def get_window_geometry() -> Dict[str, Any]:
    """Return saved Qt window geometry, validated/sanitized:
        {"x": int, "y": int, "width": int, "height": int, "maximized": bool}
    x/y are WINDOW_GEOMETRY_UNSET (-1) if no position has been saved yet —
    callers should treat that as "let the OS/Qt choose a default position"
    rather than literally moving the window to (-1, -1)."""
    cfg = load_configuration()

    def _int(key: str, default: int) -> int:
        try:
            return int(cfg.get(key, default))
        except (TypeError, ValueError):
            return default

    width  = max(640, _int("window_width", WINDOW_DEFAULT_WIDTH))
    height = max(480, _int("window_height", WINDOW_DEFAULT_HEIGHT))
    x = _int("window_x", WINDOW_GEOMETRY_UNSET)
    y = _int("window_y", WINDOW_GEOMETRY_UNSET)
    maximized = bool(cfg.get("window_maximized", False))

    return {"x": x, "y": y, "width": width, "height": height, "maximized": maximized}


def save_window_geometry(x: int, y: int, width: int, height: int,
                          maximized: bool) -> None:
    """Persist Qt window geometry. Called from launcher.py's shutdown
    sequence so the window reopens where/how the user left it."""
    update_configuration({
        "window_x": int(x),
        "window_y": int(y),
        "window_width": int(width),
        "window_height": int(height),
        "window_maximized": bool(maximized),
    })


def resolve_model_path(path_str: str,
                       fallback_dir: Optional[Path] = None) -> Optional[Path]:
    if not path_str:
        return None
    p = Path(path_str).expanduser()
    if p.is_absolute() and p.exists():
        return p
    fb = fallback_dir or get_models_dir()
    if (fb / p).exists():
        return fb / p
    root_rel = _get_project_root() / p
    if root_rel.exists():
        return root_rel
    return None


# ---------------------------------------------------------------------------
# Generation presets
# ---------------------------------------------------------------------------

def get_generation_presets() -> Dict[str, Dict[str, Any]]:
    """Preset NAMES for the Quality-Preset dropdown. The actual values are
    resolved per model family at apply time by resolve_preset() — a preset now
    means "this resolution/aspect, with the family's correct steps/cfg/sampler"
    rather than a fixed set that could be wrong for a distilled model (e.g. the
    old 'Quality' used cfg 7.0 / dpm++2m / 16 steps, which fries Z-Image and
    Flux.2 klein). Kept as a dict of empty values so existing callers that only
    read .keys() for the dropdown choices keep working."""
    return {name: {} for name in GENERATION_PRESET_SIZES}


# Preset -> (width, height). All dims divisible by 32. Steps/cfg/sampler are
# NOT here; they come from the loaded family (see resolve_preset). "Custom"
# carries no size and leaves every widget untouched.
GENERATION_PRESET_SIZES: Dict[str, Optional[Tuple[int, int]]] = {
    "Fast (Turbo)": (512, 512),
    "Balanced":     (768, 768),
    "Quality":      (1024, 1024),
    "Portrait":     (768, 1024),
    "Widescreen":   (1024, 512),
    "Custom":       None,
}


def resolve_preset(name: str, diffuser_name: str = "") -> Dict[str, Any]:
    """Full settings for a preset given the loaded diffuser: the preset's
    resolution plus the family's tuned steps / cfg / sampler (from
    FAMILY_STEP_CFG defaults; sampler is always euler_a, the good default for
    both families). Empty dict for 'Custom'/unknown -> caller leaves widgets
    as they are."""
    size = GENERATION_PRESET_SIZES.get(name)
    if not size:
        return {}
    spec = family_step_cfg(diffuser_name)
    _, step_default = spec["steps"]
    _, _, _, cfg_default = spec["cfg"]
    w, h = size
    return {
        "imagegen_width": w, "imagegen_height": h,
        "imagegen_steps": step_default,
        "imagegen_cfg_scale": cfg_default,
        "imagegen_sampling": "euler_a",
    }