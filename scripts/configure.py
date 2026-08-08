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
    # flux2-klein-4B "uncensored" text encoder (WeReCooking / Cordux). This IS a
    # Qwen3-4B under the hood -- an abliterated huihui-ai/Qwen3-4B-abliterated
    # (arch qwen3, hidden 2560, 36 layers) republished for Flux.2-klein-4B -- so
    # its specs are identical to the qwen3-4b entry above; it exists only so the
    # name-fallback recognises the flux2-klein-*-uncensored FILENAME, which the
    # qwen3-4b pattern misses (no "qwen" token). The other in-the-wild filename,
    # qwen3-4b-abl-q#.gguf, already matches the qwen3-4b entry.
    #
    # The pattern is anchored on BOTH "4b" AND "uncensored" on purpose:
    #   * "uncensored" (with "klein") keeps it off the real klein DIFFUSERS,
    #     which are never uncensored -- matching ENCODER_NAME_PATTERNS above.
    #   * "4b" keeps it from ever claiming Qwen3-4B (2560) for a hypothetical
    #     flux2-klein-9B uncensored encoder, which would be a Qwen3-8B (4096) and
    #     must fall through to metadata rather than be mislabelled here. This is
    #     mutually exclusive with every other entry (none carry "klein"), so its
    #     position in the dict does not matter.
    "flux2-klein-4b-encoder": {
        "label": "Qwen3-4B (Flux.2-klein uncensored text encoder)",
        "pattern": r"klein[-_. ]?4b.*?uncensored",
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
DIFFUSER_FAMILY_FLUX1  = "flux1"
DIFFUSER_FAMILY_FLUX2  = "flux2"
DIFFUSER_FAMILY_SDXL   = "sdxl"

# Reference-image input formats sd.cpp can decode. sd.cpp loads -r images via
# stb_image, which handles PNG, JPEG (baseline+progressive), BMP (non-RLE),
# TGA, GIF, PSD, HDR and PNM/PPM/PGM. Every common photo format the user is
# likely to hand us (png/jpg/bmp) is in that set, so NO external converter
# (nconvert etc.) is warranted -- we simply refuse, with a clear message,
# anything stb_image cannot read (WEBP, AVIF, HEIC/HEIF, TIFF, JPEG-2000).
# Note WebP is an OUTPUT-only format in sd.cpp (libwebp encode), not decodable
# as an input here. Extensions are compared lowercased, leading dot included.
SUPPORTED_REF_IMAGE_EXTS = {
    ".png", ".jpg", ".jpeg", ".jfif", ".jpe", ".bmp", ".tga", ".gif",
    ".psd", ".hdr", ".pic", ".pnm", ".ppm", ".pgm",
}

# The JPEG spellings, kept as their own list because the file picker needs
# them as a group and because they are the ones people actually have a mix of
# on disk. stb_image (what sd.cpp decodes with) sniffs the CONTENT and never
# looks at the extension, so .jpeg, .jfif and .jpe are decoded identically to
# .jpg -- the extension list is purely this program's own gate, and leaving a
# spelling out of it rejects a file the backend would have read happily.
JPEG_EXTS = (".jpg", ".jpeg", ".jfif", ".jpe")

# Reference-image handling mode when 2+ images are accumulated in the Flux.2
# "Image Edit" input list (see display.py's ref_images_state / ref_mode_radio
# and inference.generate_image's ref_images handling):
#   REF_MODE_USE_ALL   - all accumulated images are handed to a SINGLE
#                        generation as multiple -r references (sd.cpp's
#                        native multi-reference edit); one run, one batch.
#   REF_MODE_CHAIN_ALL - the list is split and each image is run through its
#                        OWN generation individually (one -r per run), in
#                        sequence; with N images and a batch count of B this
#                        produces N*B images total. Status lines show
#                        "Chain i/N" for whichever run is currently active.
# Only offered as a choice (i.e. the radio is shown at all) once 2+ reference
# images are accumulated -- with 0 or 1 images there is nothing to chain.
REF_MODE_USE_ALL   = "Use All"
REF_MODE_CHAIN_ALL = "Chain All"
REF_MODE_CHOICES   = [REF_MODE_CHAIN_ALL, REF_MODE_USE_ALL]
# Chain All is the default: sd.cpp holds every -r reference in VRAM at once
# for Use All, so as the accumulated image count grows so does the OOM risk
# (especially on a card as small as the RX 470). Chain All only ever holds
# one reference at a time, so it is the safer default; Use All remains a
# deliberate opt-in for when the user actually wants a single multi-
# reference edit.
REF_MODE_DEFAULT   = REF_MODE_CHAIN_ALL

# Encoder hidden_size per Qwen3 size, and the size each diffuser demands. The
# check in inference.check_model_compatibility() is a dimension comparison, not
# a name guess: GGUF `*.embedding_length` (already read by _probe_gguf) gives
# 2560 for Qwen3-4B and 4096 for Qwen3-8B directly.
ENCODER_DIM_QWEN3_4B = 2560
ENCODER_DIM_QWEN3_8B = 4096

# general.architecture (lowercased, exact) -> family. Z-Image and its finetunes
# report `lumina2`.
#
# `flux` is AMBIGUOUS and must never map straight to a family. Klein GGUFs
# report it, and so does every FLUX.1 file -- mhnakif/fluxunchained shows
# Architecture: flux on its HuggingFace metadata panel. A FLUX.1 file resolved
# to flux2 gets driven with --llm <Qwen3> and dies at tensor validation with no
# useful message. Only `flux2` is unambiguous; a bare `flux` falls through to
# the filename tiebreak in _flux_arch_tiebreak().
DIFFUSER_FLUX2_ARCH:  List[str] = ["flux2"]
DIFFUSER_FLUX_AMBIGUOUS_ARCH: List[str] = ["flux"]
DIFFUSER_ZIMAGE_ARCH: List[str] = ["lumina2"]
# SDXL and the SD-classic checkpoints. Unambiguous: nothing else reports these,
# so arch alone settles the family and no filename test is needed when the GGUF
# metadata is readable.
DIFFUSER_SDXL_ARCH:   List[str] = ["sdxl", "sd1", "sd2"]

# Filename/folder patterns (re.search, lowercased) for when arch is unreadable.
# Flux2 is checked FIRST: a "flux-2-klein" name must not fall through to the
# z-image bucket. These are a strict subset of DIFFUSION_NAME_PATTERNS split by
# family -- the "flux"/"klein" tokens go to flux2, everything Z-Image to zimage.
# The "2" is guarded against a following digit or dot so that
# "Persephone-Flux-2.0" -- a FLUX.1-dev finetune whose author versioned it 2.0
# -- is not claimed by flux2. "flux-2-klein" and "flux2-dev" still match.
DIFFUSER_FLUX2_NAME_PATTERNS: List[str] = [
    r"flux[-_. ]?2(?![.\d])",
    r"flux[-_. ]?klein",
    r"\bklein\b",
]
# Tested BEFORE the flux2 patterns, because the discriminating token in a
# FLUX.1 filename ("schnell", "persephone") is more specific than the generic
# "flux" both families share. kontext is listed here as a FLUX.1 derivative;
# note it wants the ref path rather than init, which the spec does not yet
# distinguish -- a Kontext file will currently be driven as plain img2img.
DIFFUSER_FLUX1_NAME_PATTERNS: List[str] = [
    r"flux[-_. ]?1",
    r"\bschnell\b",
    r"\bkontext\b",
    r"fluxed[-_. ]?up",
    r"flux[-_. ]?unchained",
    r"persephone",
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
    DIFFUSER_FAMILY_FLUX1:  "Flux.1",
    DIFFUSER_FAMILY_FLUX2:  "Flux.2-Klein",
    DIFFUSER_FAMILY_SDXL:   "SDXL",
}

# What each family's VAE box should say it wants, shown as the box's info hint.
DIFFUSER_VAE_HINTS: Dict[str, str] = {
    DIFFUSER_FAMILY_ZIMAGE: "Z-Image expects ae.safetensors (the Flux.1 VAE).",
    DIFFUSER_FAMILY_FLUX1:  ("Flux.1 expects ae.safetensors -- the SAME file "
                             "Z-Image uses, verified working on this build."),
    DIFFUSER_FAMILY_FLUX2:  ("Flux.2 expects flux2_ae "
                             "(often downloaded as diffusion_pytorch_model.safetensors)."),
    DIFFUSER_FAMILY_SDXL:   ("SDXL expects an SDXL VAE. sdxl_vae-fp16-fix "
                             "avoids black outputs; xlVAEC_c91 gives richer colour."),
}


# ---------------------------------------------------------------------------
# Per-family capability matrix -- the single source of truth for how each
# family is driven on the sd-cli command line. Adding a family means adding an
# entry here plus the matching branch in inference.generate_image().
#
#   model_flag     : "-m" for a FULL checkpoint bundling UNet + text encoders +
#                    VAE; "--diffusion-model" for standalone diffusion weights
#                    whose encoders must be supplied separately.
#   text_encoders  : which encoder flags this family needs.
#                      "llm"    -> --llm <qwen3 gguf>     (Z-Image, Flux.2)
#                      "clip_l" -> --clip_l <safetensors> (split SDXL)
#                      "clip_g" -> --clip_g <safetensors> (split SDXL)
#                    An EMPTY tuple means the encoders live inside the -m
#                    checkpoint and no flag is passed.
#   vae_required   : whether a run is refused outright without a --vae.
#   clip_skip      : whether --clip-skip means anything (CLIP-conditioned only).
#   img2img        : "init" -> -i <img> + --strength  (standard img2img)
#                    "ref"  -> -r <img>, repeatable   (Flux Kontext editing)
#   encoder_to_cpu : force te=cpu in the backend assignment regardless of the
#                    placement dropdown, for encoders too big to share the card.
# ---------------------------------------------------------------------------
DIFFUSER_FAMILY_SPECS: Dict[str, Dict[str, Any]] = {
    DIFFUSER_FAMILY_ZIMAGE: {
        "model_flag":     "--diffusion-model",
        "text_encoders":  ("llm",),
        "vae_required":   True,
        "clip_skip":      False,
        # None, not "init": this program has always driven Z-Image as
        # text-to-image only, and the input-image column has never been shown
        # for it. sd.cpp would accept -i here, so enabling img2img later is a
        # one-word change to "init" -- but that is a feature addition, not part
        # of SDXL support, so the behaviour is left exactly as it was.
        "img2img":        None,
        "encoder_to_cpu": False,
    },
    DIFFUSER_FAMILY_FLUX2: {
        "model_flag":     "--diffusion-model",
        "text_encoders":  ("llm",),
        "vae_required":   True,
        "clip_skip":      False,
        "img2img":        "ref",
        "encoder_to_cpu": True,
    },
    # FLUX.1 (schnell, dev, and the CivitAI finetunes quantized from them).
    # Standalone diffusion weights plus TWO external encoders -- CLIP-L and
    # T5-XXL -- neither of which is an LLM, so this is the first family to use
    # the clip_l slot without clip_g. Every value below was measured on an
    # RX 470 8GB at 512x512, not assumed:
    #   clip_skip      false. Flux carries no usable CLIP skip.
    #   encoder_to_cpu true, and NOT optional: the two encoders occupy
    #                  3395MB, which would evict the DiT from an 8GB card.
    #   vae_to_cpu     true, and also not optional: the VAE decode compute
    #                  buffer needs 1984MB on the GPU (measured -- it is
    #                  LARGER than the 1664MB it needs on CPU), and the run
    #                  dies at decode_first_stage after paying for a full
    #                  sampling pass. This is the first family to need the
    #                  key, so callers must treat a missing one as False.
    #   img2img        "init". -i + --strength, verified at 0.60 and 0.85.
    DIFFUSER_FAMILY_FLUX1: {
        "model_flag":     "--diffusion-model",
        "text_encoders":  ("clip_l", "t5xxl"),
        "vae_required":   True,
        "clip_skip":      False,
        "img2img":        "init",
        "encoder_to_cpu": True,
        "vae_to_cpu":     True,
    },
    # SDXL as a FULL .safetensors checkpoint (sd_xl_base_1.0.safetensors and
    # the CivitAI finetunes in their original form). One file carries the UNet,
    # both CLIP encoders and a VAE, so nothing external is required.
    DIFFUSER_FAMILY_SDXL: {
        "model_flag":     "-m",
        "text_encoders":  (),
        "vae_required":   False,
        "clip_skip":      True,
        "img2img":        "init",
        "encoder_to_cpu": False,
    },
}

# ---------------------------------------------------------------------------
# SDXL comes in TWO packagings that take DIFFERENT command lines. Both use the
# .gguf extension, so the file extension cannot tell them apart -- there are
# two separate quantizer ecosystems producing SDXL ggufs:
#
#   UNET-ONLY  (city96 / ComfyUI-GGUF lineage: hum-ma, Old-Fisherman,
#              silveroxides, and the WAI / illustrious / pony finetune repos)
#              The quantizers extract the UNet, quantise that alone with
#              city96's lcpp.patch, and ship clip_l / clip_g / vae alongside as
#              separate .safetensors. Old-Fisherman's card states it outright:
#              "Due to the steps required to extract UNet and Clip Encoders and
#              convert the UNet model to a GGUF quantized model".
#              -> --diffusion-model + --clip_l + --clip_g + --vae
#              Markers: general.architecture = "sdxl" IS present, and the file
#              is small -- hum-ma's Q4_0 is 1.49GB, a bare UNet.
#
#   FULL       (stable-diffusion.cpp / llama-box lineage: OlegSkutte, gpustack)
#              One file bundling the UNet, BOTH CLIP encoders and the VAE, made
#              by converting with sd.cpp itself. gpustack's card tabulates the
#              CLIP-L / CLIP-G / VAE quantizations packed inside.
#              -> -m, nothing external needed
#              Markers: no general.architecture key, and much larger --
#              gpustack Q4_0 is 3.94GB, OlegSkutte Q8_0 is 4.1GB.
#
# The DECIDER is whether the user has configured CLIP-L and CLIP-G, not a guess
# about the file. That is deterministic, needs no size thresholds (which would
# overlap across quantizations -- a UNet-only Q8_0 and a full Q4_0 are close in
# size), and it matches intent: supplying the split encoders IS the statement
# that the model is the split kind. File size is used only for the advisory
# label in sdxl_packaging_label(); sdxl_is_unet_only() is what decides.
# ---------------------------------------------------------------------------
SDXL_UNET_ONLY_SPEC: Dict[str, Any] = {
    "model_flag":     "--diffusion-model",
    "text_encoders":  ("clip_l", "clip_g"),
    "vae_required":   True,     # nothing else carries a VAE in this packaging
    "clip_skip":      True,
    "img2img":        "init",
    "encoder_to_cpu": False,    # CLIP-L + CLIP-G together are ~1.6GB; they fit
}

# A bare UNet at Q4_0 is ~1.5GB; a FULL checkpoint starts around 3.9GB because
# the fp16 CLIP-G alone adds 1.4GB. See sdxl_is_unet_only().
SDXL_FULL_CHECKPOINT_MIN_BYTES = 3_500_000_000


def sdxl_packaging_label(path: str, has_clips: bool = False) -> str:
    """One line describing how this SDXL file will be driven, for the
    Configuration page. Empty when the file is not SDXL or not on disk."""
    if not path or not Path(str(path)).exists():
        return ""
    if diffuser_family(path) != DIFFUSER_FAMILY_SDXL:
        return ""
    if sdxl_is_unet_only(path, has_clips):
        return ("This model contains the UNet only, so CLIP-L and CLIP-G must "
                "be supplied separately.")
    return ("This model already contains its own CLIP-L, CLIP-G and VAE, so no "
            "separate CLIP files are needed.")

# Per-family generation defaults for the distilled (fast) and base variants.
# Used by display.py to seed the correct Settings panel and by inference.py as
# fall-backs. Z-Image-Turbo keeps its existing turbo defaults elsewhere.
FLUX2_DISTILLED_DEFAULTS: Dict[str, Any] = {
    "imagegen_cfg_scale": 1.0, "imagegen_steps": 4, "imagegen_sampling": "euler",
}
FLUX2_BASE_DEFAULTS: Dict[str, Any] = {
    "imagegen_cfg_scale": 4.0, "imagegen_steps": 20, "imagegen_sampling": "euler",
}


def normalize_model_name(name: str) -> str:
    """Lower-cased final path component with separators turned into spaces.

    Model filenames separate words with underscores, dots and dashes --
    sd_xl_turbo_1.0.q8_0.gguf, noobaiXL_vPred10.gguf, juggernautXL_v9.safetensors.
    Regex \b does NOT fire next to an underscore, because underscore is a word
    character, so a pattern like \bturbo\b silently fails to match
    "sd_xl_turbo_1.0" -- which would have run SDXL-Turbo at the 30-step,
    cfg-7 base profile instead of its intended 1 step at cfg 1.0.

    Normalizing here fixes every \b-anchored pattern at once instead of
    rewriting each one, and is safe for the separator-tolerant patterns too:
    their character classes are written [-_. ] and already include a space.
    """
    return re.sub(r"[-_.]+", " ", Path(str(name)).name.lower()).strip()


# ---------------------------------------------------------------------------
# Manual family override.
#
# Auto-detection has two inputs -- general.architecture and the filename -- and
# there is a real class of model where BOTH come up empty:
#
#   * sd.cpp-native full checkpoints carry no general.architecture key at all
#     (unlike the city96/ComfyUI UNet-only quants, which do say "sdxl"), and
#   * plenty of SDXL finetunes are named with no "xl" or "sdxl" token anywhere
#     -- artiwaifu-diffusion-v1.q8_0.gguf being an SDXL 1.0 finetune whose name
#     says neither.
#
# Such a model resolves to None and falls back to the Z-Image command line,
# which is silently wrong. Rather than keep widening the filename patterns (an
# endless game, and one that risks false positives on the other families), the
# user can state the answer outright. Auto keeps the existing behaviour.
# ---------------------------------------------------------------------------
# The two SDXL entries also pin the PACKAGING, because that is the other thing
# auto-detection can get wrong and the two questions are answered by the same
# dropdown from the user's point of view ("what is this file?"):
#   full checkpoint -> encoders are inside; -m; CLIP boxes hidden
#   UNet-only       -> encoders are external; --diffusion-model + --clip_l/_g
FAMILY_OVERRIDE_AUTO = "Auto"
FAMILY_OVERRIDE_SDXL_FULL  = "SDXL (all-in-one file)"
FAMILY_OVERRIDE_SDXL_SPLIT = "SDXL (needs CLIP files)"
FAMILY_OVERRIDE_CHOICES: List[str] = [
    FAMILY_OVERRIDE_AUTO, "Z-Image", "Flux.1", "Flux.2",
    FAMILY_OVERRIDE_SDXL_FULL, FAMILY_OVERRIDE_SDXL_SPLIT,
]
_FAMILY_OVERRIDE_MAP: Dict[str, str] = {
    "z-image": DIFFUSER_FAMILY_ZIMAGE,
    "flux.1":  DIFFUSER_FAMILY_FLUX1,
    "flux.2":  DIFFUSER_FAMILY_FLUX2,
    "sdxl":    DIFFUSER_FAMILY_SDXL,
    FAMILY_OVERRIDE_SDXL_FULL.lower():  DIFFUSER_FAMILY_SDXL,
    FAMILY_OVERRIDE_SDXL_SPLIT.lower(): DIFFUSER_FAMILY_SDXL,
}

# Packaging forced by the override, or None to auto-detect. Separate from
# _family_override because "Z-Image" says nothing about SDXL packaging.
_packaging_override: Optional[bool] = None   # True = UNet-only, False = full

# Set from load_configuration() so every diffuser_family() caller honours the
# override without threading an extra argument through a dozen call sites.
_family_override: Optional[str] = None


def set_family_override(value: str) -> None:
    """Record the user's Model Family choice. "" or "Auto" restores detection."""
    global _family_override, _packaging_override
    key = str(value or "").strip().lower()
    _family_override = _FAMILY_OVERRIDE_MAP.get(key)
    if key == FAMILY_OVERRIDE_SDXL_SPLIT.lower():
        _packaging_override = True
    elif key == FAMILY_OVERRIDE_SDXL_FULL.lower():
        _packaging_override = False
    else:
        _packaging_override = None


def get_family_override() -> Optional[str]:
    return _family_override


def _flux_arch_tiebreak(low: str) -> str:
    """Split a bare general.architecture=flux between the two Flux families.

    Called only when the metadata is genuinely ambiguous.

    FLUX.1 tokens are tested FIRST, and the order is load-bearing rather than
    cosmetic. pyys/Persephone-Flux-2.0 is a FLUX.1-dev finetune whose author
    versioned it "2.0", and normalize_model_name() flattens the separators, so
    by the time the patterns run the name reads "flux 2 0" and no lookahead on
    the flux2 pattern can tell that decimal from a real Flux.2. The specific
    token ("persephone") has to win over the generic one. Nothing in the FLUX.1
    list can match a genuine Klein filename, so the reverse risk is nil.

    An unrecognised name defaults to FLUX.1, whose DiT + clip_l + t5xxl layout
    is both the older packaging and by far the more common one on HuggingFace."""
    for pat in DIFFUSER_FLUX1_NAME_PATTERNS:
        if re.search(pat, low):
            return DIFFUSER_FAMILY_FLUX1
    for pat in DIFFUSER_FLUX2_NAME_PATTERNS:
        if re.search(pat, low):
            return DIFFUSER_FAMILY_FLUX2
    return DIFFUSER_FAMILY_FLUX1


def diffuser_family(name: str, arch: str = "") -> Optional[str]:
    """Return one of the DIFFUSER_FAMILY_* constants, or None.

    Architecture metadata wins when present (same policy as classify_model());
    the FILENAME is the fallback. Flux2 is tested before Z-Image.
    `name` may be a bare filename or a full path; matching is case-insensitive.

    Only the final path component is matched, never the directories above it.
    Folder names are not evidence about a model and actively mislead: a user
    who keeps everything in a folder called "SDXL-models-GGUF" (which is what
    hum-ma's repo unzips to) would otherwise have every model inside it read as
    SDXL -- including a Z-Image gguf parked in the same place, which would then
    be driven with --clip-skip and an input-image panel it cannot use.
    """
    # An explicit user choice beats both metadata and filename: it exists
    # precisely for the models where those two disagree or say nothing.
    if _family_override:
        return _family_override

    a = str(arch).strip().lower()
    low = normalize_model_name(name)
    if a:
        if a in DIFFUSER_FLUX2_ARCH:
            return DIFFUSER_FAMILY_FLUX2
        if a in DIFFUSER_FLUX_AMBIGUOUS_ARCH:
            # Both Flux families answer to this one. Metadata has said all it
            # can, so hand the decision to the filename rather than guessing.
            return _flux_arch_tiebreak(low)
        if a in DIFFUSER_ZIMAGE_ARCH:
            return DIFFUSER_FAMILY_ZIMAGE
        if a in DIFFUSER_SDXL_ARCH:
            return DIFFUSER_FAMILY_SDXL
    for pat in DIFFUSER_FLUX1_NAME_PATTERNS:
        if re.search(pat, low):
            return DIFFUSER_FAMILY_FLUX1
    for pat in DIFFUSER_FLUX2_NAME_PATTERNS:
        if re.search(pat, low):
            return DIFFUSER_FAMILY_FLUX2
    # SDXL MUST be tested before Z-Image. DIFFUSER_ZIMAGE_NAME_PATTERNS ends
    # with a bare "turbo", which would otherwise claim sdxl-turbo (and every
    # other *-turbo SDXL distillation) for the Z-Image bucket.
    for pat in DIFFUSER_SDXL_NAME_PATTERNS:
        if re.search(pat, low):
            return DIFFUSER_FAMILY_SDXL
    for pat in DIFFUSER_ZIMAGE_NAME_PATTERNS:
        if re.search(pat, low):
            return DIFFUSER_FAMILY_ZIMAGE
    return None


def sdxl_is_unet_only(path: str, has_clips: bool = False) -> bool:
    """True when an SDXL file needs CLIP-L/CLIP-G supplied separately.

    FILE SIZE is the signal, because it is a property of the file rather than
    of what the user happens to have configured. The two ecosystems are far
    apart: a city96/ComfyUI UNet-only quant is ~1.5GB at Q4_0, while an
    sd.cpp-native full checkpoint starts around 3.9GB -- the fp16 CLIP-G inside
    it alone adds 1.4GB. Nothing in the metadata distinguishes them (the
    UNet-only files say general.architecture=sdxl, the full ones say nothing),
    so size is the best available evidence and the gap is wide enough to be
    reliable across quantizations.

    A .safetensors is assumed to be a full checkpoint: that is what SDXL
    checkpoints are, and the UNet-only ecosystem publishes GGUF only.

    Order of authority: explicit user override, then size, then -- only when
    the file cannot be measured -- whether clips happen to be configured.
    """
    if _packaging_override is not None:
        return _packaging_override
    if not str(path).lower().endswith(".gguf"):
        return False
    try:
        return Path(path).stat().st_size < SDXL_FULL_CHECKPOINT_MIN_BYTES
    except OSError:
        return has_clips


def diffuser_spec(name: str, arch: str = "",
                  has_clips: bool = False) -> Optional[Dict[str, Any]]:
    """The command-line spec for a diffuser, or None if the family is unknown.

    For SDXL the answer depends on PACKAGING rather than family alone, and both
    packagings use the .gguf extension -- so has_clips (whether the user has
    configured CLIP-L and CLIP-G) is what selects between them. Supplying the
    split encoders is the user's statement that this is the split kind;
    supplying none means the encoders are expected to be inside the file.
    See the SDXL_UNET_ONLY_SPEC block above for the two ecosystems.

    None means "cannot tell" -- callers fall back to the Z-Image-style command
    line, which is what this program did before families existed.
    """
    fam = diffuser_family(name, arch)
    if not fam:
        return None
    if fam == DIFFUSER_FAMILY_SDXL and sdxl_is_unet_only(name, has_clips):
        return SDXL_UNET_ONLY_SPEC
    return DIFFUSER_FAMILY_SPECS.get(fam)


def diffuser_family_label(name: str, arch: str = "") -> str:
    """Human label for a diffuser, or 'no model selected' when unrecognised."""
    fam = diffuser_family(name, arch)
    return DIFFUSER_FAMILY_LABELS.get(fam, "no model selected")


def is_flux2_base_variant(name: str) -> bool:
    """True for klein-base checkpoints (higher cfg / more steps), False for the
    distilled klein. `base` appears in the filename of every base variant."""
    return bool(re.search(r"\bbase\b|[-_.]base[-_.]", str(name).lower()))


def is_flux1_schnell_variant(name: str) -> bool:
    """True for the step-distilled FLUX.1 [schnell] lineage, False for the
    dev lineage (which is what every CivitAI finetune this program is likely
    to meet actually is -- fluxedUp, brainflux, and the rest are dev merges).

    DEV IS THE DEFAULT, and deliberately so. Getting it backwards is cheap in
    one direction and expensive in the other: a schnell file misread as dev
    opens at 20 steps and merely wastes time producing an image that was
    already finished at 4, whereas a dev file misread as schnell opens at 4
    steps and produces visible mush that reads as "this model is broken".
    So only an explicit schnell token flips it.

    "merge"/"fusion" files that blend the two lineages (Flux Fusion and
    friends) carry both tokens and land on schnell, which is right: they are
    published as few-step models.

    Uses the normalized name so "flux1_schnell_Q4_0" -- where the token sits
    between two underscores, which \\b cannot see past -- is recognised."""
    return bool(re.search(r"\bschnell\b", normalize_model_name(name)))


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


def vae_families(name: str) -> set:
    """The SET of diffuser families a VAE filename is valid for.

    A set rather than a single family because VAEs are shared: ae.safetensors
    is the Flux.1 autoencoder that Z-Image reuses verbatim, so a one-family
    answer would make correct pairings look like mismatches as soon as another
    family that uses the same file is added. An EMPTY set means the name
    identifies nothing (e.g. Flux.2's generic
    diffusion_pytorch_model.safetensors) -- callers treat that as "cannot
    tell" and allow it through, never as an error.

    Order matters: the SDXL patterns are tested before the Z-Image one because
    "sdxl_vae" would otherwise fall through to the boundaried bare-"ae" test.
    """
    low = normalize_model_name(name)
    for pat in VAE_FLUX2_NAME_PATTERNS:
        if re.search(pat, low):
            return {DIFFUSER_FAMILY_FLUX2}
    for pat in VAE_SDXL_NAME_PATTERNS:
        if re.search(pat, low):
            return {DIFFUSER_FAMILY_SDXL}
    for pat in VAE_ZIMAGE_NAME_PATTERNS:
        if re.search(pat, low):
            # TWO families, which is the case this function was built as a set
            # for. ae.safetensors is the Flux.1 autoencoder and Z-Image reuses
            # it verbatim, so it is correct for both and the property of
            # neither. Returning one family made vae_family() name that one,
            # and the cross-family check in display._resolve_vae then blanked a
            # perfectly good ae.safetensors the moment a FLUX.1 model was
            # picked. Two entries make vae_family() return None -- "no single
            # right answer" -- which is exactly right, while inference.py's
            # mismatch check tests membership rather than equality and so
            # keeps passing for both.
            return {DIFFUSER_FAMILY_ZIMAGE, DIFFUSER_FAMILY_FLUX1}
    return set()


def vae_family(name: str) -> Optional[str]:
    """Single-family form of vae_families(), for the display.py callers that
    want one answer. Returns None when the name identifies nothing OR when it
    is valid for more than one family (i.e. "no single right answer")."""
    fams = vae_families(name)
    return next(iter(fams)) if len(fams) == 1 else None


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
    # flux2-klein "uncensored" TEXT ENCODER (WeReCooking / Cordux). It is a
    # Qwen3-4B encoder (general.architecture=qwen3) republished for
    # Flux.2-klein-4B, but its bare FILENAME -- flux2-klein-4b-uncensored-q#.gguf
    # -- carries the flux2/klein tokens and NOT "qwen" or "encoder", so nothing
    # above catches it and it would otherwise fall through to the "flux"
    # DIFFUSION pattern and be misfiled as a diffuser. "klein" + "uncensored" is
    # the discriminator: the real klein DIFFUSERS (flux-2-klein-4b*,
    # flux-2-klein-base-4b*) are never "uncensored". Metadata still wins in
    # classify_model() -- arch=qwen3 already classifies a readable copy as an
    # encoder -- so this is purely the name-fallback for an unreadable-metadata
    # file, bringing this encoder to the same fallback parity as every other.
    # (The sibling filename qwen3-4b-abl-q#.gguf is already caught by the qwen
    # pattern above.)
    r"klein.*?uncensored",
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
    # The full "stable-diffusion-xl" spelling, which is how the official
    # checkpoints and most GGUF conversions of them are named
    # (stable-diffusion-xl-base-1.0-Q4_0.gguf). It is NOT caught by the
    # "sd?xl" pattern above, nor by the "...XL" suffix pattern below, because
    # the xl there is preceded by a hyphen rather than an alphanumeric.
    r"diffusion[-_. ]?xl",
    r"illustrious",
    # No trailing \b: the finetunes are named ponyDiffusionV6XL, ponyRealism
    # and so on, where "pony" runs straight into the next word and a closing
    # boundary can never match.
    r"\bpony",
    r"noobai",
    # The "...XL" naming convention that most SDXL finetunes follow --
    # juggernautXL, dreamshaperXL, realvisXL, animagineXL, ponyDiffusionV6XL,
    # and the separator-using variants like zavychroma5-xl. The separator is
    # optional so both "juggernautXL" and "zavychroma5-xl" match, but there
    # must be SOMETHING before the xl -- a bare leading "xl" is too weak a
    # signal on its own. The trailing test is an explicit non-alphanumeric-or-
    # end rather than \\b, for the same reason as VPRED_NAME_PATTERN: underscore
    # is a word character, so \\b would fail on juggernautXL_v9, which is the
    # single most common way these are named.
    r"[a-z0-9][-_. ]?xl(?:[^a-z0-9]|$)",
]

SD_CLASSIC_ARCHITECTURES: List[str] = ["sd1", "sd2", "sdxl"]

# The SDXL family name-fallback reuses the SD-classic token set rather than
# keeping a second copy: the models that want --clip-skip and the models that
# ARE SDXL are the same set, so two lists would only drift apart. Defined here
# rather than beside the other DIFFUSER_*_NAME_PATTERNS because it depends on
# SD_CLASSIC_NAME_PATTERNS, which is defined above; diffuser_family() resolves
# the name at call time, so the ordering within this module is not a problem.
DIFFUSER_SDXL_NAME_PATTERNS: List[str] = SD_CLASSIC_NAME_PATTERNS

# SDXL VAE filenames. Checked before the bare-"ae" Z-Image pattern in
# vae_families(). xlVAEC is the improved community VAE hum-ma ships alongside
# their GGUF quants and recommends over the base one.
VAE_SDXL_NAME_PATTERNS: List[str] = [
    r"sdxl.*?vae",
    r"vae.*?sdxl",
    r"sd[-_. ]?xl[-_. ]?vae",
    r"xlvae",
]

# Step-DISTILLED SDXL derivatives. These behave like Z-Image-Turbo and Klein
# distilled -- a handful of steps at cfg near 1.0 -- and NOT like SDXL base
# (30 steps, cfg 7). Running one at the base defaults produces burnt,
# over-saturated output, which is the most common "this model is broken"
# report for the class. "dmd2" is Distribution Matching Distillation v2, which
# several community finetunes are built on and which is named only in the
# filename -- there is no metadata flag for any of these.
SDXL_DISTILLED_NAME_PATTERNS: List[str] = [
    r"\bturbo\b",
    r"\blightning\b",
    r"\bhyper\b",
    r"\bdmd2?\b",
    r"\blcm\b",
]


def is_sdxl_distilled_variant(name: str) -> bool:
    """True for Turbo / Lightning / Hyper / DMD2 / LCM SDXL derivatives, which
    take few-step, low-cfg settings instead of the SDXL base defaults.

    Uses the normalized name so that sd_xl_turbo_1.0.q8_0.gguf -- where the
    word "turbo" sits between two underscores -- is recognised."""
    low = normalize_model_name(name)
    return any(re.search(p, low) for p in SDXL_DISTILLED_NAME_PATTERNS)


# v-prediction override. The GGUF conversion drops whatever marked a checkpoint
# as v-prediction (Old-Fisherman's card notes ComfyUI needs a custom node to
# force it back), so sd.cpp assumes eps and the output comes out washed out or
# structurally wrong. The filename is the only surviving signal. "Auto" applies
# the flag when the name says vpred and otherwise lets sd.cpp decide.
#
# The leading test is (?:^|[^a-z0-9]) rather than \b because underscore IS a
# word character: \bv[-_]?pred cannot match noobaiXL_vPred10, which is exactly
# how these files are usually named.
# ---------------------------------------------------------------------------
# Prompt-enhancement budget, per family.
#
# The enhancer is a Qwen3 gguf run through llama.cpp that rewrites the user's
# prompt before sd-cli sees it. How much text is USEFUL depends entirely on
# what will read it downstream:
#
#   SDXL      conditions through CLIP-L + CLIP-G, whose context is 77 tokens.
#             Anything past that is chunked at best and dropped at worst, so a
#             200-word cinematic paragraph mostly evaporates -- and what
#             survives is the enhancer's scene-setting preamble rather than the
#             user's actual subject. SDXL also responds better to dense
#             comma-separated keywords than to flowing prose. 77 CLIP tokens is
#             roughly 55-60 English words; max_tokens is the llama.cpp -n
#             budget, which is a different tokenizer, so it is set with slack
#             and the word count is enforced in the system prompt as well.
#
#   FLUX.1    conditions through CLIP-L (77 tokens, effectively a throwaway
#             pooled vector here) plus T5-XXL, and sd.cpp compiles the T5
#             stream at 512 tokens for dev-lineage files and 256 for schnell.
#             512 T5 tokens is roughly 350-380 English words, so the generous
#             1200-char default is close enough to the ceiling that a chatty
#             enhancer can push the tail of the prompt past it -- and what
#             gets dropped is the END of the text, which is where the
#             enhancer puts its lighting/atmosphere detail. 800 chars keeps
#             the whole rewrite inside the window with room to spare, and
#             still leaves FLUX.1 far more budget than SDXL's 320. Prose, not
#             keywords: T5 is a real language model and FLUX.1 was trained on
#             natural-language captions, so the SDXL keyword-list style would
#             actively hurt here.
#
#   Z-Image   condition through a Qwen3 LLM with a real context window, so long
#   Flux.2    descriptive prose genuinely helps and is what these were tuned
#             for. Keep the original generous budget.
#
# max_chars is a final backstop applied after generation: a model that ignores
# the word-count instruction still cannot overflow the downstream encoder.
# ---------------------------------------------------------------------------
PROMPT_ENHANCE_DEFAULTS: Dict[str, Any] = {
    "max_tokens": 256,
    "max_chars":  1200,
    "style": ("Be descriptive about style, lighting, composition, colors, and "
              "atmosphere."),
}

FAMILY_PROMPT_ENHANCE: Dict[str, Dict[str, Any]] = {
    DIFFUSER_FAMILY_SDXL: {
        "max_tokens": 96,
        "max_chars":  320,
        "style": ("Write at most 55 words as a dense, comma-separated list of "
                  "visual keywords covering subject, setting, lighting and "
                  "style. Put the main subject FIRST. No sentences, no "
                  "narration, no preamble."),
    },
    DIFFUSER_FAMILY_FLUX1: {
        "max_tokens": 200,
        "max_chars":  800,
        "style": ("Write flowing natural-language prose, at most 120 words. "
                  "Be descriptive about subject, style, lighting, composition, "
                  "colors, and atmosphere. No keyword lists, no preamble."),
    },
    DIFFUSER_FAMILY_ZIMAGE: dict(PROMPT_ENHANCE_DEFAULTS),
    DIFFUSER_FAMILY_FLUX2:  dict(PROMPT_ENHANCE_DEFAULTS),
}


def prompt_enhance_spec(diff_path: str, diff_arch: str = "") -> Dict[str, Any]:
    """Enhancement budget for the diffuser in use. Unknown models get the
    permissive default, matching how they are treated everywhere else."""
    fam = diffuser_family(diff_path, diff_arch) if diff_path else None
    return FAMILY_PROMPT_ENHANCE.get(fam, PROMPT_ENHANCE_DEFAULTS)


def trim_to_budget(text: str, max_chars: int) -> str:
    """Clean up an enhancer's output: cut it to max_chars, and drop a trailing
    fragment left behind when llama.cpp stopped at its -n token budget.

    TWO separate problems, both seen in practice:

      1. Too long. Cut to max_chars on a clause boundary, never mid-word.

      2. Truncated mid-thought, but still UNDER max_chars -- which is the
         common case once -n is small. The observed example ended
         "...No other people, animals, or distractions-total", where "total"
         is the start of a word the model never finished. Length-trimming
         alone does nothing here because the text already fits.

    For (2) the tell is that the text does not end on sentence punctuation.
    Prose that was allowed to finish ends with . ! or ? -- prose that was cut
    off does not. So when the ending is unterminated, fall back to the last
    complete clause. A keyword list (SDXL style) legitimately ends without a
    full stop, and there the same rule just drops the final partial keyword,
    which is the desired behaviour too.

    Both steps refuse a break point that would discard more than half the
    text, so a stray early comma cannot gut a whole prompt.
    """
    t = " ".join(str(text).split())
    if not t:
        return t

    def _cut_back(s: str, floor: int) -> str:
        """Trim s back to the last sentence end, else the last clause end,
        keeping at least `floor` characters. Returns s unchanged if no
        acceptable break point exists."""
        for seps, keep in ((".!?", True), (",;:", False), (" ", False)):
            idx = max(s.rfind(c) for c in seps)
            if idx > floor:
                return s[:idx + 1] if keep else s[:idx].rstrip(" ,;:-—–")
        return s

    # (1) length
    if len(t) > max_chars:
        t = _cut_back(t[:max_chars], max_chars * 0.5)

    # (2) unterminated tail -- but ONLY when the output came close to the
    # budget. llama.cpp truncates when it hits -n, which produces text at or
    # near the cap; an output well short of it stopped because the model was
    # finished, and its last word is a real one. Without this proportion test
    # the rule would eat the final word of every short, complete prompt --
    # "a man walking his dog on a meadow" would lose "meadow", and a finished
    # keyword list would lose its last keyword.
    if t and t[-1] not in ".!?" and len(t) >= max_chars * 0.6:
        t = _cut_back(t, len(t) * 0.5)

    return t.strip()


PREDICTION_AUTO = "Auto"
PREDICTION_CHOICES: List[str] = [PREDICTION_AUTO, "eps", "v", "edm_v"]
VPRED_NAME_PATTERN = r"(?:^|[^a-z0-9])v[-_]?pred"


def sdxl_prediction_override(name: str, setting: str = PREDICTION_AUTO) -> str:
    """The value for sd-cli --prediction, or "" to omit the flag entirely."""
    s = (setting or PREDICTION_AUTO).strip()
    if s and s != PREDICTION_AUTO:
        return s
    return "v" if re.search(VPRED_NAME_PATTERN, normalize_model_name(name)) else ""


# Safetensors text encoders (the split-SDXL clip_l / clip_g pair). These carry
# no architecture field at all, so the filename is the only oracle available.
# Consumed by inference.classify_model() so they land in the encoder bin rather
# than "unknown", which is what kept them out of every picker.
TEXT_ENCODER_NAME_PATTERNS: List[str] = [
    r"\bclip[-_. ]?[lg]\b",
    r"clip_[lg]",
    r"\bt5[-_. ]?xxl\b",
    r"\bumt5\b",
]

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
    "dpm++2mv2": "dpm++2mv2", "lcm": "lcm",
    # sd.cpp's sampler list has no plain "ddim" and no "plms" at all -- the
    # accepted values are euler, euler_a, heun, dpm2, dpm++2s_a, dpm++2m,
    # dpm++2mv2, ipndm, ipndm_v, lcm, ddim_trailing, tcd, res_multistep,
    # res_2s, er_sde, euler_cfg_pp, euler_a_cfg_pp. Selecting the old "ddim"
    # or "plms" handed sd-cli a value it rejects, so "ddim" now maps to the
    # real flag and "plms" is gone.
    "ddim": "ddim_trailing", "ddim_trailing": "ddim_trailing",
    "ipndm": "ipndm", "tcd": "tcd", "res_multistep": "res_multistep",
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

# FLUX.1 (dev, schnell, and the CivitAI finetunes quantized from them, e.g.
# fluxedUp). Trained at 1024 and, like Flux.2, it does NOT degrade gracefully
# downwards -- BFL document the supported band as 0.1-2.0 MP, whose lower end
# is ~320x320, and in practice anything under 512 loses coherence rather than
# just detail. So the floor is 512, matching Flux.2 rather than Z-Image's 256.
# The 2048 ceiling is 4MP, above BFL's documented 2MP recommendation but
# reachable on a big card; VRAM, not this list, is the real limit (a Q4_0
# FLUX.1 DiT is ~6.5GB before the compute buffer -- see
# inference._flux1_needs_offload, which measured 1366 MB per megapixel).
FLUX1_IMAGE_SIZES  = [512, 768, 1024, 1280, 1536, 1792, 2048]

# SDXL was trained at roughly 1MP across a set of aspect buckets, with
# 1024x1024 native. The off-square bucket sizes here (832/896/1152/1216/1344)
# are all multiples of 64, so sd.cpp will not silently round them (see the
# IMAGE_SIZES note above). Below 512 SDXL degrades badly -- it is not an SD1.5
# and does not tolerate 256 -- so the floor is 512.
SDXL_IMAGE_SIZES = [512, 768, 832, 896, 1024, 1152, 1216, 1344, 1536]

FAMILY_IMAGE_SIZES: Dict[str, List[int]] = {
    "flux1_dev":       FLUX1_IMAGE_SIZES,
    "flux1_schnell":   FLUX1_IMAGE_SIZES,
    "flux2_distilled": FLUX2_IMAGE_SIZES,
    "flux2_base":      FLUX2_IMAGE_SIZES,
    "sdxl_base":       SDXL_IMAGE_SIZES,
    "sdxl_distilled":  SDXL_IMAGE_SIZES,
    "z-image":         ZIMAGE_IMAGE_SIZES,
    "none":            IMAGE_SIZES,
}


def family_image_sizes(diffuser_name: str) -> List[int]:
    """Allowed width/height values for the loaded diffuser's family."""
    return FAMILY_IMAGE_SIZES.get(family_step_cfg_key(diffuser_name), IMAGE_SIZES)

# Diffuse Steps: ONE list, ONE default, for every family.
#
# This used to be per-family -- FLUX.1 dev opened at 20 with a floor of 16,
# SDXL base at 30, the distilled models at 4-8. That was "optimal" in the sense
# a model card means it, and wrong in the sense that matters here: on the
# hardware this program targets, 20 steps at 512x512 is minutes of waiting
# before you learn whether the prompt was any good, and a floor of 16 removed
# the ability to find out faster. A default nobody will sit through is not a
# good default.
#
# 8 is the compromise, everywhere. It is enough for a usable image out of every
# family this program drives -- generous for the distilled models (Z-Image-
# Turbo, klein, schnell, SDXL-Turbo, which are trained for 4), and soft but
# legible for the undistilled ones (FLUX.1 dev, klein base, SDXL base). It
# lands in a reasonable time on an 8GB card, which is the property being
# optimised for. Anyone who wants the model's full quality raises it, and the
# list goes to 32 so they can.
#
# The list spans both regimes deliberately: 4/6 for few-step models, 8-12 as
# the general working band, 16-32 for the undistilled families at full quality.
STEP_CHOICES      = [4, 6, 8, 10, 12, 16, 20, 24, 28, 32]

# The single step default, used by every entry in FAMILY_STEP_CFG below.
DEFAULT_STEPS     = 8
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
    # ── FLUX.1 ───────────────────────────────────────────────────────────
    # Split in two, because dev and schnell are as far apart as klein-base and
    # klein-distilled are, and a single "flux1" entry would be wrong for one of
    # them. Both are GUIDANCE-distilled, which is the part that trips people up:
    # --cfg-scale must stay at 1.0 for either. Real classifier-free guidance was
    # trained out of both, so raising cfg does not strengthen the prompt, it
    # burns the image AND doubles the step time (two forward passes instead of
    # one). What dev *does* consume is the separate embedded guidance value,
    # which sd-cli already defaults to 3.50 and this program deliberately does
    # not pass (see the is_flux1 branch in inference.generate_image). Because
    # cfg is pinned at 1.0, the NEGATIVE PROMPT does nothing for either variant
    # -- it is only read when cfg > 1.0.
    #
    # STEPS are no longer split per family -- every entry below takes the same
    # STEP_CHOICES list and the same DEFAULT_STEPS of 8; see the note there for
    # why. What still differs, and what these entries exist for, is CFG.
    #
    #   dev      : covers fluxedUp and effectively every other CivitAI FLUX.1
    #              finetune, since they are near-universally dev-lineage merges.
    #              Raise Diffuse Steps toward 20-28 for its full quality; 8 gets
    #              a legible image in a fraction of the time. NOTE the model's
    #              own card may recommend dpm++2m rather than euler -- the
    #              Sampler dropdown is deliberately never overridden by family
    #              selection, so that stays the user's call.
    #   schnell  : step-distilled, trained for 4. 8 costs a little time for
    #              nothing but does not degrade it.
    "flux1_dev": {
        "steps":  (STEP_CHOICES, DEFAULT_STEPS),
        "cfg":    (1.0, 1.5, 0.5, 1.0),
    },
    "flux1_schnell": {
        "steps":  (STEP_CHOICES, DEFAULT_STEPS),
        "cfg":    (1.0, 1.5, 0.5, 1.0),
    },
    "flux2_distilled": {
        "steps":  (STEP_CHOICES, DEFAULT_STEPS),
        "cfg":    (1.0, 2.0, 0.5, 1.0),
    },
    "flux2_base": {
        "steps":  (STEP_CHOICES, DEFAULT_STEPS),
        "cfg":    (1.0, 8.0, 0.5, 4.0),
    },
    # Z-Image-Turbo (distilled S3-DiT): 8-step model — 8 is the sweet spot,
    # 10-12 rarely helps (try a new seed instead). cfg stays low (~1.0-2.0);
    # 4+ "fries"/over-saturates a distilled model, so the range caps at 4.0.
    # Sampler is NOT forced (euler AND euler_a both work well for Z-Image),
    # unlike Flux.2 which must be euler.
    "z-image": {
        "steps":  (STEP_CHOICES, DEFAULT_STEPS),
        "cfg":    (1.0, 4.0, 0.5, 1.0),
    },
    # SDXL base and ordinary (non-distilled) finetunes: a conventional
    # classifier-free-guidance model, unlike everything else this program
    # drives. 25-30 steps at cfg 6-8 is the standard band, and the negative
    # prompt actually does something here because cfg is above 1.0.
    "sdxl_base": {
        "steps":  (STEP_CHOICES, DEFAULT_STEPS),
        "cfg":    (1.0, 12.0, 0.5, 7.0),
    },
    # Distilled SDXL: Turbo, Lightning, Hyper, DMD2, LCM. Step-distilled the
    # same way Klein and Z-Image-Turbo are, and wanting the same treatment --
    # a few steps at cfg near 1.0. See is_sdxl_distilled_variant().
    "sdxl_distilled": {
        "steps":  (STEP_CHOICES, DEFAULT_STEPS),
        "cfg":    (1.0, 3.0, 0.5, 1.5),
    },
    # No diffuser chosen yet: keep the permissive superset.
    "none": {
        "steps":  (STEP_CHOICES, DEFAULT_STEPS),
        "cfg":    (0.5, 20.0, 0.5, 1.0),
    },
}


def family_step_cfg_key(diff_path: str, diff_arch: str = "") -> str:
    """Map a diffuser to its FAMILY_STEP_CFG key: 'flux1_dev',
    'flux1_schnell', 'flux2_distilled', 'flux2_base', 'sdxl_base',
    'sdxl_distilled', 'z-image', or 'none'."""
    fam = diffuser_family(diff_path, diff_arch) if diff_path else None
    if fam == DIFFUSER_FAMILY_FLUX1:
        return "flux1_schnell" if is_flux1_schnell_variant(diff_path) else "flux1_dev"
    if fam == DIFFUSER_FAMILY_FLUX2:
        return "flux2_base" if is_flux2_base_variant(diff_path) else "flux2_distilled"
    if fam == DIFFUSER_FAMILY_SDXL:
        return "sdxl_distilled" if is_sdxl_distilled_variant(diff_path) else "sdxl_base"
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
# GLAMOUR MODE -- the streamlined single-purpose pipeline
# ---------------------------------------------------------------------------
# This program has one job: take a photo of a person and place that person at
# three fixed landmarks. What lives in this section is the tables and choice
# lists that several modules need to agree on.
#
# THE PINNED VALUES ARE NOT HERE. Sampler, seed, batch count and CFG are
# hard-coded as literals inside display.do_generate(), the one function that
# sends them, and written into generation.json from there so the Debug page
# can report what actually ran by reading that file back.
#
# They were constants in this file first, and that was the wrong shape: a
# named module-level constant looks like a dial, and a dial attracts a second
# reader. Two readers of a value nothing may change is two places to keep in
# step for no benefit. What remains here are the things that ARE choices --
# the step list, the format, the subject and scene tables -- because those
# genuinely are read from several places and need one home.
GLAMOUR_OUTPUT_FORMAT: str  = "jpg"

# Diffuse Steps -- the one numeric the user still chooses. 6 is the default
# because Klein distilled is a 4-step model: 4-6 is its ideal band, 8 is the
# practical ceiling and past that quality DROPS rather than plateaus. 10 and
# 12 are offered because a klein-base model wants more, not because the
# distilled one benefits.
GLAMOUR_STEP_CHOICES: List[int] = [6, 8, 10, 12]
GLAMOUR_DEFAULT_STEPS: int      = 6

# Subject Count -> output resolution. Framing follows headcount: one person
# wants height, a crowd wants width. The pixel budget is held roughly constant
# across all three (393k / 262k / 393k) so switching framing does not quietly
# turn a run that fits in VRAM into one that does not.
SUBJECT_ONE:   str = "1 Person"
SUBJECT_TWO:   str = "2 People"
SUBJECT_MANY:  str = "3+ People"
SUBJECT_COUNT_CHOICES: List[str] = [SUBJECT_ONE, SUBJECT_TWO, SUBJECT_MANY]
SUBJECT_COUNT_DEFAULT: str = SUBJECT_ONE
SUBJECT_COUNT_SIZES: Dict[str, tuple] = {
    SUBJECT_ONE:  (512, 768),   # portrait
    SUBJECT_TWO:  (512, 512),   # square
    SUBJECT_MANY: (768, 512),   # landscape
}

# Subject Gender -- the second half of the subject description, and the only
# TOKEN substitution in the whole prompt system.
#
# WHY IT EXISTS. Flux.2 reads the reference photo for identity but takes its
# cue on presentation from the text, and with the text silent it guesses from
# hair and build. A long-haired man comes back in a dress. Saying the word in
# the prompt costs nothing and settles it.
#
# WHY IT IS A TOKEN AND THE SUBJECT CLAUSE IS NOT. The count and the gender
# are two independent axes -- 3 counts x 3 gender settings is 9 combinations --
# and one clause per combination would be nine boxes on the Preferences page
# that all say almost the same thing. So the count still picks which clause
# runs (three boxes, unchanged) and the gender is substituted INTO it at the
# <gender> mark. That is a deliberate exception to the "no placeholders" rule
# the scene prompts follow, and it is safe for the opposite reason the scene
# prompts are: a scene prompt that loses its subject still renders something
# plausible and wrong, whereas a clause that loses <gender> is caught on save
# and reported (see display.save_prefs) because the token is a fixed literal
# that can be searched for.
GENDER_MALE:   str = "Male(s)"
GENDER_MIXED:  str = "M+F(s)"
GENDER_FEMALE: str = "Female(s)"
GENDER_CHOICES: List[str] = [GENDER_MALE, GENDER_MIXED, GENDER_FEMALE]

# Male(s), not M+F(s), is the shipped default -- NOT a statement about who is
# expected to use this, but because M+F(s) at 1 Person resolves to
# "transgender", and a factory default should not describe the subject of an
# unread photo that specifically. Male(s) at 1 Person resolves to "male",
# which is the mildest of the three wrong answers if the operator forgets to
# set the switch, and the switch sits directly under Subject Count where it
# cannot be missed.
GENDER_DEFAULT: str = GENDER_MALE

# The literal mark the substitution looks for, inside the subject clauses.
# Angle brackets because nothing else in a prose prompt uses them, so a
# search for it cannot hit ordinary prompt text. Matched case-insensitively
# at substitution time, so <Gender> and <GENDER> both work.
GENDER_TOKEN: str = "<gender>"

# (gender setting, subject count) -> the word that replaces <gender>.
#
# Nine cells, seven distinct words: "males" covers 2 People and 3+ People
# alike, as does "females", because English does not distinguish two from many
# and inventing a distinction ("a pair of males") would read as instruction to
# the model rather than description. The mixed row is the one that genuinely
# needs all three: one person of mixed gender is not a group, it is a single
# individual, and "transgender" is the word that describes them.
GENDER_WORDS: Dict[str, Dict[str, str]] = {
    GENDER_MALE: {
        SUBJECT_ONE:  "male",
        SUBJECT_TWO:  "males",
        SUBJECT_MANY: "males",
    },
    GENDER_MIXED: {
        SUBJECT_ONE:  "transgender",
        SUBJECT_TWO:  "male and female",
        SUBJECT_MANY: "males and females",
    },
    GENDER_FEMALE: {
        SUBJECT_ONE:  "female",
        SUBJECT_TWO:  "females",
        SUBJECT_MANY: "females",
    },
}

# Subject Bodyshape -- the second token, and the second thing Flux.2 invents
# when the prompt does not say.
#
# WHY IT EXISTS, and why it is not just more prompt text: Flux.2 has a strong
# prior toward gym-fit bodies. Left unsaid, a slight person comes back visibly
# broader than they are, which is the same class of error as the long-haired
# man in a dress -- the model filling a silence with its training average
# rather than reading the photo. Naming the shape closes it.
#
# UNLIKE GENDER, THIS DOES NOT VARY WITH SUBJECT COUNT. "A bodyshape of
# skinny" reads the same whether it is describing one person or four, so the
# table is a flat label -> word map rather than a grid. One shape applies to
# the whole batch, the same way one gender does; a photo whose subjects differ
# wants two runs.
BODYSHAPE_SKINNY:      str = "Skinny"
BODYSHAPE_SLIM:        str = "Slim"
BODYSHAPE_AVERAGE:     str = "Average"
BODYSHAPE_BODYBUILDER: str = "Bodybuilder"
BODYSHAPE_OVERWEIGHT:  str = "Overweight"
BODYSHAPE_OBESE:       str = "Obese"
BODYSHAPE_CHOICES: List[str] = [
    BODYSHAPE_SKINNY, BODYSHAPE_SLIM, BODYSHAPE_AVERAGE,
    BODYSHAPE_BODYBUILDER, BODYSHAPE_OVERWEIGHT, BODYSHAPE_OBESE,
]

# Average is the default, and it is NOT a no-op. Saying "a bodyshape of
# average" is the whole point -- it is what pushes back on the gym-fit prior
# for the majority of photos, and a default that substituted nothing would
# leave exactly the silence this switch exists to fill. Ordered skinny -> obese
# so the radio reads as a scale, with the default sitting in the middle of it.
BODYSHAPE_DEFAULT: str = BODYSHAPE_AVERAGE

BODYSHAPE_TOKEN: str = "<bodyshape>"

# Switch label -> the word that replaces <bodyshape>. Mostly the label
# lowercased, which is deliberate: a one-to-one mapping means what the operator
# picked on the Generation page is legible in the Debug page's prompt dump
# without a lookup. This dict is the place to tune wording if a shape comes
# out weak -- "bodybuilder" to "heavily muscled bodybuilder", say -- rather
# than editing the three subject clauses, which would have to agree.
BODYSHAPE_WORDS: Dict[str, str] = {
    BODYSHAPE_SKINNY:      "skinny",
    BODYSHAPE_SLIM:        "slim",
    BODYSHAPE_AVERAGE:     "average",
    BODYSHAPE_BODYBUILDER: "bodybuilder",
    BODYSHAPE_OVERWEIGHT:  "overweight",
    BODYSHAPE_OBESE:       "obese",
}

# WHICH SCENE(S) TO RUN. Four options, not two.
#
# A plain single/all toggle could only ever test Scene 1, which is the wrong
# shape for the job this program is actually doing right now: the three scene
# prompts are being TUNED, and tuning Scene 3 means running Scene 3 on its own
# and looking at the result. With a single/all toggle that costs a full
# three-scene run every time, two thirds of it discarded, or an edit to
# whichever prompt happens to sit in slot A.
#
# So each scene is individually selectable and "All Scenes" is the production
# setting. Nothing else changes: All Scenes still runs the three in series,
# one generation each, so N photos at All Scenes is 3N images.
SCENE_1:    str = "Scene 1"
SCENE_2:    str = "Scene 2"
SCENE_3:    str = "Scene 3"
ALL_SCENES: str = "All Scenes"
LOCATION_MODE_CHOICES: List[str] = [SCENE_1, SCENE_2, SCENE_3, ALL_SCENES]
LOCATION_MODE_DEFAULT: str = ALL_SCENES

# mode -> which entries of LOCATION_PROMPT_KEYS to run, in order. Kept as an
# explicit map rather than parsed out of the label, so renaming a scene in the
# UI cannot silently change which prompt it runs.
LOCATION_MODE_SLICES: Dict[str, List[int]] = {
    SCENE_1:    [0],
    SCENE_2:    [1],
    SCENE_3:    [2],
    ALL_SCENES: [0, 1, 2],
}

# The three location prompts and the shared negative, as shipped. These are
# DEFAULTS ONLY -- the live values live in preferences.json and are edited on
# the Preferences page. They are full, fixed templates rather than a frame
# with a {location} hole: the phrasing that works for a statue is not the
# phrasing that works for a rooftop, and a shared frame would force both into
# whichever one was written first.
# THE SCENE HALF of each prompt: landmark, lighting, camera angle. It says
# nothing about WHO is in the shot or how many -- that is the subject clause's
# job (below), and the two are concatenated at run time.
#
# This split exists because the subject wording has to change with the Subject
# Count switch and the scene wording must not. Before the split, every scene
# prompt opened "The individual from the provided image", which flatly
# contradicted the 2 People and 3+ People settings: the switch would widen the
# canvas to fit a group while the prompt still asked for one person. Editing
# that out of three prompts by hand, every time the count changed, is exactly
# the manual step worth removing.
DEFAULT_LOCATION_PROMPT_A: str = (
    "Standing next to a large stone statue of a lion, New York style. "
    "The optimal front camera angle for the scene."
)
DEFAULT_LOCATION_PROMPT_B: str = (
    "Standing on top of the Empire State Building observation deck, city "
    "skyline background. The optimal front camera angle for the scene."
)
DEFAULT_LOCATION_PROMPT_C: str = (
    "Standing in front of the Eiffel Tower at sunset. "
    "The optimal front camera angle for the scene."
)

# THE SUBJECT HALF: who is in the shot, chosen by the Subject Count switch.
# One clause per count, each a complete sentence, prepended to whichever scene
# prompt is running.
#
# This is the ONLY dynamic seam in the prompt system. Everything else is sent
# verbatim -- there are no placeholders, no substitution, nothing assembled
# from fragments. Concatenation rather than a {subject} token in the scene
# text is deliberate: a token can be deleted or mistyped while editing a scene
# on the Preferences page, and a scene prompt that silently lost its subject
# would still run and still produce an image, just of the wrong thing.
# EVERY ONE OF THE THREE MUST CONTAIN BOTH <gender> AND <bodyshape>. The
# 1-Person clause takes the gender as an adjective ("the single male
# individual") because one word slots in cleanly there; the 2- and 3+-clauses
# take it as an appositive set off by commas ("the two individuals, male and
# female,") because the mixed words are phrases, not adjectives, and "the two
# male and female individuals" does not parse.
#
# THE BODYSHAPE IS ITS OWN SENTENCE, second, not another adjective piled onto
# the first. Two reasons. Stacking ("the single skinny male individual") buries
# it mid-noun-phrase where it competes with the gender word for the same
# attention, and a short declarative sentence of its own is how sd.cpp prompts
# reliably carry a physical attribute. It stays in the SUBJECT half either way
# -- it describes the person, not the landmark -- so it inherits the subject
# clause's front position in the assembled prompt, which is where a
# front-weighted attention model needs it. Edit the wording freely, keep both
# marks.
DEFAULT_SUBJECT_CLAUSE_ONE: str = (
    "Photo-realistic glamour shot of the single <gender> individual from the "
    "provided image, alone in the foreground. A bodyshape of <bodyshape>."
)
DEFAULT_SUBJECT_CLAUSE_TWO: str = (
    "Photo-realistic glamour shot of the two individuals from the provided "
    "image, <gender>, together in the foreground. A bodyshape of <bodyshape>."
)
DEFAULT_SUBJECT_CLAUSE_MANY: str = (
    "Photo-realistic glamour shot of the individuals from the provided image, "
    "<gender>, grouped together in the foreground. A bodyshape of <bodyshape>."
)

# Subject Count label -> (preferences key, shipped default). Same one-list
# discipline as LOCATION_PROMPT_KEYS: the switch, the Preferences fields and
# the prompt builder all read this, so they cannot disagree about which clause
# belongs to which count.
SUBJECT_CLAUSE_KEYS: Dict[str, tuple] = {
    SUBJECT_ONE:  ("subject_clause_one",  DEFAULT_SUBJECT_CLAUSE_ONE),
    SUBJECT_TWO:  ("subject_clause_two",  DEFAULT_SUBJECT_CLAUSE_TWO),
    SUBJECT_MANY: ("subject_clause_many", DEFAULT_SUBJECT_CLAUSE_MANY),
}

DEFAULT_GLAMOUR_NEGATIVE: str = (
    "Cartoon. Blurry. Visual obstructions. People in the Background. "
    "Missing/mutated arms/legs. Missing/mutated hands/feet. "
    "Clothes merging with skin. Phones, headphones."
)

# preferences.json key <-> display label, in run order. One list, so the
# Preferences page, the Generation page's read-only preview and the run
# builder can never disagree about how many scenes there are or what they are
# called.
#
# THE JSON KEYS STAY a/b/c WHILE THE LABELS READ "Scene 1/2/3". The keys are
# what a hand-edited preferences.json and any older file on disk already use,
# and renaming them would strand those; the labels are what the operator sees
# and are matched to the scene selector. This list is the only place the two
# meet, so they cannot drift.
LOCATION_PROMPT_KEYS: List[tuple] = [
    ("location_prompt_a", SCENE_1, DEFAULT_LOCATION_PROMPT_A),
    ("location_prompt_b", SCENE_2, DEFAULT_LOCATION_PROMPT_B),
    ("location_prompt_c", SCENE_3, DEFAULT_LOCATION_PROMPT_C),
]

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
# Input-image gallery (Generation page, right-hand column, above Output)
# ---------------------------------------------------------------------------
# A one-row, horizontally-scrolling strip of thumbnails of whatever is
# currently in the centre column's "Add Image" list, so each reference image
# can be identified at a glance instead of by filename alone.
#
# The SIZE is a preference because it is a direct trade against the Output
# preview below it. Both live in the same column, so every pixel this row
# takes is a pixel the preview loses; too large and the Output section starts
# getting pushed off-screen, too small and the thumbnails stop being
# identifiable. 128 is the default because at 128 the row costs less than a
# third of the 500px preview and a face is still recognisable. 96 is for
# small windows, 160/196 for people running the app maximised on a large
# display.
#
# Same single-source pattern as PREVIEW_IMAGE_HEIGHT / THUMBNAIL_GALLERY_
# HEIGHT: the chosen value feeds BOTH the gr.Gallery(height=...) kwarg and the
# #input-gallery CSS in display.py, so the row height and the derived square
# cell width can never drift apart. Unlike those two it is a preference rather
# than a constant, so display.build_app() reads it once when it assembles the
# stylesheet -- which is why a change here takes effect on the next launch
# rather than the moment Save is clicked. Gradio has no runtime stylesheet
# hook to do better without shipping a <style> injection hack.
INPUT_THUMBNAIL_CHOICES: List[int] = [96, 128, 160, 196]
DEFAULT_INPUT_THUMBNAIL: int = 96

# Vertical padding Gradio's own .grid-wrap adds above and below the row (8px
# each). Subtracted when deriving the square cell width from the row height,
# exactly as the output gallery's CSS does with its literal 16.
INPUT_GALLERY_PADDING: int = 16

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

    # ── Generate page "Image Edit" state — session-only, NOT persisted ──────
    # ref_images : the accumulated "Add Image" list. It survives a completed
    #              generation on purpose; the ONLY thing that empties it is
    #              the "Clear Images" button. Mirrors the Gradio State that
    #              carries it into a run (display._add_ref_images /
    #              _clear_ref_images keep the two in step).
    # ref_mode   : Use All vs Chain All. Seeded from REF_MODE_DEFAULT at
    #              startup and then owned by the user for the rest of the
    #              session — adding more images later must NOT snap it back
    #              to the default, so nothing writes to this except the
    #              radio's own change handler.
    # Neither belongs in configuration.json: both are "what am I doing right
    # now", not a setting, and a remembered Use All across restarts would
    # silently reintroduce the multi-reference VRAM spike on the next launch.
    "ref_images": [],
    "ref_mode": REF_MODE_DEFAULT,

    # Cached copy of configuration.json's last_image_browse_dir (see the
    # session-state accessors below). Loaded once at startup, written back
    # only when the folder actually changes.
    "last_image_browse_dir": "",
}


# ---------------------------------------------------------------------------
# Session-state accessors  (the globals above, with the file I/O attached)
# ---------------------------------------------------------------------------

def init_session_state() -> None:
    """Prime APP_STATE from disk once, at startup (called by launcher.main()).

    TWO pieces of this state persist across runs and are primed here:
    last_image_browse_dir (configuration.json — where the "Add Image" picker
    opens) and ref_mode (generation.json — the Use All / Chain All switch).
    Doing both at startup means the picker's first open and the first Add cost
    no JSON read, and it keeps "load at startup" honest rather than lazy-
    loading on first use.

    The reference-image LIST is the one thing deliberately NOT restored: those
    paths are the user's own files and may not exist next launch, so the list
    always starts empty (see the GENERATION_KEYS note).
    """
    cfg = load_configuration()
    saved = cfg.get("last_image_browse_dir", "") or ""
    if not saved or not Path(saved).is_dir():
        saved = str(get_pictures_dir())
    APP_STATE["last_image_browse_dir"] = saved
    APP_STATE["ref_images"] = []
    mode = load_generation().get("imagegen_ref_mode", REF_MODE_DEFAULT)
    APP_STATE["ref_mode"] = mode if mode in REF_MODE_CHOICES else REF_MODE_DEFAULT


def get_last_image_dir() -> str:
    """Folder the "Add Image" picker should open in.

    Reads the global, not the file. Falls back to a lazy load if the process
    somehow never called init_session_state() (e.g. display.py driven without
    launcher.py), so the picker still opens in the right place either way.
    """
    current = APP_STATE.get("last_image_browse_dir", "")
    if not current:
        init_session_state()
        current = APP_STATE.get("last_image_browse_dir", "")
    if current and Path(current).is_dir():
        return current
    return str(get_pictures_dir())


def set_last_image_dir(path: str) -> None:
    """Remember the folder an image was just added from.

    Writes configuration.json ONLY when the folder actually differs from what
    is already remembered — adding six images from the same folder is one
    save at most, not six. The global is updated either way.
    """
    if not path:
        return
    folder = str(path)
    if folder == APP_STATE.get("last_image_browse_dir", ""):
        return
    APP_STATE["last_image_browse_dir"] = folder
    try:
        update_configuration({"last_image_browse_dir": folder})
    except Exception:
        # A failed write costs the next session its starting folder and
        # nothing else, so it must never break the image the user just added.
        pass


def get_ref_images() -> List[str]:
    """Copy of the accumulated reference-image list (copy, so a caller
    mutating what it gets back cannot edit session state by accident)."""
    return list(APP_STATE.get("ref_images") or [])


def set_ref_images(paths: List[str]) -> None:
    APP_STATE["ref_images"] = list(paths or [])


def get_ref_mode() -> str:
    """Current Use All / Chain All selection. Seeded from generation.json at
    startup (init_session_state) and owned by the user from then on."""
    mode = APP_STATE.get("ref_mode") or REF_MODE_DEFAULT
    return mode if mode in REF_MODE_CHOICES else REF_MODE_DEFAULT


def set_ref_mode(mode: str) -> None:
    """Record a Use All / Chain All change in session state AND on disk.

    This used to be session-only, on the grounds that a remembered "Use All"
    silently reintroduces the multi-reference VRAM spike on the next launch.
    It now persists because "every setting on the Generation page is saved on
    submission" is the rule, and this is a setting on that page.

    The original worry is defused rather than ignored: the reference-image
    list itself is NOT restored, so a launch that starts with Use All starts
    with zero images to apply it to. The radio is hidden below 2 images, and by
    the time the user has added a second one they have made a deliberate choice
    to build a multi-reference edit. If you ever want the old behaviour back,
    drop the update_generation() call here and the load_generation() read in
    init_session_state(); nothing else depends on it.

    A failed write costs the next session its remembered mode and nothing
    else, so it must never break the click that caused it."""
    if mode not in REF_MODE_CHOICES:
        return
    if APP_STATE.get("ref_mode") == mode:
        return
    APP_STATE["ref_mode"] = mode
    try:
        update_generation({"imagegen_ref_mode": mode})
    except Exception:
        pass

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
# Resolved once per run by get_pictures_dir(); see the note in its docstring.
_PICTURES_DIR: Optional[Path] = None


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


def get_generation_path() -> Path:
    """data/generation.json — everything set on the Generation page."""
    return _get_project_root() / "data" / "generation.json"


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
    Graphics for the README only (./media). Not read by the running
    program — see get_images_dir() for the program's own UI/taskbar
    graphics.
    """
    return _get_project_root() / "media"


def get_images_dir() -> Path:
    """
    Static UI/taskbar graphics for the running program (not user data,
    not generated output):
      images/banner_llama.ico      - app/taskbar icon (see launcher.py)
      images/program_no_media.jpg  - shown when output/ is empty / idle
      images/program_encoding.jpg  - shown while the prompt-encoder LLM runs
      images/program_diffusion.jpg - shown while sd.cpp diffusion runs
    """
    return _get_project_root() / "images"


def _windows_pictures_dir() -> str:
    """The logged-in user's real "Pictures" (My Pictures) folder, or "".

    Asked of the shell rather than assumed, because Pictures is a KNOWN
    FOLDER and users move it: OneDrive relocates it to
    %USERPROFILE%\\OneDrive\\Pictures, and anyone short on system-drive space
    routinely repoints it at another drive entirely. Building
    %USERPROFILE%\\Pictures by hand lands in a folder that either does not
    exist or is not the one Explorer shows, which is exactly the wrong place
    to open a picker.

    SHGetKnownFolderPath(FOLDERID_Pictures) returns wherever it actually
    lives now. Everything is wrapped: on a non-Windows host, or if any of the
    COM plumbing fails, this returns "" and get_pictures_dir() falls back.
    """
    if os.name != "nt":
        return ""
    try:
        import ctypes
        from ctypes import wintypes

        class _GUID(ctypes.Structure):
            _fields_ = [("Data1", wintypes.DWORD),
                        ("Data2", wintypes.WORD),
                        ("Data3", wintypes.WORD),
                        ("Data4", ctypes.c_byte * 8)]

        # FOLDERID_Pictures, from KnownFolders.h.
        _FOLDERID_PICTURES = "{33E28130-4E1E-4676-835A-98395C3BC3BB}"

        ole32 = ctypes.windll.ole32
        shell32 = ctypes.windll.shell32
        ole32.CLSIDFromString.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(_GUID)]
        shell32.SHGetKnownFolderPath.argtypes = [
            ctypes.POINTER(_GUID), wintypes.DWORD, wintypes.HANDLE,
            ctypes.POINTER(ctypes.c_wchar_p),
        ]

        guid = _GUID()
        if ole32.CLSIDFromString(_FOLDERID_PICTURES, ctypes.byref(guid)) != 0:
            return ""
        out = ctypes.c_wchar_p()
        # dwFlags 0, hToken None -> current user, no default-path creation.
        if shell32.SHGetKnownFolderPath(ctypes.byref(guid), 0, None,
                                        ctypes.byref(out)) != 0:
            return ""
        try:
            return out.value or ""
        finally:
            # The shell allocated this with CoTaskMemAlloc; we own freeing it.
            ole32.CoTaskMemFree(out)
    except Exception:
        return ""


def get_pictures_dir() -> Path:
    """Default starting folder for the Generate page's "Add Image" picker.

    Ladder, first hit wins:
      1. the shell's FOLDERID_Pictures (handles OneDrive / relocated folders)
      2. %USERPROFILE%\\Pictures, if it exists
      3. the user's home folder — always exists, so the picker always opens
         somewhere real rather than at the process's CWD

    This is only the FIRST-EVER default; once the user picks an image from
    anywhere else, that folder is remembered in configuration.json under
    last_image_browse_dir (see get_last_image_dir / set_last_image_dir) and
    used from then on.

    Mirrored by installer.py's _default_pictures_dir(), which seeds the same
    value into a fresh configuration.json before scripts/ is importable.

    Cached like _get_project_root(), because _default_configuration() calls
    this on EVERY load_configuration() and the known-folder lookup is a COM
    round-trip. The folder cannot move while the program is running.
    """
    global _PICTURES_DIR
    if _PICTURES_DIR is not None:
        return _PICTURES_DIR
    _PICTURES_DIR = _resolve_pictures_dir()
    return _PICTURES_DIR


def _resolve_pictures_dir() -> Path:
    known = _windows_pictures_dir()
    if known and Path(known).is_dir():
        return Path(known)
    profile = os.environ.get("USERPROFILE", "")
    if profile:
        candidate = Path(profile) / "Pictures"
        if candidate.is_dir():
            return candidate
    return Path.home()


def get_dictionaries_dir() -> Path:
    """data/dictionaries — Chromium .bdic spellcheck dictionaries.

    Read by launcher.py, which points QtWebEngine at this folder via the
    QTWEBENGINE_DICTIONARIES_PATH environment variable so the prompt boxes get
    red squiggles under misspelled words. Populated once at install time by
    installer.install_spellcheck_dictionary().

    Entirely offline: the dictionary is a local binary file and Chromium's
    spellchecker reads it in-process. Nothing is sent anywhere. (Chromium also
    has a "spelling service" mode that POSTs text to Google — that is a
    SEPARATE feature, off by default, and QtWebEngine exposes no way to switch
    it on, so there is no path by which prompt text could leave the machine.)
    """
    return _get_project_root() / "data" / "dictionaries"


SPELLCHECK_LANGUAGE: str = "en-US"


def get_build_dir() -> Path:
    return _get_project_root() / "data" / "build"


def get_llama_bin_dir() -> Path:
    return _get_project_root() / "data" / "llama_cpp_binaries"


def get_sd_bin_dir() -> Path:
    return _get_project_root() / "data" / "stable_diffusion_binaries"


def ensure_data_dirs() -> None:
    root = _get_project_root()
    for d in ("data", "output", "models", "media", "images"):
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
        # Split-SDXL text encoders. Every SDXL gguf in circulation is UNet-only
        # and needs these two supplied separately; a full SDXL .safetensors
        # carries its own and leaves them blank. Unused by Z-Image and Flux.2,
        # which condition through the Qwen3 encoder_model_path instead.
        "clip_l_model_path":   "",  "clip_l_model_name":   "",
        "clip_g_model_path":   "",  "clip_g_model_name":   "",
        # T5-XXL, FLUX.1's second text encoder. Accepts .safetensors or .gguf;
        # a gguf quant is strongly preferred because this runs on the CPU and
        # t5xxl_fp16 is 9.8GB of system RAM for one conditioning pass.
        "t5xxl_model_path":    "",  "t5xxl_model_name":    "",
        # v-prediction override for SDXL finetunes; see sdxl_prediction_override.
        "imagegen_prediction": PREDICTION_AUTO,
        # Manual model-family override for files auto-detection cannot place;
        # see FAMILY_OVERRIDE_CHOICES.
        "imagegen_family_override": FAMILY_OVERRIDE_AUTO,
        "last_model_browse_dir": ".\\models",
        # Starting folder for the Generate page's "Add Image" picker. Unlike
        # last_model_browse_dir (which sensibly starts at .\models, where this
        # program's own downloads land) reference images are the user's own
        # files, so this starts at their Pictures folder — resolved live from
        # the shell for THIS login rather than hardcoded, since it may be
        # redirected to OneDrive or another drive. Overwritten with whatever
        # folder they actually browse to (see set_last_image_dir).
        "last_image_browse_dir": str(get_pictures_dir()),
        "backend_processing": cpu_label,
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
        "imagegen_clip_skip": 2,
        # Everything the Generation page owns -- width/height/steps/cfg/seed/
        # sampler/batch/format/preset/strength/prompts -- used to live here and
        # now lives in data/generation.json (see _default_generation() and
        # GENERATION_KEYS). Nothing may be listed in both files: a key in two
        # files is written by whichever page saved last and read from whichever
        # file the caller happened to load, which is exactly the bug the
        # configuration/preferences split was made to avoid.
        "auto_save": True,
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


def default_configuration() -> Dict[str, Any]:
    """Public accessor for the canonical factory-default configuration dict.

    Same single source of truth that seeds a fresh install (installer.py's
    write_default_configuration) and backfills missing keys at load time
    (_default_configuration / _load_json_with_defaults). Exposed so the
    Configuration page's "Revert To Defaults" button resets its widgets from
    exactly these values rather than a second, drift-prone copy."""
    return _default_configuration()


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


def _load_json_with_defaults(path: Path, defaults: Dict[str, Any],
                             migrate=None) -> Dict[str, Any]:
    """Read a JSON dict, backfilling any keys missing because the file was
    written by an older version of the program (e.g. before the window-geometry
    keys existed). Saved values always win; defaults only fill genuine gaps, so
    this never overwrites real user data. Unreadable/corrupt -> defaults.

    `migrate`, if given, is applied to the RAW file contents BEFORE the defaults
    are merged in, and must return the rewritten dict. The ordering is the whole
    point and was worth a bug to learn: a migration that renames a key cannot
    run after the backfill, because by then the NEW key is present with its
    default value and the migration cannot tell "the user already has this" from
    "the backfill just invented it". Renaming backend_encoder/backend_imagegen
    to backend_processing that way silently threw away a saved GPU choice and
    reset the user to CPU. Before the merge, the raw dict says only what the
    user actually saved."""
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    if migrate is not None:
                        data = migrate(data)
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

def _migrate_split_backends(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Fold a pre-merge configuration.json's two backend keys into one.

    backend_encoder and backend_imagegen used to be separate dropdowns, on the
    theory that the Qwen3 encoder and the sd.cpp diffuser could usefully sit on
    different devices. In practice they never did: on a one-GPU machine there is
    no second device to split across, and the encoder/VAE-to-CPU decision that
    actually matters is made per FAMILY by the spec table (encoder_to_cpu /
    vae_to_cpu) and by the Diffuser Placement dropdown -- not by this control.
    Two dropdowns that were always set the same thing is one dropdown.

    The IMAGEGEN value wins when the two disagree. It is the one that governs
    the expensive half of the run, so if a user ever did set them apart, that is
    the choice worth keeping.

    Both old keys are dropped, so nothing is left behind for a later reader to
    pick up. In-place on the dict, before the caller sees it; the file itself is
    rewritten on the next save rather than eagerly here, since a stale pair of
    keys sitting in the JSON is inert once nothing reads them.
    """
    legacy = cfg.pop("backend_imagegen", None) or cfg.pop("backend_encoder", None)
    cfg.pop("backend_encoder", None)
    if legacy and not cfg.get("backend_processing"):
        cfg["backend_processing"] = legacy
    return cfg


def load_configuration() -> Dict[str, Any]:
    """Load data/configuration.json, migrating a pre-split persistent.json
    into place first if that is all that exists."""
    _migrate_legacy_persistent()
    cfg = _load_json_with_defaults(get_configuration_path(),
                                   _default_configuration(),
                                   migrate=_migrate_split_backends)
    # Refresh the module-level family override on every load, so that
    # diffuser_family() -- called from a dozen places that only have a path --
    # sees the user's current choice without an extra argument everywhere.
    set_family_override(cfg.get("imagegen_family_override", FAMILY_OVERRIDE_AUTO))
    return cfg


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
        "input_thumbnail_size": DEFAULT_INPUT_THUMBNAIL,
        "encoder_model_debug": False,
        # The three location prompts and the one shared negative. These live
        # in PREFERENCES rather than generation.json because they are standing
        # taste -- the set of landmarks this install offers -- not the state of
        # one run. Consequences that make this the right file: they survive a
        # clean install, and they are not rewritten on every Generate click,
        # so an edit made on the Preferences page cannot be clobbered by a run
        # that started before it.
        "location_prompt_a": DEFAULT_LOCATION_PROMPT_A,
        "location_prompt_b": DEFAULT_LOCATION_PROMPT_B,
        "location_prompt_c": DEFAULT_LOCATION_PROMPT_C,
        "glamour_negative_prompt": DEFAULT_GLAMOUR_NEGATIVE,
        # One subject sentence per Subject Count setting. Prepended to
        # whichever scene prompt is running -- see build_positive_prompt.
        "subject_clause_one":  DEFAULT_SUBJECT_CLAUSE_ONE,
        "subject_clause_two":  DEFAULT_SUBJECT_CLAUSE_TWO,
        "subject_clause_many": DEFAULT_SUBJECT_CLAUSE_MANY,
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


# ── generation.json (Generation page) ──────────────────────────────────────
# The THIRD settings file, and the same rule as the other two: one page owns
# one file, and no key appears in more than one.
#
#   configuration.json  this machine's wiring -- model paths, backends,
#                       threads, device indices, window geometry. Reseeded by
#                       the installer on a clean install.
#   preferences.json    standing taste -- prompt template, thumbnail counts.
#   generation.json     what was on the Generation page when Generate was last
#                       clicked.
#
# WHY IT IS SEPARATE. These values are not machine wiring and they are not
# standing taste; they are the state of one page, they change on every run, and
# they are the set most likely to want wiping or hand-editing on their own. Two
# concrete consequences of the old arrangement, both fixed by the split:
#   * a clean install deletes configuration.json, which took the user's last
#     working steps/cfg/size/prompt with it even though a reinstall invalidates
#     none of them. generation.json is seeded once and never purged, exactly
#     like preferences.json and prompt_cache.json.
#   * the Configuration page's Save wrote the whole configuration dict back,
#     so its save and the Generation page's auto-save were two writers on one
#     file racing over unrelated keys.
#
# WHAT IS NOT IN HERE: the reference-image LIST. Those paths are the user's own
# files and may well not exist next launch (a temp folder, a removed USB
# stick, a renamed directory), so restoring them would repopulate the input
# column with dead paths that only fail at generation time. The list stays
# session-only in APP_STATE, and the Generation page opens with it empty.
# imagegen_ref_mode -- the Use All / Chain All switch -- IS saved: it is a
# setting rather than a file reference, and it survives having no images to
# apply to (with 0 or 1 images the radio is hidden and the value is simply
# not consulted).
GENERATION_KEYS: List[str] = [
    "imagegen_width", "imagegen_height",
    "imagegen_steps", "imagegen_cfg_scale",
    "imagegen_sampling", "imagegen_seed",
    "imagegen_batch_count", "imagegen_quality_preset",
    "imagegen_strength", "imagegen_ref_mode",
    "imagegen_last_family",
    "output_format",
    "last_prompt", "negative_prompt",
    # Glamour mode: the three things still on the Generation page that the
    # user picks and expects to find again next launch. Subject Count and
    # Subject Gender are here rather than in preferences because they describe
    # the batch of photos in front of you right now, which is exactly what
    # generation.json is for -- the CLAUSES they select between are standing
    # taste and live in preferences.json instead.
    "imagegen_subject_count", "imagegen_location_mode",
    "imagegen_gender", "imagegen_bodyshape",
]


# ---------------------------------------------------------------------------
# Per-family generation DEFAULTS -- the full "optimal settings for this model"
# set, not just steps and cfg.
# ---------------------------------------------------------------------------
# FAMILY_STEP_CFG above already carries the two numbers that matter most, and
# it stays the source of truth for those. This table adds the rest of the
# Generation page: sampler, batch count and denoise strength. Together they are
# what family_generation_defaults() returns, and that ONE function is what both
# the "Restore To Defaults" button and the on-model-change snap apply, so the
# two can never drift into recommending different things.
#
#   sampler   Not a taste setting -- it interacts with the scheduler each
#             family was distilled against.
#               flux1     : euler, per stable-diffusion.cpp's docs/flux.md.
#                           (Individual finetunes may print a different
#                           recommendation on their model card -- fluxedUp
#                           asks for dpm++2m -- which is exactly why this is a
#                           starting point the user can change, not a lock.)
#               flux2     : euler_a. Deliberately NOT plain euler despite
#                           Flux.2 being a rectified-flow model: on this
#                           sd.cpp build euler comes out semi-garbled on the
#                           target hardware and euler_a is clean. Measured,
#                           not assumed.
#               z-image   : euler_a. Both work; this is the better default.
#               sdxl_base : dpm++2m, the long-standing SDXL standard and what
#                           nearly every finetune's card recommends.
#               sdxl_dist : euler_a. The distilled derivatives want the simple
#                           sampler their few-step schedule was trained on.
#
#   batch     How many images per Generate click. 1 for the SLOW families
#             (flux1_dev at 20 steps, flux2_base at 20, sdxl_base at 30) and 2
#             for the few-step distilled ones. This is the setting most likely
#             to make a model feel broken on an 8GB card: a batch of 2 at 20+
#             steps is several minutes before anything appears on screen, and
#             the natural read of that is "it has hung". Fast models get 2
#             because a second variation is nearly free and genuinely useful.
#
#   strength  img2img denoise. Only consulted by the families that use the -i
#             init-image path (flux1, sdxl); Flux.2 conditions through -r and
#             has no strength parameter at all. 0.65 throughout: enough to
#             restyle while keeping composition, and the value this program has
#             always used in place of sd.cpp's more destructive 0.75.
FAMILY_GENERATION_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "flux1_dev":       {"sampler": "euler",   "batch": 1, "strength": 0.65},
    "flux1_schnell":   {"sampler": "euler",   "batch": 2, "strength": 0.65},
    "flux2_distilled": {"sampler": "euler_a", "batch": 2, "strength": 0.65},
    "flux2_base":      {"sampler": "euler_a", "batch": 1, "strength": 0.65},
    "z-image":         {"sampler": "euler_a", "batch": 2, "strength": 0.65},
    "sdxl_base":       {"sampler": "dpm++2m", "batch": 1, "strength": 0.65},
    "sdxl_distilled":  {"sampler": "euler_a", "batch": 2, "strength": 0.65},
    "none":            {"sampler": "euler_a", "batch": 2, "strength": 0.65},
}

# Resolution is the ONE default that is deliberately NOT per-family.
#
# Every family here is natively 1024 and every one of them looks better there,
# so a "best settings" table would put 1024 in all eight rows -- and that is
# precisely the wrong default on the hardware this program targets. The compute
# buffer scales with pixel count, so 1024 is roughly four times the VRAM of 512
# (measured for FLUX.1 at ~1366 MB per megapixel; Flux.2 is ~3000). On an 8GB
# card that is the difference between a run that completes and a run that OOMs
# or falls back to streaming weights from system RAM at a fraction of the
# speed.
#
# So 512x512 is fixed across all families: it is the size that WORKS, on every
# sensible model, on the smallest card this program supports. Raising it is a
# deliberate act the user takes once they know their model loads -- which is
# the right way round, because a conservative default that always runs beats a
# better-looking one that sometimes cannot.
#
# Guarded with a membership test rather than assumed: 512 is in every entry of
# FAMILY_IMAGE_SIZES today, but if a future family's floor were ever raised
# above it the fallback keeps this returning a value the dropdown will accept.
DEFAULT_GENERATION_WIDTH:  int = 512
DEFAULT_GENERATION_HEIGHT: int = 512

# The factory negative prompt. Named because three separate places want the
# same literal -- _default_generation(), family_generation_defaults() (for the
# Restore To Defaults button) and installer.py's write_default_generation() --
# and three hand-copied string literals is three chances to drift.
DEFAULT_NEGATIVE_PROMPT: str = (
    "Visual obstructions. Cartoon. Blurry. Background people. Missing/mutated arms/legs. Missing/mutated hands/feet. Well covered, winter clothing. Phones, headphones."
)


def family_generation_defaults(diff_path: str, diff_arch: str = "") -> Dict[str, Any]:
    """The complete set of optimal Generation-page settings for a diffuser.

    Returns real values (not gr.updates) keyed exactly as generation.json is,
    so the same dict can seed widgets, be written to disk, or be diffed against
    what is on screen. Every key the Generation page owns is present EXCEPT the
    two prompts: the positive prompt is the user's own work and no "defaults"
    button should ever delete it, and the negative prompt is handled separately
    by the caller (see display._restore_generation_defaults) because resetting
    boilerplate is reasonable on an explicit button press but not on a model
    change.

    Composed from FAMILY_STEP_CFG (steps/cfg), FAMILY_GENERATION_DEFAULTS
    (sampler/batch/strength) and the fixed 512x512 resolution. Always returns a
    valid set: an unknown model falls back to the 'none' entry, which is the
    permissive superset.
    """
    key = family_step_cfg_key(diff_path, diff_arch)
    spec = FAMILY_STEP_CFG.get(key, FAMILY_STEP_CFG["none"])
    extra = FAMILY_GENERATION_DEFAULTS.get(key, FAMILY_GENERATION_DEFAULTS["none"])
    sizes = FAMILY_IMAGE_SIZES.get(key, IMAGE_SIZES)

    def _size(preferred: int) -> int:
        return preferred if preferred in sizes else min(sizes)

    return {
        "imagegen_width":  _size(DEFAULT_GENERATION_WIDTH),
        "imagegen_height": _size(DEFAULT_GENERATION_HEIGHT),
        "imagegen_steps":       spec["steps"][1],
        "imagegen_cfg_scale":   spec["cfg"][3],
        "imagegen_sampling":    extra["sampler"],
        "imagegen_batch_count": extra["batch"],
        "imagegen_strength":    extra["strength"],
        # -1 = a fresh random seed each run. A pinned seed is only meaningful
        # against one specific model at one specific size, so it never survives
        # into a defaults set.
        "imagegen_seed": -1,
        "output_format": "png",
        # 512x512 with the family's own steps/cfg/sampler IS the Fast (Turbo)
        # preset (see GENERATION_PRESET_SIZES / resolve_preset), so naming it
        # here keeps the dropdown honest rather than leaving it reading
        # "Custom" over a set of values it would itself produce.
        "imagegen_quality_preset": "Fast (Turbo)",
    }


def _default_generation() -> Dict[str, Any]:
    """Defaults for generation.json. Kept in step with installer.py's
    write_default_generation(); that one seeds the file at install time, this
    one backfills gaps at load time.

    Every key here must appear in GENERATION_KEYS, and none of them may appear
    in _default_configuration() or _default_preferences()."""
    return {
        # Width/height are still stored, but glamour mode DERIVES them from
        # Subject Count on every run (see subject_count_size) rather than
        # reading them back. They are seeded to the 1-Person portrait pair so
        # a file inspected by hand is not misleading about what will run.
        "imagegen_width": 512,
        "imagegen_height": 768,
        "imagegen_steps": GLAMOUR_DEFAULT_STEPS,
        # Literals, matching what display.do_generate() hard-codes and
        # overwrites on every run. Seeded here only so a generation.json read
        # by hand before the first run is not misleading.
        "imagegen_cfg_scale": 1.0,
        "imagegen_seed": -1,
        "imagegen_sampling": "euler_a",
        "imagegen_batch_count": 1,
        "imagegen_quality_preset": "Fast (Turbo)",
        "imagegen_subject_count": SUBJECT_COUNT_DEFAULT,
        "imagegen_location_mode": LOCATION_MODE_DEFAULT,
        # Subject Gender -> the word substituted into the subject clause's
        # <gender> mark. See GENDER_WORDS for the nine-cell table and
        # GENDER_DEFAULT for why Male(s) rather than M+F(s) ships as default.
        "imagegen_gender": GENDER_DEFAULT,
        # Subject Bodyshape -> the word substituted into <bodyshape>. Average
        # is the default and is NOT a no-op; see BODYSHAPE_DEFAULT.
        "imagegen_bodyshape": BODYSHAPE_DEFAULT,
        # img2img denoise strength: 0.0 returns the init image untouched, 1.0
        # ignores it entirely. sd.cpp's own default is 0.75, but I use 0.65.
        "imagegen_strength": 0.65,
        # Use All / Chain All. Chain All is the safer default (one reference
        # resident at a time); see REF_MODE_DEFAULT.
        "imagegen_ref_mode": REF_MODE_DEFAULT,
        # The family_step_cfg_key() the saved steps/cfg/size values belong to.
        # Steps and CFG mean different things per family -- cfg 1.5 is correct
        # for a distilled model and badly wrong for SDXL base, which wants ~7 --
        # so the Generation page needs to know whether the current values were
        # chosen FOR the current family or inherited from a different one.
        "imagegen_last_family": "",
        # JPG, not PNG. These are photographic outputs headed for a client,
        # and at 512x768 a PNG is several times the size for no visible gain.
        "output_format": GLAMOUR_OUTPUT_FORMAT,
        # last_prompt is vestigial in glamour mode -- the positive prompts come
        # from preferences.json, not from a box on this page -- but the key
        # stays so an older generation.json still loads without a backfill
        # warning, and so the Debug page can show what a pre-glamour build
        # left behind.
        "last_prompt": "",
        "negative_prompt": DEFAULT_GLAMOUR_NEGATIVE,
    }


def load_generation() -> Dict[str, Any]:
    """Load data/generation.json.

    On the first load after this file was introduced it does not exist yet,
    but the user's real values are sitting in configuration.json where they
    used to live. Rather than silently resetting a working setup to factory
    defaults, they are carried across here and REMOVED from the configuration
    file, so exactly one file owns each key from then on. Same one-shot
    migration pattern as load_preferences() uses for prompt_template.
    """
    path = get_generation_path()
    if path.exists():
        return _load_json_with_defaults(path, _default_generation())

    migrated = _default_generation()
    _migrate_legacy_persistent()
    cfg_path = get_configuration_path()
    if cfg_path.exists():
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                old_cfg = json.load(f)
            if isinstance(old_cfg, dict):
                moved = False
                for key in GENERATION_KEYS:
                    if key in old_cfg:
                        migrated[key] = old_cfg.pop(key)
                        moved = True
                if moved:
                    _save_json_atomic(cfg_path, old_cfg)
        except (json.JSONDecodeError, IOError, OSError):
            pass
    try:
        _save_json_atomic(path, migrated)
    except OSError:
        pass
    return migrated


def save_generation(data: Dict[str, Any]) -> None:
    """Write generation.json, keeping ONLY the keys this file owns.

    The filter matters because the caller is do_generate(), which holds a
    merged configuration+preferences+generation dict (inference.py wants one
    dict, not three). Writing that whole thing back would copy every model
    path and the prompt template into this file, recreating the duplicate-key
    problem the split exists to prevent."""
    _save_json_atomic(get_generation_path(),
                      {k: v for k, v in data.items() if k in GENERATION_KEYS})


def update_generation(updates: Dict[str, Any]) -> Dict[str, Any]:
    data = load_generation()
    data.update({k: v for k, v in updates.items() if k in GENERATION_KEYS})
    save_generation(data)
    return data


def generation_config() -> Dict[str, Any]:
    """The single merged dict a generation run needs: configuration.json,
    then preferences.json, then generation.json, later files winning.

    inference.generate_image() and enhance_prompt() take ONE cfg dict and read
    keys from all three pages out of it -- model paths and devices from
    configuration, prompt_template from preferences, everything on the
    Generation page from generation. Merged here, in the one place that knows
    about all three files, so neither inference.py nor any save handler has to
    know the split exists. The key sets are disjoint by construction, so the
    merge order only matters as a tiebreak that should never fire."""
    merged = dict(load_configuration())
    merged.update(load_preferences())
    merged.update(load_generation())
    return merged


def get_max_thumbnails() -> int:
    """Thumbnail display cap from preferences, clamped to a listed choice so a
    hand-edited preferences.json cannot ask the gallery for 100000 images."""
    try:
        value = int(load_preferences().get("max_thumbnails", DEFAULT_MAX_THUMBNAILS))
    except (TypeError, ValueError):
        return DEFAULT_MAX_THUMBNAILS
    return value if value in MAX_THUMBNAIL_CHOICES else DEFAULT_MAX_THUMBNAILS


def get_input_thumbnail_size() -> int:
    """Input-gallery row height in pixels, from preferences, clamped to a
    listed choice so a hand-edited preferences.json cannot ask for a 4000px
    row that would push the Output preview off the page entirely.

    Read by BOTH display._build_generate_tab_inner (the gr.Gallery height=
    kwarg) and display.build_app (the #input-gallery CSS, which carries
    !important and would otherwise silently win). One accessor, so the two
    cannot disagree."""
    try:
        value = int(load_preferences().get("input_thumbnail_size",
                                           DEFAULT_INPUT_THUMBNAIL))
    except (TypeError, ValueError):
        return DEFAULT_INPUT_THUMBNAIL
    return value if value in INPUT_THUMBNAIL_CHOICES else DEFAULT_INPUT_THUMBNAIL


# ---------------------------------------------------------------------------
# Glamour-mode accessors
# ---------------------------------------------------------------------------
# All four read preferences.json and all four fall back to the shipped
# default when the stored value is missing, non-string or blank. Blank
# matters: the Preferences page lets the user clear a box, and an empty
# positive prompt sent to sd-cli is a run that wastes a minute producing
# noise. Clearing a box therefore restores the default rather than disabling
# the location -- use Single Location mode to run fewer than three.

def get_location_prompts() -> List[tuple]:
    """The three (label, prompt_text) pairs, in run order A, B, C.

    Reads preferences.json once and resolves every slot against
    LOCATION_PROMPT_KEYS, so a hand-edited file that is missing a key, or has
    it set to null or "", still yields three usable prompts.
    """
    prefs = load_preferences()
    out: List[tuple] = []
    for key, label, fallback in LOCATION_PROMPT_KEYS:
        value = prefs.get(key, fallback)
        if not isinstance(value, str) or not value.strip():
            value = fallback
        out.append((label, value.strip()))
    return out


def active_location_prompts(mode: str) -> List[tuple]:
    """The (label, prompt) pairs a run should actually use, in run order.

    Scene 1/2/3 -> that one prompt alone. All Scenes -> all three, which is
    what makes one input photo produce three output images.

    An unrecognised mode -- from a hand-edited generation.json, or a file
    written by the older two-way build that stored "All Three Locations" --
    falls back to ALL_SCENES rather than to a single scene. That is the safer
    failure of the two: producing the full set the program exists to produce
    is immediately visible and re-runnable, whereas silently dropping to one
    scene looks like a successful run that quietly did a third of the work.
    """
    prompts = get_location_prompts()
    indices = LOCATION_MODE_SLICES.get(mode, LOCATION_MODE_SLICES[ALL_SCENES])
    return [prompts[i] for i in indices if i < len(prompts)]


def get_glamour_negative() -> str:
    """The single shared negative prompt, from preferences.json."""
    value = load_preferences().get("glamour_negative_prompt",
                                   DEFAULT_GLAMOUR_NEGATIVE)
    if not isinstance(value, str) or not value.strip():
        return DEFAULT_GLAMOUR_NEGATIVE
    return value.strip()


def get_subject_clause(subject_count: str) -> str:
    """The subject sentence for a Subject Count label, from preferences.json.

    Falls back to the shipped default on a missing, non-string or blank value,
    and to the 1-Person clause on an unrecognised count -- the same blank-means-
    default rule the scene prompts follow, for the same reason: an empty half
    would send a prompt that never says who is in the picture.
    """
    key, fallback = SUBJECT_CLAUSE_KEYS.get(
        subject_count, SUBJECT_CLAUSE_KEYS[SUBJECT_ONE])
    value = load_preferences().get(key, fallback)
    if not isinstance(value, str) or not value.strip():
        return fallback
    return value.strip()


def gender_word(gender: str, subject_count: str) -> str:
    """The word that replaces <gender>, for a (gender, count) pair.

    Both arguments are validated against the choice lists rather than trusted,
    because both can arrive from a hand-edited generation.json. An unknown
    gender falls back to GENDER_DEFAULT and an unknown count to SUBJECT_ONE --
    the same rule subject_count_size() already uses, and for the same reason:
    one person is the overwhelmingly common case.
    """
    row = GENDER_WORDS.get(gender, GENDER_WORDS[GENDER_DEFAULT])
    return row.get(subject_count, row[SUBJECT_ONE])


def bodyshape_word(bodyshape: str) -> str:
    """The word that replaces <bodyshape>.

    Takes no subject count, unlike gender_word: the shape word is the same for
    one person or four. Validated rather than trusted, for the same reason --
    the value can arrive from a hand-edited generation.json.
    """
    return BODYSHAPE_WORDS.get(bodyshape, BODYSHAPE_WORDS[BODYSHAPE_DEFAULT])


# The two marks a subject clause is expected to carry, as (token, what it is
# for) pairs. ONE list, so the Preferences warning, the Debug page's check and
# the substitution itself cannot disagree about how many marks there are --
# adding a third token later is an edit here plus its word table, not a hunt
# through three files.
SUBJECT_TOKENS: List[tuple] = [
    (GENDER_TOKEN,    "Subject Gender"),
    (BODYSHAPE_TOKEN, "Subject Bodyshape"),
]


def missing_subject_tokens(text: str) -> List[str]:
    """Which of SUBJECT_TOKENS a clause has lost, as a list of marks.

    Used by the Preferences page to warn on save and by the Debug page to
    flag. It is NOT used to reject or rewrite the edit: a clause without a
    mark is still a valid prompt, just one where that switch has no effect,
    and silently restoring the default over someone's deliberate wording would
    be worse than telling them what they have done.
    """
    if not isinstance(text, str):
        return [tok for tok, _ in SUBJECT_TOKENS]
    low = text.lower()
    return [tok for tok, _ in SUBJECT_TOKENS if tok.lower() not in low]


def clause_has_gender_token(text: str) -> bool:
    """True when a subject clause still carries the <gender> mark.

    Kept as its own predicate because it reads better at the one call site
    that cares only about gender; everything that wants the full picture calls
    missing_subject_tokens() instead.
    """
    if not isinstance(text, str):
        return False
    return GENDER_TOKEN.lower() in text.lower()


def apply_subject_tokens(text: str, subject_count: str, gender: str,
                         bodyshape: str) -> str:
    """Substitute BOTH marks into a clause, case-insensitively.

    <gender>, <Gender> and <GENDER> all match, and the same for <bodyshape>,
    so an edit that capitalised a mark at the start of a sentence still works.
    Text with a mark missing is returned with the others done -- see
    missing_subject_tokens for why a missing mark is a warning, not an error.

    Substitution is done in ONE pass per token against the ORIGINAL marks, so
    a word that happened to contain another mark's text could not be
    re-substituted on a later pass. The double space a substitution can leave
    behind is collapsed, because sd.cpp tokenises the prompt as written.
    """
    if not isinstance(text, str) or not text:
        return ""
    out = re.sub(re.escape(GENDER_TOKEN), gender_word(gender, subject_count),
                 text, flags=re.IGNORECASE)
    out = re.sub(re.escape(BODYSHAPE_TOKEN), bodyshape_word(bodyshape),
                 out, flags=re.IGNORECASE)
    return re.sub(r"[ \t]{2,}", " ", out).strip()


def resolve_subject_clause(subject_count: str, gender: str,
                           bodyshape: str = BODYSHAPE_DEFAULT) -> str:
    """The subject clause with both marks already substituted.

    get_subject_clause() returns what is STORED (marks and all); this returns
    what is SENT. Anything reporting on a run -- the Debug page -- wants this
    one, so that the page whose job is to say what the program does is not
    showing marks the model never sees.
    """
    return apply_subject_tokens(get_subject_clause(subject_count),
                                subject_count, gender, bodyshape)


def build_positive_prompt(scene_prompt: str, subject_count: str,
                          gender: str = GENDER_DEFAULT,
                          bodyshape: str = BODYSHAPE_DEFAULT) -> str:
    """Assemble the full positive prompt actually sent to sd-cli.

    Subject clause first, then the scene. That order because the subject is
    the thing being generated and the scene is where it is put -- and because
    diffusion attention is front-weighted, so the person the whole program
    exists to render should not be trailing a landmark description.

    This is the one place the parts meet -- gender and bodyshape into the
    clause, clause onto scene. Anything that wants to know what will be sent
    -- the run loop, the Debug page -- calls this rather than concatenating
    for itself, so there is no second copy of the join to drift.

    `gender` and `bodyshape` default to their shipped values rather than being
    required, so a call site that predates either switch still produces a
    complete, sendable prompt instead of one with a literal mark left in it.
    """
    subject = resolve_subject_clause(subject_count, gender, bodyshape)
    scene = (scene_prompt or "").strip()
    if not scene:
        return subject
    return f"{subject} {scene}"


def subject_count_size(label: str) -> tuple:
    """(width, height) for a Subject Count label; portrait for anything
    unrecognised, since one person is the overwhelmingly common case."""
    return SUBJECT_COUNT_SIZES.get(label, SUBJECT_COUNT_SIZES[SUBJECT_ONE])


def is_flux2_diffuser(diff_path: str, diff_arch: str = "") -> bool:
    """True when the configured diffuser is a Flux.2 file of either variant.

    This program is Flux.2-only by design -- the whole pipeline assumes the -r
    reference-image path, which is how Flux.2 conditions on an input photo.
    SDXL and FLUX.1 denoise from a single -i init image instead, which
    restyles the photo rather than placing the person in a new scene, and
    Z-Image takes no input image at all. Rather than let those produce
    confusing output, the Generation page gates on this.
    """
    if not diff_path:
        return False
    return diffuser_family(diff_path, diff_arch) == DIFFUSER_FAMILY_FLUX2


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
    resolution plus the family's tuned steps / cfg / sampler. Empty dict for
    'Custom'/unknown -> caller leaves widgets as they are.

    The sampler comes from FAMILY_GENERATION_DEFAULTS rather than being
    hardcoded to euler_a as it once was. That hardcoding meant picking any
    named preset silently overwrote the family's correct sampler -- handing
    euler_a to SDXL base (which wants dpm++2m) and to FLUX.1 (which wants
    euler per docs/flux.md). One table now answers "what sampler does this
    family want", for the presets, the defaults button and the model-change
    snap alike."""
    size = GENERATION_PRESET_SIZES.get(name)
    if not size:
        return {}
    spec = family_step_cfg(diffuser_name)
    _, step_default = spec["steps"]
    _, _, _, cfg_default = spec["cfg"]
    key = family_step_cfg_key(diffuser_name)
    sampler = FAMILY_GENERATION_DEFAULTS.get(
        key, FAMILY_GENERATION_DEFAULTS["none"])["sampler"]
    w, h = size
    return {
        "imagegen_width": w, "imagegen_height": h,
        "imagegen_steps": step_default,
        "imagegen_cfg_scale": cfg_default,
        "imagegen_sampling": sampler,
    }