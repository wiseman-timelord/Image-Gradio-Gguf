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
# Encoder model constraints (Qwen3-4B)
# Sourced from GGUF metadata to ensure UI sliders/dropdowns and logic 
# respect the actual architecture limits.
# ---------------------------------------------------------------------------
ENCODER_MAX_LAYERS = 36
ENCODER_MAX_CONTEXT = 40960
ENCODER_EMBEDDING_LENGTH = 2560
ENCODER_VOCAB_SIZE = 151936

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
IMAGE_SIZES       = [256, 512, 768, 1024]

# Z-Image-Turbo is a distilled few-step model; step counts are conventionally
# chosen as doubling powers of two (2/4/6/8/10/12) to match the distillation
# schedule the turbo checkpoint was trained against. Restricting the choices
# here keeps the UI from offering values that don't correspond to a step the
# turbo schedule was actually trained on.
STEP_CHOICES      = [2, 4, 6, 8, 10, 12]
BATCH_SIZE_CHOICES = [128, 256, 512, 1024, 2048]

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


def get_persistent_path() -> Path:
    return _get_project_root() / "data" / "persistent.json"


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
        devices.append({
            "index":         idx,
            "name":          sec.get(f"gpu{idx}_name", f"GPU{idx}"),
            "backend":       sec.get(f"gpu{idx}_backend", "Vulkan"),
            "vram_total_mb": int(sec.get(f"gpu{idx}_vram_mb", "0") or 0),
            "vram_free_mb":  int(sec.get(f"gpu{idx}_free_mb", "0") or 0),
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


def _default_persistent() -> Dict[str, Any]:
    """Defaults synthesised from constants.ini, for when persistent.json is
    absent or is missing keys added by a newer version. Kept in step with
    installer.py's write_default_persistent(); that one seeds the file at
    install time, this one backfills gaps at load time."""
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
        "encoder_flash_attn": True,
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
        "imagegen_width": 256,
        "imagegen_height": 256,
        "imagegen_steps": 4,
        "imagegen_cfg_scale": 1.0,
        "imagegen_seed": -1,
        "imagegen_sampling": "euler_a",
        "imagegen_batch_count": 1,
        "imagegen_clip_skip": 2,
        "imagegen_quality_preset": "Fast (Turbo)",
        "output_format": "png",
        "auto_save": True,
        "prompt_template": "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
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


def load_persistent() -> Dict[str, Any]:
    """Load persistent.json, backfilling any keys that are missing because
    the file was written by an older version of the program (e.g. before
    the window-geometry keys existed). Saved values always win; defaults
    only fill genuine gaps, so this never overwrites real user data."""
    path = get_persistent_path()
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    merged = _default_persistent()
                    merged.update(data)
                    return merged
        except (json.JSONDecodeError, IOError):
            pass
    return _default_persistent()


def save_persistent(data: Dict[str, Any]) -> None:
    path = get_persistent_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    tmp.replace(path)


def update_persistent(updates: Dict[str, Any]) -> Dict[str, Any]:
    data = load_persistent()
    data.update(updates)
    save_persistent(data)
    return data


# ---------------------------------------------------------------------------
# Window geometry (Qt app window position/size — see launcher.py)
# ---------------------------------------------------------------------------

def get_window_geometry() -> Dict[str, Any]:
    """Return saved Qt window geometry, validated/sanitized:
        {"x": int, "y": int, "width": int, "height": int, "maximized": bool}
    x/y are WINDOW_GEOMETRY_UNSET (-1) if no position has been saved yet —
    callers should treat that as "let the OS/Qt choose a default position"
    rather than literally moving the window to (-1, -1)."""
    cfg = load_persistent()

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
    update_persistent({
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
    return {
        "Fast (Turbo)": {
            "imagegen_steps": 4, "imagegen_cfg_scale": 1.0,
            "imagegen_sampling": "euler_a",
            "imagegen_width": 256, "imagegen_height": 256,
        },
        "Balanced": {
            "imagegen_steps": 8, "imagegen_cfg_scale": 1.5,
            "imagegen_sampling": "euler_a",
            "imagegen_width": 512, "imagegen_height": 512,
        },
        "Quality": {
            "imagegen_steps": 16, "imagegen_cfg_scale": 7.0,
            "imagegen_sampling": "dpm++2m",
            "imagegen_width": 768, "imagegen_height": 768,
        },
        "Portrait": {
            "imagegen_steps": 8, "imagegen_cfg_scale": 1.5,
            "imagegen_sampling": "euler_a",
            "imagegen_width": 512, "imagegen_height": 768,
        },
        "Widescreen": {
            "imagegen_steps": 8, "imagegen_cfg_scale": 1.5,
            "imagegen_sampling": "euler_a",
            "imagegen_width": 1024, "imagegen_height": 512,
        },
        "Custom": {},
    }