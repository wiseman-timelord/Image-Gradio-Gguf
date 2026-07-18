#!/usr/bin/env python3
"""
installer.py - Standalone setup script for Image Generator GGUF.
Detects hardware (CPU cores/threads, Vulkan GPUs), creates a venv,
installs Python dependencies, then installs llama-cpp-python and
stable-diffusion-cpp-python via pip wheels (pre-built where available,
compiled from source with CPU/Vulkan flags where not).
Writes:
./data/constants.ini      - hardware constants, thread counts, GPU info
./data/configuration.json - default Configuration page settings (only if absent)
./data/preferences.json   - default Preferences page settings (only if absent)
./data/prompt_cache.json  - default Positive/Negative Prompt (history) log (only if absent)
No imports from scripts.* — this is self-contained.
"""
from __future__ import annotations
import argparse
import configparser
import ctypes
import json
import math
import os
import platform
import shutil
import stat
import re
import struct
import subprocess
import sys
import threading
import time
import zipfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Project layout
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent
_DATA_DIR    = _ROOT / "data"
_VENV_DIR    = _ROOT / "venv"
_CONST_PATH  = _DATA_DIR / "constants.ini"
# Two settings files, one per UI page. configuration.json is machine-specific
# (models, backends, threads) and is reseeded here; preferences.json is the
# user's standing taste (prompt template, max thumbnails) and is seeded once.
# _LEGACY_PERSIST_PATH is the pre-split filename, removed on a clean install /
# config refresh; scripts/configure.py renames it to configuration.json if it
# is still lying around at runtime.
_CONFIG_PATH  = _DATA_DIR / "configuration.json"
_PREFS_PATH   = _DATA_DIR / "preferences.json"
_LEGACY_PERSIST_PATH = _DATA_DIR / "persistent.json"
# prompt_cache.json - "Positive/Negative Prompt (history)" rolling log (see
# scripts/configure.py's POSITIVE_HISTORY_KEYS / NEGATIVE_HISTORY_KEYS for the
# single source of truth these mirror). Seeded once, like preferences.json,
# and never purged by a clean install or a config refresh.
_PROMPT_CACHE_PATH = _DATA_DIR / "prompt_cache.json"
_PROMPT_HISTORY_SLOTS = 5
_POSITIVE_HISTORY_KEYS = [f"positive_history_{i}" for i in range(1, _PROMPT_HISTORY_SLOTS + 1)]
_NEGATIVE_HISTORY_KEYS = [f"negative_history_{i}" for i in range(1, _PROMPT_HISTORY_SLOTS + 1)]
_MODELS_DIR  = _ROOT / "models"
_OUTPUT_DIR  = _ROOT / "output"

# Only packages this project imports directly. numpy and Pillow are pulled in
# by gradio; pinning them here only risks fighting gradio's own resolver on a
# future bump, and neither is imported anywhere in scripts/.
REQUIREMENTS = [
    "gradio==6.19.0",
    "PyQt6==6.9.1",
    "PyQt6-WebEngine==6.9.0",
]

# ---------------------------------------------------------------------------
# CPU feature map  --  REPORTING ONLY
# ---------------------------------------------------------------------------
# These keys are detected, written to constants.ini, and shown in the install
# banner / Debug tab. They are deliberately NOT turned into -DGGML_*=ON cmake
# flags any more, because ggml ignores such flags whenever GGML_NATIVE is ON
# (which is the default and what we want -- see _common_cmake_defs below).
# Detection here exists so the program can tell the user the truth about their
# machine, not to steer the build.
#
# "cmake" is retained for display purposes only: it documents which ggml option
# corresponds to each feature, for anyone who later wants GGML_NATIVE=OFF.
CPU_FEATURES = [
    {"key": "has_sse3",   "name": "SSE3",   "cmake": "baseline (no separate ggml toggle)"},
    {"key": "has_ssse3",  "name": "SSSE3",  "cmake": "baseline (no separate ggml toggle)"},
    {"key": "has_sse4_1", "name": "SSE4.1", "cmake": "baseline (no separate ggml toggle)"},
    {"key": "has_sse4_2", "name": "SSE4.2", "cmake": "GGML_SSE42=ON"},
    {"key": "has_avx",    "name": "AVX",    "cmake": "GGML_AVX=ON"},
    {"key": "has_avx2",   "name": "AVX2",   "cmake": "GGML_AVX2=ON"},
    {"key": "has_f16c",   "name": "F16C",   "cmake": "GGML_F16C=ON"},
    {"key": "has_fma",    "name": "FMA",    "cmake": "GGML_FMA=ON"},
    {"key": "has_avx512", "name": "AVX512", "cmake": "GGML_AVX512=ON"},
]

# Windows PF_* constants for IsProcessorFeaturePresent (winnt.h). These report
# what the CPU has AND the OS has enabled via XSAVE -- a feature the OS has not
# enabled is unusable, so this is exactly the right question to ask.
#
# SSSE3 and SSE4.1 have NO corresponding PF_* constant -- Windows never added
# one (confirmed: PF_SSE3_INSTRUCTIONS_AVAILABLE exists and is reported via
# this API, but Microsoft's own docs note the API does not cover SSSE3, and
# there is no SSE4.1 flag either). Per standard practice for x86-64 detection
# (e.g. google/cpu_features), the fix is not a raw CPUID call for two
# instruction sets nobody ships without: every x86-64 chip since ~2006
# (Core 2 / Barcelona onward) has SSSE3 and SSE4.1, so on x86-64 they are
# treated as implied-present rather than probed. See detect_cpu().
_PF_SSE3    = 13
_PF_SSE4_2  = 38
_PF_AVX     = 39
_PF_AVX2    = 40
_PF_AVX512F = 41

# ---------------------------------------------------------------------------
# Backend source build constants
# ---------------------------------------------------------------------------
# Every network/build step gets the same generous budget. These are long,
# failure-prone operations against other people's infrastructure; giving up
# after 3 tries on a flaky link is the installer's problem, not the user's.
# (Git-specific timeout/retry-delay constants -- GIT_CLONE_TIMEOUT,
# GIT_RETRY_DELAYS -- live next to _git_clone() below, where they're used.)
RETRY_ATTEMPTS = 10

# Upstream refs. BOTH are pinned deliberately. sd.cpp previously floated on
# "master", which is the riskier half: its CLI surface is actively changing
# (the --backend module-assignment syntax replaced the older whole-component
# flags), so a floating ref means the program can break with no local edit.
# Bump these two lines together, on purpose, and retest.
# Prebuilt binaries are the PRIMARY acquisition path. They are published as
# GitHub release assets, which -- unlike git and unlike codeload source
# tarballs -- are static objects served with `accept-ranges: bytes`. That makes
# them genuinely resumable: a dropped connection costs only the bytes not yet
# written, not the whole transfer.
#
# They are also better for a public project than a local build. The official
# Windows binaries are compiled with GGML_CPU_ALL_VARIANTS + GGML_BACKEND_DL,
# shipping a ggml-cpu-*.dll per microarchitecture (sse42, sandybridge,
# ivybridge, haswell, skylakex, icelake, cascadelake, cooperlake,
# sapphirerapids, alderlake, cannonlake, piledriver, zen4, x64) and selecting
# the best one at RUNTIME. So one download is optimal on every AMD and Intel
# CPU -- strictly better than GGML_NATIVE, which is optimal only on the machine
# that compiled it. Building from source remains available as a fallback.
LLAMA_CPP_REPO       = "ggml-org/llama.cpp"
LLAMA_CPP_SOURCE_URL = "https://github.com/ggml-org/llama.cpp.git"
LLAMA_CPP_REF        = "b9670"
LLAMA_CPP_SOURCE_DIR = "lc_src"   # Short name keeps paths under MAX_PATH for MSBuild FileTracker

SD_CPP_REPO       = "leejet/stable-diffusion.cpp"
SD_CPP_SOURCE_URL = "https://github.com/leejet/stable-diffusion.cpp.git"
# sd.cpp tags every build as master-<n>-<sha>; the asset name embeds only the
# sha. Pinned rather than floating on "master": its CLI surface is actively
# changing (the --backend module-assignment syntax replaced the older
# whole-component flags), so a floating ref can break the program with no
# local edit. Bump SD_CPP_RELEASE_TAG and LLAMA_CPP_REF together, on purpose.
SD_CPP_RELEASE_TAG = "master-778-c00a9e9"
SD_CPP_REF         = SD_CPP_RELEASE_TAG
# Release TAG is master-<n>-<sha>, but the ASSET name embeds only master-<sha>.
# Keep both; they are not derivable from one another.
SD_CPP_ASSET_STEM  = "master-c00a9e9"
SD_CPP_SOURCE_DIR = "sd_src"      # Short name keeps paths under MAX_PATH for MSBuild FileTracker

LLAMA_BIN_DIR = "data/llama_cpp_binaries"
SD_BIN_DIR    = "data/stable_diffusion_binaries"

# Seconds of COMPLETE SILENCE before a build is considered hung. This is not a
# wall-clock limit: a Vulkan-enabled ggml build legitimately runs for hours
# (every compute shader is compiled), but it is never silent for 15 minutes.
# The previous code passed this to subprocess.run(timeout=) as a total ceiling,
# which killed healthy builds mid-flight and reported "cmake build failed".
BUILD_INACTIVITY_TIMEOUT = 900
CMAKE_CONFIGURE_TIMEOUT  = 600

# llama.cpp reorganised its tools: tools/main/ is gone. `llama-cli` is now an
# interactive chat REPL built on the server stack (tools/cli/, gated behind
# LLAMA_BUILD_SERVER). The one-shot completion binary we need -- the successor
# to the old main.cpp -- is `llama-completion` (tools/completion/).
LLAMA_BIN_NAME = "llama-completion.exe"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def log(msg: str = "") -> None:
    print(f"  {msg}" if msg else "")

def header(title: str) -> None:
    os.system("cls" if platform.system() == "Windows" else "clear")
    print()
    print("  " + "=" * 78)
    print(f"      {title}")
    print("  " + "=" * 78)
    print()

def section(title: str) -> None:
    print()
    print(f"  {title}")
    print("  " + "-" * len(title))

def _safe_rmtree(path: Path) -> bool:
    def _on_error(func, fpath, exc_info):
        try:
            os.chmod(fpath, stat.S_IWRITE)
            func(fpath)
        except Exception:
            pass
            
    if not path.exists():
        return True
    shutil.rmtree(path, onerror=_on_error)
    return not path.exists()

# ---------------------------------------------------------------------------
# Directory setup
# ---------------------------------------------------------------------------
def ensure_dirs() -> None:
    for d in (_DATA_DIR, _MODELS_DIR, _OUTPUT_DIR, _ROOT / "scripts"):
        d.mkdir(parents=True, exist_ok=True)
    init = _ROOT / "scripts" / "__init__.py"
    if not init.exists():
        init.write_text("# scripts package\n", encoding="utf-8")

# ---------------------------------------------------------------------------
# CPU detection
# ---------------------------------------------------------------------------
# Vendor-neutral and generation-neutral by construction: nothing here pattern
# matches on brand strings. That approach cannot work -- retail Intel names
# ("Intel(R) Core(TM) i7-10700K CPU @ 3.80GHz") contain no microarchitecture
# codename at all, and AMD tier names span generations with and without a given
# feature ("Ryzen 9" covers Zen 2, which has no AVX-512, and Zen 5, which does).
# We ask the OS instead, which knows.
# ---------------------------------------------------------------------------

def _has_cpu_feature(pf_const: int) -> bool:
    """Query one instruction set via kernel32!IsProcessorFeaturePresent."""
    if platform.system() != "Windows":
        return False
    try:
        fn = ctypes.windll.kernel32.IsProcessorFeaturePresent
        fn.argtypes = [ctypes.c_uint32]
        fn.restype = ctypes.c_int
        return bool(fn(ctypes.c_uint32(pf_const)))
    except Exception:
        return False


def _cpu_brand() -> str:
    """CPU marketing name. Registry first: it is always present and, unlike
    wmic, is not a deprecated Feature-on-Demand that Microsoft is removing."""
    if platform.system() == "Windows":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
                val, _ = winreg.QueryValueEx(k, "ProcessorNameString")
                if val:
                    return str(val).strip()
        except Exception:
            pass
        try:
            r = subprocess.run(["wmic", "cpu", "get", "Name", "/value"],
                               capture_output=True, text=True, timeout=10)
            for line in r.stdout.splitlines():
                if line.startswith("Name="):
                    name = line.split("=", 1)[1].strip()
                    if name:
                        return name
        except Exception:
            pass
    return platform.processor() or "unknown"


def _cpu_vendor() -> str:
    """AMD / Intel / unknown, from the CPUID vendor string the kernel recorded."""
    if platform.system() == "Windows":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
                vid, _ = winreg.QueryValueEx(k, "VendorIdentifier")
                v = str(vid).strip()
                if "AMD" in v:
                    return "AMD"
                if "Intel" in v:
                    return "Intel"
        except Exception:
            pass
    n = _cpu_brand().lower()
    if any(k in n for k in ("amd", "ryzen", "epyc", "threadripper")):
        return "AMD"
    if any(k in n for k in ("intel", "xeon", "pentium", "celeron")):
        return "Intel"
    return "unknown"


def _physical_cores_winapi() -> int:
    """Physical core count via GetLogicalProcessorInformationEx.

    Walks the variable-length SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX records
    and counts RelationProcessorCore entries -- one per physical core. Needed
    because os.cpu_count() reports SMT threads, and ggml compute threads are
    best matched to physical cores.
    """
    RelationProcessorCore = 0
    try:
        k32 = ctypes.windll.kernel32
        size = ctypes.c_ulong(0)
        k32.GetLogicalProcessorInformationEx(RelationProcessorCore, None,
                                             ctypes.byref(size))
        if size.value == 0:
            return 0
        buf = (ctypes.c_byte * size.value)()
        if not k32.GetLogicalProcessorInformationEx(RelationProcessorCore, buf,
                                                    ctypes.byref(size)):
            return 0
        count, off = 0, 0
        while off + 8 <= size.value:
            rel = ctypes.c_ulong.from_buffer(buf, off).value
            rec = ctypes.c_ulong.from_buffer(buf, off + 4).value
            if rec == 0:
                break
            if rel == RelationProcessorCore:
                count += 1
            off += rec
        return count
    except Exception:
        return 0


def _physical_cores(logical: int) -> int:
    """Physical cores, with graceful degradation to an SMT assumption."""
    if platform.system() == "Windows":
        n = _physical_cores_winapi()
        if n > 0:
            return n
        try:
            r = subprocess.run(["wmic", "cpu", "get", "NumberOfCores", "/value"],
                               capture_output=True, text=True, timeout=10)
            total = 0
            for line in r.stdout.splitlines():
                if line.startswith("NumberOfCores="):
                    v = line.split("=", 1)[1].strip()
                    if v.isdigit():
                        total += int(v)
            if total > 0:
                return total
        except Exception:
            pass
    return max(1, logical // 2)


def detect_cpu() -> Dict[str, Any]:
    """
    Detect CPU identity, topology and instruction sets.

    Instruction sets come from the OS (IsProcessorFeaturePresent), so this is
    correct on any AMD or Intel part, current or future, with no lookup table
    to maintain. F16C and FMA have no PF_ constant; they are inferred from
    AVX2, which is precisely what ggml itself does on MSVC:
        if (GGML_AVX2) list(APPEND ARCH_DEFINITIONS GGML_AVX2 GGML_FMA GGML_F16C)
    Every shipping x86-64 CPU with AVX2 also has F16C and FMA3.

    Thread policy (two different numbers for two different jobs):
      default_threads  = 85% of LOGICAL cores -> written to persistent.json as
                         the program's default -t for llama/sd inference.
      build_jobs       = 85% of LOGICAL cores -> cmake --parallel. Build jobs
                         are I/O-and-process bound, so SMT threads do help here.
      cores_physical   = recorded so the Configuration page can offer it: for
                         ggml compute, one thread per physical core is usually
                         faster than oversubscribing SMT siblings.
    """
    logical  = os.cpu_count() or 4
    physical = _physical_cores(logical)
    default_threads = max(1, math.ceil(logical * 0.85))

    info: Dict[str, Any] = {
        "arch": "x86_64",
        "brand": _cpu_brand(),
        "vendor": _cpu_vendor(),
        "cores_logical": logical,
        "cores_physical": physical,
        "default_threads": default_threads,
        "build_jobs": default_threads,
        "has_aocl": False,
    }

    if platform.system() == "Windows":
        avx2 = _has_cpu_feature(_PF_AVX2)
        info["has_sse3"]   = _has_cpu_feature(_PF_SSE3)
        # No PF_* constant exists for these two -- implied by x86-64 itself
        # (see the comment above _PF_SSE3). Not guessed from anything else.
        info["has_ssse3"]  = True
        info["has_sse4_1"] = True
        info["has_sse4_2"] = _has_cpu_feature(_PF_SSE4_2)
        info["has_avx"]    = _has_cpu_feature(_PF_AVX)
        info["has_avx2"]   = avx2
        info["has_avx512"] = _has_cpu_feature(_PF_AVX512F)
        info["has_f16c"]   = avx2
        info["has_fma"]    = avx2
    else:
        # Non-Windows is not a target platform for this project; report nothing
        # rather than guess. GGML_NATIVE still builds correctly regardless.
        for feat in CPU_FEATURES:
            info[feat["key"]] = False

    for p in (os.environ.get("AOCL_ROOT", ""), os.environ.get("AOCL_PATH", ""),
              r"C:\Program Files\AMD\AOCL", r"C:\AOCL"):
        if p and Path(p).exists():
            info["has_aocl"] = True
            break

    # Informational only -- see _common_cmake_defs(): the build does not consume
    # these. Kept so the Debug tab can show which options WOULD apply if a user
    # ever switched to GGML_NATIVE=OFF for portable binaries.
    info["cmake_flags"] = [f["cmake"] for f in CPU_FEATURES
                            if info.get(f["key"]) and f["cmake"].startswith("GGML_")]
    return info


# ---------------------------------------------------------------------------
# Vulkan / GPU detection  --  TWO PHASES, ON PURPOSE
# ---------------------------------------------------------------------------
# Phase 1 (pre-build): detect_vulkan_presence()
#     Is a Vulkan loader present at all? Cheap, no SDK needed. This only drives
#     the "Compile for Vulkan (detected/not detected)" install menu label.
#
# Phase 2 (post-build): probe_ggml_devices()
#     Ask the binary we just compiled to list its own devices. This is the ONLY
#     authoritative source for the index used by `-dev VulkanN` (llama.cpp) and
#     `--backend vulkanN` (sd.cpp).
#
# Why not vulkaninfo? Because ggml does not enumerate the way vulkaninfo does.
# ggml_vk_instance_init() keeps only Discrete/Integrated GPUs that pass
# ggml_vk_device_is_supported(), then deduplicates devices sharing a deviceUUID
# (one GPU exposed by two drivers, e.g. RADV + AMDVLK) resolving by driver
# priority, and finally numbers whatever survived. vulkaninfo does none of that
# and happily lists CPU/software devices too. So a vulkaninfo index is not a
# ggml index, and the mismatch is silent -- you get the wrong GPU, or none.
# Asking the actual binary removes the guesswork AND the SDK dependency
# (vulkaninfo ships with the Vulkan SDK, not with the driver).
# ---------------------------------------------------------------------------

def detect_vulkan_presence() -> Dict[str, Any]:
    """Phase 1. Is Vulkan usable on this machine? Devices are filled in later
    by probe_ggml_devices(); an empty device list here is expected, not an
    error."""
    result: Dict[str, Any] = {
        "available": False,
        "version": "unknown",
        "sdk": os.environ.get("VULKAN_SDK", ""),
        "devices": [],
    }
    if platform.system() == "Windows":
        try:
            ctypes.windll.LoadLibrary("vulkan-1.dll")
            result["available"] = True
            result["version"] = "1.x"
        except Exception:
            pass
    if not result["available"]:
        for p in (Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "vulkan-1.dll",
                  Path(os.environ.get("VULKAN_SDK", "")) / "Bin" / "vulkan-1.dll"):
            try:
                if str(p) and p.exists():
                    result["available"] = True
                    result["version"] = "1.x"
                    break
            except Exception:
                pass
    # vulkaninfo is optional and used ONLY to prettify the version string.
    vi = shutil.which("vulkaninfo")
    if vi:
        try:
            proc = subprocess.run([vi, "--summary"], capture_output=True,
                                  text=True, timeout=30)
            if proc.returncode == 0:
                result["available"] = True
                for line in proc.stdout.splitlines():
                    for tok in line.replace("=", " ").split():
                        if tok.startswith("1.") and len(tok) >= 5:
                            result["version"] = tok
                            break
                    if result["version"] != "unknown" and result["version"] != "1.x":
                        break
        except Exception:
            pass
    return result


# llama.cpp `--list-devices` prints (common/arg.cpp):
#     Available devices:
#       Vulkan0: AMD Radeon RX 580 (8192 MiB, 7800 MiB free)
# It uses printf, not the log system, so --log-disable cannot suppress it, and
# it already filters out CPU devices for us.
_GGML_DEV_RE = re.compile(
    r"^\s*([A-Za-z][A-Za-z0-9_]*?)(\d+)\s*:\s*(.+?)\s*"
    r"\((\d+)\s*MiB,\s*(\d+)\s*MiB free\)\s*$"
)


def probe_ggml_devices(exe: Path) -> List[Dict[str, Any]]:
    """Phase 2. Enumerate GPUs exactly as ggml sees them, by asking it.

    Returns [{"index", "name", "backend", "vram_total_mb", "vram_free_mb"}, ...]
    ordered as ggml orders them. index is what goes into -dev Vulkan{index}
    and --backend vulkan{index}.

    Any device ggml lists is usable; anything it does not list is not, no
    matter what vulkaninfo or Device Manager claim.
    """
    devices: List[Dict[str, Any]] = []
    if not exe or not Path(exe).exists():
        return devices
    try:
        proc = subprocess.run([str(exe), "--list-devices"],
                              capture_output=True, text=True, timeout=120,
                              encoding="utf-8", errors="replace")
    except Exception as e:
        log(f"  WARNING: could not enumerate devices via {Path(exe).name}: {e}")
        return devices

    for line in (proc.stdout + "\n" + proc.stderr).splitlines():
        m = _GGML_DEV_RE.match(line)
        if not m:
            continue
        backend, idx, name, total, free = m.groups()
        devices.append({
            "index": int(idx),
            "name": name.strip(),
            "backend": backend,
            "vram_total_mb": int(total),
            "vram_free_mb": int(free),
        })
    return devices


# ---------------------------------------------------------------------------
# Write constants.ini
# ---------------------------------------------------------------------------
def write_constants(cpu: Dict[str, Any], vk: Dict[str, Any],
                    use_vulkan: Optional[bool] = None) -> None:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    cfg = configparser.ConfigParser()
    if _CONST_PATH.exists():
        cfg.read(_CONST_PATH, encoding="utf-8")

    # [general] — only written when an install route is explicitly chosen.
    # Detection-only calls pass use_vulkan=None and leave any existing value.
    if use_vulkan is not None:
        if not cfg.has_section("general"):
            cfg.add_section("general")
        cfg["general"]["install_type"] = "vulkan" if use_vulkan else "cpu_only"

    if not cfg.has_section("cpu"):
        cfg.add_section("cpu")
    cfg["cpu"]["brand"]           = cpu["brand"]
    cfg["cpu"]["vendor"]          = cpu["vendor"]
    cfg["cpu"]["arch"]            = cpu["arch"]
    cfg["cpu"]["cores_logical"]   = str(cpu["cores_logical"])
    cfg["cpu"]["cores_physical"]  = str(cpu["cores_physical"])
    cfg["cpu"]["default_threads"] = str(cpu["default_threads"])
    cfg["cpu"]["build_jobs"]      = str(cpu["build_jobs"])
    cfg["cpu"]["has_aocl"]        = str(cpu["has_aocl"])
    # Informational: the build uses GGML_NATIVE=ON and ignores these. Recorded
    # so the Debug tab can show what a GGML_NATIVE=OFF build would enable.
    cfg["cpu"]["cmake_flags"]     = " ".join(cpu["cmake_flags"])
    cfg["cpu"]["arch_selection"]  = "GGML_NATIVE (auto-detected by ggml at build time)"
    
    for feat in CPU_FEATURES:
        cfg["cpu"][feat["key"]] = str(cpu.get(feat["key"], False))
    
    if not cfg.has_section("vulkan"):
        cfg.add_section("vulkan")
    cfg["vulkan"]["available"]    = str(vk["available"])
    cfg["vulkan"]["version"]      = vk["version"]
    cfg["vulkan"]["sdk"]          = vk["sdk"]
    
    # gpu_numbers holds ggml device indices, sourced from `--list-devices` on the
    # binary we compiled -- NOT from vulkaninfo. These are the exact values that
    # go into `-dev Vulkan{n}` and `--backend vulkan{n}`. Names may legitimately
    # contain commas, so per-device keys below are the authoritative store and
    # gpu_names is a convenience mirror only.
    gpu_indices = [str(d["index"]) for d in vk["devices"]]
    gpu_names   = [d["name"] for d in vk["devices"]]
    cfg["vulkan"]["gpu_count"]    = str(len(vk["devices"]))
    cfg["vulkan"]["gpu_numbers"]  = ",".join(gpu_indices)
    cfg["vulkan"]["gpu_names"]    = ",".join(gpu_names)
    cfg["vulkan"]["enumerated_by"] = vk.get("enumerated_by", "not yet probed")

    for d in vk["devices"]:
        i = d["index"]
        cfg["vulkan"][f"gpu{i}_name"]     = d["name"]
        cfg["vulkan"][f"gpu{i}_backend"]  = d.get("backend", "Vulkan")
        cfg["vulkan"][f"gpu{i}_vram_mb"]  = str(d.get("vram_total_mb", 0))
        cfg["vulkan"][f"gpu{i}_free_mb"]  = str(d.get("vram_free_mb", 0))
        
    with open(_CONST_PATH, "w", encoding="utf-8") as f:
        cfg.write(f)
    log(f"constants.ini written → {_CONST_PATH}")

# ---------------------------------------------------------------------------
# Write default configuration.json / preferences.json (only if missing)
# ---------------------------------------------------------------------------
def get_install_type_from_ini() -> str:
    """Read the recorded install type back from constants.ini ('vulkan'/'cpu_only')."""
    try:
        cfg = configparser.ConfigParser()
        cfg.read(_CONST_PATH, encoding="utf-8")
        if cfg.has_section("general"):
            return cfg["general"].get("install_type", "cpu_only").strip().lower()
    except Exception:
        pass
    return "cpu_only"


def _default_gpu_index(vk: Dict[str, Any]) -> int:
    """Pick a sensible starting GPU for a fresh install, on ANY machine.

    Preference: the device with the most free VRAM. On a typical desktop the
    card driving the monitors already has framebuffers and compositor surfaces
    resident, so the spare/passive card tends to win on free VRAM -- which is
    usually the one you want for inference. This is only a starting value; the
    Configuration page lists every device ggml found and the user chooses.
    Returns -1 when there are no GPUs (CPU-only machine or cpu_only install).
    """
    devices = vk.get("devices") or []
    if not devices:
        return -1
    best = max(devices, key=lambda d: (d.get("vram_free_mb", 0), d.get("vram_total_mb", 0)))
    return int(best["index"])


def write_default_configuration(cpu: Dict[str, Any],
                               vk: Optional[Dict[str, Any]] = None) -> None:
    """Seed configuration.json with defaults, if it does not already exist.

    default_threads here is the 85%-of-logical figure detected at install time;
    it becomes the program's default -t for both llama and sd inference, and the
    user can change it on the Configuration page. (See detect_cpu() for why
    cores_physical is also recorded: for ggml compute one thread per physical
    core often beats oversubscribing SMT siblings, so it is offered as a choice.)

    Device defaults are derived from whatever ggml actually enumerated on THIS
    machine -- never a hardcoded index. A machine with no GPU gets -1 and a
    Full CPU placement; a machine with three GPUs gets all three in the UI.

    Prompt Template is NOT written here: it belongs to the Preferences page and
    lives in preferences.json (see write_default_preferences). Keep these two
    key sets disjoint -- a key in both files would be written by whichever page
    the user saved last, and read from whichever file the caller happened to
    load.
    """
    if _CONFIG_PATH.exists():
        return
    vk = vk or {"devices": []}
    dt = cpu["default_threads"]
    gpu = _default_gpu_index(vk)
    has_gpu = gpu >= 0

    defaults: Dict[str, Any] = {
        "encoder_model_path": "", "encoder_model_name": "",
        "imagegen_model_path": "", "imagegen_model_name": "",
        "vae_model_path": "", "vae_model_name": "",
        "last_model_browse_dir": ".\\models",
        "backend_encoder": "CPU",
        "backend_imagegen": "CPU",
        "encoder_threads": dt,
        "encoder_batch_size": 512,
        "encoder_ctx_size": 4096,
        "encoder_flash_attn": True,
        "encoder_gpu_layers": -1,
        # Per-side device indices. Kept separate because the encoder and the
        # diffuser can legitimately sit on different devices (or one on CPU),
        # and a single shared "vulkan_device" key silently cross-wired them.
        "encoder_vulkan_device": gpu,
        "imagegen_vulkan_device": gpu,
        "imagegen_placement": ("Full GPU" if has_gpu else "Full CPU"),
        "imagegen_threads": dt,
        "imagegen_width": 256,
        "imagegen_height": 256,
        "imagegen_steps": 8,
        "imagegen_cfg_scale": 1.5,
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
        "window_x": -1,
        "window_y": -1,
        "window_width": 1280,
        "window_height": 860,
        "window_maximized": False,
    }
    tmp = _CONFIG_PATH.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(defaults, f, indent=4, ensure_ascii=False)
    tmp.replace(_CONFIG_PATH)
    log(f"configuration.json written -> {_CONFIG_PATH}")
    if has_gpu:
        log(f"  default GPU: index {gpu} (most free VRAM); change on the Configuration page")
    else:
        log("  no GPU detected -> defaults are CPU-only")


def write_default_preferences() -> None:
    """Seed preferences.json with defaults, if it does not already exist.

    Takes no hardware arguments because nothing on the Preferences page depends
    on the machine. It is also NOT purged by a clean install or a config
    refresh: these are the user's standing choices, not install state, and
    there is nothing in here that a reinstall could invalidate.

    Kept in step with scripts/configure.py's _default_preferences() /
    DEFAULT_PROMPT_TEMPLATE / DEFAULT_MAX_THUMBNAILS; that one backfills gaps
    at load time, this one seeds the file at install time.
    """
    if _PREFS_PATH.exists():
        log(f"preferences.json already present -> {_PREFS_PATH} (kept)")
        return

    defaults: Dict[str, Any] = {
        "prompt_template": "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
        "max_thumbnails": 50,
    }
    tmp = _PREFS_PATH.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(defaults, f, indent=4, ensure_ascii=False)
    tmp.replace(_PREFS_PATH)
    log(f"preferences.json written -> {_PREFS_PATH}")


def write_default_prompt_cache() -> None:
    """Seed prompt_cache.json with defaults, if it does not already exist.

    10 flat keys — positive_history_1..5, negative_history_1..5, all empty —
    one per history slot shown by the "Positive/Negative Prompt (history)"
    popouts in display.py. Takes no arguments, same as write_default_
    preferences(): nothing here depends on the machine, only on the user
    actually generating something later.

    Like preferences.json, this file is NOT purged by a clean install or a
    config refresh (see _purge_for_clean_install and menu choice 3 in
    main()) — it holds prompts the user typed, not install state, so a
    reinstall has nothing to invalidate here.

    Kept in step with scripts/configure.py's POSITIVE_HISTORY_KEYS /
    NEGATIVE_HISTORY_KEYS / _default_prompt_cache(); that one backfills gaps
    at load time, this one seeds the file at install time.
    """
    if _PROMPT_CACHE_PATH.exists():
        log(f"prompt_cache.json already present -> {_PROMPT_CACHE_PATH} (kept)")
        return

    defaults: Dict[str, Any] = {k: "" for k in _POSITIVE_HISTORY_KEYS + _NEGATIVE_HISTORY_KEYS}
    tmp = _PROMPT_CACHE_PATH.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(defaults, f, indent=4, ensure_ascii=False)
    tmp.replace(_PROMPT_CACHE_PATH)
    log(f"prompt_cache.json written -> {_PROMPT_CACHE_PATH}")


# ---------------------------------------------------------------------------
# venv helpers
# ---------------------------------------------------------------------------
def _venv_python() -> Path:
    if platform.system() == "Windows":
        return _VENV_DIR / "Scripts" / "python.exe"
    return _VENV_DIR / "bin" / "python"

def create_venv() -> bool:
    """Create the venv, retrying transient failures. Fatal if it cannot be made."""
    if _venv_python().exists():
        log(f"venv already exists at {_VENV_DIR}")
        return True
    last = ""
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        log(f"Creating venv at {_VENV_DIR}... (attempt {attempt}/{RETRY_ATTEMPTS})")
        try:
            subprocess.run(
                [sys.executable, "-m", "venv", str(_VENV_DIR)],
                check=True, capture_output=True, text=True, timeout=300,
            )
            log("venv created OK")
            return True
        except subprocess.CalledProcessError as e:
            last = (e.stderr or e.stdout or str(e)).strip()
        except Exception as e:
            last = str(e)
        log(f"  venv creation failed: {last.splitlines()[-1] if last else 'unknown'}")
        if attempt < RETRY_ATTEMPTS:
            time.sleep(min(5 * attempt, 30))
    fatal("Creating the Python virtual environment", last)
    return False


def install_deps() -> bool:
    """Install every requirement, with retries. Any package failing is fatal.

    pip already caches completed wheel downloads, so a retry resumes at package
    granularity rather than starting the whole set again. --retries/--timeout
    make pip itself persist through a flaky link before we even count an
    attempt, which matters for the big wheels (PyQt6-WebEngine-Qt6 is ~127 MB).
    """
    vpy = _venv_python()
    if not vpy.exists():
        fatal("Locating the venv Python",
              f"Expected an interpreter at {vpy} but it is not there.")

    log("Upgrading pip inside venv...")
    try:
        subprocess.run(
            [str(vpy), "-m", "pip", "install", "--upgrade", "pip",
             "--retries", str(RETRY_ATTEMPTS), "--timeout", "60"],
            check=True, capture_output=True, text=True, timeout=600,
        )
    except Exception as e:
        # Non-fatal: a stale-but-working pip can still install everything.
        log(f"pip upgrade warning (continuing): {e}")

    # Flat per-attempt ceiling. Large wheels or a slow connection can
    # legitimately take a while; output is streamed live (not captured) so
    # progress is visible instead of looking frozen.
    PKG_TIMEOUT = 1800
    for req in REQUIREMENTS:
        installed = False
        last = ""
        for attempt in range(1, RETRY_ATTEMPTS + 1):
            log(f"  Installing {req}... (attempt {attempt}/{RETRY_ATTEMPTS}, "
                f"up to {PKG_TIMEOUT // 60} min)")
            try:
                proc = subprocess.run(
                    [str(vpy), "-m", "pip", "install", req,
                     "--retries", str(RETRY_ATTEMPTS), "--timeout", "60"],
                    timeout=PKG_TIMEOUT,
                )
                if proc.returncode == 0:
                    log(f"  {req} OK")
                    installed = True
                    break
                last = f"pip exited {proc.returncode}"
            except subprocess.TimeoutExpired:
                last = f"exceeded the {PKG_TIMEOUT // 60} minute limit"
            except Exception as e:
                last = str(e)
            log(f"  FAILED: {req} - {last}")
            if attempt < RETRY_ATTEMPTS:
                delay = min(5 * attempt, 30)
                log(f"  Retrying in {delay}s (completed downloads are cached)...")
                time.sleep(delay)
        if not installed:
            fatal(f"Installing Python package {req}",
                  f"{RETRY_ATTEMPTS} attempts all failed. Last error: {last}\n"
                  f"pip caches what it already downloaded, so re-running the\n"
                  f"installer will resume rather than start over.")
    return True


# ---------------------------------------------------------------------------
# Build tools detection
# ---------------------------------------------------------------------------
def _find_cmake_in_vs_installations() -> Optional[Path]:
    prog_files_x86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    prog_files     = os.environ.get("ProgramFiles",       r"C:\Program Files")
    install_roots: List[str] = []
    vswhere_exe = Path(prog_files_x86) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if vswhere_exe.exists():
        try:
            result = subprocess.run(
                [str(vswhere_exe), "-all", "-prerelease", "-property", "installationPath"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                install_roots = [p.strip() for p in result.stdout.splitlines() if p.strip()]
        except Exception:
            pass
    for base in (prog_files_x86, prog_files):
        for year in ("2022", "2019"):
            for edition in ("BuildTools", "Enterprise", "Professional", "Community", "Preview"):
                candidate = os.path.join(base, "Microsoft Visual Studio", year, edition)
                if os.path.isdir(candidate) and candidate not in install_roots:
                    install_roots.append(candidate)
    for root in install_roots:
        cmake_bin = os.path.join(root, "Common7", "IDE", "CommonExtensions",
                                 "Microsoft", "CMake", "CMake", "bin")
        cmake_exe = os.path.join(cmake_bin, "cmake.exe")
        if os.path.isfile(cmake_exe):
            return Path(cmake_bin)
    return None

def find_cmake() -> Optional[Path]:
    c = shutil.which("cmake")
    if c:
        return Path(c)
    cmake_bin_dir = _find_cmake_in_vs_installations()
    if cmake_bin_dir:
        os.environ["PATH"] = str(cmake_bin_dir) + os.pathsep + os.environ.get("PATH", "")
        return cmake_bin_dir / "cmake.exe"
    for p in (r"C:\Program Files\CMake\bin\cmake.exe",
              r"C:\Program Files (x86)\CMake\bin\cmake.exe"):
        if Path(p).exists():
            return Path(p)
    return None

def find_git() -> Optional[Path]:
    g = shutil.which("git")
    return Path(g) if g else None


def _common_cmake_defs() -> List[str]:
    """
    cmake defs shared by llama.cpp and stable-diffusion.cpp.

    GGML_NATIVE=ON is what actually selects CPU instruction sets, and it is the
    single most important line in this installer. ggml either runs
    FindSIMD.cmake (MSVC) or passes -march=native (everything else), probing the
    machine performing the build. Because this project always compiles on the
    machine it will run on, that produces per-user optimal binaries on ANY AMD
    or Intel CPU -- Zen 1 through Zen 5, Haswell through Arrow Lake -- with no
    detection table of ours to go stale.

    We deliberately do NOT pass -DGGML_AVX2=ON and friends. When GGML_NATIVE is
    ON, ggml overwrites those options with its own probe results using a plain
    set() (ggml/src/ggml-cpu/cmake/FindSIMD.cmake), and a normal CMake variable
    shadows the cache entry a -D creates. So they were silently inert. Passing
    them only produced a configure log that lied about what was applied.

    If portable binaries are ever needed (build on one machine, ship to others),
    flip GGML_NATIVE to OFF here and pass cpu["cmake_flags"] instead -- and note
    that detect_cpu() must then be trusted, so verify it first.
    """
    return [
        "GGML_NATIVE=ON",
        "BUILD_SHARED_LIBS=OFF",
    ]


# ---------------------------------------------------------------------------
# Build root — short temp path if project root is too deep for MSBuild
# ---------------------------------------------------------------------------
# MSBuild's FileTracker (tracker.exe) cannot create .tlog files when the full
# path exceeds ~260 chars (Windows MAX_PATH). The ggml vulkan-shaders-gen
# ExternalProject adds ~100 chars of its own nesting on top of the build root,
# so any project root longer than ~150 chars risks FTK1011 errors at link time.
#
# Strategy: if the project root is short enough, build inside data/build as
# normal. If it is too long, redirect the entire build tree to C:\build_temp\igg\
# (short and fixed). After binaries are copied out, the temp tree is cleaned up.
#
# No registry modification is made. The system is not permanently altered.

_BUILD_TEMP_ROOT = Path(r"C:\build_temp\igg")
_PATH_SAFETY_THRESHOLD = 150   # chars; project root longer than this -> use temp


def _get_build_root() -> Tuple[Path, bool]:
    """
    Return (build_root_path, using_temp).
    using_temp=True means the caller must delete build_root after copying binaries.
    """
    root_len = len(str(_ROOT))
    if root_len > _PATH_SAFETY_THRESHOLD:
        log(f"  Project root is {root_len} chars — redirecting build to {_BUILD_TEMP_ROOT}")
        log(f"  (temp build tree will be deleted after binaries are copied)")
        return _BUILD_TEMP_ROOT, True
    return _ROOT / "data" / "build", False


def _cleanup_build_temp(build_root: Path, using_temp: bool) -> None:
    """Remove the temporary build tree if one was used."""
    if not using_temp:
        return
    log(f"  Cleaning up temporary build tree at {build_root} ...")
    if _safe_rmtree(build_root):
        log("  Temp build tree removed.")
    else:
        log(f"  WARNING: could not fully remove temp build tree at {build_root}.")
        log("  You may delete it manually.")

# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------
class InstallAborted(Exception):
    """Raised the moment any installation step fails.

    Every step of this installer is required: a venv without gradio cannot
    launch the program, and llama.cpp failing to download does not make
    stable-diffusion.cpp worth building. Previously a failed llama.cpp clone
    logged an error and marched straight on to sd.cpp, then printed a summary
    implying an install had happened. Now the first failure stops everything,
    prints whatever diagnostics were captured, and waits for the user.
    """


def fatal(step: str, detail: str = "") -> None:
    """Abort the install. `detail` should be something the user can act on."""
    raise InstallAborted(f"{step}\n{detail}" if detail else step)


def _report_abort(exc: InstallAborted) -> None:
    """Print the failure report and hold the console until the user reads it."""
    parts = str(exc).split("\n", 1)
    step = parts[0]
    detail = parts[1] if len(parts) > 1 else ""
    print()
    print("=" * 78)
    print("  INSTALLATION FAILED")
    print("=" * 78)
    print()
    print(f"  Step that failed: {step}")
    if detail:
        print()
        for line in detail.splitlines():
            print(f"    {line}")
    print()
    print("-" * 78)
    print("  Nothing further was attempted - every step in this installer is")
    print("  required. Downloaded sources are cached, so re-running will resume")
    print("  from where it stopped rather than starting over.")
    print("=" * 78)
    print()
    try:
        input("  Press Enter to return to the batch menu...")
    except EOFError:
        pass


# ---------------------------------------------------------------------------
# Prebuilt binary acquisition  --  the PRIMARY path
# ---------------------------------------------------------------------------
# Why this exists, and why it is tried before compiling:
#
#  1. It is RESUMABLE. GitHub release assets are static objects served with
#     `accept-ranges: bytes`, so a dropped connection costs only the bytes not
#     yet written. git clone is ATOMIC by contrast: index-pack streams into a
#     temp pack and throws it away on ANY error, so dying 4 KB short of a 35 MB
#     pack loses all 35 MB -- which is exactly the failure mode of the
#     "curl 56 schannel: server closed abruptly (missing close_notify)" bug in
#     Git-for-Windows. No git option makes a clone resumable.
#
#  2. It is FASTER and needs no toolchain. No cmake, no Visual Studio Build
#     Tools, no hour of shader compilation.
#
#  3. It is BETTER ON EVERY CPU than what we can build. Upstream's Windows
#     binaries are compiled with GGML_CPU_ALL_VARIANTS + GGML_BACKEND_DL: the
#     zip carries one ggml-cpu-*.dll per microarchitecture (sse42, sandybridge,
#     ivybridge, haswell, skylakex, icelake, cascadelake, cooperlake,
#     sapphirerapids, alderlake, cannonlake, piledriver, zen4, x64) and ggml
#     selects the best at RUNTIME. GGML_NATIVE, by contrast, is optimal only on
#     the exact machine that ran the compiler. For a public project shipped to
#     unknown hardware, runtime dispatch is strictly the better answer.
#
# Compiling from source remains the automatic fallback.
# ---------------------------------------------------------------------------

_DOWNLOAD_CHUNK = 64 * 1024


def _fmt_mb(n: int) -> str:
    return f"{n / (1024 * 1024):.1f} MB"


def _http_download(url: str, dest: Path,
                   retries: int = RETRY_ATTEMPTS) -> Tuple[bool, str]:
    """Download `url` to `dest`, resuming an interrupted transfer.

    Keeps a `.part` file between attempts and re-requests with
    `Range: bytes=N-`, so repeated network failures cost only the bytes that
    were actually lost. If the server ignores the Range header and replies 200
    instead of 206, we start over rather than corrupt the file by appending.
    """
    part = dest.with_name(dest.name + ".part")
    last_err = ""

    for attempt in range(1, retries + 1):
        have = part.stat().st_size if part.exists() else 0
        req = urllib.request.Request(url, headers={
            "User-Agent": "Image-Gradio-Gguf-installer",
            "Accept": "application/octet-stream",
        })
        if have:
            req.add_header("Range", f"bytes={have}-")
            log(f"    Resuming from {_fmt_mb(have)} (attempt {attempt}/{retries})")
        else:
            log(f"    Downloading (attempt {attempt}/{retries})")

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                code = resp.getcode()
                if have and code != 206:
                    # Range refused -- the only safe move is to start clean.
                    log("    Server ignored resume request - restarting download.")
                    part.unlink()
                    have = 0
                clen = resp.headers.get("Content-Length")
                total = (int(clen) + have) if clen and clen.isdigit() else 0

                mode = "ab" if have else "wb"
                written = have
                next_report = time.time()
                with open(part, mode) as f:
                    while True:
                        chunk = resp.read(_DOWNLOAD_CHUNK)
                        if not chunk:
                            break
                        f.write(chunk)
                        written += len(chunk)
                        if time.time() >= next_report:
                            next_report = time.time() + 0.5
                            if total:
                                pct = 100.0 * written / total
                                sys.stdout.write(
                                    f"\r      {_fmt_mb(written)} / {_fmt_mb(total)}  ({pct:5.1f}%)")
                            else:
                                sys.stdout.write(f"\r      {_fmt_mb(written)}")
                            sys.stdout.flush()
                sys.stdout.write("\n")
                sys.stdout.flush()

            if total and part.stat().st_size != total:
                last_err = (f"size mismatch: got {part.stat().st_size} bytes, "
                            f"expected {total}")
                log(f"    Incomplete ({last_err}) - will resume.")
            else:
                part.replace(dest)
                log(f"    Download complete: {_fmt_mb(dest.stat().st_size)}")
                return True, ""

        except urllib.error.HTTPError as e:
            last_err = f"HTTP {e.code} {e.reason} for {url}"
            if e.code in (404, 403):
                # A missing asset will never appear by retrying.
                return False, last_err
            log(f"    {last_err}")
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            log(f"    Transfer failed: {last_err}  (partial file kept for resume)")

        if attempt < retries:
            delay = min(5 * attempt, 30)
            log(f"    Retrying in {delay}s...")
            time.sleep(delay)

    return False, (f"Download failed after {retries} attempts.\n"
                   f"URL: {url}\nLast error: {last_err}")


def _extract_zip(zip_path: Path, dest_dir: Path,
                 required: List[str]) -> Tuple[bool, str]:
    """Extract a release zip flat into dest_dir and verify what we need is there.

    Upstream's Windows zips are flat (no wrapper directory), but we normalise
    anyway and refuse absolute or parent-relative members rather than trusting
    the archive.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path) as zf:
            bad = zf.testzip()
            if bad is not None:
                return False, f"archive is corrupt (first bad member: {bad})"
            for member in zf.infolist():
                if member.is_dir():
                    continue
                name = Path(member.filename).name
                if not name or name.startswith("..") or Path(member.filename).is_absolute():
                    continue
                with zf.open(member) as srcf, open(dest_dir / name, "wb") as outf:
                    shutil.copyfileobj(srcf, outf)
    except zipfile.BadZipFile as e:
        return False, f"not a valid zip ({e}) - the download may be truncated"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"

    missing = [r for r in required if not (dest_dir / r).exists()]
    if missing:
        return False, f"archive extracted but is missing: {', '.join(missing)}"
    return True, ""


def _install_prebuilt(label: str, repo: str, tag: str, asset: str,
                      bin_dir: Path, required: List[str]) -> Tuple[bool, str]:
    """Fetch and unpack an upstream release zip. Returns (ok, error_detail)."""
    url = f"https://github.com/{repo}/releases/download/{tag}/{asset}"
    cache = _ROOT / "data" / "downloads"
    cache.mkdir(parents=True, exist_ok=True)
    zip_path = cache / asset

    if zip_path.exists():
        log(f"    {asset} already downloaded - reusing.")
    else:
        log(f"    {label} prebuilt: {asset}")
        ok, err = _http_download(url, zip_path)
        if not ok:
            return False, err

    log(f"    Extracting to {bin_dir}...")
    ok, err = _extract_zip(zip_path, bin_dir, required)
    if not ok:
        # A corrupt cached zip must not poison every future run.
        try:
            zip_path.unlink()
            log("    Removed the bad archive; re-running will download it again.")
        except Exception:
            pass
        return False, f"{asset}: {err}"
    return True, ""


# ---------------------------------------------------------------------------
# Git source acquisition
# ---------------------------------------------------------------------------
# One method, run every attempt, no hidden variation between retries.
#
# History, so nobody re-introduces the previous approach: an earlier version
# of this installer escalated git's transport between attempts (HTTP/1.1,
# then OpenSSL as the TLS backend, then compression off), aimed at a known
# Git-for-Windows/schannel teardown bug. In practice this made failures
# strictly harder to diagnose -- "retry" silently became "try something
# different", and the diagnostic pipe it read output through failed to
# surface any of git's real error text on at least one real machine, so the
# failure report showed an empty "Last output:". Reverted in favor of the
# simple approach below, which matches an earlier, confirmed-working version
# of this installer: run plain `git clone`, let it write straight to the
# real console (capture_output=False) so its actual error is always visible,
# and just retry on failure.
#
# A valid existing clone is reused rather than re-downloaded (_clone_is_valid),
# and submodules are fetched as their own retryable step rather than via
# `git clone --recurse-submodules`, so a submodule hiccup doesn't force
# re-downloading the much larger main clone from zero.
# ---------------------------------------------------------------------------

GIT_CLONE_TIMEOUT = 900       # per-attempt wall-clock ceiling, seconds
GIT_RETRY_DELAYS = [5, 10, 15]  # seconds between attempts, last value repeats


def _clone_is_valid(git_exe: Path, dest_dir: Path,
                    sentinel: str = "CMakeLists.txt") -> bool:
    """Is a complete, checked-out clone already here?

    Stops us re-downloading source we already have, and stops a half-written
    directory being mistaken for a usable one. Requires git to agree it has a
    HEAD *and* the working tree to have actually been checked out.
    """
    if not (dest_dir / ".git").exists():
        return False
    if not (dest_dir / sentinel).exists():
        return False
    try:
        p = subprocess.run([str(git_exe), "-C", str(dest_dir), "rev-parse", "HEAD"],
                           capture_output=True, text=True, timeout=30)
        return p.returncode == 0 and len(p.stdout.strip()) >= 7
    except Exception:
        return False


def _git_submodules_needed(dest_dir: Path) -> List[str]:
    """Submodule paths this repo declares, or [] if it has none."""
    gm = dest_dir / ".gitmodules"
    try:
        if not gm.exists() or gm.stat().st_size == 0:
            return []
        paths: List[str] = []
        for line in gm.read_text(encoding="utf-8", errors="replace").splitlines():
            s = line.strip()
            if s.startswith("path"):
                paths.append(s.split("=", 1)[1].strip())
        return paths
    except Exception:
        return []


def _git_fetch_submodules(git_exe: Path, dest_dir: Path,
                          repo_name: str, retries: int = RETRY_ATTEMPTS) -> Tuple[bool, str]:
    """Fetch submodules as their own retryable step, so a submodule hiccup
    doesn't force re-downloading the (much larger) main clone from zero."""
    paths = _git_submodules_needed(dest_dir)
    if not paths:
        return True, ""

    log(f"    {repo_name} declares {len(paths)} submodule(s): {', '.join(paths)}")
    for attempt in range(1, retries + 1):
        log(f"    Fetching submodules - attempt {attempt}/{retries}")
        cmd = [str(git_exe), "-C", str(dest_dir), "submodule", "update",
              "--init", "--recursive", "--depth", "1", "--progress"]
        try:
            proc = subprocess.run(cmd, capture_output=False, text=True,
                                  timeout=GIT_CLONE_TIMEOUT)
            if proc.returncode == 0:
                log("    Submodules OK.")
                return True, ""
            log(f"    Submodule fetch failed (exit {proc.returncode}).")
        except subprocess.TimeoutExpired:
            log("    Submodule fetch timed out.")
        except Exception as e:
            log(f"    Submodule fetch error: {e}")
        if attempt < retries:
            delay = GIT_RETRY_DELAYS[min(attempt - 1, len(GIT_RETRY_DELAYS) - 1)]
            log(f"    Retrying in {delay}s...")
            time.sleep(delay)

    return False, f"Submodule fetch for {repo_name} failed after {retries} attempts."


def _git_clone(repo_url: str, dest_dir: Path, ref: str,
               retries: int = RETRY_ATTEMPTS) -> Tuple[bool, str]:
    """Clone `ref` of `repo_url` into `dest_dir`. Returns (ok, error_detail).

    A valid existing clone is reused rather than re-downloaded, so a failed
    BUILD never costs another 35 MB re-clone.
    """
    git_exe = find_git()
    if git_exe is None:
        return False, "git not found on PATH. Install Git for Windows."

    repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")

    if _clone_is_valid(git_exe, dest_dir):
        log(f"    {repo_name} source already present and valid - skipping download.")
        return _git_fetch_submodules(git_exe, dest_dir, repo_name)

    for attempt in range(1, retries + 1):
        log(f"    Cloning {repo_name} (ref: {ref}) - attempt {attempt}/{retries}")

        # git needs a non-existent or empty target. A failed clone usually
        # cleans up after itself, but not always -- make it certain.
        if dest_dir.exists():
            _safe_rmtree(dest_dir)
        dest_dir.parent.mkdir(parents=True, exist_ok=True)

        cmd = [str(git_exe), "clone", "--depth", "1", "--single-branch",
              "--branch", ref, "--progress", repo_url, str(dest_dir)]
        try:
            proc = subprocess.run(cmd, capture_output=False, text=True,
                                  timeout=GIT_CLONE_TIMEOUT)
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            log(f"    Clone timed out after {GIT_CLONE_TIMEOUT}s.")
            rc = -1
        except Exception as e:
            log(f"    Clone error: {e}")
            rc = -1

        if rc == 0 and _clone_is_valid(git_exe, dest_dir):
            log(f"    {repo_name} clone complete and verified.")
            return _git_fetch_submodules(git_exe, dest_dir, repo_name)

        if rc == 0:
            log("    Clone reported success but the tree is incomplete - retrying.")
        else:
            log(f"    Clone failed (exit {rc}). See git's own error output above.")
        if attempt < retries:
            delay = GIT_RETRY_DELAYS[min(attempt - 1, len(GIT_RETRY_DELAYS) - 1)]
            log(f"    Retrying in {delay}s...")
            time.sleep(delay)

    return False, (f"git clone of {repo_name} failed after {retries} attempts.\n"
                   f"See git's own error output above for the actual cause.")


def _run_with_inactivity_timeout(cmd: List[str], cwd: Optional[str] = None,
                                 timeout: int = BUILD_INACTIVITY_TIMEOUT) -> int:
    """Run a command, streaming its output live, killing it only after `timeout`
    seconds with NO output whatsoever.

    A compile is the wrong shape for subprocess.run(timeout=): total elapsed
    time is unbounded and legitimate, while silence is a reliable hang signal.
    Streaming also means the user watches progress instead of staring at a
    frozen console for an hour wondering if it died.
    """
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace", bufsize=1)
    last_output = [time.time()]
    finished = threading.Event()

    def _pump() -> None:
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                last_output[0] = time.time()
                sys.stdout.write(line)
                sys.stdout.flush()
        except Exception:
            pass
        finally:
            finished.set()

    t = threading.Thread(target=_pump, daemon=True)
    t.start()
    while not finished.wait(5.0):
        if time.time() - last_output[0] > timeout:
            log(f"  ERROR: no build output for {timeout // 60} minutes - assuming hung, killing.")
            try:
                proc.kill()
            except Exception:
                pass
            break
    try:
        proc.wait(timeout=30)
    except Exception:
        pass
    return proc.returncode if proc.returncode is not None else 1


def _run_cmake_build(source_dir: Path, build_dir: Path,
                     cmake_defs: List[str], jobs: int) -> bool:
    cmake_exe = find_cmake()
    if not cmake_exe:
        log("  ERROR: cmake not found. Install CMake or Visual Studio Build Tools.")
        return False

    build_dir.mkdir(parents=True, exist_ok=True)
    d_args = [f"-D{d}" for d in cmake_defs]
    configure_cmd = [
        str(cmake_exe),
        str(source_dir),
        "-B", str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
    ] + d_args

    log(f"  cmake configure: {' '.join(d_args)}")
    try:
        proc = subprocess.run(
            configure_cmd, capture_output=False,
            timeout=CMAKE_CONFIGURE_TIMEOUT, cwd=str(source_dir),
        )
        if proc.returncode != 0:
            log(f"  cmake configure failed (exit {proc.returncode})")
            return False
    except Exception as e:
        log(f"  cmake configure error: {e}")
        return False

    build_cmd = [
        str(cmake_exe), "--build", str(build_dir),
        "--config", "Release",
        "--parallel", str(jobs),
    ]
    log(f"  cmake build (--parallel {jobs}) - long builds are normal, watch for output...")
    rc = _run_with_inactivity_timeout(build_cmd)
    if rc != 0:
        log(f"  cmake build failed (exit {rc})")
        return False
    return True

def _both_paths_failed(prebuilt_error: str, git_error: str) -> str:
    """Compose a failure report. Covers both acquisition routes when both were
    tried; when the user explicitly chose "Compile" the download was never
    attempted, so prebuilt_error is empty and the report says so plainly
    rather than implying a download failure that never happened.
    """
    parts = []
    if prebuilt_error:
        parts.append("Prebuilt download failed:")
        parts.extend("  " + ln for ln in prebuilt_error.splitlines())
        parts.append("")
        parts.append("Building from source also failed:")
    else:
        parts.append("Download was skipped (Compile was chosen). Building from")
        parts.append("source failed:")
    parts.extend("  " + ln for ln in (git_error or "unknown").splitlines())
    parts.append("")
    parts.append("Common causes:")
    parts.append("  * No internet, or a proxy/firewall blocking github.com")
    parts.append("  * A corporate MITM proxy interfering with git's TLS handshake")
    parts.append("  * Antivirus quarantining the downloaded archive")
    return "\n".join(parts)


def _backend_marker(bin_dir: Path) -> Path:
    return bin_dir / ".built_backend"


def _mark_built_backend(bin_dir: Path, use_vulkan: bool, method: str = "download") -> None:
    """Record which backend AND which method (download/compile) these binaries
    came from. method must be 'download' or 'compile'."""
    try:
        backend = "vulkan" if use_vulkan else "cpu"
        _backend_marker(bin_dir).write_text(f"{backend}:{method}", encoding="utf-8")
    except Exception:
        pass


def _built_backend_matches(bin_dir: Path, use_vulkan: bool,
                           method: Optional[str] = None) -> bool:
    """True if the binaries on disk were acquired for the backend (and,
    if given, the method) now requested.

    Without the backend check, "Check/Install" could never change your mind: a
    CPU-only build followed by choosing Vulkan hit `if exe.exists(): skip`, so
    you kept CPU binaries while constants.ini recorded install_type=vulkan --
    the UI then offered GPU backends that could not possibly work, with no
    clue why.

    Without the method check, choosing "Compile" after a prior "Download" (or
    vice versa) would see a same-backend binary already on disk and silently
    skip -- doing the opposite of what was explicitly asked for on the menu.

    An unmarked (pre-upgrade) or malformed marker is treated as a mismatch so
    it gets re-acquired once, correctly.
    """
    want_backend = "vulkan" if use_vulkan else "cpu"
    try:
        raw = _backend_marker(bin_dir).read_text(encoding="utf-8").strip()
        if ":" in raw:
            have_backend, have_method = raw.split(":", 1)
        else:
            have_backend, have_method = raw, None   # pre-upgrade marker format
        if have_backend != want_backend:
            return False
        if method is not None and have_method != method:
            return False
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Backend installation
# ---------------------------------------------------------------------------
def install_llama_cpp(cpu: Dict[str, Any], use_vulkan: bool,
                      force_compile: bool = False) -> str:
    """Acquire llama-completion.exe.

    force_compile=False (default): try the prebuilt download first, compile
    from source only if that fails (used by the "Download..." menu choice).
    force_compile=True: skip the prebuilt download entirely and compile from
    source (used by the "Compile..." menu choice) -- the user asked for a
    build, not a fallback, so a prebuilt asset existing is irrelevant here.
    """
    bin_dir = _ROOT / LLAMA_BIN_DIR
    bin_dir.mkdir(parents=True, exist_ok=True)

    backend = "vulkan" if use_vulkan else "cpu"
    method  = "compile" if force_compile else "download"

    if (bin_dir / LLAMA_BIN_NAME).exists() and _built_backend_matches(bin_dir, use_vulkan, method):
        log(f"  {LLAMA_BIN_NAME} already present ({backend}, {method}), skipping.")
        return "success (already present)"

    if force_compile:
        log(f"  Compiling llama.cpp from source ({backend}) — download skipped by user choice.")
        return compile_llama_cpp(cpu, use_vulkan)

    asset = f"llama-{LLAMA_CPP_REF}-bin-win-{backend}-x64.zip"
    required = [LLAMA_BIN_NAME, "llama-completion-impl.dll", "ggml-base.dll"]
    if use_vulkan:
        required.append("ggml-vulkan.dll")

    log(f"  Trying prebuilt llama.cpp ({backend})...")
    ok, err = _install_prebuilt("llama.cpp", LLAMA_CPP_REPO, LLAMA_CPP_REF,
                                asset, bin_dir, required)
    if ok:
        _mark_built_backend(bin_dir, use_vulkan)
        log(f"  llama.cpp prebuilt installed -> {bin_dir / LLAMA_BIN_NAME}")
        return f"success (prebuilt {backend})"

    log(f"  Prebuilt unavailable: {err.splitlines()[0] if err else 'unknown'}")
    log("  Falling back to compiling from source...")
    return compile_llama_cpp(cpu, use_vulkan, prebuilt_error=err)


def install_sd_cpp(cpu: Dict[str, Any], use_vulkan: bool,
                   force_compile: bool = False) -> str:
    """Acquire sd-cli.exe. See install_llama_cpp for force_compile semantics."""
    bin_dir = _ROOT / SD_BIN_DIR
    bin_dir.mkdir(parents=True, exist_ok=True)

    backend = "vulkan" if use_vulkan else "cpu"
    method  = "compile" if force_compile else "download"

    if (bin_dir / "sd-cli.exe").exists() and _built_backend_matches(bin_dir, use_vulkan, method):
        log(f"  sd-cli.exe already present ({backend}, {method}), skipping.")
        return "success (already present)"

    if force_compile:
        log(f"  Compiling stable-diffusion.cpp from source ({backend}) — download skipped by user choice.")
        return compile_sd_cpp(cpu, use_vulkan)

    asset = f"sd-{SD_CPP_ASSET_STEM}-bin-win-{backend}-x64.zip"
    required = ["sd-cli.exe", "ggml-base.dll"]
    if use_vulkan:
        required.append("ggml-vulkan.dll")

    log(f"  Trying prebuilt stable-diffusion.cpp ({backend})...")
    ok, err = _install_prebuilt("stable-diffusion.cpp", SD_CPP_REPO,
                                SD_CPP_RELEASE_TAG, asset, bin_dir, required)
    if ok:
        _mark_built_backend(bin_dir, use_vulkan)
        log(f"  sd.cpp prebuilt installed -> {bin_dir / 'sd-cli.exe'}")
        return f"success (prebuilt {backend})"

    log(f"  Prebuilt unavailable: {err.splitlines()[0] if err else 'unknown'}")
    log("  Falling back to compiling from source...")
    return compile_sd_cpp(cpu, use_vulkan, prebuilt_error=err)


def compile_llama_cpp(cpu: Dict[str, Any], use_vulkan: bool,
                      prebuilt_error: str = "") -> str:
    """Build llama.cpp and install llama-completion.exe.

    NOT llama-cli.exe: at the pinned ref, tools/cli/ builds an interactive chat
    REPL on top of the server stack (it prints /exit, /regen, /clear commands
    and waits on stdin). Driving that from a subprocess and parsing its stdout
    cannot work. tools/completion/ builds llama-completion, the one-shot tool
    that replaced the old tools/main/.
    """
    bin_dir = _ROOT / LLAMA_BIN_DIR
    bin_dir.mkdir(parents=True, exist_ok=True)

    target = bin_dir / LLAMA_BIN_NAME
    if target.exists() and _built_backend_matches(bin_dir, use_vulkan, "compile"):
        log(f"  {LLAMA_BIN_NAME} already present (compiled for this backend), skipping clone and build.")
        return "success (binary already present)"
    if target.exists():
        log(f"  {LLAMA_BIN_NAME} present but not a matching compiled build - rebuilding.")

    log(f"  {LLAMA_BIN_NAME} not present, cloning and building...")

    build_root, using_temp = _get_build_root()
    build_root.mkdir(parents=True, exist_ok=True)

    src_dir   = build_root / "llama_cpp_src" / LLAMA_CPP_SOURCE_DIR
    build_dir = build_root / "lc_bld"

    ok, detail = _git_clone(LLAMA_CPP_SOURCE_URL, src_dir, LLAMA_CPP_REF)
    if not ok:
        fatal("Acquiring llama.cpp", _both_paths_failed(prebuilt_error, detail))

    cmake_defs: List[str] = _common_cmake_defs() + [
        # Build only what we ship. All of these default ON for a standalone
        # build; the server, its web UI, the unified app, the tests and the
        # examples are a large majority of the compile time and none of them
        # are used by this project. LLAMA_BUILD_TOOLS stays ON because
        # llama-completion lives under tools/.
        "LLAMA_BUILD_TOOLS=ON",
        "LLAMA_BUILD_SERVER=OFF",
        "LLAMA_BUILD_APP=OFF",
        "LLAMA_BUILD_TESTS=OFF",
        "LLAMA_BUILD_EXAMPLES=OFF",
    ]
    if use_vulkan:
        cmake_defs.append("GGML_VULKAN=ON")

    mode = "Vulkan" if use_vulkan else "CPU"
    log(f"  Compiling llama.cpp ({mode}) - this will take a while...")
    jobs = max(1, cpu.get("build_jobs", cpu.get("default_threads", 4)))
    if not _run_cmake_build(src_dir, build_dir, cmake_defs, jobs):
        fatal(f"Compiling llama.cpp ({mode})",
              "cmake build failed - see the output above for the compiler error.\n"
              "The source is cached, so re-running the installer retries the\n"
              "build without re-downloading anything.")

    candidates = [
        build_dir / "bin" / "Release" / LLAMA_BIN_NAME,
        build_dir / "bin" / LLAMA_BIN_NAME,
        build_dir / "Release" / LLAMA_BIN_NAME,
        build_dir / LLAMA_BIN_NAME,
    ]
    src_exe: Optional[Path] = next((c for c in candidates if c.exists()), None)
    if src_exe is None:
        found = list(build_dir.rglob(LLAMA_BIN_NAME))
        src_exe = found[0] if found else None

    if not src_exe:
        fatal(f"Locating {LLAMA_BIN_NAME} after build",
              f"The build reported success but produced no {LLAMA_BIN_NAME}.\n"
              f"Searched under: {build_dir}")

    dest_exe = bin_dir / LLAMA_BIN_NAME
    shutil.copy2(str(src_exe), str(dest_exe))
    for dll in src_exe.parent.glob("*.dll"):
        shutil.copy2(str(dll), str(bin_dir / dll.name))

    _mark_built_backend(bin_dir, use_vulkan, method="compile")
    log(f"  {LLAMA_BIN_NAME} -> {dest_exe}")
    _cleanup_build_temp(build_root, using_temp)
    return f"success (compiled {mode})"


def compile_sd_cpp(cpu: Dict[str, Any], use_vulkan: bool,
                   prebuilt_error: str = "") -> str:
    bin_dir = _ROOT / SD_BIN_DIR
    bin_dir.mkdir(parents=True, exist_ok=True)

    have = (bin_dir / "sd-cli.exe").exists() or (bin_dir / "sd.exe").exists()
    if have and _built_backend_matches(bin_dir, use_vulkan, "compile"):
        log("  sd-cli.exe already present (compiled for this backend), skipping clone and build.")
        return "success (binary already present)"
    if have:
        log("  sd-cli.exe present but not a matching compiled build - rebuilding.")

    log("  sd-cli.exe not present, cloning and building...")

    build_root, using_temp = _get_build_root()
    build_root.mkdir(parents=True, exist_ok=True)

    src_dir   = build_root / "sd_cpp_src" / SD_CPP_SOURCE_DIR
    build_dir = build_root / "sd_bld"

    ok, detail = _git_clone(SD_CPP_SOURCE_URL, src_dir, SD_CPP_REF)
    if not ok:
        fatal("Acquiring stable-diffusion.cpp", _both_paths_failed(prebuilt_error, detail))

    cmake_defs: List[str] = _common_cmake_defs() + ["SD_BUILD_EXAMPLES=ON"]
    if use_vulkan:
        cmake_defs.append("SD_VULKAN=ON")

    sdk = os.environ.get("VULKAN_SDK", "")
    if use_vulkan and sdk:
        cmake_defs.append(f"Vulkan_INCLUDE_DIR={sdk}/Include")
        cmake_defs.append(f"Vulkan_LIBRARY={sdk}/Lib/vulkan-1.lib")

    mode = "Vulkan" if use_vulkan else "CPU"
    log(f"  Compiling stable-diffusion.cpp ({mode}) - this will take a while...")
    jobs = max(1, cpu.get("build_jobs", cpu.get("default_threads", 4)))
    if not _run_cmake_build(src_dir, build_dir, cmake_defs, jobs):
        fatal(f"Compiling stable-diffusion.cpp ({mode})",
              "cmake build failed - see the output above for the compiler error.\n"
              "The source is cached, so re-running the installer retries the\n"
              "build without re-downloading anything.")

    cli_candidates = [
        build_dir / "bin" / "Release" / "sd-cli.exe",
        build_dir / "bin" / "sd-cli.exe",
        build_dir / "Release" / "sd-cli.exe",
        build_dir / "examples" / "cli" / "Release" / "sd-cli.exe",
        build_dir / "bin" / "Release" / "sd.exe",
        build_dir / "bin" / "sd.exe",
        build_dir / "Release" / "sd.exe",
        build_dir / "examples" / "cli" / "Release" / "sd.exe",
        build_dir / "sd.exe",
    ]
    src_exe: Optional[Path] = next((c for c in cli_candidates if c.exists()), None)
    if src_exe is None:
        for name in ("sd-cli.exe", "sd.exe"):
            found = list(build_dir.rglob(name))
            if found:
                src_exe = found[0]
                break

    if not src_exe:
        fatal("Locating sd-cli.exe after build",
              "The build reported success but produced no sd-cli.exe.\n"
              f"Searched under: {build_dir}")

    dest_exe = bin_dir / src_exe.name
    shutil.copy2(str(src_exe), str(dest_exe))
    log(f"  {src_exe.name} -> {dest_exe}")

    srv_src = src_exe.parent / "sd-server.exe"
    if srv_src.exists():
        shutil.copy2(str(srv_src), str(bin_dir / "sd-server.exe"))
        log(f"  sd-server.exe -> {bin_dir / 'sd-server.exe'}")

    for dll in src_exe.parent.glob("*.dll"):
        shutil.copy2(str(dll), str(bin_dir / dll.name))

    _mark_built_backend(bin_dir, use_vulkan, method="compile")


# ---------------------------------------------------------------------------
# Install menu helpers
# ---------------------------------------------------------------------------
def _detect_build_tools() -> Tuple[Optional[Path], Optional[Path]]:
    return find_git(), find_cmake()

def _print_install_banner(cpu: Dict[str, Any], vk: Dict[str, Any]) -> None:
    git, cmake = _detect_build_tools()
    header("Image-Gradio-Gguf — Install Method")
    print()
    print()
    print("  System Detections...")
    print(f"     Platform: Windows {platform.version().split('.')[0] if platform.system() == 'Windows' else platform.system()};"
          f" Python {platform.python_version()}")
    print(f"     Build Tools: Git {'OK' if git else 'NOT FOUND'};"
          f" CMake {'OK' if cmake else 'NOT FOUND'}")
    
    # Clean comma-separated list of detected architecture features from central map
    arch_features = [feat["name"] for feat in CPU_FEATURES if cpu.get(feat["key"])]
    arch_str = ", ".join(arch_features) if arch_features else "Baseline x86_64"
    print(f"     Architecture: {arch_str}")
    
    if vk.get("devices"):
        gpu_str = ", ".join(f"{d['backend']}{d['index']} {d['name']}" for d in vk["devices"])
    else:
        gpu_str = "enumerated after build" if vk.get("available") else "none"
    print(f"     Hardware: {cpu['cores_logical']} logical / {cpu['cores_physical']} physical cores;"
          f" Vulkan {vk['version']}")
    print(f"     GPUs: {gpu_str}")
    print()
    print()
    print("  " + "-" * 79)
    print()
    print()
    print("     1. Clean Install (Purge First)")
    print()
    print("     2. Check/Install (Fix Missing Packages/Libraries)")
    print()
    print("     3. Refresh Configs (Only Remake Ini/Json)")
    print()
    print()
    print()
    print("  " + "=" * 79)

def _purge_for_clean_install() -> None:
    section("Purging previous installation...")
    targets = [
        (_VENV_DIR,                        "venv"),
        (_ROOT / LLAMA_BIN_DIR,            "llama_cpp_binaries"),
        (_ROOT / SD_BIN_DIR,               "stable_diffusion_binaries"),
        (_ROOT / "data" / "build",         "data/build"),
        (_BUILD_TEMP_ROOT,                 "build_temp (C:\\build_temp\\igg)"),
    ]
    for target, label in targets:
        if target.exists():
            log(f"Removing {label} at {target}...")
            if _safe_rmtree(target):
                log(f"{label} removed.")
            else:
                log(f"WARNING: could not fully remove {label} at {target}.")
                log(f"  Close any programs using it (antivirus, explorer) and retry.")
        else:
            log(f"{label} not present, skipping.")
    if _CONFIG_PATH.exists():
        _CONFIG_PATH.unlink()
        log("configuration.json removed (will be regenerated).")
    if _LEGACY_PERSIST_PATH.exists():
        _LEGACY_PERSIST_PATH.unlink()
        log("persistent.json removed (superseded by configuration.json).")
    # preferences.json and prompt_cache.json are deliberately NOT purged --
    # they hold the user's own standing choices and typed prompt history,
    # none of which a reinstall can invalidate.
    if _CONST_PATH.exists():
        _CONST_PATH.unlink()
        log("constants.ini removed (will be regenerated).")

def _run_deps(cpu: Dict[str, Any]) -> None:
    """Both steps are required; create_venv/install_deps raise InstallAborted
    on failure rather than logging a warning and letting main() carry on to the
    compile stage with no usable Python environment."""
    section("Python virtual environment...")
    create_venv()
    section("Python dependencies...")
    install_deps()
    log("All packages installed OK.")

def _missing_build_tools() -> List[str]:
    """Tools required by the Compile path (both llama.cpp and sd.cpp need a
    git clone, then a cmake configure+build). Returns the missing subset, by
    display name, so callers can gate/annotate the Compile menu options with
    something more useful than a failure 5 minutes into a pip install."""
    git, cmake = _detect_build_tools()
    missing = []
    if git is None:
        missing.append("Git")
    if cmake is None:
        missing.append("CMake")
    return missing

def _print_backend_banner(vk: Dict[str, Any], missing_tools: List[str]) -> None:
    header("Image-Gradio-Gguf — Backend Selection")
    print()
    print()
    print()
    print()
    print()
    print()
    print()
    print()
    compile_suffix = f"   [BLOCKED - missing: {', '.join(missing_tools)}]" if missing_tools else ""
    if vk["available"]:
        print(f"     1. Download Llama.Cpp/StableDiffusion for CPU")
        print()
        print(f"     2. Download Llama.Cpp/StableDiffusion for Vulkan")
        print()
        print(f"     3. Compile Llama.Cpp/StableDiffusion for CPU{compile_suffix}")
        print()
        print(f"     4. Compile Llama.Cpp/StableDiffusion for Vulkan{compile_suffix}")
        print()
    else:
        print(f"     1. Download Llama.Cpp/StableDiffusion for CPU")
        print()
        print(f"     2. Compile Llama.Cpp/StableDiffusion for CPU{compile_suffix}")
        print()
    if missing_tools:
        print()
        print(f"  NOTE: Compile options require {', '.join(missing_tools)}, which "
              f"{'was' if len(missing_tools) == 1 else 'were'} not found on PATH")
        print(f"        (or, for CMake, bundled with a Visual Studio install). Install "
              f"{'it' if len(missing_tools) == 1 else 'them'} to unlock Compile,")
        print(f"        or use a Download option above - no build tools required.")
    print()
    print()
    print()
    print()
    print()
    print("  " + "=" * 79)

def _choose_backend(vk: Dict[str, Any]) -> Tuple[bool, bool]:
    """Returns (use_vulkan, force_compile). force_compile=True skips the
    prebuilt-download attempt entirely and goes straight to compiling from
    source, regardless of whether a prebuilt asset exists for this ref.

    Compile options are checked against git/cmake presence BEFORE returning,
    not after: the caller runs a multi-minute venv + pip install right after
    this returns, then a git clone. Discovering a missing tool only at the
    clone step wastes that whole window. Detection here is authoritative and
    re-run fresh each loop, so installing Git/CMake mid-menu and reselecting
    works without restarting the installer.
    """
    has_vk = vk["available"]
    while True:
        missing_tools = _missing_build_tools()
        _print_backend_banner(vk, missing_tools)
        max_choice = 4 if has_vk else 2
        choice = input(f"  Selection; Menu Options = 1-{max_choice}, Abandon Install = A: ").strip().upper()
        if choice == "A":
            print()
            print("  Abandoning install — returning to batch menu.")
            print()
            raise SystemExit(0)

        compile_choice = choice in (("3", "4") if has_vk else ("2",))
        if compile_choice and missing_tools:
            print()
            print(f"  Compile is unavailable - missing: {', '.join(missing_tools)}.")
            print(f"  Install {'it' if len(missing_tools) == 1 else 'them'} and try again, "
                  f"or pick a Download option instead.")
            print()
            input("  Press Enter to continue...")
            continue

        if has_vk:
            if choice == "1":
                return False, False   # download, CPU
            if choice == "2":
                return True, False    # download, Vulkan
            if choice == "3":
                return False, True    # compile, CPU
            if choice == "4":
                return True, True     # compile, Vulkan
        else:
            if choice == "1":
                return False, False   # download, CPU
            if choice == "2":
                return False, True    # compile, CPU
        print()
        print("  Invalid selection, please try again.")
        print()

def _run_build(cpu: Dict[str, Any], use_vulkan: bool, force_compile: bool = False) -> None:
    """Both backends are required. compile_* raise InstallAborted on failure, so
    a llama.cpp failure no longer falls through into an sd.cpp build that could
    not produce a working install anyway."""
    mode = "Vulkan" if use_vulkan else "CPU"
    method = "compile" if force_compile else "download"
    section(f"Backend acquisition  ({mode}, {method})  -  llama.cpp + stable-diffusion.cpp...")
    log("llama.cpp...")
    llama_status = install_llama_cpp(cpu, use_vulkan, force_compile=force_compile)
    log(f"  llama.cpp  ->  {llama_status}")
    log()
    log("stable-diffusion.cpp...")
    sd_status = install_sd_cpp(cpu, use_vulkan, force_compile=force_compile)
    log(f"  stable-diffusion.cpp  ->  {sd_status}")

def _run_summary(t0: float) -> None:
    """Report what is actually on disk, not what was attempted.

    Only reached when no step raised InstallAborted, but it still verifies
    rather than asserts: a summary that lists paths without checking them is
    how an install ends up looking successful when it is not.
    """
    elapsed = round(time.time() - t0, 1)
    section("Installation summary")
    log(f"Time elapsed : {elapsed}s")

    def _state(p: Path, what: str) -> str:
        return f"OK   {p}" if p.exists() else f"MISSING ({what})  {p}"

    log(f"constants.ini: {_state(_CONST_PATH, 'hardware constants')}")
    log(f"configuration: {_state(_CONFIG_PATH, 'configuration page settings')}")
    log(f"preferences  : {_state(_PREFS_PATH, 'preferences page settings')}")
    log(f"venv python  : {_state(_venv_python(), 'python environment')}")
    log(f"llama        : {_state(_ROOT / LLAMA_BIN_DIR / LLAMA_BIN_NAME, 'encoder binary')}")

    sd_dir = _ROOT / SD_BIN_DIR
    sd_exe = next((sd_dir / n for n in ("sd-cli.exe", "sd.exe")
                   if (sd_dir / n).exists()), sd_dir / "sd-cli.exe")
    log(f"sd           : {_state(sd_exe, 'image generation binary')}")
    log()
    log("Press Enter to return to the batch menu...")
    input()

# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------
def run_detection() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Phase 1 hardware detection. GPU *devices* are not enumerated here -- see
    probe_ggml_devices() / _run_build(), which asks the compiled binary."""
    section("Hardware detection...")
    cpu = detect_cpu()
    vk  = detect_vulkan_presence()
    log(f"CPU  : {cpu['brand']}")
    log(f"Arch : {cpu['arch']}  Vendor: {cpu['vendor']}")
    log(f"Cores: {cpu['cores_logical']} logical / {cpu['cores_physical']} physical")
    log(f"       -> {cpu['default_threads']} default inference threads (85% of logical)")
    log(f"       -> {cpu['build_jobs']} build jobs")

    arch_features = [feat["name"] for feat in CPU_FEATURES if cpu.get(feat["key"])]
    arch_str = ", ".join(arch_features) if arch_features else "Baseline x86_64"
    log(f"Features: {arch_str}   (reported by the OS)")
    log("Build   : ggml auto-selects instruction sets for THIS cpu (GGML_NATIVE=ON)")

    log()
    log(f"Vulkan : {vk['available']}  ver={vk['version']}")
    log(f"SDK    : {vk['sdk'] or 'not set'}")
    log("GPUs   : enumerated after build, by the compiled binary itself")
    return cpu, vk


def _enumerate_devices_post_build(vk: Dict[str, Any], use_vulkan: bool) -> Dict[str, Any]:
    """Phase 2. Fill vk["devices"] from the binary we just built.

    Runs `llama-completion --list-devices`, which prints ggml's own device list
    -- the same ggml that sd.cpp vendors, so the indices are valid for both
    `-dev VulkanN` and `--backend vulkanN`.
    """
    vk = dict(vk)
    vk["devices"] = []
    vk["enumerated_by"] = "none"
    if not use_vulkan:
        vk["enumerated_by"] = "n/a (cpu-only install)"
        return vk

    exe = _ROOT / LLAMA_BIN_DIR / LLAMA_BIN_NAME
    section("GPU enumeration (asking the compiled binary)...")
    if not exe.exists():
        log(f"  {LLAMA_BIN_NAME} not found - cannot enumerate GPUs.")
        log("  The Configuration page will offer CPU only.")
        return vk

    devices = probe_ggml_devices(exe)
    vk["devices"] = devices
    vk["enumerated_by"] = f"{LLAMA_BIN_NAME} --list-devices"
    if devices:
        log(f"  ggml reports {len(devices)} device(s):")
        for d in devices:
            log(f"    {d['backend']}{d['index']}: {d['name']}  "
                f"({d['vram_total_mb']} MiB total, {d['vram_free_mb']} MiB free)")
        log("  These indices are what the program will pass to -dev / --backend.")
    else:
        log("  ggml reports no GPU devices.")
        log("  Vulkan drivers may be missing, or the build has no Vulkan support.")
    return vk


def main() -> None:
    parser = argparse.ArgumentParser(description="Image Generator GGUF Installer")
    parser.add_argument("--detect-only",  action="store_true", help="Detect hardware and write constants.ini only")
    parser.add_argument("--deps-only",    action="store_true", help="Create venv and install Python packages only")
    parser.add_argument("--build-only",   action="store_true", help="Build llama.cpp and sd.cpp only")
    args = parser.parse_args()
    
    ensure_dirs()
    header("Image-Gradio-Gguf — Initialize Install")
    
    if args.detect_only:
        cpu, vk = run_detection()
        write_constants(cpu, vk)
        write_default_configuration(cpu, vk)
        write_default_preferences()
        write_default_prompt_cache()
        log("Detection complete.")
        return

    if args.deps_only:
        cpu, vk = run_detection()
        write_constants(cpu, vk)
        write_default_configuration(cpu, vk)
        write_default_preferences()
        write_default_prompt_cache()
        t0 = time.time()
        _run_deps(cpu)
        _run_summary(t0)
        return
        
    if args.build_only:
        cpu, vk = run_detection()
        t0 = time.time()
        use_vulkan, force_compile = _choose_backend(vk)
        write_constants(cpu, vk, use_vulkan=use_vulkan)
        _run_build(cpu, use_vulkan, force_compile=force_compile)
        # Re-write constants now that real GPU indices are knowable.
        write_constants(cpu, _enumerate_devices_post_build(vk, use_vulkan),
                        use_vulkan=use_vulkan)
        _run_summary(t0)
        return
        
    # Interactive menu
    cpu, vk = run_detection()
    while True:
        _print_install_banner(cpu, vk)
        choice = input("  Selection; Menu Options = 1-3, Abandon Install = A: ").strip().upper()
        if choice == "A":
            print()
            print("  Abandoning install — returning to batch menu.")
            print()
            return
        if choice == "1":
            t0 = time.time()
            use_vulkan, force_compile = _choose_backend(vk)
            header("Image-Gradio-Gguf - Installation")
            _purge_for_clean_install()
            write_constants(cpu, vk, use_vulkan=use_vulkan)
            _run_deps(cpu)
            _run_build(cpu, use_vulkan, force_compile=force_compile)
            vk = _enumerate_devices_post_build(vk, use_vulkan)
            write_constants(cpu, vk, use_vulkan=use_vulkan)
            # configuration.json is written LAST so its device defaults can
            # be anchored to a GPU that actually exists on this machine.
            write_default_configuration(cpu, vk)
            write_default_preferences()
            write_default_prompt_cache()
            _run_summary(t0)
            return
        if choice == "2":
            t0 = time.time()
            use_vulkan, force_compile = _choose_backend(vk)
            header("Image-Gradio-Gguf - Installation")
            write_constants(cpu, vk, use_vulkan=use_vulkan)
            _run_deps(cpu)
            _run_build(cpu, use_vulkan, force_compile=force_compile)
            vk = _enumerate_devices_post_build(vk, use_vulkan)
            write_constants(cpu, vk, use_vulkan=use_vulkan)
            write_default_configuration(cpu, vk)
            write_default_preferences()
            write_default_prompt_cache()
            _run_summary(t0)
            return
        if choice == "3":
            t0 = time.time()
            section("Refreshing configs...")
            # Re-enumerate from whatever binaries are already installed, so a
            # config refresh cannot downgrade a good device list to an empty one.
            existing_vulkan = get_install_type_from_ini() == "vulkan"
            vk_now = _enumerate_devices_post_build(vk, existing_vulkan)
            write_constants(cpu, vk_now)
            if _CONFIG_PATH.exists():
                _CONFIG_PATH.unlink()
                log("configuration.json removed (will be regenerated with defaults).")
            if _LEGACY_PERSIST_PATH.exists():
                _LEGACY_PERSIST_PATH.unlink()
                log("persistent.json removed (superseded by configuration.json).")
            write_default_configuration(cpu, vk_now)
            # preferences.json survives a config refresh: nothing in it is
            # machine-derived, so there is nothing to refresh.
            write_default_preferences()
            write_default_prompt_cache()
            _run_summary(t0)
            return
            
        print()
        print("  Invalid selection, please try again.")
        print()

def _main_guarded() -> int:
    """Entry point. Any step raising InstallAborted prints a report, waits for
    the user, and returns a non-zero code to the batch menu. Ctrl+C is treated
    as a deliberate abort, not a crash."""
    try:
        main()
        return 0
    except InstallAborted as e:
        _report_abort(e)
        return 1
    except KeyboardInterrupt:
        print("\n\n  Interrupted by user. Nothing further was attempted.")
        print("  Downloaded sources are cached; re-running will resume.")
        try:
            input("\n  Press Enter to return to the batch menu...")
        except EOFError:
            pass
        return 130
    except Exception:
        # An unexpected crash is still an install failure: show the traceback
        # rather than letting the console close on it, and still pause.
        import traceback
        print()
        print("=" * 78)
        print("  INSTALLATION FAILED - unexpected error")
        print("=" * 78)
        print()
        traceback.print_exc()
        print()
        print("  This is a bug in the installer. Please report the trace above.")
        try:
            input("\n  Press Enter to return to the batch menu...")
        except EOFError:
            pass
        return 1


if __name__ == "__main__":
    sys.exit(_main_guarded())