"""
utilities.py - General utility code: hardware detection, build/package status,
system information helpers. Code not more appropriate in other scripts.
"""

from __future__ import annotations

import ctypes
import os
import platform
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import scripts.configure as configure


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def human_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.1f} KB"
    if size_bytes < 1024 ** 3:
        return f"{size_bytes / (1024 ** 2):.1f} MB"
    return f"{size_bytes / (1024 ** 3):.2f} GB"


def quick_hash(file_path: str, nbytes: int = 8192) -> str:
    import hashlib
    try:
        with open(file_path, "rb") as f:
            return hashlib.md5(f.read(nbytes)).hexdigest()[:12]
    except Exception:
        return "????????????"


# ---------------------------------------------------------------------------
# CPU feature detection
# ---------------------------------------------------------------------------
# CPU feature detection (instruction sets, vendor/brand, thread defaults)
# happens once, at install time, in installer.py's detect_cpu() — the
# canonical detector, run before scripts/ even exists on a fresh install,
# which writes the results to data/constants.ini. At runtime this script's
# package reads that file back via configure.get_cpu_info() /
# configure.get_default_threads() rather than re-detecting; there is no
# detect_cpu_features() here to avoid a second, easily-drifting copy of the
# same logic (see installer.py's CPU_FEATURES list and detect_cpu() for the
# real implementation, and configure.py's CPU_FEATURES mirror for the
# documented list of instruction-set keys that round-trip through
# constants.ini).

# ---------------------------------------------------------------------------
# Vulkan detection  --  intentionally NOT here
# ---------------------------------------------------------------------------
# There used to be a second Vulkan detector in this file that shelled out to
# `vulkaninfo --summary` and parsed it. It is gone, for three reasons:
#
#   1. It could never work. It looked for lines starting with "GPU" containing
#      "=", but --summary prints "GPU0:" on its own line with "deviceName = ..."
#      indented beneath, so it always returned zero devices.
#   2. Even correct, vulkaninfo is the wrong oracle. ggml keeps only supported
#      discrete/integrated GPUs and deduplicates multi-driver duplicates, then
#      numbers the survivors -- so a vulkaninfo index is not a ggml index, and
#      `-dev VulkanN` / `--backend vulkanN` want the ggml one.
#   3. vulkaninfo ships with the Vulkan SDK, not the driver, so depending on it
#      broke machines that had working Vulkan but no SDK.
#
# The installer now asks the compiled binary (`llama-completion --list-devices`)
# and records the result in constants.ini. At runtime, read it back with
# configure.get_vulkan_info(). One source of truth, and it is ggml's own.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Memory info
# ---------------------------------------------------------------------------

def get_memory_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    if platform.system() != "Windows":
        return info
    try:
        class MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        mem = MS()
        mem.dwLength = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem))
        total = mem.ullTotalPhys // (1024 * 1024)
        avail = mem.ullAvailPhys // (1024 * 1024)
        info = {"ram_total_mb": total, "ram_used_mb": total - avail,
                "ram_percent": mem.dwMemoryLoad}
    except Exception:
        pass
    return info


# ---------------------------------------------------------------------------
# Build / binary status
# ---------------------------------------------------------------------------
# The installer compiles llama.cpp and stable-diffusion.cpp from source and
# places the executables in:
#   ./data/llama_cpp_binaries/llama-completion.exe
#   ./data/stable_diffusion_binaries/sd-cli.exe  (sd-server.exe also copied)
#
# get_build_status() checks those locations first, then falls back to PATH.
# ---------------------------------------------------------------------------

def _find_exe_in_dir(directory: Path, names: List[str]) -> Optional[Path]:
    """Return the first name found in directory, or None."""
    for name in names:
        p = directory / name
        if p.exists():
            return p
    return None


def get_build_status() -> Dict[str, Any]:
    """
    Check whether the compiled C++ backend executables are present.

    Looks in:
      1. ./data/llama_cpp_binaries/       for llama-completion.exe
      2. ./data/stable_diffusion_binaries/ for sd-cli.exe (or legacy sd.exe)
      3. PATH (shutil.which) as fallback

    Returns a dict compatible with launcher.py and display.py:
        llama_built  : bool  - llama-completion.exe found
        llama_path   : str   - path string or ""
        sd_built     : bool  - sd-cli.exe (or sd.exe) found
        sd_path      : str   - path string or ""
    """
    llama_bin_dir = configure.get_llama_bin_dir()
    sd_bin_dir    = configure.get_sd_bin_dir()

    # llama-completion (NOT llama-cli, which is now an interactive chat REPL)
    llama_exe = _find_exe_in_dir(llama_bin_dir,
                                 ["llama-completion.exe", "llama-completion"])
    if not llama_exe:
        found = shutil.which("llama-completion")
        llama_exe = Path(found) if found else None

    # sd-cli.exe is the current output name; sd.exe kept as legacy fallback
    sd_exe = _find_exe_in_dir(sd_bin_dir, ["sd-cli.exe", "sd-cli", "sd.exe", "sd"])
    if not sd_exe:
        found = shutil.which("sd-cli") or shutil.which("sd")
        sd_exe = Path(found) if found else None

    return {
        "llama_built":  llama_exe is not None,
        "llama_path":   str(llama_exe) if llama_exe else "",
        "sd_built":     sd_exe is not None,
        "sd_path":      str(sd_exe) if sd_exe else "",
    }


# ---------------------------------------------------------------------------
# Tool finders (cmake / git) — kept for any future use; not used at runtime
# ---------------------------------------------------------------------------

_CMAKE_CANDIDATES = [
    r"C:\Program Files\CMake\bin\cmake.exe",
    r"C:\Program Files (x86)\CMake\bin\cmake.exe",
]


def find_cmake() -> Optional[Path]:
    c = shutil.which("cmake")
    if c:
        return Path(c)
    for p in _CMAKE_CANDIDATES:
        if Path(p).exists():
            return Path(p)
    return None


def find_git() -> Optional[Path]:
    g = shutil.which("git")
    return Path(g) if g else None


def check_prerequisites() -> Dict[str, Any]:
    cmake, git = find_cmake(), find_git()
    return {
        "cmake": cmake is not None, "cmake_path": str(cmake) if cmake else "",
        "git": git is not None, "git_path": str(git) if git else "",
        "vulkan_sdk": bool(os.environ.get("VULKAN_SDK")),
        "vulkan_sdk_path": os.environ.get("VULKAN_SDK", ""),
        "ninja": shutil.which("ninja") is not None,
    }


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def get_relevant_env() -> Dict[str, str]:
    keys = ["VULKAN_SDK", "VK_INSTANCE_LAYERS", "VK_LAYER_PATH",
            "CUDA_PATH", "HIP_PATH", "GGML_VK_VISIBLE_DEVICES",
            "AOCL_ROOT", "AOCL_PATH", "NUMBER_OF_PROCESSORS",
            "PROCESSOR_ARCHITECTURE", "PATH"]
    result: Dict[str, str] = {}
    for k in keys:
        v = os.environ.get(k, "")
        if v:
            if k == "PATH" and len(v) > 200:
                v = v[:200] + "..."
            result[k] = v
    return result