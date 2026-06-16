#!/usr/bin/env python3
"""
installer.py - Standalone setup script for Image Generator GGUF.

Detects hardware (CPU cores/threads, Vulkan GPUs), creates a venv,
installs Python dependencies into it, then builds llama.cpp and
stable-diffusion.cpp with optimal flags for Zen 2 + Vulkan.

Writes:
    ./data/constants.ini   - hardware constants, thread counts, GPU info
    ./data/persistent.json - default user config (only if absent)

No imports from scripts.* — this is self-contained.

Usage (called by batch menu, option 2):
    python installer.py
    python installer.py --deps-only
    python installer.py --build-only
    python installer.py --detect-only
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
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Project layout
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent
_DATA_DIR    = _ROOT / "data"
_VENV_DIR    = _ROOT / "venv"
_BUILD_DIR   = _DATA_DIR / "build"
_LLAMA_DIR   = _BUILD_DIR / "llama.cpp"
_SD_DIR      = _BUILD_DIR / "sd.cpp"
_CONST_PATH  = _DATA_DIR / "constants.ini"
_PERSIST_PATH = _DATA_DIR / "persistent.json"
_MODELS_DIR  = _ROOT / "models"
_OUTPUT_DIR  = _ROOT / "output"

REQUIREMENTS = [
    "gradio>=5.0",
    "Pillow>=10.0",
    "numpy>=1.26",
]


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
    """Inline section label — no cls, scrolls with the install log."""
    print()
    print(f"  {title}")
    print("  " + "-" * len(title))


# ---------------------------------------------------------------------------
# Directory setup
# ---------------------------------------------------------------------------

def ensure_dirs() -> None:
    for d in (_DATA_DIR, _BUILD_DIR, _MODELS_DIR, _OUTPUT_DIR,
              _ROOT / "scripts"):
        d.mkdir(parents=True, exist_ok=True)
    # Ensure scripts/__init__.py exists
    init = _ROOT / "scripts" / "__init__.py"
    if not init.exists():
        init.write_text("# scripts package\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CPU detection
# ---------------------------------------------------------------------------

def detect_cpu() -> Dict[str, Any]:
    logical = os.cpu_count() or 4
    default_threads = max(1, math.ceil(logical * 0.85))
    arch = _cpu_arch()
    brand = platform.processor() or "unknown"
    vendor = "unknown"

    info: Dict[str, Any] = {
        "arch": arch,
        "brand": brand,
        "vendor": vendor,
        "cores_logical": logical,
        "default_threads": default_threads,
        "has_avx": False,
        "has_avx2": False,
        "has_f16c": False,
        "has_fma": False,
        "has_avx512": False,
        "has_sse4_2": False,
        "has_aocl": False,
    }

    if platform.system() == "Windows":
        try:
            r = subprocess.run(["wmic", "cpu", "get", "Name", "/value"],
                               capture_output=True, text=True, timeout=10)
            for line in r.stdout.strip().splitlines():
                if line.startswith("Name="):
                    info["brand"] = line.split("=", 1)[1].strip()
                    break
        except Exception:
            pass

    # Try py-cpuinfo for exact flags
    try:
        import cpuinfo  # type: ignore
        ci = cpuinfo.get_cpu_info()
        fl = [x.lower() for x in ci.get("flags", [])]
        info.update(
            has_avx="avx" in fl,
            has_avx2="avx2" in fl,
            has_f16c="f16c" in fl,
            has_fma="fma" in fl,
            has_sse4_2="sse4_2" in fl,
            has_avx512=any("avx512" in x for x in fl),
        )
        if ci.get("brand_raw"):
            info["brand"] = ci["brand_raw"]
    except ImportError:
        # Fallback: infer from CPU name
        n = info["brand"].lower()
        is_amd = any(k in n for k in ("amd", "ryzen", "epyc", "threadripper"))
        is_intel = any(k in n for k in ("intel", "core", "xeon"))
        if is_amd:
            info["vendor"] = "AMD"
            info.update(has_avx=True, has_avx2=True, has_f16c=True,
                        has_fma=True, has_sse4_2=True)
        elif is_intel:
            info["vendor"] = "Intel"
            info.update(has_avx=True, has_sse4_2=True)
            if any(k in n for k in ("haswell", "broadwell", "skylake",
                                    "kaby", "coffee", "comet", "ice",
                                    "tiger", "alder", "raptor", "arrow",
                                    "meteor", "ultra")):
                info.update(has_avx2=True, has_f16c=True, has_fma=True)
        elif arch == "x86_64":
            info.update(has_avx=True, has_avx2=True, has_f16c=True,
                        has_fma=True, has_sse4_2=True)

    # AOCL
    for p in (os.environ.get("AOCL_ROOT", ""), os.environ.get("AOCL_PATH", ""),
              r"C:\Program Files\AMD\AOCL", r"C:\AOCL"):
        if p and Path(p).exists():
            info["has_aocl"] = True
            break

    # Vendor from brand if not set
    if info["vendor"] == "unknown":
        n = info["brand"].lower()
        if any(k in n for k in ("amd", "ryzen", "epyc")):
            info["vendor"] = "AMD"
        elif "intel" in n:
            info["vendor"] = "Intel"

    # cmake flags
    flags = []
    for feat, flag in (("has_avx", "GGML_AVX=ON"), ("has_avx2", "GGML_AVX2=ON"),
                       ("has_f16c", "GGML_F16C=ON"), ("has_fma", "GGML_FMA=ON"),
                       ("has_avx512", "GGML_AVX512=ON")):
        if info[feat]:
            flags.append(flag)
    info["cmake_flags"] = flags

    return info


def _cpu_arch() -> str:
    m = platform.machine().lower()
    if m in ("amd64", "x86_64"):
        return "x86_64"
    if m in ("i386", "i686", "x86"):
        return "x86"
    if "aarch64" in m:
        return "aarch64"
    return m


# ---------------------------------------------------------------------------
# Vulkan / GPU detection
# ---------------------------------------------------------------------------
def _parse_vk_devices_from_text(text: str) -> List[Dict[str, Any]]:
    """Parse GPU devices from full vulkaninfo text output.
    Lines look like: "GPU id = 0 (NVIDIA GeForce GTX 1060 3GB)"
    Captures full name even when it contains parentheses.
    """
    import re
    devices = []
    # Pattern: GPU id = digits, space, (, any characters, ) at end of line
    pattern = re.compile(r"GPU id = (\d+)\s*\((.*)\)$")
    for line in text.splitlines():
        m = pattern.search(line)
        if m:
            idx = int(m.group(1))
            name = m.group(2).strip()
            if not any(d["index"] == idx for d in devices):
                devices.append({"index": idx, "name": name, "type": ""})
    # If the above fails, fallback to block parsing (GPU0: ... deviceName = ...)
    if not devices:
        devices = _parse_vk_devices_from_blocks(text)
    return devices

def _parse_vk_devices_from_blocks(text: str) -> List[Dict[str, Any]]:
    """Fallback: parse from GPU0: / GPU1: blocks."""
    import re
    devices = []
    blocks = re.split(r'\nGPU(\d+):\n', text)
    for i in range(1, len(blocks), 2):
        idx = int(blocks[i])
        block = blocks[i+1]
        name_match = re.search(r'deviceName\s*=\s*([^\n]+)', block)
        name = name_match.group(1).strip() if name_match else f"GPU{idx}"
        type_match = re.search(r'deviceType\s*=\s*([^\n]+)', block)
        dev_type = type_match.group(1).strip() if type_match else ""
        devices.append({"index": idx, "name": name, "type": dev_type})
    return devices

def detect_vulkan() -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "available": False,
        "version": "unknown",
        "sdk": os.environ.get("VULKAN_SDK", ""),
        "devices": [],        # list of {"index": int, "name": str, "type": str}
    }

    vi = shutil.which("vulkaninfo")
    if not vi:
        return result

    # Try JSON output first (most reliable)
    try:
        proc = subprocess.run([vi, "--json"], capture_output=True, text=True, timeout=30)
        if proc.returncode == 0 and proc.stdout.strip():
            data = json.loads(proc.stdout)
            devices = []
            if "physicalDevices" in data:
                dev_list = data["physicalDevices"]
            elif isinstance(data, list):
                dev_list = data
            else:
                dev_list = []
                for v in data.values():
                    if isinstance(v, list) and v and "deviceName" in v[0]:
                        dev_list = v
                        break
            for i, d in enumerate(dev_list):
                if isinstance(d, dict):
                    idx = d.get("deviceID", d.get("deviceId", i))
                    name = d.get("deviceName", f"GPU{idx}")
                    dev_type = d.get("deviceType", "")
                    devices.append({"index": idx, "name": name, "type": dev_type})
            if devices:
                result["available"] = True
                result["devices"] = devices
                result["version"] = _parse_vk_version("")
                return result
    except Exception:
        pass

    # Fallback: parse full text output
    try:
        proc = subprocess.run([vi], capture_output=True, text=True, timeout=30)
        if proc.returncode == 0:
            result["available"] = True
            result["devices"] = _parse_vk_devices_from_text(proc.stdout)
            result["version"] = _parse_vk_version(proc.stdout)
    except Exception:
        pass

    # Last resort: check if DLL exists (Windows)
    if not result["available"] and platform.system() == "Windows":
        try:
            ctypes.windll.LoadLibrary("vulkan-1.dll")
            result["available"] = True
            result["version"] = "1.x"
        except Exception:
            pass

    return result


def _parse_vk_version(stdout: str) -> str:
    for line in stdout.splitlines():
        for tok in line.split():
            if tok.startswith("1.") and len(tok) >= 3:
                return tok
    return "detected"

def _parse_vk_devices(stdout: str) -> List[Dict[str, Any]]:
    devices: List[Dict[str, Any]] = []
    current: Dict[str, Any] = {}
    for line in stdout.splitlines():
        s = line.strip()
        if s.startswith("GPU") and "=" in s:
            if current:
                devices.append(current)
            idx_str = s.split("=")[0].replace("GPU", "").strip()
            try:
                idx = int(idx_str)
            except ValueError:
                idx = len(devices)
            current = {"index": idx, "name": s.split("=", 1)[1].strip(), "type": ""}
        elif current:
            sl = s.lower().replace(" ", "")
            if sl.startswith("devicetype"):
                current["type"] = s.split("=", 1)[-1].strip()
    if current:
        devices.append(current)
    return devices


# ---------------------------------------------------------------------------
# Write constants.ini
# ---------------------------------------------------------------------------

def write_constants(cpu: Dict[str, Any], vk: Dict[str, Any]) -> None:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    cfg = configparser.ConfigParser()

    # Read existing so we don't wipe user edits in other sections
    if _CONST_PATH.exists():
        cfg.read(_CONST_PATH, encoding="utf-8")

    # [cpu]
    if not cfg.has_section("cpu"):
        cfg.add_section("cpu")
    cfg["cpu"]["brand"]           = cpu["brand"]
    cfg["cpu"]["vendor"]          = cpu["vendor"]
    cfg["cpu"]["arch"]            = cpu["arch"]
    cfg["cpu"]["cores_logical"]   = str(cpu["cores_logical"])
    cfg["cpu"]["default_threads"] = str(cpu["default_threads"])
    cfg["cpu"]["has_avx"]         = str(cpu["has_avx"])
    cfg["cpu"]["has_avx2"]        = str(cpu["has_avx2"])
    cfg["cpu"]["has_f16c"]        = str(cpu["has_f16c"])
    cfg["cpu"]["has_fma"]         = str(cpu["has_fma"])
    cfg["cpu"]["has_avx512"]      = str(cpu["has_avx512"])
    cfg["cpu"]["has_sse4_2"]      = str(cpu["has_sse4_2"])
    cfg["cpu"]["has_aocl"]        = str(cpu["has_aocl"])
    cfg["cpu"]["cmake_flags"]     = " ".join(cpu["cmake_flags"])

    # [vulkan]
    if not cfg.has_section("vulkan"):
        cfg.add_section("vulkan")
    cfg["vulkan"]["available"]    = str(vk["available"])
    cfg["vulkan"]["version"]      = vk["version"]
    cfg["vulkan"]["sdk"]          = vk["sdk"]
    # gpu_count: number of discrete GPUs found
    gpu_indices = [str(d["index"]) for d in vk["devices"]]
    gpu_names   = [d["name"] for d in vk["devices"]]
    cfg["vulkan"]["gpu_count"]    = str(len(vk["devices"]))
    cfg["vulkan"]["gpu_numbers"]  = ",".join(gpu_indices)   # e.g. "0,1"
    cfg["vulkan"]["gpu_names"]    = ",".join(gpu_names)     # e.g. "RX 580,RX 470"
    # Per-GPU entries for easy lookup
    for d in vk["devices"]:
        cfg["vulkan"][f"gpu{d['index']}_name"] = d["name"]
        cfg["vulkan"][f"gpu{d['index']}_type"] = d.get("type", "")

    with open(_CONST_PATH, "w", encoding="utf-8") as f:
        cfg.write(f)
    log(f"constants.ini written → {_CONST_PATH}")


# ---------------------------------------------------------------------------
# Write default persistent.json (only if missing)
# ---------------------------------------------------------------------------

def write_default_persistent(cpu: Dict[str, Any]) -> None:
    if _PERSIST_PATH.exists():
        return  # never overwrite user config
    dt = cpu["default_threads"]
    defaults: Dict[str, Any] = {
        "encoder_model_path": "", "encoder_model_name": "",
        "imagegen_model_path": "", "imagegen_model_name": "",
        "vae_model_path": "", "vae_model_name": "",
        "backend_encoder": "CPU",
        "backend_imagegen": "CPU",
        "encoder_threads": dt,
        "encoder_batch_size": 512,
        "encoder_ctx_size": 4096,
        "encoder_flash_attn": True,
        "encoder_gpu_layers": -1,
        "imagegen_threads": dt,
        "imagegen_width": 512,
        "imagegen_height": 512,
        "imagegen_steps": 4,
        "imagegen_cfg_scale": 1.0,
        "imagegen_seed": -1,
        "imagegen_sampling": "euler_a",
        "imagegen_batch_count": 1,
        "imagegen_clip_skip": 2,
        "vulkan_device": 0,
        "output_format": "png",
        "auto_save": True,
        "prompt_template": "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
        "negative_prompt": "",
        "ui_theme": "Default",
        "first_run": True,
    }
    tmp = _PERSIST_PATH.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(defaults, f, indent=4, ensure_ascii=False)
    tmp.replace(_PERSIST_PATH)
    log(f"persistent.json written → {_PERSIST_PATH}")


# ---------------------------------------------------------------------------
# venv helpers
# ---------------------------------------------------------------------------

def _venv_python() -> Path:
    if platform.system() == "Windows":
        return _VENV_DIR / "Scripts" / "python.exe"
    return _VENV_DIR / "bin" / "python"


def _venv_pip() -> Path:
    if platform.system() == "Windows":
        return _VENV_DIR / "Scripts" / "pip.exe"
    return _VENV_DIR / "bin" / "pip"


def create_venv() -> bool:
    if _venv_python().exists():
        log(f"venv already exists at {_VENV_DIR}")
        return True
    log(f"Creating venv at {_VENV_DIR} ...")
    try:
        subprocess.run(
            [sys.executable, "-m", "venv", str(_VENV_DIR)],
            check=True, capture_output=True, text=True, timeout=120,
        )
        log("venv created OK")
        return True
    except Exception as e:
        log(f"ERROR creating venv: {e}")
        return False


def install_deps() -> bool:
    vpy = _venv_python()
    if not vpy.exists():
        log("ERROR: venv python not found — run venv creation first")
        return False
    log("Upgrading pip inside venv...")
    try:
        subprocess.run(
            [str(vpy), "-m", "pip", "install", "--upgrade", "pip"],
            check=True, capture_output=True, text=True, timeout=120,
        )
    except Exception as e:
        log(f"pip upgrade warning: {e}")

    all_ok = True
    for req in REQUIREMENTS:
        log(f"  Installing {req} ...")
        try:
            subprocess.run(
                [str(vpy), "-m", "pip", "install", req],
                check=True, capture_output=True, text=True, timeout=300,
            )
            log(f"  {req} OK")
        except subprocess.CalledProcessError as e:
            log(f"  FAILED: {req}")
            log(f"    {e.stderr[-300:] if e.stderr else ''}")
            all_ok = False
    return all_ok


# ---------------------------------------------------------------------------
# Build tools detection
# ---------------------------------------------------------------------------
def _find_cmake_in_vs_installations() -> Optional[Path]:
    """Search for cmake.exe inside Visual Studio / Build Tools installations.
    Returns Path to the directory containing cmake.exe, or None.
    """
    prog_files_x86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    prog_files     = os.environ.get("ProgramFiles",       r"C:\Program Files")
    install_roots: List[str] = []

    # 1. Use vswhere to get all installation paths
    vswhere_exe = Path(prog_files_x86) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if vswhere_exe.exists():
        try:
            result = subprocess.run(
                [
                    str(vswhere_exe),
                    "-all",               # all installed products
                    "-prerelease",        # include pre-release
                    "-property", "installationPath",
                ],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                install_roots = [p.strip() for p in result.stdout.splitlines() if p.strip()]
        except Exception:
            pass

    # 2. Hard-coded default roots for VS 2019 & 2022, all editions
    for base in (prog_files_x86, prog_files):
        for year in ("2022", "2019"):
            for edition in ("BuildTools", "Enterprise", "Professional", "Community", "Preview"):
                candidate = os.path.join(base, "Microsoft Visual Studio", year, edition)
                if os.path.isdir(candidate) and candidate not in install_roots:
                    install_roots.append(candidate)

    # 3. Walk each root looking for cmake.exe
    for root in install_roots:
        cmake_bin = os.path.join(root, "Common7", "IDE", "CommonExtensions",
                                 "Microsoft", "CMake", "CMake", "bin")
        cmake_exe = os.path.join(cmake_bin, "cmake.exe")
        if os.path.isfile(cmake_exe):
            return Path(cmake_bin)

    return None

def find_cmake() -> Optional[Path]:
    # First check PATH
    c = shutil.which("cmake")
    if c:
        return Path(c)
    # Then search inside VS installations
    cmake_bin_dir = _find_cmake_in_vs_installations()
    if cmake_bin_dir:
        # Also add to PATH for subsequent subprocesses
        os.environ["PATH"] = str(cmake_bin_dir) + os.pathsep + os.environ.get("PATH", "")
        return cmake_bin_dir / "cmake.exe"
    # Fallback to hardcoded paths
    for p in (r"C:\Program Files\CMake\bin\cmake.exe",
              r"C:\Program Files (x86)\CMake\bin\cmake.exe"):
        if Path(p).exists():
            return Path(p)
    return None

def find_git() -> Optional[Path]:
    g = shutil.which("git")
    return Path(g) if g else None

def _clean_build_dir(build_path: Path) -> None:
    """Remove CMake cache and files to force a fresh configuration."""
    cache = build_path / "CMakeCache.txt"
    if cache.exists():
        cache.unlink()
    cmake_files = build_path / "CMakeFiles"
    if cmake_files.exists():
        shutil.rmtree(cmake_files, ignore_errors=True)

def _detect_generator() -> str:
    """Return a usable CMake generator string.
    Prefers Ninja if it works, otherwise tries Visual Studio generators,
    finally falls back to no explicit generator (CMake default).
    """
    # Test Ninja
    if shutil.which("ninja"):
        try:
            subprocess.run(["ninja", "--version"], capture_output=True, check=True, timeout=5)
            return "Ninja"
        except Exception:
            pass

    # Check for Visual Studio 2022
    for vs_path in [
        r"C:\Program Files\Microsoft Visual Studio\2022\Community",
        r"C:\Program Files\Microsoft Visual Studio\2022\Professional",
        r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise",
        r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools",
    ]:
        if Path(vs_path).exists():
            return "Visual Studio 17 2022"

    # Check for Visual Studio 2019
    for vs_path in [
        r"C:\Program Files\Microsoft Visual Studio\2019\Community",
        r"C:\Program Files\Microsoft Visual Studio\2019\Professional",
        r"C:\Program Files\Microsoft Visual Studio\2019\Enterprise",
        r"C:\Program Files\Microsoft Visual Studio\2019\BuildTools",
    ]:
        if Path(vs_path).exists():
            return "Visual Studio 16 2019"

    # No generator specified – let CMake pick default
    return ""


def _git_clone_or_update(git: Path, url: str, target: Path,
                         recursive: bool = False) -> bool:
    def _do_clone() -> bool:
        cmd = [str(git), "clone", "--depth", "1"]
        if recursive:
            cmd.extend(["--recurse-submodules", "--shallow-submodules"])
        cmd.extend([url, str(target)])
        log(f"  Cloning {url} ...")
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if r.returncode != 0:
                log(f"  Clone failed: {r.stderr[-400:]}")
                return False
            if not (target / "CMakeLists.txt").exists():
                log(f"  Clone appeared to succeed but CMakeLists.txt is missing.")
                return False
            return True
        except Exception as e:
            log(f"  Clone failed: {e}")
            return False

    # Detect a broken existing directory: present but missing CMakeLists.txt
    if target.exists() and not (target / "CMakeLists.txt").exists():
        log(f"  {target.name} directory incomplete — wiping and re-cloning ...")
        shutil.rmtree(target, ignore_errors=False)
        # Confirm it's gone; on Windows a file lock can prevent removal
        if target.exists():
            log(f"  ERROR: could not remove {target} — close any programs using it and retry.")
            return False

    if not target.exists():
        return _do_clone()

    # Directory exists and CMakeLists.txt is present — update in place
    log(f"  Updating {target.name} ...")
    subprocess.run([str(git), "pull", "--ff-only"], cwd=str(target),
                   check=False, capture_output=True, text=True, timeout=120)
    if recursive:
        subprocess.run(
            [str(git), "submodule", "update", "--init", "--recursive"],
            cwd=str(target), check=False, capture_output=True,
            text=True, timeout=300,
        )
    return True


# ---------------------------------------------------------------------------
# Build llama.cpp
# ---------------------------------------------------------------------------

def build_llamacpp(cpu: Dict[str, Any], has_vk: bool) -> str:
    git, cmake = find_git(), find_cmake()
    if not git:
        return "error: git not found"
    if not cmake:
        return "error: cmake not found"

    # llama.cpp requires submodules (cpp-httplib, pocs, tools, etc.)
    if not _git_clone_or_update(
            git,
            "https://github.com/ggerganov/llama.cpp.git",
            _LLAMA_DIR,
            recursive=True):
        return "error: clone/update failed"

    bdir = _LLAMA_DIR / "build"
    bdir.mkdir(exist_ok=True)
    _clean_build_dir(bdir)

    gen = _detect_generator()
    cmake_args = [
        str(cmake), "-B", str(bdir), "-S", str(_LLAMA_DIR),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DGGML_NATIVE=OFF",
        "-DLLAMA_BUILD_TESTS=OFF",
        "-DLLAMA_BUILD_EXAMPLES=ON",
        "-DLLAMA_BUILD_SERVER=OFF",
    ]
    if gen:
        cmake_args.extend(["-G", gen])
    if gen and "Visual Studio" in gen:
        cmake_args.extend(["-A", "x64"])

    # CPU optimisation flags
    for flag in cpu["cmake_flags"]:
        cmake_args.append(f"-D{flag}")

    cmake_args.append(f"-DGGML_VULKAN={'ON' if has_vk else 'OFF'}")

    log("  Configuring llama.cpp ...")
    r = subprocess.run(cmake_args, capture_output=True, text=True,
                       timeout=300)
    if r.returncode != 0:
        log(f"  cmake configure failed:\n{r.stderr[-800:]}")
        return "error: cmake configure"

    log("  Compiling llama.cpp (this takes several minutes) ...")
    ncpu = min(os.cpu_count() or 4, 16)
    build_cmd = [str(cmake), "--build", str(bdir), "--config", "Release"]
    if gen and "Ninja" in gen:
        build_cmd += ["--parallel", str(ncpu)]
    elif gen and "Visual Studio" in gen:
        build_cmd += ["--", f"/m:{ncpu}"]
    r = subprocess.run(build_cmd, capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        log(f"  build failed:\n{r.stderr[-600:]}")
        return "error: build failed"
    return "success"


# ---------------------------------------------------------------------------
# Build stable-diffusion.cpp
# ---------------------------------------------------------------------------

def build_sdcpp(cpu: Dict[str, Any], has_vk: bool) -> str:
    git, cmake = find_git(), find_cmake()
    if not git:
        return "error: git not found"
    if not cmake:
        return "error: cmake not found"

    # sd.cpp requires submodules (ggml, etc.)
    if not _git_clone_or_update(
            git,
            "https://github.com/leejet/stable-diffusion.cpp.git",
            _SD_DIR,
            recursive=True):
        return "error: clone/update failed"

    bdir = _SD_DIR / "build"
    bdir.mkdir(exist_ok=True)
    _clean_build_dir(bdir)

    gen = _detect_generator()

    def _make_cmake_args(use_vk: bool) -> List[str]:
        args = [
            str(cmake), "-B", str(bdir), "-S", str(_SD_DIR),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DGGML_NATIVE=OFF",
            "-DSD_BUILD_SHARED_LIBS=OFF",
        ]
        if gen:
            args.extend(["-G", gen])
        if gen and "Visual Studio" in gen:
            args.extend(["-A", "x64"])
        # CPU optimisation flags
        for flag in cpu["cmake_flags"]:
            args.append(f"-D{flag}")
        if use_vk:
            args.append("-DSD_VULKAN=ON")
        return args

    log("  Configuring sd.cpp ...")
    cmake_args = _make_cmake_args(has_vk)
    r = subprocess.run(cmake_args, capture_output=True, text=True, timeout=300)
    if r.returncode != 0 and has_vk:
        log("  Vulkan configure failed, retrying CPU-only ...")
        _clean_build_dir(bdir)
        cmake_args = _make_cmake_args(False)
        r = subprocess.run(cmake_args, capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        log(f"  cmake configure failed:\n{r.stderr[-800:]}")
        return "error: cmake configure"

    log("  Compiling sd.cpp (this takes several minutes) ...")
    ncpu = min(os.cpu_count() or 4, 16)
    build_cmd = [str(cmake), "--build", str(bdir), "--config", "Release"]
    if gen and "Ninja" in gen:
        build_cmd += ["--parallel", str(ncpu)]
    elif gen and "Visual Studio" in gen:
        build_cmd += ["--", f"/m:{ncpu}"]
    r = subprocess.run(build_cmd, capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        log(f"  build failed:\n{r.stderr[-600:]}")
        return "error: build failed"
    return "success"


# ---------------------------------------------------------------------------
# Install menu helpers
# ---------------------------------------------------------------------------

def _detect_build_tools() -> Tuple[Optional[Path], Optional[Path]]:
    return find_git(), find_cmake()


def _print_install_banner(cpu: Dict[str, Any], vk: Dict[str, Any]) -> None:
    git, cmake = _detect_build_tools()
    header("Image-Generator-Gguf — Install Method")
    print("  System Detections...")
    print(f"     Platform: Windows {platform.version().split('.')[0] if platform.system() == 'Windows' else platform.system()};"
          f" Python {platform.python_version()}")
    print(f"     Build Tools: Git {'OK' if git else 'NOT FOUND'};"
          f" CMake {'OK' if cmake else 'NOT FOUND'}")
    print(f"     Architecture: AVX {cpu['has_avx']};"
          f" AVX2 {cpu['has_avx2']};"
          f" F16C {cpu['has_f16c']};"
          f" FMA {cpu['has_fma']}")
    gpu_str = ", ".join(str(d["index"]) for d in vk["devices"]) if vk["devices"] else "none"
    print(f"     Hardware: CPUs {cpu['cores_logical']};"
          f" GPUs {gpu_str};"
          f" Vulkan {vk['version']}")
    print()
    print()
    print("  " + "-" * 79)
    print()
    print("     1. Clean Install (Purge First)")
    print()
    print("     2. Check/Install (Fix Missing Packages/Libraries)")
    print()
    print("     3. Refresh Configs (Only Remake Ini/Json)")
    print()
    print()
    print("  " + "=" * 79)


def _purge_for_clean_install() -> None:
    """Remove venv and build dirs so everything is rebuilt from scratch."""
    section("Purging previous installation...")
    for target, label in ((_VENV_DIR, "venv"), (_BUILD_DIR, "build")):
        if target.exists():
            log(f"Removing {label} at {target} ...")
            shutil.rmtree(target, ignore_errors=True)
            log(f"{label} removed.")
        else:
            log(f"{label} not present, skipping.")
    # Also remove persistent.json so defaults are regenerated
    if _PERSIST_PATH.exists():
        _PERSIST_PATH.unlink()
        log("persistent.json removed (will be regenerated).")


def _run_deps(cpu: Dict[str, Any]) -> None:
    """Create venv + install Python deps."""
    section("Python virtual environment...")
    if not create_venv():
        log("FATAL: could not create venv.")
        return

    section("Python dependencies...")
    if not install_deps():
        log("WARNING: some packages failed — the app may not work correctly.")
    else:
        log("All packages installed OK.")


def _run_build(cpu: Dict[str, Any], vk: Dict[str, Any]) -> None:
    """Build llama.cpp and sd.cpp."""
    section("Backend build  (llama.cpp + stable-diffusion.cpp)...")
    git  = find_git()
    cmake = find_cmake()
    log(f"git   : {git or 'NOT FOUND'}")
    log(f"cmake : {cmake or 'NOT FOUND'}")
    log()

    if not git or not cmake:
        log("SKIPPING BUILD — git and/or cmake not found.")
        log("Install them then re-run and choose option 1 or 2.")
        return

    llama_status = build_llamacpp(cpu, vk["available"])
    log(f"  llama.cpp  →  {llama_status}")
    sd_status = build_sdcpp(cpu, vk["available"])
    log(f"  sd.cpp     →  {sd_status}")


def _run_summary(t0: float) -> None:
    elapsed = round(time.time() - t0, 1)
    section("Installation summary")
    log(f"Time elapsed : {elapsed}s")
    log(f"constants.ini: {_CONST_PATH}")
    log(f"persistent   : {_PERSIST_PATH}")
    log(f"venv         : {_VENV_DIR}")
    log()
    log("Press Enter to return to the batch menu...")
    input()


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------

def run_detection() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    section("Hardware detection...")
    cpu = detect_cpu()
    vk  = detect_vulkan()

    log(f"CPU  : {cpu['brand']}")
    log(f"Arch : {cpu['arch']}  Vendor: {cpu['vendor']}")
    log(f"Cores: {cpu['cores_logical']} logical  →  {cpu['default_threads']} threads (85%)")
    log(f"AVX  : {cpu['has_avx']}  AVX2: {cpu['has_avx2']}  F16C: {cpu['has_f16c']}  FMA: {cpu['has_fma']}")
    log()
    log(f"Vulkan : {vk['available']}  ver={vk['version']}")
    log(f"SDK    : {vk['sdk'] or 'not set'}")
    if vk["devices"]:
        log("GPUs :")
        for d in vk["devices"]:
            log(f"  GPU{d['index']}: {d['name']}  ({d.get('type','')})")
    else:
        log("GPUs   : none detected via vulkaninfo")
    return cpu, vk


def main() -> None:
    parser = argparse.ArgumentParser(description="Image Generator GGUF Installer")
    parser.add_argument("--detect-only",  action="store_true",
                        help="Detect hardware and write constants.ini only")
    parser.add_argument("--deps-only",    action="store_true",
                        help="Create venv and install Python packages only")
    parser.add_argument("--build-only",   action="store_true",
                        help="Build llama.cpp and sd.cpp only")
    args = parser.parse_args()

    ensure_dirs()
    header("Image-Generator-Gguf — Initialize Install")
    if args.detect_only:
        cpu, vk = run_detection()
        write_constants(cpu, vk)
        write_default_persistent(cpu)
        log("Detection complete.")
        return

    if args.deps_only:
        cpu, vk = run_detection()
        write_constants(cpu, vk)
        write_default_persistent(cpu)
        t0 = time.time()
        _run_deps(cpu)
        _run_summary(t0)
        return

    if args.build_only:
        cpu, vk = run_detection()
        write_constants(cpu, vk)
        t0 = time.time()
        _run_build(cpu, vk)
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
            _purge_for_clean_install()
            write_constants(cpu, vk)
            write_default_persistent(cpu)
            _run_deps(cpu)
            _run_build(cpu, vk)
            _run_summary(t0)
            return

        if choice == "2":
            t0 = time.time()
            write_constants(cpu, vk)
            write_default_persistent(cpu)
            _run_deps(cpu)
            _run_build(cpu, vk)
            _run_summary(t0)
            return

        if choice == "3":
            t0 = time.time()
            section("Refreshing configs...")
            # Always overwrite constants; persistent only if absent
            write_constants(cpu, vk)
            if _PERSIST_PATH.exists():
                _PERSIST_PATH.unlink()
                log("persistent.json removed (will be regenerated with defaults).")
            write_default_persistent(cpu)
            _run_summary(t0)
            return

        # Invalid input — redisplay menu
        print()
        print("  Invalid selection, please try again.")
        print()


if __name__ == "__main__":
    main()