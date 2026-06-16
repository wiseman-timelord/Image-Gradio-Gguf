"""
utilities.py - General utility code: hardware detection, build system,
system information helpers. Code not more appropriate in other scripts.
"""

from __future__ import annotations

import ctypes
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

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

def detect_cpu_features() -> Dict[str, Any]:
    import math
    logical = os.cpu_count() or 4
    f: Dict[str, Any] = {
        "architecture": _cpu_arch(),
        "vendor": "unknown",
        "brand": platform.processor() or "unknown",
        "cores_logical": logical,
        "default_threads": max(1, math.ceil(logical * 0.85)),
        "has_avx": False, "has_avx2": False, "has_f16c": False,
        "has_fma": False, "has_avx512": False, "has_sse4_2": False,
        "has_aocl": False, "compile_flags": [],
    }
    if f["architecture"] in ("x86_64", "x86"):
        _detect_x86_features(f)

    for p in (os.environ.get("AOCL_ROOT", ""), os.environ.get("AOCL_PATH", ""),
              r"C:\Program Files\AMD\AOCL", r"C:\AOCL"):
        if p and Path(p).exists():
            f["has_aocl"] = True
            break

    flags = []
    for feat, flag in (("has_avx", "GGML_AVX=ON"), ("has_avx2", "GGML_AVX2=ON"),
                       ("has_f16c", "GGML_F16C=ON"), ("has_fma", "GGML_FMA=ON"),
                       ("has_avx512", "GGML_AVX512=ON")):
        if f[feat]:
            flags.append(flag)
    f["compile_flags"] = flags
    return f


def _cpu_arch() -> str:
    m = platform.machine().lower()
    if m in ("amd64", "x86_64"):
        return "x86_64"
    if m in ("i386", "i686", "x86"):
        return "x86"
    if "aarch64" in m:
        return "aarch64"
    if "arm" in m:
        return "arm"
    return m


def _detect_x86_features(f: Dict[str, Any]) -> None:
    cpu_name = f["brand"].lower()
    if platform.system() == "Windows":
        try:
            r = subprocess.run(["wmic", "cpu", "get", "Name", "/value"],
                               capture_output=True, text=True, timeout=10)
            if r.returncode == 0:
                for line in r.stdout.strip().splitlines():
                    if line.startswith("Name="):
                        f["brand"] = line.split("=", 1)[1].strip()
                        cpu_name = f["brand"].lower()
                        break
        except Exception:
            pass

    try:
        import cpuinfo  # type: ignore
        info = cpuinfo.get_cpu_info()
        fl = [x.lower() for x in info.get("flags", [])]
        f.update(has_avx="avx" in fl, has_avx2="avx2" in fl,
                 has_f16c="f16c" in fl, has_fma="fma" in fl,
                 has_sse4_2="sse4_2" in fl,
                 has_avx512=any("avx512" in x for x in fl))
        if info.get("brand_raw"):
            f["brand"] = info["brand_raw"]
        return
    except ImportError:
        pass

    _infer_from_cpu_name(cpu_name, f)


def _infer_from_cpu_name(name: str, f: Dict[str, Any]) -> None:
    n = name.lower()
    is_amd = any(k in n for k in ("amd", "ryzen", "epyc", "threadripper"))
    is_intel = any(k in n for k in ("intel", "core", "xeon"))

    if is_amd:
        f["vendor"] = "AMD"
        if any(k in n for k in ("ryzen", "epyc", "threadripper")):
            f.update(has_avx=True, has_avx2=True, has_f16c=True,
                     has_fma=True, has_sse4_2=True)
    elif is_intel:
        f["vendor"] = "Intel"
        f.update(has_avx=True, has_sse4_2=True)
        if any(k in n for k in ("ultra", "meteor", "arrow", "raptor", "alder",
                                "tiger", "ice", "comet", "coffee", "kaby",
                                "skylake", "broadwell", "haswell")):
            f.update(has_avx2=True, has_f16c=True, has_fma=True)
    elif f["architecture"] == "x86_64":
        f.update(has_avx=True, has_avx2=True, has_f16c=True,
                 has_fma=True, has_sse4_2=True)


# ---------------------------------------------------------------------------
# Vulkan detection
# ---------------------------------------------------------------------------

def detect_vulkan() -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "vulkan_available": False, "vulkan_version": "unknown",
        "devices": [], "vulkan_sdk": os.environ.get("VULKAN_SDK", ""),
        "loader_path": "", "error": "",
    }
    vi = shutil.which("vulkaninfo")
    if vi:
        try:
            proc = subprocess.run([vi, "--summary"], capture_output=True,
                                  text=True, timeout=30)
            if proc.returncode == 0:
                result["vulkan_available"] = True
                result["vulkan_version"] = _parse_vk_version(proc.stdout)
                result["devices"] = _parse_vk_devices(proc.stdout)
        except Exception as e:
            result["error"] = str(e)

    if not result["vulkan_available"] and platform.system() == "Windows":
        try:
            ctypes.windll.LoadLibrary("vulkan-1.dll")
            result["vulkan_available"] = True
            if result["vulkan_version"] == "unknown":
                result["vulkan_version"] = "1.x"
        except Exception as e:
            result["error"] = str(e)

    if not result["vulkan_available"] and platform.system() == "Windows":
        for p in (Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "vulkan-1.dll",
                  Path(os.environ.get("VULKAN_SDK", "")) / "Bin" / "vulkan-1.dll"):
            if p.exists():
                result["vulkan_available"] = True
                result["loader_path"] = str(p)
                if result["vulkan_version"] == "unknown":
                    result["vulkan_version"] = "1.x"
                break
    return result


def _parse_vk_version(stdout: str) -> str:
    for line in stdout.splitlines():
        for p in line.split():
            if p.startswith("1.") and len(p) >= 3:
                return p
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
            current = {"index": idx, "name": s.split("=", 1)[1].strip()}
        elif current:
            sl = s.lower().replace(" ", "")
            if sl.startswith("apiversion"):
                current["api_version"] = s.split("=", 1)[-1].strip()
            elif sl.startswith("devicetype"):
                current["type"] = s.split("=", 1)[-1].strip()
            elif "heap" in sl and "size" in sl:
                current["heap_size"] = s.split("=", 1)[-1].strip()
    if current:
        devices.append(current)
    return devices


def get_preferred_device() -> int:
    vk = detect_vulkan()
    if not vk.get("vulkan_available"):
        return -1
    cfg = configure.load_persistent()
    configured = cfg.get("vulkan_device", 1)
    if configured < len(vk.get("devices", [])):
        return configured
    return 0


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
# Build system
# ---------------------------------------------------------------------------

_BUILD_DIR = configure.get_build_dir()
_LLAMACPP_DIR = _BUILD_DIR / "llama.cpp"
_SD_DIR = _BUILD_DIR / "sd.cpp"

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


def _detect_generator() -> str:
    if shutil.which("ninja"):
        return "Ninja"
    for ver, gen in [("2022", "Visual Studio 17 2022"),
                     ("2019", "Visual Studio 16 2019")]:
        base = Path(rf"C:\Program Files\Microsoft Visual Studio\{ver}")
        if base.exists():
            for ed in ("Community", "Professional", "Enterprise"):
                if (base / ed).exists():
                    return gen
    return "Ninja"


def _git_clone_or_update(git: Path, url: str, target: Path, msg: Callable,
                         recursive: bool = False) -> bool:
    if not target.exists():
        cmd = [str(git), "clone", "--depth", "1"]
        if recursive:
            cmd.append("--recursive")
        cmd.extend([url, str(target)])
        try:
            subprocess.run(cmd, check=True, capture_output=True,
                           text=True, timeout=300)
            return True
        except Exception as e:
            msg(f"Clone failed: {e}")
            return False
    msg(f"Updating {target.name}...")
    subprocess.run([str(git), "pull", "--ff-only"], cwd=str(target),
                   check=False, capture_output=True, text=True, timeout=60)
    if recursive:
        subprocess.run([str(git), "submodule", "update", "--init", "--recursive"],
                       cwd=str(target), check=False, capture_output=True,
                       text=True, timeout=60)
    return True


def build_all(progress_callback: Optional[Callable[[str, float], None]] = None
              ) -> Dict[str, Any]:
    t0 = time.time()
    status: Dict[str, Any] = {
        "success": False, "llama_status": "pending", "sd_status": "pending",
        "messages": [], "elapsed_seconds": 0.0,
    }

    def _msg(text: str, progress: float = 0.0):
        status["messages"].append(text)
        if progress_callback:
            try:
                progress_callback(text, progress)
            except Exception:
                pass

    prereq = check_prerequisites()
    if not prereq["git"] or not prereq["cmake"]:
        missing = [k for k in ("git", "cmake") if not prereq[k]]
        _msg(f"ERROR: {', '.join(missing)} not found.")
        status["llama_status"] = status["sd_status"] = "skipped"
        return status

    _BUILD_DIR.mkdir(parents=True, exist_ok=True)
    cpu = detect_cpu_features()
    has_vk = detect_vulkan()["vulkan_available"]

    _msg("--- Building llama.cpp ---", 0.05)
    status["llama_status"] = _build_llamacpp(has_vk, cpu, _msg)
    _msg(f"llama.cpp: {status['llama_status']}", 0.45)

    _msg("--- Building stable-diffusion.cpp ---", 0.50)
    status["sd_status"] = _build_sdcpp(has_vk, cpu, _msg)
    _msg(f"sd.cpp: {status['sd_status']}", 0.95)

    status["success"] = (status["llama_status"] == "success"
                         and status["sd_status"] == "success")
    status["elapsed_seconds"] = round(time.time() - t0, 1)
    _msg(f"Build completed in {status['elapsed_seconds']}s.", 1.0)
    return status


def _build_llamacpp(has_vk: bool, cpu: Dict, _msg: Callable) -> str:
    git, cmake = find_git(), find_cmake()
    if not git or not cmake:
        return "error: missing tools"
    try:
        if not _git_clone_or_update(
                git, "https://github.com/ggerganov/llama.cpp.git",
                _LLAMACPP_DIR, _msg):
            return "error: clone failed"

        bdir = _LLAMACPP_DIR / "build"
        bdir.mkdir(exist_ok=True)
        gen = _detect_generator()
        args = [str(cmake), "-B", str(bdir), "-S", str(_LLAMACPP_DIR),
                "-G", gen]
        if "Visual Studio" in gen:
            args.extend(["-A", "x64"])
        args.extend([
            "-DCMAKE_BUILD_TYPE=Release", "-DGGML_NATIVE=OFF",
            "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_EXAMPLES=ON",
            "-DLLAMA_BUILD_SERVER=OFF",
            "-DGGML_AVX=ON", "-DGGML_AVX2=ON",
            "-DGGML_F16C=ON", "-DGGML_FMA=ON",
            f"-DGGML_VULKAN={'ON' if has_vk else 'OFF'}",
        ])
        if cpu.get("has_aocl"):
            aocl = os.environ.get("AOCL_ROOT") or os.environ.get("AOCL_PATH")
            if aocl:
                args.extend(["-DBLAS_VENDOR=AOCL",
                             f"-DCMAKE_PREFIX_PATH={aocl}"])

        _msg("Configuring llama.cpp...", 0.15)
        r = subprocess.run(args, capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            _msg(f"CMake failed: {r.stderr[-500:]}")
            return "error: cmake configure"

        _msg("Compiling llama.cpp...", 0.25)
        bcmd = [str(cmake), "--build", str(bdir), "--config", "Release"]
        if "Ninja" in gen:
            bcmd.extend(["--parallel",
                         str(min(os.cpu_count() or 4, 16))])
        r = subprocess.run(bcmd, capture_output=True, text=True, timeout=900)
        return "success" if r.returncode == 0 else "error: build failed"
    except Exception as e:
        return f"error: {e}"


def _build_sdcpp(has_vk: bool, cpu: Dict, _msg: Callable) -> str:
    git, cmake = find_git(), find_cmake()
    if not git or not cmake:
        return "error: missing tools"
    try:
        if not _git_clone_or_update(
                git, "https://github.com/leejet/stable-diffusion.cpp.git",
                _SD_DIR, _msg, recursive=True):
            return "error: clone failed"

        bdir = _SD_DIR / "build"
        bdir.mkdir(exist_ok=True)
        gen = _detect_generator()
        args = [str(cmake), "-B", str(bdir), "-S", str(_SD_DIR), "-G", gen]
        if "Visual Studio" in gen:
            args.extend(["-A", "x64"])
        args.append("-DCMAKE_BUILD_TYPE=Release")
        if has_vk:
            _msg("Enabling Vulkan for sd.cpp", 0.63)
            args.append("-DSD_VULKAN=ON")

        _msg("Configuring sd.cpp...", 0.60)
        r = subprocess.run(args, capture_output=True, text=True, timeout=120)
        if r.returncode != 0 and has_vk and "SD_VULKAN" in r.stderr:
            _msg("Retrying without SD_VULKAN...", 0.62)
            args = [a for a in args if "SD_VULKAN" not in a]
            r = subprocess.run(args, capture_output=True, text=True,
                               timeout=120)
        if r.returncode != 0:
            _msg(f"CMake failed: {r.stderr[-500:]}")
            return "error: cmake configure"

        _msg("Compiling sd.cpp...", 0.70)
        bcmd = [str(cmake), "--build", str(bdir), "--config", "Release"]
        if "Ninja" in gen:
            bcmd.extend(["--parallel",
                         str(min(os.cpu_count() or 4, 16))])
        r = subprocess.run(bcmd, capture_output=True, text=True, timeout=900)
        return "success" if r.returncode == 0 else "error: build failed"
    except Exception as e:
        return f"error: {e}"


def get_build_status() -> Dict[str, Any]:
    llama_exe = _find_exe("llama-cli", _LLAMACPP_DIR) or _find_exe("main", _LLAMACPP_DIR)
    sd_exe = _find_exe("sd", _SD_DIR)
    return {
        "llama_built": llama_exe is not None,
        "llama_path": str(llama_exe) if llama_exe else "",
        "sd_built": sd_exe is not None,
        "sd_path": str(sd_exe) if sd_exe else "",
        "llama_source_exists": _LLAMACPP_DIR.exists(),
        "sd_source_exists": _SD_DIR.exists(),
    }


def _find_exe(name: str, base: Path) -> Optional[Path]:
    for sub in ("build/bin/Release", "build/bin", "build/Release", "build"):
        p = base / sub / f"{name}.exe"
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def get_relevant_env() -> Dict[str, str]:
    keys = ["VULKAN_SDK", "VK_INSTANCE_LAYERS", "VK_LAYER_PATH",
            "CUDA_PATH", "HIP_PATH", "GGML_VULKAN_DEVICE",
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