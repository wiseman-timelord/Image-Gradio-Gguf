# ============================================================================
# INSTALLER.PY - CRITICAL FAILURE MODE
# Agentic Chatbot Installation Script
# Windows 10, Python 3.12
# ============================================================================
import os
import subprocess
import sys
import time
import shutil
import configparser
import threading
import argparse
import json
import copy
import datetime
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

if sys.platform == "win32":
    # Force UTF-8 output in the Windows console
    sys.stdout.reconfigure(encoding='utf-8')

# ============================================================================
# EMBEDDED REQUIREMENTS - ALL CRITICAL (NO OPTIONAL)
# ============================================================================
REQUIREMENTS_INSTALL = [
    "requests==2.31.0",
    "psutil==7.2.1",
    "py-cpuinfo==9.0.0",
]

REQUIREMENTS_PROGRAM = {
    "core": [
        "numpy==1.26.4",
        "requests==2.31.0",
        "psutil==7.2.1",
        "py-cpuinfo==9.0.0",
        "beautifulsoup4==4.12.3",
        "lxml==5.3.0",
        "googlesearch-python==1.3.0",
        "gradio==5.50.0",
        "PyQt6==6.7.0",
        "PyQt6-WebEngine==6.7.0",
        "pillow==10.4.0",
        "pillow-avif-plugin==1.4.0",
        "pandas==2.2.2",
        "sentence-transformers==3.0.1",
        "tiktoken==0.7.0",
        "openai>=1.30.0",       # Manager HTTP client → WSL2 vLLM (OpenAI-compatible API)
    ],
    "vulkan": ["vulkan==1.3.275.0"],
    "rag": [],
    "tts": [],
}

# NOTE: pyamdgpuinfo REMOVED - Windows incompatible, use WMI detection in program instead
PACKAGE_SIZES = {
    "numpy": 50,
    "requests": 5,
    "psutil": 2,
    "py-cpuinfo": 1,
    "beautifulsoup4": 2,
    "lxml": 15,
    "googlesearch-python": 1,
    "gradio": 50,
    "PyQt6": 100,
    "PyQt6-WebEngine": 150,
    "pillow": 15,
    "pillow-avif-plugin": 2,
    "pandas": 30,
    "vulkan": 1,
    "sentence-transformers": 150,
    "transformers": 200,
    "tiktoken": 5,
    "torchaudio": 100,
    "llama-cpp-python": 50,
}

LLAMACPP_VERSION = "b9585"   # source tag cloned + built by build_llama_agents()

# llama_agents — Windows Vulkan build used by playground.py (agents + TTS)
# Lives under data/ alongside other non-Python binaries
LLAMA_AGENTS_DIR = "data/llama_agents"

# sd_agent — stable-diffusion.cpp Windows Vulkan build, used by
# playground.py's run_stable_diffusion() (image_gen_agent, one-shot CLI —
# see configure.py's SD_AGENTS_DIR docs for the exact sd-cli.exe flags used)
SD_AGENTS_DIR = "data/sd_agent"
SD_CPP_SOURCE_URL = "https://github.com/leejet/stable-diffusion.cpp.git"

# WSL2 manager server
MANAGER_PORT = 8080
MANAGER_SCRIPT = "helper.sh"   # permanent file in BASE_DIR root (not generated)

RETRY_DELAY = 5
MAX_RETRIES = 5

def get_flattened_requirements() -> List[str]:
    return [pkg for category in REQUIREMENTS_PROGRAM.values() for pkg in category]

# ============================================================================
# PROGRESS BAR - SINGLE LINE THAT OVERWRITES ITSELF
# ============================================================================
def format_size(bytes_val: float) -> str:
    if bytes_val >= 1024 * 1024:
        return f"{bytes_val / (1024 * 1024):.1f} MB"
    elif bytes_val >= 1024:
        return f"{bytes_val / 1024:.1f} KB"
    else:
        return f"{bytes_val:.0f} B"

def format_time(seconds: float) -> str:
    if seconds >= 60:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    else:
        return f"{int(seconds)}s"

def format_speed(bytes_per_sec: float) -> str:
    if bytes_per_sec >= 1024 * 1024:
        return f"{bytes_per_sec / (1024 * 1024):.1f} MB/s"
    elif bytes_per_sec >= 1024:
        return f"{bytes_per_sec / 1024:.0f} KB/s"
    else:
        return f"{bytes_per_sec:.0f} B/s"

def show_progress(pkg_name: str, percent: float, current: float, total: float,
                  elapsed: float, speed: float) -> None:
    bar_filled = int(20 * min(percent, 100) / 100)
    bar = "=" * bar_filled + "-" * (20 - bar_filled)
    name = pkg_name[:16].ljust(16)
    time_str = format_time(elapsed)
    speed_str = format_speed(speed)
    line = f"\r  Installing: {name} [{bar}] {min(percent, 100):3.0f}% | {time_str}"
    sys.stdout.write(line.ljust(79)[:79])
    sys.stdout.flush()

def show_done(pkg_name: str, total: float, elapsed: float) -> None:
    name = pkg_name[:16].ljust(16)
    bar = "=" * 20
    line = f"\r  Installing: {name} [{bar}] 100% | {format_time(elapsed)}"
    sys.stdout.write(line.ljust(79)[:79] + "\n")
    sys.stdout.flush()

def show_skipped(pkg_name: str) -> None:
    name = pkg_name[:16].ljust(16)
    line = f"\r  ○ {name} SKIPPED (already installed)"
    sys.stdout.write(line.ljust(79)[:79] + "\n")
    sys.stdout.flush()

def show_failed(pkg_name: str, elapsed: float) -> None:
    name = pkg_name[:16].ljust(16)
    line = f"\r  ✗ {name} FAILED after {format_time(elapsed)}"
    sys.stdout.write(line.ljust(79)[:79] + "\n")
    sys.stdout.flush()

# ============================================================================
# BOOTSTRAP PHASE
# ============================================================================
BASE_DIR = Path(__file__).parent

def bootstrap_install() -> bool:
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
                   capture_output=True, check=False)
    for package in REQUIREMENTS_INSTALL:
        print(f"  Installing {package}...")
        result = subprocess.run([sys.executable, "-m", "pip", "install", package],
                                capture_output=True, text=True, check=False)
        if result.returncode != 0:
            print(f"  Failed: {result.stderr}")
            return False
    print("Bootstrap complete!")
    return True

import requests
import psutil
import cpuinfo

APP_NAME = "Agentic Chatbot"
VENV_DIR = BASE_DIR / ".venv"
TEMP_DIR = BASE_DIR / "data" / "temp"
SELECTED_GRADIO = "5.50.0"
PYTHON_VERSION = "3.12"

DIRECTORIES = [
    "data", "data/history", "data/temp", "data/vectors", "data/embedding_cache",
    "data/visualizes", "data/cache", "data/cache/webengine", "data/llama_agents",
    "models", "scripts", "anims"
]

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def clear_screen():
    os.system('cls')

def display_separator_thick():
    print("=" * 79)

def display_separator_thin():
    print("-" * 79)

def display_header(title):
    clear_screen()
    display_separator_thick()
    print(f"    {title}")
    display_separator_thick()
    print()


def display_main_header():
    display_header("Agentic Chatbot: Installer")

def display_critical_failure(title: str, error_details: str):
    """Display critical failure and wait for user input before exit."""
    print("\n\n *** CRITICAL INSTALLATION FAILURE ***\n")
    print(f"\n  Error Details:\n  {error_details}\n")
    display_separator_thick()
    print("\n  Critical error during installation, take a moment to assess the issue,")
    print("  then press enter to return to the menu...")
    input()
    sys.exit(1)

def print_status(msg, success=True):
    print(f"  {'[OK]' if success else '[FAIL]'} {msg}")
    time.sleep(0.2 if success else 0.5)

def print_info(msg):
    print(f"  {msg}")

def detect_cpu_features() -> Dict[str, Any]:
    try:
        info = cpuinfo.get_cpu_info()
        flags = info.get('flags', [])
        features = {
            'avx': 'avx' in flags,
            'avx2': 'avx2' in flags,
            'avx512': any('avx512' in f for f in flags),
            'f16c': 'f16c' in flags,
            'fma': 'fma' in flags,
            'cpu_name': info.get('brand_raw', 'Unknown'),
            'cores_physical': psutil.cpu_count(logical=False),
            'cores_logical': psutil.cpu_count(logical=True),
            'arch': info.get('arch', 'X86_64')
        }
        print(f"\n  CPU: {features['cpu_name']}")
        print(f"  Cores: {features['cores_physical']} physical / {features['cores_logical']} logical")
        return features
    except:
        return {
            'avx': True, 'avx2': True, 'avx512': False, 'f16c': True, 'fma': True,
            'cpu_name': 'Unknown', 'cores_logical': 24, 'cores_physical': 12, 'arch': 'X86_64'
        }

def check_vulkan() -> Tuple[bool, str]:
    try:
        import vulkan
        return True, "1.3"
    except:
        try:
            result = subprocess.run(['vulkaninfo'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                return True, "Unknown"
        except:
            pass
    return False, "Not Available"

def detect_msvc() -> str:
    """Detect installed MSVC version."""
    try:
        vswhere_path = Path(os.environ.get('ProgramFiles(x86)', r'C:\Program Files (x86)')) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
        if vswhere_path.exists():
            result = subprocess.run([str(vswhere_path), '-latest', '-property', 'catalog_productDisplayVersion'],
                                    capture_output=True, text=True, timeout=5)
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        try:
            import winreg
            key_path = r"SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\X64"
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as key:
                version, _ = winreg.QueryValueEx(key, "Version")
                if version:
                    return f"MSVC++ {version[:4]}"
        except:
            pass
    except:
        pass
    return "Not Detected"

def create_directories():
    for d in DIRECTORIES:
        (BASE_DIR / d).mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    print_status("Directory structure created")

def create_init_files() -> bool:
    try:
        scripts_dir = BASE_DIR / "scripts"
        scripts_dir.mkdir(parents=True, exist_ok=True)
        with open(scripts_dir / "__init__.py", 'w') as f:
            f.write('"""Agentic Chatbot Python package."""\n')
        print_status("Created Python package files")
        return True
    except Exception as e:
        print_status(f"Failed: {e}", False)
        return False

def create_venv() -> bool:
    try:
        if VENV_DIR.exists() and (VENV_DIR / "Scripts" / "python.exe").exists():
            print_status("Virtual environment already exists (skipped)")
            return True
        result = subprocess.run([sys.executable, "-m", "venv", str(VENV_DIR)],
                                capture_output=True, text=True)
        if result.returncode != 0:
            print_status(f"Failed: {result.stderr}", False)
            return False
        print_status("Created virtual environment")
        return True
    except Exception as e:
        print_status(f"Failed: {e}", False)
        return False

# ============================================================================
# RETRY / BACKOFF
# ============================================================================
# Escalating backoff delays for source clone/fetch retries (seconds)
RETRY_DELAYS = [5, 15, 30, 60, 90]

def parse_package_name(package_spec: str) -> str:
    """Extract display name from package spec (handles URLs)."""
    if package_spec.startswith(('http://', 'https://')):
        filename = package_spec.split('/')[-1]
        if filename.endswith('.whl'):
            return filename.split('-')[0]
        return filename[:16]
    for sep in ['<', '>', '=', '[', ';', '!']:
        if sep in package_spec:
            return package_spec.split(sep)[0].strip()
    return package_spec.strip()

def install_package_with_progress(pip_exe: str, package: str, max_retries: int = MAX_RETRIES) -> Tuple[bool, str]:
    """Install a single package with retries. Returns (success, error_message)."""
    pkg_name = parse_package_name(package)
    estimated_size = PACKAGE_SIZES.get(pkg_name, 30) * 1024 * 1024
    attempt = 1
    last_error = ""
    
    while attempt <= max_retries:
        start_time = time.time()
        try:
            cmd = [
                pip_exe, "install",
                "--retries", "10",
                "--timeout", "300",
                "--no-input",
                "--progress-bar", "off",
                package
            ]
            
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            stdout_data = []
            stderr_data = []
            
            def read_stdout():
                for line in iter(process.stdout.readline, b''):
                    stdout_data.append(line)
                process.stdout.close()  
            
            def read_stderr():
                for line in iter(process.stderr.readline, b''):
                    stderr_data.append(line)
                process.stderr.close()
             
            stdout_thread = threading.Thread(target=read_stdout, daemon=True)
            stderr_thread = threading.Thread(target=read_stderr, daemon=True)
            stdout_thread.start() 
            stderr_thread.start()
             
            last_update = start_time
            
            while process.poll() is None:
                now = time.time()
                if now - last_update >= 0.15:
                    last_update = now
                    elapsed = now - start_time
                    fake_percent = min(95, 5 + (elapsed / 300) * 90)
                    fake_current = estimated_size * fake_percent / 100
                    fake_speed = fake_current / elapsed if elapsed > 0 else 0
                    show_progress(pkg_name, fake_percent, fake_current, estimated_size, elapsed, fake_speed)
                time.sleep(0.05)
            
            stdout_thread.join(timeout=1)
            stderr_thread.join(timeout=1)
            
            elapsed = time.time() - start_time
            
            if process.returncode == 0:
                show_done(pkg_name, estimated_size, elapsed)
                return True, ""
            else:
                stderr = b''.join(stderr_data).decode('utf-8', errors='replace')
                
                if "already satisfied" in stderr.lower() or "requirement already" in stderr.lower():
                    show_skipped(pkg_name)
                    return True, ""
                
                last_error = stderr[:500]
                show_failed(pkg_name, elapsed)
                print(f"  [RETRY {attempt}/{max_retries}] {pkg_name}")
                time.sleep(RETRY_DELAY)
                attempt += 1
                continue

        except KeyboardInterrupt:
            elapsed = time.time() - start_time
            show_failed(pkg_name, elapsed)
            return False, "Cancelled by user"
        
        except Exception as e:
            last_error = str(e)
            elapsed = time.time() - start_time
            show_failed(pkg_name, elapsed)
            print(f"  [RETRY {attempt}/{max_retries}] {pkg_name}: {str(e)[:50]}")
            time.sleep(RETRY_DELAY)
            attempt += 1
            continue

    # All retries exhausted
    return False, last_error

def install_base_deps() -> Tuple[bool, str]:
    """Install all base dependencies. CRITICAL: Fail immediately if any package fails."""
    pip_exe = str(VENV_DIR / "Scripts" / "pip.exe")
    print_info("Upgrading pip...")
    subprocess.run([pip_exe, "install", "--upgrade", "pip"], capture_output=True, check=False)
    subprocess.run([pip_exe, "config", "set", "global.timeout", "300"], capture_output=True, check=False)
    subprocess.run([pip_exe, "config", "set", "global.retries", "10"], capture_output=True, check=False)
    
    all_packages = get_flattened_requirements()
    for package in all_packages:
        pkg_name = parse_package_name(package)
        success, error = install_package_with_progress(pip_exe, package)
        
        if not success:
            error_msg = f"CRITICAL: Failed to install {pkg_name} after {MAX_RETRIES} retries.\n\nError: {error[:500]}"
            return False, error_msg

    print_status("Base dependencies installed")
    return True, ""

def check_package_installed(pip_exe: str, package_spec: str) -> bool:
    """Return True if a package is already installed in the venv."""
    pkg_name = parse_package_name(package_spec)
    result = subprocess.run(
        [pip_exe, "show", pkg_name],
        capture_output=True, text=True, check=False
    )
    return result.returncode == 0

def refresh_packages_only() -> bool:
    """Option 3 – check every requirement and install only what is missing."""
    print("\nSTEP 1: Ensuring Configuration Files Exist")
    display_separator_thin()
    cpu = detect_cpu_features()
    vulkan, _ = check_vulkan()

    if not (BASE_DIR / "data" / "constants.ini").exists():
        if not create_system_ini(vulkan, cpu):
            display_critical_failure("Failed to create constants.ini", "Configuration file creation failed")
            return False
    if not (BASE_DIR / "data" / "configuration.json").exists():
        if not create_configuration_json(vulkan, cpu):
            display_critical_failure("Failed to create configuration.json", "Settings file creation failed")
            return False
    if not (BASE_DIR / "data" / "profiles.json").exists():
        if not create_profiles_json():
            display_critical_failure("Failed to create profiles.json", "Profiles file creation failed")
            return False

    print("\nSTEP 2: Ensuring Virtual Environment Exists")
    display_separator_thin()
    if not create_venv():
        display_critical_failure("Failed to create virtual environment", "venv creation failed")
        return False
    pip_exe = str(VENV_DIR / "Scripts" / "pip.exe")
    print_info("Upgrading pip...")
    subprocess.run([pip_exe, "install", "--upgrade", "pip"], capture_output=True, check=False)

    print("\nSTEP 3: Checking / Installing Missing Packages")
    display_separator_thin()
    all_packages = get_flattened_requirements()
    missing = []
    for package in all_packages:
        if not check_package_installed(pip_exe, package):
            missing.append(package)
        else:
            show_skipped(parse_package_name(package))
    if missing:
        print_info(f"\n  {len(missing)} missing package(s) to install:")
        for package in missing:
            pkg_name = parse_package_name(package)
            success, error = install_package_with_progress(pip_exe, package)
            if not success:
                display_critical_failure(
                    f"Failed to install {pkg_name}",
                    f"Package installation failed after {MAX_RETRIES} retries.\n\nError: {error[:500]}"
                )
                return False
    else:
        print_status("All packages already present - nothing to install")

    print("\nSTEP 4: Ensuring Package Files")
    display_separator_thin()
    create_init_files()

    print("\nSTEP 5: Checking WSL2 Manager Server")
    display_separator_thin()
    wsl_ok, wsl_msg = check_wsl2()
    if wsl_ok:
        _cflags = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
        verify_cmd = "~/vllm-env/bin/python -c \"import vllm\" 2>/dev/null && echo OK"
        try:
            verify = subprocess.run(
                ["wsl", "-e", "bash", "-c", verify_cmd],
                capture_output=True, text=True, timeout=10, creationflags=_cflags,
            )
            if "OK" not in verify.stdout:
                print_info("  WSL2 vLLM missing - setting up...")
                wsl_pw = prompt_wsl_sudo_password()
                if not setup_wsl2_vllm(wsl_pw):
                    display_critical_failure("Failed to set up WSL2 vLLM", "Setup process failed or timed out.")
                    return False
            else:
                show_skipped("WSL2 vLLM")
        except Exception as e:
            print_info(f"  [!] Could not verify WSL2 vLLM: {e}")
            
        models_dir = BASE_DIR / "models"
        manager_models_dir = _resolve_manager_models_dir(models_dir)
        print_info("  Verifying helper.sh and constants.ini...")
        if manager_models_dir != models_dir:
            print_info(f"  Manager model directory: {manager_models_dir} (separate from agents' models_dir)")
        generate_start_manager_sh(models_dir, cpu, manager_models_dir)
        
        wslconfig_path = Path.home() / ".wslconfig"
        try:
            has_wslconfig = wslconfig_path.exists() and "Agentic Chatbot" in wslconfig_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            has_wslconfig = False
            
        if not has_wslconfig:
            print_info("  .wslconfig missing or outdated - creating...")
            wsl_ram_gb = prompt_wsl_profile_size()
            create_wslconfig(wsl_ram_gb)
        else:
            show_skipped(".wslconfig")
            
        print_info("  Verifying port proxy...")
        setup_portproxy()
    else:
        print_info(f"  [!] WSL2 not available: {wsl_msg}")

    print("\nSTEP 6: Checking llama-cpp-python")
    display_separator_thin()
    if not check_package_installed(pip_exe, "llama-cpp-python"):
        print_info("  llama-cpp-python missing - running compile installer...")
        print_info("  - Requires Visual Studio Build Tools + Vulkan SDK")
        success, error = install_llama_compile()
        if not success:
            display_critical_failure("Failed to compile llama-cpp-python", error)
            return False
    else:
        show_skipped("llama-cpp-python")

    print("\nSTEP 7: Checking Native Binaries for Agents and Image Gen")
    display_separator_thin()
    
    server_exe = BASE_DIR / LLAMA_AGENTS_DIR / "llama-server.exe"
    if not server_exe.exists():
        print_info("  llama-server.exe missing - building...")
        success, error = build_llama_agents(cpu)
        if not success:
            display_critical_failure("Failed to build llama-server.exe", error)
            return False
    else:
        show_skipped("llama-server.exe")
        
    sd_found = any((BASE_DIR / SD_AGENTS_DIR / name).exists() for name in ("sd-cli.exe", "sd.exe"))
    if not sd_found:
        print_info("  sd-cli.exe missing - building...")
        success, error = build_sd_agent(cpu)
        if not success:
            display_critical_failure("Failed to build sd-cli.exe", error)
            return False
    else:
        show_skipped("sd-cli.exe")

    print("\n")
    display_separator_thick()
    print("    REFRESH COMPLETE!")
    display_separator_thick()
    return True

def recreate_jsons_only() -> bool:
    """Option 4 – recreate configuration.json AND profiles.json with defaults
    (keeps venv & packages).

    Both JSON files hold runtime settings/identity only — nothing in either
    one is install-specific, so they can be regenerated without touching the
    venv, packages, or constants.ini."""
    print("\n  Re-creating configuration.json and profiles.json with defaults...")
    if not (BASE_DIR / "data").exists():
        (BASE_DIR / "data").mkdir(parents=True, exist_ok=True)
        print_status("Created data/ directory")

    cpu = detect_cpu_features()
    vulkan, _ = check_vulkan()

    ok_config = create_configuration_json(vulkan, cpu)
    ok_profiles = create_profiles_json()

    if ok_config and ok_profiles:
        print("\n")
        display_separator_thick()
        print("    JSONS RE-CREATED!")
        display_separator_thick()
        return True
    return False

def detect_vulkan_sdk() -> Optional[str]:
    """Detect Vulkan SDK installation path on Windows."""
    sdk_path = os.environ.get('VULKAN_SDK', '')
    if sdk_path and Path(sdk_path).exists():
        return sdk_path
        
    common_paths = [
        Path(os.environ.get('ProgramFiles', r'C:\Program Files')) / "VulkanSDK",
        Path(r"C:\VulkanSDK"),
    ]
    for base in common_paths:
        if base.exists():
            versions = sorted([d for d in base.iterdir() if d.is_dir()], reverse=True)
            if versions:
                return str(versions[0])
    return None

def detect_msvc_env() -> Optional[str]:
    """Detect Visual Studio Build Tools and return the path to vcvarsall.bat."""
    vswhere_path = Path(os.environ.get('ProgramFiles(x86)', r'C:\Program Files (x86)')) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if vswhere_path.exists():
        try:
            result = subprocess.run(
                [str(vswhere_path), '-latest', '-products', '*',
                 '-requires', 'Microsoft.VisualStudio.Component.VC.Tools.x86.x64',
                 '-property', 'installationPath'],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0 and result.stdout.strip():
                vs_path = Path(result.stdout.strip())
                vcvarsall = vs_path / "VC" / "Auxiliary" / "Build" / "vcvarsall.bat"
                if vcvarsall.exists():
                    return str(vcvarsall)
        except Exception:
            pass
            
    for year in ["2022", "2019", "2017"]:
        for edition in ["BuildTools", "Community", "Professional", "Enterprise"]:
            vcvarsall = Path(rf"C:\Program Files (x86)\Microsoft Visual Studio\{year}\{edition}\VC\Auxiliary\Build\vcvarsall.bat")
            if vcvarsall.exists():
                return str(vcvarsall)
    return None

def build_cmake_args(cpu_features: Dict) -> str:
    """Build CMAKE_ARGS string for llama-cpp-python compilation."""
    args = ["-DGGML_VULKAN=ON"]
    if cpu_features.get('avx512'):
        args.append("-DGGML_AVX512=ON")
        print_info("  CPU: AVX-512 detected — enabled")
    if cpu_features.get('avx2'):
        args.append("-DGGML_AVX2=ON")
        print_info("  CPU: AVX2 detected — enabled")
    if cpu_features.get('avx'):
        args.append("-DGGML_AVX=ON")
        print_info("  CPU: AVX detected — enabled")
    if cpu_features.get('f16c'):
        args.append("-DGGML_F16C=ON")
        print_info("  CPU: F16C detected — enabled")
    if cpu_features.get('fma'):
        args.append("-DGGML_FMA=ON")
        print_info("  CPU: FMA detected — enabled")
        
    cores = cpu_features.get('cores_logical', os.cpu_count() or 4)
    args.append(f"-DCMAKE_BUILD_PARALLEL_LEVEL={cores}")
    
    cmake_args = " ".join(args)
    print_info(f"  CMAKE_ARGS: {cmake_args}")
    return cmake_args

def install_llama_compile() -> Tuple[bool, str]:
    """Compile llama-cpp-python with CPU-specific optimizations (manual option). CRITICAL MODE."""
    python_exe = str(VENV_DIR / "Scripts" / "python.exe")
    pip_exe = str(VENV_DIR / "Scripts" / "pip.exe")
    cpu_features = detect_cpu_features()
    cmake_args = build_cmake_args(cpu_features)
    print_status(f"Compiling with Vulkan + CPU optimizations")
    print_info(f"  CMAKE_ARGS: {cmake_args}")
    
    vulkan_available, _ = check_vulkan()
    if not vulkan_available:
        print_status("Vulkan not available on system", False)
        error_details = "This installation requires Vulkan. Please install Vulkan drivers.\nhttps://vulkan.lunarg.com/sdk/home"
        return False, error_details
        
    build_env = os.environ.copy()
    build_env["CMAKE_ARGS"] = cmake_args
    build_env["FORCE_CMAKE"] = "1"
    cores = cpu_features.get('cores_logical', os.cpu_count() or 4)
    build_env["CMAKE_BUILD_PARALLEL_LEVEL"] = str(cores)
    build_env["MAKEFLAGS"] = f"-j{cores}"

    # Inject Vulkan SDK paths so CMake can locate headers and libraries
    vulkan_sdk = detect_vulkan_sdk()
    if vulkan_sdk:
        build_env["VULKAN_SDK"] = vulkan_sdk
        vulkan_bin = str(Path(vulkan_sdk) / "Bin")
        if vulkan_bin not in build_env.get("PATH", ""):
            build_env["PATH"] = vulkan_bin + ";" + build_env.get("PATH", "")
        print_status(f"Vulkan SDK: {vulkan_sdk}")
    else:
        print_info("  WARNING: Vulkan SDK not detected — compilation may fail")
        print_info("  Install from: https://vulkan.lunarg.com/sdk/home")

    vcvarsall = detect_msvc_env()
    if vcvarsall:
        try:
            cmd_activate = f'"{vcvarsall}" x64 && set'
            result = subprocess.run(cmd_activate, shell=True, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    if '=' in line:
                        key, _, value = line.partition('=')
                        build_env[key] = value
                print_status("MSVC environment activated (x64)")
        except Exception as e:
            print_info(f"  Warning: Failed to activate MSVC env: {e}")

    subprocess.run([pip_exe, "install", 
                  "cmake==3.30.2", 
                  "scikit-build-core==0.10.7", 
                  "setuptools==75.1.0", 
                  "wheel==0.44.0",
                  "ninja"],  # Required by scikit-build-core when isolation is off
               capture_output=True, check=False, env=build_env)

    cmd = [
        pip_exe, "install", "llama-cpp-python",
        "--force-reinstall",
        "--no-cache-dir",
        "--no-binary", "llama-cpp-python",  # Only compile llama-cpp-python from source
        "--no-build-isolation",
        "--verbose",
    ]

    attempt = 1
    error_details = ""
    while attempt <= MAX_RETRIES:
        start_time = time.time()
        last_update = start_time
        pkg_name = "llama-cpp-python"
        
        try:
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=build_env
            )

            # Consume stdout while process runs (prevents pipe buffer deadlock)
            build_log_lines = []
            while process.poll() is None:
                line = process.stdout.readline()
                if line:
                    build_log_lines.append(line.decode('utf-8', errors='replace').strip())
                now = time.time()
                if now - last_update >= 0.5:
                    last_update = now
                    elapsed = now - start_time
                    fake_percent = min(95, 5 + (elapsed / 600) * 90)
                    show_progress(pkg_name, fake_percent, 0, 0, elapsed, 0)
                time.sleep(0.05)

            # Drain any remaining output
            remaining = process.stdout.read()
            if remaining:
                for line in remaining.decode('utf-8', errors='replace').splitlines():
                    build_log_lines.append(line.strip())

            elapsed = time.time() - start_time
            
            if process.returncode == 0:
                show_done(pkg_name, 50*1024*1024, elapsed)
                print_status(f"llama-cpp-python compiled (Vulkan + CPU, {format_time(elapsed)})")
                return True, ""
            else:
                show_failed(pkg_name, elapsed)
                error_details = f"Build error (last 5 lines):\n" + "\n".join(build_log_lines[-5:])
                print(f"  [RETRY {attempt}/{MAX_RETRIES}] Compile failed")
                
                if attempt >= MAX_RETRIES:
                    return False, error_details
                
                time.sleep(RETRY_DELAY)
                attempt += 1
                continue
                
        except Exception as e:
            elapsed = time.time() - start_time
            show_failed(pkg_name, elapsed)
            error_details = f"Compile error: {str(e)}"
            time.sleep(RETRY_DELAY)
            attempt += 1
            continue

    return False, error_details

def create_sample_visualizes():
    """Create anims directory for AVIF avatar animations."""
    anims_dir = BASE_DIR / "anims"
    anims_dir.mkdir(parents=True, exist_ok=True)
    waiting_avif    = anims_dir / "robot_eye_waiting.avif"
    working_avif    = anims_dir / "robot_eye_working.avif"
    organizing_avif = anims_dir / "robot_eye_organizing.avif"

    if waiting_avif.exists() and working_avif.exists() and organizing_avif.exists():
        print_status("Avatar files found")
        return

    info_file = anims_dir / "README_AVATARS.txt"
    if not info_file.exists():
        with open(info_file, "w") as f:
            f.write("""Avatar Animation Files
Place your AVIF avatar animation files in this directory:
Required files:
robot_eye_waiting.avif     (shown when AI is waiting for user input)
robot_eye_working.avif     (shown when AI is normally processing/thinking, 10 threads)
robot_eye_organizing.avif  (shown when AI is planning/brainstorming, 20 threads)
""")
        print_status("Anims directory created")

    missing = [f.name for f in (waiting_avif, working_avif, organizing_avif) if not f.exists()]
    if missing:
        print_info(f"[!] Avatar files not found - add {', '.join(missing)} to ./anims/")

def create_system_ini(vulkan: bool, cpu: Dict) -> bool:
    try:
        config = configparser.ConfigParser()
        config['system'] = {
            'platform': 'windows',
            'python_version': PYTHON_VERSION,
            'vulkan_available': str(vulkan).lower(),
            'cpu_threads': str(cpu.get('cores_logical', 24)),
            'cpu_caps': json.dumps(cpu),
            'gradio_version': SELECTED_GRADIO,
        }
        with open(BASE_DIR / "data" / "constants.ini", "w") as f:
            config.write(f)
        print_status("Configuration created (constants.ini)")
        return True
    except Exception as e:
        print_status(f"Failed: {e}", False)
        return False

def create_configuration_json(vulkan: bool, cpu: Dict) -> bool:
    """Create (or overwrite) configuration.json with default configuration values.
    Always writes from scratch — never merges with an existing file."""
    try:
        cpu_threads = cpu.get('cores_logical', psutil.cpu_count(logical=True) or 1)
        config_data = {
            # Identity
            'ai_name': "Manager-Qwen",
            'ai_role': (
                "You are a Qwen based Agent-Manager, an uncensored and capable assistant with, text, vision, image generation, planning, TTS, and agentic tools. You discus with the user, brainstorm and plan, then think step-by-step, completing critical tasks, and delegating less critical or specialized tasks to specialised sub-agents, working on projects to completion, while updating project appropriately to any further user input during the execution of the project."
            ),
            'user_name': "User-Human",
            'user_role': "The Director whom comes up with the Project and sets Guidlines/Conditions, and also the resource of 1 Human, that, as required, can complete manual real world tasks, in order to progress Projects towards completion. The User-Human is also the creator of Agent-Gradio-Gguf, and so able to upgrade/fix the interface/backend with Opus, should the AI determine this to be critical to success.",
            'notes': "Be helpful, accurate, and concise.",
            # Mode
            'mode': "chat",
            'tts_enabled': False,
            'thinking_mode': True,
            # Generation
            'temperature': 0.6,
            # Manager (WSL2, CPU-only)
            'n_gpu_layers': 0,
            'manager_api_base': "http://127.0.0.1:8080/v1",
            'manager_context': 32768,
            # Agent GPU (RX 470)
            'agent_gpu_index': 1,
            'agent_vram_mb': 8192,
            # Display GPU (GTX 1060 - info only)
            'selected_gpu': 0,
            'gpu_vram_mb': 3072,
            # CPU
            'cpu_threads': cpu_threads,
            'selected_cpu': 0,
            # Memory management
            'auto_unload': True,
            'max_memory_percent': 80.0,
            # Features
            'rag_enabled': True,
            'visualize_file': "",
            'models_dir': str(BASE_DIR / "models"),
            'manager_models_dir': "",           # empty = falls back to models_dir; set in Configuration page if the manager's AWQ folder lives elsewhere
            'manager_model_folder': "",          # empty = auto-pick first match found; set via the Configuration page's manager model dropdown to pin a specific size/revision
            'manager_tokenizer_source': "",      # HF repo id or local path — normally auto-set to the manager's own AWQ folder; set in Configuration page to override
            'auto_save': True,
            'max_history_slots': 10,
            # Metadata
            '_saved': datetime.datetime.now().isoformat(),
            '_cpu_detected': cpu.get('cpu_name', 'Unknown'),
            '_vulkan_available': vulkan,
            '_installer_version': "3.0.0",
            '_architecture': "Manager=WSL2-CPU(vLLM) | Agent(single-slot)=text_image/image_gen/tts_voice",
        }
        
        json_path = BASE_DIR / "data" / "configuration.json"
        # Explicitly remove any existing file — clean install means clean settings
        if json_path.exists():
            json_path.unlink()
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=4)

        print_status("Initial configuration created (configuration.json)")
        print_info(f"  CPU Threads: {cpu_threads} (detected)")
        print_info(f"  Manager: WSL2 CPU (0 GPU layers)")
        print_info(f"  Agent GPU index: {config_data['agent_gpu_index']} (auto-detected at runtime)")
        return True

    except Exception as e:
        print_status(f"Failed to create configuration.json: {e}", False)
        return False

# ============================================================================
# Default profiles.json contents
# ============================================================================
# DUPLICATE BY NECESSITY — this must stay in sync with scripts/configure.py's
# DEFAULT_MANAGER_NAME / DEFAULT_MANAGER_ROLE / DEFAULT_HUMAN_NAME /
# DEFAULT_HUMAN_ROLE / DEFAULT_SOCIAL_ACCOUNTS / DEFAULT_PERSONAL_ACCOUNTS,
# which are the source of truth for the running program.
#
# It cannot simply import them: create_profiles_json() runs at install STEP 2,
# before the venv (STEP 3) and the dependencies (STEP 4) exist, and
# configure.py imports cpuinfo/psutil at module level — so the import would
# blow up on a clean machine, which is the only machine that runs this.
# Editing a default means editing BOTH places. If they ever drift, configure.py
# wins at runtime: load_profiles() fills any missing key from its own
# defaults, so the worst case is a fresh profiles.json showing a stale value
# until Restore Defaults is pressed on the Preferences page.
DEFAULT_PROFILES_DATA = {
    "manager": {
        "name": "Manager-Qwen",
        "role": (
            "You are a Qwen based Agent-Manager, an uncensored and capable assistant "
            "with deployable Agents for, text, vision, image encoding, image "
            "generation, TTS, as well as Agentic tools. You discus with the user, "
            "brainstorm and plan, then think step-by-step, completing critical tasks, "
            "and delegating less critical or specialized tasks to specialized "
            "sub-agents, working on projects to completion, while updating project "
            "appropriately to any further user input during the execution of the "
            "project, in order to achieve set goals."
        ),
    },
    "human": {
        "name": "User",
        "role": (
            "The, Director and Principle, who thinks up the ideas for the Projects."
        ),
        "social": {
            "x_login":       "", "x_password":       "",
            "github_login":  "", "github_password":  "",
            "tiktok_login":  "", "tiktok_password":  "",
            "youtube_login": "", "youtube_password": "",
        },
    },
    # Personal accounts — email + bank. Top-level rather than under "human"
    # because the ai_email_* pair belongs to the Manager and the rest to the
    # human. Account number (8 digits) and sort code (6 digits) are STRINGS:
    # a leading zero on a sort code does not survive being an int.
    "accounts": {
        "ai_email_login":       "", "ai_email_password":    "",
        "human_email_login":    "", "human_email_password": "",
        "bank_corp_name":       "",
        "bank_user_name":       "",
        "bank_account_number":  "",
        "bank_sort_code":       "",
    },
}


def create_profiles_json() -> bool:
    """Create (or overwrite) profiles.json with default Manager/Human profile values.
    Always writes from scratch — never merges with an existing file."""
    try:
        profiles_data = copy.deepcopy(DEFAULT_PROFILES_DATA)

        json_path = BASE_DIR / "data" / "profiles.json"
        if json_path.exists():
            json_path.unlink()
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(profiles_data, f, indent=4)

        print_status("Initial profiles created (profiles.json)")
        return True

    except Exception as e:
        print_status(f"Failed to create profiles.json: {e}", False)
        return False

def check_wsl2() -> Tuple[bool, str]:
    """Check WSL2 is available by running a simple echo through it.
    Uses 'wsl echo ok' — avoids the UTF-16LE output issue from
    'wsl --list --verbose' which causes false negatives on Windows 10."""
    try:
        result = subprocess.run(
            ["wsl", "echo", "ok"],
            capture_output=True, text=True, timeout=10,
            creationflags=subprocess.CREATE_NO_WINDOW
        )
        if result.returncode == 0 and "ok" in result.stdout:
            return True, "WSL2 available"
        return False, (
            f"WSL2 not responding (exit {result.returncode}). "
            "Run: wsl --install  then reboot."
        )
    except FileNotFoundError:
        return False, (
            "wsl.exe not found. Install WSL2:\n"
            "  1. Open PowerShell as Administrator\n"
            "  2. Run: wsl --install\n"
            "  3. Reboot and complete Ubuntu setup"
        )
    except Exception as e:
        return False, f"WSL2 check failed: {e}"

def _has_internet() -> bool:
    try:
        requests.get("https://huggingface.co", timeout=5)
        return True
    except:
        return False

def print_detection_table() -> Tuple[bool, List[str]]:
    """Run system checks and print a two‑column table.
    Returns (critical_ok, list_of_warning_messages)."""
    import ctypes
    import platform
    
    # 1. Windows version
    win_ver = platform.release()
    win_ok = win_ver in ("10", "11")
    win_ver_str = win_ver if win_ok else f"{win_ver} (unsupported)"
    
    # 2. Administrator rights
    is_admin = False
    try:
        is_admin = ctypes.windll.shell32.IsUserAnAdmin() != 0
    except:
        pass

    # 3. MSVC Build Tools
    msvc_ok = detect_msvc_env() is not None

    # 4. Vulkan SDK / runtime
    vulkan_ok, _ = check_vulkan()
    vulkan_sdk_ok = detect_vulkan_sdk() is not None
    vulkan_ok = vulkan_ok or vulkan_sdk_ok

    # 5. WSL2
    wsl_ok, wsl_msg = check_wsl2()

    # 6. Git
    git_ok = shutil.which("git") is not None

    # 7. Internet (non-critical)
    internet_ok = _has_internet()

    # Prepare two‑column output
    left_col = []
    right_col = []

    checks = [
        ("Windows 10+", win_ok, True, win_ver_str),
        ("Administrator", is_admin, True, "   "),
        ("MSVC Build Tools", msvc_ok, True, "   "),
        ("Vulkan SDK", vulkan_ok, True, "   "),
        ("WSL2 + distro v2", wsl_ok, True, wsl_msg if not wsl_ok else "   "),
        ("Git", git_ok, False, "   "),
        ("Internet", internet_ok, False, "Ensure internet is on and Python is allowed through the firewall before proceeding."),
    ]

    critical_ok = True
    warnings = []

    for i, (label, ok, critical, extra) in enumerate(checks):
        if ok:
            status = "OK    "
        elif not ok and critical:
            status = "FAIL "
            critical_ok = False
        elif not ok and not critical:
            status = "WARN "
            warnings.append(f"{label}: {extra}")
        else:
            status = "FAIL "

        line = f"  {status} {label:<22}   "
        if i % 2 == 0:
            left_col.append(line)
        else:
            right_col.append(line)

    print()
    print("Detections...")
    max_rows = max(len(left_col), len(right_col))
    for i in range(max_rows):
        left = left_col[i] if i < len(left_col) else "   "
        right = right_col[i] if i < len(right_col) else "   "
        print(f"{left:<40} {right}")

    if not critical_ok:
        print("\n" + "=" * 79)
        print("  CRITICAL: Missing required components. Please fix before installing.")
        print("  - MSVC Build Tools: https://visualstudio.microsoft.com/visual-cpp-build-tools/")
        print("  - Vulkan SDK: https://vulkan.lunarg.com/sdk/home")
        print("  - WSL2: Run 'wsl --install' as Administrator")
        print("=" * 79)
    elif warnings:
        print("\n  Note: Non-critical warnings – installation may still proceed, but:")
        for w in warnings:
            print(f"  - {w}")

    return critical_ok, warnings

def setup_portproxy() -> Tuple[bool, str]:
    """
    Create a Windows port proxy: 127.0.0.1:8080 → WSL2_IP:8080
    This makes the WSL2 vLLM server reachable at a stable localhost address
    even after WSL2's internal IP changes on reboot.
    Requires Administrator privileges (checked by the .bat before launch).
    """
    try:
        # Get WSL2 IP
        ip_result = subprocess.run(
            ["wsl", "hostname", "-I"],
            capture_output=True, text=True, timeout=10,
            creationflags=subprocess.CREATE_NO_WINDOW
        )
        if ip_result.returncode != 0 or not ip_result.stdout.strip():
            return False, "Could not determine WSL2 IP address — is WSL2 running?"
            
        wsl_ip = ip_result.stdout.strip().split()[0]
        print_info(f"  WSL2 IP: {wsl_ip}")
        
        # Remove any existing rule first (ignore failure — may not exist)
        subprocess.run(
            ["netsh", "interface", "portproxy", "delete", "v4tov4",
             "listenport=8080", "listenaddress=127.0.0.1"],
            capture_output=True, timeout=10,
            creationflags=subprocess.CREATE_NO_WINDOW
        )

        # Add new rule
        add_result = subprocess.run(
            ["netsh", "interface", "portproxy", "add", "v4tov4",
             "listenport=8080", "listenaddress=127.0.0.1",
             f"connectport={MANAGER_PORT}", f"connectaddress={wsl_ip}"],
            capture_output=True, text=True, timeout=10,
            creationflags=subprocess.CREATE_NO_WINDOW
        )
        if add_result.returncode != 0:
            return False, (
                f"netsh portproxy failed (code {add_result.returncode}).\n"
                f"stderr: {add_result.stderr.strip()}\n"
                "Ensure the installer is running as Administrator."
            )

        print_status(f"Port proxy: 127.0.0.1:8080 → {wsl_ip}:{MANAGER_PORT}")
        return True, f"Proxy set: 127.0.0.1:8080 → {wsl_ip}:{MANAGER_PORT}"

    except Exception as e:
        return False, f"Port proxy setup failed: {e}"

def build_llama_agents(cpu: Dict) -> Tuple[bool, str]:
    """
    Clone llama.cpp b9585, compile llama-server.exe for Windows with:
    - Vulkan (agent GPU, RX 470)
    - AVX2 + FMA + F16C (CPU fallback threads)
    Output binaries are copied to llama_agents/.
    Requires: cmake, git, Vulkan SDK, MSVC Build Tools.
    """
    agents_dir = BASE_DIR / LLAMA_AGENTS_DIR
    server_exe = agents_dir / "llama-server.exe"
    if server_exe.exists():
        print_status(f"llama-server.exe already built: {server_exe}")
        return True, ""

    vcvarsall = detect_msvc_env()
    if not vcvarsall:
        return False, (
            "MSVC Build Tools not found.\n"
            "Install from https://visualstudio.microsoft.com/visual-cpp-build-tools/\n"
            "Select: 'Desktop development with C++' workload."
        )

    vulkan_sdk = detect_vulkan_sdk()
    if not vulkan_sdk:
        return False, (
            "Vulkan SDK not found.\n"
            "Install from https://vulkan.lunarg.com/sdk/home\n"
            "Then re-run the installer."
        )

    # CRITICAL CHECK: Ensure glslangValidator exists (required for vulkan-shaders-gen)
    vulkan_bin = str(Path(vulkan_sdk) / "Bin")
    glslang_validator = Path(vulkan_bin) / "glslangValidator.exe"
    if not glslang_validator.exists():
        return False, (
            f"Vulkan SDK found, but glslangValidator.exe is missing from:\n{vulkan_bin}\n"
            "This tool is REQUIRED to compile Vulkan shaders for llama.cpp.\n"
            "Please ensure you installed the FULL Vulkan SDK (not just the runtime) from:\n"
            "https://vulkan.lunarg.com/sdk/home\n"
            "During installation, ensure 'Shader Compiler' / 'glslang' components are selected."
        )

    # ── CRITICAL: Use a SHORT path with NO special characters for the build. ──────
    # The project path contains '+' (e.g. "ZAI+Claude") which breaks the
    # vulkan-shaders-gen custom build step — Windows command-line argument
    # parsing and CMake's ExternalProject treat '+' as a path separator in
    # some contexts, causing vulkan-shaders-gen.exe to exit with code 1.
    # We clone and build in C:\llama_bld\ to guarantee a clean path.
    safe_build_root = Path(r"C:\llama_bld")
    clone_dir = safe_build_root / f"llama-{LLAMACPP_VERSION}"
    safe_build_root.mkdir(parents=True, exist_ok=True)
    agents_dir.mkdir(parents=True, exist_ok=True)

    # Clean any stale build dirs from previous runs (both old "build" and "build_agents" names).
    for stale_name in ("build", "build_agents"):
        stale = clone_dir / stale_name
        if stale.exists():
            print_info(f"  Clearing stale build dir: {stale_name}/")
            try:
                shutil.rmtree(stale, ignore_errors=True)
            except Exception:
                pass

    if not (clone_dir / "CMakeLists.txt").exists():
        # Retry git clone with escalating backoff — network drops mid-transfer
        # (schannel / early EOF) are common on Windows and always fixable by retrying.
        clone_delays = [0, 5, 15, 30, 60, 90]
        clone_success = False
        clone_error = ""
        for attempt, delay in enumerate(clone_delays, 1):
            if delay > 0:
                print_info(f"  Retrying clone in {delay}s (attempt {attempt}/{len(clone_delays)})...")
                time.sleep(delay)

            # Always wipe any partial clone before each attempt
            if clone_dir.exists():
                print_info("  Clearing partial clone...")
                try:
                    shutil.rmtree(clone_dir, ignore_errors=True)
                    if clone_dir.exists():
                        subprocess.run(
                            ["cmd", "/c", "rmdir", "/s", "/q", str(clone_dir)],
                            capture_output=True, check=False
                        )
                except Exception:
                    pass

            print_info(f"  Cloning llama.cpp {LLAMACPP_VERSION} (attempt {attempt}/{len(clone_delays)})...")
            r = subprocess.run(
                ["git", "clone",
                 "--depth", "1",
                 "--branch", LLAMACPP_VERSION,
                 # Increase curl buffer and retry settings to survive flaky connections
                 "-c", "http.postBuffer=524288000",
                 "-c", "core.compression=0",
                 "https://github.com/ggml-org/llama.cpp",
                 str(clone_dir)],
                capture_output=True, text=True, timeout=600
            )
            if r.returncode == 0 and (clone_dir / "CMakeLists.txt").exists():
                clone_success = True
                break
            clone_error = r.stderr[-400:]
            print_info(f"  Clone attempt {attempt} failed: {clone_error[-120:]}")

        if not clone_success:
            return False, f"git clone failed after {len(clone_delays)} attempts:\n{clone_error}"
        print_status(f"Cloned llama.cpp {LLAMACPP_VERSION}")
    else:
        print_status(f"llama.cpp source already present: {clone_dir}")

    # Build environment with MSVC + Vulkan SDK
    build_env = os.environ.copy()
    build_env["VULKAN_SDK"] = vulkan_sdk
    if vulkan_bin not in build_env.get("PATH", ""):
        build_env["PATH"] = vulkan_bin + ";" + build_env.get("PATH", "")

    # Activate MSVC x64
    try:
        r = subprocess.run(
            f'"{vcvarsall}" x64 && set',
            shell=True, capture_output=True, text=True, timeout=30
        )
        if r.returncode == 0:
            for line in r.stdout.splitlines():
                if "=" in line:
                    k, _, v = line.partition("=")
                    build_env[k.strip()] = v.strip()
            print_status("MSVC x64 environment activated")
        else:
            print_info(f"  Warning: vcvarsall exited {r.returncode}")
    except Exception as e:
        print_info(f"  Warning: MSVC activation failed: {e}")

    build_dir = clone_dir / "build_agents"
    avx2 = "ON" if cpu.get("avx2") else "OFF"
    fma = "ON" if cpu.get("fma") else "OFF"
    f16c = "ON" if cpu.get("f16c") else "OFF"
    cores = cpu.get("cores_logical", os.cpu_count() or 4)

    # Always wipe the build directory before configuring — a stale CMakeCache.txt
    # from a previous run with a different generator silently overrides the -G flag.
    if build_dir.exists():
        print_info("  Clearing stale build directory...")
        shutil.rmtree(build_dir, ignore_errors=True)
    build_dir.mkdir(parents=True, exist_ok=True)

    print_info(f"  CMake: Vulkan=ON  AVX2={avx2}  FMA={fma}  F16C={f16c}")

    # Absolute path to cmake.exe inside the venv
    cmake_exe = str(VENV_DIR / "Scripts" / "cmake.exe")
    
    # SAFEGUARD: Ensure cmake is installed in the venv
    if not Path(cmake_exe).exists():
        print_info("  CMake not found in venv, installing...")
        pip_exe = str(VENV_DIR / "Scripts" / "pip.exe")
        subprocess.run([pip_exe, "install", "cmake==3.30.2"], capture_output=True, check=False)

    # Verify glslc is available — vulkan-shaders-gen passes --glslc path to compile shaders
    glslc_exe = Path(vulkan_bin) / "glslc.exe"
    if not glslc_exe.exists():
        return False, (
            f"glslc.exe not found in Vulkan SDK Bin:\n  {vulkan_bin}\n"
            "glslc is REQUIRED to compile Vulkan GLSL shaders to SPIR-V.\n"
            "Reinstall the Vulkan SDK and ensure 'Shader Compiler' is selected.\n"
            "https://vulkan.lunarg.com/sdk/home"
        )

    # Build CMAKE_PREFIX_PATH so CMake can find SPIRV-Headers (required by ggml-vulkan).
    # The Vulkan SDK ships SPIRV-Headers CMake config at <SDK>/share/cmake/SPIRV-Headers
    # or <SDK>/cmake/SPIRV-Headers depending on SDK version.
    spirv_prefix_candidates = [
        str(Path(vulkan_sdk) / "share"),          # SDK 1.3.250+
        str(Path(vulkan_sdk) / "cmake"),           # older layouts
        vulkan_sdk,                                 # fallback: root
    ]
    cmake_prefix_path = ";".join(spirv_prefix_candidates)

    # Use the Visual Studio generator — llama.cpp b9585's ExternalProject_Add for
    # vulkan-shaders-gen uses $<CONFIG> generator expressions which ONLY work correctly
    # with multi-config generators (VS). NMake would produce a literal "$<CONFIG>" path.
    # Detect VS version from the activated environment to pick the right generator string.
    vs_generator = "Visual Studio 17 2022"  # default: VS 2022
    vs_version_map = {
        "16.": "Visual Studio 16 2019",
        "17.": "Visual Studio 17 2022",
    }
    vs_install_ver = build_env.get("VisualStudioVersion", "")
    for prefix, gen in vs_version_map.items():
        if vs_install_ver.startswith(prefix):
            vs_generator = gen
            break
    print_info(f"  Generator: {vs_generator}")

    cmake_cfg = subprocess.run(
        [cmake_exe, "-S", str(clone_dir), "-B", str(build_dir),
         "-G", vs_generator,
         "-A", "x64",
         f"-DCMAKE_PREFIX_PATH={cmake_prefix_path}",
         f"-DVULKAN_SDK={vulkan_sdk}",
         f"-DVulkan_INCLUDE_DIR={Path(vulkan_sdk) / 'Include'}",
         f"-DVulkan_LIBRARY={Path(vulkan_sdk) / 'Lib' / 'vulkan-1.lib'}",
         f"-DVulkan_GLSLC_EXECUTABLE={glslc_exe}",
         "-DGGML_VULKAN=ON",
         f"-DGGML_AVX2={avx2}",
         f"-DGGML_FMA={fma}",
         f"-DGGML_F16C={f16c}",
         "-DBUILD_SHARED_LIBS=OFF",
         "-DLLAMA_BUILD_TESTS=OFF",
         "-DLLAMA_BUILD_EXAMPLES=OFF",
         "-DLLAMA_BUILD_SERVER=ON",
         f"-DCMAKE_BUILD_PARALLEL_LEVEL={cores}"],
        capture_output=True, text=True, env=build_env, timeout=120
    )
    if cmake_cfg.returncode != 0:
        return False, f"CMake configure failed:\n--- STDERR ---\n{cmake_cfg.stderr[-1000:]}\n--- STDOUT ---\n{cmake_cfg.stdout[-1000:]}"
    print_status("CMake configured")

    print_info(f"  Building with {cores} cores (this takes ~5 min)...")

    cmake_build = subprocess.run(
        [cmake_exe, "--build", str(build_dir),
         "--config", "Release",
         "--target", "llama-server",
         "--",
         f"/m:{cores}"],        # MSBuild parallel jobs
        capture_output=True, text=True, env=build_env, timeout=900
    )
    
    if cmake_build.returncode != 0:
        err_msg = (
            f"CMake build failed:\n"
            f"--- STDERR (last 2000 chars) ---\n{cmake_build.stderr[-2000:]}\n"
            f"--- STDOUT (last 2000 chars) ---\n{cmake_build.stdout[-2000:]}"
        )
        return False, err_msg

    # VS generator places output in bin/Release/
    src_bin = build_dir / "bin" / "Release"
    if not src_bin.exists():
        src_bin = build_dir / "bin"   # fallback if Release subdir absent

    copied = 0
    for f in src_bin.glob("*.exe"):
        shutil.copy2(f, agents_dir / f.name)
        copied += 1
    for f in src_bin.glob("*.dll"):
        shutil.copy2(f, agents_dir / f.name)

    if copied == 0:
        return False, f"No .exe found in {src_bin} — check build output"

    print_status(f"llama-server.exe built → {agents_dir}/ ({copied} exe, Vulkan+AVX2)")
    return True, ""

def build_sd_agent(cpu: Dict) -> Tuple[bool, str]:
    """
    Clone stable-diffusion.cpp, compile sd-cli.exe for Windows with Vulkan +
    AVX2/FMA/F16C. Output binary is copied to sd_agent/.

    Used by playground.py's run_stable_diffusion() (image_gen_agent) — this
    is a ONE-SHOT CLI, not a server, so unlike build_llama_agents() there's
    no --target llama-server equivalent to pin down: sd.cpp's example CLI
    target name isn't guaranteed stable across versions, so we build
    everything (SD_BUILD_EXAMPLES=ON) and search multiple known output
    locations for the resulting sd-cli.exe (or legacy sd.exe), the same
    defensive pattern a working reference implementation (a separate
    Z-Image-Turbo app using the same stable-diffusion.cpp) uses.

    Requires: cmake, git, Vulkan SDK, MSVC Build Tools (same as llama.cpp).
    """
    sd_dir = BASE_DIR / SD_AGENTS_DIR
    existing = None
    for name in ("sd-cli.exe", "sd.exe"):
        if (sd_dir / name).exists():
            existing = sd_dir / name
            break
    if existing:
        print_status(f"{existing.name} already built: {existing}")
        return True, ""

    vcvarsall = detect_msvc_env()
    if not vcvarsall:
        return False, (
            "MSVC Build Tools not found.\n"
            "Install from https://visualstudio.microsoft.com/visual-cpp-build-tools/\n"
            "Select: 'Desktop development with C++' workload."
        )

    vulkan_sdk = detect_vulkan_sdk()
    if not vulkan_sdk:
        return False, (
            "Vulkan SDK not found.\n"
            "Install from https://vulkan.lunarg.com/sdk/home\n"
            "Then re-run the installer."
        )

    # Same shader-compiler check as build_llama_agents() — sd.cpp vendors
    # the same ggml-vulkan backend and shader-gen step.
    vulkan_bin = str(Path(vulkan_sdk) / "Bin")
    glslc_exe = Path(vulkan_bin) / "glslc.exe"
    if not glslc_exe.exists():
        return False, (
            f"glslc.exe not found in Vulkan SDK Bin:\n  {vulkan_bin}\n"
            "glslc is REQUIRED to compile Vulkan GLSL shaders to SPIR-V.\n"
            "Please ensure you installed the FULL Vulkan SDK (not just the runtime) from:\n"
            "https://vulkan.lunarg.com/sdk/home\n"
            "During installation, ensure 'Shader Compiler' / 'glslang' components are selected."
        )

    # ── Same short-path fix as build_llama_agents() ──────────────────────────
    # A path containing '+' (e.g. "ZAI+Claude") breaks vulkan-shaders-gen's
    # ExternalProject_Add step. Build in C:\sd_bld\ to guarantee a clean path,
    # regardless of where this project itself is installed.
    safe_build_root = Path(r"C:\sd_bld")
    clone_dir = safe_build_root / "stable-diffusion.cpp"
    safe_build_root.mkdir(parents=True, exist_ok=True)
    sd_dir.mkdir(parents=True, exist_ok=True)

    build_dir = clone_dir / "build_agents"
    if build_dir.exists():
        print_info("  Clearing stale build directory...")
        shutil.rmtree(build_dir, ignore_errors=True)

    # A previous run (before this installer fixed the submodule bug) may
    # have left a clone with CMakeLists.txt present but ggml/ empty — check
    # for that too, or the stale partial clone gets reused as-is and fails
    # the exact same "does not contain a CMakeLists.txt file" error again.
    needs_clone = (
        not (clone_dir / "CMakeLists.txt").exists()
        or not (clone_dir / "ggml" / "CMakeLists.txt").exists()
    )
    if needs_clone:
        if clone_dir.exists():
            print_info("  Existing clone is incomplete (missing ggml submodule) — re-cloning...")
        clone_delays = [0, 5, 15, 30, 60, 90]
        clone_success = False
        clone_error = ""
        for attempt, delay in enumerate(clone_delays, 1):
            if delay > 0:
                print_info(f"  Retrying clone in {delay}s (attempt {attempt}/{len(clone_delays)})...")
                time.sleep(delay)

            if clone_dir.exists():
                print_info("  Clearing partial clone...")
                try:
                    shutil.rmtree(clone_dir, ignore_errors=True)
                    if clone_dir.exists():
                        subprocess.run(
                            ["cmd", "/c", "rmdir", "/s", "/q", str(clone_dir)],
                            capture_output=True, check=False
                        )
                except Exception:
                    pass

            print_info(f"  Cloning stable-diffusion.cpp (attempt {attempt}/{len(clone_delays)})...")
            # NOTE: no version tag pinned here (unlike LLAMACPP_VERSION for
            # llama.cpp) — tracking master, same as the working reference
            # implementation this was ported from. Pin to a specific commit
            # here if you need build reproducibility.
            #
            # --recurse-submodules is REQUIRED: stable-diffusion.cpp vendors
            # ggml as a git submodule (unlike llama.cpp, which inlines it).
            # Without this flag, ggml/ is an empty directory and CMake fails
            # at add_subdirectory(ggml) with "does not contain a
            # CMakeLists.txt file" — confirmed the actual cause of a real
            # build failure during testing.
            r = subprocess.run(
                ["git", "clone",
                 "--depth", "1",
                 "--recurse-submodules",
                 "--shallow-submodules",
                 "-c", "http.postBuffer=524288000",
                 "-c", "core.compression=0",
                 SD_CPP_SOURCE_URL,
                 str(clone_dir)],
                capture_output=True, text=True, timeout=600
            )
            ggml_ok = (clone_dir / "ggml" / "CMakeLists.txt").exists()
            if r.returncode == 0 and (clone_dir / "CMakeLists.txt").exists() and ggml_ok:
                clone_success = True
                break
            if r.returncode == 0 and not ggml_ok:
                # Clone itself succeeded but the submodule didn't come down
                # (seen with some git/proxy configs even with
                # --recurse-submodules) — fetch it explicitly as a fallback
                # before giving up on this attempt.
                print_info("  ggml submodule missing after clone, fetching explicitly...")
                sub_r = subprocess.run(
                    ["git", "submodule", "update", "--init", "--recursive", "--depth", "1"],
                    cwd=str(clone_dir), capture_output=True, text=True, timeout=300
                )
                if sub_r.returncode == 0 and (clone_dir / "ggml" / "CMakeLists.txt").exists():
                    clone_success = True
                    break
                clone_error = sub_r.stderr[-400:]
                print_info(f"  submodule fetch failed: {clone_error[-120:]}")
                continue
            clone_error = r.stderr[-400:]
            print_info(f"  Clone attempt {attempt} failed: {clone_error[-120:]}")

        if not clone_success:
            return False, f"git clone failed after {len(clone_delays)} attempts:\n{clone_error}"
        print_status("Cloned stable-diffusion.cpp")
    else:
        print_status(f"stable-diffusion.cpp source already present: {clone_dir}")

    build_env = os.environ.copy()
    build_env["VULKAN_SDK"] = vulkan_sdk
    if vulkan_bin not in build_env.get("PATH", ""):
        build_env["PATH"] = vulkan_bin + ";" + build_env.get("PATH", "")

    try:
        r = subprocess.run(
            f'"{vcvarsall}" x64 && set',
            shell=True, capture_output=True, text=True, timeout=30
        )
        if r.returncode == 0:
            for line in r.stdout.splitlines():
                if "=" in line:
                    k, _, v = line.partition("=")
                    build_env[k.strip()] = v.strip()
            print_status("MSVC x64 environment activated")
        else:
            print_info(f"  Warning: vcvarsall exited {r.returncode}")
    except Exception as e:
        print_info(f"  Warning: MSVC activation failed: {e}")

    avx2 = "ON" if cpu.get("avx2") else "OFF"
    fma = "ON" if cpu.get("fma") else "OFF"
    f16c = "ON" if cpu.get("f16c") else "OFF"
    cores = cpu.get("cores_logical", os.cpu_count() or 4)

    build_dir.mkdir(parents=True, exist_ok=True)
    print_info(f"  CMake: SD_VULKAN=ON  AVX2={avx2}  FMA={fma}  F16C={f16c}")

    cmake_exe = str(VENV_DIR / "Scripts" / "cmake.exe")
    if not Path(cmake_exe).exists():
        print_info("  CMake not found in venv, installing...")
        pip_exe = str(VENV_DIR / "Scripts" / "pip.exe")
        subprocess.run([pip_exe, "install", "cmake==3.30.2"], capture_output=True, check=False)

    spirv_prefix_candidates = [
        str(Path(vulkan_sdk) / "share"),
        str(Path(vulkan_sdk) / "cmake"),
        vulkan_sdk,
    ]
    cmake_prefix_path = ";".join(spirv_prefix_candidates)

    # Same VS multi-config generator as build_llama_agents() — sd.cpp vendors
    # the same ggml ExternalProject_Add shader-gen step that needs it.
    vs_generator = "Visual Studio 17 2022"
    vs_version_map = {"16.": "Visual Studio 16 2019", "17.": "Visual Studio 17 2022"}
    vs_install_ver = build_env.get("VisualStudioVersion", "")
    for prefix, gen in vs_version_map.items():
        if vs_install_ver.startswith(prefix):
            vs_generator = gen
            break
    print_info(f"  Generator: {vs_generator}")

    cmake_cfg = subprocess.run(
        [cmake_exe, "-S", str(clone_dir), "-B", str(build_dir),
         "-G", vs_generator,
         "-A", "x64",
         f"-DCMAKE_PREFIX_PATH={cmake_prefix_path}",
         f"-DVULKAN_SDK={vulkan_sdk}",
         f"-DVulkan_INCLUDE_DIR={Path(vulkan_sdk) / 'Include'}",
         f"-DVulkan_LIBRARY={Path(vulkan_sdk) / 'Lib' / 'vulkan-1.lib'}",
         f"-DVulkan_GLSLC_EXECUTABLE={glslc_exe}",
         "-DSD_VULKAN=ON",
         "-DSD_BUILD_EXAMPLES=ON",
         f"-DGGML_AVX2={avx2}",
         f"-DGGML_FMA={fma}",
         f"-DGGML_F16C={f16c}",
         "-DBUILD_SHARED_LIBS=OFF",
         f"-DCMAKE_BUILD_PARALLEL_LEVEL={cores}"],
        capture_output=True, text=True, env=build_env, timeout=120
    )
    if cmake_cfg.returncode != 0:
        return False, f"CMake configure failed:\n--- STDERR ---\n{cmake_cfg.stderr[-1000:]}\n--- STDOUT ---\n{cmake_cfg.stdout[-1000:]}"
    print_status("CMake configured")

    print_info(f"  Building with {cores} cores (this takes several minutes)...")

    # No --target: sd.cpp's example CLI target name isn't guaranteed stable
    # across versions, so build everything and search for the output below
    # (SD_BUILD_EXAMPLES=ON is what makes the CLI/server examples build at all).
    cmake_build = subprocess.run(
        [cmake_exe, "--build", str(build_dir),
         "--config", "Release",
         "--",
         f"/m:{cores}"],
        capture_output=True, text=True, env=build_env, timeout=1200
    )
    if cmake_build.returncode != 0:
        err_msg = (
            f"CMake build failed:\n"
            f"--- STDERR (last 2000 chars) ---\n{cmake_build.stderr[-2000:]}\n"
            f"--- STDOUT (last 2000 chars) ---\n{cmake_build.stdout[-2000:]}"
        )
        return False, err_msg

    # Search known output locations first, then fall back to a recursive
    # search — sd.cpp's example output layout has shifted between releases.
    cli_candidates = [
        build_dir / "bin" / "Release" / "sd-cli.exe",
        build_dir / "bin" / "sd-cli.exe",
        build_dir / "Release" / "sd-cli.exe",
        build_dir / "examples" / "cli" / "Release" / "sd-cli.exe",
        build_dir / "bin" / "Release" / "sd.exe",
        build_dir / "bin" / "sd.exe",
        build_dir / "Release" / "sd.exe",
        build_dir / "examples" / "cli" / "Release" / "sd.exe",
    ]
    src_exe = next((c for c in cli_candidates if c.exists()), None)
    if src_exe is None:
        for name in ("sd-cli.exe", "sd.exe"):
            found = list(build_dir.rglob(name))
            if found:
                src_exe = found[0]
                break

    if src_exe is None:
        return False, f"No sd-cli.exe (or sd.exe) found under {build_dir} — check build output"

    shutil.copy2(str(src_exe), str(sd_dir / src_exe.name))
    print_status(f"{src_exe.name} built → {sd_dir}/")

    # sd-server.exe may be built alongside the CLI — copy it too if present,
    # even though playground.py's run_stable_diffusion() uses the CLI only.
    srv_src = src_exe.parent / "sd-server.exe"
    if srv_src.exists():
        shutil.copy2(str(srv_src), str(sd_dir / "sd-server.exe"))
        print_info(f"  sd-server.exe → {sd_dir}/ (not used by this project, kept for reference)")

    dll_count = 0
    for dll in src_exe.parent.glob("*.dll"):
        shutil.copy2(str(dll), str(sd_dir / dll.name))
        dll_count += 1

    print_status(f"stable-diffusion.cpp built → {sd_dir}/ ({src_exe.name} + {dll_count} dll, Vulkan+AVX2)")
    return True, ""

def _resolve_manager_models_dir(fallback: Path) -> Path:
    """Read manager_models_dir from configuration.json if the user has set
    one — this is intentionally separate from the shared models_dir, since
    the manager's AWQ folder routinely lives somewhere else entirely (its own
    drive, etc). Falls back to `fallback` (normally models_dir) if unset,
    so existing single-directory setups keep working unchanged."""
    json_path = BASE_DIR / "data" / "configuration.json"
    try:
        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            configured = data.get("manager_models_dir", "")
            if configured:
                return Path(configured)
    except Exception:
        pass
    return fallback


def generate_start_manager_sh(models_dir: Path, cpu: Dict, manager_models_dir: Optional[Path] = None) -> bool:
    """
    helper.sh is a permanent project file — NOT generated by the installer.
    This function no longer writes any model path or tokenizer info to
    constants.ini: launcher.py resolves the manager's model path and
    tokenizer source itself (via configure.py's runtime config) and passes
    them to helper.sh directly as env vars at launch time. helper.sh never
    globs a filename or reads constants.ini for model location at all —
    that keeps it future-proof against model changes, and keeps model
    resolution in exactly one place (Python) instead of two (Python +
    bash) that could silently drift out of sync, which is what happened
    before this was split out.

    manager_models_dir: the MANAGER's own model directory (may be a
    different drive entirely from the shared agents' models_dir). Not
    scanned or globbed here — models_dir/manager_models_dir is just a
    placeholder default path at install time ("./models"), which the user
    is never expected to have actually populated yet (they point it at a
    real folder later, from the Configuration page, once the model files
    are in place). configure.py's runtime discovery is the only place
    model presence is ever checked — duplicating that check here just to
    print an install-time status line isn't worth a stale/misleading
    result if the user installs first and copies the model in afterward.

    constants.ini [wsl] still keeps manager_port — a genuinely structural,
    rarely-changing setting, unlike the model path/tokenizer which vary
    per model choice and belong in configuration.json instead.
    """
    ini_path = BASE_DIR / "data" / "constants.ini"
    try:
        config = configparser.ConfigParser()
        if ini_path.exists():
            config.read(str(ini_path), encoding="utf-8")
        if not config.has_section("wsl"):
            config.add_section("wsl")
        config.set("wsl", "manager_port",    str(MANAGER_PORT))
        config.set("wsl", "manager_script",  MANAGER_SCRIPT)
        # Clean up now-obsolete keys from earlier installer versions, if present —
        # leaving them behind is harmless but misleading since nothing reads them.
        for stale_key in ("models_dir_wsl", "manager_tokenizer_source"):
            if config.has_option("wsl", stale_key):
                config.remove_option("wsl", stale_key)
        with open(ini_path, "w", encoding="utf-8") as f:
            config.write(f)
        print_status(f"Updated constants.ini [wsl] manager_port = {MANAGER_PORT}")
    except Exception as e:
        print_status(f"Failed to update constants.ini: {e}", False)
        return False

    # NOTE: deliberately no manager-model check here. The installer doesn't
    # scan for model files at all — models_dir/manager_models_dir are just
    # reference locations the user points at whenever they get around to
    # placing files there; requiring a model to be present at install time
    # would break the common case of installing first and copying multi-GB
    # model files in afterward. Model discovery (including which specific
    # Huihui-Qwen3.x-#B-abliterated-AWQ folder is in use) is a runtime
    # concern — launcher.py / configure.py's _validate_paths, run every
    # time the app actually starts — not the installer's job.

    # Verify helper.sh exists (permanent file — just warn if missing)
    script_path = BASE_DIR / MANAGER_SCRIPT
    if not script_path.exists():
        print_status(f"WARNING: {MANAGER_SCRIPT} not found at {script_path}", False)
        print_info("  Ensure helper.sh is present in the project root.")
        print_info("  It is a permanent file, not generated by the installer.")
    else:
        # chmod +x so WSL2 can execute it
        try:
            drive_l   = script_path.drive.rstrip(":").lower()
            path_rest = script_path.as_posix().split(":", 1)[1]
            wsl_path  = f"/mnt/{drive_l}{path_rest}"
            subprocess.run(
                ["wsl", "chmod", "+x", wsl_path],
                capture_output=True, timeout=5,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            print_status(f"helper.sh found and marked executable (WSL2 chmod +x)")
        except Exception as e:
            print_info(f"  chmod +x skipped: {e}")

    print_info(f"  Threads: 10 (fixed)  |  Context: 32768  |  Port: {MANAGER_PORT}")
    return True

def create_wslconfig(wsl_ram_gb: Optional[int] = None) -> bool:
    r"""
    Create or update %USERPROFILE%\.wslconfig with the WSL2 memory
    allocation the user picked at prompt_wsl_profile_size() (16/24/36 GB,
    sized for the 9B/14B/27B tiers of the Huihui-Qwen3.5/3.6-#B-abliterated-
    AWQ family respectively).

    wsl_ram_gb: pass the menu's choice through explicitly. Falls back to
    the old auto-detect-from-total-system-RAM heuristic (capped at 32 GB)
    only if called with no argument at all — kept for any caller that
    doesn't go through the menu (e.g. a future non-interactive/scripted
    install path), not because dynamic-per-model sizing is offered here.

    WHY THIS IS ONE FILE WITH ONE ACTIVE VALUE, NOT REAL PER-MODEL
    PROFILES: .wslconfig's [wsl2] memory= setting applies to the whole
    lightweight WSL2 VM, not per-distro — WSL2 has no concept of a
    per-distro memory profile. Genuinely separate profiles would mean a
    second (or third) full WSL2 Ubuntu install just to get a different
    number, at many GB of disk each — not worth it for one integer. This
    function just writes a different value into the SAME single
    .wslconfig depending on the menu choice, so "switching profiles" costs
    nothing beyond re-running that menu — see prompt_wsl_profile_size().

    Skips safely if the user already has a hand-crafted .wslconfig without our
    marker comment — prints recommended lines instead of clobbering their config.
    """
    wslconfig_path = Path.home() / ".wslconfig"
    import psutil as _psutil
    total_ram_gb = round(_psutil.virtual_memory().total / (1024 ** 3))

    if wsl_ram_gb is None:
        # Legacy auto-detect fallback — see docstring. Not used by either
        # of the two normal call paths (both now go through
        # prompt_wsl_profile_size() first), kept only so this function
        # still does something reasonable if ever called directly.
        if total_ram_gb >= 64:
            wsl_ram_gb = 32
        elif total_ram_gb >= 32:
            wsl_ram_gb = int(total_ram_gb * 0.5)
        else:
            wsl_ram_gb = max(8, int(total_ram_gb * 0.5))
    elif wsl_ram_gb >= total_ram_gb:
        # The user's explicit choice — honor it regardless, but flag if it
        # leaves little to nothing for Windows itself (WSL2's VM memory
        # limit is separate from, and in addition to, whatever Windows and
        # every other running program also needs).
        print_info(f"  [!] {wsl_ram_gb} GB WSL2 allocation is close to or above "
                   f"total system RAM ({total_ram_gb} GB) — Windows may run low.")

    model_tier = {16: "9B", 24: "9B + headroom", 36: "9B + max headroom / future"}.get(wsl_ram_gb, "selected")
    content = (
        "[wsl2]\n"
        "# Agentic Chatbot — WSL2 resource allocation\n"
        f"# System RAM: {total_ram_gb} GB  |  WSL2 allocation: {wsl_ram_gb} GB "
        f"(profile: {model_tier} manager model)\n"
        "# Chosen at install time via the WSL Profile Size menu — re-run\n"
        "# the installer's Check/Install or Reinstall option to change it.\n"
        f"memory={wsl_ram_gb}GB\n"
        "# Do NOT set 'processors' — the --threads flag in helper.sh\n"
        "# controls actual CPU usage; a processor cap hurts scheduling flexibility.\n"
        "swap=8GB\n"
    )

    if wslconfig_path.exists():
        try:
            existing = wslconfig_path.read_text(encoding="utf-8")
            if "Agentic Chatbot" in existing:
                # Ours — overwrite with updated values
                pass
            else:
                # User's own config — don't clobber it
                print_info(f"  Existing .wslconfig found at {wslconfig_path}")
                print_info(f"  Add these lines if not already present:")
                print_info(f"    [wsl2]")
                print_info(f"    memory={wsl_ram_gb}GB")
                print_info(f"    swap=8GB")
                print_status("Skipped .wslconfig - existing user config preserved")
                return True
        except Exception:
            pass

    try:
        with open(wslconfig_path, "w", encoding="utf-8") as f:
            f.write(content)
        print_status(f"Created .wslconfig: {wsl_ram_gb} GB RAM allocated to WSL2 ({model_tier} profile)")
        print_info(f"  Path: {wslconfig_path}")
        print_info("  Restart WSL2 to apply: wsl --shutdown")
    except Exception as e:
        print_status(f"Failed to create .wslconfig: {e}", False)
        print_info(f"  Create {wslconfig_path} manually with:")
        print_info(f"    [wsl2]")
        print_info(f"    memory={wsl_ram_gb}GB")
        print_info(f"    swap=8GB")
        return False

    return True

def prompt_wsl_profile_size() -> int:
    """
    Menu shown right before the WSL2 sudo-password prompt, letting the
    user pick how much RAM to give the whole WSL2 VM via .wslconfig —
    sized for whichever manager model tier (9B/14B/27B, from the
    Huihui-Qwen3.5/3.6-#B-abliterated-AWQ family) they intend to run.
    See create_wslconfig()'s docstring for why this is one file with one
    active value, not real separate per-size profiles.

    Only shown if WSL2 is actually available — same guard as
    prompt_wsl_sudo_password(); skipping it if not just means .wslconfig
    won't be touched at all, matching how the rest of this file already
    treats "no WSL2" (manager server setup is skipped, not an error).

    Returns the chosen size in GB (16, 24, or 36).
    """
    wsl_ok, _ = check_wsl2()
    if not wsl_ok:
        return 32  # unused when WSL2 isn't available - create_wslconfig() is never called

    while True:
        display_header("Agent-Gradio-Gguf - WSL Profile Size")
        print("\n\n\n\n\n")
        print("    1.  Size of 16GB  - fits the 9B manager\n")
        print("    2.  Size of 24GB  - 9B with extra RAM headroom\n")
        print("    3.  Size of 36GB  - maximum headroom / future larger manager\n")
        print("\n\n\n\n\n")
        display_separator_thick()
        choice = input("Selection; Menu Options = 1-3, Exit to Menu = B:   ").strip()
        if choice.lower() == 'b':
            sys.exit(0)
        if choice == "1":
            return 16
        if choice == "2":
            return 24
        if choice == "3":
            return 36
        print("Invalid selection. Please try again.")
        time.sleep(1)

VLLM_PIN_VERSION = "v0.25.0"   # pinned tag — everything checked against this exact release

# Official source distribution from the v0.25.0 release — the installer always
# builds vLLM from source (VLLM_TARGET_DEVICE=cpu), so CPU-specific
# optimizations get baked in rather than using a generic prebuilt artifact.
# This is vLLM's own prepared sdist (from the release assets), not a raw
# `git clone` of the whole repo — a single ~36MB tarball download instead of
# cloning the project's entire git history, which is both far smaller and a
# single HTTP GET instead of the git protocol's own negotiation (the previous
# git-clone approach is what kept failing on a shaky connection).
VLLM_SDIST_URL = (
    "https://github.com/vllm-project/vllm/releases/download/v0.25.0/vllm-0.25.0.tar.gz"
)

def setup_wsl2_vllm(wsl_password: str = "") -> bool:
    """
    Set up vLLM inside a dedicated WSL2 venv for the manager server, always as
    a CPU-only SOURCE BUILD (the Download-vs-Compile choice was removed — the
    installer only compiles now). Building from source with
    VLLM_TARGET_DEVICE=cpu bakes in CPU-specific optimizations a generic
    prebuilt wheel can't target, and excludes the CUDA kernel toolchain.

    The manager is vLLM-only, throughout — llama.cpp/llama-server is used
    exclusively by the Windows-side Vulkan sub-agents (build_llama_agents),
    never inside WSL2.

    Why this excludes the default PyPI package ("pip install vllm" with no
    other flags): that plain package defaults to a CUDA target and pulls a
    large GPU kernel-authoring toolchain (Triton, CUTLASS-DSL, TileLang, a full
    CUDA toolkit) as hard dependencies — multiple GB this program will never
    use, on hardware whose display GPU (GTX 1060) must never be touched by
    manager inference. VLLM_TARGET_DEVICE=cpu during the build compiles only
    the CPU backend instead.

    This CPU (Zen 2, no AVX512) runs vLLM's CPU backend on the documented
    "avx2 (Limited features)" path; the build script detects that from host
    CPU flags automatically (no special flag needed).

    wsl_password: if the distro needs a sudo password (no NOPASSWD entry), pass
    it here to feed it via stdin — never via argv/command string, so it never
    shows up in a process list. Only used transiently for apt-get calls and
    then discarded; nothing about it is written to disk. If empty, falls back
    to non-interactive sudo -n (fails fast, no hang).

    Setup time: a real compile — dependency installs are a few minutes each,
    the actual C++ CPU-kernel build is the longest single step, well beyond 30
    minutes. Requires internet access throughout.
    """
    print_info("\n  Setting up vLLM in WSL2 (CPU backend, source build)...")
    print_info(f"  Pinned to {VLLM_PIN_VERSION}. Requires internet access; a real compile, not quick.")
    display_separator_thin()

    wsl_ok, wsl_msg = check_wsl2()
    if not wsl_ok:
        print_status(f"WSL2 not available: {wsl_msg}", False)
        return False

    _cflags = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0

    def _sudo_prefix() -> str:
        """Shared sudo-handling fragment used by every apt-get call below."""
        if wsl_password:
            return (
                "read -r WSLPW && "
                "export DEBIAN_FRONTEND=noninteractive && "
            )
        return "export DEBIAN_FRONTEND=noninteractive && "

    def _apt_install(packages: str, label: str) -> bool:
        """Run one apt-get install with the shared password/timeout/error
        handling — every system-dependency step below is one of these."""
        if wsl_password:
            cmd = (
                "read -r WSLPW && "
                "export DEBIAN_FRONTEND=noninteractive && "
                "echo \"$WSLPW\" | sudo -S -p '' apt-get update -qq && "
                f"echo \"$WSLPW\" | sudo -S -p '' apt-get install -y --no-install-recommends {packages}; "
                "unset WSLPW"
            )
        else:
            cmd = (
                "export DEBIAN_FRONTEND=noninteractive && "
                "sudo -n apt-get update -qq && "
                f"sudo -n apt-get install -y --no-install-recommends {packages}"
            )
        try:
            proc = subprocess.Popen(
                ["wsl", "-e", "bash", "-c", cmd],
                stdin=subprocess.PIPE if wsl_password else None,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                creationflags=_cflags,
            )
            if wsl_password:
                try:
                    proc.stdin.write((wsl_password + "\n").encode("utf-8"))
                    proc.stdin.flush()
                    proc.stdin.close()
                except Exception:
                    pass
            lines = []
            import threading

            def _drain(p, buf):
                for raw in iter(p.stdout.readline, b""):
                    line = raw.decode("utf-8", errors="replace").rstrip()
                    buf.append(line)
                    if any(kw in line for kw in ("Get:", "Unpacking", "Setting up", "Processing", "Err:", "E:", "WARNING")):
                        print_info(f"    {line[:120]}")

            t = threading.Thread(target=_drain, args=(proc, lines), daemon=True)
            t.start()
            t.join(timeout=600)
            if t.is_alive():
                proc.kill()
                print_status(f"{label} install timed out - 10 min", False)
                return False
            proc.wait()
            if proc.returncode != 0:
                print_status(f"{label} install failed", False)
                for l in lines[-10:]:
                    print_info(f"    {l[:120]}")
                print_info("")
                if wsl_password:
                    print_info("  The password entered was rejected, or another error occurred above.")
                else:
                    print_info("  Your WSL2 sudo requires a password, which this installer cannot")
                    print_info("  supply non-interactively. Enable passwordless sudo (one-time):")
                    print_info("    Run in WSL2:  sudo visudo")
                    print_info("    Add at the end:  %sudo ALL=(ALL) NOPASSWD:ALL")
                return False
            print_status(f"{label} installed")
            return True
        except subprocess.TimeoutExpired:
            print_status(f"{label} install timed out", False)
            return False
        except Exception as e:
            print_status(f"{label} install error: {e}", False)
            return False

    def _run(cmd: str, timeout: int, label: str, stream: bool = False) -> Tuple[bool, str]:
        """Run one WSL2 shell command. stream=True prints matching output
        lines live instead of only reporting pass/fail — used for the long
        steps (dependency installs, the download, the build) so a
        multi-minute step doesn't look hung."""
        try:
            if not stream:
                result = subprocess.run(
                    ["wsl", "-e", "bash", "-c", cmd],
                    capture_output=True, text=True, timeout=timeout, creationflags=_cflags,
                )
                if result.returncode != 0:
                    print_status(f"{label} failed", False)
                    print_info(f"  {result.stderr[-500:] or result.stdout[-500:]}")
                    return False, result.stdout + result.stderr
                return True, result.stdout
            else:
                proc = subprocess.Popen(
                    ["wsl", "-e", "bash", "-c", cmd],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    creationflags=_cflags,
                )
                lines = []
                import threading

                def _drain(p, buf):
                    for raw in iter(p.stdout.readline, b""):
                        line = raw.decode("utf-8", errors="replace").rstrip()
                        buf.append(line)
                        if any(kw in line for kw in (
                            "Collecting", "Installing", "Successfully", "ERROR", "error:",
                            "Building wheel", "running build", "CMake Error",
                            "Downloading", "%",
                        )):
                            print_info(f"    {line[:130]}")

                t = threading.Thread(target=_drain, args=(proc, lines), daemon=True)
                t.start()
                t.join(timeout=timeout)
                if t.is_alive():
                    proc.kill()
                    print_status(f"{label} timed out - {timeout // 60} min", False)
                    return False, "\n".join(lines)
                proc.wait()
                if proc.returncode != 0:
                    print_status(f"{label} failed", False)
                    for l in lines[-20:]:
                        print_info(f"    {l[:130]}")
                    return False, "\n".join(lines)
                print_status(f"{label} done")
                return True, "\n".join(lines)
        except subprocess.TimeoutExpired:
            print_status(f"{label} timed out", False)
            return False, ""
        except Exception as e:
            print_status(f"{label} error: {e}", False)
            return False, str(e)

    # ── Fast path: already installed and matches the pinned version ─────────
    check_cmd = (
        "~/vllm-env/bin/python -c \"import vllm; print(vllm.__version__)\" 2>/dev/null"
    )
    try:
        check = subprocess.run(
            ["wsl", "-e", "bash", "-c", check_cmd],
            capture_output=True, text=True, timeout=15, creationflags=_cflags,
        )
        installed_version = check.stdout.strip()
        pin_bare = VLLM_PIN_VERSION.lstrip("v")
        # Match on the pinned version number only — NOT a literal "cpu"
        # substring in vllm.__version__. That substring isn't guaranteed
        # present in either route's __version__ string, so requiring it
        # here would be the same false-negative bug that forced a full
        # reinstall/rebuild on every single run before this was fixed.
        # ~/vllm-env is a venv dedicated exclusively to this project's
        # vLLM install — nothing else ever installs a CUDA vllm into it —
        # so matching the version number alone is sufficient.
        if check.returncode == 0 and installed_version.startswith(pin_bare):
            print_status(f"vLLM already installed and matches {VLLM_PIN_VERSION}: {installed_version}")
            return True
        elif installed_version:
            action = "rebuilding"
            print_info(f"  Found vLLM {installed_version}, but it doesn't match "
                       f"{VLLM_PIN_VERSION} — {action}.")
    except Exception:
        pass

    # ── Step 1: verify Python 3.12 specifically inside WSL2 ─────────────────
    total_steps = 8
    print_info(f"  Step 1/{total_steps}: Checking Python 3.12 in WSL2...")
    py_check_cmd = "python3 --version 2>&1"
    py_version_ok = False
    py_version_str = ""
    try:
        py_result = subprocess.run(
            ["wsl", "-e", "bash", "-c", py_check_cmd],
            capture_output=True, text=True, timeout=15, creationflags=_cflags,
        )
        py_version_str = py_result.stdout.strip() or py_result.stderr.strip()
        py_version_ok = "Python 3.12" in py_version_str
    except Exception as e:
        print_info(f"  Python version check failed ({e})")

    if py_version_ok:
        print_status(f"Python found: {py_version_str}")
    else:
        print_info(f"  Found: {py_version_str or 'no python3 on PATH'} — need 3.12.x")
        py_pkgs = "python3.12 python3.12-venv python3.12-dev"
        if not _apt_install(py_pkgs, "python3.12"):
            print_info("")
            print_info("  If this distro's default repos don't carry python3.12")
            print_info("  (e.g. Ubuntu 22.04 ships 3.10), add the deadsnakes PPA first:")
            print_info("    sudo add-apt-repository ppa:deadsnakes/ppa")
            print_info("    sudo apt-get update")
            print_info(f"    sudo apt-get install -y {py_pkgs}")
            return False
        py_result = subprocess.run(
            ["wsl", "-e", "bash", "-c", py_check_cmd],
            capture_output=True, text=True, timeout=15, creationflags=_cflags,
        )
        py_version_str = py_result.stdout.strip() or py_result.stderr.strip()
        if "Python 3.12" not in py_version_str:
            print_status(f"python3.12 installed but 'python3' still resolves to: {py_version_str}", False)
            print_info("  Check update-alternatives --list python3, or invoke python3.12 explicitly")
            return False

    # ── Step 2: remaining system dependencies ────────────────────────────────
    # numactl/libnuma-dev: vLLM's CPU backend NUMA-aware thread binding at
    #   RUNTIME (VLLM_CPU_OMP_THREADS_BIND in helper.sh).
    # libtcmalloc-minimal4: vLLM's CPU backend strongly recommends preloading
    #   TCMalloc for allocation throughput / cache locality — helper.sh finds
    #   libtcmalloc_minimal.so.4 and prepends it to LD_PRELOAD at launch.
    # python3.12-dev (CMake's FindPython), curl (rustup fetch, step 3),
    #   build-essential/cmake/git (the C++ toolchain): needed for the source build.
    print_info(f"  Step 2/{total_steps}: Checking other system dependencies...")
    sys_pkgs = "python3.12-venv python3.12-dev python3-pip build-essential cmake git numactl libnuma-dev libtcmalloc-minimal4 curl"
    check_cmd2 = f"dpkg -s {sys_pkgs} >/dev/null 2>&1 && echo ALL_PRESENT || echo MISSING"
    deps_present = False
    try:
        check_result = subprocess.run(
            ["wsl", "-e", "bash", "-c", check_cmd2],
            capture_output=True, text=True, timeout=30, creationflags=_cflags,
        )
        deps_present = "ALL_PRESENT" in check_result.stdout
    except Exception as e:
        print_info(f"  Dependency check failed ({e}), will attempt install")

    if deps_present:
        print_status("System dependencies already installed")
    else:
        print_info("  Installing missing system dependencies (apt)...")
        if not _apt_install(sys_pkgs, "System dependencies"):
            print_info("")
            print_info("  Install these yourself instead, then re-run:")
            print_info("    sudo apt-get update && sudo apt-get install -y \\")
            print_info(f"      --no-install-recommends {sys_pkgs}")
            return False

    # ── Step 3: Rust toolchain ───────────────────────────────────────────────
    # vLLM's setup.py imports setuptools_rust unconditionally. Whether a given
    # build actually invokes cargo depends on which extensions get compiled —
    # safer to always have a real toolchain present than find out partway
    # through a 20+ minute build that it's missing.
    print_info(f"  Step 3/{total_steps}: Checking Rust toolchain...")
    rust_check = "test -x ~/.cargo/bin/rustc && ~/.cargo/bin/rustc --version"
    try:
        rust_result = subprocess.run(
            ["wsl", "-e", "bash", "-c", rust_check],
            capture_output=True, text=True, timeout=10, creationflags=_cflags,
        )
        rust_present = rust_result.returncode == 0
    except Exception:
        rust_present = False

    if rust_present:
        print_status(f"Rust already installed: {rust_result.stdout.strip()}")
    else:
        print_info("  Installing Rust via rustup (a few minutes)...")
        rustup_cmd = (
            "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs "
            "| sh -s -- -y --default-toolchain stable -q"
        )
        ok, _ = _run(rustup_cmd, timeout=300, label="Rust install")
        if not ok:
            return False

    # ── Step 4: create the dedicated vLLM venv ──────────────────────────────
    # python3.12 explicitly, not the ambiguous python3 — avoids silently
    # building/installing against the wrong interpreter on a distro with
    # more than one Python version installed side by side.
    venv_step = 4
    print_info(f"  Step {venv_step}/{total_steps}: Creating vLLM venv at ~/vllm-env...")
    venv_cmd = (
        "if [ ! -x ~/vllm-env/bin/python ]; then "
        "  python3.12 -m venv ~/vllm-env; "
        "fi && "
        "~/vllm-env/bin/pip install --upgrade pip -q"
    )
    ok, _ = _run(venv_cmd, timeout=120, label="venv creation")
    if not ok:
        return False
    print_status("vLLM venv ready (~/vllm-env)")

    # ── Step 5 (compile route): fetch the pinned source distribution ────
    # A single tarball download (VLLM_SDIST_URL, vLLM's own prepared
    # sdist from the release assets), NOT `git clone` of the whole repo
    # — far smaller, one HTTP GET instead of the git protocol's own
    # negotiation. The marker file skips re-downloading on a re-run once
    # this exact pinned version is already extracted.
    print_info(f"  Step 5/{total_steps}: Fetching vLLM source (sdist), pinned to {VLLM_PIN_VERSION}...")
    pin_bare = VLLM_PIN_VERSION.lstrip("v")
    fetch_cmd = (
        "mkdir -p ~/vllm-src && cd ~/vllm-src && "
        f"if [ ! -f .fetched-{pin_bare} ]; then "
        f"  rm -rf vllm-{pin_bare} vllm-{pin_bare}.tar.gz && "
        f"  curl -fL -o vllm-{pin_bare}.tar.gz \"{VLLM_SDIST_URL}\" && "
        f"  tar xzf vllm-{pin_bare}.tar.gz && "
        f"  touch .fetched-{pin_bare}; "
        "fi"
    )
    ok, _ = _run(fetch_cmd, timeout=900, label="vLLM source fetch", stream=True)
    if not ok:
        return False

    # ── Step 6: install Python-side runtime + build dependencies ────────
    # requirements/cpu.txt: what vLLM needs at runtime on the CPU backend.
    # requirements/build/cpu.txt: what's needed to build it — cmake, ninja,
    #   setuptools-rust, and a matching torch build. This is what pulls
    #   setuptools_rust in automatically; a standalone `pip install
    #   setuptools_rust` beforehand isn't needed if this runs first.
    print_info(f"  Step 6/{total_steps}: Installing Python dependencies (runtime + build)...")
    deps_cmd = (
        f"cd ~/vllm-src/vllm-{pin_bare} && "
        "~/vllm-env/bin/pip install -q -r requirements/cpu.txt "
        "--extra-index-url https://download.pytorch.org/whl/cpu && "
        "~/vllm-env/bin/pip install -q -r requirements/build/cpu.txt "
        "--extra-index-url https://download.pytorch.org/whl/cpu"
    )
    ok, _ = _run(deps_cmd, timeout=1800, label="Python dependency install", stream=True)
    if not ok:
        return False

    # ── Step 7: the actual build ──────────────────────────────────────────
    # VLLM_TARGET_DEVICE=cpu is what excludes the CUDA kernel toolchain —
    # the build script inspects host CPU flags itself (no AVX512 here,
    # confirmed via lscpu) and compiles the AVX2 path automatically.
    print_info(f"  Step 7/{total_steps}: Building vLLM from source (CPU) — this is the long step...")
    print_info("            Real C++ compile. Can run well beyond 30 minutes.")
    build_cmd = (
        f"cd ~/vllm-src/vllm-{pin_bare} && "
        "source $HOME/.cargo/env 2>/dev/null; "
        "VLLM_TARGET_DEVICE=cpu ~/vllm-env/bin/pip install -q . --no-build-isolation"
    )
    ok, _ = _run(build_cmd, timeout=5400, label="vLLM build", stream=True)
    if not ok:
        print_info("")
        print_info("  If this failed with a missing-module or missing-header error,")
        print_info("  it's very likely a single missing system/python package — check")
        print_info("  the error above for the specific name, install it, and re-run.")
        print_info("  This exact build has needed one-at-a-time fixes before")
        print_info("  (python3.12-dev, setuptools_rust, a Rust toolchain) — that's")
        print_info("  normal for a source build, not a sign of a deeper problem.")
        return False

    # ── Verify import and version ────────────────────────────────────────────
    print_info(f"  Step {total_steps}/{total_steps}: Verifying vLLM import and version...")
    try:
        verify = subprocess.run(
            ["wsl", "-e", "bash", "-c",
             "~/vllm-env/bin/python -c \"import vllm; print(vllm.__version__)\""],
            capture_output=True, text=True, timeout=60, creationflags=_cflags,
        )
        version = verify.stdout.strip()
        if verify.returncode != 0 or not version:
            print_status("vLLM installed but failed to import", False)
            print_info(f"  {verify.stderr[-400:]}")
            return False
        pin_bare = VLLM_PIN_VERSION.lstrip("v")
        if not version.startswith(pin_bare):
            print_info(f"  WARNING: installed version is {version}, expected to start with {pin_bare}")
        print_status(f"vLLM ready — version {version}")
    except Exception as e:
        print_status(f"vLLM verification failed: {e}", False)
        return False

    return True

def verify_installation() -> Tuple[bool, List[str]]:
    print_status("Verifying...")
    missing = []
    for path, name in [
        (VENV_DIR / "Scripts" / "python.exe", "Python venv"),
        (BASE_DIR / "data" / "constants.ini", "Config"),
        (BASE_DIR / "scripts" / "__init__.py", "Scripts package"),
        (BASE_DIR / "data" / "configuration.json", "Persistent config"),
        (BASE_DIR / "data" / "profiles.json", "Profiles config"),
        (BASE_DIR / LLAMA_AGENTS_DIR / "llama-server.exe", "llama-server.exe (agents)"),
        (BASE_DIR / MANAGER_SCRIPT, "helper.sh"),
    ]:
        exists = path.exists()
        print_info(f"  {'Found' if exists else 'Missing'}: {name}")
        if not exists:
            missing.append(name)

    # Image generation
    sd_found = any(
        (BASE_DIR / SD_AGENTS_DIR / name).exists() for name in ("sd-cli.exe", "sd.exe")
    )
    print_info(f"  {'Found' if sd_found else 'Missing'}: sd-cli.exe - image generation")
    if not sd_found:
        missing.append("sd-cli.exe")

    wsl_ok, _ = check_wsl2()
    if wsl_ok:
        _cflags = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
        wsl_server_ok = False
        try:
            verify_cmd = "~/vllm-env/bin/python -c \"import vllm\" 2>/dev/null && echo OK"
            verify = subprocess.run(
                ["wsl", "-e", "bash", "-c", verify_cmd],
                capture_output=True, text=True, timeout=10, creationflags=_cflags,
            )
            wsl_server_ok = "OK" in verify.stdout
        except Exception:
            pass
        print_info(f"  {'Found' if wsl_server_ok else 'Missing'}: WSL2 vLLM")
        if not wsl_server_ok:
            missing.append("WSL2 vLLM")
    else:
        print_info("  Missing: WSL2 vLLM - WSL2 not available")
        missing.append("WSL2 vLLM")

    return len(missing) == 0, missing

# ============================================================================
# MENUS
# ============================================================================
# ============================================================================
# MAIN INSTALLATION
# ============================================================================
def run_installation(cpu, vulkan, vulkan_ver, msvc_ver, wsl_password="", wsl_ram_gb=32):
    start = time.time()
    status_parts = [
        "Installing with Compile options",
        f"on Windows 10 with Python v{PYTHON_VERSION}",
        f"and Vulkan v{vulkan_ver}" if vulkan else "without Vulkan",
    ]
    if msvc_ver != "Not Detected":
        status_parts.append(f"and {msvc_ver}")
    print(f"\n  {', '.join(status_parts)}.")

    print("\nSTEP 1: Creating Directories")
    create_directories()
    
    print("\nSTEP 2: Creating Configuration")
    if not create_system_ini(vulkan, cpu):
        display_critical_failure("Failed to create constants.ini", "Configuration file creation failed")
        return True
    if not create_configuration_json(vulkan, cpu):
        display_critical_failure("Failed to create configuration.json", "Settings file creation failed")
        return True
    if not create_profiles_json():
        display_critical_failure("Failed to create profiles.json", "Profiles file creation failed")
        return True

    print("\nSTEP 3: Creating Virtual Environment")
    if not create_venv():
        display_critical_failure("Failed to create virtual environment", "venv creation failed")
        return True
        
    print("\nSTEP 4: Installing Python Dependencies")
    success, error = install_base_deps()
    if not success:
        display_critical_failure("Failed to install Python dependencies", error)
        return True
        
    print("\nSTEP 5: Creating Package Files")
    create_init_files()

    models_dir = BASE_DIR / "models"

    print("\nSTEP 6: WSL2 Check & Manager Server Setup")
    wsl_ok, wsl_msg = check_wsl2()
    if wsl_ok:
        print_status("WSL2 available")
        # ── Set up vLLM (manager is vLLM-only, throughout) — always a source
        # build (Compile), for CPU-specific optimizations a generic prebuilt
        # wheel can't target. ────────────────────────────────────────────────
        if not setup_wsl2_vllm(wsl_password):
            display_critical_failure("Failed to set up WSL2 vLLM", "Setup process failed or timed out.")
            return True
        print_status("WSL2 vLLM ready")

        # ── Update constants.ini [wsl] with models path for helper.sh ──────────
        manager_models_dir = _resolve_manager_models_dir(models_dir)
        if manager_models_dir != models_dir:
            print_info(f"  Manager model directory: {manager_models_dir} (separate from agents' models_dir)")
        if generate_start_manager_sh(models_dir, cpu, manager_models_dir):
            print_status("helper.sh verified + constants.ini [wsl] updated")
        else:
            print_info("  [!] constants.ini [wsl] update failed — check helper.sh models path manually")

        # ── .wslconfig — allocate RAM per the chosen WSL profile size ────────
        create_wslconfig(wsl_ram_gb)

        # ── Port proxy ────────────────────────────────────────────────────────
        proxy_ok, proxy_msg = setup_portproxy()
        if proxy_ok:
            print_status(f"Port proxy configured: {proxy_msg}")
        else:
            print_info(f"  [!] Port proxy: {proxy_msg}")
            print_info("  [!] Run as Administrator to configure port proxy.")
            print_info("  [!] Or configure manually: netsh interface portproxy add v4tov4 ...")
    else:
        print_info(f"  [!] WSL2: {wsl_msg}")
        print_info("  [!] Manager server requires WSL2. Install it, then re-run installer.")
        print_info("  [!] Agent models and UI will still work without WSL2.")

    print("\nSTEP 7: Installing llama.cpp")
    success, error = install_llama_compile()
    if not success:
        display_critical_failure("Failed to compile llama-cpp-python", error)
        return True

    print("\nSTEP 8a: Building llama-server - Windows Vulkan Agent GPU")
    success, error = build_llama_agents(cpu)
    if not success:
        display_critical_failure("Failed to build llama-server.exe - agents", error)
        return True

    print("\nSTEP 8b: Building sd-cli.exe - Windows Vulkan Image Generation")
    success, error = build_sd_agent(cpu)
    if not success:
        display_critical_failure("Failed to build sd-cli.exe", error)
        return True
    else:
        print_status("sd-cli.exe ready")

    print("\nSTEP 9: Setting Up Media Directory")
    create_sample_visualizes()

    print("\nSTEP 10: Verification")
    success, missing = verify_installation()

    elapsed = time.time() - start

    if success:
        print("\n")
        display_separator_thick()
        print("    INSTALLATION COMPLETE!")
        display_separator_thick()
        print(f"\n  Time: {format_time(elapsed)}")
        print(f"  Backend (agents): Vulkan (compile)")
        print(f"  Backend (manager): WSL2 CPU (vLLM)")
        print(f"  Vulkan: {'Yes' if vulkan else 'No'}")
        print()
        display_separator_thick()
        return False
    else:
        display_critical_failure("Installation completed with missing components", f"Missing: {', '.join(missing)}")
        return True

def show_installer_menu() -> str:
    """Display the Reinstall / Refresh sub-menu with detection table above."""
    while True:
        display_header("Agent-Gradio-Gguf - Reinstall / Refresh")
        # Run detection table (re‑run for display; critical check already passed in install())
        critical_ok, _ = print_detection_table()
        if not critical_ok:
            # Should not happen because install() already checked, but safety
            print("\nPress ENTER to exit...")
            input()
            sys.exit(1)
            
        # Thin separator after detections
        print()
        print()
        display_separator_thin()
        print()
        print()
        
        # Menu options (no extra blank lines)
        print("    1. Clean Reinstall All     (delete venv + full reinstall + reset settings)\n")
        print("    2. Reinstall Dependencies  (force reinstall packages, keep venv & settings)\n")
        print("    3. Refresh Packages Only   (check + install missing, keep venv & settings)\n")
        print("    4. Re-Create Jsons Only    (configuration.json + profiles.json → defaults)\n")
        
        print()
        print()
        display_separator_thick()
        choice = input("Selection; Menu Option = 1-4, Abandon = A:   ").strip()
        if choice.lower() == 'a':
            sys.exit(0)
        if choice in ["1", "2", "3", "4"]:
            return choice
        print("Invalid selection. Please try again.")
        time.sleep(1)

def prompt_wsl_sudo_password() -> str:
    """
    Last menu before installation starts (right after
    prompt_wsl_profile_size()): ask for the WSL2 Ubuntu sudo password, so
    STEP 6's apt-get system-dependency install doesn't fail if
    passwordless sudo isn't configured.

    This password is used ONLY transiently, piped to `sudo -S` over stdin
    for that one apt-get call (never written to disk, never placed in a
    command string/argv). Nothing else in this program needs it — the
    manager restart and helper.sh never call sudo — so it is deliberately
    NOT saved to constants.ini or configuration.json.

    Returns "" if skipped (e.g. no WSL2, or the user already has
    passwordless sudo and just presses Enter) — setup_wsl2_vllm()
    falls back to the old fail-fast sudo -n behavior in that case.
    """
    wsl_ok, _ = check_wsl2()
    if not wsl_ok:
        return ""

    display_header("Agent-Gradio-Gguf - WSL2 Sudo Password")
    print("  STEP 6 installs system dependencies inside WSL2 via 'sudo apt-get'.")
    print("  If your WSL2 distro already has passwordless sudo configured,")
    print("  or already has these packages installed, just press Enter to skip.")
    print()
    print("  This password is used once, right now, for that apt-get call only.")
    print("  It is never saved to disk, and nothing else in this program needs")
    print("  it afterward (helper.sh and the manager restart never call sudo).")
    print()
    print("  NOTE: the password IS shown as you type it, on purpose - so you can")
    print("  confirm it is correct and tell a wrong password apart from other")
    print("  apt-get failures. This console is local to you.")
    print()
    try:
        # Deliberately input(), not getpass: the user asked to SEE the password
        # as it is entered, to diagnose auth failures. Visible echo is the point.
        pw = input("  WSL2 Ubuntu sudo password (visible; Enter to skip): ")
    except Exception:
        pw = ""
    return pw.strip()


def install():
    parser = argparse.ArgumentParser(description="Agentic Chatbot Installer", add_help=False)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--check", action="store_true")
    args, _ = parser.parse_known_args()
    
    # ── Pre‑flight detection (always run, critical failures abort) ──
    critical_ok, _ = print_detection_table()
    if not critical_ok:
        print("\nPress ENTER to exit...")
        input()
        sys.exit(1)

    # ── Determine menu choice from args or user input ──
    if args.clean:
        menu_choice = "1"
    elif args.refresh:
        menu_choice = "2"
    elif args.check:
        menu_choice = "3"
    else:
        menu_choice = show_installer_menu()   # only shows options 1-4

    if menu_choice == "3":
        display_header("Agentic Chatbot: Installer (Refresh Packages Only)")
        refresh_packages_only()
        return

    if menu_choice == "4":
        display_header("Agentic Chatbot: Installer (Re-Create Jsons Only)")
        recreate_jsons_only()
        return

    # Save the clean-install choice to a flag now, but DON'T purge yet — the
    # purge only runs after the user has been through the remaining menus and
    # actually entered their WSL password (below), so cancelling or mistyping
    # at those prompts can't wipe an existing install first.
    is_clean_install = (menu_choice == "1")
    if is_clean_install:
        display_header("Agent-Gradio-Gguf - Installation")
        print("  Clean install selected.")
        print("  Existing install will be purged AFTER the WSL password step.")
    else:
        print("  Reinstalling dependencies (keeping venv & settings)...")

    cpu = detect_cpu_features()
    vulkan, vulkan_ver = check_vulkan()
    msvc_ver = detect_msvc()
    print(f"  Vulkan: {'Yes' if vulkan else 'No'} ({vulkan_ver})")
    print(f"  MSVC: {msvc_ver}")

    # The installer always compiles (source builds) — the Download-vs-Compile
    # menu and its plumbing were removed. Nothing to choose here.
    wsl_ram_gb = prompt_wsl_profile_size()
    wsl_password = prompt_wsl_sudo_password()

    # Deferred clean-install purge — runs ONLY once all menus are done and the
    # WSL password has been entered, per the requirement that the purge happen
    # at this point rather than immediately after the first menu.
    if is_clean_install:
        display_header("Agent-Gradio-Gguf - Clean Install Purge")
        print("  Purging existing installation for clean install...")
        targets = [VENV_DIR, BASE_DIR / "data"]
        for target in targets:
            if target.exists():
                print(f"  Removing {target.name}/...")
                shutil.rmtree(target, ignore_errors=True)
        # Belt-and-braces: remove configuration.json/profiles.json if they somehow survived data/ purge
        for stray_name in ("configuration.json", "profiles.json"):
            stray_json = BASE_DIR / "data" / stray_name
            if stray_json.exists():
                stray_json.unlink()
        print_status("Purge complete - ready for clean install")
        time.sleep(1)

    while True:
        display_main_header()
        should_retry = run_installation(cpu, vulkan, vulkan_ver, msvc_ver, wsl_password, wsl_ram_gb)
        if not should_retry:
            break

if __name__ == "__main__":
    try:
        install()
    except KeyboardInterrupt:
        print("\n\nCancelled. Run again to resume.")
        sys.exit(0)
    except Exception as e:
        display_critical_failure("Fatal installer error", f"{str(e)}\n\n{traceback.format_exc()}")
        sys.exit(1)