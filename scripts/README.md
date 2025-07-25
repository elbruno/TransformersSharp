# TransformersSharp Scripts Documentation

This directory contains utility scripts for managing Python environments, CUDA installations, and troubleshooting TransformersSharp setups across different platforms.

## 📋 Quick Reference

| Script | Platform | Purpose | Usage |
|--------|----------|---------|-------|
| `check-gpu-cuda-windows.ps1` | Windows | Detect GPU/CUDA capabilities | `.\check-gpu-cuda-windows.ps1` |
| `fix-pytorch-compatibility-windows.ps1` | Windows | Fix PyTorch compatibility issues | `.\fix-pytorch-compatibility-windows.ps1` |
| `install-cuda-windows.ps1` | Windows | Install CUDA-enabled PyTorch | `.\install-cuda-windows.ps1` |
| `install-cuda-windows.bat` | Windows | Install CUDA PyTorch (batch) | `install-cuda-windows.bat` |
| `install-cuda-linux.sh` | Linux | Install CUDA-enabled PyTorch | `./install-cuda-linux.sh` |
| `load-venv-windows.ps1` | Windows | Load/create virtual environment | `.\load-venv-windows.ps1` |
| `load-venv-windows.bat` | Windows | Load virtual environment (batch) | `load-venv-windows.bat` |
| `load-venv-linux.sh` | Linux | Load/create virtual environment | `./load-venv-linux.sh` |
| `cleanup-venv-windows.ps1` | Windows | Clean virtual environment | `.\cleanup-venv-windows.ps1` |
| `cleanup-venv-macos.sh` | macOS | Clean virtual environment | `./cleanup-venv-macos.sh` |
| `cleanup-venv-linux.sh` | Linux | Clean virtual environment | `./cleanup-venv-linux.sh` |
| `test_cuda.py` | All | Test CUDA installation | `python test_cuda.py` |

## 🚀 Getting Started

### Windows Users

1. **Check your system capabilities:**
   ```powershell
   .\scripts\check-gpu-cuda-windows.ps1
   ```

2. **If you have NVIDIA GPU and want GPU acceleration:**
   ```powershell
   .\scripts\fix-pytorch-compatibility-windows.ps1
   ```

3. **If you encounter compatibility issues:**
   ```powershell
   .\scripts\fix-pytorch-compatibility-windows.ps1
   ```

### Linux Users

1. **Load virtual environment:**
   ```bash
   ./scripts/load-venv-linux.sh
   ```

2. **Install CUDA support (if you have NVIDIA GPU):**
   ```bash
   ./scripts/install-cuda-linux.sh
   ```

### Testing Your Installation

Test CUDA availability:
```bash
python scripts/test_cuda.py
```

## 📚 Detailed Script Documentation

### 🔍 Diagnostic Scripts

#### `check-gpu-cuda-windows.ps1`
**Purpose:** Comprehensive system analysis for GPU and CUDA capabilities.

**What it does:**
- Detects NVIDIA GPUs using WMI
- Checks NVIDIA driver installation (`nvidia-smi`)
- Analyzes current PyTorch CUDA support
- Provides specific recommendations based on findings

**Output sections:**
1. **GPU Detection:** Lists installed NVIDIA hardware
2. **Driver Status:** Verifies nvidia-smi availability
3. **PyTorch Analysis:** Checks current PyTorch CUDA support
4. **Recommendations:** Tailored next steps based on your setup

**Example output:**
```
✅ NVIDIA GPU detected:
   - NVIDIA GeForce RTX 3060
   - Driver Version: 31.0.15.2849

✅ nvidia-smi is available
   NVIDIA drivers are properly installed

✅ PyTorch check results:
   PyTorch version: 2.1.0+cu121
   CUDA available: True
   CUDA devices: 1
```

#### `test_cuda.py`
**Purpose:** Python-based CUDA functionality testing.

**What it does:**
- Reports PyTorch version and Python version
- Tests CUDA availability and device count
- Performs actual tensor operations on GPU/CPU
- Provides diagnostic information for troubleshooting

**Use cases:**
- Verify installation after setup
- Debug CUDA-related issues
- Performance testing preparation

---

### 🔧 Installation Scripts

#### `fix-pytorch-compatibility-windows.ps1`
**Purpose:** Comprehensive solution for PyTorch/xFormers compatibility issues.

**Problem it solves:**
- xFormers compatibility warnings
- PyTorch version mismatches
- CPU/CUDA PyTorch conflicts
- Python version compatibility issues

**What it does:**
1. **Analysis Phase:**
   - Detects current Python version
   - Analyzes existing PyTorch installation
   - Identifies xFormers conflicts

2. **Cleanup Phase:**
   - Removes incompatible xFormers
   - Uninstalls conflicting PyTorch versions

3. **Installation Phase:**
   - Python 3.12: Installs CPU-only PyTorch (maximum compatibility)
   - Python 3.10/3.11: Attempts CUDA PyTorch with fallback
   - Other versions: Defaults to CPU-only

4. **Verification Phase:**
   - Tests new installation
   - Verifies package imports
   - Reports final configuration

**Supported Python versions:**
- **Python 3.12:** CPU-only (recommended for stability)
- **Python 3.10/3.11:** CUDA-enabled with xFormers support
- **Other versions:** CPU-only fallback

#### `install-cuda-windows.ps1` / `install-cuda-windows.bat`
**Purpose:** Install CUDA-enabled PyTorch in existing environments.

**Prerequisites:**
- Existing virtual environment
- NVIDIA GPU with drivers
- Compatible CUDA version

**Installation strategy:**
```powershell
# Attempts CUDA installation first
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Falls back to CPU if CUDA fails
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### `install-cuda-linux.sh`
**Purpose:** Linux-specific CUDA PyTorch installation.

**Features:**
- Detects Linux distribution
- Checks for NVIDIA drivers
- Installs appropriate CUDA toolkit if needed
- Configures PyTorch with CUDA support

---

### 🌐 Environment Management Scripts

#### `load-venv-windows.ps1` / `load-venv-windows.bat`
**Purpose:** Create or activate TransformersSharp virtual environment.

**Environment path priority:**
1. `$env:TRANSFORMERS_SHARP_VENV_PATH` (user-defined)
2. `%LOCALAPPDATA%\TransformersSharp\venv` (default)

**Auto-creation:** Creates new environment if none exists.

**Features:**
- Respects user-defined paths
- Automatic environment creation
- Cross-session path persistence

#### `load-venv-linux.sh`
**Purpose:** Linux/Unix virtual environment management.

**Features:**
- Bash-compatible activation
- Automatic path detection
- Environment validation

---

### 🧹 Cleanup Scripts

#### `cleanup-venv-windows.ps1`
**Purpose:** Remove and recreate virtual environment on Windows.

**Use cases:**
- Corrupted environment recovery
- Fresh installation after major updates
- Disk space cleanup

**Safety features:**
- Confirmation prompts
- Backup recommendations
- Graceful error handling

#### `cleanup-venv-linux.sh` / `cleanup-venv-macos.sh`
**Purpose:** Unix-based environment cleanup.

**Platform-specific optimizations:**
- Linux: Uses `rm -rf` with safety checks
- macOS: Handles case-sensitive filesystem considerations

---

## 🔧 Troubleshooting Guide

### Common Issues

#### ❌ "CUDA requested but not available"
**Cause:** PyTorch was installed without CUDA support.

**Solution:**
```powershell
# Windows
.\scripts\fix-pytorch-compatibility-windows.ps1

# Linux
./scripts/install-cuda-linux.sh
```

#### ❌ "DLL load failed while importing _C"
**Cause:** Incompatible PyTorch/diffusers versions.

**Solution:**
```powershell
.\scripts\fix-pytorch-compatibility-windows.ps1
```

#### ❌ "xFormers can't load C++/CUDA extensions"
**Cause:** Version mismatch between xFormers, PyTorch, and Python.

**Solution:** The fix script automatically handles this by:
- Removing incompatible xFormers
- Installing compatible PyTorch
- Skipping xFormers for Python 3.12

#### ❌ "Virtual environment not found"
**Cause:** Environment path issues or missing environment.

**Solution:**
```powershell
# Create/activate environment
.\scripts\load-venv-windows.ps1

# Or set custom path
$env:TRANSFORMERS_SHARP_VENV_PATH = "C:\your\custom\path"
.\scripts\load-venv-windows.ps1
```

### Advanced Diagnostics

#### Environment Variables
- `TRANSFORMERS_SHARP_VENV_PATH`: Custom virtual environment location
- `XFORMERS_MORE_DETAILS`: Set to "1" for detailed xFormers debugging
- `XFORMERS_DISABLED`: Set to "1" to disable xFormers completely

#### Manual Testing
```python
# Test in Python REPL
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")

import transformers
print(f"Transformers: {transformers.__version__}")

import diffusers  
print(f"Diffusers: {diffusers.__version__}")
```

## 📋 Best Practices

### For Developers

1. **Always run diagnostics first:**
   ```powershell
   .\scripts\check-gpu-cuda-windows.ps1
   ```

2. **Use the fix script for any compatibility issues:**
   ```powershell
   .\scripts\fix-pytorch-compatibility-windows.ps1
   ```

3. **Test your installation:**
   ```python
   python scripts/test_cuda.py
   ```

### For CI/CD

1. **Use cleanup scripts for fresh environments:**
   ```bash
   ./scripts/cleanup-venv-linux.sh
   ./scripts/load-venv-linux.sh
   ```

2. **Verify installation in automated tests:**
   ```bash
   python scripts/test_cuda.py
   ```

### For Production

1. **Use specific Python versions:**
   - Python 3.11 for CUDA environments
   - Python 3.12 for CPU-only (maximum stability)

2. **Set environment variables for consistency:**
   ```bash
   export TRANSFORMERS_SHARP_VENV_PATH="/opt/transformers-sharp/venv"
   export XFORMERS_DISABLED="1"  # For production stability
   ```

## 🆘 Getting Help

If these scripts don't resolve your issue:

1. **Check the full error message** in your application
2. **Run diagnostic script** for detailed system analysis
3. **Verify environment variables** are set correctly
4. **Try the fix script** even if the issue seems unrelated
5. **Test with the CUDA test script** to verify installation

For additional support, include the output of:
```powershell
.\scripts\check-gpu-cuda-windows.ps1
python scripts/test_cuda.py
```

---

*This documentation covers TransformersSharp utility scripts. For library-specific documentation, see the main README.*