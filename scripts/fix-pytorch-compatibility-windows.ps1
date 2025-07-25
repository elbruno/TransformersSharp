# PowerShell script to fix PyTorch compatibility issues for TransformersSharp
# This script addresses xFormers compatibility warnings and Python version mismatches

Write-Host "=== TransformersSharp PyTorch Compatibility Fix ===" -ForegroundColor Green
Write-Host ""

# Get the virtual environment path
$venvPath = $env:TRANSFORMERS_SHARP_VENV_PATH
if (-not $venvPath) {
    $appDataPath = [System.IO.Path]::Combine($env:LOCALAPPDATA, "TransformersSharp")
    $venvPath = [System.IO.Path]::Combine($appDataPath, "venv")
}

Write-Host "Virtual Environment Path: $venvPath" -ForegroundColor Yellow

# Check if virtual environment exists
if (-not (Test-Path $venvPath)) {
    Write-Host "❌ Virtual environment not found at: $venvPath" -ForegroundColor Red
    Write-Host "Please run your TransformersSharp application first to create the environment." -ForegroundColor Red
    exit 1
}

# Activate the virtual environment
$activateScript = Join-Path $venvPath "Scripts\Activate.ps1"
if (-not (Test-Path $activateScript)) {
    Write-Host "❌ Virtual environment activation script not found" -ForegroundColor Red
    exit 1
}

Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& $activateScript

# Get Python version
Write-Host "Checking Python version..." -ForegroundColor Yellow
$pythonVersion = & python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
Write-Host "Python version: $pythonVersion" -ForegroundColor Cyan

# Check current PyTorch installation
Write-Host "Checking current PyTorch installation..." -ForegroundColor Yellow
$torchInfo = & python -c "
try:
    import torch
    print(f'PyTorch version: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    print(f'CPU-only build: {'+cpu' in torch.__version__}')
except ImportError:
    print('PyTorch not installed')
" 2>$null

Write-Host $torchInfo -ForegroundColor Cyan

# Check for xformers
Write-Host "Checking xFormers installation..." -ForegroundColor Yellow
$xformersStatus = & python -c "
try:
    import xformers
    print(f'xFormers version: {xformers.__version__}')
    print('xFormers is installed')
except ImportError:
    print('xFormers not installed')
" 2>$null

Write-Host $xformersStatus -ForegroundColor Cyan

# Uninstall problematic packages first
Write-Host ""
Write-Host "=== Fixing Package Compatibility ===" -ForegroundColor Green

if ($xformersStatus -like "*xFormers is installed*") {
    Write-Host "Uninstalling incompatible xFormers..." -ForegroundColor Yellow
    & python -m pip uninstall -y xformers
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ xFormers uninstalled successfully" -ForegroundColor Green
    }
    else {
        Write-Host "⚠️  Failed to uninstall xFormers (this is usually OK)" -ForegroundColor Yellow
    }
}

# Reinstall PyTorch based on Python version
Write-Host ""
Write-Host "Installing compatible PyTorch..." -ForegroundColor Yellow

switch ($pythonVersion) {
    "3.12" {
        Write-Host "Python 3.12 detected - installing CPU-only PyTorch for maximum compatibility..." -ForegroundColor Cyan
        & python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --force-reinstall
        Write-Host "Note: xFormers will be skipped for Python 3.12 to avoid compatibility warnings." -ForegroundColor Yellow
    }
    { $_ -in "3.10", "3.11" } {
        Write-Host "Python $pythonVersion detected - attempting CUDA PyTorch installation..." -ForegroundColor Cyan
        & python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --force-reinstall
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ CUDA PyTorch installed successfully" -ForegroundColor Green
            
            # Try to install compatible xformers
            Write-Host "Installing compatible xFormers..." -ForegroundColor Yellow
            & python -m pip install xformers 2>$null
            if ($LASTEXITCODE -eq 0) {
                Write-Host "✅ xFormers installed successfully" -ForegroundColor Green
            }
            else {
                Write-Host "⚠️  xFormers installation failed - proceeding without it" -ForegroundColor Yellow
                Write-Host "   (This is OK - basic functionality will work without optimization)" -ForegroundColor Gray
            }
        }
        else {
            Write-Host "⚠️  CUDA PyTorch installation failed - falling back to CPU version" -ForegroundColor Yellow
            & python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --force-reinstall
        }
    }
    default {
        Write-Host "Python $pythonVersion - installing CPU-only PyTorch..." -ForegroundColor Cyan
        & python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --force-reinstall
    }
}

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ PyTorch installation completed successfully" -ForegroundColor Green
}
else {
    Write-Host "❌ PyTorch installation failed" -ForegroundColor Red
    exit 1
}

# Verify the installation
Write-Host ""
Write-Host "=== Verifying Installation ===" -ForegroundColor Green

$verificationResult = & python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')

try:
    import transformers
    print(f'Transformers version: {transformers.__version__}')
except ImportError as e:
    print(f'Transformers import failed: {e}')

try:
    import diffusers
    print(f'Diffusers version: {diffusers.__version__}')
except ImportError as e:
    print(f'Diffusers import failed: {e}')

try:
    import xformers
    print(f'xFormers version: {xformers.__version__}')
except ImportError:
    print('xFormers not installed (this is OK for Python 3.12)')

print('Installation verification complete')
"

Write-Host $verificationResult -ForegroundColor Cyan

Write-Host ""
Write-Host "=== Setup Complete ===" -ForegroundColor Green
Write-Host "You can now run your TransformersSharp application." -ForegroundColor Green
Write-Host "The compatibility warnings should be resolved." -ForegroundColor Green
Write-Host ""
Write-Host "If you still see warnings, they should be non-critical and not affect functionality." -ForegroundColor Yellow
