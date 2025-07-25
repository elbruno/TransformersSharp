# PowerShell script to install CUDA-enabled PyTorch in the TransformersSharp venv

# Use TRANSFORMERS_SHARP_VENV_PATH if set, otherwise use LocalAppData
$venvPath = $env:TRANSFORMERS_SHARP_VENV_PATH
if (-not $venvPath) {
    $venvPath = Join-Path $env:LocalAppData 'TransformersSharp\venv'
}

if (-Not (Test-Path $venvPath)) {
    Write-Host "Virtual environment not found at $venvPath. Please run load-venv-windows.ps1 first."
    exit 1
}

Write-Host "Activating venv at $venvPath"
. "$venvPath\Scripts\Activate.ps1"

Write-Host "Attempting to install CUDA-enabled PyTorch..."
try {
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    Write-Host "CUDA-enabled PyTorch installed successfully."
}
catch {
    Write-Host "CUDA install failed. Attempting CPU-only install..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    Write-Host "CPU-only PyTorch installed."
}
