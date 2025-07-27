# fix-gpu-pytorch-windows.ps1
# This script activates the TransformersSharp venv, uninstalls CPU-only torch/torchvision/torchaudio,
# and reinstalls the CUDA-enabled versions for GPU support.
# Usage: Just run this script from anywhere in PowerShell.

# Find the venv path (default location)
$venvPath = "$env:LOCALAPPDATA\TransformersSharp\venv"
$activateScript = Join-Path $venvPath 'Scripts\Activate.ps1'

if (-Not (Test-Path $activateScript)) {
    Write-Host "Could not find venv activation script at $activateScript. Please check your venv location."
    exit 1
}

Write-Host "Activating venv at $venvPath..."
. $activateScript

Write-Host "Uninstalling existing torch, torchvision, torchaudio..."
pip uninstall torch torchvision torchaudio -y

Write-Host "Installing CUDA-enabled torch, torchvision, torchaudio (CUDA 12.1)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

Write-Host "Verifying installation..."
python -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('cuda version:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"

Write-Host "Done. If 'cuda available' is True, your environment is ready for GPU!"