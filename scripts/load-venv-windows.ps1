# PowerShell script to activate Python venv for TransformersSharp
# Use TRANSFORMERS_SHARP_VENV_PATH if set, otherwise use LocalAppData
$venvPath = $env:TRANSFORMERS_SHARP_VENV_PATH
if (-not $venvPath) {
    $venvPath = Join-Path $env:LocalAppData 'TransformersSharp\venv'
}

if (-Not (Test-Path $venvPath)) {
    Write-Host "Virtual environment not found at $venvPath. Creating new venv..."
    python -m venv $venvPath
}
Write-Host "Activating venv at $venvPath"
. "$venvPath\Scripts\Activate.ps1"
