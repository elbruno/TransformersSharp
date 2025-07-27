# PowerShell script to clean up TransformersSharp virtual environment on Windows
# This will remove the existing virtual environment and optionally recreate it

Write-Host "=== TransformersSharp Virtual Environment Cleanup ===" -ForegroundColor Green
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
    Write-Host "✅ No virtual environment found at: $venvPath" -ForegroundColor Green
    Write-Host "Environment is already clean." -ForegroundColor Green
    exit 0
}

# Confirm deletion
Write-Host ""
Write-Host "⚠️  This will permanently delete the virtual environment and all installed packages." -ForegroundColor Yellow
Write-Host "Virtual environment location: $venvPath" -ForegroundColor Cyan
Write-Host ""

$confirmation = Read-Host "Are you sure you want to continue? (y/N)"
if ($confirmation -ne "y" -and $confirmation -ne "Y") {
    Write-Host "Operation cancelled." -ForegroundColor Yellow
    exit 0
}
# Check for -y parameter
$autoYes = $false
if ($args.Count -gt 0 -and $args[0] -eq '-y') {
    $autoYes = $true
}

if (-not $autoYes) {
    $confirmation = Read-Host "Are you sure you want to continue? (y/N)"
    if ($confirmation -ne "y" -and $confirmation -ne "Y") {
        Write-Host "Operation cancelled." -ForegroundColor Yellow
        exit 0
    }
}
else {
    Write-Host "Auto-confirmation enabled (-y parameter detected). Proceeding with deletion..." -ForegroundColor Yellow
}

# Attempt to deactivate any active environment
try {
    if ($env:VIRTUAL_ENV) {
        Write-Host "Deactivating current virtual environment..." -ForegroundColor Yellow
        deactivate 2>$null
    }
}
catch {
    # Ignore errors - the environment might not be active
}

# Remove the virtual environment directory
Write-Host "Removing virtual environment..." -ForegroundColor Yellow
try {
    Remove-Item -Path $venvPath -Recurse -Force -ErrorAction Stop
    Write-Host "✅ Virtual environment removed successfully." -ForegroundColor Green
}
catch {
    Write-Host "❌ Failed to remove virtual environment: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "You may need to:" -ForegroundColor Yellow
    Write-Host "1. Close any applications using the environment" -ForegroundColor Yellow
    Write-Host "2. Run this script as Administrator" -ForegroundColor Yellow
    Write-Host "3. Manually delete the folder: $venvPath" -ForegroundColor Yellow
    exit 1
}

# Ask if user wants to recreate the environment
Write-Host ""
$recreate = Read-Host "Would you like to create a new clean virtual environment? (Y/n)"
if ($recreate -ne "n" -and $recreate -ne "N") {
    Write-Host "Creating new virtual environment..." -ForegroundColor Yellow
    
    try {
        # Create new virtual environment
        python -m venv $venvPath
        
        if (Test-Path $venvPath) {
            Write-Host "✅ New virtual environment created successfully." -ForegroundColor Green
            
            # Activate and upgrade pip
            $activateScript = Join-Path $venvPath "Scripts\Activate.ps1"
            if (Test-Path $activateScript) {
                Write-Host "Activating new environment and upgrading pip..." -ForegroundColor Yellow
                & $activateScript
                python -m pip install --upgrade pip
                Write-Host "✅ Virtual environment ready for use." -ForegroundColor Green
                Write-Host ""
                Write-Host "You can now run your TransformersSharp application to install dependencies." -ForegroundColor Cyan
            }
        }
        else {
            Write-Host "❌ Failed to create virtual environment." -ForegroundColor Red
        }
    }
    catch {
        Write-Host "❌ Failed to create virtual environment: $($_.Exception.Message)" -ForegroundColor Red
    }
}
else {
    Write-Host "✅ Cleanup complete. Virtual environment removed." -ForegroundColor Green
}

Write-Host ""
Write-Host "=== Cleanup Complete ===" -ForegroundColor Green
