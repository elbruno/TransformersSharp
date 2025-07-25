# PowerShell script to check GPU availability and CUDA installation
Write-Host "=== GPU and CUDA Detection Script ===" -ForegroundColor Green
Write-Host ""

# Check for NVIDIA GPU
Write-Host "1. Checking for NVIDIA GPU..." -ForegroundColor Yellow
try {
    $nvidiaInfo = Get-WmiObject -Class Win32_VideoController | Where-Object { $_.Name -like "*NVIDIA*" }
    if ($nvidiaInfo) {
        Write-Host "✅ NVIDIA GPU detected:" -ForegroundColor Green
        $nvidiaInfo | ForEach-Object {
            Write-Host "   - $($_.Name)" -ForegroundColor Cyan
            Write-Host "   - Driver Version: $($_.DriverVersion)" -ForegroundColor Cyan
        }
    }
    else {
        Write-Host "❌ No NVIDIA GPU detected" -ForegroundColor Red
        Write-Host "   This system will use CPU-only processing" -ForegroundColor Yellow
    }
}
catch {
    Write-Host "❌ Error checking GPU: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host ""

# Check nvidia-smi availability
Write-Host "2. Checking nvidia-smi (NVIDIA driver utility)..." -ForegroundColor Yellow
try {
    $nvidiaSmi = & nvidia-smi --version 2>$null
    if ($nvidiaSmi) {
        Write-Host "✅ nvidia-smi is available" -ForegroundColor Green
        Write-Host "   NVIDIA drivers are properly installed" -ForegroundColor Cyan
        
        # Get GPU details
        Write-Host ""
        Write-Host "GPU Details:" -ForegroundColor Yellow
        & nvidia-smi --query-gpu=name, driver_version, memory.total --format=csv, noheader, nounits 2>$null | ForEach-Object {
            $parts = $_ -split ","
            if ($parts.Length -ge 3) {
                Write-Host "   GPU: $($parts[0].Trim())" -ForegroundColor Cyan
                Write-Host "   Driver: $($parts[1].Trim())" -ForegroundColor Cyan  
                Write-Host "   Memory: $($parts[2].Trim()) MB" -ForegroundColor Cyan
            }
        }
    }
    else {
        Write-Host "❌ nvidia-smi not found" -ForegroundColor Red
        Write-Host "   NVIDIA drivers may not be installed or not in PATH" -ForegroundColor Yellow
    }
}
catch {
    Write-Host "❌ nvidia-smi not available" -ForegroundColor Red
    Write-Host "   NVIDIA drivers may not be installed" -ForegroundColor Yellow
}

Write-Host ""

# Check current Python environment and PyTorch CUDA support
Write-Host "3. Checking Python PyTorch CUDA support..." -ForegroundColor Yellow

$venvPath = $env:TRANSFORMERS_SHARP_VENV_PATH
if (-not $venvPath) {
    $appDataPath = [System.IO.Path]::Combine($env:LOCALAPPDATA, "TransformersSharp")
    $venvPath = [System.IO.Path]::Combine($appDataPath, "venv")
}

if (Test-Path $venvPath) {
    $activateScript = Join-Path $venvPath "Scripts\Activate.ps1"
    if (Test-Path $activateScript) {
        Write-Host "   Activating Python virtual environment..." -ForegroundColor Cyan
        & $activateScript
        
        # Check PyTorch CUDA availability
        $torchCheck = & python -c "
try:
    import torch
    print(f'PyTorch version: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'CUDA devices: {torch.cuda.device_count()}')
        for i in range(torch.cuda.device_count()):
            print(f'  Device {i}: {torch.cuda.get_device_name(i)}')
    else:
        print('CUDA not available in PyTorch - using CPU only')
        if '+cpu' in torch.__version__:
            print('CPU-only PyTorch detected')
        else:
            print('PyTorch CUDA support may need to be installed')
except ImportError:
    print('PyTorch not installed')
except Exception as e:
    print(f'Error checking PyTorch: {e}')
" 2>$null

        if ($torchCheck) {
            Write-Host "✅ PyTorch check results:" -ForegroundColor Green
            $torchCheck -split "`n" | ForEach-Object {
                Write-Host "   $_" -ForegroundColor Cyan
            }
        }
        else {
            Write-Host "❌ Could not check PyTorch CUDA support" -ForegroundColor Red
        }
    }
    else {
        Write-Host "❌ Virtual environment activation script not found" -ForegroundColor Red
    }
}
else {
    Write-Host "❌ Virtual environment not found at: $venvPath" -ForegroundColor Red
}

Write-Host ""
Write-Host "=== Recommendations ===" -ForegroundColor Green

# Provide recommendations based on findings
$hasNvidiaGpu = $null -ne (Get-WmiObject -Class Win32_VideoController | Where-Object { $_.Name -like "*NVIDIA*" })
$hasNvidiaSmi = $false
try { 
    & nvidia-smi --version 2>$null | Out-Null
    $hasNvidiaSmi = $true
}
catch { }

if ($hasNvidiaGpu -and $hasNvidiaSmi) {
    Write-Host "✅ GPU Setup Complete: You have an NVIDIA GPU with drivers" -ForegroundColor Green
    Write-Host "   Recommendation: Install CUDA PyTorch for optimal performance" -ForegroundColor Yellow
    Write-Host "   Run: .\scripts\fix-pytorch-compatibility-windows.ps1" -ForegroundColor Cyan
}
elseif ($hasNvidiaGpu -and -not $hasNvidiaSmi) {
    Write-Host "⚠️  GPU Present but Drivers Missing" -ForegroundColor Yellow
    Write-Host "   You have an NVIDIA GPU but drivers are not properly installed" -ForegroundColor Yellow
    Write-Host "   Recommendation: Install latest NVIDIA drivers from nvidia.com" -ForegroundColor Cyan
}
else {
    Write-Host "ℹ️  CPU-Only Configuration" -ForegroundColor Blue
    Write-Host "   No NVIDIA GPU detected - will use CPU processing" -ForegroundColor Yellow
    Write-Host "   This is perfectly fine for testing and development" -ForegroundColor Cyan
}
