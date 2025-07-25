@echo off
REM Batch script to install CUDA-enabled PyTorch in the TransformersSharp venv
setlocal
if defined TRANSFORMERS_SHARP_VENV_PATH (
    set "venvPath=%TRANSFORMERS_SHARP_VENV_PATH%"
) else (
    set "venvPath=%LocalAppData%\TransformersSharp\venv"
)

if not exist "%venvPath%" (
    echo Virtual environment not found at %venvPath%. Please run load-venv-windows.bat first.
    exit /b 1
)

call "%venvPath%\Scripts\activate.bat"
echo Attempting to install CUDA-enabled PyTorch...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 (
    echo CUDA install failed. Attempting CPU-only install...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    echo CPU-only PyTorch installed.
) else (
    echo CUDA-enabled PyTorch installed successfully.
)
endlocal
