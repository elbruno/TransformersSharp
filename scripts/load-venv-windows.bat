@echo off
REM Use TRANSFORMERS_SHARP_VENV_PATH if set, otherwise use LocalAppData
setlocal
if defined TRANSFORMERS_SHARP_VENV_PATH (
    set "venvPath=%TRANSFORMERS_SHARP_VENV_PATH%"
) else (
    set "venvPath=%LocalAppData%\TransformersSharp\venv"
)

if not exist "%venvPath%" (
    echo Virtual environment not found at %venvPath%. Creating new venv...
    python -m venv "%venvPath%"
)
echo Activating venv at %venvPath%
call "%venvPath%\Scripts\activate.bat"
endlocal
