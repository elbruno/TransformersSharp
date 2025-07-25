#!/bin/bash
# Bash script to install CUDA-enabled PyTorch in the TransformersSharp venv

if [ -n "$TRANSFORMERS_SHARP_VENV_PATH" ]; then
  venvPath="$TRANSFORMERS_SHARP_VENV_PATH"
else
  venvPath="$HOME/.local/share/TransformersSharp/venv"
fi

if [ ! -d "$venvPath" ]; then
  echo "Virtual environment not found at $venvPath. Please run load-venv-linux.sh first."
  exit 1
fi

source "$venvPath/bin/activate"
echo "Attempting to install CUDA-enabled PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
if [ $? -ne 0 ]; then
  echo "CUDA install failed. Attempting CPU-only install..."
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
  echo "CPU-only PyTorch installed."
else
  echo "CUDA-enabled PyTorch installed successfully."
fi
