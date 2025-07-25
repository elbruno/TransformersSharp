#!/bin/bash
# Use TRANSFORMERS_SHARP_VENV_PATH if set, otherwise use $HOME/.local/share/TransformersSharp/venv
if [ -n "$TRANSFORMERS_SHARP_VENV_PATH" ]; then
  venvPath="$TRANSFORMERS_SHARP_VENV_PATH"
else
  venvPath="$HOME/.local/share/TransformersSharp/venv"
fi

if [ ! -d "$venvPath" ]; then
  echo "Virtual environment not found at $venvPath. Creating new venv..."
  python3 -m venv "$venvPath"
fi
echo "Activating venv at $venvPath"
source "$venvPath/bin/activate"
