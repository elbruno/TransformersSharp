#!/bin/bash
# Shell script to clean up TransformersSharp virtual environment on Linux
# This will remove the existing virtual environment and optionally recreate it

echo "=== TransformersSharp Virtual Environment Cleanup ==="
echo ""

# Get the virtual environment path
VENV_PATH="${TRANSFORMERS_SHARP_VENV_PATH:-$HOME/.local/share/TransformersSharp/venv}"

echo "Virtual Environment Path: $VENV_PATH"

# Check if virtual environment exists
if [ ! -d "$VENV_PATH" ]; then
    echo "✅ No virtual environment found at: $VENV_PATH"
    echo "Environment is already clean."
    exit 0
fi

# Check for -y parameter
AUTO_YES=0
if [ "$1" = "-y" ]; then
    AUTO_YES=1
fi

# Confirm deletion
echo ""
echo "⚠️  This will permanently delete the virtual environment and all installed packages."
echo "Virtual environment location: $VENV_PATH"
echo ""

if [ $AUTO_YES -eq 0 ]; then
    read -p "Are you sure you want to continue? (y/N): " confirmation
    if [[ ! "$confirmation" =~ ^[Yy]$ ]]; then
        echo "Operation cancelled."
        exit 0
    fi
else
    echo "Auto-confirmation enabled (-y parameter detected). Proceeding with deletion..."
fi

# Deactivate any active environment
if [ -n "$VIRTUAL_ENV" ]; then
    echo "Deactivating current virtual environment..."
    deactivate 2>/dev/null || true
fi

# Remove the virtual environment directory
echo "Removing virtual environment..."
if rm -rf "$VENV_PATH"; then
    echo "✅ Virtual environment removed successfully."
else
    echo "❌ Failed to remove virtual environment."
    echo "You may need to:"
    echo "1. Close any applications using the environment"
    echo "2. Run this script with sudo"
    echo "3. Manually delete the folder: $VENV_PATH"
    exit 1
fi

# Ask if user wants to recreate the environment
echo ""
read -p "Would you like to create a new clean virtual environment? (Y/n): " recreate
if [[ ! "$recreate" =~ ^[Nn]$ ]]; then
    echo "Creating new virtual environment..."
    
    # Create directory if it doesn't exist
    mkdir -p "$(dirname "$VENV_PATH")"
    
    if python3 -m venv "$VENV_PATH"; then
        echo "✅ New virtual environment created successfully."
        
        # Activate and upgrade pip
        echo "Activating new environment and upgrading pip..."
        source "$VENV_PATH/bin/activate"
        python -m pip install --upgrade pip
        echo "✅ Virtual environment ready for use."
        echo ""
        echo "You can now run your TransformersSharp application to install dependencies."
    else
        echo "❌ Failed to create virtual environment."
    fi
else
    echo "✅ Cleanup complete. Virtual environment removed."
fi

echo ""
echo "=== Cleanup Complete ==="
