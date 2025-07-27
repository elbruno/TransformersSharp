# Fixing PyTorch GPU Support Issues in TransformersSharp (Windows)

If you encounter issues where your Python environment does not recognize your GPU (e.g., `torch.cuda.is_available()` returns `False` or your image generation is not using the GPU), follow these steps to resolve the problem.

## Problem

- PyTorch is installed with CPU-only support, even though you have a compatible NVIDIA GPU and CUDA drivers.
- Running GPU-accelerated code results in slow performance or no GPU usage.
- You may see 0-byte or blank image files when running text-to-image pipelines on GPU.

## Solution: Use the Provided Fix Script

A PowerShell script is provided to automatically fix your environment by uninstalling CPU-only PyTorch packages and installing the correct CUDA-enabled versions.

### Steps

1. **Open PowerShell**
2. **Navigate to your project root folder** (if not already there)
3. **Run the fix script:**

   ```powershell
   ./scripts/fix-gpu-pytorch-windows.ps1
   ```

   This script will:
   - Activate your TransformersSharp virtual environment
   - Uninstall any existing torch, torchvision, and torchaudio packages
   - Install the CUDA-enabled versions of these packages (for CUDA 12.1)
   - Verify the installation and print your GPU info

4. **Check the output:**
   - If you see `cuda available: True` and your GPU name, your environment is ready for GPU acceleration.
   - If not, check your NVIDIA drivers and CUDA compatibility.

## Script Location

- [scripts/fix-gpu-pytorch-windows.ps1](../scripts/fix-gpu-pytorch-windows.ps1)

## Additional Notes

- This script assumes your venv is in the default location: `%LOCALAPPDATA%\TransformersSharp\venv`
- If you have a different setup, edit the script to point to your venv's `Activate.ps1`.
- For other CUDA versions, change the `--index-url` in the script accordingly (e.g., `cu118` for CUDA 11.8).

## Troubleshooting

- Ensure you have the latest NVIDIA drivers installed.
- Make sure your Python version is supported by the CUDA-enabled PyTorch wheels.
- If you still have issues, try creating a new venv and running the script again.

---

For more help, see the [PyTorch official installation guide](https://pytorch.org/get-started/locally/).
