"""
Device Management Module for TransformersSharp

This module provides device detection and management functionality for CUDA/CPU devices.
Handles device validation, availability checking, and automatic fallback mechanisms.
"""

from typing import Any, Dict, Optional
import torch


def is_cuda_available() -> bool:
    """Check if CUDA is available for PyTorch."""
    return torch.cuda.is_available()


def get_device_info() -> Dict[str, Any]:
    """Get information about available devices."""
    cuda_available = torch.cuda.is_available()
    return {
        "cuda_available": cuda_available,
        "cuda_device_count": torch.cuda.device_count() if cuda_available else 0,
        "current_device": torch.cuda.current_device() if cuda_available else None,
        "device_name": torch.cuda.get_device_name() if cuda_available else None
    }


def validate_and_get_device(requested_device: Optional[str] = None, silent: bool = False) -> str:
    """
    Validate the requested device and return the best available device.
    
    Args:
        requested_device: The requested device ('cuda', 'cpu', etc.)
        silent: If True, suppress warning messages for graceful fallback
    
    Returns:
        The validated device string
    """
    if requested_device is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    
    if requested_device.lower() == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        else:
            if not silent:
                print("Warning: CUDA requested but not available. PyTorch was not compiled with CUDA support. Falling back to CPU.")
            return "cpu"
    
    return requested_device