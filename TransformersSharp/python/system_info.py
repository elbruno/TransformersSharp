"""
System Information Module for TransformersSharp

This module provides comprehensive system information gathering capabilities including
CPU, memory, GPU, and PyTorch installation details for performance analysis and debugging.
"""

from typing import Any, Dict
import torch


def get_detailed_system_info() -> Dict[str, Any]:
    """Get comprehensive system information for performance analysis."""
    import platform
    import psutil
    
    # Base system information
    info = {
        "cpu": _get_cpu_info(),
        "memory": _get_memory_info(),
        "pytorch": _get_pytorch_info()
    }
    
    # Add GPU information if available
    if torch.cuda.is_available():
        info["gpu"] = _get_gpu_info()
    else:
        info["gpu"] = {
            "device_count": 0,
            "available": False,
            "message": "CUDA not available"
        }
    
    return info


def _get_cpu_info() -> Dict[str, Any]:
    """Get CPU information."""
    import platform
    import psutil
    
    cpu_freq = psutil.cpu_freq()
    return {
        "processor": platform.processor(),
        "architecture": platform.architecture()[0],
        "logical_cores": psutil.cpu_count(logical=True),
        "physical_cores": psutil.cpu_count(logical=False),
        "cpu_freq_current": round(cpu_freq.current, 2) if cpu_freq else None,
        "cpu_freq_max": round(cpu_freq.max, 2) if cpu_freq else None,
    }


def _get_memory_info() -> Dict[str, Any]:
    """Get memory information."""
    import psutil
    
    memory = psutil.virtual_memory()
    return {
        "total_gb": round(memory.total / (1024**3), 2),
        "available_gb": round(memory.available / (1024**3), 2),
        "used_percent": memory.percent
    }


def _get_pytorch_info() -> Dict[str, Any]:
    """Get PyTorch information."""
    return {
        "version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() and torch.backends.cudnn.is_available() else None
    }


def _get_gpu_info() -> Dict[str, Any]:
    """Get GPU information."""
    gpu_info = {
        "device_count": torch.cuda.device_count(),
        "current_device": torch.cuda.current_device(),
        "devices": []
    }
    
    for i in range(torch.cuda.device_count()):
        device_props = torch.cuda.get_device_properties(i)
        gpu_info["devices"].append({
            "index": i,
            "name": device_props.name,
            "compute_capability": f"{device_props.major}.{device_props.minor}",
            "total_memory_gb": round(device_props.total_memory / (1024**3), 2),
            "multiprocessor_count": device_props.multi_processor_count,
            "allocated_memory_gb": round(torch.cuda.memory_allocated(i) / (1024**3), 2),
            "cached_memory_gb": round(torch.cuda.memory_reserved(i) / (1024**3), 2)
        })
    
    return gpu_info