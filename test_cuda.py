#!/usr/bin/env python3
"""
Test script to check CUDA availability and PyTorch installation.
"""

import torch
import sys

def test_cuda():
    print("=" * 50)
    print("PyTorch CUDA Test")
    print("=" * 50)
    
    # Basic PyTorch info
    print(f"PyTorch version: {torch.__version__}")
    print(f"Python version: {sys.version}")
    
    # CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of CUDA devices: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
        
        # Test tensor operations
        print("\nTesting CUDA operations...")
        try:
            x = torch.randn(3, 3).cuda()
            y = torch.randn(3, 3).cuda()
            z = torch.matmul(x, y)
            print("✓ CUDA tensor operations working correctly")
            print(f"  Test result shape: {z.shape}")
            print(f"  Test result device: {z.device}")
        except Exception as e:
            print(f"✗ CUDA tensor operations failed: {e}")
    else:
        print("CUDA is not available. This could be due to:")
        print("1. No NVIDIA GPU installed")
        print("2. NVIDIA drivers not installed")
        print("3. PyTorch installed without CUDA support")
        print("4. Incompatible CUDA versions")
        
        # Test CPU operations
        print("\nTesting CPU operations...")
        try:
            x = torch.randn(3, 3)
            y = torch.randn(3, 3)
            z = torch.matmul(x, y)
            print("✓ CPU tensor operations working correctly")
        except Exception as e:
            print(f"✗ CPU tensor operations failed: {e}")
    
    print("=" * 50)

if __name__ == "__main__":
    test_cuda()
