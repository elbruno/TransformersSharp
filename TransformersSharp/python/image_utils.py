"""
Image Utilities Module for TransformersSharp

This module provides image processing and conversion utilities for handling various
image formats and converting them to bytes for C# interop.
"""

import numpy as np
from io import BytesIO
from PIL import Image as PILImage


def convert_image_to_bytes(image) -> bytes:
    """
    Convert various image formats to PNG bytes for C# interop.
    
    Args:
        image: Image object (PIL Image, numpy array, or tensor)
        
    Returns:
        PNG image as bytes
    """
    # Handle PIL Image
    if hasattr(image, 'save'):
        buffer = BytesIO()
        image.save(buffer, format='PNG')
        return buffer.getvalue()
    
    # Handle numpy array or tensor
    if hasattr(image, 'numpy'):
        image = image.numpy()
    
    # Convert to uint8 if needed
    if hasattr(image, 'dtype') and image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
        
    # Convert numpy array to PIL Image and then to bytes
    if len(image.shape) == 3 and image.shape[2] == 3:  # RGB
        pil_image = PILImage.fromarray(image, mode='RGB')
    else:
        pil_image = PILImage.fromarray(image)
        
    buffer = BytesIO()
    pil_image.save(buffer, format='PNG')
    return buffer.getvalue()