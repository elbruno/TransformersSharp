"""
TransformersSharp Python wrapper for machine learning pipelines and utilities.

This module provides Python-side functionality for TransformersSharp, including:
- Device detection and management (CUDA/CPU)
- Pipeline creation and execution 
- System information gathering
- Package compatibility checking
- Text-to-image generation
- Speech recognition and text-to-audio
- Image classification and object detection
"""

from typing import Any, Generator, Optional, Tuple, Dict, List
from transformers import pipeline as TransformersPipeline, Pipeline, TextGenerationPipeline
from huggingface_hub import login
import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from collections.abc import Buffer
from PIL import Image


# ==============================================================================
# Device Management and System Information
# ==============================================================================

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


# ==============================================================================
# Device Validation and Pipeline Creation
# ==============================================================================

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


# ==============================================================================
# Package Compatibility Checking
# ==============================================================================

def check_text_to_image_compatibility() -> Dict[str, Any]:
    """
    Check if text-to-image pipelines can be created successfully.
    Returns detailed information about compatibility issues.
    """
    try:
        from diffusers import AutoPipelineForText2Image
        import sys
        
        result = {
            "compatible": True,
            "can_create_pipeline": True,
            "pytorch_version": torch.__version__,
            "python_version": sys.version.split()[0],
            "cuda_available": torch.cuda.is_available(),
            "is_cuda_pytorch": '+cu' in torch.__version__,
            "message": "Text-to-image pipelines should work correctly"
        }
        
        return result
        
    except ImportError as e:
        return _handle_import_error(e)
    except Exception as e:
        return _handle_unknown_error(e)


def _handle_import_error(error: ImportError) -> Dict[str, Any]:
    """Handle import errors during compatibility check."""
    import sys
    
    base_info = {
        "compatible": False,
        "can_create_pipeline": False,
        "pytorch_version": torch.__version__,
        "python_version": sys.version.split()[0],
        "cuda_available": torch.cuda.is_available(),
        "is_cuda_pytorch": '+cu' in torch.__version__,
        "error_details": str(error)
    }
    
    if "_C" in str(error) or "DLL load failed" in str(error):
        base_info.update({
            "error_type": "dll_compatibility_error",
            "message": "Diffusers cannot load C extensions - likely PyTorch/diffusers version incompatibility",
            "recommended_action": "reinstall_compatible_packages"
        })
    else:
        base_info.update({
            "error_type": "import_error",
            "message": f"Cannot import diffusers: {str(error)}",
            "recommended_action": "install_diffusers"
        })
    
    return base_info


def _handle_unknown_error(error: Exception) -> Dict[str, Any]:
    """Handle unknown errors during compatibility check."""
    import sys
    
    return {
        "compatible": False,
        "can_create_pipeline": False,
        "error_type": "unknown_error",
        "pytorch_version": torch.__version__,
        "python_version": sys.version.split()[0],
        "cuda_available": torch.cuda.is_available(),
        "message": f"Unknown error checking compatibility: {str(error)}",
        "error_details": str(error),
        "recommended_action": "check_installation"
    }


def check_package_compatibility() -> Tuple[bool, Optional[str]]:
    """
    Check if the current PyTorch and diffusers installation is compatible.
    
    Returns:
        Tuple of (is_compatible, error_message)
    """
    try:
        from diffusers import AutoPipelineForText2Image
        # Test basic import - if this succeeds, we're likely compatible
        return True, None
    except ImportError as e:
        if "_C" in str(e) or "DLL load failed" in str(e):
            return False, "diffusers_dll_error"
        else:
            return False, f"diffusers_import_error: {str(e)}"
    except Exception as e:
        return False, f"diffusers_unknown_error: {str(e)}"

# ==============================================================================
# Pipeline Creation and Management
# ==============================================================================

def pipeline(
    task: Optional[str] = None, 
    model: Optional[str] = None, 
    tokenizer: Optional[str] = None, 
    torch_dtype: Optional[str] = None, 
    device: Optional[str] = None, 
    trust_remote_code: bool = False, 
    silent_device_fallback: bool = False
):
    """
    Create a pipeline for a specific task using the Hugging Face Transformers library.
    
    Args:
        task: The task type (e.g., 'text-classification', 'text-to-image')
        model: Model name or path
        tokenizer: Tokenizer name or path  
        torch_dtype: PyTorch data type as string
        device: Target device ('cuda', 'cpu', etc.)
        trust_remote_code: Whether to trust remote code
        silent_device_fallback: Whether to suppress device fallback warnings
        
    Returns:
        Configured pipeline object
    """
    # Validate torch dtype
    if torch_dtype is not None:
        if not hasattr(torch, torch_dtype.lower()):
            raise ValueError(f"Unsupported torch_dtype: {torch_dtype}")
        torch_dtype = getattr(torch, torch_dtype.lower())
    
    # Validate and get the best available device
    device = validate_and_get_device(device, silent=silent_device_fallback)
    
    # Handle text-to-image using diffusers
    if task == "text-to-image":
        return _create_text_to_image_pipeline(model, torch_dtype, device)
    
    # Handle other tasks using transformers
    return TransformersPipeline(
        task=task, 
        model=model, 
        tokenizer=tokenizer, 
        torch_dtype=torch_dtype, 
        device=device, 
        trust_remote_code=trust_remote_code
    )


def _create_text_to_image_pipeline(model: Optional[str], torch_dtype, device: str):
    """Create a text-to-image pipeline with error handling."""
    # Check package compatibility first
    is_compatible, error_msg = check_package_compatibility()
    
    if not is_compatible:
        if error_msg == "diffusers_dll_error":
            raise RuntimeError(_create_dll_error_message())
        else:
            raise RuntimeError(f"Cannot create text-to-image pipeline: {error_msg}")
    
    try:
        from diffusers import AutoPipelineForText2Image
        return AutoPipelineForText2Image.from_pretrained(
            model, 
            torch_dtype=torch_dtype
        ).to(device)
    except Exception as e:
        raise RuntimeError(_create_pipeline_error_message(str(e)))


def _create_dll_error_message() -> str:
    """Create detailed error message for DLL compatibility issues."""
    pytorch_version = torch.__version__
    import sys
    python_version = sys.version
    
    return f"""
Text-to-image pipeline creation failed due to package compatibility issues.

ISSUE: Diffusers library cannot load required C extensions
CAUSE: Incompatible PyTorch and diffusers versions for your system

Current Environment:
- PyTorch: {pytorch_version}
- Python: {python_version.split()[0]}

SOLUTION:
1. Uninstall current packages:
   pip uninstall torch torchvision torchaudio diffusers xformers -y

2. Install compatible versions:
   For CPU-only:
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   pip install diffusers[torch] --no-deps
   pip install safetensors accelerate

   For CUDA (if you have NVIDIA GPU):
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   pip install diffusers xformers

3. Restart your application

This error commonly occurs when:
- PyTorch CPU version is mixed with CUDA-compiled extensions
- Python version mismatches with compiled packages
- Missing Visual C++ Redistributables (Windows)
"""


def _create_pipeline_error_message(error_str: str) -> str:
    """Create error message for general pipeline creation failures."""
    return f"""
Text-to-image pipeline creation failed: {error_str}

This may be due to:
1. Model download issues - ensure internet connectivity
2. Insufficient disk space for model files
3. Model incompatibility with current diffusers version

Try:
1. Check your internet connection
2. Ensure you have sufficient disk space (models can be several GB)
3. Try a different model like 'runwayml/stable-diffusion-v1-5'
4. Update diffusers: pip install --upgrade diffusers
"""


# ==============================================================================
# Authentication
# ==============================================================================

def huggingface_login(token: str) -> None:
    """Login to Hugging Face with the provided token."""
    login(token=token)


# ==============================================================================
# Pipeline Execution Functions
# ==============================================================================

def call_pipeline(pipeline: Pipeline, input: str, **kwargs) -> List[Dict[str, Any]]:
    """Execute a pipeline with string input."""
    return pipeline(input, **kwargs)


def call_pipeline_with_list(pipeline: Pipeline, input: List[str], **kwargs) -> List[Dict[str, Any]]:
    """Execute a pipeline with list input."""
    return pipeline(input, **kwargs)


def invoke_text_generation_pipeline_with_template(
    pipeline: TextGenerationPipeline, 
    messages: List[Dict[str, str]],
    max_length: Optional[int] = None,
    max_new_tokens: Optional[int] = None,
    min_length: Optional[int] = None,
    min_new_tokens: Optional[int] = None,
    stop_strings: Optional[List[str]] = None,
    temperature: Optional[float] = 1.0,
    top_k: Optional[int] = 50,
    top_p: Optional[float] = 1.0,
    min_p: Optional[float] = None,
) -> List[Dict[str, str]]:
    """
    Invoke a text generation pipeline with a chat template.
    Use pytorch for intermediate tensors (template -> generate)
    """
    result = pipeline(
        messages, 
        max_length=max_length, 
        max_new_tokens=max_new_tokens, 
        min_length=min_length, 
        min_new_tokens=min_new_tokens, 
        stop=stop_strings, 
        temperature=temperature, 
        top_k=top_k, 
        top_p=top_p, 
        min_p=min_p
    )
    return result[0]['generated_text']


def stream_text_generation_pipeline_with_template(
    pipeline: TextGenerationPipeline, 
    messages: List[Dict[str, str]],
    max_length: Optional[int] = None,
    max_new_tokens: Optional[int] = None,
    min_length: Optional[int] = None,
    min_new_tokens: Optional[int] = None,
    stop_strings: Optional[List[str]] = None,
    temperature: Optional[float] = 1.0,
    top_k: Optional[int] = 50,
    top_p: Optional[float] = 1.0,
    min_p: Optional[float] = None,
) -> Generator[Dict[str, str], None, None]:
    """
    Invoke a text generation pipeline with a chat template and stream results.
    Use pytorch for intermediate tensors (template -> generate)
    """
    result = pipeline(
        messages, 
        max_length=max_length, 
        max_new_tokens=max_new_tokens, 
        min_length=min_length, 
        min_new_tokens=min_new_tokens, 
        stop=stop_strings, 
        temperature=temperature, 
        top_k=top_k, 
        top_p=top_p, 
        min_p=min_p
    )
    
    # TODO: Implement actual streaming - for now yield all messages
    for message in result[0]['generated_text']:
        yield message


# ==============================================================================
# Tokenization Functions  
# ==============================================================================

def tokenizer_from_pretrained(
    model: str, 
    cache_dir: Optional[str] = None, 
    force_download: bool = False, 
    revision: Optional[str] = 'main', 
    trust_remote_code: bool = False
) -> PreTrainedTokenizerBase:
    """Load a pretrained tokenizer."""
    return AutoTokenizer.from_pretrained(
        model, 
        cache_dir=cache_dir, 
        force_download=force_download, 
        revision=revision, 
        trust_remote_code=trust_remote_code
    )


def tokenize_text_with_attention(tokenizer: PreTrainedTokenizerBase, text: str) -> Tuple[List[int], List[int]]:
    """Tokenize text and return input IDs and attention mask."""
    result = tokenizer(text)
    return result['input_ids'], result['attention_mask']


def tokenizer_text_as_ndarray(
    tokenizer: PreTrainedTokenizerBase, 
    text: str, 
    add_special_tokens: Optional[bool] = True
) -> Buffer:
    """Tokenize text and return as numpy array."""
    result = tokenizer(text, return_tensors='np', return_attention_mask=False, add_special_tokens=add_special_tokens)
    return result['input_ids'][0]


def tokenizer_text_with_offsets(
    tokenizer: PreTrainedTokenizerBase, 
    text: str, 
    add_special_tokens: Optional[bool] = True
) -> Tuple[Buffer, Buffer]:
    """Tokenize text and return input IDs and offset mapping."""
    result = tokenizer(text, return_tensors='np', return_offsets_mapping=True, return_attention_mask=False, add_special_tokens=add_special_tokens)
    return result['input_ids'][0], result['offset_mapping'][0]


def tokenizer_decode(
    tokenizer: PreTrainedTokenizerBase, 
    input_ids: List[int], 
    skip_special_tokens: bool = False,
    clean_up_tokenization_spaces: Optional[bool] = None
) -> str:
    """Decode token IDs back to text."""
    return tokenizer.decode(input_ids, skip_special_tokens=skip_special_tokens, clean_up_tokenization_spaces=clean_up_tokenization_spaces)


# ==============================================================================
# Specialized Pipeline Functions
# ==============================================================================

def invoke_image_classification_pipeline(
    pipeline: Pipeline, 
    image: str,
    function_to_apply: Optional[str] = None,
    top_k: Optional[int] = 5,
    timeout: Optional[float] = None
) -> List[Dict[str, Any]]:
    """Invoke an image classification pipeline."""
    return pipeline(image, top_k=top_k, timeout=timeout, function_to_apply=function_to_apply)


def invoke_image_classification_from_bytes(
    pipeline: Pipeline, 
    data: bytes,
    width: int,
    height: int,
    pixel_mode: str,
    function_to_apply: Optional[str] = None,
    top_k: int = 5,
    timeout: Optional[float] = None
) -> List[Dict[str, Any]]:
    """Invoke an image classification pipeline from raw bytes."""
    image = Image.frombytes(pixel_mode, (width, height), data)
    return pipeline(image, top_k=top_k, timeout=timeout, function_to_apply=function_to_apply)


def invoke_object_detection_pipeline(
    pipeline: Pipeline, 
    image: str,
    threshold: float = 0.5,
    timeout: Optional[float] = None
) -> Generator[Tuple[str, float, Tuple[int, int, int, int]], None, None]:
    """Invoke an object detection pipeline."""
    results = pipeline(image, threshold=threshold, timeout=timeout)
    for result in results:
        box = result["box"]
        yield (result["label"], result["score"], (box["xmin"], box["ymin"], box["xmax"], box["ymax"]))


def invoke_text_to_audio_pipeline(
    pipeline: Pipeline, 
    text: str,
    generate_kwargs: Optional[Dict[str, Any]] = None
) -> Tuple[Buffer, int]:
    """Invoke a text-to-audio pipeline."""
    if not generate_kwargs:
        generate_kwargs = {}
    result = pipeline(text, **generate_kwargs)
    return result['audio'], result['sampling_rate']


def invoke_automatic_speech_recognition_pipeline(pipeline: Pipeline, audio: str) -> str:
    """
    Invoke an automatic speech recognition pipeline and return the result.

    Args:
        pipeline: The ASR pipeline object
        audio: A local file path or a URL
    Returns:
        The text detected
    """
    result = pipeline(audio, return_timestamps=False)
    return result['text']


def invoke_automatic_speech_recognition_pipeline_from_bytes(pipeline: Pipeline, audio: bytes) -> str:
    """
    Invoke an automatic speech recognition pipeline and return the result.
    
    Args:
        pipeline: The ASR pipeline object
        audio: The bytes of the audio file
    Returns:
        The text detected
    """
    result = pipeline(audio, return_timestamps=False)
    return result['text']


def invoke_text_to_image_pipeline(
    pipeline, 
    text: str,
    num_inference_steps: Optional[int] = 50,
    guidance_scale: Optional[float] = 7.5,
    height: Optional[int] = 512,
    width: Optional[int] = 512
) -> Buffer:
    """
    Invoke a text-to-image pipeline.
    
    Args:
        pipeline: The text-to-image pipeline object (could be diffusers or transformers)
        text: The text prompt for image generation
        num_inference_steps: Number of inference steps
        guidance_scale: Guidance scale for generation
        height: Height of generated image
        width: Width of generated image
    Returns:
        The generated image as bytes
    """
    # Suppress FutureWarning about callback_steps deprecation
    import warnings
    from io import BytesIO
    import numpy as np
    
    # Generate image based on pipeline type
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, message=".*callback_steps.*")
        
        if hasattr(pipeline, '__call__') and hasattr(pipeline, 'unet'):
            # This is a diffusers pipeline
            result = pipeline(
                text, 
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width
            )
            image = result.images[0]
        else:
            # This is a transformers pipeline
            result = pipeline(
                text, 
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width
            )
            
            if hasattr(result, 'images') and len(result.images) > 0:
                image = result.images[0]
            else:
                image = result
    
    # Convert PIL Image to bytes
    return _convert_image_to_bytes(image)


def _convert_image_to_bytes(image) -> bytes:
    """Convert various image formats to PNG bytes."""
    from io import BytesIO
    import numpy as np
    from PIL import Image as PILImage
    
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
