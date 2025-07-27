"""
TransformersSharp Python wrapper for machine learning pipelines and utilities.

This module provides Python-side functionality for TransformersSharp, including:
- Pipeline creation and execution 
- Package compatibility checking
- Text-to-image generation
- Speech recognition and text-to-audio
- Image classification and object detection
- Authentication and tokenization

Device management, system information, and image utilities are provided by separate modules.
"""

from typing import Any, Generator, Optional, Tuple, Dict, List
from transformers import pipeline as TransformersPipeline, Pipeline, TextGenerationPipeline
from huggingface_hub import login
import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from collections.abc import Buffer
from PIL import Image

# Import functionality from separate modules with fallbacks
try:
    from device_manager import is_cuda_available, get_device_info, validate_and_get_device
except ImportError:
    # Fallback implementations if module not available
    def is_cuda_available() -> bool:
        """Check if CUDA is available for PyTorch."""
        return torch.cuda.is_available()

    def get_device_info() -> dict:
        """Get information about available devices."""
        cuda_available = torch.cuda.is_available()
        return {
            "cuda_available": cuda_available,
            "cuda_device_count": torch.cuda.device_count() if cuda_available else 0,
            "current_device": torch.cuda.current_device() if cuda_available else None,
            "device_name": torch.cuda.get_device_name() if cuda_available else None
        }

    def validate_and_get_device(requested_device: Optional[str] = None, silent: bool = False) -> str:
        """Validate the requested device and return the best available device."""
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

try:
    from system_info import get_detailed_system_info
except ImportError:
    # Fallback implementation if module not available
    def get_detailed_system_info() -> dict:
        """Get detailed system information."""
        import platform
        import psutil
        
        return {
            "platform": f"{platform.system()} {platform.release()}",
            "architecture": "64-bit" if platform.machine().endswith('64') else "32-bit",
            "processor_count": psutil.cpu_count(),
            "python_version": f"{platform.python_version()}",
            "memory_info": {
                "working_set_mb": psutil.Process().memory_info().rss / 1024 / 1024
            }
        }

try:
    from image_utils import convert_image_to_bytes
except ImportError:
    # Fallback implementation if module not available  
    def convert_image_to_bytes(image) -> bytes:
        """Convert PIL Image to bytes."""
        from io import BytesIO
        
        if hasattr(image, 'save'):
            # PIL Image
            buffer = BytesIO()
            image.save(buffer, format='PNG')
            return buffer.getvalue()
        else:
            # Already bytes or other format
            return bytes(image)


# ==============================================================================
# Model Support and Validation
# ==============================================================================

def get_supported_text_to_image_models() -> List[Dict[str, Any]]:
    """
    Get list of supported text-to-image models with their configurations.
    
    Returns:
        List of model dictionaries with metadata
    """
    return [
        {
            "model_id": "kandinsky-community/kandinsky-2-2-decoder",
            "pipeline_class": "KandinskyV22Pipeline",
            "default": True,
            "description": "Artistic style generation with unique aesthetic",
            "recommended_size": (512, 512),
            "memory_requirements": "Medium",
            "speed": "Medium"
        },
        {
            "model_id": "black-forest-labs/FLUX.1-dev",
            "pipeline_class": "FluxPipeline",
            "default": False,
            "description": "State-of-the-art text-to-image generation with high quality and prompt adherence",
            "recommended_size": (1024, 1024),
            "memory_requirements": "High",
            "speed": "Medium",
            "notes": "Requires HuggingFace token acceptance and significant VRAM"
        }
    ]


def validate_text_to_image_model(model: str) -> Dict[str, Any]:
    """
    Validate if a model is supported and get its configuration.
    
    Args:
        model: Model identifier to validate
        
    Returns:
        Validation result with model information
    """
    supported_models = get_supported_text_to_image_models()
    
    # Check if model is in our supported list
    for model_info in supported_models:
        if model_info["model_id"] == model:
            return {
                "supported": True,
                "model_info": model_info,
                "pipeline_class": model_info["pipeline_class"],
                "recommendations": {
                    "size": model_info["recommended_size"],
                    "memory": model_info["memory_requirements"],
                    "speed": model_info["speed"]
                }
            }
    
    # Check if it's a variant or similar model
    model_lower = model.lower()
    if "kandinsky" in model_lower:
        return {
            "supported": True,
            "model_info": {
                "model_id": model,
                "pipeline_class": "KandinskyV22Pipeline", 
                "description": "Kandinsky variant (auto-detected)",
                "recommended_size": (512, 512)
            },
            "pipeline_class": "KandinskyV22Pipeline",
            "note": "Auto-detected as Kandinsky variant"
        }
    elif "flux" in model_lower and ("1-dev" in model_lower or "1.1-dev" in model_lower):
        return {
            "supported": True,
            "model_info": {
                "model_id": model,
                "pipeline_class": "FluxPipeline",
                "description": "FLUX variant (auto-detected)",
                "recommended_size": (1024, 1024)
            },
            "pipeline_class": "FluxPipeline",
            "note": "Auto-detected as FLUX variant"
        }
    else:
        return {
            "supported": False,
            "model_info": None,
            "pipeline_class": "AutoPipelineForText2Image",
            "note": f"Unknown model '{model}' - will attempt to use AutoPipelineForText2Image",
            "recommendation": "Use one of the officially supported models for best results"
        }



# ==============================================================================
# C# Interop Functions for Device Management  
# ==============================================================================

def is_cuda_available_wrapper() -> bool:
    """Check if CUDA is available (wrapper for C# interop)."""
    return is_cuda_available()


# ==============================================================================
# C# Interop Functions for Model Support
# ==============================================================================

def get_default_text_to_image_model() -> str:
    """Get the default text-to-image model identifier."""
    supported_models = get_supported_text_to_image_models()
    for model in supported_models:
        if model.get("default", False):
            return model["model_id"]
    return "kandinsky-community/kandinsky-2-2-decoder"  # Fallback


def get_model_pipeline_class_name(model: str) -> str:
    """Get the pipeline class name for a given model."""
    validation_result = validate_text_to_image_model(model)
    return validation_result["pipeline_class"]


def get_recommended_settings_for_model(model: str) -> Dict[str, Any]:
    """
    Get recommended generation settings for a specific model.
    
    Returns:
        Dictionary with recommended inference steps, guidance scale, and image dimensions
    """
    validation_result = validate_text_to_image_model(model)
    
    if validation_result["supported"] and validation_result.get("model_info"):
        model_info = validation_result["model_info"]
        recommended_size = model_info.get("recommended_size", (512, 512))
        
        # Model-specific recommendations
        model_lower = model.lower()
        if "kandinsky" in model_lower:
            return {
                "num_inference_steps": 25,
                "guidance_scale": 7.0,
                "height": recommended_size[1],
                "width": recommended_size[0],
                "optimal_for": "artistic and stylized images"
            }
        elif "flux" in model_lower and ("1-dev" in model_lower or "1.1-dev" in model_lower):
            return {
                "num_inference_steps": 20,
                "guidance_scale": 3.5,
                "height": recommended_size[1],
                "width": recommended_size[0],
                "optimal_for": "high quality photorealistic and artistic images"
            }
    
    # Default settings
    return {
        "num_inference_steps": 20,
        "guidance_scale": 7.5,
        "height": 512,
        "width": 512,
        "optimal_for": "general purpose image generation"
    }


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
    """Create a text-to-image pipeline with model-specific handling."""
    # Check package compatibility first
    is_compatible, error_msg = check_package_compatibility()
    
    if not is_compatible:
        if error_msg == "diffusers_dll_error":
            raise RuntimeError(_create_dll_error_message())
        else:
            raise RuntimeError(f"Cannot create text-to-image pipeline: {error_msg}")
    
    try:
        # Use model-specific pipeline based on the model identifier
        pipeline_class, pipeline_kwargs = _get_pipeline_for_model(model)

        # Special handling for Kandinsky: use torch_dtype only if CUDA
        if model and "kandinsky" in model.lower():
            if device == "cuda" and torch.cuda.is_available():
                pipeline = pipeline_class.from_pretrained(model, torch_dtype=torch.float16, **pipeline_kwargs)
            else:
                pipeline = pipeline_class.from_pretrained(model, **pipeline_kwargs)
        # FLUX models work best with bfloat16 on CUDA, or float32 on CPU
        elif model and "flux" in model.lower() and ("1-dev" in model.lower() or "1.1-dev" in model.lower()):
            if device == "cuda" and torch.cuda.is_available():
                pipeline = pipeline_class.from_pretrained(model, torch_dtype=torch.bfloat16, **pipeline_kwargs)
            else:
                pipeline = pipeline_class.from_pretrained(model, torch_dtype=torch.float32, **pipeline_kwargs)
        else:
            # Common parameters for all other pipelines
            common_kwargs = {'torch_dtype': torch_dtype, **pipeline_kwargs} if torch_dtype is not None else {**pipeline_kwargs}
            pipeline = pipeline_class.from_pretrained(model, **common_kwargs)

        # Move to device
        pipeline = pipeline.to(device)
        return pipeline
    except Exception as e:
        raise RuntimeError(_create_pipeline_error_message(str(e)))


def _get_pipeline_for_model(model: Optional[str]) -> Tuple[Any, Dict[str, Any]]:
    """
    Determine the appropriate pipeline class and kwargs for a specific model.
    
    Returns:
        Tuple of (pipeline_class, additional_kwargs)
    """
    if not model:
        model = "kandinsky-community/kandinsky-2-2-decoder"
    
    model_lower = model.lower()
    
    # Kandinsky models
    if "kandinsky" in model_lower:
        from diffusers import KandinskyV22Pipeline
        return KandinskyV22Pipeline, {}
    
    # FLUX models - use specific FluxPipeline implementation
    elif "flux" in model_lower and ("1-dev" in model_lower or "1.1-dev" in model_lower):
        try:
            from diffusers import FluxPipeline
            return FluxPipeline, {}
        except ImportError:
            # Fallback to AutoPipeline if FluxPipeline is not available
            from diffusers import AutoPipelineForText2Image
            return AutoPipelineForText2Image, {}
    
    # Default: try AutoPipelineForText2Image for unknown models
    else:
        from diffusers import AutoPipelineForText2Image
        return AutoPipelineForText2Image, {}


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
    width: Optional[int] = 512,
    max_sequence_length: Optional[int] = None,
    seed: Optional[int] = None,
    enable_model_cpu_offload: bool = False
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
        max_sequence_length: Maximum sequence length for FLUX models
        seed: Random seed for reproducible generation
        enable_model_cpu_offload: Enable CPU offloading for memory optimization (only if GPU available)
    Returns:
        The generated image as bytes
    """
    # Suppress FutureWarning about callback_steps deprecation
    import warnings
    from io import BytesIO
    import numpy as np
    import torch
    
    # Enable CPU offloading only if GPU is available and requested
    # This follows the FLUX.1-dev sample code pattern
    if enable_model_cpu_offload and torch.cuda.is_available() and hasattr(pipeline, 'enable_model_cpu_offload'):
        pipeline.enable_model_cpu_offload()
    
    # Setup generator for reproducible results
    generator = None
    if seed is not None:
        # For FLUX models, use CPU generator as in the sample code
        pipeline_class_name = pipeline.__class__.__name__.lower()
        if "flux" in pipeline_class_name:
            generator = torch.Generator("cpu").manual_seed(seed)
        else:
            # For other models, use device-appropriate generator
            device = getattr(pipeline, 'device', 'cpu')
            if hasattr(device, 'type'):
                device = device.type
            generator = torch.Generator(device).manual_seed(seed)
    
    # Generate image based on pipeline type
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, message=".*callback_steps.*")
        
        # Check if this is a Kandinsky V22 Pipeline that needs special handling
        pipeline_class_name = pipeline.__class__.__name__
        
        if "KandinskyV22" in pipeline_class_name and "Combined" not in pipeline_class_name:
            # This is a regular KandinskyV22Pipeline that needs prior embeddings
            # We need to use the combined pipeline approach
            try:
                from diffusers import KandinskyV22PriorPipeline
                
                # Extract model path from the pipeline
                model_path = getattr(pipeline, '_name_or_path', 'kandinsky-community/kandinsky-2-2-decoder')
                prior_model_path = model_path.replace('-decoder', '-prior')
                
                # Create prior pipeline to generate embeddings
                prior_pipeline = KandinskyV22PriorPipeline.from_pretrained(
                    prior_model_path,
                    torch_dtype=pipeline.dtype if hasattr(pipeline, 'dtype') else None
                ).to(pipeline.device)
                
                # Generate embeddings using the prior
                image_embeds, negative_image_embeds = prior_pipeline(
                    text,
                    guidance_scale=guidance_scale
                ).to_tuple()
                
                # Use the decoder pipeline with the embeddings
                kwargs = {
                    'image_embeds': image_embeds,
                    'negative_image_embeds': negative_image_embeds,
                    'num_inference_steps': num_inference_steps,
                    'guidance_scale': guidance_scale,
                    'height': height,
                    'width': width
                }
                
                # Add generator if seed is provided
                if generator is not None:
                    kwargs['generator'] = generator
                
                result = pipeline(**kwargs)
                image = result.images[0]
                
            except Exception as e:
                # Fallback: try to handle as a regular diffusers pipeline
                kwargs = {
                    'prompt': text,
                    'num_inference_steps': num_inference_steps,
                    'guidance_scale': guidance_scale,
                    'height': height,
                    'width': width
                }
                
                # Add generator if seed is provided
                if generator is not None:
                    kwargs['generator'] = generator
                
                result = pipeline(**kwargs)
                image = result.images[0] if hasattr(result, 'images') else result
                
        elif hasattr(pipeline, '__call__') and (hasattr(pipeline, 'unet') or "flux" in pipeline_class_name.lower()):
            # This is a diffusers pipeline (including FLUX)
            kwargs = {
                'prompt': text,
                'num_inference_steps': num_inference_steps,
                'guidance_scale': guidance_scale,
                'height': height,
                'width': width
            }
            
            # Add generator if seed is provided
            if generator is not None:
                kwargs['generator'] = generator
                
            # Add max_sequence_length for FLUX models
            if max_sequence_length is not None and 'flux' in pipeline_class_name.lower():
                kwargs['max_sequence_length'] = max_sequence_length
            
            result = pipeline(**kwargs)
            image = result.images[0]
        else:
            # This is a transformers pipeline
            kwargs = {
                'prompt': text,
                'num_inference_steps': num_inference_steps,
                'guidance_scale': guidance_scale,
                'height': height,
                'width': width
            }
            
            # Add generator if seed is provided
            if generator is not None:
                kwargs['generator'] = generator
            
            result = pipeline(**kwargs)
            
            if hasattr(result, 'images') and len(result.images) > 0:
                image = result.images[0]
            else:
                image = result
    

    # Debug: Validate image before converting to bytes
    if image is None:
        raise RuntimeError("Image generation failed: image is None")
    if not hasattr(image, 'save'):
        raise RuntimeError(f"Image generation failed: result is not a PIL Image, got {type(image)}")

    # Convert PIL Image to bytes
    return convert_image_to_bytes(image)

# Note: _convert_image_to_bytes function has been moved to image_utils.py module
# and is now imported as convert_image_to_bytes
