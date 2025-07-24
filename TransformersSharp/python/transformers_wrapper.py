from typing import Any, Generator, Optional
from transformers import pipeline as TransformersPipeline, Pipeline, TextGenerationPipeline
from huggingface_hub import login
import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from collections.abc import Buffer
from PIL import Image

def is_cuda_available() -> bool:
    """
    Check if CUDA is available for PyTorch.
    """
    return torch.cuda.is_available()

def get_device_info() -> dict[str, Any]:
    """
    Get information about available devices.
    """
    return {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None,
        "device_name": torch.cuda.get_device_name() if torch.cuda.is_available() else None
    }

def validate_and_get_device(requested_device: Optional[str] = None) -> str:
    """
    Validate the requested device and return the best available device.
    If CUDA is requested but not available, fall back to CPU with a warning.
    """
    if requested_device is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    
    if requested_device.lower() == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        else:
            print("Warning: CUDA requested but not available. PyTorch was not compiled with CUDA support. Falling back to CPU.")
            return "cpu"
    
    return requested_device

def pipeline(task: Optional[str] = None, model: Optional[str] = None, tokenizer: Optional[str] = None, torch_dtype: Optional[str] = None, device: Optional[str] = None, trust_remote_code: bool = False):
    """
    Create a pipeline for a specific task using the Hugging Face Transformers library.
    """
    if torch_dtype is not None:
        if not hasattr(torch, torch_dtype.lower()):
            raise ValueError(f"Unsupported torch_dtype: {torch_dtype}")
        else:
            torch_dtype = getattr(torch, torch_dtype.lower())
    
    # Validate and get the best available device
    device = validate_and_get_device(device)
    
    # Handle text-to-image using diffusers
    if task == "text-to-image":
        from diffusers import AutoPipelineForText2Image
        # Remove trust_remote_code for diffusers pipelines to avoid warning
        return AutoPipelineForText2Image.from_pretrained(
            model, 
            torch_dtype=torch_dtype
        ).to(device)
    
    # Only pass trust_remote_code for transformers pipelines
    return TransformersPipeline(task=task, model=model, tokenizer=tokenizer, torch_dtype=torch_dtype, device=device, trust_remote_code=trust_remote_code)


def invoke_text_generation_pipeline_with_template(pipeline: TextGenerationPipeline, 
                             messages: list[dict[str, str]],
                             max_length: Optional[int] = None,
                             max_new_tokens: Optional[int] = None,
                             min_length: Optional[int] = None,
                             min_new_tokens: Optional[int] = None,
                             stop_strings: Optional[list[str]] = None,
                             temperature: Optional[float] = 1.0,
                             top_k: Optional[int] = 50,
                             top_p: Optional[float] = 1.0,
                             min_p: Optional[float] = None,
                            ) -> list[dict[str, str]]:
    """
    Invoke a text generation pipeline with a chat template.
    Use pytorch for intermediate tensors (template -> generate)
    """
    # Apply template to messages
    r = pipeline(messages, max_length=max_length, max_new_tokens=max_new_tokens, min_length=min_length, min_new_tokens=min_new_tokens, stop=stop_strings, temperature=temperature, top_k=top_k, top_p=top_p, min_p=min_p)
    return r[0]['generated_text']


def huggingface_login(token: str) -> None:
    login(token=token)


def call_pipeline(pipeline: Pipeline, input: str, **kwargs) -> list[dict[str, Any]]:
    return pipeline(input, **kwargs)


def call_pipeline_with_list(pipeline: Pipeline, input: list[str], **kwargs) -> list[dict[str, Any]]:
    return pipeline(input, **kwargs)


def tokenizer_from_pretrained(model: str, 
                              cache_dir: Optional[str] = None, 
                              force_download: bool = False, 
                              revision: Optional[str] = 'main', 
                              trust_remote_code: bool  = False) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(model, cache_dir=cache_dir, force_download=force_download, revision=revision, trust_remote_code=trust_remote_code)


def tokenize_text_with_attention(tokenizer: PreTrainedTokenizerBase, text: str) -> tuple[list[int], list[int]]:
    result = tokenizer(text)
    return result['input_ids'], result['attention_mask']

def tokenizer_text_as_ndarray(tokenizer: PreTrainedTokenizerBase, text: str, add_special_tokens: Optional[bool] = True) -> Buffer:
    result = tokenizer(text, return_tensors='np', return_attention_mask=False, add_special_tokens=add_special_tokens)
    return result['input_ids'][0]

def tokenizer_text_with_offsets(tokenizer: PreTrainedTokenizerBase, text: str, add_special_tokens: Optional[bool] = True) -> tuple[Buffer, Buffer]:
    result = tokenizer(text, return_tensors='np', return_offsets_mapping=True, return_attention_mask=False, add_special_tokens=add_special_tokens)
    input_ids = result['input_ids']
    offsets = result['offset_mapping']
    return input_ids[0], offsets[0]

def tokenizer_decode(tokenizer: PreTrainedTokenizerBase, input_ids: list[int], skip_special_tokens: bool = False,
                     clean_up_tokenization_spaces: Optional[bool] = None) -> str:
    decoded = tokenizer.decode(input_ids, skip_special_tokens=skip_special_tokens, clean_up_tokenization_spaces=clean_up_tokenization_spaces)
    return decoded


def stream_text_generation_pipeline_with_template(pipeline: TextGenerationPipeline, 
                             messages: list[dict[str, str]],
                             max_length: Optional[int] = None,
                             max_new_tokens: Optional[int] = None,
                             min_length: Optional[int] = None,
                             min_new_tokens: Optional[int] = None,
                             stop_strings: Optional[list[str]] = None,
                             temperature: Optional[float] = 1.0,
                             top_k: Optional[int] = 50,
                             top_p: Optional[float] = 1.0,
                             min_p: Optional[float] = None,
                            ) -> Generator[dict[str, str], None, None]:
    """
    Invoke a text generation pipeline with a chat template.
    Use pytorch for intermediate tensors (template -> generate)
    """
    # Apply template to messages
    r = pipeline(messages, max_length=max_length, max_new_tokens=max_new_tokens, min_length=min_length, 
                 min_new_tokens=min_new_tokens, stop=stop_strings, temperature=temperature, top_k=top_k, 
                 top_p=top_p, min_p=min_p)

    # TODO : stream messages
    for message in r[0]['generated_text']:
        yield message


def invoke_image_classification_pipeline(pipeline: Pipeline, 
                             image: str,
                             function_to_apply: Optional[str] = None,
                             top_k: Optional[int] = 5,
                             timeout: Optional[float] = None) -> list[dict[str, Any]]:
    """
    Invoke an image classification pipeline.
    """
    r = pipeline(image, top_k=top_k, timeout=timeout, function_to_apply=function_to_apply)
    return r

def invoke_image_classification_from_bytes(pipeline: Pipeline, 
                             data: bytes,
                             width: int,
                             height: int,
                             pixel_mode: str,
                             function_to_apply: Optional[str] = None,
                             top_k: int = 5,
                             timeout: Optional[float] = None) -> list[dict[str, Any]]:
    """
    Invoke an image classification pipeline.
    """
    image = Image.frombytes(pixel_mode, (width, height), data)
    r = pipeline(image, top_k=top_k, timeout=timeout, function_to_apply=function_to_apply)
    return r

def invoke_object_detection_pipeline(pipeline: Pipeline, 
                             image: str,
                             threshold: float = 0.5,
                             timeout: Optional[float] = None) -> Generator[tuple[str, float, tuple[int, int, int, int]], None, None]:
    """
    Invoke an object detection pipeline.
    """
    return ((r["label"], r["score"], ((box := r["box"])["xmin"], box["ymin"], box["xmax"], box["ymax"]))
            for r in pipeline(image, threshold=threshold, timeout=timeout))



def invoke_text_to_audio_pipeline(pipeline: Pipeline, 
                             text: str,
                             generate_kwargs: Optional[dict[str, Any]] = None) -> tuple[Buffer, int]:
    """
    Invoke a text-to-audio pipeline.
    """
    if not generate_kwargs:
        generate_kwargs = {}
    r = pipeline(text, **generate_kwargs)
    return r['audio'], r['sampling_rate']


def invoke_automatic_speech_recognition_pipeline(pipeline: Pipeline, audio: str) -> str:
    """
    Invoke an automatic speech recognition pipeline and return the result.

    Args:
        pipeline: The ASR pipeline object
        audio: A local file path or a URL
    Returns:
        The text detected
    """
    r = pipeline(audio, return_timestamps=False)
    return r['text']

def invoke_automatic_speech_recognition_pipeline_from_bytes(pipeline: Pipeline, audio: bytes) -> str:
    """
    Invoke an automatic speech recognition pipeline and return the result.
    
    Args:
        pipeline: The ASR pipeline object
        audio: The bytes of the audio file
    Returns:
        The text detected
    """
    r = pipeline(audio, return_timestamps=False)
    return r['text']


def invoke_text_to_image_pipeline(pipeline, 
                                text: str,
                                num_inference_steps: Optional[int] = 50,
                                guidance_scale: Optional[float] = 7.5,
                                height: Optional[int] = 512,
                                width: Optional[int] = 512) -> Buffer:
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
    # For diffusers pipelines
    if hasattr(pipeline, '__call__') and hasattr(pipeline, 'unet'):
        # This is a diffusers pipeline
        # Remove deprecated callback_steps, use callback_on_step_end if needed
        result = pipeline(
            text, 
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width
            # callback_on_step_end can be added here if needed
        )
        # Diffusers returns a result with .images attribute
        image = result.images[0]
    else:
        # This is a transformers pipeline
        result = pipeline(text, 
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    height=height,
                    width=width)
        
        # Get the generated image (it should be a PIL Image)
        if hasattr(result, 'images') and len(result.images) > 0:
            image = result.images[0]
        else:
            image = result
    
    # Convert PIL Image to bytes
    from io import BytesIO
    import numpy as np
    
    # Convert PIL Image to numpy array and then to bytes
    if hasattr(image, 'save'):
        # It's a PIL Image
        buffer = BytesIO()
        image.save(buffer, format='PNG')
        return buffer.getvalue()
    else:
        # It might already be a numpy array or tensor
        if hasattr(image, 'numpy'):
            image = image.numpy()
        
        # Convert to uint8 if needed
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
            
        # Convert numpy array to PIL Image and then to bytes
        from PIL import Image as PILImage
        if len(image.shape) == 3 and image.shape[2] == 3:  # RGB
            pil_image = PILImage.fromarray(image, mode='RGB')
        else:
            pil_image = PILImage.fromarray(image)
            
        buffer = BytesIO()
        pil_image.save(buffer, format='PNG')
        return buffer.getvalue()
