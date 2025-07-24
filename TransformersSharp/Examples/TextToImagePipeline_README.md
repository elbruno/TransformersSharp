# TextToImagePipeline

The `TextToImagePipeline` class provides a C# interface for generating images from text prompts using state-of-the-art text-to-image models.

## Overview

This pipeline integrates with the Hugging Face ecosystem, supporting both transformers and diffusers libraries for text-to-image generation. It automatically uses the appropriate backend (diffusers for text-to-image tasks) while maintaining a consistent C# interface.

## Features

- **Text-to-Image Generation**: Convert text prompts into high-quality images
- **Configurable Parameters**: Control image dimensions, inference steps, and guidance scale
- **Multiple Model Support**: Compatible with various text-to-image models from Hugging Face
- **PNG Output**: Generated images are returned as PNG-formatted byte arrays
- **Consistent API**: Follows the same pattern as other pipelines in TransformersSharp

## Usage

### Basic Usage

```csharp
using TransformersSharp.Pipelines;

// Create the pipeline with a specific model
var pipeline = TextToImagePipeline.FromModel(
    "kandinsky-community/kandinsky-2-2-decoder", 
    trustRemoteCode: true);

// Generate an image
var result = pipeline.Generate("A beautiful sunset over mountains");

// Save the image
File.WriteAllBytes("generated_image.png", result.ImageBytes);
```

### Advanced Usage

```csharp
// Generate with custom parameters
var result = pipeline.Generate(
    prompt: "A futuristic cityscape at night",
    numInferenceSteps: 30,
    guidanceScale: 8.0f,
    height: 768,
    width: 768
);

Console.WriteLine($"Generated image: {result.Width}x{result.Height}");
Console.WriteLine($"File size: {result.ImageBytes.Length} bytes");
```

## Supported Models

The pipeline works with text-to-image models from Hugging Face, including:

- **Kandinsky 2.2**: `kandinsky-community/kandinsky-2-2-decoder`
- **Stable Diffusion models**: Various SD models that support diffusers
- **Custom models**: Any text-to-image model compatible with diffusers

## Parameters

### Generation Parameters

- `prompt` (string): The text description of the image to generate
- `numInferenceSteps` (int): Number of denoising steps (default: 50)
- `guidanceScale` (float): Guidance scale for classifier-free guidance (default: 7.5)
- `height` (int): Height of the generated image in pixels (default: 512)
- `width` (int): Width of the generated image in pixels (default: 512)

### Model Parameters

- `model` (string): Hugging Face model identifier
- `torchDtype` (TorchDtype?): Data type for model weights
- `device` (string?): Device to run the model on ("cuda", "cpu", etc.)
- `trustRemoteCode` (bool): Whether to trust remote code in the model

## Output

The `Generate` method returns an `ImageGenerationResult` struct containing:

- `ImageBytes`: The generated image as a PNG-formatted byte array
- `Width`: The width of the generated image
- `Height`: The height of the generated image

## Requirements

The pipeline requires the following Python packages:
- `transformers`
- `diffusers`
- `torch`
- `pillow`

These are automatically installed when using the TransformersSharp environment.

## Error Handling

```csharp
try
{
    var pipeline = TextToImagePipeline.FromModel("model-name");
    var result = pipeline.Generate("text prompt");
    // Process result...
}
catch (Exception ex)
{
    Console.WriteLine($"Error: {ex.Message}");
}
```

## Performance Tips

1. **Use smaller inference steps** for faster generation (e.g., 20-30 steps)
2. **Reduce image dimensions** for quicker results (e.g., 256x256 or 512x512)
3. **Use appropriate hardware** (GPU recommended for larger models)
4. **Cache models** by reusing the same pipeline instance

## Example Applications

- **Content Creation**: Generate illustrations for articles or stories
- **Prototyping**: Create visual mockups from descriptions
- **Art Generation**: Produce artistic images from creative prompts
- **Educational Tools**: Visualize concepts described in text