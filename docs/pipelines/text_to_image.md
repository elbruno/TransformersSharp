# Text-to-Image Pipeline

The Text-to-Image pipeline in TransformersSharp enables you to generate high-quality images from text descriptions using state-of-the-art diffusion models. This pipeline supports multiple model architectures and provides seamless integration between C# and Python-based diffusion models.

## Supported Models

TransformersSharp automatically detects and configures the appropriate pipeline for different model types:

### Kandinsky 2.2 (Default)
- **Model**: `kandinsky-community/kandinsky-2-2-decoder`
- **Pipeline**: KandinskyV22Pipeline
- **Use Case**: Artistic style generation with unique aesthetic, now the default for balanced performance
- **Default Size**: 256x256 pixels (optimized for performance, can be customized)

### FLUX.1-dev
- **Model**: `black-forest-labs/FLUX.1-dev`
- **Pipeline**: FluxPipeline (with AutoPipeline fallback)
- **Use Case**: State-of-the-art text-to-image generation with high quality and prompt adherence
- **Default Size**: 256x256 pixels (can be increased to 1024x1024 for maximum quality)
- **Note**: Requires HuggingFace token and model access approval
- **Authentication**: Gated model requiring HuggingFace login
- **Advanced Features**: Supports max_sequence_length, seed for reproducibility, and CPU offloading

## Architecture

The Text-to-Image pipeline is built on TransformersSharp's modular Python architecture:

- **Core Pipeline**: Implemented in `transformers_wrapper.py` for model management and generation
- **Device Management**: Handled by `device_manager.py` for CUDA/CPU optimization
- **Image Processing**: Managed by `image_utils.py` for format conversion and C# interop
- **System Information**: Provided by `system_info.py` for performance analysis

This modular design ensures better maintainability and allows for independent optimization of each component.

## Basic Usage

### Simple Image Generation

```csharp
using TransformersSharp.Pipelines;

// Create pipeline with default model (Kandinsky 2.2)
var pipeline = TextToImagePipeline.FromModel();

// Generate image
var result = pipeline.Generate("A beautiful sunset over mountains");

// Save image
File.WriteAllBytes("sunset.png", result.ImageBytes);
```

### Advanced FLUX.1-dev Configuration

FLUX.1-dev supports additional parameters for enhanced control and reproducibility:

```csharp
using TransformersSharp;
using TransformersSharp.Pipelines;
using TransformersSharp.Models;

// Create pipeline with specific model and device
var pipeline = TextToImagePipeline.FromModel(
    model: "black-forest-labs/FLUX.1-dev",
    device: "cuda",  // Use GPU if available
    silentDeviceFallback: true,  // Silently fall back to CPU if CUDA unavailable
    huggingFaceToken: "your_hf_token_here"  // Required for gated models
);

// Generate with FLUX-optimized parameters
var result = pipeline.Generate(
    prompt: "A futuristic cityscape with flying cars",
    numInferenceSteps: 50,          // FLUX recommended steps (vs 20 in old docs)
    guidanceScale: 3.5f,            // FLUX optimized guidance scale
    height: 256,                    // Default optimized size (can go up to 1024)
    width: 256,                     // Default optimized size (can go up to 1024)
    maxSequenceLength: 512,         // FLUX-specific parameter for prompt processing
    seed: 42,                       // For reproducible results
    enableModelCpuOffload: true     // Memory optimization
);

Console.WriteLine($"Generated {result.Width}x{result.Height} image");
```

### HuggingFace Authentication

For gated models like FLUX.1-dev, you need to authenticate with HuggingFace. TransformersSharp supports multiple authentication methods:

```csharp
// Method 1: Pass token directly to pipeline
var pipeline = TextToImagePipeline.FromModel(
    model: "black-forest-labs/FLUX.1-dev",
    huggingFaceToken: "your_hf_token_here"
);

// Method 2: Global authentication (applies to all subsequent operations)
TransformerEnvironment.Login("your_hf_token_here");
var pipeline = TextToImagePipeline.FromModel("black-forest-labs/FLUX.1-dev");

// Method 3: Environment variable (recommended for security)
// Set HF_TOKEN or HUGGINGFACE_TOKEN environment variable
// The pipeline will automatically use it when provided

// Method 4: User Secrets (recommended for development)
// Use dotnet user-secrets for secure local development
// dotnet user-secrets set "HF_TOKEN" "your_token_here"
```

#### Getting a HuggingFace Token

1. Visit [HuggingFace Token Settings](https://huggingface.co/settings/tokens)
2. Create a new token with "Read" permissions
3. Request access to gated models at their model pages (e.g., [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev))
4. Wait for approval (usually automatic for FLUX.1-dev)

#### User Secrets for Development

For secure local development, use .NET User Secrets:

```bash
# Set token using User Secrets (recommended for development)
dotnet user-secrets set "HF_TOKEN" "your_token_here"

# Or use the alternative key
dotnet user-secrets set "HUGGINGFACE_TOKEN" "your_token_here"
```

Then in your application:

```csharp
using Microsoft.Extensions.Configuration;

// Build configuration to read from multiple sources
var config = new ConfigurationBuilder()
    .AddEnvironmentVariables()
    .AddUserSecrets<Program>()  // Add user secrets support
    .Build();

// Check for token from multiple sources
var hfToken = Environment.GetEnvironmentVariable("HF_TOKEN") ?? 
              Environment.GetEnvironmentVariable("HUGGINGFACE_TOKEN") ??
              config["HF_TOKEN"] ??
              config["HUGGINGFACE_TOKEN"];

if (!string.IsNullOrEmpty(hfToken))
{
    var pipeline = TextToImagePipeline.FromModel(
        "black-forest-labs/FLUX.1-dev", 
        huggingFaceToken: hfToken
    );
}
```

### Using ImageGenerator (Recommended for Benchmarking)

The `ImageGenerator` class in `Demo10_text-to-image_benchmark` provides a high-level interface with automatic pipeline management and performance testing:

```csharp
using Demo10_text_to_image_benchmark;

// Create generator with custom settings optimized for performance
var settings = new ImageGenerationSettings
{
    NumInferenceSteps = 30,
    GuidanceScale = 7.5f,
    Height = 256,          // Optimized default size
    Width = 256,           // Optimized default size
    OutputFolder = @"C:\MyImages"
};

using var generator = new ImageGenerator(
    model: "kandinsky-community/kandinsky-2-2-decoder",  // Default model
    device: "cuda",
    settings: settings
);

// Generate and save image automatically
var result = generator.GenerateImage("A majestic eagle soaring through clouds");

Console.WriteLine($"Image saved to: {result.FileGenerated}");
Console.WriteLine($"Generation took: {result.TimeTakenSeconds:F2} seconds");
Console.WriteLine($"Used device: {result.DeviceType}");
```

## Model Selection Guide

### Choosing the Right Model

| Model | Best For | Speed | Quality | Memory |
|-------|----------|-------|---------|---------|
| Kandinsky 2.2 | Artistic, stylized images (Default) | Medium | Good | Medium |
| FLUX.1-dev | State-of-the-art, photorealistic images | Medium | Excellent | High |

### Model-Specific Examples

```csharp
// Kandinsky 2.2 - Artistic style (Default)
var kandinsky = TextToImagePipeline.FromModel("kandinsky-community/kandinsky-2-2-decoder");
var art = kandinsky.Generate("An abstract painting of music in vibrant colors");

// FLUX.1-dev - State-of-the-art quality (requires HuggingFace token)
var hfToken = Environment.GetEnvironmentVariable("HF_TOKEN");
var flux = TextToImagePipeline.FromModel("black-forest-labs/FLUX.1-dev", huggingFaceToken: hfToken);
var photo = flux.Generate("A photorealistic portrait of a wise old wizard", height: 1024, width: 1024);
```

## Performance Optimization

### Device Selection

```csharp
// Check CUDA availability
if (TransformerEnvironment.IsCudaAvailable())
{
    Console.WriteLine("Using GPU acceleration");
    var pipeline = TextToImagePipeline.FromModel(device: "cuda");
}
else
{
    Console.WriteLine("Using CPU (slower but still works)");
    var pipeline = TextToImagePipeline.FromModel(device: "cpu");
}
```

### Performance Optimization

All TransformersSharp demos now use 256x256 pixel images by default for optimal performance. You can customize this for your specific use case:

```csharp
// Fast generation (default optimized settings)
var quick = pipeline.Generate(
    prompt: "A simple sketch of a house",
    numInferenceSteps: 20,      // Optimized for speed
    guidanceScale: 3.5f,        // FLUX-optimized guidance
    height: 256,                // Default optimized size
    width: 256,                 // Default optimized size
    seed: 42                    // Reproducible results
);

// High quality generation (larger output)
var detailed = pipeline.Generate(
    prompt: "A highly detailed oil painting of a Victorian mansion",
    numInferenceSteps: 50,      // More steps for FLUX quality
    guidanceScale: 3.5f,        // FLUX-optimized guidance
    height: 1024,               // Increase for maximum detail
    width: 1024,                // Increase for maximum detail
    maxSequenceLength: 512,     // FLUX-specific prompt processing
    enableModelCpuOffload: true // Memory optimization
);
```

## Prompt Engineering Tips

### Effective Prompting

```csharp
// Good: Specific and descriptive
var result1 = pipeline.Generate(
    "A golden retriever puppy playing in a sunny meadow with wildflowers, " +
    "soft natural lighting, professional photography, high detail"
);

// Good: Style-specific
var result2 = pipeline.Generate(
    "A cyberpunk cityscape at night, neon lights, digital art style, " +
    "highly detailed, trending on artstation"
);

// Good: Artist-inspired
var result3 = pipeline.Generate(
    "A landscape painting in the style of Van Gogh, swirling clouds, " +
    "vibrant colors, impressionist brushstrokes"
);
```

### Negative Prompts (Advanced)

Some models support negative prompts to exclude unwanted elements:

```csharp
// Note: Negative prompts may require model-specific implementation
var result = pipeline.Generate(
    "A beautiful woman portrait",
    // Negative prompt would go here if supported by the model
    numInferenceSteps: 50,
    guidanceScale: 7.5f
);
```

## Error Handling

### Robust Pipeline Creation

```csharp
try
{
    var pipeline = TextToImagePipeline.FromModel(
        model: "kandinsky-community/kandinsky-2-2-decoder",
        device: "cuda",
        silentDeviceFallback: true
    );
    
    var result = pipeline.Generate("A peaceful garden scene");
    Console.WriteLine("Image generated successfully!");
}
catch (InvalidOperationException ex) when (ex.Message.Contains("compatibility"))
{
    Console.WriteLine("Package compatibility issue detected:");
    Console.WriteLine(ex.Message);
    // Handle specific compatibility errors
}
catch (Exception ex)
{
    Console.WriteLine($"Image generation failed: {ex.Message}");
    // Handle other errors
}
```

### Device Fallback

```csharp
// Automatic fallback with user notification
var generator = new ImageGenerator(
    model: "stable-diffusion-v1-5/stable-diffusion-v1-5",
    device: "cuda"  // Will automatically fall back to CPU if CUDA unavailable
);

// Manual device checking
if (!TransformerEnvironment.IsCudaAvailable())
{
    Console.WriteLine("CUDA not available. Using CPU mode.");
    Console.WriteLine("For GPU acceleration:");
    Console.WriteLine("1. Install NVIDIA drivers");
    Console.WriteLine("2. Run: TransformerEnvironment.InstallCudaPyTorch()");
}
```

## System Requirements

### Minimum Requirements
- **RAM**: 8GB (16GB recommended)
- **Storage**: 5GB free space for model downloads
- **CPU**: Multi-core processor (4+ cores recommended)

### GPU Requirements (Optional)
- **NVIDIA GPU** with CUDA Compute Capability 6.0+
- **VRAM**: 4GB minimum (8GB+ recommended for high-resolution images)
- **CUDA Drivers**: Version 11.0 or later

### Installation

```csharp
// Install CPU-only PyTorch (works on any system)
TransformerEnvironment.InstallCpuPyTorch();

// Install CUDA PyTorch (requires NVIDIA GPU)
TransformerEnvironment.InstallCudaPyTorch();

// Verify installation
var deviceInfo = TransformerEnvironment.GetDeviceInfo();
Console.WriteLine($"CUDA Available: {deviceInfo.CudaAvailable}");
```

## Troubleshooting

### Common Issues

1. **DLL Load Failed Error**
   ```
   Solution: Reinstall compatible PyTorch and diffusers versions
   pip uninstall torch torchvision diffusers -y
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   pip install diffusers[torch]
   ```

2. **Out of Memory (CUDA)**
   ```csharp
   // Use optimized defaults or reduce image size further
   var result = pipeline.Generate(
       "Your prompt",
       height: 256,  // Optimized default size
       width: 256,
       numInferenceSteps: 20,     // Balanced steps
       enableModelCpuOffload: true // Enable CPU offloading for memory
   );
   ```

3. **Model Download Issues**
   ```
   Solution: Check internet connection and disk space
   Models are downloaded to: %USERPROFILE%\.cache\huggingface\transformers
   ```

4. **Authentication Issues (Gated Models)**
   ```
   Error: "Access to model X is restricted"
   Solution: 
   1. Get HuggingFace token from https://huggingface.co/settings/tokens
   2. Request access to the specific model
   3. Set HF_TOKEN environment variable or pass token to pipeline
   ```
5. **Slow Generation on CPU**
   ```
   This is normal. Consider:
   - Using smaller image sizes (256x256 instead of 512x512)
   - Reducing inference steps (10-20 instead of 50)
   - Installing CUDA PyTorch for GPU acceleration
   ```

## Advanced Usage

### Batch Generation

```csharp
var prompts = new[]
{
    "A cat in a spacesuit",
    "A robot gardener watering flowers",
    "A magical forest with glowing mushrooms"
};

using var generator = new ImageGenerator();

foreach (var prompt in prompts)
{
    var result = generator.GenerateImage(prompt, refreshPipeline: false);
    Console.WriteLine($"Generated: {result.FileGenerated}");
}
```

### Custom Output Handling

```csharp
var pipeline = TextToImagePipeline.FromModel();
var result = pipeline.Generate("A sunset over the ocean");

// Convert to different formats
using var stream = new MemoryStream(result.ImageBytes);
using var image = System.Drawing.Image.FromStream(stream);

// Save as JPEG with compression
image.Save("output.jpg", System.Drawing.Imaging.ImageFormat.Jpeg);

// Get image properties
Console.WriteLine($"Image size: {result.Width}x{result.Height}");
Console.WriteLine($"Data size: {result.ImageBytes.Length} bytes");
```

## See Also

- [Pipeline Overview](index.md)
- [System Requirements](../index.md)
- [CUDA Installation Guide](../CUDA_FIX_SUMMARY.md)
- [Performance Optimization Tips](../index.md)