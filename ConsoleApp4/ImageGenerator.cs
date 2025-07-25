using System.Diagnostics;
using TransformersSharp;
using TransformersSharp.Pipelines;

namespace ConsoleApp4;

/// <summary>
/// High-performance image generator for text-to-image generation with optimized pipeline management.
/// </summary>
public class ImageGenerator : IDisposable
{
    private TextToImagePipeline? _pipeline;
    private bool _disposed = false;
    private readonly string _model;
    private readonly string _device;
    private readonly ImageGenerationSettings _settings;

    /// <summary>
    /// Configuration settings for image generation.
    /// </summary>
    public class ImageGenerationSettings
    {
        public int NumInferenceSteps { get; set; } = 10;
        public float GuidanceScale { get; set; } = 7.5f;
        public int Height { get; set; } = 256;
        public int Width { get; set; } = 256;
        public string OutputFolder { get; set; } = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Desktop), "TransformersSharpImages");
    }

    /// <summary>
    /// Initializes a new instance of the ImageGenerator class.
    /// </summary>
    /// <param name="model">The model identifier</param>
    /// <param name="device">The device to run the model on</param>
    /// <param name="settings">Optional generation settings</param>
    public ImageGenerator(
        string model = "stabilityai/stable-diffusion-2-1-base", 
        string device = "cpu", 
        ImageGenerationSettings? settings = null)
    {
        _model = model;
        _device = device;
        _settings = settings ?? new ImageGenerationSettings();
        
        ValidateDeviceAndProvideGuidance();
        CreatePipeline();
    }

    /// <summary>
    /// Validates device availability and provides user guidance.
    /// </summary>
    private void ValidateDeviceAndProvideGuidance()
    {
        if (_device.ToLower() == "cuda" && !TransformerEnvironment.IsCudaAvailable())
        {
            Console.WriteLine($"⚠️  CUDA requested but not available. Using CPU instead.");
            Console.WriteLine("   To enable GPU acceleration, ensure you have:");
            Console.WriteLine("   - Compatible NVIDIA GPU");
            Console.WriteLine("   - NVIDIA drivers installed");
            Console.WriteLine("   - CUDA-enabled PyTorch (run TransformerEnvironment.InstallCudaPyTorch())");
            Console.WriteLine();
        }
    }

    /// <summary>
    /// Creates a fresh pipeline instance with proper resource management.
    /// </summary>
    private void CreatePipeline()
    {
        DisposePipeline();
        
        try
        {
            _pipeline = TextToImagePipeline.FromModel(
                model: _model,
                device: _device,
                silentDeviceFallback: true);
        }
        catch (Exception ex) when (IsCompatibilityError(ex))
        {
            throw new InvalidOperationException(CreateCompatibilityErrorMessage(ex), ex);
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException(CreateGeneralErrorMessage(ex), ex);
        }
    }

    /// <summary>
    /// Checks if the exception indicates a package compatibility issue.
    /// </summary>
    private static bool IsCompatibilityError(Exception ex)
    {
        var message = ex.Message;
        return message.Contains("DLL load failed") || 
               message.Contains("diffusers") || 
               message.Contains("_C") ||
               message.Contains("xFormers") ||
               message.Contains("package compatibility");
    }

    /// <summary>
    /// Creates a detailed error message for compatibility issues.
    /// </summary>
    private string CreateCompatibilityErrorMessage(Exception ex)
    {
        return $@"
❌ Text-to-image pipeline creation failed due to package compatibility issues.

PROBLEM: The current PyTorch and diffusers installation has compatibility issues.
This commonly occurs when:
- CPU-only PyTorch is mixed with CUDA-compiled extensions
- Package versions are incompatible
- Missing system dependencies (Visual C++ Redistributables on Windows)

CURRENT SETUP ISSUES:
{ex.Message}

SOLUTIONS:
🔧 Option 1 - Reinstall compatible packages (Recommended):
1. pip uninstall torch torchvision torchaudio diffusers xformers -y
2. pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
3. pip install diffusers --no-deps && pip install safetensors accelerate
4. Restart your application

🔧 Option 2 - Install CUDA version (if you have NVIDIA GPU):
1. Ensure NVIDIA drivers are installed
2. Run: TransformerEnvironment.InstallCudaPyTorch()
3. Restart your application

For more help, see: https://pytorch.org/get-started/locally/";
    }

    /// <summary>
    /// Creates a detailed error message for general pipeline creation issues.
    /// </summary>
    private string CreateGeneralErrorMessage(Exception ex)
    {
        return $@"
❌ Failed to create text-to-image pipeline.

ERROR: {ex.Message}

POSSIBLE CAUSES:
- Network connectivity issues during model download
- Insufficient disk space (models can be several GB)
- Model unavailable or access restricted
- Invalid model name: '{_model}'

SOLUTIONS:
1. Check your internet connection
2. Ensure sufficient disk space (at least 5GB free)
3. Try a different model name
4. Verify the model exists at: https://huggingface.co/{_model}

If the problem persists, try using a well-known model like:
'stabilityai/stable-diffusion-2-1-base' or 'runwayml/stable-diffusion-v1-5'";
    }

    /// <summary>
    /// Disposes the current pipeline and forces garbage collection.
    /// </summary>
    private void DisposePipeline()
    {
        if (_pipeline is IDisposable disposablePipeline)
        {
            disposablePipeline.Dispose();
        }
        _pipeline = null;
        
        // Force garbage collection to ensure Python objects are released
        GC.Collect();
        GC.WaitForPendingFinalizers();
    }

    /// <summary>
    /// Refreshes the pipeline by disposing the current instance and creating a new one.
    /// </summary>
    public void RefreshPipeline()
    {
        if (_disposed) throw new ObjectDisposedException(nameof(ImageGenerator));
        CreatePipeline();
    }

    /// <summary>
    /// Generates an image from the specified text prompt.
    /// </summary>
    /// <param name="prompt">The text prompt for image generation</param>
    /// <param name="desiredFolder">Optional custom output folder</param>
    /// <param name="refreshPipeline">Whether to refresh the pipeline before generation</param>
    /// <returns>Image generation result with timing and file information</returns>
    public ImageGenerationResult GenerateImage(string prompt, string? desiredFolder = null, bool refreshPipeline = true)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(ImageGenerator));

        if (refreshPipeline)
        {
            RefreshPipeline();
        }

        EnsureOutputFolder(desiredFolder);
        
        var stopwatch = Stopwatch.StartNew();
        var result = _pipeline!.Generate(
            prompt,
            numInferenceSteps: _settings.NumInferenceSteps,
            guidanceScale: _settings.GuidanceScale,
            height: _settings.Height,
            width: _settings.Width);
        stopwatch.Stop();

        var filepath = SaveImageToFile(result.ImageBytes, desiredFolder);

        return new ImageGenerationResult
        {
            Prompt = prompt,
            FileGenerated = filepath,
            TimeTakenSeconds = stopwatch.Elapsed.TotalSeconds,
            DeviceType = _pipeline.DeviceType
        };
    }

    /// <summary>
    /// Ensures the output folder exists.
    /// </summary>
    private void EnsureOutputFolder(string? desiredFolder)
    {
        var folder = desiredFolder ?? _settings.OutputFolder;
        if (!Directory.Exists(folder))
        {
            Directory.CreateDirectory(folder);
        }
    }

    /// <summary>
    /// Saves the generated image to a file with a timestamped filename.
    /// </summary>
    private string SaveImageToFile(byte[] imageBytes, string? desiredFolder)
    {
        var folder = desiredFolder ?? _settings.OutputFolder;
        var timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss_fff");
        var filename = $"image_{_pipeline!.DeviceType}_{timestamp}.png";
        var filepath = Path.Combine(folder, filename);
        
        File.WriteAllBytes(filepath, imageBytes);
        return filepath;
    }

    /// <summary>
    /// Disposes of the image generator and its resources.
    /// </summary>
    public void Dispose()
    {
        if (!_disposed)
        {
            DisposePipeline();
            _disposed = true;
        }
    }
}
