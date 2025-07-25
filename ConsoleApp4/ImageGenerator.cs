using System.Diagnostics;
using TransformersSharp;
using TransformersSharp.Pipelines;

namespace ConsoleApp4;

/// <summary>
/// Image generator that creates images from text prompts using text-to-image pipelines.
/// </summary>
public class ImageGenerator : IDisposable
{
    private TextToImagePipeline? _pipeline;
    private bool _disposed = false;
    private readonly string _model;
    private readonly string _device;

    /// <summary>
    /// Initializes a new instance of the ImageGenerator class.
    /// </summary>
    /// <param name="model">The model identifier (e.g., "kandinsky-community/kandinsky-2-2-decoder")</param>
    /// <param name="device">The device to run the model on (e.g., "cpu", "cuda")</param>
    public ImageGenerator(string model = "stabilityai/stable-diffusion-2-1-base", string device = "cpu")
    {
        _model = model;
        _device = device;
        
        // Check CUDA availability and provide user feedback
        if (device.ToLower() == "cuda" && !TransformerEnvironment.IsCudaAvailable())
        {
            Console.WriteLine($"⚠️  CUDA requested but not available. Using CPU instead.");
            Console.WriteLine("   To enable GPU acceleration, ensure you have:");
            Console.WriteLine("   - Compatible NVIDIA GPU");
            Console.WriteLine("   - NVIDIA drivers installed");
            Console.WriteLine("   - CUDA-enabled PyTorch (run TransformerEnvironment.InstallCudaPyTorch())");
            Console.WriteLine();
        }
        
        CreatePipeline();
    }

    /// <summary>
    /// Creates a fresh pipeline instance, disposing any existing pipeline objects.
    /// This ensures no reuse of in-memory pipeline objects while preserving downloaded model files.
    /// </summary>
    private void CreatePipeline()
    {
        // Dispose existing pipeline if it exists
        if (_pipeline is IDisposable disposablePipeline)
        {
            disposablePipeline.Dispose();
        }
        _pipeline = null;

        // Force garbage collection to ensure Python objects are released
        GC.Collect();
        GC.WaitForPendingFinalizers();

        try
        {
            // Create new pipeline instance with silent device fallback
            _pipeline = TextToImagePipeline.FromModel(
                model: _model,
                device: _device,
                silentDeviceFallback: true);
        }
        catch (Exception ex) when (ex.Message.Contains("DLL load failed") || 
                                  ex.Message.Contains("diffusers") || 
                                  ex.Message.Contains("_C") ||
                                  ex.Message.Contains("xFormers") ||
                                  ex.Message.Contains("package compatibility"))
        {
            // Handle diffusers/PyTorch compatibility issues
            throw new InvalidOperationException($@"
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
1. Uninstall conflicting packages:
   pip uninstall torch torchvision torchaudio diffusers xformers -y

2. Install CPU-compatible versions:
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   pip install diffusers --no-deps
   pip install safetensors accelerate

3. Restart your application

🔧 Option 2 - Install CUDA version (if you have NVIDIA GPU):
1. Ensure NVIDIA drivers are installed
2. Run: TransformerEnvironment.InstallCudaPyTorch()
3. Restart your application

🔧 Option 3 - Use a different model:
Try a lighter model that may have better compatibility:
- stabilityai/stable-diffusion-2-1-base (current)
- runwayml/stable-diffusion-v1-5
- kandinsky-community/kandinsky-2-1

For more help, see: https://pytorch.org/get-started/locally/
", ex);
        }
        catch (Exception ex)
        {
            // Handle other pipeline creation errors
            throw new InvalidOperationException($@"
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
'stabilityai/stable-diffusion-2-1-base' or 'runwayml/stable-diffusion-v1-5'
", ex);
        }
    }

    /// <summary>
    /// Refreshes the pipeline by disposing the current instance and creating a new one.
    /// This ensures fresh pipeline objects without re-downloading the model files.
    /// </summary>
    public void RefreshPipeline()
    {
        if (_disposed) throw new ObjectDisposedException(nameof(ImageGenerator));
        CreatePipeline();
    }

    public ImageGenerationResult GenerateImage(string prompt, string? desiredFolder = null, bool refreshPipeline = true)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(ImageGenerator));

        // Refresh pipeline before generation to ensure fresh objects
        if (refreshPipeline)
        {
            RefreshPipeline();
        }

        var folder = desiredFolder ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Desktop), "TransformersSharpImages");
        if (!Directory.Exists(folder))
            Directory.CreateDirectory(folder);

        var stopwatch = Stopwatch.StartNew();
        var result = _pipeline!.Generate(
            prompt,
            numInferenceSteps: 10,
            guidanceScale: 7.5f,
            height: 256,
            width: 256);

        stopwatch.Stop();

        var timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss_fff");
        var filename = $"image_{_pipeline.DeviceType}_{timestamp}.png";
        var filepath = Path.Combine(folder, filename);
        File.WriteAllBytes(filepath, result.ImageBytes);

        return new ImageGenerationResult
        {
            Prompt = prompt,
            FileGenerated = filepath,
            TimeTakenSeconds = stopwatch.Elapsed.TotalSeconds,
            DeviceType = _pipeline.DeviceType
        };
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            if (_pipeline is IDisposable disposablePipeline)
            {
                disposablePipeline.Dispose();
            }
            _pipeline = null;
            _disposed = true;
        }
    }
}
