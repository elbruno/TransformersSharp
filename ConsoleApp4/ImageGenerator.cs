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
    public ImageGenerator(string model = "kandinsky-community/kandinsky-2-2-decoder", string device = "cpu")
    {
        _model = model;
        _device = device;
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

        // Create new pipeline instance
        _pipeline = TextToImagePipeline.FromModel(
            model: _model,
            device: _device);
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
