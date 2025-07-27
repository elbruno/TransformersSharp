using System.Diagnostics;
using TransformersSharp;
using TransformersSharp.Pipelines;

namespace Demo10_text_to_image_benchmark;

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
        public int NumInferenceSteps { get; set; } = 30;
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
        string model = "kandinsky-community/kandinsky-2-2-decoder",
        string device = "cpu",
        ImageGenerationSettings? settings = null)
    {
        _model = model;
        _device = device;
        _settings = settings ?? new ImageGenerationSettings();

        CreatePipeline();
    }

    /// <summary>
    /// Creates a fresh pipeline instance with proper resource management.
    /// </summary>
    private void CreatePipeline()
    {
        DisposePipeline();

        if (_device == "cuda")
        {
            _pipeline = TextToImagePipeline.FromModel(
                        model: _model,
                        device: _device,
                        torchDtype: TorchDtype.Float16);
        }
        else
        {
            _pipeline = TextToImagePipeline.FromModel(
            model: _model,
            device: _device);
        }
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
        var safeDeviceType = _pipeline!.DeviceType.Replace(":", "-").Replace("/", "-");
        var filename = $"image_{safeDeviceType}_{timestamp}.png";
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
