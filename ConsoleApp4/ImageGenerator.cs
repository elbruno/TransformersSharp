using System.Diagnostics;
using TransformersSharp;
using TransformersSharp.Pipelines;

namespace ConsoleApp4;

public class ImageGenerator : IDisposable
{
    private TextToImagePipeline? _pipeline;
    private bool _disposed = false;

    public ImageGenerator(string model = "kandinsky-community/kandinsky-2-2-decoder", string device = "cpu")
    {
        _pipeline = TextToImagePipeline.FromModel(
            model: model, 
            torchDtype: TorchDtype.BFloat16,
            device: device);
    }

    public ImageGenerationResult GenerateImage(string prompt, string? desiredFolder = null)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(ImageGenerator));

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
