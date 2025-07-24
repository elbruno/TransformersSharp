using System.Diagnostics;
using TransformersSharp.Pipelines;

namespace ConsoleApp4;

public static class ImageGenerator
{
    public static ImageGenerationResult GenerateImage(string device, string prompt, string? desiredFolder = null, string model = "kandinsky-community/kandinsky-2-2-decoder")
    {
        var folder = desiredFolder ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Desktop), "TransformersSharpImages");
        if (!Directory.Exists(folder))
            Directory.CreateDirectory(folder);

        var stopwatch = Stopwatch.StartNew();
        var pipeline = TextToImagePipeline.FromModel(
            model,
            device: device);

        var result = pipeline.Generate(
            prompt,
            numInferenceSteps: 10,
            guidanceScale: 7.5f,
            height: 256,
            width: 256);

        stopwatch.Stop();

        var timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss_fff");
        var filename = $"image_{device}_{timestamp}.png";
        var filepath = Path.Combine(folder, filename);
        File.WriteAllBytes(filepath, result.ImageBytes);

        return new ImageGenerationResult
        {
            Prompt = prompt,
            FileGenerated = filepath,
            TimeTakenSeconds = stopwatch.Elapsed.TotalSeconds,
            DeviceType = device
        };
    }
}
