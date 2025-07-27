using TransformersSharp;
using TransformersSharp.Pipelines;

Console.WriteLine("=== TransformersSharp Text-to-Image Generation - FLUX.1-dev (Console4) ===");
Console.WriteLine();

var model = "black-forest-labs/FLUX.1-dev";

Console.WriteLine($"Model: {model}");
Console.WriteLine("Prompt: A majestic dragon flying over a medieval castle at sunset");
Console.WriteLine();

Console.WriteLine("Creating text-to-image pipeline...");
var pipeline = TextToImagePipeline.FromModel(model: model, device: "cuda");
Console.WriteLine("✅ Pipeline created successfully");
Console.WriteLine($"Using device: {pipeline.DeviceType}");
Console.WriteLine();

Console.WriteLine("Generating image...");
var result = pipeline.Generate(
    "A majestic dragon flying over a medieval castle at sunset",
    numInferenceSteps: 20,
    guidanceScale: 3.5f,
    height: 1024,
    width: 1024);

Console.WriteLine("✅ Image generation completed");
Console.WriteLine();

Console.WriteLine("Saving image to Desktop...");
var desktopPath = Environment.GetFolderPath(Environment.SpecialFolder.Desktop);
var folderPath = Path.Combine(desktopPath, "TransformersSharpImages");
Directory.CreateDirectory(folderPath);
var timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss_fff");
var filename = $"image_flux_{pipeline!.DeviceType}_{timestamp}.png";
var destinationPath = Path.Combine(folderPath, filename);
File.WriteAllBytes(destinationPath, result.ImageBytes);

Console.WriteLine($"✅ Image saved successfully to: {destinationPath}");
Console.WriteLine($"📷 Image size: {result.ImageBytes.Length} bytes");