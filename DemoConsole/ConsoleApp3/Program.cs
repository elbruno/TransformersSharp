using TransformersSharp;
using TransformersSharp.Pipelines;

Console.WriteLine("=== TransformersSharp Text-to-Image Generation (Console3) ===");
Console.WriteLine();

var model = "kandinsky-community/kandinsky-2-2-decoder";

Console.WriteLine($"Model: {model}");
Console.WriteLine("Prompt: A pixelated image of a beaver in Canada");
Console.WriteLine();

Console.WriteLine("Creating text-to-image pipeline...");
var pipeline = TextToImagePipeline.FromModel(model: model, device: "cuda");
Console.WriteLine("✅ Pipeline created successfully");
Console.WriteLine($"Using device: {pipeline.DeviceType}");
Console.WriteLine();

Console.WriteLine("Generating image...");
var result = pipeline.Generate(
    "A pixelated image of a beaver in Canada",
    numInferenceSteps: 30,
    guidanceScale: 7.5f,
    height: 512,
    width: 512);

Console.WriteLine("✅ Image generation completed");
Console.WriteLine();

Console.WriteLine("Saving image to Desktop...");
var desktopPath = Environment.GetFolderPath(Environment.SpecialFolder.Desktop);
var folderPath = Path.Combine(desktopPath, "TransformersSharpImages");
Directory.CreateDirectory(folderPath);
var timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss_fff");
var filename = $"image_{pipeline!.DeviceType}_{timestamp}.png";
var destinationPath = Path.Combine(folderPath, filename);
File.WriteAllBytes(destinationPath, result.ImageBytes);

Console.WriteLine($"✅ Image saved successfully to: {destinationPath}");
Console.WriteLine($"📷 Image size: {result.ImageBytes.Length} bytes");