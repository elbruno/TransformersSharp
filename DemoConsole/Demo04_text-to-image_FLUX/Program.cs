using TransformersSharp;
using TransformersSharp.Pipelines;

Console.WriteLine("=== TransformersSharp Text-to-Image Generation - FLUX.1-dev (Console4) ===");
Console.WriteLine();

var model = "black-forest-labs/FLUX.1-dev";

Console.WriteLine($"Model: {model}");
Console.WriteLine("Prompt: A majestic dragon flying over a medieval castle at sunset");
Console.WriteLine();

// Check for HuggingFace token
var hfToken = Environment.GetEnvironmentVariable("HF_TOKEN") ?? Environment.GetEnvironmentVariable("HUGGINGFACE_TOKEN");

if (string.IsNullOrEmpty(hfToken))
{
    Console.WriteLine("⚠️  FLUX.1-dev requires HuggingFace authentication.");
    Console.WriteLine("   Please set one of these environment variables with your HuggingFace token:");
    Console.WriteLine("   - HF_TOKEN=your_token_here");
    Console.WriteLine("   - HUGGINGFACE_TOKEN=your_token_here");
    Console.WriteLine();
    Console.WriteLine("   You can get a token from: https://huggingface.co/settings/tokens");
    Console.WriteLine("   You also need to request access to FLUX.1-dev at: https://huggingface.co/black-forest-labs/FLUX.1-dev");
    Console.WriteLine();
    Console.WriteLine("Press any key to exit...");
    Console.ReadKey();
    return;
}

Console.WriteLine("Creating text-to-image pipeline with authentication...");
try
{
    var pipeline = TextToImagePipeline.FromModel(
        model: model, 
        device: "cuda", 
        huggingFaceToken: hfToken);
    
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
}
catch (Exception ex)
{
    Console.WriteLine($"❌ Error: {ex.Message}");
    Console.WriteLine();
    Console.WriteLine("Common solutions:");
    Console.WriteLine("1. Ensure you have access to FLUX.1-dev model on HuggingFace");
    Console.WriteLine("2. Verify your HuggingFace token is valid");
    Console.WriteLine("3. Check your internet connection");
    Console.WriteLine("4. Ensure sufficient disk space for model downloads");
}