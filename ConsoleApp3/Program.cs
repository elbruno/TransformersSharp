using TransformersSharp;
using TransformersSharp.Pipelines;

Console.WriteLine("=== TransformersSharp Text-to-Image Generation (Console3) ===");
Console.WriteLine();

try
{
    // Check .NET version first
    var dotnetVersion = Environment.Version;
    Console.WriteLine($"Running on .NET {dotnetVersion}");
    
    if (dotnetVersion.Major < 9)
    {
        Console.WriteLine("⚠️  WARNING: This application requires .NET 9.0 or later.");
        Console.WriteLine("   Current version may cause compatibility issues.");
        Console.WriteLine("   Please install .NET 9.0 from https://dotnet.microsoft.com/download");
        Console.WriteLine();
    }

    // Test model options (uncomment as needed)
    //var model = "kandinsky-community/kandinsky-2-2-decoder";
    //var model = "sd-legacy/stable-diffusion-v1-5";
    var model = "stabilityai/stable-diffusion-2-1";

    Console.WriteLine($"Model: {model}");
    Console.WriteLine("Device: CUDA (with CPU fallback)");
    Console.WriteLine("Prompt: A pixelated image of a beaver in Canada");
    Console.WriteLine();

    Console.WriteLine("Creating text-to-image pipeline...");
    var pipeline = TextToImagePipeline.FromModel(
        model: model,
        device: "cuda");

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
    var destinationPath = Path.Combine(folderPath, "console3_image.png");
    File.WriteAllBytes(destinationPath, result.ImageBytes);

    Console.WriteLine($"✅ Image saved successfully to: {destinationPath}");
    Console.WriteLine($"📊 Final device used: {pipeline.DeviceType}");
    Console.WriteLine($"📷 Image size: {result.ImageBytes.Length} bytes");
}
catch (DllNotFoundException ex) when (ex.Message.Contains("python") || ex.Message.Contains("Python"))
{
    Console.WriteLine("❌ Python runtime not found.");
    Console.WriteLine("   This application requires Python to be installed and configured.");
    Console.WriteLine("   Please ensure Python 3.8+ is installed and accessible.");
    Console.WriteLine($"   Error: {ex.Message}");
}
catch (InvalidOperationException ex) when (ex.Message.Contains("Python environment is not initialized"))
{
    Console.WriteLine("❌ Python environment initialization failed.");
    Console.WriteLine("   The Python virtual environment could not be set up.");
    Console.WriteLine("   This may be due to missing dependencies or configuration issues.");
    Console.WriteLine($"   Error: {ex.Message}");
}
catch (InvalidOperationException ex) when (ex.Message.Contains("package compatibility"))
{
    Console.WriteLine("❌ Package compatibility issue detected.");
    Console.WriteLine();
    Console.WriteLine("🚨 SOLUTION:");
    Console.WriteLine("1. Uninstall incompatible packages:");
    Console.WriteLine("   pip uninstall torch torchvision torchaudio diffusers xformers -y");
    Console.WriteLine();
    Console.WriteLine("2. Install compatible CPU versions:");
    Console.WriteLine("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu");
    Console.WriteLine("   pip install diffusers --no-deps && pip install safetensors accelerate");
    Console.WriteLine();
    Console.WriteLine($"Full error: {ex.Message}");
}
catch (Exception ex) when (ex.Message.Contains("NETSDK1045") || ex.Message.Contains(".NET SDK does not support"))
{
    Console.WriteLine("❌ .NET version compatibility issue.");
    Console.WriteLine("   This application requires .NET 9.0 SDK or later.");
    Console.WriteLine();
    Console.WriteLine("🚨 SOLUTION:");
    Console.WriteLine("   Download and install .NET 9.0 SDK from:");
    Console.WriteLine("   https://dotnet.microsoft.com/download/dotnet/9.0");
    Console.WriteLine();
    Console.WriteLine($"Error: {ex.Message}");
}
catch (Exception ex)
{
    Console.WriteLine($"❌ Unexpected error occurred: {ex.GetType().Name}");
    Console.WriteLine($"   Message: {ex.Message}");
    Console.WriteLine();
    Console.WriteLine("🔍 Troubleshooting suggestions:");
    Console.WriteLine("   1. Ensure .NET 9.0 SDK is installed");
    Console.WriteLine("   2. Verify Python 3.8+ is available");
    Console.WriteLine("   3. Check internet connectivity for package downloads");
    Console.WriteLine("   4. Try running Console4 for more detailed diagnostics");
    Console.WriteLine();
    Console.WriteLine("Full stack trace:");
    Console.WriteLine(ex.ToString());
}

Console.WriteLine();
Console.WriteLine("=== Console3 Complete ===");
Console.WriteLine("Press any key to exit...");
Console.ReadKey();
