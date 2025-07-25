// https://tonybaloney.github.io/TransformersSharp/pipelines/text_classification/

using TransformersSharp;
using TransformersSharp.Pipelines;

Console.WriteLine("=== TransformersSharp Text Generation (Console1) ===");
Console.WriteLine();

try
{
    // Check .NET version
    var dotnetVersion = Environment.Version;
    if (dotnetVersion.Major < 9)
    {
        Console.WriteLine("⚠️  WARNING: This application requires .NET 9.0 or later.");
        Console.WriteLine("   See docs/requirements.md for installation instructions.");
        Console.WriteLine();
    }

    Console.WriteLine("Creating text generation pipeline...");
    Console.WriteLine("Model: Qwen/Qwen2.5-0.5B");
    Console.WriteLine("Torch Data Type: BFloat16");
    
    var pipeline = TextGenerationPipeline.FromModel("Qwen/Qwen2.5-0.5B", TorchDtype.BFloat16);
    Console.WriteLine("✅ Pipeline created successfully");
    Console.WriteLine();

    // Prepare input as IReadOnlyList<IReadOnlyDictionary<string, string>>
    var messages = new List<Dictionary<string, string>>
    {
        new Dictionary<string, string>
        {
            { "role", "user" },
            { "content", "Tell me a story about a brave knight." }
        }
    };

    Console.WriteLine("Generating text...");
    var results = pipeline.Generate(messages, maxLength: 100, temperature: 0.7);
    
    Console.WriteLine("✅ Text generation completed");
    Console.WriteLine();
    Console.WriteLine("Results:");
    Console.WriteLine("───────────────────");
    
    foreach (var result in results)
    {
        foreach (var kvp in result)
        {
            Console.WriteLine($"{kvp.Key}: {kvp.Value}");
        }
    }
}
catch (Exception ex) when (ex.Message.Contains("NETSDK1045") || ex.Message.Contains(".NET SDK does not support"))
{
    Console.WriteLine("❌ .NET version compatibility issue.");
    Console.WriteLine("   This application requires .NET 9.0 SDK or later.");
    Console.WriteLine("   Download from: https://dotnet.microsoft.com/download/dotnet/9.0");
}
catch (Exception ex)
{
    Console.WriteLine($"❌ Error: {ex.GetType().Name}");
    Console.WriteLine($"   {ex.Message}");
    Console.WriteLine();
    Console.WriteLine("💡 Try Console4 for comprehensive diagnostics");
}

Console.WriteLine();
Console.WriteLine("=== Console1 Complete ===");