using Microsoft.Extensions.AI;
using TransformersSharp;
using TransformersSharp.MEAI;

Console.WriteLine("=== TransformersSharp Chat Client (Console2) ===");
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

    //var model = "Qwen/Qwen2.5-0.5B";
    var model = "Qwen/Qwen3-0.6B";

    Console.WriteLine($"Model: {model}");
    Console.WriteLine("Creating Microsoft.Extensions.AI chat client...");
    
    var client = TextGenerationPipelineChatClient.FromModel(model, TorchDtype.BFloat16, trustRemoteCode: true);
    Console.WriteLine("✅ Chat client created successfully");
    Console.WriteLine();

    Console.WriteLine("Sending message: 'tell me a story about kittens'");
    var response = await client.GetResponseAsync("tell me a story about kittens");

    Console.WriteLine("✅ Response received");
    Console.WriteLine();
    Console.WriteLine("Response:");
    Console.WriteLine("───────────────────");
    Console.WriteLine(response.Text);
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
Console.WriteLine("=== Console2 Complete ===");