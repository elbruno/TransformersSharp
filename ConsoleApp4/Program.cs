
using System.Diagnostics;
using ConsoleApp4;
using TransformersSharp.Pipelines;


// ...existing code...

Console.WriteLine("=== TransformersSharp CUDA Performance Test ===");
Console.WriteLine();


var samplePrompts = new[]
{
    "A pixelated image of a beaver in Canada.",
    "A futuristic city skyline at sunset.",
    "A cat riding a skateboard in a park.",
    "A surreal landscape with floating islands.",
    "A robot painting a portrait in an art studio."
};

var cpuResults = new List<ImageGenerationResult>();
var gpuResults = new List<ImageGenerationResult>();

Console.WriteLine("=== TransformersSharp Image Generation Performance Test ===\n");
Console.WriteLine("Image size: 256x256 pixels\n");

Console.WriteLine("--- CPU Generation ---");
foreach (var prompt in samplePrompts)
{
    Console.WriteLine($"Prompt: {prompt}");
    try
    {
        var result = ImageGenerator.GenerateImage("cpu", prompt);
        cpuResults.Add(result);
        Console.WriteLine($"✅ CPU: {result.TimeTakenSeconds:F2} seconds, Saved: {result.FileGenerated}");
    }
    catch (Exception ex)
    {
        Console.WriteLine($"❌ CPU test failed: {ex.Message}");
    }
}

Console.WriteLine("\n--- GPU (CUDA) Generation ---");
foreach (var prompt in samplePrompts)
{
    Console.WriteLine($"Prompt: {prompt}");
    try
    {
        var result = ImageGenerator.GenerateImage("cuda", prompt);
        gpuResults.Add(result);
        Console.WriteLine($"✅ GPU: {result.TimeTakenSeconds:F2} seconds, Saved: {result.FileGenerated}");
    }
    catch (Exception ex)
    {
        Console.WriteLine($"❌ GPU test failed: {ex.Message}");
    }
}

Console.WriteLine("\n=== Performance Comparison ===");
for (int i = 0; i < samplePrompts.Length; i++)
{
    var cpuTime = cpuResults.Count > i ? cpuResults[i].TimeTakenSeconds : double.NaN;
    var gpuTime = gpuResults.Count > i ? gpuResults[i].TimeTakenSeconds : double.NaN;
    Console.WriteLine($"Prompt {i + 1}: {samplePrompts[i]}");
    Console.WriteLine($"  CPU: {cpuTime:F2} seconds");
    Console.WriteLine($"  GPU: {gpuTime:F2} seconds");
    if (!double.IsNaN(cpuTime) && !double.IsNaN(gpuTime))
    {
        var diff = cpuTime - gpuTime;
        var faster = diff > 0 ? "GPU" : "CPU";
        Console.WriteLine($"  {faster} was faster by {Math.Abs(diff):F2} seconds\n");
    }
    else
    {
        Console.WriteLine("  (One or both tests failed)\n");
    }
}

Console.WriteLine("=== Test Complete ===");
