
using ConsoleApp4;


var samplePrompts = new[]
{
    "A pixelated image of a beaver in Canada.",
    "A futuristic city skyline at sunset.",
    "A cat riding a skateboard in a park.",
    "A surreal landscape with floating islands.",
    "A robot playing soccer on the moon."
};

var cpuResults = new List<ImageGenerationResult>();
var gpuResults = new List<ImageGenerationResult>();

Console.WriteLine("=== TransformersSharp Performance Test ===");
Console.WriteLine();
Console.WriteLine("=== TransformersSharp Image Generation Performance Test ===\n");
Console.WriteLine("Image size: 256x256 pixels\n");

Console.WriteLine("----------------------");
Console.WriteLine("--- CPU Generation ---");
foreach (var prompt in samplePrompts)
{
    Console.WriteLine(" >> start image generation ...");
    Console.WriteLine($" >> Prompt: {prompt}");
    using var cpuGen = new ImageGenerator(device: "cpu");
    try
    {
        var result = cpuGen.GenerateImage(prompt);
        cpuResults.Add(result);
        Console.WriteLine($" >> ✅ CPU: {result.TimeTakenSeconds:F2} seconds, Saved: {result.FileGenerated}");
    }
    catch (Exception ex)
    {
        Console.WriteLine($">> ❌ CPU test failed: {ex.Message}");
    }
    Console.WriteLine(" >> image generation complete");
    Console.WriteLine(" >> ");
    Console.WriteLine();
}
Console.WriteLine("----------------------");
Console.WriteLine("--- GPU (CUDA) Generation ---");

foreach (var prompt in samplePrompts)
{
    Console.WriteLine(" >> start image generation ...");
    Console.WriteLine($" >> Prompt: {prompt}");
    using var gpuGen = new ImageGenerator(device: "cuda");
    try
    {
        var result = gpuGen.GenerateImage(prompt);
        gpuResults.Add(result);
        Console.WriteLine($" >> ✅ GPU: {result.TimeTakenSeconds:F2} seconds, Saved: {result.FileGenerated}");
    }
    catch (Exception ex)
    {
        Console.WriteLine($" >> ❌ GPU test failed: {ex.Message}");
    }
    Console.WriteLine(" >> image generation complete");
    Console.WriteLine(" >> ");
    Console.WriteLine();
}
Console.WriteLine("----------------------");

Console.WriteLine("==============================");
Console.WriteLine("=== Performance Comparison ===");
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

// Calculate and display total runtime comparison
var totalCpuTime = cpuResults.Sum(r => r.TimeTakenSeconds);
var totalGpuTime = gpuResults.Sum(r => r.TimeTakenSeconds);
var totalDiff = totalCpuTime - totalGpuTime;
var overallFaster = totalDiff > 0 ? "GPU" : "CPU";

Console.WriteLine("==============================");
Console.WriteLine("=== Total Runtime Comparison ===");
Console.WriteLine($"Total CPU Time: {totalCpuTime:F2} seconds");
Console.WriteLine($"Total GPU Time: {totalGpuTime:F2} seconds");
Console.WriteLine($"Total Difference: {Math.Abs(totalDiff):F2} seconds");
Console.WriteLine($"Overall Winner: {overallFaster} was faster by {Math.Abs(totalDiff):F2} seconds");

if (cpuResults.Count > 0 && gpuResults.Count > 0)
{
    var percentageDiff = (Math.Abs(totalDiff) / Math.Max(totalCpuTime, totalGpuTime)) * 100;
    Console.WriteLine($"Performance Improvement: {percentageDiff:F1}%");
}

Console.WriteLine("==============================");
Console.WriteLine("=== Test Complete ===");
