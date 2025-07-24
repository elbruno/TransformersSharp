
using ConsoleApp4;
using TransformersSharp;

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

// Check device capabilities
Console.WriteLine("=== Device Capability Check ===");
var deviceInfo = TransformerEnvironment.GetDeviceInfo();
bool cudaAvailable = TransformerEnvironment.IsCudaAvailable();

Console.WriteLine($"CUDA Available: {(cudaAvailable ? "✅ Yes" : "❌ No")}");
if (cudaAvailable)
{
    if (deviceInfo.ContainsKey("cuda_device_count"))
    {
        Console.WriteLine($"CUDA Devices: {deviceInfo["cuda_device_count"]}");
    }
    if (deviceInfo.ContainsKey("device_name"))
    {
        Console.WriteLine($"GPU: {deviceInfo["device_name"]}");
    }
}
else
{
    Console.WriteLine("GPU acceleration unavailable - using CPU only");
    Console.WriteLine("To enable GPU acceleration:");
    Console.WriteLine("  1. Ensure you have a compatible NVIDIA GPU");
    Console.WriteLine("  2. Install NVIDIA drivers");
    Console.WriteLine("  3. Run: TransformerEnvironment.InstallCudaPyTorch()");
}
Console.WriteLine();

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

if (cudaAvailable)
{
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
}
else
{
    Console.WriteLine("⚠️  Skipping GPU tests - CUDA not available");
    Console.WriteLine("   Run on a system with NVIDIA GPU and CUDA support for GPU acceleration");
    Console.WriteLine();
}
Console.WriteLine("----------------------");

Console.WriteLine("==============================");
Console.WriteLine("=== Performance Comparison ===");

if (gpuResults.Count > 0)
{
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
}
else
{
    Console.WriteLine("GPU tests were skipped - no performance comparison available");
    Console.WriteLine($"CPU completed {cpuResults.Count} out of {samplePrompts.Length} tests");
}

// Calculate and display total runtime comparison
var totalCpuTime = cpuResults.Sum(r => r.TimeTakenSeconds);
var totalGpuTime = gpuResults.Sum(r => r.TimeTakenSeconds);

Console.WriteLine("==============================");
Console.WriteLine("=== Total Runtime Comparison ===");
Console.WriteLine($"Total CPU Time: {totalCpuTime:F2} seconds");

if (gpuResults.Count > 0)
{
    var totalDiff = totalCpuTime - totalGpuTime;
    var overallFaster = totalDiff > 0 ? "GPU" : "CPU";
    
    Console.WriteLine($"Total GPU Time: {totalGpuTime:F2} seconds");
    Console.WriteLine($"Total Difference: {Math.Abs(totalDiff):F2} seconds");
    Console.WriteLine($"Overall Winner: {overallFaster} was faster by {Math.Abs(totalDiff):F2} seconds");

    if (cpuResults.Count > 0 && gpuResults.Count > 0)
    {
        var percentageDiff = (Math.Abs(totalDiff) / Math.Max(totalCpuTime, totalGpuTime)) * 100;
        Console.WriteLine($"Performance Improvement: {percentageDiff:F1}%");
    }
}
else
{
    Console.WriteLine("Total GPU Time: N/A (CUDA not available)");
    Console.WriteLine("Performance comparison not available - enable CUDA for GPU testing");
}

Console.WriteLine("==============================");
Console.WriteLine("=== Test Complete ===");
