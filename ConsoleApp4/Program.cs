
using ConsoleApp4;
using TransformersSharp;

var samplePrompts = new[]
{
    "A pixelated image of a beaver in Canada.",
    // "A futuristic city skyline at sunset.",
    // "A cat riding a skateboard in a park.",
    // "A surreal landscape with floating islands.",
    // "A robot playing soccer on the moon."
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
    Console.WriteLine("  3. Installing CUDA PyTorch...");

    try
    {
        TransformerEnvironment.InstallCudaPyTorch();
        Console.WriteLine("✅ CUDA PyTorch installation initiated");
        Console.WriteLine("⚠️  Please restart the application after installation completes");
    }
    catch (Exception ex)
    {
        Console.WriteLine($"❌ Failed to install CUDA PyTorch: {ex.Message}");
        Console.WriteLine("   You may need to install manually - see installation instructions above");
    }
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

// Display detailed system information used for the tests
Console.WriteLine();
Console.WriteLine("==============================");
Console.WriteLine("=== System Information ===");
try
{
    var systemInfo = TransformerEnvironment.GetDetailedSystemInfo();

    // Display System Information
    if (systemInfo.ContainsKey("system") && systemInfo["system"] is Dictionary<string, object> sysInfo)
    {
        Console.WriteLine("--- System Information ---");
        if (sysInfo.ContainsKey("platform")) Console.WriteLine($"Platform: {sysInfo["platform"]}");
        if (sysInfo.ContainsKey("architecture")) Console.WriteLine($"Architecture: {sysInfo["architecture"]}");
        if (sysInfo.ContainsKey("processor_count")) Console.WriteLine($"Processor Count: {sysInfo["processor_count"]}");
        if (sysInfo.ContainsKey("dotnet_version")) Console.WriteLine($".NET Version: {sysInfo["dotnet_version"]}");
        Console.WriteLine();
    }

    // Display CUDA Information
    if (systemInfo.ContainsKey("cuda") && systemInfo["cuda"] is Dictionary<string, object> cudaInfo)
    {
        Console.WriteLine("--- CUDA Information ---");
        if (cudaInfo.ContainsKey("available"))
        {
            bool available = Convert.ToBoolean(cudaInfo["available"]);
            Console.WriteLine($"CUDA Available: {(available ? "✅ Yes" : "❌ No")}");
        }
        if (cudaInfo.ContainsKey("description")) Console.WriteLine($"Status: {cudaInfo["description"]}");
        Console.WriteLine();
    }

    // Display Memory Information
    if (systemInfo.ContainsKey("memory") && systemInfo["memory"] is Dictionary<string, object> memInfo)
    {
        Console.WriteLine("--- Memory Information ---");
        if (memInfo.ContainsKey("working_set_mb")) Console.WriteLine($"Working Set: {memInfo["working_set_mb"]} MB");
        Console.WriteLine();
    }

    // Display test execution summary
    Console.WriteLine("--- Test Execution Summary ---");
    Console.WriteLine($"CPU Tests Completed: {cpuResults.Count}/{samplePrompts.Length}");
    Console.WriteLine($"GPU Tests Completed: {gpuResults.Count}/{samplePrompts.Length}");
    if (cpuResults.Count > 0)
    {
        var avgCpuTime = cpuResults.Average(r => r.TimeTakenSeconds);
        Console.WriteLine($"Average CPU Time per Image: {avgCpuTime:F2} seconds");
    }
    if (gpuResults.Count > 0)
    {
        var avgGpuTime = gpuResults.Average(r => r.TimeTakenSeconds);
        Console.WriteLine($"Average GPU Time per Image: {avgGpuTime:F2} seconds");

        if (cpuResults.Count > 0)
        {
            var avgCpuTime = cpuResults.Average(r => r.TimeTakenSeconds);
            var speedup = avgCpuTime / avgGpuTime;
            Console.WriteLine($"GPU Speedup Factor: {speedup:F2}x");
        }
    }
}
catch (Exception ex)
{
    Console.WriteLine($"❌ Could not retrieve detailed system information: {ex.Message}");
}

Console.WriteLine("==============================");
