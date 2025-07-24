using System.Diagnostics;
using TransformersSharp.Pipelines;

namespace ConsoleApp3;

public static class TestCudaPerformance
{
    public static void RunPerformanceTest()
    {
        Console.WriteLine("=== TransformersSharp CUDA Performance Test ===");
        Console.WriteLine();

        // Test with CUDA
        Console.WriteLine("Testing with CUDA device...");
        var stopwatchCuda = Stopwatch.StartNew();

        try
        {
            var pipelineCuda = TextToImagePipeline.FromModel(
                "kandinsky-community/kandinsky-2-2-decoder",
                trustRemoteCode: true,
                device: "cuda");

            Console.WriteLine($"Pipeline created. Device: {pipelineCuda.DeviceType}");

            var resultCuda = pipelineCuda.Generate(
                "A small 256x256 test image of a sunset over mountains",
                numInferenceSteps: 10, // Reduced for faster testing
                guidanceScale: 7.5f,
                height: 256,
                width: 256);

            stopwatchCuda.Stop();
            Console.WriteLine($"✅ CUDA generation completed in: {stopwatchCuda.Elapsed.TotalSeconds:F2} seconds");
            Console.WriteLine($"Image size: {resultCuda.ImageBytes.Length} bytes");
        }
        catch (Exception ex)
        {
            stopwatchCuda.Stop();
            Console.WriteLine($"❌ CUDA test failed: {ex.Message}");
        }

        Console.WriteLine();

        // Test with CPU fallback
        Console.WriteLine("Testing with CPU device...");
        var stopwatchCpu = Stopwatch.StartNew();

        try
        {
            var pipelineCpu = TextToImagePipeline.FromModel(
                "kandinsky-community/kandinsky-2-2-decoder",
                trustRemoteCode: true,
                device: "cpu");

            Console.WriteLine($"Pipeline created. Device: {pipelineCpu.DeviceType}");

            var resultCpu = pipelineCpu.Generate(
                "A small 256x256 test image of a sunset over mountains",
                numInferenceSteps: 10, // Reduced for faster testing
                guidanceScale: 7.5f,
                height: 256,
                width: 256);

            stopwatchCpu.Stop();
            Console.WriteLine($"✅ CPU generation completed in: {stopwatchCpu.Elapsed.TotalSeconds:F2} seconds");
            Console.WriteLine($"Image size: {resultCpu.ImageBytes.Length} bytes");
        }
        catch (Exception ex)
        {
            stopwatchCpu.Stop();
            Console.WriteLine($"❌ CPU test failed: {ex.Message}");
        }

        Console.WriteLine();
        Console.WriteLine("=== Test Complete ===");
    }
}
