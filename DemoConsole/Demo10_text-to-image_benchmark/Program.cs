
using Demo10_text_to_image_benchmark;
using TransformersSharp;

namespace Demo10_text_to_image_benchmark
{
    /// <summary>
    /// Console application for testing and benchmarking TransformersSharp image generation performance.
    /// Updated for 256x256 image generation benchmarks.
    /// </summary>
    internal class Program
    {
        private static readonly string[] SamplePrompts =
        {
            "A pixelated image of a beaver in Canada.",
            "A futuristic city skyline at sunset.",
            "A cat riding a skateboard in a park.",
            "A surreal landscape with floating islands and waterfalls.",
            "A robot painting a portrait of a human.",
            "A fantasy castle surrounded by a magical forest.",
            "A vintage car driving through a neon-lit street.",
            "A cozy cabin in the mountains during winter.",
            "A dragon flying over a medieval village.",
            "A space scene with planets and stars in the background.",
            "A close-up of a flower with dew drops on its petals.",
            "A whimsical underwater scene with colorful fish and coral.",
            "A steampunk-inspired airship sailing through the clouds.",
            "A futuristic robot serving coffee in a cafe."
        };

        private static readonly List<ImageGenerationResult> CpuResults = new();
        private static readonly List<ImageGenerationResult> GpuResults = new();

        static void Main(string[] args)
        {
            Console.WriteLine("=== TransformersSharp Text-to-Image Benchmark - 256x256 Generation ===\n");
            Console.WriteLine("Image size: 256x256 pixels (optimized for performance testing)\n");
            Console.WriteLine("=== Benchmark Start ===");

            PerformCpuTests();
            PerformGpuTests();

            DisplayPerformanceComparison();
            DisplayTestExecutionSummary();

            Console.WriteLine("=== Benchmark Complete ===");
        }


        /// <summary>
        /// Performs CPU-based image generation tests.
        /// </summary>
        private static void PerformCpuTests()
        {
            Console.WriteLine("----------------------");
            Console.WriteLine("--- CPU Generation ---");

            foreach (var prompt in SamplePrompts)
            {
                RunImageGenerationTest(prompt, "cpu", CpuResults);                
            }
            Console.WriteLine("----------------------");
        }

        /// <summary>
        /// Performs GPU-based image generation tests if CUDA is available.
        /// </summary>
        private static void PerformGpuTests()
        {
            Console.WriteLine("----------------------");
            Console.WriteLine("--- GPU (CUDA) Generation ---");

            foreach (var prompt in SamplePrompts)
            {
                RunImageGenerationTest(prompt, "cuda", GpuResults);
            }
            Console.WriteLine("----------------------");
        }

        /// <summary>
        /// Runs a single image generation test for the specified device.
        /// </summary>
        /// <param name="prompt">The text prompt</param>
        /// <param name="device">The device to use (cpu/cuda)</param>
        /// <param name="results">The results collection to add to</param>
        /// <returns>True if critical error occurred and testing should stop</returns>
        private static bool RunImageGenerationTest(string prompt, string device, List<ImageGenerationResult> results)
        {
            Console.WriteLine(" >> start image generation ...");
            Console.WriteLine($" >> Prompt: {prompt}");

            try
            {
                using var generator = new ImageGenerator(device: device);
                var result = generator.GenerateImage(prompt);
                results.Add(result);
                Console.WriteLine($" >> ✅ {device.ToUpper()}: {result.TimeTakenSeconds:F2} seconds, Saved: {result.FileGenerated}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($">> ❌ {device.ToUpper()} test failed: {ex.Message}");
            }

            Console.WriteLine(" >> image generation complete");
            Console.WriteLine(" >> ");
            Console.WriteLine();
            return false; // Continue testing
        }

        /// <summary>
        /// Displays comprehensive performance comparison results.
        /// </summary>
        private static void DisplayPerformanceComparison()
        {
            Console.WriteLine("==============================");
            Console.WriteLine("=== Performance Comparison ===");

            if (GpuResults.Count > 0)
            {
                DisplayDetailedComparison();
            }
            else
            {
                DisplayCpuOnlyResults();
            }

            DisplayTotalRuntimeComparison();
        }

        /// <summary>
        /// Displays detailed CPU vs GPU comparison.
        /// </summary>
        private static void DisplayDetailedComparison()
        {
            for (int i = 0; i < SamplePrompts.Length; i++)
            {
                var cpuTime = CpuResults.Count > i ? CpuResults[i].TimeTakenSeconds : double.NaN;
                var gpuTime = GpuResults.Count > i ? GpuResults[i].TimeTakenSeconds : double.NaN;

                Console.WriteLine($"Prompt {i + 1}: {SamplePrompts[i]}");
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

        /// <summary>
        /// Displays CPU-only results when GPU tests were skipped.
        /// </summary>
        private static void DisplayCpuOnlyResults()
        {
            Console.WriteLine("GPU tests were skipped - no performance comparison available");
            Console.WriteLine($"CPU completed {CpuResults.Count} out of {SamplePrompts.Length} tests");
        }

        /// <summary>
        /// Displays total runtime comparison.
        /// </summary>
        private static void DisplayTotalRuntimeComparison()
        {
            var totalCpuTime = CpuResults.Sum(r => r.TimeTakenSeconds);
            var totalGpuTime = GpuResults.Sum(r => r.TimeTakenSeconds);

            Console.WriteLine("==============================");
            Console.WriteLine("=== Total Runtime Comparison ===");
            Console.WriteLine($"Total CPU Time: {totalCpuTime:F2} seconds");

            if (GpuResults.Count > 0)
            {
                DisplayGpuComparison(totalCpuTime, totalGpuTime);
            }
            else
            {
                Console.WriteLine("Total GPU Time: N/A (CUDA not available)");
                Console.WriteLine("Performance comparison not available - enable CUDA for GPU testing");
            }
        }

        /// <summary>
        /// Displays GPU performance comparison metrics.
        /// </summary>
        private static void DisplayGpuComparison(double totalCpuTime, double totalGpuTime)
        {
            var totalDiff = totalCpuTime - totalGpuTime;
            var overallFaster = totalDiff > 0 ? "GPU" : "CPU";

            Console.WriteLine($"Total GPU Time: {totalGpuTime:F2} seconds");
            Console.WriteLine($"Total Difference: {Math.Abs(totalDiff):F2} seconds");
            Console.WriteLine($"Overall Winner: {overallFaster} was faster by {Math.Abs(totalDiff):F2} seconds");

            if (CpuResults.Count > 0 && GpuResults.Count > 0)
            {
                var percentageDiff = (Math.Abs(totalDiff) / Math.Max(totalCpuTime, totalGpuTime)) * 100;
                Console.WriteLine($"Performance Improvement: {percentageDiff:F1}%");
            }
        }

        /// <summary>
        /// Displays test execution summary with performance metrics.
        /// </summary>
        private static void DisplayTestExecutionSummary()
        {
            Console.WriteLine("==============================");
            Console.WriteLine("--- Test Execution Summary ---");
            Console.WriteLine($"CPU Tests Completed: {CpuResults.Count}/{SamplePrompts.Length}");
            Console.WriteLine($"GPU Tests Completed: {GpuResults.Count}/{SamplePrompts.Length}");

            if (CpuResults.Count > 0)
            {
                var avgCpuTime = CpuResults.Average(r => r.TimeTakenSeconds);
                Console.WriteLine($"Average CPU Time per Image: {avgCpuTime:F2} seconds");
            }

            if (GpuResults.Count > 0)
            {
                var avgGpuTime = GpuResults.Average(r => r.TimeTakenSeconds);
                Console.WriteLine($"Average GPU Time per Image: {avgGpuTime:F2} seconds");

                if (CpuResults.Count > 0)
                {
                    var avgCpuTime = CpuResults.Average(r => r.TimeTakenSeconds);
                    var speedup = avgCpuTime / avgGpuTime;
                    Console.WriteLine($"GPU Speedup Factor: {speedup:F2}x");
                }
            }
            Console.WriteLine("==============================");
        }
    }
}
