
using static Demo10_text_to_image_benchmark.ImageGenerator;

namespace Demo10_text_to_image_benchmark;

internal class Program
{
    private static int sampleCount = 2;
    private static int ImageSize = 128; // Fixed image size for benchmarking
    private static string model = "kandinsky-community/kandinsky-2-2-decoder";
    private static string[]? SamplePrompts;
    private static readonly List<ImageGenerationResult> CpuResults = new();
    private static readonly List<ImageGenerationResult> GpuResults = new();

    static void Main(string[] args)
    {
        Console.WriteLine($"=== TransformersSharp Text-to-Image Benchmark - {ImageSize}x{ImageSize} Generation ===\n");
        Console.WriteLine("=== Benchmark Start ===");

        var random = new Random();
        string promptFile = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "sample_prompts.txt");
        SamplePrompts = File.ReadAllLines(promptFile);
        SamplePrompts = [.. SamplePrompts.Where(p => !string.IsNullOrWhiteSpace(p) && !p.StartsWith("#"))];
        SamplePrompts = [.. SamplePrompts.OrderBy(x => random.Next()).Take(sampleCount)];

        //PerformCpuTests();
        //PerformGpuTests();

        // Use mock data for export testing
        MockDataGenerator.GenerateMockCpuResults(CpuResults, SamplePrompts!);
        MockDataGenerator.GenerateMockGpuResults(GpuResults, SamplePrompts!);

        var allResults = CpuResults.Concat(GpuResults).ToList();
        var exportPaths = ExportManager.ExportAll(allResults, SamplePrompts!, CpuResults, GpuResults);
        if (!string.IsNullOrEmpty(exportPaths.CsvPath))
            Console.WriteLine($"CSV results saved to: {exportPaths.CsvPath}");
        if (!string.IsNullOrEmpty(exportPaths.MarkdownPath))
            Console.WriteLine($"Markdown report saved to: {exportPaths.MarkdownPath}");
        if (!string.IsNullOrEmpty(exportPaths.HtmlPath))
            Console.WriteLine($"HTML report saved to: {exportPaths.HtmlPath}");

        // All reporting is now handled by ExportManager

        Console.WriteLine("=== Benchmark Complete ===");
    }

    /// <summary>
    /// Performs CPU-based image generation tests.
    /// </summary>
    private static void PerformCpuTests()
    {
        Console.WriteLine("----------------------");
        Console.WriteLine("--- CPU Generation ---");

        foreach (var prompt in SamplePrompts!)
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

        foreach (var prompt in SamplePrompts!)
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
            var imgGenSettings = new ImageGenerationSettings()
            {
                Height = ImageSize,
                Width = ImageSize,
            };
            using var generator = new ImageGenerator(
                device: device,
                model: model,
                settings: imgGenSettings);
            var result = generator.GenerateImage(prompt);
            results.Add(result);
            Console.WriteLine($" >> ✅ {device.ToUpper()}: {result.TimeTakenSeconds:F2} seconds, Saved: {result.FileName}");
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

}