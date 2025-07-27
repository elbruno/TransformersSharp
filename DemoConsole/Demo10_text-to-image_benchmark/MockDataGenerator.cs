using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace Demo10_text_to_image_benchmark
{
    public static class MockDataGenerator
    {
        public static void GenerateMockCpuResults(List<ImageGenerationResult> cpuResults, string[] prompts)
        {
            cpuResults.Clear();
            var rnd = new Random(42);
            string desktopImagesFolder = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Desktop), "TransformersSharpImages");
            foreach (var prompt in prompts)
            {
                cpuResults.Add(new ImageGenerationResult
                {
                    Prompt = prompt,
                    Width = 128,
                    Height = 128,
                    FileFullPath = Path.Combine(desktopImagesFolder, $"mock_cpu_{prompt.GetHashCode()}.png"),
                    TimeTakenSeconds = Math.Round(40 + rnd.NextDouble() * 20, 2),
                    DeviceType = "cpu"
                });
            }
        }

        public static void GenerateMockGpuResults(List<ImageGenerationResult> gpuResults, string[] prompts)
        {
            gpuResults.Clear();
            var rnd = new Random(99);
            string desktopImagesFolder = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Desktop), "TransformersSharpImages");
            foreach (var prompt in prompts)
            {
                gpuResults.Add(new ImageGenerationResult
                {
                    Prompt = prompt,
                    Width = 128,
                    Height = 128,
                    FileFullPath = Path.Combine(desktopImagesFolder, $"mock_gpu_{prompt.GetHashCode()}.png"),
                    TimeTakenSeconds = Math.Round(5 + rnd.NextDouble() * 5, 2),
                    DeviceType = "cuda"
                });
            }
        }
    }
}
