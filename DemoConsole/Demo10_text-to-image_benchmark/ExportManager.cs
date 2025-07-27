using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;

namespace Demo10_text_to_image_benchmark
{
    public class ExportResultPaths
    {
        public string? CsvPath { get; set; }
        public string? MarkdownPath { get; set; }
    }

    public static class ExportManager
    {
        /// <summary>
        /// Exports results to CSV and Markdown, determines output folder, and returns file paths.
        /// </summary>
        public static ExportResultPaths ExportAll(List<ImageGenerationResult> allResults, string[] samplePrompts, List<ImageGenerationResult> cpuResults, List<ImageGenerationResult> gpuResults)
        {
            // Get default output folder from ImageGenerator settings
            string outputFolder = GetDefaultOutputFolder();
            var firstResult = allResults.FirstOrDefault();
            if (firstResult != null && !string.IsNullOrEmpty(firstResult.FileFullPath))
            {
                var folder = Path.GetDirectoryName(firstResult.FileFullPath);
                if (!string.IsNullOrEmpty(folder))
                    outputFolder = folder;
            }

            // Use a single timestamp for both files
            DateTime exportTimestamp = DateTime.Now;
            var csvPath = ExportToCsv(allResults, outputFolder, exportTimestamp);
            var mdPath = ExportToMarkdown(cpuResults, samplePrompts, gpuResults, outputFolder, exportTimestamp);

            return new ExportResultPaths { CsvPath = csvPath, MarkdownPath = mdPath };
        }

        private static string GetDefaultOutputFolder()
        {
            return Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Desktop), "TransformersSharpImages");
        }

        /// <summary>
        /// Exports the image generation results to a CSV file in the images folder.
        /// </summary>
        public static string ExportToCsv(IEnumerable<ImageGenerationResult> results, string outputFolder, DateTime timestamp)
        {
            if (results == null || !results.Any())
                return null;

            string fileName = $"TransformersSharp_Benchmark_{timestamp:yyyyMMdd_HHmmss}.csv";
            string csvPath = Path.Combine(outputFolder, fileName);

            // Ensure output directory exists
            Directory.CreateDirectory(outputFolder);

            using (var writer = new StreamWriter(csvPath, false, Encoding.UTF8))
            {
                writer.WriteLine("Prompt,Width,Height,FileName,FileFullPath,TimeTakenSeconds,DeviceType");
                foreach (var r in results)
                {
                    string safePrompt = r.Prompt?.Replace("\"", "\"\"") ?? string.Empty;
                    string safeFileName = r.FileName?.Replace("\"", "\"\"") ?? string.Empty;
                    string safeFileFullPath = r.FileFullPath?.Replace("\"", "\"\"") ?? string.Empty;
                    string safeDeviceType = r.DeviceType?.Replace("\"", "\"\"") ?? string.Empty;
                    writer.WriteLine($"\"{safePrompt}\",{r.Width},{r.Height},\"{safeFileName}\",\"{safeFileFullPath}\",{r.TimeTakenSeconds.ToString(CultureInfo.InvariantCulture)},{safeDeviceType}");
                }
            }
            return csvPath;
        }

        /// <summary>
        /// Generates a markdown report summarizing the benchmark results.
        /// </summary>
        public static string ExportToMarkdown(IEnumerable<ImageGenerationResult> cpuResults, string[] prompts, IEnumerable<ImageGenerationResult> gpuResults, string outputFolder, DateTime timestamp)
        {
            string fileName = $"TransformersSharp_Benchmark_{timestamp:yyyyMMdd_HHmmss}.md";
            string mdPath = Path.Combine(outputFolder, fileName);
            // Ensure output directory exists
            Directory.CreateDirectory(outputFolder);
            var sb = new StringBuilder();
            sb.AppendLine($"# TransformersSharp Benchmark Report");
            sb.AppendLine($"_Generated: {timestamp:yyyy-MM-dd HH:mm:ss}_\n");
            sb.AppendLine($"## Test Prompts");
            for (int i = 0; i < prompts.Length; i++)
            {
                sb.AppendLine($"{i + 1}. {prompts[i]}");
            }
            sb.AppendLine();
            sb.AppendLine("## Results Table");
            sb.AppendLine("| Prompt | Device | Time (s) | File | ");
            sb.AppendLine("|--------|--------|----------|------|");
            for (int i = 0; i < prompts.Length; i++)
            {
                var cpu = cpuResults.ElementAtOrDefault(i);
                var gpu = gpuResults.ElementAtOrDefault(i);
                if (cpu != null)
                    sb.AppendLine($"| {prompts[i]} | CPU | {cpu.TimeTakenSeconds:F2} | {cpu.FileName} |");
                if (gpu != null)
                    sb.AppendLine($"| {prompts[i]} | GPU | {gpu.TimeTakenSeconds:F2} | {gpu.FileName} |");
            }
            sb.AppendLine();
            sb.AppendLine("## Summary");
            double cpuTotal = cpuResults.Sum(r => r.TimeTakenSeconds);
            double gpuTotal = gpuResults.Sum(r => r.TimeTakenSeconds);
            sb.AppendLine($"- Total CPU Time: {cpuTotal:F2} seconds");
            sb.AppendLine($"- Total GPU Time: {(gpuResults.Any() ? gpuTotal.ToString("F2") : "N/A")} seconds");
            if (cpuResults.Any() && gpuResults.Any())
            {
                double diff = cpuTotal - gpuTotal;
                string winner = diff > 0 ? "GPU" : "CPU";
                sb.AppendLine($"- {winner} was faster by {Math.Abs(diff):F2} seconds");
                double percent = (Math.Abs(diff) / Math.Max(cpuTotal, gpuTotal)) * 100;
                sb.AppendLine($"- Performance Improvement: {percent:F1}%");
            }
            sb.AppendLine();
            File.WriteAllText(mdPath, sb.ToString(), Encoding.UTF8);
            return mdPath;
        }
    }
}
