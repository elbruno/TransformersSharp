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
        public string? HtmlPath { get; set; }
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
            var htmlPath = ExportToHtml(cpuResults, samplePrompts, gpuResults, outputFolder, exportTimestamp);

            return new ExportResultPaths { CsvPath = csvPath, MarkdownPath = mdPath, HtmlPath = htmlPath };
        }

        /// <summary>
        /// Generates an HTML report summarizing the benchmark results.
        /// </summary>
        public static string ExportToHtml(IEnumerable<ImageGenerationResult> cpuResults, string[] prompts, IEnumerable<ImageGenerationResult> gpuResults, string outputFolder, DateTime timestamp)
        {
            string fileName = $"TransformersSharp_Benchmark_{timestamp:yyyyMMdd_HHmmss}.html";
            string htmlPath = Path.Combine(outputFolder, fileName);
            Directory.CreateDirectory(outputFolder);
            var sb = new StringBuilder();
            sb.AppendLine("<!DOCTYPE html>");
            sb.AppendLine("<html lang=\"en\">\n<head>");
            sb.AppendLine("<meta charset=\"UTF-8\">\n<title>TransformersSharp Benchmark Report</title>");
            sb.AppendLine("<style>body{font-family:sans-serif;}table{border-collapse:collapse;}th,td{border:1px solid #ccc;padding:6px;}th{background:#f0f0f0;}tr:nth-child(even){background:#fafafa;}</style>");
            sb.AppendLine("</head>\n<body>");
            sb.AppendLine($"<h1>TransformersSharp Benchmark Report</h1>");
            sb.AppendLine($"<p><em>Generated: {timestamp:yyyy-MM-dd HH:mm:ss}</em></p>");
            sb.AppendLine("<h2>Test Prompts</h2><ol>");
            for (int i = 0; i < prompts.Length; i++)
            {
                sb.AppendLine($"<li>{System.Net.WebUtility.HtmlEncode(prompts[i])}</li>");
            }
            sb.AppendLine("</ol>");
            sb.AppendLine("<h2>Results Table</h2>");
            sb.AppendLine("<table><tr><th>Prompt</th><th>Device</th><th>Time (s)</th><th>File</th></tr>");
            for (int i = 0; i < prompts.Length; i++)
            {
                var cpu = cpuResults.ElementAtOrDefault(i);
                var gpu = gpuResults.ElementAtOrDefault(i);
                if (cpu != null)
                    sb.AppendLine($"<tr><td>{System.Net.WebUtility.HtmlEncode(prompts[i])}</td><td>CPU</td><td>{cpu.TimeTakenSeconds:F2}</td><td>{System.Net.WebUtility.HtmlEncode(cpu.FileName)}</td></tr>");
                if (gpu != null)
                    sb.AppendLine($"<tr><td>{System.Net.WebUtility.HtmlEncode(prompts[i])}</td><td>GPU</td><td>{gpu.TimeTakenSeconds:F2}</td><td>{System.Net.WebUtility.HtmlEncode(gpu.FileName)}</td></tr>");
            }
            sb.AppendLine("</table>");
            sb.AppendLine("<h2>Summary</h2><ul>");
            double cpuTotal = cpuResults.Sum(r => r.TimeTakenSeconds);
            double gpuTotal = gpuResults.Sum(r => r.TimeTakenSeconds);
            sb.AppendLine($"<li>Total CPU Time: {cpuTotal:F2} seconds</li>");
            sb.AppendLine($"<li>Total GPU Time: {(gpuResults.Any() ? gpuTotal.ToString("F2") : "N/A")} seconds</li>");
            if (cpuResults.Any() && gpuResults.Any())
            {
                double diff = cpuTotal - gpuTotal;
                string winner = diff > 0 ? "GPU" : "CPU";
                sb.AppendLine($"<li>{winner} was faster by {Math.Abs(diff):F2} seconds</li>");
                double percent = (Math.Abs(diff) / Math.Max(cpuTotal, gpuTotal)) * 100;
                sb.AppendLine($"<li>Performance Improvement: {percent:F1}%</li>");
            }
            sb.AppendLine("</ul>");
            sb.AppendLine("</body>\n</html>");
            File.WriteAllText(htmlPath, sb.ToString(), Encoding.UTF8);
            return htmlPath;
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
                return string.Empty;

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
