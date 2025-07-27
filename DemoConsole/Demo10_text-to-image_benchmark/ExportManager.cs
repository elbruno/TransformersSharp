using System.Globalization;
using System.Text;

namespace Demo10_text_to_image_benchmark;

public class ExportResultPaths
{
    public string? CsvPath { get; set; }
    public string? MarkdownPath { get; set; }
    public string? HtmlPath { get; set; }
}
public static class ExportManager
{
    public static ExportResultPaths ExportAll(List<ImageGenerationResult> allResults, string[] samplePrompts, List<ImageGenerationResult> cpuResults, List<ImageGenerationResult> gpuResults)
    {
        string outputFolder = GetDefaultOutputFolder();
        var firstResult = allResults.FirstOrDefault();
        if (firstResult != null && !string.IsNullOrEmpty(firstResult.FileFullPath))
        {
            var folder = Path.GetDirectoryName(firstResult.FileFullPath);
            if (!string.IsNullOrEmpty(folder))
                outputFolder = folder;
        }
        DateTime exportTimestamp = DateTime.Now;
        var csvPath = ExportToCsv(allResults, outputFolder, exportTimestamp);
        var mdPath = ExportToMarkdown(cpuResults, samplePrompts, gpuResults, outputFolder, exportTimestamp);
        var htmlPath = ExportToHtml(cpuResults, samplePrompts, gpuResults, outputFolder, exportTimestamp);
        return new ExportResultPaths { CsvPath = csvPath, MarkdownPath = mdPath, HtmlPath = htmlPath };
    }

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
        sb.AppendLine("<h2>Configuration</h2><ul>");
        sb.AppendLine($"<li>Model: {System.Net.WebUtility.HtmlEncode(ProgramModelName())}</li>");
        sb.AppendLine($"<li>Image Size: {ProgramImageSize()}x{ProgramImageSize()} px</li>");
        sb.AppendLine($"<li>Sample Count: {prompts.Length}</li>");
        sb.AppendLine($"<li>.NET Version: {System.Environment.Version}</li>");
        sb.AppendLine($"<li>OS: {System.Runtime.InteropServices.RuntimeInformation.OSDescription}</li>");
        sb.AppendLine("</ul>");
        sb.AppendLine("<h2>Test Prompts</h2><ol>");
        for (int i = 0; i < prompts.Length; i++)
            sb.AppendLine($"<li>{System.Net.WebUtility.HtmlEncode(prompts[i])}</li>");
        sb.AppendLine("</ol>");
        sb.AppendLine("<h2>Results Table</h2>");
        sb.AppendLine("<table><tr><th>Prompt</th><th>Device</th><th>Time (s)</th><th>Speedup</th><th>Status</th><th>File</th><th>Preview</th></tr>");
        for (int i = 0; i < prompts.Length; i++)
        {
            var cpu = cpuResults.ElementAtOrDefault(i);
            var gpu = gpuResults.ElementAtOrDefault(i);
            if (cpu != null)
            {
                string speedup = (gpu != null && gpu.TimeTakenSeconds > 0) ? (cpu.TimeTakenSeconds / gpu.TimeTakenSeconds).ToString("F2") : "-";
                string status = !string.IsNullOrEmpty(cpu.FileFullPath) ? "Success" : "Fail";
                string fileLink = !string.IsNullOrEmpty(cpu.FileFullPath) ? $"<a href=\"file:///{cpu.FileFullPath.Replace("\\", "/")}\">{System.Net.WebUtility.HtmlEncode(cpu.FileName)}</a>" : "";
                string imgTag = !string.IsNullOrEmpty(cpu.FileFullPath) ? $"<img src=\"file:///{cpu.FileFullPath.Replace("\\", "/")}\" alt=\"CPU Image\" width=\"64\">" : "";
                sb.AppendLine($"<tr><td>{System.Net.WebUtility.HtmlEncode(prompts[i])}</td><td>CPU</td><td>{cpu.TimeTakenSeconds:F2}</td><td>{speedup}</td><td>{status}</td><td>{fileLink}</td><td>{imgTag}</td></tr>");
            }
            if (gpu != null)
            {
                string speedup = (cpu != null && gpu.TimeTakenSeconds > 0) ? (cpu.TimeTakenSeconds / gpu.TimeTakenSeconds).ToString("F2") : "-";
                string status = !string.IsNullOrEmpty(gpu.FileFullPath) ? "Success" : "Fail";
                string fileLink = !string.IsNullOrEmpty(gpu.FileFullPath) ? $"<a href=\"file:///{gpu.FileFullPath.Replace("\\", "/")}\">{System.Net.WebUtility.HtmlEncode(gpu.FileName)}</a>" : "";
                string imgTag = !string.IsNullOrEmpty(gpu.FileFullPath) ? $"<img src=\"file:///{gpu.FileFullPath.Replace("\\", "/")}\" alt=\"GPU Image\" width=\"64\">" : "";
                sb.AppendLine($"<tr><td>{System.Net.WebUtility.HtmlEncode(prompts[i])}</td><td>GPU</td><td>{gpu.TimeTakenSeconds:F2}</td><td>{speedup}</td><td>{status}</td><td>{fileLink}</td><td>{imgTag}</td></tr>");
            }
        }
        sb.AppendLine("</table>");
        sb.AppendLine("<h2>Aggregate Statistics</h2><ul>");
        sb.AppendLine($"<li>CPU Min: {cpuResults.Min(r => r.TimeTakenSeconds):F2} s, Max: {cpuResults.Max(r => r.TimeTakenSeconds):F2} s, Avg: {cpuResults.Average(r => r.TimeTakenSeconds):F2} s</li>");
        sb.AppendLine($"<li>GPU Min: {(gpuResults.Any() ? gpuResults.Min(r => r.TimeTakenSeconds).ToString("F2") : "N/A")} s, Max: {(gpuResults.Any() ? gpuResults.Max(r => r.TimeTakenSeconds).ToString("F2") : "N/A")} s, Avg: {(gpuResults.Any() ? gpuResults.Average(r => r.TimeTakenSeconds).ToString("F2") : "N/A")} s</li>");
        sb.AppendLine("</ul>");
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

    public static string ExportToCsv(IEnumerable<ImageGenerationResult> results, string outputFolder, DateTime timestamp)
    {
        if (results == null || !results.Any())
            return string.Empty;
        string fileName = $"TransformersSharp_Benchmark_{timestamp:yyyyMMdd_HHmmss}.csv";
        string csvPath = Path.Combine(outputFolder, fileName);
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

    public static string ExportToMarkdown(IEnumerable<ImageGenerationResult> cpuResults, string[] prompts, IEnumerable<ImageGenerationResult> gpuResults, string outputFolder, DateTime timestamp)
    {
        string fileName = $"TransformersSharp_Benchmark_{timestamp:yyyyMMdd_HHmmss}.md";
        string mdPath = Path.Combine(outputFolder, fileName);
        Directory.CreateDirectory(outputFolder);
        var sb = new StringBuilder();
        sb.AppendLine($"# TransformersSharp Benchmark Report");
        sb.AppendLine($"_Generated: {timestamp:yyyy-MM-dd HH:mm:ss}_\n");
        sb.AppendLine($"## Configuration\n- Model: {ProgramModelName()}\n- Image Size: {ProgramImageSize()}x{ProgramImageSize()} px\n- Sample Count: {prompts.Length}\n- .NET Version: {System.Environment.Version}\n- OS: {System.Runtime.InteropServices.RuntimeInformation.OSDescription}\n");
        sb.AppendLine($"## Test Prompts");
        for (int i = 0; i < prompts.Length; i++)
            sb.AppendLine($"{i + 1}. {prompts[i]}");
        sb.AppendLine();
        sb.AppendLine("## Results Table");
        sb.AppendLine("| Prompt | Device | Time (s) | Speedup | Status | File |");
        sb.AppendLine("|--------|--------|----------|---------|--------|------|");
        for (int i = 0; i < prompts.Length; i++)
        {
            var cpu = cpuResults.ElementAtOrDefault(i);
            var gpu = gpuResults.ElementAtOrDefault(i);
            if (cpu != null)
            {
                string speedup = (gpu != null && gpu.TimeTakenSeconds > 0) ? (cpu.TimeTakenSeconds / gpu.TimeTakenSeconds).ToString("F2") : "-";
                string status = !string.IsNullOrEmpty(cpu.FileFullPath) ? "Success" : "Fail";
                string fileLink = !string.IsNullOrEmpty(cpu.FileFullPath) ? $"[{cpu.FileName}]({cpu.FileFullPath})" : "";
                sb.AppendLine($"| {prompts[i]} | CPU | {cpu.TimeTakenSeconds:F2} | {speedup} | {status} | {fileLink} |");
            }
            if (gpu != null)
            {
                string speedup = (cpu != null && gpu.TimeTakenSeconds > 0) ? (cpu.TimeTakenSeconds / gpu.TimeTakenSeconds).ToString("F2") : "-";
                string status = !string.IsNullOrEmpty(gpu.FileFullPath) ? "Success" : "Fail";
                string fileLink = !string.IsNullOrEmpty(gpu.FileFullPath) ? $"[{gpu.FileName}]({gpu.FileFullPath})" : "";
                sb.AppendLine($"| {prompts[i]} | GPU | {gpu.TimeTakenSeconds:F2} | {speedup} | {status} | {fileLink} |");
            }
        }
        sb.AppendLine();
        sb.AppendLine($"## Aggregate Statistics\n- CPU Min: {cpuResults.Min(r => r.TimeTakenSeconds):F2} s, Max: {cpuResults.Max(r => r.TimeTakenSeconds):F2} s, Avg: {cpuResults.Average(r => r.TimeTakenSeconds):F2} s");
        sb.AppendLine($"- GPU Min: {(gpuResults.Any() ? gpuResults.Min(r => r.TimeTakenSeconds).ToString("F2") : "N/A")} s, Max: {(gpuResults.Any() ? gpuResults.Max(r => r.TimeTakenSeconds).ToString("F2") : "N/A")} s, Avg: {(gpuResults.Any() ? gpuResults.Average(r => r.TimeTakenSeconds).ToString("F2") : "N/A")} s\n");
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

    private static string ProgramModelName()
    {
        var type = Type.GetType("Demo10_text_to_image_benchmark.Program");
        var field = type?.GetField("model", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static);
        return field?.GetValue(null)?.ToString() ?? "unknown";
    }
    private static int ProgramImageSize()
    {
        var type = Type.GetType("Demo10_text_to_image_benchmark.Program");
        var field = type?.GetField("ImageSize", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static);
        return field != null ? (int)(field.GetValue(null) ?? 0) : 0;
    }
}