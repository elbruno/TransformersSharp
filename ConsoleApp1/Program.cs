// https://tonybaloney.github.io/TransformersSharp/pipelines/text_classification/

using TransformersSharp;
using TransformersSharp.Pipelines;

Console.WriteLine("=== TransformersSharp Text Generation (Console1) ===");
Console.WriteLine("Creating text generation pipeline...");
Console.WriteLine("Model: Qwen/Qwen2.5-0.5B");
Console.WriteLine("Torch Data Type: BFloat16");

var pipeline = TextGenerationPipeline.FromModel("Qwen/Qwen2.5-0.5B", TorchDtype.BFloat16);
Console.WriteLine("✅ Pipeline created successfully");
Console.WriteLine();

var messages = new List<Dictionary<string, string>>
    {
        new() {
            { "role", "user" },
            { "content", "Tell me a story about a brave knight." }
        }
    };

Console.WriteLine("Generating text...");
var results = pipeline.Generate(messages, maxLength: 100, temperature: 0.7);

Console.WriteLine("✅ Text generation completed");
Console.WriteLine();
Console.WriteLine("Results:");
Console.WriteLine("───────────────────");

foreach (var result in results)
{
    foreach (var kvp in result)
    {
        Console.WriteLine($"{kvp.Key}: {kvp.Value}");
    }
}