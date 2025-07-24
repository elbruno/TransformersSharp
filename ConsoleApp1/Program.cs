// https://tonybaloney.github.io/TransformersSharp/pipelines/text_classification/

using TransformersSharp;
using TransformersSharp.Pipelines;

var pipeline = TextGenerationPipeline.FromModel("Qwen/Qwen2.5-0.5B", TorchDtype.BFloat16);

// Prepare input as IReadOnlyList<IReadOnlyDictionary<string, string>>
var messages = new List<Dictionary<string, string>>
{
    new Dictionary<string, string>
    {
        { "role", "user" },
        { "content", "Tell me a story about a brave knight." }
    }
};

var results = pipeline.Generate(messages, maxLength: 100, temperature: 0.7);
foreach (var result in results)
{
    foreach (var kvp in result)
    {
        Console.WriteLine($"{kvp.Key}: {kvp.Value}");
    }
}