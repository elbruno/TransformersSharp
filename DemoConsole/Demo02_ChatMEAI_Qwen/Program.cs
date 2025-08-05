using TransformersSharp;
using TransformersSharp.Pipelines;

var model = "openai/gpt-oss-20b";
var pipeline = TextGenerationPipeline.FromModel(model, device: "cpu");
Console.WriteLine("✅ Pipeline created successfully");

var prompt = "Why the sky is blue?";
var results = pipeline.Generate(prompt);

foreach (var result in results)
{
    Console.WriteLine(result);
}
