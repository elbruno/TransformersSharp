using Microsoft.Extensions.AI;
using TransformersSharp;
using TransformersSharp.MEAI;

Console.WriteLine("=== TransformersSharp Chat Client (Console2) ===");
Console.WriteLine();

//var model = "Qwen/Qwen2.5-0.5B";
var model = "Qwen/Qwen3-0.6B";

Console.WriteLine($"Model: {model}");
Console.WriteLine("Creating Microsoft.Extensions.AI chat client...");

var client = TextGenerationPipelineChatClient.FromModel(model, TorchDtype.BFloat16, trustRemoteCode: true);
Console.WriteLine("✅ Chat client created successfully");
Console.WriteLine();

Console.WriteLine("Sending message: 'tell me a story about kittens'");
var response = await client.GetResponseAsync("tell me a story about kittens");

Console.WriteLine("✅ Response received");
Console.WriteLine();
Console.WriteLine("Response:");
Console.WriteLine("───────────────────");
Console.WriteLine(response.Text);
