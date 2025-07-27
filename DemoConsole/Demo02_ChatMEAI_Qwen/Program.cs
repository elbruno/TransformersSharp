using Microsoft.Extensions.AI;
using TransformersSharp;
using TransformersSharp.MEAI;

var model = "Qwen/Qwen2.5-0.5B";
var client = TextGenerationPipelineChatClient.FromModel(model, TorchDtype.BFloat16, trustRemoteCode: true);
var response = await client.GetResponseAsync("tell me a story about kittens");
Console.WriteLine(response.Text);
