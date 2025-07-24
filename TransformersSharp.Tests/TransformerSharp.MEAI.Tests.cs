using TransformersSharp.MEAI;
using TransformersSharp.Pipelines;
using Microsoft.Extensions.AI;

namespace TransformersSharp.Tests;

public class TransformerSharpMEAITests
{
    [Fact]
    public async Task TestChatClient()
    {
        var chatClient = TextGenerationPipelineChatClient.FromModel("Qwen/Qwen2.5-0.5B", TorchDtype.BFloat16, trustRemoteCode: true);
        var messages = new List<ChatMessage>
        {
            new(ChatRole.System, "You are a helpful little robot."),
            new(ChatRole.User, "how many helicopters can a human eat in one sitting?!")
        };
        var response = await chatClient.GetResponseAsync(messages, new() { Temperature = 0.7f });
        
        Assert.NotNull(response);
        Assert.Contains("helicopter", response.Text, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task TestChatClientStreaming()
    {
        var chatClient = TextGenerationPipelineChatClient.FromModel("Qwen/Qwen2.5-0.5B", TorchDtype.BFloat16, trustRemoteCode: true);
        var messages = new List<ChatMessage>
        {
            new(ChatRole.System, "You are a helpful little robot."),
            new(ChatRole.User, "how many helicopters can a human eat in one sitting?!")
        };
        var response = chatClient.GetStreamingResponseAsync(messages, new() { Temperature = 0.7f });
        await foreach (var update in response)
        {
            Assert.NotNull(update);
            Assert.NotEmpty(update.Text);
        }
    }

    [Fact]
    public async Task TestSpeechToTextClient()
    {
        var speechClient = SpeechToTextClient.FromModel("openai/whisper-tiny");
        using var audioStream = new MemoryStream(File.ReadAllBytes("sample.flac"));
        var response = await speechClient.GetTextAsync(audioStream);
        
        Assert.NotNull(response);
        Assert.NotEmpty(response.Text);
        Assert.Contains("stew for dinner", response.Text, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task TestSpeechToTextClientStreaming()
    {
        var speechClient = SpeechToTextClient.FromModel("openai/whisper-tiny");
        using var audioStream = new MemoryStream(File.ReadAllBytes("sample.flac"));
        var response = speechClient.GetStreamingTextAsync(audioStream);
        await foreach (var update in response)
        {
            Assert.NotNull(update);
            Assert.NotEmpty(update.Text);
            Assert.Contains("stew for dinner", update.Text, StringComparison.OrdinalIgnoreCase);
        }
    }

    [Fact]
    public void TestTextToImagePipeline()
    {
        var textToImagePipeline = TextToImagePipeline.FromModel("kandinsky-community/kandinsky-2-2-decoder", trustRemoteCode: true);
        var result = textToImagePipeline.Generate("A beautiful sunset over mountains", numInferenceSteps: 20, height: 256, width: 256);
        
        Assert.NotNull(result.ImageBytes);
        Assert.True(result.ImageBytes.Length > 0);
        Assert.Equal(256, result.Width);
        Assert.Equal(256, result.Height);
        
        // Verify it's a valid image by checking for PNG header
        Assert.True(result.ImageBytes.Length >= 8);
        // PNG header: 89 50 4E 47 0D 0A 1A 0A
        Assert.Equal(0x89, result.ImageBytes[0]);
        Assert.Equal(0x50, result.ImageBytes[1]);
        Assert.Equal(0x4E, result.ImageBytes[2]);
        Assert.Equal(0x47, result.ImageBytes[3]);
    }
}
