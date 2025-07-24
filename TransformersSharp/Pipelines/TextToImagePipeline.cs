using CSnakes.Runtime.Python;
using TransformersSharp.Models;

namespace TransformersSharp.Pipelines;

public class TextToImagePipeline : Pipeline
{
    public struct ImageGenerationResult
    {
        public byte[] ImageBytes { get; set; }
        public int Width { get; set; }
        public int Height { get; set; }
    }

    internal TextToImagePipeline(PyObject pipelineObject) : base(pipelineObject)
    {
    }

    public static TextToImagePipeline FromModel(string model, TorchDtype? torchDtype = null, string? device = null, bool trustRemoteCode = false)
    {
        return new TextToImagePipeline(TransformerEnvironment.TransformersWrapper.Pipeline(
            "text-to-image",
            model,
            null,
            torchDtype?.ToString(),
            device,
            trustRemoteCode));
    }

    /// <summary>
    /// Generates an image from a text prompt using the text-to-image pipeline.
    /// </summary>
    /// <param name="prompt">The text prompt for image generation</param>
    /// <param name="numInferenceSteps">Number of inference steps (default: 50)</param>
    /// <param name="guidanceScale">Guidance scale for generation (default: 7.5)</param>
    /// <param name="height">Height of generated image (default: 512)</param>
    /// <param name="width">Width of generated image (default: 512)</param>
    /// <returns>Generated image result</returns>
    public ImageGenerationResult Generate(string prompt, int numInferenceSteps = 50, float guidanceScale = 7.5f, int height = 512, int width = 512)
    {
        var imageBuffer = TransformerEnvironment.TransformersWrapper.InvokeTextToImagePipeline(
            PipelineObject, 
            prompt, 
            numInferenceSteps, 
            guidanceScale, 
            height, 
            width);

        return new ImageGenerationResult
        {
            ImageBytes = imageBuffer.AsByteReadOnlySpan().ToArray(),
            Width = width,
            Height = height
        };
    }
}