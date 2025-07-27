using CSnakes.Runtime.Python;
using TransformersSharp.Models;

namespace TransformersSharp.Pipelines;

/// <summary>
/// A pipeline for generating images from text prompts using text-to-image models.
/// This pipeline uses diffusers library for text-to-image generation when the task is "text-to-image".
/// </summary>
public class TextToImagePipeline : Pipeline
{
    /// <summary>
    /// Represents the result of image generation, containing the image data and dimensions.
    /// </summary>
    public struct ImageGenerationResult
    {
        /// <summary>
        /// The generated image as a byte array in PNG format.
        /// </summary>
        public byte[] ImageBytes { get; set; }

        /// <summary>
        /// The width of the generated image in pixels.
        /// </summary>
        public int Width { get; set; }

        /// <summary>
        /// The height of the generated image in pixels.
        /// </summary>
        public int Height { get; set; }
    }

    internal TextToImagePipeline(PyObject pipelineObject) : base(pipelineObject)
    {
    }

    /// <summary>
    /// Creates a new TextToImagePipeline from a pre-trained model.
    /// </summary>
    /// <param name="model">The model identifier (e.g., "kandinsky-community/kandinsky-2-2-decoder")</param>
    /// <param name="torchDtype">Optional torch data type for model weights</param>
    /// <param name="device">Optional device to run the model on (e.g., "cuda", "cpu")</param>
    /// <param name="trustRemoteCode">Whether to trust remote code in the model</param>
    /// <param name="silentDeviceFallback">If true, suppresses warnings when falling back from CUDA to CPU</param>
    /// <param name="huggingFaceToken">Optional HuggingFace token for accessing gated models (e.g., FLUX.1-dev)</param>
    /// <returns>A new TextToImagePipeline instance</returns>
    public static TextToImagePipeline FromModel(string model, TorchDtype? torchDtype = null, string? device = null, bool trustRemoteCode = false, bool silentDeviceFallback = false, string? huggingFaceToken = null)
    {
        // Authenticate with HuggingFace if token is provided
        if (!string.IsNullOrEmpty(huggingFaceToken))
        {
            TransformerEnvironment.Login(huggingFaceToken);
        }

        return new TextToImagePipeline(TransformerEnvironment.TransformersWrapper.Pipeline(
            "text-to-image",
            model,
            null,
            torchDtype?.ToString(),
            device,
            trustRemoteCode,
            silentDeviceFallback));
    }

    /// <summary>
    /// Creates a new TextToImagePipeline using the default model (Kandinsky 2.2).
    /// </summary>
    /// <param name="torchDtype">Optional torch data type for model weights</param>
    /// <param name="device">Optional device to run the model on (e.g., "cuda", "cpu")</param>
    /// <param name="trustRemoteCode">Whether to trust remote code in the model</param>
    /// <param name="silentDeviceFallback">If true, suppresses warnings when falling back from CUDA to CPU</param>
    /// <param name="huggingFaceToken">Optional HuggingFace token for accessing gated models</param>
    /// <returns>A new TextToImagePipeline instance</returns>
    public static TextToImagePipeline FromModel(TorchDtype? torchDtype = null, string? device = null, bool trustRemoteCode = false, bool silentDeviceFallback = false, string? huggingFaceToken = null)
    {
        return FromModel("kandinsky-community/kandinsky-2-2-decoder", torchDtype, device, trustRemoteCode, silentDeviceFallback, huggingFaceToken);
    }

    /// <summary>
    /// Generates an image from a text prompt using the text-to-image pipeline.
    /// </summary>
    /// <param name="prompt">The text prompt for image generation</param>
    /// <param name="numInferenceSteps">Number of inference steps (default: 50)</param>
    /// <param name="guidanceScale">Guidance scale for generation (default: 7.5)</param>
    /// <param name="height">Height of generated image (default: 512)</param>
    /// <param name="width">Width of generated image (default: 512)</param>
    /// <param name="maxSequenceLength">Maximum sequence length for FLUX models (default: null)</param>
    /// <param name="seed">Random seed for reproducible generation (default: null)</param>
    /// <param name="enableModelCpuOffload">Enable CPU offloading for memory optimization (default: false)</param>
    /// <returns>Generated image result containing image bytes and dimensions</returns>
    public ImageGenerationResult Generate(string prompt, int numInferenceSteps = 50, float guidanceScale = 7.5f, int height = 512, int width = 512, int? maxSequenceLength = null, int? seed = null, bool enableModelCpuOffload = false)
    {
        var imageBuffer = TransformerEnvironment.TransformersWrapper.InvokeTextToImagePipeline(
            PipelineObject,
            prompt,
            numInferenceSteps,
            guidanceScale,
            height,
            width,
            maxSequenceLength,
            seed,
            enableModelCpuOffload);

        return new ImageGenerationResult
        {
            ImageBytes = imageBuffer.AsByteReadOnlySpan().ToArray(),
            Width = width,
            Height = height
        };
    }
}