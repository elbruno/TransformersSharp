using TransformersSharp.Pipelines;
using System.IO;

namespace TransformersSharp.Examples;

/// <summary>
/// Example demonstrating how to use the TextToImagePipeline
/// </summary>
public static class TextToImageExample
{
    /// <summary>
    /// Generates an image from text and saves it to a file.
    /// This example uses the Kandinsky 2.2 decoder model for text-to-image generation.
    /// </summary>
    /// <param name="prompt">The text prompt to generate an image from</param>
    /// <param name="outputPath">Path where to save the generated image</param>
    /// <param name="width">Width of the generated image (default: 512)</param>
    /// <param name="height">Height of the generated image (default: 512)</param>
    /// <param name="numInferenceSteps">Number of inference steps (default: 20 for faster generation)</param>
    /// <param name="guidanceScale">Guidance scale for generation (default: 7.5)</param>
    public static void GenerateImage(
        string prompt, 
        string outputPath,
        int width = 512, 
        int height = 512, 
        int numInferenceSteps = 20, 
        float guidanceScale = 7.5f)
    {
        // Create the text-to-image pipeline using the Kandinsky model
        var pipeline = TextToImagePipeline.FromModel(
            "kandinsky-community/kandinsky-2-2-decoder", 
            trustRemoteCode: true);

        Console.WriteLine($"Generating image for prompt: '{prompt}'");
        Console.WriteLine($"Image dimensions: {width}x{height}");
        Console.WriteLine($"Inference steps: {numInferenceSteps}");

        // Generate the image
        var result = pipeline.Generate(
            prompt, 
            numInferenceSteps: numInferenceSteps,
            guidanceScale: guidanceScale,
            height: height, 
            width: width);

        // Save the image to file
        File.WriteAllBytes(outputPath, result.ImageBytes);
        
        Console.WriteLine($"Image generated and saved to: {outputPath}");
        Console.WriteLine($"Image size: {result.ImageBytes.Length} bytes");
        Console.WriteLine($"Dimensions: {result.Width}x{result.Height}");
    }

    /// <summary>
    /// Example usage of the TextToImagePipeline
    /// </summary>
    public static void RunExample()
    {
        try
        {
            GenerateImage(
                "A beautiful sunset over mountains with a lake reflection",
                "generated_image.png",
                width: 256,
                height: 256,
                numInferenceSteps: 20
            );
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error generating image: {ex.Message}");
        }
    }
}