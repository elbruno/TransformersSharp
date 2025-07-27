using System;
using TransformersSharp;
using TransformersSharp.Pipelines;

namespace Demo10_text_to_image_benchmark
{
    /// <summary>
    /// Example program demonstrating the enhanced text-to-image model support.
    /// </summary>
    public class ModelSupportExample
    {
        private static readonly string[] SupportedModels = {
            "stable-diffusion-v1-5/stable-diffusion-v1-5",     // Default - balanced performance
            "stabilityai/stable-diffusion-2-1",                // Higher quality
            "kandinsky-community/kandinsky-2-2-decoder",       // Artistic style
            "DeepFloyd/IF-I-M-v1.0"                           // Photorealistic (requires HF token)
        };

        public static void DemonstrateModelSupport()
        {
            Console.WriteLine("=== TransformersSharp Enhanced Model Support Demo ===\n");

            // Check CUDA availability
            var deviceInfo = TransformerEnvironment.GetDeviceInfo();
            bool cudaAvailable = TransformerEnvironment.IsCudaAvailable();
            string device = cudaAvailable ? "cuda" : "cpu";
            
            Console.WriteLine($"Device: {device} ({(cudaAvailable ? "GPU Acceleration" : "CPU Only")})");
            Console.WriteLine();

            // Demonstrate each supported model
            foreach (var model in SupportedModels)
            {
                DemonstrateModel(model, device);
                Console.WriteLine();
            }
        }

        private static void DemonstrateModel(string modelId, string device)
        {
            Console.WriteLine($"=== Model: {modelId} ===");
            
            try
            {
                // Create pipeline with silent fallback
                var pipeline = TextToImagePipeline.FromModel(
                    model: modelId,
                    device: device,
                    silentDeviceFallback: true
                );

                Console.WriteLine($"✅ Pipeline created successfully");
                Console.WriteLine($"   Device: {pipeline.DeviceType}");
                
                // Generate a test image with model-appropriate settings
                var result = GenerateTestImage(pipeline, modelId);
                
                Console.WriteLine($"   Image generated: {result.Width}x{result.Height} pixels");
                Console.WriteLine($"   Data size: {result.ImageBytes.Length:N0} bytes");

                // Note: TextToImagePipeline doesn't implement IDisposable in the current version
                // pipeline.Dispose();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ Failed to create pipeline for {modelId}");
                Console.WriteLine($"   Error: {ex.Message}");
                
                if (ex.Message.Contains("compatibility"))
                {
                    Console.WriteLine("   This appears to be a package compatibility issue.");
                    Console.WriteLine("   Please follow the installation guide in the documentation.");
                }
            }
        }

        private static TextToImagePipeline.ImageGenerationResult GenerateTestImage(
            TextToImagePipeline pipeline, 
            string modelId)
        {
            // Use model-appropriate settings
            var settings = GetModelSettings(modelId);
            
            return pipeline.Generate(
                prompt: "A beautiful sunset over mountains",
                numInferenceSteps: settings.inferenceSteps,
                guidanceScale: settings.guidanceScale,
                height: settings.height,
                width: settings.width
            );
        }

        private static (int inferenceSteps, float guidanceScale, int height, int width) GetModelSettings(string modelId)
        {
            // Optimize settings based on model characteristics
            return modelId.ToLower() switch
            {
                var id when id.Contains("stable-diffusion-v1-5") => (20, 7.5f, 512, 512),
                var id when id.Contains("stable-diffusion-2-1") => (30, 8.0f, 768, 768),
                var id when id.Contains("kandinsky") => (25, 7.0f, 512, 512),
                var id when id.Contains("deepfloyd") => (50, 7.5f, 256, 256),  // Start smaller for IF
                _ => (20, 7.5f, 512, 512)  // Default
            };
        }
    }
}