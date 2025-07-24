
using System.Diagnostics;
using ConsoleApp4;
using TransformersSharp.Pipelines;


// ...existing code...

Console.WriteLine("=== TransformersSharp CUDA Performance Test ===");
Console.WriteLine();

// prompt to be user 
var imagePrompt = "A pixelated image of a beaver in Canada.";
Console.WriteLine($"Prompt: {imagePrompt}");
Console.WriteLine($"Image size: 256x256 pixels");
Console.WriteLine();


// Test with CUDA

// ...existing code...
Console.WriteLine("Testing with CUDA device...");
try
{
    var cudaResult = ImageGenerator.GenerateImage("cuda", imagePrompt);
    Console.WriteLine($"✅ CUDA generation completed in: {cudaResult.TimeTakenSeconds:F2} seconds");
    Console.WriteLine($"Image saved: {cudaResult.FileGenerated}");
}
catch (Exception ex)
{
    Console.WriteLine($"❌ CUDA test failed: {ex.Message}");
}

Console.WriteLine();
Console.WriteLine("Testing with CPU device...");
try
{
    var cpuResult = ImageGenerator.GenerateImage("cpu", imagePrompt);
    Console.WriteLine($"✅ CPU generation completed in: {cpuResult.TimeTakenSeconds:F2} seconds");
    Console.WriteLine($"Image saved: {cpuResult.FileGenerated}");
}
catch (Exception ex)
{
    Console.WriteLine($"❌ CPU test failed: {ex.Message}");
}

Console.WriteLine();
Console.WriteLine("=== Test Complete ===");
