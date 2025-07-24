using TransformersSharp.Pipelines;

try
{
    var pipeline = TextToImagePipeline.FromModel(
        "kandinsky-community/kandinsky-2-2-decoder",
        device: "cuda");

    var result = pipeline.Generate(
        "A pixelated image of a beaver in Canada",
        numInferenceSteps: 30,
        guidanceScale: 7.5f,
        height: 512,
        width: 512);

    var desktopPath = Environment.GetFolderPath(Environment.SpecialFolder.Desktop);
    var folderPath = Path.Combine(desktopPath, "TransformersSharpImages");
    Directory.CreateDirectory(folderPath);
    var destinationPath = Path.Combine(folderPath, "image.png");
    File.WriteAllBytes(destinationPath, result.ImageBytes);

    Console.WriteLine($"Image generated successfully and saved to: {destinationPath}");
    Console.WriteLine($"Used device: {pipeline.DeviceType}");
}
catch (Exception ex) when (ex.Message.Contains("CUDA") || ex.Message.Contains("Torch not compiled with CUDA"))
{
    Console.WriteLine("CUDA not available, falling back to CPU...");

    var pipeline = TextToImagePipeline.FromModel(
        "kandinsky-community/kandinsky-2-2-decoder",
        trustRemoteCode: true, device: "cpu");

    var result = pipeline.Generate(
        "A pixelated image of a beaver in Canada",
        numInferenceSteps: 30,
        guidanceScale: 7.5f,
        height: 512,
        width: 512);

    var desktopPath = Environment.GetFolderPath(Environment.SpecialFolder.Desktop);
    var folderPath = Path.Combine(desktopPath, "TransformersSharpImages");
    Directory.CreateDirectory(folderPath);
    var destinationPath = Path.Combine(folderPath, "image.png");
    File.WriteAllBytes(destinationPath, result.ImageBytes);

    Console.WriteLine($"Image generated successfully and saved to: {destinationPath}");
    Console.WriteLine($"Used device: {pipeline.DeviceType}");
}