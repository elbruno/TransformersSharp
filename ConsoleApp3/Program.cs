using TransformersSharp;
using TransformersSharp.Pipelines;

//var model = "kandinsky-community/kandinsky-2-2-decoder";
var model = "sd-legacy/stable-diffusion-v1-5";

var pipeline = TextToImagePipeline.FromModel(
    model: model,
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
