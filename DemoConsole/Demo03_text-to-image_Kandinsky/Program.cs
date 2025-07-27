using TransformersSharp;
using TransformersSharp.Pipelines;

var model = "kandinsky-community/kandinsky-2-2-decoder";
var pipeline = TextToImagePipeline.FromModel(model: model, device: "cpu");
// var pipeline = TextToImagePipeline.FromModel(model: model,
//     torchDtype: TorchDtype.Float16,
//     device: "cuda");
var result = pipeline.Generate(
    "A pixelated image of a beaver in Canada",
    numInferenceSteps: 30,
    guidanceScale: 0.75f,
    height: 256,  
    width: 256);  

var imageName = GetImageName(pipeline!.DeviceType);
File.WriteAllBytes(imageName, result.ImageBytes);
Console.WriteLine($"Image saved to: {imageName}");

string GetImageName(string deviceType, int width = 256, int height = 256)
{
    var desktopPath = Environment.GetFolderPath(Environment.SpecialFolder.Desktop);
    var folderPath = Path.Combine(desktopPath, "TransformersSharpImages");
    Directory.CreateDirectory(folderPath);

    // Remove or replace invalid filename characters (e.g., colon)
    var safeDeviceType = deviceType.Replace(":", "-").Replace("/", "-");
    var timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss_fff");
    var filename = $"image_{safeDeviceType}_{timestamp}_{width}x{height}.png"; 
    return Path.Combine(folderPath, filename);
}
