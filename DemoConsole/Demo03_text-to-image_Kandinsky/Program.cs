using TransformersSharp;
using TransformersSharp.Pipelines;

var model = "kandinsky-community/kandinsky-2-2-decoder";
var pipeline = TextToImagePipeline.FromModel(model: model, device: "cuda");
var result = pipeline.Generate(
    "A pixelated image of a beaver in Canada",
    numInferenceSteps: 30,
    guidanceScale: 7.5f,
    height: 256,  // Updated to 256x256
    width: 256);  // Updated to 256x256

var desktopPath = Environment.GetFolderPath(Environment.SpecialFolder.Desktop);
var folderPath = Path.Combine(desktopPath, "TransformersSharpImages");
Directory.CreateDirectory(folderPath);
var timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss_fff");
var filename = $"image_{pipeline!.DeviceType}_{timestamp}_256x256.png";
var destinationPath = Path.Combine(folderPath, filename);
File.WriteAllBytes(destinationPath, result.ImageBytes);
