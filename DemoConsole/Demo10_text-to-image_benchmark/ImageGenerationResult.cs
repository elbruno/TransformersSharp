namespace Demo10_text_to_image_benchmark;

public class ImageGenerationResult
{
    public string? Prompt { get; set; }

    public int? Width { get; set; }
    public int? Height { get; set; }
    public string? FileFullPath { get; set; }

    public string? FileName
    {
        get
        {
            if (string.IsNullOrEmpty(FileFullPath))
                return null;
            return Path.GetFileName(FileFullPath);
        }
    }
    public double TimeTakenSeconds { get; set; }
    public string? DeviceType { get; set; }
}
