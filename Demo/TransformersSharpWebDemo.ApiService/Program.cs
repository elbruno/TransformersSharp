using TransformersSharp.MEAI;
using TransformersSharp.Pipelines;
using static TransformersSharp.Pipelines.ObjectDetectionPipeline;

var builder = WebApplication.CreateBuilder(args);

// Add service defaults & Aspire client integrations.
builder.AddServiceDefaults();

// Add services to the container.
builder.Services.AddProblemDetails();

// Learn more about configuring OpenAPI at https://aka.ms/aspnet/openapi
builder.Services.AddOpenApi();

// Configure detailed logging
builder.Logging.ClearProviders();
builder.Logging.AddConsole();
builder.Logging.AddDebug();
builder.Logging.SetMinimumLevel(LogLevel.Debug);

// add detailed logging

var app = builder.Build();

// Configure the HTTP request pipeline.
app.UseExceptionHandler();

if (app.Environment.IsDevelopment())
{
    app.MapOpenApi();
}

var logger = app.Services.GetRequiredService<ILoggerFactory>().CreateLogger("ApiService");

var objectDetectionPipeline = ObjectDetectionPipeline.FromModel("facebook/detr-resnet-50");
var speechToTextClient = SpeechToTextClient.FromModel("openai/whisper-small");

app.MapPost("/detect", (DetectRequest r) =>
{
    logger.LogInformation("/detect called with Url: {Url}", r.Url);
    try
    {
        var result = objectDetectionPipeline.Detect(r.Url);
        logger.LogDebug("Detection result: {@Result}", result);
        return result;
    }
    catch (Exception ex)
    {
        logger.LogError(ex, "Error in /detect endpoint for Url: {Url}", r.Url);
        throw;
    }

}).Accepts<DetectRequest>("application/json")
    .Produces<DetectionResult>(StatusCodes.Status200OK)
    .WithName("Detect");

app.MapPost("/transcribe", async (HttpRequest request) =>
{
    logger.LogInformation("/transcribe called");
    if (!request.HasFormContentType)
    {
        logger.LogWarning("/transcribe called without form content type");
        return Results.BadRequest("File upload required");
    }

    var form = await request.ReadFormAsync();

    string output = string.Empty;
    foreach (var file in form.Files)
    {
        logger.LogInformation("Processing file: {FileName}, Size: {Length}", file.FileName, file.Length);
        using var stream = file.OpenReadStream();
        try
        {
            var text = await speechToTextClient.GetTextAsync(stream);
            output += text + "\n";
        }
        catch (Exception ex)
        {
            logger.LogError(ex, "Error processing file: {FileName}", file.FileName);
            return Results.BadRequest($"Error processing file: {file.FileName}");
        }
    }

    return Results.Ok(output);
});

app.MapDefaultEndpoints();

app.Run();

public record DetectRequest(string Url)
{
}
