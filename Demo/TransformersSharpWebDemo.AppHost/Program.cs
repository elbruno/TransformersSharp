var builder = DistributedApplication.CreateBuilder(args);

var apiService = builder.AddProject<Projects.TransformersSharpWebDemo_ApiService>("apiservice")
    .WithExternalHttpEndpoints();

builder.AddProject<Projects.TransformersSharpWebDemo_Web>("webfrontend")
    .WithExternalHttpEndpoints()
    .WithReference(apiService)
    .WaitFor(apiService);

builder.Build().Run();
