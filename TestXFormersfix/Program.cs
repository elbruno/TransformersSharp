using TransformersSharp;

Console.WriteLine("=== TransformersSharp xFormers Compatibility Test ===");
Console.WriteLine();

try
{
    // Set environment variables to suppress xFormers warnings before initialization
    Environment.SetEnvironmentVariable("XFORMERS_MORE_DETAILS", "0");
    Environment.SetEnvironmentVariable("XFORMERS_DISABLED", "1");

    Console.WriteLine("Testing TransformerEnvironment initialization...");

    // Test basic device info - this should not show xFormers warnings anymore
    var deviceInfo = TransformerEnvironment.GetDeviceInfo();

    Console.WriteLine("✅ TransformerEnvironment initialized successfully");
    Console.WriteLine($"CUDA Available: {deviceInfo.GetValueOrDefault("cuda_available", false)}");

    if (deviceInfo.ContainsKey("error"))
    {
        Console.WriteLine($"Note: {deviceInfo["error"]}");
    }

    Console.WriteLine();
    Console.WriteLine("✅ Test completed - xFormers warnings should be suppressed");
}
catch (Exception ex)
{
    Console.WriteLine($"❌ Test failed: {ex.Message}");
    Console.WriteLine();
    Console.WriteLine("This indicates the Python environment needs to be rebuilt.");
    Console.WriteLine("Please run the fix script: .\\scripts\\fix-pytorch-compatibility-windows.ps1");
}

Console.WriteLine();
Console.WriteLine("Press any key to exit...");
Console.ReadKey();
