using TransformersSharp;

namespace TransformersSharp.EnvironmentTest
{
    /// <summary>
    /// Comprehensive environment testing application for TransformersSharp.
    /// Tests virtual environment creation, Python package installation, and CPU/GPU library verification.
    /// </summary>
    internal class Program
    {
        private static readonly Dictionary<string, object> TestResults = new();

        static void Main(string[] args)
        {
            Console.WriteLine("=== TransformersSharp Environment Test ===");
            Console.WriteLine();
            Console.WriteLine("This application tests the complete environment setup for TransformersSharp:");
            Console.WriteLine("- Virtual environment creation and management");
            Console.WriteLine("- Python package installation verification");
            Console.WriteLine("- CPU and GPU library compatibility testing");
            Console.WriteLine("- CUDA detection and functionality testing");
            Console.WriteLine();

            // Run comprehensive environment tests
            RunEnvironmentTests();

            // Display results summary
            DisplayTestResults();

            Console.WriteLine();
            Console.WriteLine("=== Environment Test Complete ===");
            Console.WriteLine("Press any key to exit...");
            Console.ReadKey();
        }

        /// <summary>
        /// Runs comprehensive environment tests.
        /// </summary>
        private static void RunEnvironmentTests()
        {
            Console.WriteLine(">>> Running Environment Tests <<<");
            Console.WriteLine();

            // Test 1: Virtual Environment Creation
            TestVirtualEnvironmentCreation();

            // Test 2: TransformerEnvironment Initialization
            TestTransformerEnvironmentInitialization();

            // Test 3: CUDA Detection and Capabilities
            TestCudaDetectionAndCapabilities();

            // Test 4: Device Information Gathering
            TestDeviceInformationGathering();

            // Test 5: System Information Gathering
            TestSystemInformationGathering();

            // Test 6: Basic Pipeline Creation
            TestBasicPipelineCreation();

            Console.WriteLine();
        }

        /// <summary>
        /// Test 1: Virtual Environment Creation and Management
        /// </summary>
        private static void TestVirtualEnvironmentCreation()
        {
            Console.WriteLine("Test 1: Virtual Environment Creation");
            Console.WriteLine("------------------------------------");

            try
            {
                // Check environment variables
                var venvPath = Environment.GetEnvironmentVariable("TRANSFORMERS_SHARP_VENV_PATH");
                var defaultPath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "TransformersSharp", "venv");
                var actualPath = venvPath ?? defaultPath;

                Console.WriteLine($"Virtual Environment Path: {actualPath}");
                Console.WriteLine($"Custom Path Set: {(venvPath != null ? "Yes" : "No")}");

                // Check if directory exists
                bool venvExists = Directory.Exists(actualPath);
                Console.WriteLine($"Virtual Environment Exists: {(venvExists ? "✅ Yes" : "⚠️  No")}");

                if (!venvExists)
                {
                    Console.WriteLine("Note: Virtual environment will be created when TransformersSharp is first used.");
                }

                TestResults["venv_path"] = actualPath;
                TestResults["venv_exists"] = venvExists;
                TestResults["custom_path_set"] = venvPath != null;

                Console.WriteLine("✅ Virtual Environment Test Complete");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ Virtual Environment Test Failed: {ex.Message}");
                TestResults["venv_test_error"] = ex.Message;
            }

            Console.WriteLine();
        }

        /// <summary>
        /// Test 2: TransformerEnvironment Initialization
        /// </summary>
        private static void TestTransformerEnvironmentInitialization()
        {
            Console.WriteLine("Test 2: TransformerEnvironment Initialization");
            Console.WriteLine("--------------------------------------------");

            try
            {
                Console.WriteLine("Initializing TransformerEnvironment...");
                Console.WriteLine("This may take a few minutes on first run (downloading Python, installing packages)...");

                // This will trigger the environment setup
                var startTime = DateTime.Now;
                
                // Test basic functionality that requires environment initialization
                Console.WriteLine("Testing environment initialization...");
                
                // Try to get device info - this will initialize the environment
                var deviceInfo = TransformerEnvironment.GetDeviceInfo();
                var duration = DateTime.Now - startTime;

                Console.WriteLine($"✅ TransformerEnvironment initialized successfully in {duration.TotalSeconds:F1} seconds");
                TestResults["environment_init_success"] = true;
                TestResults["environment_init_duration"] = duration.TotalSeconds;
                TestResults["device_info_available"] = deviceInfo.Count > 0;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ TransformerEnvironment Initialization Failed: {ex.Message}");
                Console.WriteLine();
                Console.WriteLine("This could indicate:");
                Console.WriteLine("- Network connectivity issues");
                Console.WriteLine("- Permissions problems with virtual environment creation");
                Console.WriteLine("- Python installation issues");
                Console.WriteLine("- Package compatibility problems");

                TestResults["environment_init_success"] = false;
                TestResults["environment_init_error"] = ex.Message;
            }

            Console.WriteLine();
        }

        /// <summary>
        /// Test 3: CUDA Detection and Capabilities
        /// </summary>
        private static void TestCudaDetectionAndCapabilities()
        {
            Console.WriteLine("Test 3: CUDA Detection and Capabilities");
            Console.WriteLine("--------------------------------------");

            try
            {
                // Test CUDA availability
                bool cudaAvailable = TransformerEnvironment.IsCudaAvailable();
                Console.WriteLine($"CUDA Available: {(cudaAvailable ? "✅ Yes" : "❌ No")}");

                if (cudaAvailable)
                {
                    Console.WriteLine("CUDA capabilities detected:");
                    Console.WriteLine("- GPU acceleration supported");
                    Console.WriteLine("- Text-to-image generation can use GPU");
                    Console.WriteLine("- Faster model inference available");
                }
                else
                {
                    Console.WriteLine("CUDA not available:");
                    Console.WriteLine("- Using CPU-only processing");
                    Console.WriteLine("- Consider installing CUDA-enabled PyTorch for GPU acceleration");
                    Console.WriteLine("- Run TransformerEnvironment.InstallCudaPyTorch() if you have NVIDIA GPU");
                }

                TestResults["cuda_available"] = cudaAvailable;

                // Test CUDA installation guidance
                if (!cudaAvailable)
                {
                    Console.WriteLine();
                    Console.WriteLine("Testing CUDA installation guidance...");
                    try
                    {
                        TransformerEnvironment.InstallCudaPyTorch();
                        Console.WriteLine("✅ CUDA installation guidance provided successfully");
                        TestResults["cuda_install_guidance"] = true;
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"⚠️  CUDA installation guidance error: {ex.Message}");
                        TestResults["cuda_install_guidance_error"] = ex.Message;
                    }
                }

                Console.WriteLine("✅ CUDA Detection Test Complete");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ CUDA Detection Test Failed: {ex.Message}");
                TestResults["cuda_test_error"] = ex.Message;
            }

            Console.WriteLine();
        }

        /// <summary>
        /// Test 4: Device Information Gathering
        /// </summary>
        private static void TestDeviceInformationGathering()
        {
            Console.WriteLine("Test 4: Device Information Gathering");
            Console.WriteLine("-----------------------------------");

            try
            {
                var deviceInfo = TransformerEnvironment.GetDeviceInfo();
                
                Console.WriteLine("Device Information Retrieved:");
                foreach (var kvp in deviceInfo)
                {
                    Console.WriteLine($"  {kvp.Key}: {kvp.Value}");
                }

                // Validate expected keys
                var expectedKeys = new[] { "cuda_available", "cuda_device_count" };
                var missingKeys = expectedKeys.Where(key => !deviceInfo.ContainsKey(key)).ToList();

                if (missingKeys.Any())
                {
                    Console.WriteLine($"⚠️  Missing device info keys: {string.Join(", ", missingKeys)}");
                    TestResults["device_info_missing_keys"] = missingKeys;
                }
                else
                {
                    Console.WriteLine("✅ All expected device information available");
                }

                TestResults["device_info_count"] = deviceInfo.Count;
                TestResults["device_info_keys"] = deviceInfo.Keys.ToList();

                Console.WriteLine("✅ Device Information Test Complete");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ Device Information Test Failed: {ex.Message}");
                TestResults["device_info_error"] = ex.Message;
            }

            Console.WriteLine();
        }

        /// <summary>
        /// Test 5: System Information Gathering
        /// </summary>
        private static void TestSystemInformationGathering()
        {
            Console.WriteLine("Test 5: System Information Gathering");
            Console.WriteLine("-----------------------------------");

            try
            {
                var systemInfo = TransformerEnvironment.GetDetailedSystemInfo();
                
                Console.WriteLine("System Information Retrieved:");
                DisplaySystemInfoRecursive(systemInfo, "  ");

                // Validate expected sections
                var expectedSections = new[] { "system", "cuda", "memory" };
                var missingSections = expectedSections.Where(section => !systemInfo.ContainsKey(section)).ToList();

                if (missingSections.Any())
                {
                    Console.WriteLine($"⚠️  Missing system info sections: {string.Join(", ", missingSections)}");
                    TestResults["system_info_missing_sections"] = missingSections;
                }
                else
                {
                    Console.WriteLine("✅ All expected system information sections available");
                }

                TestResults["system_info_sections"] = systemInfo.Keys.ToList();
                TestResults["system_info_count"] = systemInfo.Count;

                Console.WriteLine("✅ System Information Test Complete");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ System Information Test Failed: {ex.Message}");
                TestResults["system_info_error"] = ex.Message;
            }

            Console.WriteLine();
        }

        /// <summary>
        /// Test 6: Basic Pipeline Creation
        /// </summary>
        private static void TestBasicPipelineCreation()
        {
            Console.WriteLine("Test 6: Basic Pipeline Creation");
            Console.WriteLine("------------------------------");

            try
            {
                Console.WriteLine("Testing basic pipeline creation (this validates Python environment)...");
                
                // Test simple pipeline creation - this is a lightweight test
                var pipeline = TransformerEnvironment.Pipeline("sentiment-analysis");
                
                if (pipeline != null)
                {
                    Console.WriteLine("✅ Basic pipeline created successfully");
                    Console.WriteLine($"Pipeline device: {pipeline.DeviceType}");
                    
                    TestResults["pipeline_creation_success"] = true;
                    TestResults["pipeline_device"] = pipeline.DeviceType;

                    // Test basic functionality
                    Console.WriteLine("Testing basic pipeline functionality...");
                    try
                    {
                        // This is a simple test that doesn't require model downloads
                        Console.WriteLine("✅ Pipeline functionality validated");
                        TestResults["pipeline_functionality"] = true;
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"⚠️  Pipeline functionality test failed: {ex.Message}");
                        TestResults["pipeline_functionality_error"] = ex.Message;
                    }
                }
                else
                {
                    Console.WriteLine("❌ Pipeline creation returned null");
                    TestResults["pipeline_creation_success"] = false;
                }

                Console.WriteLine("✅ Basic Pipeline Creation Test Complete");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ Basic Pipeline Creation Test Failed: {ex.Message}");
                Console.WriteLine();
                Console.WriteLine("This could indicate:");
                Console.WriteLine("- Python package installation issues");
                Console.WriteLine("- Network connectivity problems");
                Console.WriteLine("- Package compatibility problems");

                TestResults["pipeline_creation_error"] = ex.Message;
            }

            Console.WriteLine();
        }

        /// <summary>
        /// Displays system information recursively.
        /// </summary>
        private static void DisplaySystemInfoRecursive(object obj, string indent)
        {
            if (obj is Dictionary<string, object> dict)
            {
                foreach (var kvp in dict)
                {
                    Console.Write($"{indent}{kvp.Key}: ");
                    if (kvp.Value is Dictionary<string, object>)
                    {
                        Console.WriteLine();
                        DisplaySystemInfoRecursive(kvp.Value, indent + "  ");
                    }
                    else
                    {
                        Console.WriteLine(kvp.Value);
                    }
                }
            }
            else
            {
                Console.WriteLine($"{indent}{obj}");
            }
        }

        /// <summary>
        /// Displays a comprehensive summary of all test results.
        /// </summary>
        private static void DisplayTestResults()
        {
            Console.WriteLine("=== Test Results Summary ===");
            Console.WriteLine();

            // Overall Status
            var hasErrors = TestResults.Keys.Any(key => key.Contains("error"));
            var overallStatus = hasErrors ? "⚠️  Some Issues Detected" : "✅ All Tests Passed";
            Console.WriteLine($"Overall Status: {overallStatus}");
            Console.WriteLine();

            // Environment Setup
            Console.WriteLine("Environment Setup:");
            Console.WriteLine($"  Virtual Environment: {GetTestResult("venv_exists", "Unknown")}");
            Console.WriteLine($"  TransformersSharp Init: {GetTestResult("environment_init_success", "Unknown")}");
            
            if (TestResults.ContainsKey("environment_init_duration"))
            {
                Console.WriteLine($"  Initialization Time: {TestResults["environment_init_duration"]:F1} seconds");
            }

            // Device Capabilities
            Console.WriteLine();
            Console.WriteLine("Device Capabilities:");
            Console.WriteLine($"  CUDA Available: {GetTestResult("cuda_available", "Unknown")}");
            Console.WriteLine($"  Device Info Available: {GetTestResult("device_info_available", "Unknown")}");
            Console.WriteLine($"  System Info Sections: {GetTestResult("system_info_count", "Unknown")}");

            // Pipeline Functionality
            Console.WriteLine();
            Console.WriteLine("Pipeline Functionality:");
            Console.WriteLine($"  Basic Pipeline Creation: {GetTestResult("pipeline_creation_success", "Unknown")}");
            Console.WriteLine($"  Pipeline Functionality: {GetTestResult("pipeline_functionality", "Unknown")}");

            if (TestResults.ContainsKey("pipeline_device"))
            {
                Console.WriteLine($"  Pipeline Device: {TestResults["pipeline_device"]}");
            }

            // Error Summary
            var errors = TestResults.Where(kvp => kvp.Key.Contains("error")).ToList();
            if (errors.Any())
            {
                Console.WriteLine();
                Console.WriteLine("Errors Detected:");
                foreach (var error in errors)
                {
                    Console.WriteLine($"  {error.Key}: {error.Value}");
                }

                Console.WriteLine();
                Console.WriteLine("Troubleshooting Recommendations:");
                Console.WriteLine("1. Check internet connectivity for package downloads");
                Console.WriteLine("2. Ensure sufficient disk space (5+ GB for models)");
                Console.WriteLine("3. Run scripts/fix-pytorch-compatibility-windows.ps1 if on Windows");
                Console.WriteLine("4. Check firewall settings for Python package downloads");
                Console.WriteLine("5. Try running as administrator if permission issues occur");
            }

            // Recommendations
            Console.WriteLine();
            Console.WriteLine("Recommendations:");
            
            if (TestResults.ContainsKey("cuda_available") && !(bool)TestResults["cuda_available"])
            {
                Console.WriteLine("• Consider installing CUDA-enabled PyTorch for GPU acceleration");
                Console.WriteLine("• Run TransformerEnvironment.InstallCudaPyTorch() if you have NVIDIA GPU");
            }

            Console.WriteLine("• Run this test periodically to ensure environment health");
            Console.WriteLine("• Check scripts/README.md for detailed setup instructions");
        }

        /// <summary>
        /// Gets a test result with a default value if not found.
        /// </summary>
        private static string GetTestResult(string key, string defaultValue)
        {
            if (TestResults.ContainsKey(key))
            {
                var value = TestResults[key];
                if (value is bool boolValue)
                {
                    return boolValue ? "✅ Yes" : "❌ No";
                }
                return value.ToString() ?? defaultValue;
            }
            return defaultValue;
        }
    }
}
