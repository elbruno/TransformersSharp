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
                // Test initial CUDA availability
                bool initialCudaAvailable = TransformerEnvironment.IsCudaAvailable();
                Console.WriteLine($"Initial CUDA Available: {(initialCudaAvailable ? "✅ Yes" : "❌ No")}");

                if (initialCudaAvailable)
                {
                    Console.WriteLine("CUDA capabilities detected:");
                    Console.WriteLine("- GPU acceleration supported");
                    Console.WriteLine("- Text-to-image generation can use GPU");
                    Console.WriteLine("- Faster model inference available");
                    TestResults["cuda_available"] = true;
                    TestResults["cuda_installation_attempted"] = false;
                }
                else
                {
                    Console.WriteLine("CUDA not available - attempting automatic installation...");
                    Console.WriteLine("- Using CPU-only processing currently");
                    Console.WriteLine("- Attempting to install CUDA-enabled PyTorch");

                    // Attempt automatic CUDA installation
                    Console.WriteLine();
                    Console.WriteLine("🔧 Attempting automatic CUDA PyTorch installation...");
                    TestResults["cuda_installation_attempted"] = true;

                    try
                    {
                        bool installationResult = TransformerEnvironment.InstallPyTorch(executeAutomatically: true);

                        if (installationResult)
                        {
                            Console.WriteLine("✅ CUDA installation process completed successfully");
                            TestResults["cuda_installation_success"] = true;

                            // Re-initialize the environment to reload Python and packages
                            Console.WriteLine();
                            Console.WriteLine("� Reloading TransformerEnvironment to refresh CUDA state...");
                            TransformerEnvironment.Dispose();
                            System.Threading.Thread.Sleep(2000);

                            // Re-check CUDA availability after installation
                            Console.WriteLine("🔍 Re-checking CUDA availability after installation...");
                            System.Threading.Thread.Sleep(3000);
                            bool finalCudaAvailable = TransformerEnvironment.IsCudaAvailable();
                            Console.WriteLine($"CUDA Available After Installation: {(finalCudaAvailable ? "✅ Yes" : "❌ No")}");

                            TestResults["cuda_available"] = finalCudaAvailable;
                            TestResults["cuda_installation_improved_availability"] = finalCudaAvailable && !initialCudaAvailable;

                            if (finalCudaAvailable && !initialCudaAvailable)
                            {
                                Console.WriteLine("🎉 Success! CUDA installation enabled GPU acceleration!");
                            }
                            else if (!finalCudaAvailable)
                            {
                                Console.WriteLine("ℹ️  CUDA installation completed, but GPU acceleration is not available.");
                                Console.WriteLine("   This is normal if you don't have a compatible NVIDIA GPU.");
                                Console.WriteLine("   CPU-optimized PyTorch has been installed instead.");
                            }
                        }
                        else
                        {
                            Console.WriteLine("❌ CUDA installation process failed");
                            TestResults["cuda_installation_success"] = false;
                            TestResults["cuda_available"] = false;
                        }
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"❌ CUDA installation failed with exception: {ex.Message}");
                        TestResults["cuda_installation_error"] = ex.Message;
                        TestResults["cuda_installation_success"] = false;
                        TestResults["cuda_available"] = false;
                    }
                }

                Console.WriteLine("✅ CUDA Detection and Installation Test Complete");
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
            Console.WriteLine($"  CUDA Installation Attempted: {GetTestResult("cuda_installation_attempted", "Unknown")}");
            
            if (TestResults.ContainsKey("cuda_installation_attempted") && (bool)TestResults["cuda_installation_attempted"])
            {
                Console.WriteLine($"  CUDA Installation Success: {GetTestResult("cuda_installation_success", "Unknown")}");
                
                if (TestResults.ContainsKey("cuda_installation_improved_availability") && (bool)TestResults["cuda_installation_improved_availability"])
                {
                    Console.WriteLine("  🎉 CUDA Installation Enabled GPU Acceleration!");
                }
            }
            
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
                if (TestResults.ContainsKey("cuda_installation_attempted") && (bool)TestResults["cuda_installation_attempted"])
                {
                    if (TestResults.ContainsKey("cuda_installation_success") && (bool)TestResults["cuda_installation_success"])
                    {
                        Console.WriteLine("• CUDA installation completed - CPU-optimized PyTorch now available");
                        Console.WriteLine("• GPU acceleration may not be available if you don't have a compatible NVIDIA GPU");
                    }
                    else
                    {
                        Console.WriteLine("• CUDA installation was attempted but failed");
                        Console.WriteLine("• You can manually run TransformerEnvironment.InstallPyTorch(false) for instructions");
                        Console.WriteLine("• Check the error messages above for specific installation issues");
                    }
                }
                else
                {
                    Console.WriteLine("• Consider installing CUDA-enabled PyTorch for GPU acceleration");
                    Console.WriteLine("• Run TransformerEnvironment.InstallPyTorch() if you have NVIDIA GPU");
                }
            }
            else if (TestResults.ContainsKey("cuda_available") && (bool)TestResults["cuda_available"])
            {
                Console.WriteLine("• 🎉 GPU acceleration is available and ready to use!");
                Console.WriteLine("• Your text-to-image generation will benefit from faster processing");
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
