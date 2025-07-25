using CSnakes.Runtime;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using TransformersSharp.Pipelines;

namespace TransformersSharp
{
    /// <summary>
    /// Provides centralized access to Python-based machine learning environments and utilities for TransformersSharp.
    /// </summary>
    public static class TransformerEnvironment
    {
        private static readonly IPythonEnvironment? _env;
        private static readonly Lock _setupLock = new();
        private static readonly string[] _coreRequirements = 
        {
            "transformers",
            "sentence_transformers", 
            "torch",
            "torchvision", 
            "torchaudio",
            "pillow",
            "timm",
            "einops",
            "diffusers",
            "accelerate",
            "psutil",
            "safetensors",
            "scipy",
            "numpy"
            // Note: Standard torch packages included - pip will choose appropriate CPU/CUDA versions
            // based on system compatibility and available package variants
        };

        static TransformerEnvironment()
        {
            lock (_setupLock)
            {
                _env = InitializePythonEnvironment();
            }
        }

        /// <summary>
        /// Initializes the Python environment with proper configuration and dependencies.
        /// </summary>
        /// <returns>Configured Python environment</returns>
        private static IPythonEnvironment InitializePythonEnvironment()
        {
            var builder = Host.CreateDefaultBuilder()
                .ConfigureServices(services =>
                {
                    ConfigureLogging(services);
                    var paths = SetupPaths();
                    ConfigureEnvironmentVariables();
                    CreateRequirementsFile(paths.appDataPath);
                    CreateCudaInstallationScript(paths.appDataPath);
                    ConfigurePythonServices(services, paths.appDataPath, paths.venvPath);
                });

            var app = builder.Build();
            return app.Services.GetRequiredService<IPythonEnvironment>();
        }

        /// <summary>
        /// Configures logging for the Python environment.
        /// </summary>
        private static void ConfigureLogging(IServiceCollection services)
        {
            services.AddLogging(loggingBuilder =>
            {
                loggingBuilder.AddConsole();
                loggingBuilder.AddDebug();
                loggingBuilder.SetMinimumLevel(LogLevel.Debug);
            });
        }

        /// <summary>
        /// Sets up application and virtual environment paths.
        /// </summary>
        /// <returns>Tuple containing appDataPath and venvPath</returns>
        private static (string appDataPath, string venvPath) SetupPaths()
        {
            string appDataPath = Path.Join(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "TransformersSharp");
            string? envPath = Environment.GetEnvironmentVariable("TRANSFORMERS_SHARP_VENV_PATH");
            string venvPath = envPath ?? Path.Join(appDataPath, "venv");

            // Create the directory if it doesn't exist
            if (!Directory.Exists(appDataPath))
                Directory.CreateDirectory(appDataPath);

            // Save the TRANSFORMERS_SHARP_VENV_PATH for the current user
            Environment.SetEnvironmentVariable("TRANSFORMERS_SHARP_VENV_PATH", venvPath, EnvironmentVariableTarget.User);

            return (appDataPath, venvPath);
        }

        /// <summary>
        /// Configures environment variables to suppress compatibility warnings.
        /// </summary>
        private static void ConfigureEnvironmentVariables()
        {
            Environment.SetEnvironmentVariable("XFORMERS_MORE_DETAILS", "0", EnvironmentVariableTarget.Process);
            Environment.SetEnvironmentVariable("XFORMERS_DISABLED", "1", EnvironmentVariableTarget.Process);
        }

        /// <summary>
        /// Creates the requirements.txt file with core dependencies.
        /// </summary>
        private static void CreateRequirementsFile(string appDataPath)
        {
            string requirementsPath = Path.Join(appDataPath, "requirements.txt");
            File.WriteAllText(requirementsPath, string.Join('\n', _coreRequirements));

        }

        /// <summary>
        /// Creates the CUDA PyTorch installation script.
        /// </summary>
        private static void CreateCudaInstallationScript(string appDataPath)
        {
            string installCudaScript = Path.Join(appDataPath, "install_cuda_pytorch.py");
            string cudaInstallScript = GetCudaInstallationScriptContent();
            File.WriteAllText(installCudaScript, cudaInstallScript);
        }

        /// <summary>
        /// Gets the content for the CUDA installation script.
        /// </summary>
        /// <returns>Python script content for CUDA installation</returns>
        private static string GetCudaInstallationScriptContent()
        {
            return @"
import subprocess
import sys
import os

def install_pytorch_for_version(python_version):
    """"""Install PyTorch based on Python version.""""""
    print(f'Installing PyTorch for Python {python_version}...')
    
    if python_version == '3.12':
        return install_cpu_pytorch('Python 3.12 detected - installing CPU-only for compatibility')
    elif python_version in ['3.10', '3.11']:
        return install_cuda_pytorch_with_fallback(python_version)
    else:
        return install_cpu_pytorch(f'Python {python_version} - installing CPU-only for compatibility')

def install_cpu_pytorch(message):
    """"""Install CPU-only PyTorch.""""""
    print(message)
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'torch', 'torchvision', 'torchaudio', '--index-url', 'https://download.pytorch.org/whl/cpu'])
        print('CPU-only PyTorch installed successfully.')
        return True
    except subprocess.CalledProcessError as e:
        print(f'Failed to install PyTorch: {e}')
        return False

def install_cuda_pytorch_with_fallback(python_version):
    """"""Try CUDA PyTorch, fallback to CPU if needed.""""""
    print(f'Python {python_version} detected - attempting CUDA PyTorch installation...')
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'torch', 'torchvision', 'torchaudio', '--index-url', 'https://download.pytorch.org/whl/cu121'])
        print('CUDA PyTorch installed successfully.')
        
        # Try to install xformers for optimization
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'xformers'])
            print('xFormers installed successfully for optimized performance.')
        except subprocess.CalledProcessError:
            print('xFormers installation failed - proceeding without it.')
        return True
    except subprocess.CalledProcessError as e:
        print(f'CUDA PyTorch installation failed: {e}')
        print('Falling back to CPU-only version...')
        return install_cpu_pytorch('Installing CPU-only PyTorch as fallback')

def main():
    python_version = f'{sys.version_info.major}.{sys.version_info.minor}'
    print(f'Current Python version: {python_version}')
    
    success = install_pytorch_for_version(python_version)
    if success:
        print('PyTorch installation complete!')
    else:
        print('PyTorch installation failed!')
        sys.exit(1)

if __name__ == '__main__':
    main()
";
        }

        /// <summary>
        /// Configures Python services with the specified paths.
        /// </summary>
        private static void ConfigurePythonServices(IServiceCollection services, string appDataPath, string venvPath)
        {
            // Use Python 3.12 for now (CUDA support limited to CPU-only packages)
            // Future: Switch to Python 3.10/3.11 when CSnakes properly supports them
            services
                .WithPython()
                .WithHome(appDataPath)
                .WithVirtualEnvironment(venvPath)
                .WithUvInstaller()
                .FromRedistributable(); // Use default Python 3.12
        }

        /// <summary>
        /// Gets the Python environment instance.
        /// </summary>
        private static IPythonEnvironment Env => _env ?? throw new InvalidOperationException("Python environment is not initialized.");

        /// <summary>
        /// Gets the Transformers wrapper for Python interop.
        /// </summary>
        internal static ITransformersWrapper TransformersWrapper => Env.TransformersWrapper();
        
        /// <summary>
        /// Gets the Sentence Transformers wrapper for Python interop.
        /// </summary>
        internal static ISentenceTransformersWrapper SentenceTransformersWrapper => Env.SentenceTransformersWrapper();

        /// <summary>
        /// Login to Huggingface with a token.
        /// </summary>
        /// <param name="token">HuggingFace authentication token</param>
        public static void Login(string token)
        {
            var wrapperModule = Env.TransformersWrapper();
            wrapperModule.HuggingfaceLogin(token);
        }

        /// <summary>
        /// Creates a pipeline for the specified task and model.
        /// </summary>
        /// <param name="task">The task type (optional)</param>
        /// <param name="model">The model name (optional)</param>
        /// <param name="tokenizer">The tokenizer name (optional)</param>
        /// <param name="torchDtype">The torch data type (optional)</param>
        /// <returns>Configured pipeline</returns>
        public static Pipeline Pipeline(string? task = null, string? model = null, string? tokenizer = null, TorchDtype? torchDtype = null)
        {
            var wrapperModule = Env.TransformersWrapper();
            string? torchDtypeStr = torchDtype?.ToString() ?? null;
            var pipeline = wrapperModule.Pipeline(task, model, tokenizer, torchDtypeStr);

            return new Pipeline(pipeline);
        }

        /// <summary>
        /// Installs CUDA-enabled PyTorch. Call this method if you need GPU acceleration.
        /// </summary>
        /// <param name="executeAutomatically">If true, executes the installation automatically. If false, displays instructions only.</param>
        /// <returns>True if installation succeeded or was completed, false otherwise.</returns>
        public static bool InstallCudaPyTorch(bool executeAutomatically = true)
        {
            try
            {
                var paths = GetApplicationPaths();
                string installScript = Path.Join(paths.appDataPath, "install_cuda_pytorch.py");

                if (!File.Exists(installScript))
                {
                    Console.WriteLine($"Installation script not found at: {installScript}");
                    Console.WriteLine("Please ensure TransformersSharp is properly initialized.");
                    
                    if (executeAutomatically)
                    {
                        // Try to recreate the script
                        try
                        {
                            CreateCudaInstallationScript(paths.appDataPath);
                            Console.WriteLine("Installation script recreated successfully.");
                        }
                        catch (Exception ex)
                        {
                            Console.WriteLine($"Failed to recreate installation script: {ex.Message}");
                            DisplayManualInstallationInstructions();
                            return false;
                        }
                    }
                    else
                    {
                        return false;
                    }
                }

                if (executeAutomatically)
                {
                    return ExecuteCudaInstallation(paths, installScript);
                }
                else
                {
                    Console.WriteLine("Installing PyTorch and compatible packages...");
                    Console.WriteLine("This may take a few minutes...");
                    DisplayInstallationInstructions(paths);
                    return true; // Instructions displayed successfully
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Failed to install CUDA PyTorch: {ex.Message}");
                DisplayManualInstallationInstructions();
                return false;
            }
        }

        /// <summary>
        /// Executes the CUDA PyTorch installation automatically.
        /// </summary>
        /// <param name="paths">Application paths</param>
        /// <param name="installScript">Path to the installation script</param>
        /// <returns>True if installation succeeded, false otherwise</returns>
        private static bool ExecuteCudaInstallation((string appDataPath, string venvPath) paths, string installScript)
        {
            try
            {
                Console.WriteLine("🚀 Starting automatic CUDA PyTorch installation...");
                Console.WriteLine("This may take several minutes depending on your internet connection.");
                Console.WriteLine();

                // Get the Python executable from the virtual environment
                string pythonExecutable = GetPythonExecutablePath(paths.venvPath);
                
                if (!File.Exists(pythonExecutable))
                {
                    Console.WriteLine($"❌ Python executable not found at: {pythonExecutable}");
                    Console.WriteLine("Virtual environment may not be properly initialized.");
                    return false;
                }

                Console.WriteLine($"📍 Using Python: {pythonExecutable}");
                Console.WriteLine($"📍 Running script: {installScript}");
                Console.WriteLine();

                // Execute the installation script
                var processInfo = new System.Diagnostics.ProcessStartInfo
                {
                    FileName = pythonExecutable,
                    Arguments = $"\"{installScript}\"",
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true,
                    WorkingDirectory = paths.appDataPath
                };

                using var process = new System.Diagnostics.Process { StartInfo = processInfo };
                
                // Handle output in real-time
                process.OutputDataReceived += (sender, e) => {
                    if (!string.IsNullOrEmpty(e.Data))
                        Console.WriteLine($"📦 {e.Data}");
                };
                
                process.ErrorDataReceived += (sender, e) => {
                    if (!string.IsNullOrEmpty(e.Data))
                        Console.WriteLine($"⚠️  {e.Data}");
                };

                process.Start();
                process.BeginOutputReadLine();
                process.BeginErrorReadLine();
                
                process.WaitForExit();

                if (process.ExitCode == 0)
                {
                    Console.WriteLine();
                    Console.WriteLine("✅ CUDA PyTorch installation completed successfully!");
                    Console.WriteLine("🔄 Checking CUDA availability...");
                    
                    // Wait a moment for the installation to take effect
                    System.Threading.Thread.Sleep(2000);
                    
                    // Verify installation by checking CUDA availability
                    try
                    {
                        bool cudaAvailable = IsCudaAvailable();
                        if (cudaAvailable)
                        {
                            Console.WriteLine("🎉 CUDA is now available for GPU acceleration!");
                        }
                        else
                        {
                            Console.WriteLine("⚠️  Installation completed, but CUDA is still not available.");
                            Console.WriteLine("   This is normal if you don't have a compatible NVIDIA GPU.");
                            Console.WriteLine("   CPU-only PyTorch has been installed instead.");
                        }
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"⚠️  Could not verify CUDA availability after installation: {ex.Message}");
                    }
                    
                    return true;
                }
                else
                {
                    Console.WriteLine();
                    Console.WriteLine($"❌ Installation failed with exit code: {process.ExitCode}");
                    Console.WriteLine("Falling back to manual installation instructions...");
                    DisplayManualInstallationInstructions();
                    return false;
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ Automatic installation failed: {ex.Message}");
                Console.WriteLine("Falling back to manual installation instructions...");
                DisplayManualInstallationInstructions();
                return false;
            }
        }

        /// <summary>
        /// Gets the path to the Python executable in the virtual environment.
        /// </summary>
        /// <param name="venvPath">Path to the virtual environment</param>
        /// <returns>Path to the Python executable</returns>
        private static string GetPythonExecutablePath(string venvPath)
        {
            if (Environment.OSVersion.Platform == PlatformID.Win32NT)
            {
                return Path.Join(venvPath, "Scripts", "python.exe");
            }
            else
            {
                return Path.Join(venvPath, "bin", "python");
            }
        }

        /// <summary>
        /// Gets application paths for data and virtual environment.
        /// </summary>
        /// <returns>Tuple containing appDataPath and venvPath</returns>
        private static (string appDataPath, string venvPath) GetApplicationPaths()
        {
            string appDataPath = Path.Join(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "TransformersSharp");
            string venvPath = Environment.GetEnvironmentVariable("TRANSFORMERS_SHARP_VENV_PATH") ?? Path.Join(appDataPath, "venv");
            return (appDataPath, venvPath);
        }

        /// <summary>
        /// Displays installation instructions for the user.
        /// </summary>
        private static void DisplayInstallationInstructions((string appDataPath, string venvPath) paths)
        {
            Console.WriteLine("To install PyTorch with proper compatibility, please run:");
            Console.WriteLine($"1. Open PowerShell and navigate to: {paths.appDataPath}");
            Console.WriteLine($"2. Activate the virtual environment: {paths.venvPath}\\Scripts\\Activate.ps1");
            Console.WriteLine($"3. Run the installation script: python install_cuda_pytorch.py");
            Console.WriteLine("4. Restart your application after installation completes");
            Console.WriteLine();
            Console.WriteLine("The script will automatically choose the best PyTorch version for your Python installation.");
        }

        /// <summary>
        /// Displays manual installation instructions as fallback.
        /// </summary>
        private static void DisplayManualInstallationInstructions()
        {
            Console.WriteLine("You can manually install compatible PyTorch by running:");
            Console.WriteLine("For Python 3.12: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu");
            Console.WriteLine("For Python 3.10/3.11: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121");
            Console.WriteLine("Note: Avoid installing xformers with Python 3.12 to prevent compatibility warnings.");
        }

        /// <summary>
        /// Checks if CUDA is available for GPU acceleration.
        /// </summary>
        /// <returns>True if CUDA is available, false otherwise</returns>
        public static bool IsCudaAvailable()
        {
            try
            {
                var wrapperModule = Env.TransformersWrapper();
                return wrapperModule.IsCudaAvailableWrapper();
            }
            catch (Exception)
            {
                return false;
            }
        }

        /// <summary>
        /// Gets detailed information about available devices including CUDA capabilities.
        /// </summary>
        /// <returns>A dictionary containing device information</returns>
        public static Dictionary<string, object> GetDeviceInfo()
        {
            try
            {
                bool cudaAvailable = IsCudaAvailable();
                var result = new Dictionary<string, object>
                {
                    {"cuda_available", cudaAvailable},
                    {"cuda_device_count", cudaAvailable ? 1 : 0}
                };

                if (cudaAvailable)
                {
                    try
                    {
                        result["device_name"] = "CUDA Device";
                    }
                    catch
                    {
                        // Ignore errors getting additional info
                    }
                }

                return result;
            }
            catch (Exception ex)
            {
                return new Dictionary<string, object>
                {
                    {"error", ex.Message},
                    {"cuda_available", false}
                };
            }
        }

        /// <summary>
        /// Gets comprehensive system information including CPU, memory, and GPU details for performance analysis.
        /// </summary>
        /// <returns>A dictionary containing comprehensive system information</returns>
        public static Dictionary<string, object> GetDetailedSystemInfo()
        {
            try
            {
                var systemInfo = new Dictionary<string, object>
                {
                    ["system"] = CreateSystemInfo(),
                    ["cuda"] = CreateCudaInfo(),
                    ["memory"] = CreateMemoryInfo()
                };

                return systemInfo;
            }
            catch (Exception ex)
            {
                return new Dictionary<string, object>
                {
                    {"error", ex.Message},
                    {"available", false}
                };
            }
        }

        /// <summary>
        /// Creates system information dictionary.
        /// </summary>
        /// <returns>Dictionary with system information</returns>
        private static Dictionary<string, object> CreateSystemInfo()
        {
            return new Dictionary<string, object>
            {
                {"platform", Environment.OSVersion.ToString()},
                {"architecture", Environment.Is64BitOperatingSystem ? "64-bit" : "32-bit"},
                {"processor_count", Environment.ProcessorCount},
                {"dotnet_version", Environment.Version.ToString()}
            };
        }

        /// <summary>
        /// Creates CUDA information dictionary.
        /// </summary>
        /// <returns>Dictionary with CUDA information</returns>
        private static Dictionary<string, object> CreateCudaInfo()
        {
            bool cudaAvailable = IsCudaAvailable();
            return new Dictionary<string, object>
            {
                {"available", cudaAvailable},
                {"description", cudaAvailable ? "CUDA is available for GPU acceleration" : "CUDA not available - using CPU only"}
            };
        }

        /// <summary>
        /// Creates memory information dictionary.
        /// </summary>
        /// <returns>Dictionary with memory information</returns>
        private static Dictionary<string, object> CreateMemoryInfo()
        {
            return new Dictionary<string, object>
            {
                {"working_set_mb", Math.Round(Environment.WorkingSet / (1024.0 * 1024.0), 2)}
            };
        }

        /// <summary>
        /// Gets the default text-to-image model identifier.
        /// </summary>
        /// <returns>Default model identifier</returns>
        public static string GetDefaultTextToImageModel()
        {
            return TransformersWrapper.GetDefaultTextToImageModel();
        }

        /// <summary>
        /// Gets the pipeline class name for a specific text-to-image model.
        /// </summary>
        /// <param name="model">Model identifier</param>
        /// <returns>Pipeline class name</returns>
        public static string GetModelPipelineClassName(string model)
        {
            return TransformersWrapper.GetModelPipelineClassName(model);
        }

        /// <summary>
        /// Gets recommended generation settings for a specific text-to-image model.
        /// </summary>
        /// <param name="model">Model identifier</param>
        /// <returns>Dictionary with recommended settings</returns>
        public static Dictionary<string, object> GetRecommendedSettingsForModel(string model)
        {
            var pythonDict = TransformersWrapper.GetRecommendedSettingsForModel(model);
            
            // Convert Python dict to C# Dictionary
            var result = new Dictionary<string, object>();
            foreach (var key in pythonDict.Keys)
            {
                result[key.ToString()] = pythonDict[key];
            }
            return result;
        }

        /// <summary>
        /// Disposes of the Python environment.
        /// </summary>
        public static void Dispose()
        {
            _env?.Dispose();
        }
    }
}
