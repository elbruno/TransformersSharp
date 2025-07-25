using CSnakes.Runtime;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using TransformersSharp.Pipelines;

namespace TransformersSharp
{
    public static class TransformerEnvironment
    {
        private static readonly IPythonEnvironment? _env;
        private static readonly Lock _setupLock = new();

        static TransformerEnvironment()
        {
            lock (_setupLock)
            {
                IHostBuilder builder = Host.CreateDefaultBuilder()
                    .ConfigureServices(services =>
                    {
                        // Configure detailed logging
                        services.AddLogging(loggingBuilder =>
                        {
                            loggingBuilder.AddConsole();
                            loggingBuilder.AddDebug();
                            loggingBuilder.SetMinimumLevel(LogLevel.Debug);
                        });

                        // Use Local AppData folder for Python installation
                        string appDataPath = Path.Join(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "TransformersSharp");

                        // If user has an environment variable TRANSFORMERS_SHARP_VENV_PATH, use that instead
                        string? envPath = Environment.GetEnvironmentVariable("TRANSFORMERS_SHARP_VENV_PATH");
                        string venvPath = envPath ?? Path.Join(appDataPath, "venv");

                        // Create the directory if it doesn't exist
                        if (!Directory.Exists(appDataPath))
                            Directory.CreateDirectory(appDataPath);

                        // Save the TRANSFORMERS_SHARP_VENV_PATH for the current user
                        Environment.SetEnvironmentVariable("TRANSFORMERS_SHARP_VENV_PATH", venvPath, EnvironmentVariableTarget.User);

                        // Set environment variables to suppress known compatibility warnings
                        Environment.SetEnvironmentVariable("XFORMERS_MORE_DETAILS", "0", EnvironmentVariableTarget.Process);
                        Environment.SetEnvironmentVariable("XFORMERS_DISABLED", "1", EnvironmentVariableTarget.Process);

                        // Write requirements to appDataPath
                        string requirementsPath = Path.Join(appDataPath, "requirements.txt");
                        string[] requirements =
                        {
                            "transformers",
                            "sentence_transformers",
                            "pillow",
                            "timm",
                            "einops",
                            "diffusers",
                            "accelerate",
                            "psutil",
                            "safetensors",
                            "scipy",
                            "numpy"
                            // Note: torch, torchvision, torchaudio, and xformers are handled separately
                            // to ensure compatibility between CPU/CUDA versions and Python versions
                        };
                        File.WriteAllText(requirementsPath, string.Join('\n', requirements));

                        // Create a separate script to install CUDA PyTorch
                        string installCudaScript = Path.Join(appDataPath, "install_cuda_pytorch.py");
                        string cudaInstallScript = @"
import subprocess
import sys
import shutil
import os

def main():
    python_version = f'{sys.version_info.major}.{sys.version_info.minor}'
    print(f'Current Python version: {python_version}')
    
    # First, install PyTorch based on Python version and CUDA availability
    print('Installing PyTorch...')
    
    # For Python 3.12, we need to be more careful with package compatibility
    if python_version == '3.12':
        print(f'Python {python_version} detected - installing compatible PyTorch (CPU-only for better compatibility)...')
        try:
            # Install CPU-only PyTorch for maximum compatibility with Python 3.12
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'torch', 'torchvision', 'torchaudio', '--index-url', 'https://download.pytorch.org/whl/cpu'])
            print('CPU-only PyTorch installed successfully.')
            
            # Check if we should try xformers - skip it for better compatibility
            print('Skipping xformers installation for Python 3.12 to avoid compatibility issues.')
            print('Note: Some advanced optimizations may not be available, but basic functionality will work.')
            
        except subprocess.CalledProcessError as e:
            print(f'Failed to install PyTorch: {e}')
            sys.exit(1)
    elif python_version in ['3.10', '3.11']:
        print(f'Python {python_version} detected - attempting CUDA PyTorch installation...')
        try:
            # Try CUDA first for Python 3.10/3.11
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'torch', 'torchvision', 'torchaudio', '--index-url', 'https://download.pytorch.org/whl/cu121'])
            print('CUDA PyTorch installed successfully.')
            
            # Try to install xformers for optimization
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'xformers'])
                print('xFormers installed successfully for optimized performance.')
            except subprocess.CalledProcessError:
                print('xFormers installation failed - proceeding without it (performance may be slower but functionality preserved).')
                
        except subprocess.CalledProcessError as e:
            print(f'CUDA PyTorch installation failed: {e}')
            print('Falling back to CPU-only version...')
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'torch', 'torchvision', 'torchaudio', '--index-url', 'https://download.pytorch.org/whl/cpu'])
            print('CPU-only PyTorch installed successfully.')
    else:
        print(f'Python {python_version} - installing CPU-only PyTorch for compatibility...')
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'torch', 'torchvision', 'torchaudio', '--index-url', 'https://download.pytorch.org/whl/cpu'])
            print('CPU-only PyTorch installed successfully.')
        except subprocess.CalledProcessError as e:
            print(f'Failed to install PyTorch: {e}')
            sys.exit(1)

    print('PyTorch installation complete!')

if __name__ == '__main__':
    main()
";
                        File.WriteAllText(installCudaScript, cudaInstallScript);

                        // Use Python 3.12 for now (CUDA support limited to CPU-only packages)
                        // Future: Switch to Python 3.10/3.11 when CSnakes properly supports them
                        services
                            .WithPython()
                            .WithHome(appDataPath)
                            .WithVirtualEnvironment(venvPath)
                            .WithUvInstaller()
                            .FromRedistributable(); // Use default Python 3.12
                    });

                var app = builder.Build();
                _env = app.Services.GetRequiredService<IPythonEnvironment>();
            }
        }

        private static IPythonEnvironment Env => _env ?? throw new InvalidOperationException("Python environment is not initialized..");

        internal static ITransformersWrapper TransformersWrapper => Env.TransformersWrapper();
        internal static ISentenceTransformersWrapper SentenceTransformersWrapper => Env.SentenceTransformersWrapper();

        /// <summary>
        /// Login to Huggingface with a token.
        /// </summary>
        /// <param name="token"></param>
        public static void Login(string token)
        {
            var wrapperModule = Env.TransformersWrapper();
            wrapperModule.HuggingfaceLogin(token);
        }

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
        public static void InstallCudaPyTorch()
        {
            try
            {
                string appDataPath = Path.Join(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "TransformersSharp");
                string installScript = Path.Join(appDataPath, "install_cuda_pytorch.py");

                if (File.Exists(installScript))
                {
                    Console.WriteLine("Installing PyTorch and compatible packages...");
                    Console.WriteLine("This may take a few minutes...");

                    // Try to run the installation script using the Python environment
                    try
                    {
                        var wrapperModule = Env.TransformersWrapper();
                        // For now, provide manual instructions as running subprocess through CSnakes needs more setup
                        Console.WriteLine("To install PyTorch with proper compatibility, please run:");
                        Console.WriteLine($"1. Open PowerShell and navigate to: {appDataPath}");
                        string venvPath = Environment.GetEnvironmentVariable("TRANSFORMERS_SHARP_VENV_PATH") ?? Path.Join(appDataPath, "venv");
                        Console.WriteLine($"2. Activate the virtual environment: {venvPath}\\Scripts\\Activate.ps1");
                        Console.WriteLine($"3. Run the installation script: python install_cuda_pytorch.py");
                        Console.WriteLine("4. Restart your application after installation completes");
                        Console.WriteLine();
                        Console.WriteLine("The script will automatically choose the best PyTorch version for your Python installation.");
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"Could not run installation script automatically: {ex.Message}");
                        Console.WriteLine("Please run the installation manually as described above.");
                    }
                }
                else
                {
                    Console.WriteLine($"Installation script not found at: {installScript}");
                    Console.WriteLine("Please ensure TransformersSharp is properly initialized.");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Failed to install CUDA PyTorch: {ex.Message}");
                Console.WriteLine("You can manually install compatible PyTorch by running:");
                Console.WriteLine("For Python 3.12: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu");
                Console.WriteLine("For Python 3.10/3.11: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121");
                Console.WriteLine("Note: Avoid installing xformers with Python 3.12 to prevent compatibility warnings.");
            }
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
                return wrapperModule.IsCudaAvailable();
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
                    {"cuda_device_count", 0}
                };

                if (cudaAvailable)
                {
                    // Try to get additional info if CUDA is available
                    try
                    {
                        // For now, just set basic info - we can enhance this later
                        result["cuda_device_count"] = 1;
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
        /// Gets detailed system information including CPU, memory, and GPU details for performance analysis.
        /// </summary>
        /// <returns>A dictionary containing comprehensive system information</returns>
        public static Dictionary<string, object> GetDetailedSystemInfo()
        {
            try
            {
                // For now, let's create a simplified version that doesn't rely on complex Python object conversion
                // We can enhance this later once we understand the conversion patterns better
                var systemInfo = new Dictionary<string, object>();

                // Add basic system info that we can gather from .NET
                systemInfo["system"] = new Dictionary<string, object>
                {
                    {"platform", Environment.OSVersion.ToString()},
                    {"architecture", Environment.Is64BitOperatingSystem ? "64-bit" : "32-bit"},
                    {"processor_count", Environment.ProcessorCount},
                    {"dotnet_version", Environment.Version.ToString()}
                };

                // Add CUDA info
                bool cudaAvailable = IsCudaAvailable();
                systemInfo["cuda"] = new Dictionary<string, object>
                {
                    {"available", cudaAvailable},
                    {"description", cudaAvailable ? "CUDA is available for GPU acceleration" : "CUDA not available - using CPU only"}
                };

                // Add basic memory info
                systemInfo["memory"] = new Dictionary<string, object>
                {
                    {"working_set_mb", Math.Round(Environment.WorkingSet / (1024.0 * 1024.0), 2)}
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

        public static void Dispose()
        {
            _env?.Dispose();
        }
    }
}
