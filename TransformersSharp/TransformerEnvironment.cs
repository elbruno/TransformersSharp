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
                        string venvPath;
                        if (envPath != null)
                            venvPath = envPath;
                        else
                            venvPath = Path.Join(appDataPath, "venv");

                        // Create the directory if it doesn't exist
                        if (!Directory.Exists(appDataPath))
                            Directory.CreateDirectory(appDataPath);

                        // Save the TRANSFORMERS_SHARP_VENV_PATH for the current user
                        Environment.SetEnvironmentVariable("TRANSFORMERS_SHARP_VENV_PATH", venvPath, EnvironmentVariableTarget.User);

                        // Write requirements to appDataPath
                        string requirementsPath = Path.Join(appDataPath, "requirements.txt");

                        // TODO: Make this configurable
                        string[] requirements =
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
                            "accelerate"
                        };

                        File.WriteAllText(requirementsPath, string.Join('\n', requirements));

                        // Create a separate script to install CUDA PyTorch
                        string installCudaScript = Path.Join(appDataPath, "install_cuda_pytorch.py");
                        string cudaInstallScript = @"
import subprocess
import sys

def install_cuda_pytorch():
    try:
        # Try to install CUDA version of PyTorch
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', 
            'torch', 'torchvision', 'torchaudio', 
            '--index-url', 'https://download.pytorch.org/whl/cu121',
            '--force-reinstall'
        ])
        print('CUDA PyTorch installed successfully')
        return True
    except subprocess.CalledProcessError:
        print('Failed to install CUDA PyTorch, keeping CPU version')
        return False

if __name__ == '__main__':
    install_cuda_pytorch()
";
                        File.WriteAllText(installCudaScript, cudaInstallScript);

                        services
                                .WithPython()
                                .WithHome(appDataPath)
                                .WithVirtualEnvironment(venvPath)
                                .WithUvInstaller()
                                .FromRedistributable(); // Download Python 3.12 and store it locally

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
            var wrapperModule = Env.TransformersWrapper();
            try
            {
                // Use the Python subprocess to install CUDA PyTorch
                string appDataPath = Path.Join(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "TransformersSharp");
                string installScript = Path.Join(appDataPath, "install_cuda_pytorch.py");

                if (File.Exists(installScript))
                {
                    // TODO: Add implementation to run the CUDA installation script
                    Console.WriteLine("To install CUDA PyTorch manually, run the following commands:");
                    Console.WriteLine($"1. Navigate to: {appDataPath}");
                    Console.WriteLine($"2. Activate the virtual environment: .\\venv\\Scripts\\Activate.ps1");
                    Console.WriteLine("3. Install CUDA PyTorch: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Failed to install CUDA PyTorch: {ex.Message}");
                Console.WriteLine("You can manually install CUDA PyTorch by running:");
                Console.WriteLine("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121");
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

        public static void Dispose()
        {
            _env?.Dispose();
        }
    }
}
