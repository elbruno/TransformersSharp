"""
Device Manager Extensions for CSnakes Integration

This file defines extension methods that would allow direct access to device_manager.py
functions through CSnakes when the framework supports direct module access.
"""

# When CSnakes supports direct module access, this would enable:
# var deviceManager = Env.DeviceManager();
# bool cudaAvailable = deviceManager.IsCudaAvailable();

# For now, all functionality is available through the main TransformersWrapper

# Example future C# extension methods:
"""
namespace TransformersSharp.Extensions
{
    public static class DeviceManagerExtensions
    {
        public static IDeviceManager DeviceManager(this IPythonEnvironment env)
        {
            return env.GetModule<IDeviceManager>("device_manager");
        }
    }

    public interface IDeviceManager
    {
        bool IsCudaAvailable();
        Dictionary<string, object> GetDeviceInfo();
        string ValidateAndGetDevice(string? requestedDevice = null, bool silent = false);
    }
}
"""