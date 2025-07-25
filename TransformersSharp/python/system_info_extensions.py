"""
System Info Extensions for CSnakes Integration

This file defines extension methods that would allow direct access to system_info.py
functions through CSnakes when the framework supports direct module access.
"""

# When CSnakes supports direct module access, this would enable:
# var systemInfo = Env.SystemInfo();
# var details = systemInfo.GetDetailedSystemInfo();

# For now, all functionality is available through the main TransformersWrapper

# Example future C# extension methods:
"""
namespace TransformersSharp.Extensions
{
    public static class SystemInfoExtensions
    {
        public static ISystemInfo SystemInfo(this IPythonEnvironment env)
        {
            return env.GetModule<ISystemInfo>("system_info");
        }
    }

    public interface ISystemInfo
    {
        Dictionary<string, object> GetDetailedSystemInfo();
        Dictionary<string, object> GetCpuInfo();
        Dictionary<string, object> GetMemoryInfo();
        Dictionary<string, object> GetPytorchInfo();
        Dictionary<string, object> GetGpuInfo();
    }
}
"""