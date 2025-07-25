"""
Image Utils Extensions for CSnakes Integration

This file defines extension methods that would allow direct access to image_utils.py
functions through CSnakes when the framework supports direct module access.
"""

# When CSnakes supports direct module access, this would enable:
# var imageUtils = Env.ImageUtils();
# byte[] imageBytes = imageUtils.ConvertImageToBytes(image);

# For now, all functionality is available through the main TransformersWrapper

# Example future C# extension methods:
"""
namespace TransformersSharp.Extensions
{
    public static class ImageUtilsExtensions
    {
        public static IImageUtils ImageUtils(this IPythonEnvironment env)
        {
            return env.GetModule<IImageUtils>("image_utils");
        }
    }

    public interface IImageUtils
    {
        byte[] ConvertImageToBytes(object image);
    }
}
"""