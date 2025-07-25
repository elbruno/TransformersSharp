# Python Module Architecture

TransformersSharp's Python wrapper has been reorganized into a modular architecture for better maintainability and separation of concerns.

## Module Structure

### Core Modules

#### `transformers_wrapper.py` - Main Pipeline Module
The primary module that handles:
- Pipeline creation and execution
- Model support and validation  
- Package compatibility checking
- Authentication and tokenization
- Text-to-image generation
- Speech recognition and text-to-audio
- Image classification and object detection

#### `device_manager.py` - Device Management
Handles all CUDA/CPU device detection and management:
- `is_cuda_available()` - Check if CUDA is available
- `get_device_info()` - Get information about available devices
- `validate_and_get_device()` - Validate and return best available device

#### `system_info.py` - System Information
Provides comprehensive system information gathering:
- `get_detailed_system_info()` - Get comprehensive system information
- `_get_cpu_info()` - Get CPU information
- `_get_memory_info()` - Get memory information
- `_get_pytorch_info()` - Get PyTorch information
- `_get_gpu_info()` - Get GPU information

#### `image_utils.py` - Image Processing
Contains image processing and conversion utilities:
- `convert_image_to_bytes()` - Convert various image formats to PNG bytes

### Benefits of Modular Architecture

1. **Separation of Concerns**: Each module has a specific responsibility
2. **Better Maintainability**: Easier to locate and modify specific functionality
3. **Reduced Code Duplication**: Shared utilities are centralized
4. **Improved Testing**: Each module can be tested independently
5. **Clear Dependencies**: Explicit imports make relationships clear

## Usage Patterns

### Direct Module Usage (Future Enhancement)
When CSnakes supports direct module imports, you could potentially use:

```csharp
// Direct device management
var deviceManager = Env.DeviceManager();
bool cudaAvailable = deviceManager.IsCudaAvailable();

// Direct system information
var systemInfo = Env.SystemInfo();
var details = systemInfo.GetDetailedSystemInfo();

// Direct image utilities
var imageUtils = Env.ImageUtils();
byte[] imageBytes = imageUtils.ConvertImageToBytes(image);
```

### Current Usage (Through Main Wrapper)
Currently, all functionality is accessed through the main wrapper:

```csharp
// All functions are re-exported through TransformersWrapper
var wrapper = TransformerEnvironment.TransformersWrapper;
bool cudaAvailable = wrapper.IsCudaAvailable();
var systemInfo = wrapper.GetDetailedSystemInfo();
```

## Migration Impact

This refactoring maintains **100% backward compatibility**:
- All existing C# code continues to work unchanged
- All functions are re-exported through the main wrapper
- No changes to public APIs or method signatures
- No performance impact - functions are simply imported

## File Organization

```
TransformersSharp/python/
├── transformers_wrapper.py     # Main pipeline and orchestration
├── device_manager.py          # CUDA/CPU device management
├── system_info.py             # System information gathering
├── image_utils.py             # Image processing utilities
└── sentence_transformers_wrapper.py  # Sentence transformers (unchanged)
```

## Future Enhancements

This modular architecture enables:
1. **Independent Module Updates**: Update device management without affecting image processing
2. **Selective Loading**: Load only required modules for specific use cases
3. **Plugin Architecture**: Add new modules without modifying existing ones
4. **Better Error Isolation**: Issues in one module don't affect others
5. **Performance Optimization**: Optimize specific modules independently