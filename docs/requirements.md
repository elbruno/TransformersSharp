# TransformersSharp Requirements

## .NET Version Requirement

**TransformersSharp requires .NET 9.0 SDK or later.**

### Why .NET 9.0?

TransformersSharp uses several modern .NET features and libraries that require .NET 9.0:

- **CSnakes.Runtime 1.0.34+**: Python interop library optimized for .NET 9.0
- **System.Numerics.Tensors**: Advanced tensor operations
- **Lock type**: Modern thread synchronization (replaces traditional lock statements)
- **Collection expressions**: Modern C# syntax features
- **Enhanced nullable reference types**: Improved null safety

### Installation

1. **Download .NET 9.0 SDK**:
   - Visit: https://dotnet.microsoft.com/download/dotnet/9.0
   - Download the SDK (not just the runtime)

2. **Verify installation**:
   ```bash
   dotnet --version
   ```
   Should show 9.0.x or later

3. **Check SDK availability**:
   ```bash
   dotnet --list-sdks
   ```
   Should include a 9.0.x version

## Common .NET Version Issues

### Build Error: NETSDK1045
```
error NETSDK1045: The current .NET SDK does not support targeting .NET 9.0. 
Either target .NET 8.0 or lower, or use a version of the .NET SDK that supports .NET 9.0.
```

**Solution**: Install .NET 9.0 SDK from the link above.

### CSnakes Analyzer Warnings
```
warning CS9057: The analyzer assembly references version '4.9.0.0' of the compiler, 
which is newer than the currently running version '4.8.0.0'.
```

**Solution**: This warning appears when using .NET 8.0 SDK with .NET 9.0 projects. Install .NET 9.0 SDK to resolve.

## Python Requirements

In addition to .NET 9.0, TransformersSharp requires:

- **Python 3.8+**: For running machine learning models
- **Virtual Environment Support**: Automatically managed by TransformersSharp
- **Internet Access**: For downloading models and packages

## Console Applications

All console applications (located in the `DemoConsole/` folder) require .NET 9.0:

- **DemoConsole/ConsoleApp1**: Text generation pipeline example
- **DemoConsole/ConsoleApp2**: Microsoft.Extensions.AI chat client example  
- **DemoConsole/ConsoleApp3**: Text-to-image generation example
- **DemoConsole/ConsoleApp4**: Comprehensive performance testing and diagnostics
- **DemoConsole/TransformersSharp.EnvironmentTest**: Environment validation and CUDA installation testing

## Troubleshooting

### If you cannot install .NET 9.0

If you're restricted to .NET 8.0, you would need to:

1. Downgrade CSnakes.Runtime to a .NET 8.0 compatible version (if available)
2. Replace .NET 9.0 specific features:
   - Replace `Lock` with `object` + traditional lock statements
   - Replace collection expressions with explicit constructors
   - Handle nullable reference type differences

**Note**: This is not officially supported and may cause compatibility issues.

### Getting Help

- Check DemoConsole/ConsoleApp4 for comprehensive diagnostics
- Run DemoConsole/TransformersSharp.EnvironmentTest for environment validation
- Review error messages for specific guidance
- See the troubleshooting sections in each console application