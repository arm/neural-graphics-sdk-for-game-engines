# Neural Super Sampling

Mobile devices increasingly require more complex graphic experience, which increases pressure on GPU performance, power budgets, and memory bandwidth. Traditional rendering techniques are effective, but they are reaching their limits in delivering high-resolution, high-frame-rate graphics within the tight constraints of mobile hardware.

The Neural Super Sampling addresses these limitations. The Neural Super Sampling is a temporal frame upsampling technique that improves visual fidelity by reconstructing high-resolution frames from lower-resolution inputs. The Neural Super Sampling is specifically optimized for mobile applications to achieve:

• High image quality

• Computational efficiency

• Minimal bandwidth usage

## Platform Support

Windows 11 x64.

Linux x64.

Android. (Validated on Android 16)

## Integration guidelines

### Build

Build from source code: copy SDK folder into your project, add sdk as a sub project through CmakeLists.txt:

```cmake
add_subdirectory(path/to/sdk)
```

Or if you use prebuilt libs, need to link them to your project.

### Header files

These header files need to be included:

```cpp
#define FFX_CPU
#include <path/to/sdk/ffx-api/include/ffx_api/ffx_api.hpp>
#include <path/to/sdk/ffx-api/include/ffx_api/ffx_nss.hpp>
#include <path/to/sdk/ffx-api/include/ffx_api/ffx_api_types.h>
#include <path/to/sdk/ffx-api/include/ffx_api/vk/ffx_api_vk.hpp>
```

Need to add "#define FFX_CPU" before including these for the SDK to resolve some types in some common headers.

### API interface

NSS is built based on FFX SDK 1.1.3 APIs, declared in ffx_api.h:

• ffxCreateContext

• ffxDestroyContext

• ffxDispatch

• ffxQuery

• ffxConfigure

#### ffxCreateContext

Creates a FFX object context. Depending on the desc structures provided to this function, the context will be created with the desired version and attributes. Pointers passed in desc must remain live until ffxDestroyContext is called on the context. MemCb may be null; the system allocator (malloc/free) will be used in this case.

Sample code:

```cpp
ffx::CreateBackendVKDesc backendDesc{};
backendDesc.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK;
backendDesc.vkDevice = getVulkanDevice();
backendDesc.vkPhysicalDevice = getVulkanPhysicalDevice();
backendDesc.vkInstance = getVulkanInstance();
backendDesc.vkDeviceProcAddr = vkGetDeviceProcAddr; //vulkan function pointer
backendDesc.vkGetInstanceProcAddr = vkGetInstanceProcAddr; //vulkan function pointer

ffx::CreateContextDescNss createContextNss{};
createContextNss.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_NSS;
createContextNss.maxRenderSize = {initSettings.m_srcRes.x(), initSettings.m_srcRes.y()};
createContextNss.maxUpscaleSize = {initSettings.m_targetRes.x(), initSettings.m_targetRes.y()};
createContextNss.flags = 0;
createContextNss.flags |= (FFX_API_NSS_CONTEXT_FLAG_QUANTIZED |
                           FFX_API_NSS_CONTEXT_FLAG_HIGH_DYNAMIC_RANGE |
                           FFX_API_NSS_CONTEXT_FLAG_DEPTH_INFINITE |
                           FFX_API_NSS_CONTEXT_FLAG_READ_TENSORS_AS_IMAGES |
                           FFX_API_NSS_CONTEXT_FLAG_ENABLE_DEBUG_CHECKING);
createContextNss.fpMessage = &nssMsgCallback;
ffx::ReturnCode retCode = ffx::CreateContext(m_nssContext, nullptr, createContextNss, backendDesc);
```

More details about [`ffxApiCreateContextDescNss`](../../ffx-api/include/ffx_api/ffx_nss.h#L74):

header.type: Must be [`FFX_API_CREATE_CONTEXT_DESC_TYPE_NSS`](../../ffx-api/include/ffx_api/ffx_nss.h#L72).

maxRenderSize/maxUpscaleSize: The SDK supports flexible upscaling ratios, but 2x is recommended for best quality and performance due to dedicated optimizations. Support for other ratios will be improved in future updates.

flags: These flags are defined in "enum [`FfxApiCreateContextNssFlags`](../../ffx-api/include/ffx_api/ffx_nss.h#L51)":

qualityMode: Select shader quality mode. This bit is currently unused. It will be required in the future when more optimizations are introduced in the shader.

| Flag bits | Description |
|-----------|-------------|
| FFX_API_NSS_CONTEXT_FLAG_QUANTIZED | Use a quantized data graph. Resources will be quantized to 8 bits. Currently, SDK only support quantized data graph. So this bit must be set. |
| FFX_API_NSS_CONTEXT_FLAG_HIGH_DYNAMIC_RANGE | Input color data provided is using a high-dynamic range. Currently this flag must be set. |
| FFX_API_NSS_CONTEXT_FLAG_DEPTH_INVERTED | Input depth buffer data provided is inverted [1..0]. |
| FFX_API_NSS_CONTEXT_FLAG_DEPTH_INFINITE | Input depth buffer data provided is using an infinite far plane. |
| FFX_API_NSS_CONTEXT_FLAG_RESAMPLE_BICUBIC | Sample using Bicubic filtering. |
| FFX_API_NSS_CONTEXT_FLAG_READ_TENSORS_AS_IMAGES | Tensor image aliasing is enabled. Will load tensors through "texture" functions. |
| FFX_API_NSS_CONTEXT_FLAG_ALLOW_16BIT | Runtime should allow 16bit resources to be used. |
| FFX_API_NSS_CONTEXT_FLAG_ENABLE_DEBUG_CHECKING | 	Runtime should check some API values and report issues. |

#### ffxDestroyContext

Destroys an FFX object context. MemCb must be compatible with the callbacks passed into ffxCreateContext.

#### ffxConfigure

Configures the provided FFX object context. If context is null, configure operates on any global state. For NSS, there is no specific configuration.

#### ffxQuery

Queries the provided FFX object context.

NSS provide these query types:

| Query type | Data structure | comments |
|------------|----------------|----------|
| [`FFX_API_QUERY_DESC_TYPE_NSS_GETJITTERPHASECOUNT`](../../ffx-api/include/ffx_api/ffx_nss.h#L147) | ffxApiQueryDescNssGetJitterPhaseCount | Get jitter phase count. |
| [`FFX_API_QUERY_DESC_TYPE_NSS_GETJITTEROFFSET`](../../ffx-api/include/ffx_api/ffx_nss.h#L59) | ffxApiQueryDescNssGetJitterOffset | Get jitter offset for specific index. |

If context is null, query operates on any global state. For example, to query a provider ID:

```cpp
ffx::QueryDescGetVersions versionQuery{};
versionQuery.createDescType = FFX_API_CREATE_CONTEXT_DESC_TYPE_NSS;
uint64_t versionCount = 0;
versionQuery.outputCount = &versionCount;
ffxQuery(nullptr, &versionQuery.header); //Null context pointer

std::vector<uint64_t> versionIds;
std::vector<const char*> versionNames;
versionIds.resize(versionCount);
versionNames.resize(versionCount);
versionQuery.versionIds = versionIds.data();
versionQuery.versionNames = versionNames.data();
ffxQuery(nullptr, &versionQuery.header); //Null context pointer
```

#### ffxDispatch

ffxDispatch need to be called at each end of frame, to do the upscale work. Dispatches work on the given FFX object context defined by the dispatch descriptor. Need to fill up ffxApiDispatchDescNss to let upscale work.

Sample code:

```cpp
ffx::DispatchDescNss dispatchNss{};
dispatchNss.header.type = FFX_API_DISPATCH_DESC_TYPE_NSS;
dispatchNss.commandList   = static_cast<CommandBufferImpl&>(cmdb).getHandle();
 
//Get input resources
dispatchNss.color = getColor;
dispatchNss.depth = getDepth;
dispatchNss.motionVectors = getMotion;
dispatchNss.output = getOutput;
 
dispatchNss.jitterOffset.x      = jitterOffset.x();
dispatchNss.jitterOffset.y      = jitterOffset.y();
dispatchNss.renderSize.width    = renderSize.x();
dispatchNss.renderSize.height   = renderSize.y();
dispatchNss.upscaleSize.width   = upscaleSize.x();
dispatchNss.upscaleSize.height  = upscaleSize.y();
dispatchNss.cameraNear             = cameraNear;
dispatchNss.cameraFar              = cameraFar;
dispatchNss.cameraFovAngleVertical = cameraFovAngleVertical;
dispatchNss.exposure            = exposure;
dispatchNss.motionVectorScale.x = -1.0f * renderSize.x(); //NSS expects the motion vectors in left-handed coordinates
dispatchNss.motionVectorScale.y = -1.0f * renderSize.y();
dispatchNss.frameTimeDelta         = frameTimeDelta;
dispatchNss.reset                  = reset;
dispatchNss.flags                  = enableDebugView ? FFX_API_NSS_DISPATCH_FLAG_DRAW_DEBUG_VIEW : 0;

ffx::ReturnCode retCode = ffx::Dispatch(m_nssContext, dispatchNss);
```

When debug view is enabled, the output splits into 12 pieces:
|  | Column 1 | Column 2 | Column 3 | Column 4 |
|----------|------------|--------|-----------|---------|
| Row 1 | Warpped history | jittered color | feed back tensor | disocclusion mask and luma derivative|
| Row 2 | Internal tensors K0 | Internal tensors K1 | Internal tensors K2 | Internal tensors K3 |
| Row 3 | Motion vector | KPN weight(calculated from internal tensors) | temporal parameters | upscaledOutput |


##### ffx::DispatchDescNss

More details about struct [`ffxApiDispatchDescNss`](../../ffx-api/include/ffx_api/ffx_nss.h#L88):

jitterOffset: jitter offset for each frame. SDK expect a "uv" offset, so They must be in range [-0.5, 0.5].

renderSize/upscaleSize: Same as [`ffxApiCreateContextDescNss`](../../ffx-api/include/ffx_api/ffx_nss.h#L74).

cameraNear: The distance to the near plane of the camera.

cameraFar: The distance to the far plane of the camera.

cameraFovAngleVertical: The camera angle field of view in the vertical direction (expressed in radians).

exposure: The exposure value. Cannot be zero.

motionVectorScale: It should be set as renderSize. NSS need motion vector recorded in left-handed coordinate system. If the motions provided by your application is not, need to multiply it with -1.0.

frameTimeDelta: The time elapsed since the last frame (expressed in milliseconds).

reset: Need reset history, due to changes in render size, camera transitions, etc.

flags: Only contains [`FFX_API_NSS_DISPATCH_FLAG_DRAW_DEBUG_VIEW`](../../ffx-api/include/ffx_api/ffx_nss.h#L65), when set this flag, debug view will be rendered rather than normal upscaled output.

##### Resources

| Resource | Resolution | Format | Data type | Comment |
|----------|------------|--------|-----------|---------|
| color | render resolution | R11G11B10 | float | The render resolution color buffer for the current frame provided by the application. Must be HDR format. |
| depth | render resolution | R32 | float | The render resolution depth buffer for the current frame provided by the application. |
| motionVectors | render resolution | R16G16 | float | The 2D motion vectors for the current frame are provided by the application.<br>UV(T-1) = UV(T) + Motion.<br>Normalized vectors, range in [-1, 1]. |
| output | upscaled resolution | R11G11B10 | float | Upscaled output. |

## NSS Sample

You can run the NSS sample to see how to use NSS for high‑quality temporal upscaling. The NSS sample project is derived from the Khronos Vulkan Samples project, commit b9961792604af2ede4c9d0868947de2a8eccd549.

### Overview

The program flow of the NSS sample is:

1. The scene pipeline produces low-resolution **color**, **velocity**, and **depth** images.
2. These images are passed to **NSS dispatch** to upscale into a high-resolution output.
3. The result is passed to the post-processing pipeline to render the final image to the screen.

You can choose a display mode:

- High-resolution output
- Low-resolution color/velocity/depth buffers
- NSS debug view

### Requirements

#### Before you begin

The dependencies for the NSS sample are:

- CMake 3.16+
- Python 3
- On Windows:
    - Visual Studio 2019 or higher
- On Android:
    - Android Studio 2024.2.1+ (recommended)
    - Android NDK r23+
    - Android SDK
    - JDK 21

**Note**: Sample code is not supported on Linux.

#### About the sample

The NSS sample is based on the SDK and must meet all SDK environment prerequisites. See: [SDK README](../../README.md)

#### Procedure

1. Before you can build the NSS sample, you must build the static SDK library. To build the static SDK library, run `build.py`. For example:

```
python build.py --clean -t Debug -b vk_windows_x64 --cmake-args="-DFFX_BUILD_AS_DLL=OFF"
python build.py --clean -t Release -b vk_windows_x64 --cmake-args="-DFFX_BUILD_AS_DLL=OFF"

$env:NDK_ROOT="path\to\ndk"
python build.py --clean -t Release -b vk_android_arm --cmake-args="-DFFX_BUILD_AS_DLL=OFF"
python build.py --clean -t Debug -b vk_android_arm --cmake-args="-DFFX_BUILD_AS_DLL=OFF"
```

2. Initialize the submodules:

```
git submodule status
git submodule update --init --recursive
```

### Build the sample

**Note**: Always build the sample from the `samples` directory.

#### Windows

To build the sample on a Windows platform:

```
cd samples

# Generate Visual Studio project
cmake -G "Visual Studio 17 2022" -A x64 -B build/windows

# Build
cmake --build build/windows --config Debug --target vulkan_samples --parallel 16
```

#### Android

To build the sample on an Android platform:

```
cd samples

# Generate Android Gradle project
./scripts/generate.py android

# Build with Gradle
cd build/android_gradle
chmod +x gradlew
./gradlew clean

# You must run with JDK 21
# When building and the android device is properly connected, it pushes assets and shaders to the device automatically.
./gradlew assembleRelease

# Install on device
# It does not push assets and shaders to device.
adb install app/build/outputs/apk/release/vulkan_samples-release.apk
```

Alternatively, you can use Android Studio to open `build/android_gradle` and build directly in Android Studio.

### Run the sample

#### Windows

To run the sample on Windows, use:

```
cd samples
.\build\windows\app\bin\Debug\AMD64\vulkan_samples.exe sample nss
```

**Note**: On Windows, always run the NSS sample from the `samples` directory. Otherwise, the NSS sample cannot find the `assets` and `shaders`.

#### Android

To run the sample on Android, use:

```
# If you did not push assets and shaders, you can push them manually.
adb push /path/to/samples/assets/scenes/sponza /sdcard/Android/data/com.khronos.vulkan_samples/files/assets/scenes/sponza
adb push /path/to/samples/assets/fonts /sdcard/Android/data/com.khronos.vulkan_samples/files/assets/fonts
adb push /path/to/samples/shaders /sdcard/Android/data/com.khronos.vulkan_samples/files/shaders

adb shell "chmod -R 777 /sdcard/Android/data/com.khronos.vulkan_samples/files/assets"
adb shell "chmod -R 777 /sdcard/Android/data/com.khronos.vulkan_samples/files/shaders"

adb shell am start-activity \
	-n com.khronos.vulkan_samples/com.khronos.vulkan_samples.SampleLauncherActivity \
	-e sample nss
```

#### Command line options

The following tables show the sample command line options and NSS sample arguments.

| Option | Description | Usage |
|---|---|---|
| `--width <pixels>` | Window width | `--width 1920` |
| `--height <pixels>` | Window height | `--height 1080` |
| `--hideui` | Hide UI | `--hideui` |
| `--headless_surface` | Use `VK_EXT_headless_surface` | `--headless_surface` |
| `--app-arg` | Forward arbitrary KEY=VALUE arguments to the selected sample | `--app-arg NSS_ENABLE=0` |

| NSS argument | Description | Usage |
|---|---|---|
| `NSS_ENABLE` | enable or disable NSS | `--app-arg NSS_ENABLE=0` |
| `NSS_SCALE_FACTOR` | NSS Upscale Factor [1.0, 3.0] | `--app-arg NSS_SCALE_FACTOR=2.0` |
| `NSS_FLAGS` | NSS create context override flags | `--app-arg NSS_FLAGS=0x05` |

#### Windows example

An example of how to use command line options on Windows is:

```
# Run with 2× NSS upscaling (944×496 → 1888×992):
.\build\windows\app\bin\Debug\AMD64\vulkan_samples.exe sample nss --width 1888 --height 992
# Disable NSS
.\build\windows\app\bin\Debug\AMD64\vulkan_samples.exe sample nss --app-arg NSS_ENABLE=0
```

**Note**: NSS is optimized for 2× upscaling. Other scale factors are supported but may have reduced quality.

#### Android example

An example of how to use command line options on Android is:

```
adb shell am start -n com.khronos.vulkan_samples/com.khronos.vulkan_samples.SampleLauncherActivity --es cmd "sample\ nss\ --hideui"
adb shell am start -n com.khronos.vulkan_samples/com.khronos.vulkan_samples.SampleLauncherActivity --es cmd "sample\ nss\ --app-arg\ NSS_ENABLE=1,NSS_SCALE_FACTOR=2.0,NSS_FLAGS=0x85"
```

### User interface elements

The NSS sample provides display modes to help debug and understand the NSS upscaling pipeline.

The display modes are:

- **Final Output**: the final upscaled image produced by NSS.
- **Velocity**: color-coded motion vectors. Red/Green channels represent X/Y direction; brightness indicates magnitude.
- **Depth**: depth buffer visualization (white = near, black = far).
 - **NSS Debug View**: NSS internal visualization. See Debug View output in section [ffxDispatch](#ffxdispatch)
- **Low-Res Color**: raw low-resolution rendering without NSS upscaling.

#### Temporal Anti-Aliasing settings

The following table shows the Temporal Anti-Aliasing (TAA) settings.

| Setting | Value | Description |
|---|---|---|
| Jitter Scale | 0.0 - 1.0 | Sub-pixel jitter multiplier (0.0 = off, 1.0 = full) |
| Current Jitter | Read-only | Current frame jitter offset in pixels |
| ScaleFactor | 1.0 - 3.0 | NSS upscale factor |
| Flags | Depends on `ffx_nss.h` | NSS create context override flags; if `0`, do not override |
| Render | Read-only | Render resolution (low-res), computed from ScaleFactor |
| Display | Read-only | Display resolution (high-res) |
| LOD Bias | Read-only | Mipmap bias for texture sampling, computed from ScaleFactor |

### Additional Resources

- Vulkan Samples: https://github.com/KhronosGroup/Vulkan-Samples

## Limitations

Only quantized data graphs are supported.