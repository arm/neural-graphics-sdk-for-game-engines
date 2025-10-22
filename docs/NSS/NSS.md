# Neural Super Sampling

Mobile devices increasingly require more complex graphic experience, which increases pressure on GPU performance, power budgets, and memory bandwidth. Traditional rendering techniques are effective, but they are reaching their limits in delivering high-resolution, high-frame-rate graphics within the tight constraints of mobile hardware.

Neural Super Sampling addresses these limitations as a temporal frame upsampling technique powered by neural technologies, improving visual fidelity by reconstructing high-resolution frames from lower-resolution inputs. It is specifically optimized for mobile applications to achieve:

• High image quality

• Computational efficiency

• Minimal bandwidth usage

## Integration guidelines

### Build

Build from source code: copy SDK folder into your project, add sdk as a sub project through CmakeLists.txt:

```cmake
add_subdirectory(path/to/sdk).
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
dispatchNss.depthTm1 = getDepthTm1
dispatchNss.motionVectors = getMotion;
dispatchNss.outputTm1 = getOutputTm1;
dispatchNss.output = getOutput;
dispatchNss.debugViews = getDebugViwe;
 
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
| depthTm1 | render resolution | R32 | float | Last frame's depth texture. "Tm1" means "time minus 1". |
| motionVectors | render resolution | R16G16 | float | The 2D motion vectors for the current frame are provided by the application.<br>UV(T-1) = UV(T) + Motion.<br>Normalized vectors, range in [-1, 1]. |
| output | upscaled resolution | R11G11B10 | float | Upscaled output. |
| outputTm1 | upscaled resolution | R11G11B10 | float | Last frame's upscaled output. |
| debugView | upscaled resolution | R11G11B10 | float | Render internal resources for easy debug. Debug view will split output into 12 pieces:<br>Row1: Warpped history, jittered color, feed back tensor, disocclusion mask and luma derivative.<br>Row2: Internal tensors, K0~K3.<br>Row3: Motion vector, KPN weight(calculated from internal tensors), temporal parameters, upscaledOutput |

##### Padding and truncate

Padding input: Clamp_net requires the width/height("render resolution") of the input in multiple of 8, need to pad for input if necessary. For example, if your input resolution is 960x540, need to pad it to 960x544.

It's recommended to use "Mirror padding" in bottom-right corner:

![invert](./NSS_MirrorPadding.svg "A diagram showing mirror padding.")

Sample code:
```glsl
#version 450
 
layout(push_constant) uniform Constants {
    uvec2 inputSize;
    uvec2 paddedInputSize;
} g_pc;
 
layout(location = 0) in vec2 uv;
 
layout(location = 0) out vec4 outColor;
layout(location = 1) out float outDepth;
layout(location = 2) out vec2 outFlow;
 
layout(set = 0, binding = 0) uniform sampler2D inputColorTex;
layout(set = 0, binding = 1) uniform sampler2D inputDepthTex;
layout(set = 0, binding = 2) uniform sampler2D inputFlowTex;
 
void main() {
    Vec2 inputUV = (uv * Vec2(m_paddedInputSize) / Vec2(g_pc.m_inputSize));
 
    F32 MirrorBoundary = 1.0 - 0.5 / g_pc.m_inputSize;
    if(inputUV.x >= 1.0f)
    {
        inputUV.x = 2 * MirrorBoundary - inputUV.x;
    }
    if(inputUV.y >= 1.0f)
    {
        inputUV.y = 2 * MirrorBoundary - inputUV.y;
    }
    uv = inputUV;
 
    outColor = texture(inputColorTex, coord);
    outDepth = texture(inputDepthTex, coord).r;
    outFlow = motionVectorFactor * texture(inputFlowTex, coord).rg;
}
```

Truncate output: Output's bottom-right corner need to be truncated. For example, if your input dimension is 960x540, scale is x2, expect a 1920x1080 output. Then padded input dimension will be 960x544, padded output will be 1920*1088, output need to be truncated to 1920x1080.

## Limitations

Only support quantized data graph.