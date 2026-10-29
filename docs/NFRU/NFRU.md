# Neural Frame Rate Upscaling

## Dataset capture

NFRU capture configuration, lifetime, synchronization, dataset layout, and
replay constraints are documented in the [Dataset Capture guide](../Capture/Capture.md).

Neural Frame Rate Upscaling (NFRU) is a deep‑learning–based frame interpolation technique that increases delivered frame rate between consecutive rendered frames. It combines engine-provided motion vectors, optical-flow–derived correspondence fields, and a neural network to produce temporally consistent and visually smooth interpolated frames. NFRU operates with a known, fixed additional frame of latency, making pacing predictable in real-time graphics on mobile.

> NFRU can be used on its own or alongside an upscaler (such as NSS). When paired with an upscaler, the application renders at a reduced `renderSize`, upscales to `displaySize`, and NFRU generates one additional interpolated frame per rendered frame. When used standalone, `renderSize` and `displaySize` are simply set equal. In both cases NFRU produces one additional interpolated frame per rendered frame.

## Table of Contents

- [Neural Frame Rate Upscaling](#neural-frame-rate-upscaling)
  - [Table of Contents](#table-of-contents)
  - [Platform Support](#platform-support)
  - [Integration Guidelines](#integration-guidelines)
    - [Build](#build)
    - [Placement in the frame](#placement-in-the-frame)
    - [Header Files](#header-files)
    - [API Interface](#api-interface)
      - [Integration Overview](#integration-overview)
      - [ffxCreateContext](#ffxcreatecontext)
      - [ffxDispatch](#ffxdispatch)
        - [Prepare Dispatch](#prepare-dispatch)
        - [Dispatch](#dispatch)
        - [Resources](#resources)
      - [ffxQuery](#ffxquery)
      - [ffxConfigure](#ffxconfigure)
    - [Runtime logging](#runtime-logging)
    - [Lifetime and synchronization](#lifetime-and-synchronization)
    - [Troubleshooting NFRU Integration](#troubleshooting-nfru-integration)
      - [Depth convention and disocclusion](#depth-convention-and-disocclusion)
      - [Motion-vector artifacts](#motion-vector-artifacts)
      - [View-projection matrix and dynamic mask](#view-projection-matrix-and-dynamic-mask)
      - [Candidate alignment](#candidate-alignment)
  - [Internal Resolution Capping](#internal-resolution-capping)
  - [Limitations](#limitations)
  - [Presentation Pacing](#presentation-pacing)
  - [Integration Checklist](#integration-checklist)

---

## Platform Support

| OS | Architecture |
|----|--------------|
| Windows 11 | x64 |
| Linux | x64 |
| Android | AArch64 |

## Integration Guidelines

### Build

There are two ways to integrate the SDK:

- **Build from source**: copy the SDK folder into your project and add it as a sub-project from your `CMakeLists.txt`:

  ```cmake
  add_subdirectory(path/to/sdk)
  ```

- **Use prebuilt libraries**: link the prebuilt libraries into your project directly.

### Placement in the frame

NFRU should normally consume the final scene colour before UI composition. It is robust to many colour-only post-processing operations, but effects whose apparent motion is not represented by the supplied depth and motion vectors can produce interpolation artifacts.

Use this ordering as a starting point:

```text
view geometry -> optional upscaler -> scene post-processing -> NFRU -> UI composition -> present
```

Compose UI and effects that cannot provide temporally consistent colour, depth, and motion after NFRU. For isolation steps and a placement illustration, see [Post-Processing Moves Incorrectly](../integration_troubleshooting.md#post-processing-moves-incorrectly).

### Header Files

These header files need to be included:

```cpp
#define FFX_CPU
#include <path/to/sdk/ffx-api/include/ffx_api/ffx_api.hpp>
#include <path/to/sdk/ffx-api/include/ffx_api/ffx_framegeneration.hpp>
#include <path/to/sdk/ffx-api/include/ffx_api/ffx_api_types.h>
#include <path/to/sdk/ffx-api/include/ffx_api/vk/ffx_api_vk.hpp>
```

`#define FFX_CPU` must appear before the header includes so that the SDK selects the correct macros at compile time in some common headers.

### API Interface

NFRU uses the API defined in [ffx_api.h](../../ffx-api/include/ffx_api/ffx_api.h).

• ffxCreateContext

• ffxDestroyContext

• ffxDispatch

• ffxConfigure

• ffxQuery

For descriptions of the core API functions and return codes, see [User Guide - Core API](../user_guide.md#core-api). Below is a detailed walkthrough of each function as used by NFRU.

#### Integration Overview

NFRU does not require a preceding upscaler: `renderSize` may equal `displaySize` (standalone use) or be smaller than it (when an upscaler such as NSS runs first). The optical flow and frame interpolation contexts are managed internally, so a typical integration only deals with one frame generation context and the following lifecycle:

```text
ffxCreateContext -> ffxConfigure -> per frame: [ffxDispatch prepare -> ffxDispatch] -> wait for GPU completion -> ffxDestroyContext
```

A minimal per-frame loop. The `createFg` / `backendDesc` descriptors are filled as shown in [ffxCreateContext](#ffxcreatecontext); the prepare and dispatch descriptors as shown in [Prepare dispatch](#prepare-dispatch) and [Dispatch](#dispatch):

```cpp
// Once, at startup
ffx::CreateContext(m_FrameGenContext, nullptr, createFg, backendDesc);

ffx::ConfigureDescFrameGeneration config{};
config.frameGenerationEnabled = true;
config.flags = FFX_API_FG_DISPATCH_FLAG_NO_SWAPCHAIN_CONTEXT_NOTIFY;
ffx::Configure(m_FrameGenContext, config);

// Every frame. frameID must increment by exactly 1 (see frameID below).
for (uint64_t frameID = 0; running; ++frameID)
{
    renderScene();   // produces render-resolution depth, motion vectors and color
    upscale();       // optional preceding upscale (e.g. NSS); skip when rendering directly at display resolution

    // 1) prepare: build the internal resources needed for interpolation
    ffx::DispatchDescFrameGenerationPrepare prepare{};
    prepare.frameID = frameID;
    // ... fill prepare (commandList, depth, motionVectors, camera, viewProjection, ...)
    // Supply depthTm1/colorTm1 unless the selected path manages them internally.
    ffx::Dispatch(m_FrameGenContext, prepare);

    // 2) dispatch: generate the interpolated frame
    ffx::DispatchDescFrameGeneration dispatch{};
    dispatch.frameID = frameID;   // same frameID as the prepare of this frame
    dispatch.numGeneratedFrames = 1;
    // ... fill dispatch (commandList, presentColor, outputs, reset, ...)
    ffx::Dispatch(m_FrameGenContext, dispatch);

    submitAndWaitForOutput();     // submit recorded work and synchronize before presentation
    present(interpolatedFrame);   // generated frame; reset/first dispatch copies realFrame
    present(realFrame);           // rendered frame
}

// Once, at shutdown, after all GPU work using this context has completed
ffx::DestroyContext(m_FrameGenContext);
```

NFRU only generates the interpolated image. How the rendered and generated frames are submitted to the swap chain and paced is the responsibility of the presentation layer. NFRU operates with a known, fixed additional frame of latency to make pacing predictable.

#### ffxCreateContext

Create frame generation context. Two internal contexts will be created internally: optical flow context and frame interpolation context. These contexts will be managed by SDK internally. Users needn't to configure optical flow or frame interpolation related parameters explicitly.

Sample code:

```cpp
ffx::CreateBackendVKDesc backendDesc{};
backendDesc.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK;
backendDesc.vkDevice = getVulkanDevice();
backendDesc.vkPhysicalDevice = getVulkanPhysicalDevice();
backendDesc.vkInstance = getVulkanInstance();
backendDesc.vkDeviceProcAddr = vkGetDeviceProcAddr; //vulkan function pointer
backendDesc.vkGetInstanceProcAddr = vkGetInstanceProcAddr; //vulkan function pointer

ffx::CreateContextDescFrameGeneration createFg{};
createFg.displaySize = {m_settings.m_targetRes.x(), m_settings.m_targetRes.y()};
createFg.renderSize = {m_settings.m_srcRes.x(), m_settings.m_srcRes.y()};
createFg.backBufferFormat = getSwapChainFormat();
memcpy(createFg.initialViewProjection, initialViewProjectionColumnMajor, sizeof(createFg.initialViewProjection));
createFg.fpMessage = &nfruMsgCallback;
createFg.flags = 0;
createFg.flags |= (FFX_API_FG_CONTEXT_FLAG_ENABLE_HIGH_DYNAMIC_RANGE |
                   FFX_API_FG_CONTEXT_FLAG_ENABLE_DEPTH_INVERTED);
ffx::ReturnCode retCode = ffx::CreateContext(m_FrameGenContext, nullptr, createFg, backendDesc);
```

More details about [ffxApiCreateContextDescFrameGeneration](../../ffx-api/include/ffx_api/ffx_framegeneration.h#L78):

header.type: Must be [`FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION`](../../ffx-api/include/ffx_api/ffx_framegeneration.h#L75).

renderSize: The resolution at which the application renders the scene, before upscaling. For example `{1280, 720}`.

mvDepthSize: Optional shared resolution of the supplied depth and motion-vector images, independent of rendered color. Leave it zero-initialized (`{0, 0}`) to use `renderSize`; this value-style field does not take a null pointer. Both dimensions must be at least 4 when supplied and must have the same aspect ratio as `renderSize`. Partially zero dimensions are invalid. The size is fixed for the lifetime of the context.

For 2160p display, 1080p rendered color, and inputs already reduced to 540p:

```cpp
createFg.displaySize = {3840, 2160};
createFg.renderSize = {1920, 1080};
createFg.mvDepthSize = {960, 540};
```

NFRU still downsamples color for optical flow and preprocessing, but does not run the all-input downsample pass on these already reduced depth/motion inputs.

displaySize: The resolution at which both rendered and generated frames are presented. When NFRU is used standalone it equals `renderSize`; when a preceding upscaler is used it is larger than `renderSize` (for example, a `{1280, 720}` renderSize with a `{2560, 1440}` displaySize corresponds to a 2x upscale).

backBufferFormat: The surface format of the backbuffer, one of the values from `FfxApiSurfaceFormat` (for example `FFX_API_SURFACE_FORMAT_B8G8R8A8_UNORM`). It should match the format reported by your swap chain.

initialViewProjection: Frame interpolation needs the view-projection matrices of both the frame before and the frame after each interpolated frame. At context creation there is no previous frame yet, so this field seeds the matrix of the frame preceding the first interpolation; pass the same matrix you will use for the first rendered frame.

flags: A bitwise-OR of values from enum [`FfxApiCreateContextFramegenerationFlags`](../../ffx-api/include/ffx_api/ffx_framegeneration.h#L39). They fall into two groups:

- **Input-description flags** describe how your depth, motion-vector and color inputs are laid out (inverted depth, infinite far plane, HDR color, display-resolution or jitter-cancelled motion vectors). Set these to match how your engine produces its buffers.
- **Scheduling flags** (the `*_COMPUTE` bits) select compute jobs for individual pipeline stages. When a compute bit is unset, the corresponding stage uses its default fragment job.

| Flag bits | Description |
|-----------|-------------|
|FFX_API_FG_CONTEXT_FLAG_ENABLE_DISPLAY_RESOLUTION_MOTION_VECTORS|A bit indicating if the motion vectors are rendered at display resolution.|
|FFX_API_FG_CONTEXT_FLAG_ENABLE_MOTION_VECTORS_JITTER_CANCELLATION|Removes sample-position jitter from backward motion vectors using `jitterOffset` in render pixels. Leave disabled when the application already supplies unjittered motion vectors.|
|FFX_API_FG_CONTEXT_FLAG_ENABLE_DEPTH_INVERTED|A bit indicating that the input depth buffer data provided is inverted [1..0].|
|FFX_API_FG_CONTEXT_FLAG_ENABLE_DEPTH_INFINITE|A bit indicating that the input depth buffer data provided is using an infinite far plane.|
|FFX_API_FG_CONTEXT_FLAG_ENABLE_HIGH_DYNAMIC_RANGE|A bit indicating if the input color data provided to all inputs is using a high-dynamic range.|
|FFX_API_FG_CONTEXT_FLAG_WARP_COMPUTE|Use compute jobs for the warp stages (MV TP1 + flow TM1). When unset, these stages use fragment jobs.|
|FFX_API_FG_CONTEXT_FLAG_FILL_HOLES_COMPUTE|Use compute jobs for the fill-holes stages (TP1 + TM1). When unset, these stages use fragment jobs.|
|FFX_API_FG_CONTEXT_FLAG_PRE_PROCESS_COMPUTE|Use a compute job for preprocess. When unset, preprocess uses a fragment job.|
|FFX_API_FG_CONTEXT_FLAG_POSTPROCESS_COMPUTE|Use a compute job for postprocess. When unset, postprocess uses a fragment job.|
|FFX_API_FG_CONTEXT_FLAG_DEBUG_VIEW_COMPUTE|Use a compute job for debug view. When unset, debug view uses a fragment job.|
|FFX_API_FG_CONTEXT_FLAG_ALL_STAGES_COMPUTE|Convenience flag that selects compute jobs for all selectable stages listed above.|
|FFX_API_FG_CONTEXT_FLAG_MV_HINTS_COMPUTE|Generate motion vector hints in a compute shader when hints are enabled. When unset, the fragment shader is used.|
|FFX_API_FG_CONTEXT_FLAG_ENABLE_MV_HINTS| Enables generation and use of optical-flow motion vector hints. |
|FFX_API_FG_CONTEXT_FLAG_MANAGE_PREVIOUS_DEPTH|A bit indicating that the SDK manages the previous depth internally. When set, the SDK allocates and ping-pongs the depthTm1 resource; when not set, the user must supply depthTm1 on each dispatch.|
|FFX_API_FG_CONTEXT_FLAG_MANAGE_PREVIOUS_COLOR|A bit indicating that the SDK manages the previous color internally. When set, the SDK allocates and ping-pongs the colorTm1 resource; when not set, the user must supply colorTm1 on each dispatch.|

The default fragment warp and preprocess stages require Vulkan's `fragmentStoresAndAtomics` feature to be supported and enabled on `VkDevice`. If the physical device does not support it, set both `FFX_API_FG_CONTEXT_FLAG_WARP_COMPUTE` and `FFX_API_FG_CONTEXT_FLAG_PRE_PROCESS_COMPUTE`; setting only the warp flag leaves preprocess on the fragment path and context creation rejects it.

NFRU selects FP16 shader permutations when `fp16Supported` reports physical-device support. On Vulkan, applications must enable `shaderFloat16`, `storageBuffer16BitAccess`, and `uniformAndStorageBuffer16BitAccess` when creating `VkDevice`; `fp16Supported` cannot report which optional features were enabled. NFRU does not expose a context flag to force FP32.

When FP16 is supported, the frame-interpolation shader-blob accessor selects
16-bit variants for internal passes where those variants are defined. They do
not change caller-supplied resource formats or the INT8 data graph. FP16
rounding can change reconstruction pixels relative to FP32; bit-identical
results and a universal per-pixel FP16-to-FP32 error bound are not promised.

NFRU does not promise bit-identical output across devices, drivers, shader
compilers, precision paths, or execution modes. Validate image quality with
representative inputs on the target device and configuration; test tolerances
are not universal per-pixel or run-to-run determinism bounds.

For the host Optical Flow API, `mvDepthSize` may be `{0, 0}` to use `maxRenderSize` as the depth/motion input size; explicit dimensions must both be nonzero and fit signed 32-bit shader coordinates. Partial-zero and out-of-range values are rejected during context creation. When hints are enabled, the current `depth` must match the input size, while previous `depthTm1` may match either the input size or resolved lane. Dispatch rejects other dimensions before registering resources or scheduling GPU work. The returned `depthTm1` and `depthTm1Next` resource descriptions use the resolved lane dimensions.

Frame-generation and frame-interpolation context `displaySize`, `maxRenderSize`, and resolved `mvDepthSize` dimensions must fit the signed 32-bit shader-coordinate range. Derived and 8-pixel-aligned internal NFRU lanes are checked against the same limit before resource creation.

Note:
Motion vector hints are disabled by default. Set `FFX_API_FG_CONTEXT_FLAG_ENABLE_MV_HINTS` in the frame-generation context flags to enable them; the frame-generation provider then passes `FFX_OPTICALFLOW_ENABLE_MV_HINTS` to the Optical Flow API. Enabled hints use the fragment path by default; add `FFX_API_FG_CONTEXT_FLAG_MV_HINTS_COMPUTE` to select compute. The setting applies for the lifetime of the context, so recreate the context to change it.

Typical flag combinations:

- Standard LDR pipeline with a normal depth buffer and default fragment scheduling: `flags = 0`.
- Reverse-Z depth (most common): `FFX_API_FG_CONTEXT_FLAG_ENABLE_DEPTH_INVERTED`.
- Reverse-Z with an infinite far plane (cameraFar is then ignored): `... | FFX_API_FG_CONTEXT_FLAG_ENABLE_DEPTH_INFINITE`.
- HDR color inputs: `... | FFX_API_FG_CONTEXT_FLAG_ENABLE_HIGH_DYNAMIC_RANGE`.
- Run all selectable stages as compute jobs: `... | FFX_API_FG_CONTEXT_FLAG_ALL_STAGES_COMPUTE`. By default, selectable stages use fragment jobs to reduce bandwidth and power on tile-based GPUs. Fragment-only downsample stages remain fragment jobs even when all selectable compute flags are set.

Input resources read by both fixed fragment downsample passes and compute passes must use `FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ`. `ALL_STAGES_COMPUTE` does not turn those fixed downsample passes into compute jobs. Set each output's state according to the stage that writes it.

#### ffxDispatch

ffxDispatch is divided into two steps: "prepare dispatch" and "dispatch". "Prepare dispatch" is used to prevent situations where some information cannot be obtained when calling "dispatch".

##### Prepare Dispatch

Sample code:

```cpp
ffx::DispatchDescFrameGenerationPrepare dispatchFgPrep{};
dispatchFgPrep.commandList         = static_cast<CommandBufferImpl&>(cmdb).getHandle();
dispatchFgPrep.depth               = getDepth;
dispatchFgPrep.motionVectors       = getMotionVector;
// Required unless the selected context/path manages these histories internally.
dispatchFgPrep.depthTm1            = getPreviousDepth;
dispatchFgPrep.colorTm1            = getPreviousPresentationColor;
dispatchFgPrep.jitterOffset.x      = jitterOffset.x();
dispatchFgPrep.jitterOffset.y      = jitterOffset.y();
dispatchFgPrep.motionVectorScale.x = motionVectorsScale.x();
dispatchFgPrep.motionVectorScale.y = motionVectorsScale.y();
dispatchFgPrep.mvSimilarityThreshold = 0.25f;
dispatchFgPrep.mvSimilarityNoiseThreshold = 0.125f;
dispatchFgPrep.mvSimilarityRelativeThreshold = 0.002f;
dispatchFgPrep.frameTimeDelta      = max(deltaTimeMS, 1.0f);
dispatchFgPrep.cameraFovAngleVertical  = cameraFovV;
dispatchFgPrep.cameraFar               = cameraFar;
dispatchFgPrep.cameraNear              = cameraNear;
dispatchFgPrep.viewSpaceToMetersFactor = 0.f;
dispatchFgPrep.frameID                 = frameIdx;
memcpy(dispatchFgPrep.viewProjection, viewProjectionColumnMajor, sizeof(dispatchFgPrep.viewProjection));
ffx::ReturnCode retCode = ffx::Dispatch(m_FrameGenContext, dispatchFgPrep);
```

`viewProjectionColumnMajor` points to 16 contiguous floats containing the unjittered view-projection matrix.

The SDK's matrix contract is:

- **Math:** column-vector multiplication, `clip = viewProjection * point`.
- **Storage:** column-major raw storage, where element `(row, column)` is at `matrix[column * 4 + row]`.
- **Clip space:** Vulkan NDC, including a depth range of `0..1`.
- **Temporal state:** the matrix must be unjittered; pass temporal jitter separately in `jitterOffset`.

If an engine uses row-vector math or row-major storage, transpose or repack the matrix once before copying it into `viewProjection`. Keep the camera, depth, motion-vector, and render-size coordinate conventions consistent.

Math convention and memory layout are independent dimensions. Switching the vector side also transposes the logical matrix, so `column-vector math + row-major storage` and `row-vector math + column-major storage` can describe the same transform when the transpose is applied consistently. This explains why an integration may appear to work after changing both conventions together. It is not a second NFRU input contract: the SDK still expects the canonical column-vector/column-major raw representation above. For the SDK's canonical pair, the corresponding dual reading is row-vector math with row-major storage.

More details about [ffxApiDispatchDescFrameGenerationPrepare](../../ffx-api/include/ffx_api/ffx_framegeneration.h#L150):

| Field | Description |
|-------|-------------|
| `header.type` | Must be [`FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION_PREPARE`](../../ffx-api/include/ffx_api/ffx_framegeneration.h#L147). |
| `frameID` | Identifier used to select internal resources. Must increment by exactly 1 each frame; any other delta resets the frame generation logic. The prepare and dispatch of the same frame must use the same `frameID`. For example, `100 -> 101 -> 102` is valid, while `100 -> 105` resets the internal history and degrades interpolation quality. |
| `commandList` | The command list to record frame generation commands into. |
| `jitterOffset` | The sample-position offset in `renderSize` pixels, not MV/depth or display pixels. Use the same convention and jitter as your upscaler for the matching `frameID`, for example `{0.13f, -0.27f}`. Each output pixel at `x` samples the unjittered image at `x + jitter`; equivalently, a point whose unjittered position is `p` is rasterized at `p - jitter`. Here, `x`, `p`, and `jitter` use the same render-pixel coordinate system. If you do not jitter the camera, pass `{0, 0}`. |
| `motionVectorScale` | Converts stored motion-vector values into pixel offsets in the resolved `mvDepthSize` coordinate system (`renderSize` when omitted). For signed UV-space vectors, use `{mvDepthSize.width, mvDepthSize.height}`; for vectors already in MV/depth pixel units, use `{1, 1}`. If downsampled vectors retain render-pixel units, use `{mvDepthSize.width / float(renderSize.width), mvDepthSize.height / float(renderSize.height)}`. The display-resolution motion-vector flag retains its existing convention: scale into `displaySize` pixels instead. |
| `mvSimilarityThreshold` | Dynamic-mask similarity threshold (`eps`) used to compare camera motion and game motion in the warp-flow stage. Larger values make dynamic-mask detection less sensitive; smaller values make it more sensitive. If this value is `<= 0`, the SDK falls back to the default `0.25`. |
| `mvSimilarityNoiseThreshold` | Dynamic-mask noise threshold (`tau`) used as the low-motion floor in dynamic-mask classification. If this value is `<= 0`, the SDK falls back to the default `0.125`. |
| `mvSimilarityRelativeThreshold` | Dynamic-mask relative threshold (`rel_eps`) used with the larger motion magnitude in dynamic-mask classification. If this value is `<= 0`, the SDK falls back to the default `0.002`; NaN and positive or negative infinity return `FFX_ERROR_INVALID_ARGUMENT` from the host API and `FFX_API_RETURN_ERROR_PARAMETER` through `ffxDispatch`. |
| `frameTimeDelta` | Time elapsed since the previous frame, in milliseconds (for example about `16.6` ms at 60 fps). The sample clamps it to a minimum of `1.0` ms to avoid degenerate values on the first frame or after a stall. |
| `cameraNear` | The distance to the camera near plane, in world units (for example `0.1f`). |
| `cameraFar` | The distance to the camera far plane, in world units (for example `1000.0f`). Used only for non-infinite depth; ignored when `FFX_API_FG_CONTEXT_FLAG_ENABLE_DEPTH_INFINITE` is set. |
| `cameraFovAngleVertical` | The field of view angle (in radians) along the height (Y) axis of `renderSize`. For a landscape render (width > height) this is the usual vertical FOV; for a portrait render (height > width, e.g. with Android swapchain pre-rotation) it must be the FOV along the now-longer height axis, which equals the landscape horizontal FOV. A mismatched value degrades depth reconstruction and interpolation quality. |
| `viewSpaceToMetersFactor` | The scale factor converting view-space units to meters. Pass `1.0f` when your view space is already in meters. As a convenience, a non-positive value (`<= 0`) selects the default factor of `1.0f`. |
| `depth` | The depth buffer image resource for the current frame. |
| `motionVectors` | The motion vector image resource for the current frame. |
| `depthTm1` | Previous frame's depth, at `mvDepthSize` or the resolved internal MV/depth lane. Required unless `MANAGE_PREVIOUS_DEPTH` is set or all-input downsampling provides internal history; ignored otherwise. |
| `colorTm1` | Previous frame's display-resolution presentation color, matching `presentColor` in format and coordinate system. Required unless `MANAGE_PREVIOUS_COLOR` is set; ignored otherwise. |
| `viewProjection` | This frame's unjittered ViewProjection matrix in column-major raw storage for column-vector math (`matrix[column * 4 + row]`), using Vulkan NDC and the same camera and coordinate convention as `depth`, `motionVectors`, and `renderSize`. Pass temporal jitter separately in `jitterOffset`; baking jitter into this matrix corrupts camera-motion comparison and the dynamic mask. The SDK reprojects between frames as `currentVP * inverse(lastVP)`, so no rotation compensation is needed as long as consecutive frames use the same coordinate definition; do not pre-multiply a rotation matrix (for example, to account for swapchain pre-rotation). |

###### Motion-vector jitter cancellation

When `FFX_API_FG_CONTEXT_FLAG_ENABLE_MOTION_VECTORS_JITTER_CANCELLATION` is
enabled, NFRU removes jitter from backward motion (current frame to previous
frame) after converting the stored vectors to UV space:

```text
T = displaySize if display-resolution motion vectors are enabled, otherwise mvDepthSize
scaledMotionUV = rawMotion * motionVectorScale / T
correctedMotionUV = scaledMotionUV - (currentJitter - previousJitter) / renderSize
```

All operations are component-wise. Jitter always uses the original `renderSize` (`maxRenderSize` in the component host API), even with reduced MV/depth inputs or display-resolution motion. For example, a 0.75-render-pixel horizontal jitter delta at render width 1920 is `0.75 / 1920` UV regardless of MV texture width. `motionVectorScale` already handles stored units and per-axis polarity; do not apply another sign change to the jitter delta. Cancellation precedes interpolation timestep scaling and is applied once, in the MV warp pass, not to raw downsampled motion, camera-generated motion, or optical flow.

##### Dispatch

Two dispatches will be triggered: optical flow dispatch and frame interpolation dispatch. The optical flow output is automatically passed as input to the frame interpolation process.

Sample code:

```cpp
ffx::DispatchDescFrameGeneration dispatchDesc{};
dispatchDesc.commandList                = static_cast<CommandBufferImpl&>(cmdb).getHandle();
dispatchDesc.outputs[0]                 = interpolatedResource;
dispatchDesc.presentColor               = backbuffer;
dispatchDesc.reset                      = bReset;
dispatchDesc.numGeneratedFrames         = 1;
dispatchDesc.frameID                    = currentFrameID;
ffx::ReturnCode retCode = ffx::Dispatch(m_FrameGenContext, dispatchDesc);
```

More details about [ffxApiDispatchDescFrameGeneration](../../ffx-api/include/ffx_api/ffx_framegeneration.h#L110):

| Field | Description |
|-------|-------------|
| `header.type` | Must be [`FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION`](../../ffx-api/include/ffx_api/ffx_framegeneration.h#L107). |
| `commandList` | The command list on which to register frame generation commands. |
| `presentColor` | The current frame's presentation color image, used as the source for frame generation. It is a display-resolution image, typically the swap chain image you are about to present. |
| `outputs` | The destination image(s) that receive the generated frame(s), one entry per generated frame. With `numGeneratedFrames = 1`, only `outputs[0]` is used. Each output must be a display-resolution image matching `presentColor`'s format and coordinate system. |
| `numGeneratedFrames` | The number of frames to generate from the passed-in color target. NFRU currently generates one interpolated frame per rendered frame, so this is `1`. |
| `reset` | A boolean which, when true, indicates the camera moved discontinuously or the rendered scene changed. Set it on frames where interpolation against the previous frame is invalid (for example a camera teleport, a cut-scene cut, or the first frame after a level load); such frames are copied from the latest `presentColor` instead of being interpolated. |
| `frameID` | Identifier used to select internal resources. Must increment by exactly 1 each frame; any other delta resets the frame generation logic. |

##### Resources

Reset does not waive the required previous-colour or previous-depth resource bindings in external-history mode.

**Coordinate system:** All inputs and outputs (`depth`, `motionVectors`, `presentColor`, `outputs`, `renderSize`, `mvDepthSize`, `displaySize`, `viewProjection`, `cameraFovAngleVertical`) must be defined in the same screen-space coordinate system. On Android a swapchain surface pre-rotation (e.g. `VK_SURFACE_TRANSFORM_ROTATE_90_BIT_KHR`) rotates the actual image memory layout: a landscape render such as 1600×720 can map to a portrait swapchain image such as 1440×3200. In that case every parameter must follow the rotated layout. The SDK is orientation-agnostic and applies no rotation itself, so keeping all inputs consistent is the application's responsibility.

**Fragment-pass image usage:** When using fragment stages, do not add
`VK_IMAGE_USAGE_STORAGE_BIT` to external images that are only sampled or used as color/depth
attachments, including `depth` and `motionVectors`. Use attachment and sampled usages that
match the application's access pattern. This lets eligible images retain AFBC compression on
Mali GPUs. Add storage-image usage only for resources that the compute path accesses as
storage images.

The resources used during the prepare and dispatch stage:

| Resource | stage | Resolution | Format | Data type | Comment |
|----------|-------|------------|--------|-----------|---------|
| depth | prepare | resolved `mvDepthSize` | R32 | float | Depth buffer for the current frame provided by the application. |
| motionVectors | prepare | resolved `mvDepthSize` | R16G16 | float |  The 2D motion vectors for the current frame are provided by the application. <br>UV(T-1) = UV(T+1) + Motion.<br>For example, if a pixel at UV (0.50, 0.50) in the current frame was at (0.52, 0.50) in the previous frame, Motion is (0.02, 0.00). |
| depthTm1 | prepare | `mvDepthSize` or internal MV/depth lane | R32 | float | Previous depth when not managed internally; retain it through the paired dispatch and GPU completion. |
| colorTm1 | prepare | display resolution | Same as presentColor | Same as presentColor | Previous presentation color when not managed internally; retain it through the paired dispatch and GPU completion. |
| presentColor | dispatch | display resolution | BGRA8 | Unorm | Current frame's presentation color, usually a swap chain image. Usually not a HDR format. Must use the same coordinate system as all other inputs (see note above); with swapchain pre-rotation the memory layout is rotated accordingly. |
| outputs | dispatch | display resolution | BGRA8 | Unorm | Destination for the generated frame. Must match presentColor's coordinate system and dimensions. |

These are typical formats, not a requirement to tonemap every integration to
BGRA8. HDR input must match `ENABLE_HIGH_DYNAMIC_RANGE` and the configured
`backBufferFormat`. With display-resolution motion enabled, the motion image
may instead use `displaySize`; depth still uses the configured `mvDepthSize`.
Output images must support transfer-destination usage for reset copies, plus
color-attachment usage for fragment postprocess/debug or storage-image usage
for compute postprocess/debug. Inputs and external histories must support
their sampled access, and `presentColor` must support transfer-source access
for reset copies. Supply accurate resource descriptions and current states.

### Lifetime and synchronization

Calls on one context are thread-affine and must be externally serialized by
its owning thread. Prepare copies descriptor values but borrows native
resources; keep those resources alive through the matching dispatch and GPU
completion. Record and execute prepare/dispatch work in order. Separate
command buffers or queues require application-provided synchronization.

Keep the Vulkan device, command buffers, input/output images, history, and
allocator callback functions/user data valid for their use. Wait for all GPU
work using the context before destruction. `fpMessage` is synchronous; retain
its implementation until destruction, consume message text during the call,
and do not throw or re-enter the active context. See
[Dataset Capture](../Capture/Capture.md) for capture-specific failure and
shutdown requirements.

#### ffxQuery

NFRU provides the following memory query types:

| Query type | Data structure | Description |
|------------|----------------|-------------|
| [`FFX_API_QUERY_DESC_TYPE_FG_GPU_MEMORY_USAGE`](../../ffx-api/include/ffx_api/ffx_framegeneration.h#L214) | `ffxApiQueryDescFrameGenerationGetGPUMemoryUsage` | Get GPU memory usage tracked by the NFRU context. |
| [`FFX_API_QUERY_DESC_TYPE_FG_CPU_MEMORY_USAGE`](../../ffx-api/include/ffx_api/ffx_framegeneration.h#L224) | `ffxApiQueryDescFrameGenerationGetCPUMemoryUsage` | Get current and peak CPU heap memory requested through the NFRU context's tracked allocator. |

To query NFRU's internal GPU memory usage after creating the frame generation context:

```cpp
FfxApiEffectMemoryUsage vramUsage{};
ffx::QueryDescFrameGenerationGetGPUMemoryUsage queryDesc{};
queryDesc.gpuMemoryUsageFrameGeneration = &vramUsage;
ffx::ReturnCode retCode = ffx::Query(m_FrameGenContext, queryDesc);
```

`queryDesc.header.type` is initialized to [`FFX_API_QUERY_DESC_TYPE_FG_GPU_MEMORY_USAGE`](../../ffx-api/include/ffx_api/ffx_framegeneration.h#L214) by `ffx::QueryDescFrameGenerationGetGPUMemoryUsage`. `vramUsage` contains the combined usage of the optical flow, frame interpolation, and shared NFRU resources:

| Field | Description |
|-------|-------------|
| `totalUsageInBytes` | Total GPU memory allocated by NFRU. |
| `textureUsageInBytes` | GPU memory allocated for texture resources. |
| `bufferUsageInBytes` | GPU memory allocated for buffer resources, including backend uniform-buffer memory. |
| `tensorUsageInBytes` | GPU memory allocated for tensor resources. |
| `dataGraphUsageInBytes` | GPU memory bound to data-graph pipeline sessions. |
| `aliasableUsageInBytes` | GPU memory that can be aliased. This is included in `totalUsageInBytes`. |

The returned byte counts cover effect and backend resources, but not application-owned resources, swapchain allocations, or driver residency. Query after the frame generation dispatch that produces the interpolated frame, rather than after the prepare dispatch, when measuring current-frame usage.

To query NFRU's CPU heap memory usage after creating the frame generation context:

This query is available only when the SDK is built with `FFX_ENABLE_CPU_HEAP_TRACKING=ON`. Otherwise, `ffx::Query` returns `FFX_API_RETURN_ERROR_UNKNOWN_DESCTYPE`.

```cpp
FfxApiEffectCpuMemoryUsage heapUsage{};
ffx::QueryDescFrameGenerationGetCPUMemoryUsage queryDesc{};
queryDesc.cpuMemoryUsageFrameGeneration = &heapUsage;
ffx::ReturnCode retCode = ffx::Query(m_FrameGenContext, queryDesc);
```

`queryDesc.header.type` is initialized to [`FFX_API_QUERY_DESC_TYPE_FG_CPU_MEMORY_USAGE`](../../ffx-api/include/ffx_api/ffx_framegeneration.h#L224) by `ffx::QueryDescFrameGenerationGetCPUMemoryUsage`. `heapUsage` contains CPU heap bytes requested through the NFRU context's tracked allocator:

| Field | Description |
|-------|-------------|
| `currentUsageInBytes` | Outstanding tracked CPU heap allocation size. |
| `peakUsageInBytes` | Maximum tracked size since the effect context was created. |

The CPU heap values include the provider context and allocations made through its tracked allocator (currently backend scratch buffers). They exclude stack memory, STL container allocations such as `std::vector` storage, allocator bookkeeping, GPU driver allocations, and dynamic allocations internal to third-party libraries.

#### ffxConfigure

Use `ffxConfigure` to control frame-generation state, debug visualization, and verbose runtime logging.

Sample code:

```cpp
ffx::ConfigureDescFrameGeneration configDesc{};
configDesc.frameGenerationEnabled = true;
configDesc.flags = FFX_API_FG_DISPATCH_FLAG_NO_SWAPCHAIN_CONTEXT_NOTIFY;
if(bEnableDebugView)
{
    configDesc.flags |= FFX_API_FG_DISPATCH_FLAG_DRAW_DEBUG_VIEW;
}
if(bEnableDebugChecking)
{
    configDesc.flags |= FFX_API_FG_DISPATCH_FLAG_ENABLE_DEBUG_CHECKING;
}
ffx::ReturnCode retCode = ffx::Configure(m_FrameGenContext, configDesc);
```

header.type: Must be [`FFX_API_CONFIGURE_DESC_TYPE_FRAMEGENERATION`](../../ffx-api/include/ffx_api/ffx_framegeneration.h#L126).

flags: `FFX_API_FG_DISPATCH_FLAG_DRAW_DEBUG_VIEW` enables the debug view. `FFX_API_FG_DISPATCH_FLAG_ENABLE_DEBUG_CHECKING` enables verbose per-frame runtime logs. `FFX_API_FG_DISPATCH_FLAG_NO_SWAPCHAIN_CONTEXT_NOTIFY` selects the manual dispatch path used by the NFRU sample.

When the debug view is enabled, the output is split into a 3x3 grid of internal surfaces ("tm1" means "time minus 1", the previous frame; "tp1" means "time plus 1", the next frame), which is useful for diagnosing interpolation artifacts:

|  | Column 1 | Column 2 | Column 3 |
|--|----------|----------|----------|
| Row 1 | warped depth tm1 | warped depth tp1 | disocclusion mask tm1 |
| Row 2 | disocclusion mask tp1 | motion warped tm1 color | motion warped tp1 color |
| Row 3 | OF warped tm1 color | OF warped tp1 color | dynamic mask blended on current-frame color |

The first eight tiles dequantize the stored model-input tensor features; they
are not raw resource dumps. Depth and mask tiles use the red channel. The
ninth tile samples the current presentation color and overlays the TP1 dynamic
mask. Use captured raw inputs as well as these tiles when validating formats
and depth conventions.

### Runtime logging

NFRU logs versions, context state changes, and structured errors. The optional `fpMessage` callback receives these records independently of the built-in sink level.

Set `FFX_SDK_LOG_LEVEL=info` before the first SDK call to display INFO records. Add `FFX_API_FG_DISPATCH_FLAG_ENABLE_DEBUG_CHECKING` to enable per-frame Prepare, Dispatch, and CPU timing records:

```cpp
ffx::ConfigureDescFrameGeneration configDesc{};
configDesc.frameGenerationEnabled = true;
configDesc.flags = FFX_API_FG_DISPATCH_FLAG_NO_SWAPCHAIN_CONTEXT_NOTIFY |
                   FFX_API_FG_DISPATCH_FLAG_ENABLE_DEBUG_CHECKING;
ffx::Configure(m_FrameGenContext, configDesc);
```

Per-frame records use `ctx_id` and `frame_id` and summarize resources, reset state, OF/FI stages, and CPU-side scheduling time. `cpu_*_us` fields do not measure GPU or neural execution. Prepare and Dispatch must use the same frame ID, with one pending Prepare per context. Input-validation errors may be corrected and retried; after an OF/FI backend failure, discard the command list and recreate the context. The SDK reports required fallback but does not perform it.

### Troubleshooting NFRU Integration

Use the [Common Validation Workflow](../integration_troubleshooting.md#common-validation-workflow) for the input-first diagnostic order and the [NFRU troubleshooting guide](../integration_troubleshooting.md#nfru-troubleshooting) for symptom-specific procedures and visual examples. The [3x3 debug-view layout](#ffxconfigure) defines the tiles used in these checks.

#### Depth convention and disocclusion

For depth and mask comparisons, see [Incorrect Disocclusion Regions](../integration_troubleshooting.md#incorrect-disocclusion-regions). Input dimensions and coordinate requirements are defined in [Resources](#resources).

#### Motion-vector artifacts

For polarity, scale, decoding, and frame-pairing checks, see [Block Artifacts: Motion-Vector Polarity or Scale](../integration_troubleshooting.md#block-artifacts-motion-vector-polarity-or-scale). The `motionVectorScale` contract is defined in [Prepare Dispatch](#prepare-dispatch).

#### View-projection matrix and dynamic mask

For dynamic-mask comparisons, see [Incorrect Dynamic Mask: View-Projection Matrix](../integration_troubleshooting.md#incorrect-dynamic-mask-view-projection-matrix). The matrix and jitter contracts are defined in [Prepare Dispatch](#prepare-dispatch).

#### Candidate alignment

For both-time-direction comparisons and expected alignment, see [Misaligned Candidate Frames](../integration_troubleshooting.md#misaligned-candidate-frames).

## Internal Resolution Capping

NFRU derives a fixed set of internal lanes at context creation to bound bandwidth and compute cost. Applications do not configure these lanes manually; they follow `displaySize`, the resolved `mvDepthSize` (`renderSize` when omitted), and the required optical-flow grid:

| Lane | Size rule | Notes |
|------|-----------|-------|
| Full-resolution colour / final output | `displaySize` | Also the full-resolution postprocess history. |
| OF input colour | `displaySize` up to `1080p`; above that, height is fixed to `1080` and width is rounded with positive ties-to-even | This is the dedicated OF colour lane and is independent from preprocess colour history. |
| Preprocess / MV-depth logical lane | `mvDepthSize` while `mvDepthSize.height < 540`; otherwise height is fixed to `540` and width uses the `displaySize` aspect ratio with the same ties-to-even rounding | This is the logical lane consumed by warp, preprocess, and the graph input. |
| NFRU flow lane | Height is `max(1, floor(ofInputColour.height / 4))`; width is rounded from that height using the `displaySize` aspect ratio and positive ties-to-even | The physical OF texture remains independently ceiling-sized for its supported `1x1`, `2x2`, `4x4`, or `8x8` grid and is sampled into this lane. |
| Tensor lane (physical storage) | `align8(preprocessLane)` per axis | The tensor lane is **not** capped at `270p`; it tracks the logical preprocess lane and is only padded for the graph input tensor. |
| Params / gate lanes (physical storage) | `paddedTensorLane / 2` | The model writes half-resolution params and gate outputs into the padded tensor domain. The logical content remains `floor(mvDepthLane / 2)` inside that physical image. |

Context creation rejects dimensions whose aspect ratio would round any derived lane width to zero.

NFRU then selects one of three pre-warp topologies, plus an optional OF-colour prepass:

- **All-input downsample**: selected when the MV/depth-input-to-preprocess ratio reaches `1.5x` or more on the fixed `540p` path. Colour, depth, and motion are all reduced into the `540p` preprocess lane, and reduced depth / reduced motion resources are kept at that lane size. Example: `3840x2160` display with `1920x1080` render and omitted `mvDepthSize` uses `1920x1080` OF colour, `960x540` preprocess, `480x270` flow, `960x544` padded tensor, and `480x272` physical params/gate.
- **Colour-only downsample**: selected when the MV/depth-input-to-preprocess ratio stays below `1.5x` but display-to-preprocess reaches `1.5x` or more. Only colour is explicitly reduced before warp. Example: `3840x2160` display, `1920x1080` render, and explicit `960x540` MV/depth select this path, with the same internal lane dimensions as above but no additional MV/depth reduction. `2560x1440` display with `1280x720` render and omitted `mvDepthSize` also selects this path.
- **No-downsample**: selected when both ratios stay below `1.5x`. No pre-warp downsample pass runs; warp reads the original MV/depth inputs by normalized coordinates.

The histories are independent:

- Full-resolution postprocess colour history always stays at `displaySize`.
- OF colour `P1/M1` history exists only when the `1080p` OF cap is active.
- Preprocess colour `P1/M1` history exists only for the all-input and colour-only topologies.
- SDK-managed depth history uses `mvDepthLane`. All-input downsample writes it; the other topologies write it from the warp pass. All-input downsampling requires internal depth history even if `MANAGE_PREVIOUS_DEPTH` is unset. Otherwise, when that flag is unset, the application must supply `depthTm1` at `mvDepthSize` or already reduced to `mvDepthLane`; NFRU samples external history by normalized coordinates. Optical-flow hints use the same logical lane and the actual bound history extent.

Prepare validates current depth and motion against the configured input size, and depth history against its input or internal lane size, before updating temporal state or registering resources. Legacy display-sized motion resources remain accepted with the display-resolution motion-vector flag.

Implementation note: samplerless integer fetches in the warp, preprocess, and postprocess paths clamp derived pixel coordinates to the valid image extent before `texelFetch`/raw load operations. Off-image reprojection therefore reuses the nearest edge texel instead of relying on undefined out-of-bounds reads.

## Limitations

- Only quantized data graphs are supported.

---

## Presentation Pacing

NFRU generates one interpolated frame per rendered frame, adding a fixed latency of one frame. The SDK only produces the interpolated image; **how the rendered and generated frames are submitted to the swap chain and timed is entirely the application's responsibility**.

A typical presentation sequence:

1. Render frame N (optionally upscaling to display resolution with NSS or another upscaler).
2. Call `ffxDispatch` (prepare + dispatch); NFRU generates the interpolated frame for N.
3. Present frame N-interpolated first (earlier presentation time).
4. Present frame N (real frame, later presentation time).

On the first dispatch, an explicit reset, or a skipped/decreasing frame ID,
the generated output is a copy of the current frame rather than an
interpolation. Avoid treating that duplicate as an additional temporal sample.

Maintain consistent frame pacing to avoid judder. Do not present both frames at identical timestamps.

---

## Integration Checklist

See the [NFRU Integration Checklist](../integration_troubleshooting.md#nfru-integration-checklist) in the integration troubleshooting guide.