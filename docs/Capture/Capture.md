<!--
SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
SPDX-License-Identifier: MIT
-->

# Dataset Capture

Dataset capture is an optional per-context diagnostic component exposed by
`ffxApiConfigureDescCapture` and `ffxApiQueryDescCaptureStatus`. It records
runtime NSS and NFRU inputs, outputs, and temporal metadata for offline replay.
The capture API writes datasets; it does not replay them.

## Build configuration and bounded inactive-path cost

### Default build configuration

Dataset capture is **enabled by default** in the SDK build
(`FFX_CAPTURE_ENABLED=ON`), compiling both the capture API and the
environment/property configuration path into the SDK library:

This default makes capture readily available to developers during integration
and debugging. When capture is not requested, its runtime overhead is expected
to be negligible, but it is not strictly zero.

For production builds, we still recommend setting `FFX_CAPTURE_ENABLED=OFF` to
remove capture support and achieve the best runtime performance.

```bash
# Default build compiles dataset capture into the SDK library automatically:
python build.py -t Release -b vk_windows_x64
python build.py -t Release -b vk_linux_x64
python build.py -t Release -b vk_android_arm
```

To build an SDK library with capture completely stripped from binary code:

```bash
python build.py -t Release -b vk_linux_x64 --cmake-args="-DFFX_CAPTURE_ENABLED=OFF"
```

When compiled with `FFX_CAPTURE_ENABLED=OFF`, the capture component is omitted
from the library. Submitting capture descriptors to `ffxConfigure` or querying
capture status via `ffxQuery` synchronously returns
`FFX_API_RETURN_ERROR_PARAMETER`.

When capture is compiled in, the per-context controller is created eagerly
during context creation. The controller is lightweight: capture-session
bookkeeping remains present, while staging/readback resources and the worker
remain lazy until the first eligible dispatch. Payload files are created
asynchronously after capture data is submitted. The inactive path still
performs fixed controller and frame bookkeeping; it is bounded rather than a
zero-overhead guarantee.

### Inactive-path overhead

Compiling capture into the SDK does not make every context an active capture
session. When no capture is pending or active:

- context creation allocates only the eager controller and fixed bookkeeping;
- no staging pool, readback buffer, worker thread, or payload output file is
  created;
- dispatch and prepare do not record readback commands, map staging memory, or
  perform filesystem I/O; and
- environment/property polling remains disabled unless a valid startup
  configuration was found during context creation.

When an existing build directory was configured with an older default of
`FFX_CAPTURE_ENABLED=OFF`, CMake preserves that cached value. Use `--clean` or
pass `-DFFX_CAPTURE_ENABLED=ON` explicitly when upgrading such a build.

### Configuration migration

Capture is not yet part of a released SDK. The current configuration contract
is therefore the contract to use for new integrations:

- `FFX_CAPTURE_ENABLED` is `ON` by default. `FFX_CAPTURE_ENABLED=OFF` removes
  capture support, including the programmatic capture descriptors.
- Environment/property capture requires a complete request, including
  `CONFIG_REVISION`/`revision`, to be published before context creation.
  Context creation performs the startup read; use the programmatic API when
  configuration must be supplied after startup.
- API BEGIN output directories and environment/property `DIR`/`dir` values are
  limited to 512 bytes. Overlong API paths return
  `FFX_API_RETURN_ERROR_PARAMETER`; oversized environment/property values are
  rejected.
- Use `debug.ngsdk.nss.*` / `debug.ngsdk.nfru.*` for Android capture.
- Use `NGSDK_CAPTURE_POOL_HOST_VISIBLE_RATIO` on desktop and
  `debug.ngsdk.pool_ratio` on Android for the staging-pool ratio.
- Use `NGSDK_CAPTURE_PERFORMANCE` on desktop and `debug.ngsdk.capture.perf`
  on Android for performance diagnostics.

## Developer integration guide

### 1. Workflow options: Option A (Non-intrusive CLI) vs. Option B (Programmatic API)

Developers and test engineers have **two alternative ways** to capture
datasets. Use one active control source per context. API mode is available
without any environment/property configuration and can permanently take over
when an API request is accepted:

| Option | Primary use case | Requires code changes? | Control method |
| --- | --- | --- | --- |
| **Option A: Environment/property control** | Existing applications, QA, automated CI, and non-intrusive capture | **Zero code changes** | Set desktop environment variables or Android system properties before context creation |
| **Option B: Programmatic capture API** | In-engine tools, hotkeys, and automated runtime control | Minimal `ffxConfigure` / `ffxQuery` calls | Submit `FFX_API_CAPTURE_BEGIN` / `FFX_API_CAPTURE_CANCEL` and query status with `ffxQuery` |

Both paths require captured Vulkan images and buffers to have been created
with the corresponding transfer-source usage flags. Capture also requires the
submitted command buffer to run on a queue family with non-zero timestamp
support.

#### Priority and relationship between Option A and Option B

- A valid `FFX_API_CAPTURE_BEGIN` always takes priority over pending
  environment/property requests and replaces those pending requests. A second
  API `BEGIN` is rejected with `FFX_API_RETURN_ERROR_PARAMETER` while the first
  API request is still pending.
- Environment/property configuration is optional for API capture. A valid
  startup environment/property configuration is needed only to enable the
  environment fallback polling path.
- `FFX_API_CAPTURE_CANCEL` stops an active capture and selects API mode for the
  context. Subsequent environment/property revisions are ignored.
- Do not mix both control paths on one context. Use Option A for non-intrusive
  testing and Option B for explicit in-engine control.

> **Prerequisites for both options:** Every captured image and buffer must have
> been created with the corresponding transfer-source usage flag
> (`VK_IMAGE_USAGE_TRANSFER_SRC_BIT` or `VK_BUFFER_USAGE_TRANSFER_SRC_BIT`).
> The command buffer must be submitted to a queue family with non-zero
> timestamp support. If these requirements are already satisfied, Option A
> works without code changes.

### 2. Option A: Non-intrusive CLI / environment control (Android vs. PC)

With Option A, no capture API calls are required. Configure the OS environment
variables or Android system properties before launching the application or
before creating the SDK context.

#### Environment Variables and System Properties Reference

The following table lists all configuration variables for NSS and NFRU.

| Variable suffix | Desktop environment variable | Android system property | Required / default | Description |
| --- | --- | --- | --- | --- |
| `DIR` / `dir` | `NSS_CAPTURE_DIR` / `NFRU_CAPTURE_DIR` | `debug.ngsdk.nss.dir` / `debug.ngsdk.nfru.dir` | **Required** | Output root; it must not already exist. `DIR`/`dir` is limited to 512 bytes, including API BEGIN output directories. |
| `FRAME_COUNT` / `frames` | `NSS_CAPTURE_FRAME_COUNT` / `NFRU_CAPTURE_FRAME_COUNT` | `debug.ngsdk.nss.frames` / `debug.ngsdk.nfru.frames` | **Required** | Maximum number of frames; must be greater than zero. |
| `CONFIG_REVISION` / `revision` | `NSS_CAPTURE_CONFIG_REVISION` / `NFRU_CAPTURE_CONFIG_REVISION` | `debug.ngsdk.nss.revision` / `debug.ngsdk.nfru.revision` | **Required (`uint64`)** | Monotonic revision; publish it last after all other settings have been written. |
| `START_FRAME` / `start` | `NSS_CAPTURE_START_FRAME` / `NFRU_CAPTURE_START_FRAME` | `debug.ngsdk.nss.start` / `debug.ngsdk.nfru.start` | Optional (`0`) | Absolute dispatch frame at which recording starts. |
| `MV_UNORM_BIAS` / `mv_unorm_bias` | `NSS_CAPTURE_MV_UNORM_BIAS` / `NFRU_CAPTURE_MV_UNORM_BIAS` | `debug.ngsdk.nss.mv_unorm_bias` / `debug.ngsdk.nfru.mv_unorm_bias` | Optional (`0.5`) | Finite motion-vector UNORM bias. |
| `MV_UNORM_SCALE` / `mv_unorm_scale` | `NSS_CAPTURE_MV_UNORM_SCALE` / `NFRU_CAPTURE_MV_UNORM_SCALE` | `debug.ngsdk.nss.mv_unorm_scale` / `debug.ngsdk.nfru.mv_unorm_scale` | Optional (`2.0`) | Finite, non-zero motion-vector UNORM scale. |
| `IMAGE_OUTPUT` / `image_output` | `NSS_CAPTURE_IMAGE_OUTPUT` / `NFRU_CAPTURE_IMAGE_OUTPUT` | `debug.ngsdk.nss.image_output` / `debug.ngsdk.nfru.image_output` | Optional (`exr`) | `exr` output or `raw` binary output converted by Bin2Exr. |
| Value length | NSS/NFRU request values | NSS/NFRU request values | N/A | Values read by the NSS/NFRU environment configuration reader are limited to 4096 bytes; `DIR`/`dir` and API BEGIN output directories are limited to 512 bytes. Oversized NSS/NFRU request values are rejected synchronously. The diagnostic-performance setting is parsed independently, and an invalid or oversized staging-pool ratio keeps the default `0.125` ratio. |
| Diagnostic timestamps | `NGSDK_CAPTURE_PERFORMANCE` | `debug.ngsdk.capture.perf` | Optional (`0`) | Set to `1` or `true` to emit `[NSS-PERF]` / `[NFRU-PERF]` markers. |
| Staging pool ratio | `NGSDK_CAPTURE_POOL_HOST_VISIBLE_RATIO` | `debug.ngsdk.pool_ratio` | Optional (`0.125`) | Ratio of the host-visible memory budget used for staging pages. |

#### How to capture after navigating to a specific game scene

##### On Android

Set the properties before launching the game or creating the context:

```bash
adb shell setprop debug.ngsdk.nss.dir /sdcard/Android/data/com.example.game/files/captures/scene_boss
adb shell setprop debug.ngsdk.nss.frames 60
adb shell setprop debug.ngsdk.nss.start 1800
adb shell setprop debug.ngsdk.nss.revision 1  # Publish revision last
```

Launch the game, then pull the resulting dataset:

```bash
adb pull /sdcard/Android/data/com.example.game/files/captures/scene_boss ./captured_scene_boss
```

##### On PC (Launch-time environment variables)

Set the variables before launching the application. `START_FRAME` can delay
capture until the desired scene; for example, 1800 frames is approximately
30 seconds at 60 FPS.

```text
# Windows CMD
set NSS_CAPTURE_DIR=C:\captures\scene_boss
set NSS_CAPTURE_FRAME_COUNT=60
set NSS_CAPTURE_START_FRAME=1800
set NSS_CAPTURE_CONFIG_REVISION=1
my_vulkan_game.exe

# Linux Bash
export NSS_CAPTURE_DIR=/tmp/captures/scene_boss
export NSS_CAPTURE_FRAME_COUNT=60
export NSS_CAPTURE_START_FRAME=1800
export NSS_CAPTURE_CONFIG_REVISION=1
./my_vulkan_game
```

For NFRU, substitute the `NSS_CAPTURE_` prefix with `NFRU_CAPTURE_`.

For exact interactive capture after navigating in gameplay, use Option B and
bind a key or debug-menu action to `FFX_API_CAPTURE_BEGIN`.

The environment/property path writes all fields first and publishes the
revision last. The reader samples the revision before and after reading the
settings and ignores configurations whose revision changes during the read.
This is a best-effort publication protocol: reading the revision does not
reserve or lock the other fields, and the reader cannot guarantee that it will
finish reading them before another process or thread updates them. Producers
must therefore keep all fields unchanged until the revision has been published
and avoid modifying a configuration while a reader may be consuming it. On
Android, the adapter reads system properties only and never falls back to
`getenv()`.

An accepted startup configuration enables bounded environment polling while
the controller is IDLE, including while a queued request waits for its `start`
frame. Polling continues after the initial capture completes.

### 3. Option B: Programmatic capture API (In-engine integration)

Use Option B for programmatic, in-game control such as a developer hotkey,
debug UI, or automated test harness.

#### 1. Header setup

Include `<ffx_api/ffx_api.h>` or `<ffx_api/ffx_api.hpp>`, along with
`<ffx_api/ffx_nss.h>` or `<ffx_api/ffx_framegeneration.h>`.

#### 2. Data structures and lifecycle

- `ffxApiConfigureDescCapture` is passed to `ffxConfigure` to begin or cancel
  capture.
- `ffxApiQueryDescCaptureStatus` is passed to `ffxQuery` to inspect `phase`,
  `result`, `capturedFrames`, and `requestedFrames`.
- Phases are `IDLE`, `CAPTURING`, and `DRAINING`.
- Results are `NONE`, `SUCCESS`, `FAILED`, and `CANCELLED`.

#### 3. Allocator callback lifetime

When custom `ffxAllocationCallbacks` are passed to `ffxCreateContext`, the
allocation functions and `pUserData` pointer must remain valid through
`ffxDestroyContext`. The SDK may retain and invoke them during later capture
configuration and context destruction.

#### 4. Render-thread execution flow (Pseudo-code)

This schematic flow arms the already-created controller before the target
dispatch. Initialize each descriptor's `header.type` to the corresponding
`FFX_API_CONFIGURE_DESC_TYPE_CAPTURE`, `FFX_API_DISPATCH_DESC_TYPE_NSS`, or
`FFX_API_QUERY_DESC_TYPE_CAPTURESTATUS`, and terminate its extension chain.
The functions below represent application-side wrappers, not Python bindings
provided by the SDK.

```python
isCapturing = False
capturePending = False

def OnGameRenderTick(frameIndex, nssContext):
    if frameIndex == 100 and not isCapturing:
        captureBeginDesc = ffxApiConfigureDescCapture(
            operation       = FFX_API_CAPTURE_BEGIN,
            outputDirectory = "./captures/nss_run_100",
            frameCount      = 60,
            startDispatch   = frameIndex,
            motionVectorUnormBias  = 0.5,
            motionVectorUnormScale = 2.0
        )
        ret = ffxConfigure(nssContext, captureBeginDesc)
        if ret == FFX_API_RETURN_OK:
            isCapturing = True
            capturePending = True
            captureStartFrame = frameIndex

    if isCapturing and frameIndex == captureStartFrame + 10:
        captureCancelDesc = ffxApiConfigureDescCapture(
            operation = FFX_API_CAPTURE_CANCEL
        )
        ret = ffxConfigure(nssContext, captureCancelDesc)
        if ret == FFX_API_RETURN_OK:
            # Keep polling while the cancelled capture drains.
            capturePending = True

    cmdBuffer = BeginCommandBuffer()
    RenderScene(cmdBuffer)
    nssDispatchDesc = ffxApiDispatchDescNss(
        commandList   = cmdBuffer,
        color         = sceneColorImage,
        depth         = sceneDepthImage,
        motionVectors = sceneMotionVectorsImage,
        output        = presentationOutputImage,
        jitterOffset  = currentJitterOffset,
        frameTimeDelta= deltaTimeMs,
        ...
    )
    ffxDispatch(nssContext, nssDispatchDesc)
    SubmitCommandBuffer(cmdBuffer)

    if capturePending:
        status = FfxApiCaptureStatus()
        statusDesc = ffxApiQueryDescCaptureStatus()
        statusDesc.pOutStatus = &status
        ffxQuery(nssContext, statusDesc)
        if (status.phase == FFX_API_CAPTURE_PHASE_IDLE and
          status.result != FFX_API_CAPTURE_RESULT_NONE):
            print("Capture complete! Result code:", status.result)
            print("Captured frames:", status.capturedFrames)
            capturePending = False
            isCapturing = False
```

#### Order of operations summary

1. **Configure before dispatch:** issue `BEGIN` before the dispatch whose data
   should be captured.
2. **Snapshot in dispatch:** the component dispatch records input/output
   snapshots into host-visible staging buffers.
3. **Cancel before dispatch:** issue `CANCEL` before the later dispatch that
   should not be captured.
4. **Poll status after dispatch:** check the query return code, then wait for
  `phase == IDLE` and a terminal result (`SUCCESS`, `FAILED`, or `CANCELLED`).
  An armed request waiting for its start dispatch can still be `IDLE` with
  result `NONE`; that is not completion. `DRAINING` is not complete either.

Check dispatch and submission errors before polling as well. On a dispatch
error after capture commands were recorded, discard that command buffer as
described in [Lifecycle and synchronization](#lifecycle-and-synchronization).

Normal frame-count completion reports `SUCCESS`. An early `CANCEL` reports
`CANCELLED` after the drain completes, and payload, metadata, allocation, or
file errors report `FAILED`.

NFRU capture follows the same lifecycle across each
`ffxApiDispatchDescFrameGenerationPrepare` and
`ffxApiDispatchDescFrameGeneration` pair. A custom `ffxAllocationCallbacks`
passed to context creation must remain valid through context destruction,
because capture may use those callbacks during later configuration and
shutdown.

API `BEGIN` has priority over pending environment/property requests and
replaces them. A second API `BEGIN` is rejected while an API request is
pending. An accepted API `BEGIN` or `CANCEL` selects API mode for the context;
later environment/property revisions are ignored.

### 4. Captured dataset storage location and directory structure

All files are saved inside the directory specified by `outputDirectory`.

#### Example output locations

- **PC:** The path supplied to `outputDirectory`, such as
  `./captures/<name>` relative to the current working directory.
- **Android:** An application-writable path supplied by the caller, such as
  `/sdcard/Android/data/<package_name>/files/<custom_capture_folder>`.

#### Dataset directory layout

##### NSS dataset layout

Inside `<outputDirectory>`:

```text
<outputDirectory>/
├── 0000.json                        # Dataset metadata and camera parameters
├── results/
│   └── NGSDK/
│       ├── 0000.exr                 # Upscaled output frame 0
│       ├── 0001.exr                 # Upscaled output frame 1
│       └── ...
├── x2/                              # Ratio directory, e.g. x2, x1_5, x1_333333
│   ├── color/0000/
│   │   ├── 0000.exr                 # Low-resolution HDR/LDR input color
│   │   └── ...
│   ├── depth/0000/
│   │   ├── 0000.exr                 # Render-resolution depth buffer
│   │   └── ...
│   └── motion/0000/
│       ├── 0000.exr                 # Render-resolution motion vectors
│       └── ...
└── exposure/0000/                   # Only with the exposure-texture flag
    ├── 0000.exr
    └── ...
```

##### NFRU dataset layout

Inside `<outputDirectory>`:

```text
<outputDirectory>/
├── 0000.json                        # Frame-generation metadata and timing
├── ground_truth/0000/
│   ├── 0000.exr                     # Real rendered frame inputs
│   ├── 0002.exr
│   └── ...
├── depth/0000/                      # Or <ratio>/depth/0000 at lower resolution
│   ├── 0000.exr                     # Depth buffer
│   └── ...
├── motion_m2/0000/                  # Or <ratio>/motion_m2/0000
│   ├── 0000.exr                     # Backward motion vectors
│   └── ...
├── results/
│   └── NGSDK/
│       ├── 0001.exr                 # Generated/interpolated frame output
│       ├── 0003.exr
│       └── ...
└── optical_flow_p2/0000/
    ├── 0001.exr                     # Optical flow field
    └── ...
```

The ratio directory formats the ratio to six decimal places, replaces the
decimal point with `_`, and removes trailing zeroes, giving names such as
`x1_5` or `x1_333333`. This is rounded spelling, not an exact rational encoding.
Consumers must use the dataset/frame index and
metadata rather than infer ordering from asynchronous worker completion order.

#### Convert raw captures to EXR

With `IMAGE_OUTPUT=raw`, image payloads use `.bin` instead of `.exr`. After
capture has completed, run the bundled converter on the dataset root:

```text
Bin2Exr <capture-directory> [--depth-is-d24]
```

The converter reads `0000.json`, including its `Streams` descriptors, and
writes EXRs beside the raw files; it does not delete the `.bin` inputs. Use
`--depth-is-d24` only for 24-bit depth packed into a 32-bit value. Capture roots
and stream paths must be ASCII. Metadata must be valid JSON, no larger than
16 MiB, with a nonempty contiguous frame timeline starting at zero. Missing
payloads or byte sizes inconsistent with metadata fail conversion. Exit codes
are `0` for success, `1` for conversion/path failures, and `2` for usage or
metadata failures. Partial EXR output may remain after an error.

Build the tool with `BUILD_TOOLS=ON` and capture enabled. Source-package builds
must provide the JSON header as described in the
[Bin2Exr build and usage guide](../../sdk/tools/capture/README.md).

## Capture control API

The capture descriptor is submitted to `ffxConfigure` with an NSS or NFRU
context. It is not global, and a null context is invalid for this descriptor.
On `FFX_API_CAPTURE_BEGIN`, `outputDirectory` must be a non-null,
NUL-terminated, non-empty path and is copied before `ffxConfigure` returns.
The directory must not already exist; choose a unique path per capture because
the status API reports counts and results, not a rewritten output path.
The other BEGIN settings are copied as value data; CANCEL ignores the
BEGIN-only fields.

BEGIN queues a capture for the configured dispatch range. GPU readback,
metadata, and file output complete asynchronously, so applications should
poll `ffxApiQueryDescCaptureStatus` with `ffxQuery`. Reaching `frameCount`
automatically completes the capture successfully. CANCEL stops an active
capture early and requests a drain that marks the capture cancelled. A
DRAINING status is not complete; continue polling until IDLE.
`FAILED` is reported for payload, metadata, or worker I/O failures, and the
result remains available in the final IDLE status.

`frameCount` must be non-zero. It is a maximum request rather than a promise:
capture may stop earlier when the staging-memory budget, outstanding-slot cap,
device progress, or file output becomes unavailable. Dataset capture preserves
the raw runtime metadata values supplied by each dispatch, including atypical,
out-of-range, and non-finite values. The derived NSS and NFRU header
`EmulatedFramerate` safely falls back when its source timing cannot be
converted; it does not change the recorded per-frame `FrameTimeDelta`.
Non-finite numeric output may be non-strict JSON. Replay and other consumers
must decide whether to reject or diagnose metadata or datasets they cannot
consume.

The staging-memory budget accounts for the actual Vulkan memory allocation
requirement, including alignment overhead, rather than only the requested
buffer size. A page whose actual requirement exceeds the remaining budget or
the device maximum allocation size fails the capture without affecting the
component dispatch.

Asynchronous completion order is not part of the capture contract. Worker
callbacks may complete frames in a different order from dispatch submission;
each completed frame remains self-contained and carries its `Frame`/dataset
index and per-frame metadata. Consumers must use that index rather than infer
ordering from callback arrival or metadata write timing.

The motion-vector UNORM scale must be finite and non-zero. The documented
R8G8_UNORM defaults are a bias of `0.5` and a scale of `2.0`.

## Lifecycle and synchronization

Context destruction waits up to 30 seconds for an active capture to drain. If
Vulkan progress or file output does not return in that interval, capture
shutdown detaches the incomplete capture and retains its resources rather than
risk destroying memory that may still be in use. This bounds context
destruction, but it cannot make simultaneous `VkDevice` destruction or module
unload safe: the application must keep the device and backend code valid until
detached Vulkan or file operations have returned and command buffers containing
capture commands have completed or been permanently discarded. A permanently
blocked operation or unresolved command buffer may therefore retain its capture
resources for the rest of the process.

The capture state machine accepts one pending or active API capture per context.
BEGIN is accepted while the context is IDLE and no API BEGIN is pending,
regardless of whether environment/property configuration exists; the request
becomes CAPTURING on the first eligible component dispatch. If environment/
property requests are pending, an accepted API BEGIN supersedes them. BEGIN is
rejected while another API BEGIN is pending, CAPTURING, or DRAINING.
Environment/property configuration is read once during context creation. A
valid startup configuration enables runtime polling while IDLE, and at most 16
valid, distinct environment revisions may wait for their `startFrame`.
Additional revisions observed while that queue is full are rejected and
logged, and are not retried automatically. API BEGIN supersedes pending
environment revisions, but a second API BEGIN is rejected while an API request
is pending. Accepting API BEGIN selects API configuration for the context and
disables environment fallback.
CANCEL is accepted only while CAPTURING; it is rejected while IDLE, while a
BEGIN is still pending, or while the worker is already DRAINING. Reaching
`frameCount` is the normal automatic successful completion; CANCEL is only for
an early stop and cancellation. CANCEL drains with a CANCELLED result and
discards only the incomplete transaction. After the worker reaches IDLE, the
final result and frame counts remain queryable until the next accepted BEGIN,
which resets the result to NONE.

Configure, status query, dispatch, and destroy calls for one context must be
externally serialized by its owning thread. Separate contexts may be used
concurrently.

Capture borrows the resources and command list supplied by the component
dispatch. It records readback commands and temporary resource barriers but
does not take ownership of native resources. The application must preserve
normal Vulkan resource, command-buffer, and synchronization lifetimes until
the submitted work completes, and must provide accurate resource descriptions
and states. Every image and buffer captured by this path must have been
created with the corresponding Vulkan transfer-source usage
(`VK_IMAGE_USAGE_TRANSFER_SRC_BIT` or `VK_BUFFER_USAGE_TRANSFER_SRC_BIT`);
the capture descriptor does not carry enough native usage information to
validate this requirement.

Capture staging buffers use exclusive queue ownership. Submit all command
buffers containing capture work for one device through the same queue family;
the opaque public command-list contract does not carry a queue-family index.

Capture records readback commands before the component dispatch completes all
of its validation. If an NSS or NFRU dispatch returns an error after capture
commands have been recorded, the application must permanently discard that
command buffer rather than submit or retry it. Correct the dispatch inputs and
record a new command buffer. This applies to NSS validation errors and NFRU
optical-flow, frame-interpolation, and validation errors. Capture records its
snapshot before component work so successful datasets retain the original
input timing; it cannot safely retain staging storage across a recoverable
component dispatch failure.

Vulkan completion uses timestamp queries recorded into the supplied command
buffer. The command buffer must be submitted to a queue family with nonzero
timestamp support, and the application must not destroy or reuse its resources
until that submission completes. The capture runtime cannot infer the executing
queue family from the opaque command list. Submitting it to a queue family
without timestamp support violates the capture contract and is not a condition
the runtime can detect or recover from.

## NSS and NFRU datasets

NSS capture records the dispatch inputs and output needed by the NSS dataset
replay flow. NFRU capture records the prepare/dispatch pair, including the
ground-truth, depth, motion, generated-output, and optical-flow attachments
used by NFRU replay.

NSS and NFRU camera and timing metadata faithfully preserves the raw dispatch
or prepare descriptor values. Capture does not clamp, substitute, or reject
values for a replay consumer; consumers that cannot accept a value, including
a non-finite one, must diagnose or reject it themselves.

NFRU metadata also records whether motion-vector jitter cancellation or
display-resolution motion vectors were enabled. Capture continues when either
mode is enabled; consumers must diagnose or implement the associated replay
semantics themselves.

Environment fallback is component-scoped. On desktop, NSS uses the
`NSS_CAPTURE_` environment-variable prefix and NFRU uses `NFRU_CAPTURE_`;
both support the suffixes `DIR`, `FRAME_COUNT`, `START_FRAME`,
`MV_UNORM_BIAS`, `MV_UNORM_SCALE`, `IMAGE_OUTPUT`, and `CONFIG_REVISION`.
On Android, NSS uses the `debug.ngsdk.nss.` system-property prefix and NFRU
uses `debug.ngsdk.nfru.`; both support the suffixes `dir`, `frames`, `start`,
`mv_unorm_bias`, `mv_unorm_scale`, `image_output`, and `revision`.
Android reads system properties only and never falls back to `getenv()`.

The output directory (`DIR`/`dir`) and non-zero frame count
(`FRAME_COUNT`/`frames`) are required. The start frame
(`START_FRAME`/`start`) defaults to zero; omitted motion-vector bias and scale
use their configuration defaults. `IMAGE_OUTPUT`/`image_output` accepts
`exr` or `raw`, case-insensitively; an omitted or empty value selects `exr`.
Raw image output requires the Bin2Exr conversion step in the EXR workflow.
The public API BEGIN has no encoding field, so raw output is selected through
the environment/property configuration path.

Publish adapter settings by updating all fields first and
`CONFIG_REVISION`/`revision` last. Each changed request needs a new unique
revision. The reader samples the revision before and after reading the
settings, and ignores incomplete configurations or settings whose revision
changes during the read. This protocol does not provide a lock or an
instantaneous snapshot; callers must not update fields while a reader may be
reading them.

An accepted API BEGIN selects API configuration for the context and disables
environment/property fallback. CANCEL also selects API-only mode for future
requests. Environment fallback is checked only while the context is IDLE and
only when a valid startup configuration was found; use the capture API when no
startup configuration is present. The environment/property path is always
compiled together with capture and has no independent build switch.

Set `NGSDK_CAPTURE_PERFORMANCE=1` before the first active capture dispatch on
desktop to emit `[NSS-PERF]` or `[NFRU-PERF]` dispatch timestamp markers. This
opt-in is cached, remains quiet when unset or false, and uses the Android
property `debug.ngsdk.capture.perf` rather than an environment variable.

NFRU capture records the resources supplied by each prepare/dispatch pair.
Capture does not relax normal NFRU context layout constraints: depth must
match the context's fixed `mvDepthSize`; motion vectors must match
`mvDepthSize`, or the context's fixed `displaySize` when
display-resolution motion vectors are enabled. Recreate the NFRU context
before capturing a size that requires a different depth or motion-vector
layout. Render-size changes are supported only when the regular NFRU dispatch
validation accepts them; capture does not provide a general per-frame
dynamic-layout replay contract. NFRU `Frame` metadata is emitted from the full
64-bit dataset index mapping (`datasetIndex * 2`), without narrowing it for
legacy file-name limits.

NSS metadata records whether motion-vector jitter cancellation is enabled.
Capture continues silently with
`FFX_API_NSS_CONTEXT_FLAG_ENABLE_MOTION_VECTORS_JITTER_CANCELLATION`; replay
and other consumers must diagnose or implement its semantics themselves.

Beginning a capture forces a cold-start reset on the first captured dispatch
so the dataset contains valid temporal state initialization. Applications
should expect this intentional history transition when capture begins
mid-stream.

NSS capture with `FFX_API_NSS_CONTEXT_FLAG_EXPOSURE_TEXTURE` writes an exposure
stream that the bundled NSS replay sample does not currently upload. Use
scalar exposure for datasets intended for that sample, or provide an
integration that binds the captured exposure texture.

NSS capture output directories encode the rounded six-decimal upscale ratio
(for example, `x1_5` or `x1_333333`); replay and validation must select the
matching ratio directory.
