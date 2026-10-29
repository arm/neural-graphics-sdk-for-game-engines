# Sample Code for Neural Graphics SDK for Game Engines

This directory contains the sample applications for the Neural Graphics SDK for Game Engines. The samples demonstrate how to integrate and use the SDK's neural rendering components in a real Vulkan application.

The sample code is derived from the [Khronos Vulkan Samples](https://github.com/KhronosGroup/Vulkan-Samples) project (commit `b9961792`) and currently includes the **NSS** (Neural Super Sampling) and **NFRU** (Neural Frame Rate Upscaling) samples.

## Table of Contents

- [Sample Code for Neural Graphics SDK for Game Engines](#sample-code-for-neural-graphics-sdk-for-game-engines)
  - [Table of Contents](#table-of-contents)
  - [Samples Overview](#samples-overview)
  - [Prerequisites](#prerequisites)
  - [Fetch Sample Dependencies](#fetch-sample-dependencies)
  - [Build the Samples](#build-the-samples)
    - [Windows](#windows)
    - [Linux x64](#linux-x64)
    - [Android](#android)
  - [Vulkan Emulation Layer Setup](#vulkan-emulation-layer-setup)
  - [Run the Samples](#run-the-samples)
    - [Windows](#windows-1)
    - [Linux](#linux)
    - [Android](#android-1)
  - [Command Line Options](#command-line-options)
    - [General options](#general-options)
    - [NSS-specific arguments (via `--app-arg`)](#nss-specific-arguments-via---app-arg)
    - [NFRU-specific arguments (via `--app-arg`)](#nfru-specific-arguments-via---app-arg)
  - [Dataset Replay Config](#dataset-replay-config)
    - [NSS Config](#nss-config)
    - [NFRU Config](#nfru-config)
  - [Command Line Options Example Commands](#command-line-options-example-commands)
  - [NSS Sample Details](#nss-sample-details)
    - [Overview](#overview)
    - [UI controls - display modes](#ui-controls---display-modes)
    - [NSS sample UI panel](#nss-sample-ui-panel)
  - [NFRU Sample Details](#nfru-sample-details)
    - [Overview](#overview-1)
    - [NFRU sample UI panel](#nfru-sample-ui-panel)

---

## Samples Overview

| Sample | Component | Description | Source |
|--------|-----------|-------------|--------|
| **NSS** | Neural Super Sampling | Renders a Sponza scene or replays EXR datasets at a lower resolution and upscales to the display resolution using NSS. Demonstrates context creation, per-frame dispatch, jitter, motion vectors, history management, and the debug view. | `src/nss/` |
| **NFRU** | Neural Frame Rate Upscaling | Renders a Sponza scene or replays an EXR dataset, then uses NFRU to generate interpolated frames between consecutive rendered frames. Demonstrates context creation, frame-generation dispatch, history management, and the debug view. | `src/nfru/` |

---

## Prerequisites

| Requirement | Version |
|-------------|---------|
| CMake | See [Compatibility Matrix - Build Tool Versions](../docs/compatibility_matrix.md#build-tool-versions). |
| Python | See [Compatibility Matrix - Build Tool Versions](../docs/compatibility_matrix.md#build-tool-versions). |
| Vulkan SDK | See [Compatibility Matrix - Build Tool Versions](../docs/compatibility_matrix.md#build-tool-versions); [download from LunarG](https://www.lunarg.com/vulkan-sdk/). |
| **Windows only** | Visual Studio 2019 or higher |
| **Linux only** | C++20 compiler; install system packages: `cmake g++ xorg-dev libglu1-mesa-dev libwayland-dev libxkbcommon-dev` |
| **Android only** | Android Studio 2024.2.1+ (recommended), Android NDK (see [Compatibility Matrix - Build Tool Versions](../docs/compatibility_matrix.md#build-tool-versions)), Android SDK, JDK 21 |

> The samples also inherit all SDK prerequisites. See [User Guide - Before you begin](../docs/user_guide.md#before-you-begin).

---

## Fetch Sample Dependencies

The sample third-party repositories and large asset payloads are fetched by the platform script below.

The fetch scripts use partial clone and sparse checkout by default, so they only materialize the files needed by the sample build (including only `assets/fonts` and `assets/scenes/sponza` from the upstream asset repository).

**Windows:**

```powershell
powershell -ExecutionPolicy Bypass -File samples/scripts/fetch_sample_dependencies.ps1
```

**Linux:**

```bash
bash samples/scripts/fetch_sample_dependencies.sh
```

For source-only analysis such as Black Duck CI, the Linux script accepts
`--skip-assets` to fetch the third-party libraries without downloading sample
assets. Omit this option when preparing to run the samples.

---

## Build the Samples

After fetching the sample dependencies, build the SDK and the samples together using the `build.py` script at the repository root with `--build-samples`.

### Windows

```powershell
python build.py --clean -t Debug -b vk_windows_x64 --build-samples
```

### Linux x64

```bash
# Headless / Direct-to-Display (default)
python build.py -t Release -b vk_linux_x64 --build-samples

# Desktop with X11 (XCB) window
python build.py -t Release -b vk_linux_x64 --build-samples --cmake-args="-DVKB_WSI_SELECTION=XCB"

# Desktop with Wayland window
python build.py -t Release -b vk_linux_x64 --build-samples --cmake-args="-DVKB_WSI_SELECTION=WAYLAND"
```

> `VKB_WSI_SELECTION` also accepts `XLIB` (X11). Without `--cmake-args`, the build defaults to Direct-to-Display (`D2D`), which needs no desktop windowing libraries and can also be run headless (see [Run the Samples](#run-the-samples)).

### Android

On Windows PowerShell:

```powershell
# Set environment variables
$env:JAVA_HOME = "path\to\jdk21"
$env:NDK_ROOT  = "path\to\ndk"

python build.py --clean -t Release -b vk_android_arm --build-samples
```

On Linux:

```bash
python build.py --docker -t Release -b vk_android_arm --android-api 33 --build-samples
```

> On Android the build produces an APK at `samples/build/android_gradle/app/build/outputs/apk/release/vulkan_samples-release.apk`.

---

## Vulkan Emulation Layer Setup

The SDK requires Arm Vulkan ML extensions (`VK_ARM_tensors`, `VK_ARM_data_graph`). On desktop machines without Arm Neural Engine hardware, you must enable the [Vulkan ML Emulation Layer](https://github.com/arm/ai-ml-sdk-for-vulkan) before running any sample.

The samples request Vulkan 1.4, and NFRU additionally requests
`VK_ARM_data_graph_optical_flow`. Check the selected device, enabled features,
and compatible emulation-layer version in the
[compatibility matrix](../docs/compatibility_matrix.md); having a Vulkan SDK
installed does not establish runtime support.

Follow the [User Guide - ML Emulation Layer for Vulkan](../docs/user_guide.md#ml-emulation-layer-for-vulkan) to download and configure the layer. The guide covers the repository downloader, Vulkan Configurator, environment variables, and layer ordering. The emulation layer must be configured before running a desktop sample.

---

## Run the Samples

Run from the repository root.

### Windows

```powershell
.\build\Debug\samples\app\bin\debug\AMD64\vulkan_samples.exe sample nss
```

For release builds, use `build\Release\samples\app\bin\release\AMD64\vulkan_samples.exe`.

### Linux

**Interactive (desktop, requires an XCB/XLIB/WAYLAND build):**

```bash
./build_x64/Release/samples/app/bin/Release/x86_64/vulkan_samples sample nss
```

**Headless (no display required, default build):**

```bash
./build_x64/Release/samples/app/bin/Release/x86_64/vulkan_samples \
    sample nss --headless-surface --hideui
```

> On desktop, remember to configure the emulation layer as described in the [User Guide - ML Emulation Layer for Vulkan](../docs/user_guide.md#ml-emulation-layer-for-vulkan) before running.

### Android

**Step 1 - Install the APK:**

```bash
adb install samples/build/android_gradle/app/build/outputs/apk/release/vulkan_samples-release.apk
```

**Step 2 - Push assets and shaders** (required if no device was connected during the build):

```bash
adb push samples/assets/scenes/sponza \
    /sdcard/Android/data/com.khronos.vulkan_samples/files/assets/scenes/sponza

adb push samples/assets/fonts \
    /sdcard/Android/data/com.khronos.vulkan_samples/files/assets/fonts

adb push samples/shaders \
    /sdcard/Android/data/com.khronos.vulkan_samples/files/shaders

adb shell "chmod -R 777 /sdcard/Android/data/com.khronos.vulkan_samples/files/assets"
adb shell "chmod -R 777 /sdcard/Android/data/com.khronos.vulkan_samples/files/shaders"
```

**Step 3 - Launch the NSS sample:**

```bash
adb shell am start-activity \
    -n com.khronos.vulkan_samples/com.khronos.vulkan_samples.SampleLauncherActivity \
    -e sample nss
```

---

## Command Line Options

### General options

| Option | Description | Example |
|---------|-------------|---------|
| `--width <pixels>` | Window/surface width | `--width 1920` |
| `--height <pixels>` | Window/surface height | `--height 1080` |
| `--hideui` | Hide the on-screen UI | `--hideui` |
| `--always-render` | Continue rendering when the window is not focused | `--always-render` |
| `--headless-surface` | Use `VK_EXT_headless_surface` (no display required) | `--headless-surface` |
| `--stop-after-frame <n>` | Automatically stop after frame N | `--stop-after-frame 300` |
| `--force-close` | On Windows, suppress the final `Press return to continue` prompt on non-success exit | `--force-close` |
| `--log-file <path>` | Write runtime logs to a file; an existing log is truncated | `--log-file sample-run.log` |
| `--app-arg <KEY=VALUE>` | Forward sample-specific arguments (may be repeated or comma-separated) | `--app-arg NSS_ENABLE=1` |

### NSS-specific arguments (via `--app-arg`)

| NSS argument | Description | Default | Usage |
|---|---|---|---|
| `NSS_ENABLE` | Enable (`1`) or disable (`0`) NSS upscaling. | `1` | `--app-arg NSS_ENABLE=0` |
| `NSS_SCALE_FACTOR` | Upscale factor in `[1.0, 3.0]`. | `2.0` | `--app-arg NSS_SCALE_FACTOR=2.0` |
| `NSS_QUALITY` | Quality mode: `0`=Quality, `1`=Balanced, `2`=Performance. | `1` | `--app-arg NSS_QUALITY=1` |
| `NSS_USE_COMPUTE` | Preprocess/postprocess path: `0`=fragment, `1`=compute. | `0` | `--app-arg NSS_USE_COMPUTE=1` |
| `NSS_LDR_MODE` | Experimental input mode: `0`=HDR, `1`=LDR. Realtime mode preprocesses rendered color to LDR; dataset mode treats the stored color as LDR without preprocessing it. | `0` | `--app-arg NSS_LDR_MODE=1` |
| `NSS_COLOR_FORMAT` | Initializes both NSS input and output/history color formats: `R11G11B10`, `RGBA16F`, or `RGBA32F`. | `R11G11B10` | `--app-arg NSS_COLOR_FORMAT=RGBA16F` |
| `NSS_INPUT_COLOR_FORMAT` | Optional input color format override. Accepted values are `R11G11B10`, `RGBA16F`, and `RGBA32F`; overrides `NSS_COLOR_FORMAT` for the input image only. | Value of `NSS_COLOR_FORMAT` | `--app-arg NSS_INPUT_COLOR_FORMAT=RGBA32F` |
| `NSS_OUTPUT_COLOR_FORMAT` | Optional output/history color format override. Accepted values are `R11G11B10`, `RGBA16F`, and `RGBA32F`; overrides `NSS_COLOR_FORMAT` for output and history images only. | Value of `NSS_COLOR_FORMAT` | `--app-arg NSS_OUTPUT_COLOR_FORMAT=R11G11B10` |
| `NSS_MANAGE_HISTORY` | History ownership: `0`=the sample supplies `outputTm1`, `1`=the SDK manages history. | `0` | `--app-arg NSS_MANAGE_HISTORY=1` |
| `NSS_DEBUG_CHECKING` | Enable runtime jitter diagnostics: `0`=disabled, `1`=enabled. | `0` | `--app-arg NSS_DEBUG_CHECKING=1` |
| `NSS_DEBUG_VIEW_MODE` | Debug view tile: `0`=4x4 overview, `1`-`16`=one fullscreen tile. Providing this argument enables the debug view. | `0` | `--app-arg NSS_DEBUG_VIEW_MODE=6` |
| `NSS_DATASET_DIR` | Dataset root directory. Providing this argument enables dataset mode. | Not set | `--app-arg NSS_DATASET_DIR=path\to\dataset` |
| `NSS_DATASET_SEQUENCE` | Dataset sequence identifier. | `0000` | `--app-arg NSS_DATASET_SEQUENCE=0000` |
| `NSS_SAVE_EXR` | Save NSS outputs in dataset mode. This argument is ignored in realtime mode. | `0` | `--app-arg NSS_SAVE_EXR=1` |
| `NSS_OUTPUT_DIR` | Dataset output root. Relative paths are resolved from `NSS_DATASET_DIR`; absolute paths are used directly. The sample appends `nss_<ratio>/<sequence>`. | `result` | `--app-arg NSS_OUTPUT_DIR=results\baseline` |
| `NSS_FLAGS` | NSS create-context override flags (decimal or `0x`-prefixed hex). When omitted, the sample selects its required and configured flags. NSS uses HDR and FP16 by default; set `FFX_API_NSS_CONTEXT_FLAG_LOW_DYNAMIC_RANGE` for experimental LDR or `FFX_API_NSS_CONTEXT_FLAG_DISABLE_16BIT` for FP32. The preprocess/postprocess compute flags are `FFX_API_NSS_CONTEXT_FLAG_PRE_PROCESS_COMPUTE` and `FFX_API_NSS_CONTEXT_FLAG_POST_PROCESS_COMPUTE`; the SDK supports them independently, but this sample requires both flags to be set together because its resource state configuration is shared. | Not set | `--app-arg NSS_FLAGS=0x045` |

### NFRU-specific arguments (via `--app-arg`)

| NFRU argument | Description | Default | Usage |
|---|---|---|---|
| `NFRU_DATASET_DIR` | Dataset root directory. Providing this argument enables dataset mode. | Not set | `--app-arg NFRU_DATASET_DIR=path\to\dataset` |
| `NFRU_DATASET_SEQUENCE` | Dataset sequence identifier. | `0000` | `--app-arg NFRU_DATASET_SEQUENCE=0000` |
| `NFRU_SAVE_EXR` | Save generated frames. In realtime mode, generated frames are written to `result/nfru/interpolated/` and reference frames to `result/nfru/ground_truth/`. In dataset mode, interpolated frames are written to `<dataset>/result/nfru_x2/<sequence>/`. Realtime saving is disabled on Android. | `0` | `--app-arg NFRU_SAVE_EXR=1` |
| `NFRU_ORBIT_SPEED` | Realtime camera rotation speed in degrees per frame, in `[0, 10]`. | `0.5` | `--app-arg NFRU_ORBIT_SPEED=2` |
| `NFRU_USE_FRAGMENT` | Selectable-stage path: `0`=compute, `1`=fragment. Fixed downsample stages remain fragment. | `1` | `--app-arg NFRU_USE_FRAGMENT=0` |
| `NFRU_ENABLE_MV_HINTS` | Enable optical-flow motion vector hints: `0`=disabled, `1`=enabled. | `0` | `--app-arg NFRU_ENABLE_MV_HINTS=1` |
| `NFRU_DEBUG_VIEW` | Enable the NFRU debug view at startup: `0`=disabled, `1`=enabled. | `0` | `--app-arg NFRU_DEBUG_VIEW=1` |
| `NFRU_DEBUG_CHECKING` | Enable verbose per-frame NFRU runtime logs: `0`=disabled, `1`=enabled. | `0` | `--app-arg NFRU_DEBUG_CHECKING=1` |
| `NFRU_MANAGE_HISTORY` | History ownership: `0`=the sample supplies previous color and depth, `1`=the SDK manages history. | `1` | `--app-arg NFRU_MANAGE_HISTORY=0` |

Multiple arguments can be combined with commas:

```bash
--app-arg NSS_ENABLE=1,NSS_SCALE_FACTOR=2.0,NSS_FLAGS=0x045,NSS_QUALITY=1,NSS_DEBUG_VIEW_MODE=0
```

## Dataset Replay Config

Dataset replay optionally reads `config.json` from the dataset root. The file and every field are optional.

With `NSS_SAVE_EXR=0` or `NFRU_SAVE_EXR=0`, replay setup does not create result
directories. Enabling export creates directories and can overwrite matching
EXRs. NSS supports an external `NSS_OUTPUT_DIR`; NFRU has no output-root
override, so use an authorized disposable dataset copy when the original must
remain unchanged. Both samples keep displaying the last result after all
inputs are consumed. Close them explicitly or use `--stop-after-frame` plus a
process timeout; the frame limit is an application-frame count, not a dataset
timeline index, and must be large enough to finish the intended replay and
readbacks.

### NSS Config

For NSS, the supported fields are:

```json
{
    "Exposure": 7.389056,
    "FlipDepth": false
}
```

- `Exposure`: Exposure multiplier override. The value must be greater than zero. The sample passes `Exposure` to NSS for every frame. Without the override, it uses each frame's `Exposure` value from the sequence JSON.
- `FlipDepth`: Upload `1.0 - depth` when `true`. The effective reversed-Z convention is the sequence JSON `ReverseZ` value XOR this setting. The default is `false`.

NSS reads `<sequence>.json` and images under `x<ratio>/color/<sequence>`, `x<ratio>/depth/<sequence>`, and `x<ratio>/motion/<sequence>`. It also accepts the legacy `colour` directory name. Decimal ratios use underscores, so scale factor `1.5` maps to directory `x1_5` and JSON key `x1_5_index` under `UpscalingRatiosIndices`. With `NSS_SAVE_EXR=1`, output is written to `<dataset>/result/nss_x<ratio>/<sequence>/####.exr` by default. Set `NSS_OUTPUT_DIR` to change the output root; relative values are resolved from the dataset directory and absolute values are used directly, after which the sample appends `nss_x<ratio>/<sequence>/####.exr`.

NSS also reads the optional per-frame `MotionVectorScale` array from the sequence JSON. It must contain exactly two finite numbers, `[x, y]`, which are passed directly to the dispatch without changing their signs or multiplying by the resolution. When absent, the sample retains the legacy default `[-renderWidth, -renderHeight]` for that frame. Malformed values cause dataset loading to fail.

### NFRU Config

For NFRU, the supported fields are:

```json
{
    "Tonemap": true,
    "Exposure": 7.389056,
    "FlipDepth": true,
    "TransposeViewProjectionOnLoad": true
}
```

- `Tonemap`: If `true`, the sample applies Reinhard tonemapping to the color EXR before converting it to `R8G8B8A8_UNORM`. The default is `true`. The tonemap multiplies the color by `Exposure`. If `false`, the sample skips this tonemapping step and `Exposure` has no effect.
- `Exposure`: The exposure multiplier used by the tonemap. The value must be greater than zero.
- `FlipDepth`: Upload `1.0 - depth` when `true`. The effective reversed-Z convention is the sequence JSON `ReverseZ` value XOR this setting. The default is `false`.
- `TransposeViewProjectionOnLoad`: Transpose each JSON `ViewProjection` matrix while loading. The default is `true`.

NFRU dataset replay reads `<sequence>.json`, full-resolution color from `ground_truth/<sequence>`, low-resolution depth from `x2/depth/<sequence>`, and two-frame motion vectors from `x2/motion_m2/<sequence>`. If `motion_m2` is absent, the sample uses `x2/motion_t-2/<sequence>`. Replay consumes even-numbered source frames. Any source frame marked `Reset` initializes history and is not saved; subsequent interpolated outputs use their source timeline index, for example `0001.exr`, `0003.exr`, and `0005.exr`.

## Command Line Options Example Commands

**Windows - run at 1920x1080 (2x NSS upscaling from 960x540):**

```powershell
.\build\Debug\samples\app\bin\debug\AMD64\vulkan_samples.exe sample nss --width 1920 --height 1080
```

> **Note:** Exact 2x NSS uses the static LUT path. Non-2x ratios are supported through the dynamic LUT pass; for better performance, prefer input/output sizes that reduce to smaller modulo values.

**Windows - disable NSS:**

```powershell
.\build\Debug\samples\app\bin\debug\AMD64\vulkan_samples.exe sample nss --app-arg NSS_ENABLE=0
```

**Windows - headless automation, exit after frame 300:**

```powershell
.\build\Debug\samples\app\bin\debug\AMD64\vulkan_samples.exe sample nss --headless-surface --hideui --stop-after-frame 300 --force-close
```

**Windows - NSS dataset replay:**

```powershell
.\build\Debug\samples\app\bin\debug\AMD64\vulkan_samples.exe sample nss --always-render --stop-after-frame 300 --force-close --app-arg "NSS_DATASET_DIR=path\to\dataset,NSS_DATASET_SEQUENCE=0000,NSS_SCALE_FACTOR=2.0,NSS_SAVE_EXR=1"
```

To keep replay output outside the dataset, provide an absolute output root:

```powershell
.\build\Debug\samples\app\bin\debug\AMD64\vulkan_samples.exe sample nss --always-render --stop-after-frame 300 --force-close --app-arg "NSS_DATASET_DIR=path\to\dataset,NSS_DATASET_SEQUENCE=0000,NSS_SCALE_FACTOR=2.0,NSS_SAVE_EXR=1,NSS_OUTPUT_DIR=D:\replay-results"
```

**Windows - NFRU:**

```powershell
# Rotate 1 degree per frame:
.\build\Debug\samples\app\bin\debug\AMD64\vulkan_samples.exe sample nfru --app-arg NFRU_ORBIT_SPEED=1

# Save generated and reference frames in realtime mode:
.\build\Debug\samples\app\bin\debug\AMD64\vulkan_samples.exe sample nfru --app-arg NFRU_SAVE_EXR=1

# Dataset replay:
.\build\Debug\samples\app\bin\debug\AMD64\vulkan_samples.exe sample nfru --always-render --app-arg "NFRU_DATASET_DIR=path\to\dataset,NFRU_DATASET_SEQUENCE=0000"
```

**Android - hide UI:**

```bash
adb shell am start-activity \
    -n com.khronos.vulkan_samples/com.khronos.vulkan_samples.SampleLauncherActivity \
    --es cmd "sample\ nss\ --hideui"
```

**Android - custom NSS settings:**

```bash
adb shell am start-activity \
    -n com.khronos.vulkan_samples/com.khronos.vulkan_samples.SampleLauncherActivity \
    --es cmd "sample\ nss\ --app-arg\ NSS_ENABLE=1,NSS_SCALE_FACTOR=2.0,NSS_QUALITY=1"
```

**Android - NSS dataset replay:**

```bash
adb push /path/to/dataset /sdcard/Android/data/com.khronos.vulkan_samples/files/
adb shell "chmod -R 777 /sdcard/Android/data/com.khronos.vulkan_samples/files/dataset"
adb shell am start-activity -n com.khronos.vulkan_samples/com.khronos.vulkan_samples.SampleLauncherActivity --es cmd "sample\ nss\ --app-arg\ NSS_DATASET_DIR=/sdcard/Android/data/com.khronos.vulkan_samples/files/dataset,NSS_DATASET_SEQUENCE=0000,NSS_SAVE_EXR=1"

# Wait until the replay finishes
adb pull /sdcard/Android/data/com.khronos.vulkan_samples/files/dataset/result/nss_x2 /path/to/dataset/result
```

---

## NSS Sample Details

### Overview

The NSS sample renders the Sponza scene at a lower render resolution (determined by `NSS_SCALE_FACTOR`) and upscales it to the window/surface resolution using NSS. A first-person camera lets you fly through the scene to evaluate temporal upscaling quality.

The sample supports two input modes:

- **Realtime mode (default).** The sample renders the low-resolution Sponza scene and generates the NSS output each frame.
- **Dataset mode (when `NSS_DATASET_DIR` is provided).** The sample loads color, depth, motion-vector, and frame metadata from EXR and JSON files instead of rendering Sponza.

The program flow is:

1. The scene pipeline produces low-resolution **color**, **velocity** (motion vector), and **depth** images.
2. These images are passed to the **NSS dispatch** to upscale into a high-resolution output.
3. The result is passed to the post-processing pipeline to render the final image to the screen.

### UI controls - display modes

| Display mode | Description |
|-------------|-------------|
| **Final Output** | Final upscaled image produced by NSS |
| **NSS Debug** | NSS internal visualization tiles. See the [Debug view tile reference](../docs/NSS/NSS.md#debug-view-tile-reference) section.|

When **NSS Debug** is selected, a **Debug View Mode** combo box is shown:

- `0` - 4x4 tiled overview of all 16 debug tiles.
- `1`-`16` - single debug tile fullscreen.

### NSS sample UI panel

| Element | Description |
|---|---|
| Mode | Current input mode: `Realtime` or `Dataset`. |
| NSS | Whether the NSS context is enabled. |
| Path | Active execution path: `Fragment` or `Compute`. |
| Display mode | Selects `Final Output` or `NSS Debug`. |
| Debug View Mode | `Overview (4x4)` or one of the 16 debug tiles. |
| Jitter Scale | Realtime mode only. Sub-pixel jitter multiplier. |
| Current Jitter | Realtime mode only. Current frame jitter offset in pixels. |
| Frame | Dataset mode only. Current frame and total frame count. |
| Sequence | Dataset mode only. Sequence identifier. |
| Save EXR | Whether dataset output saving is enabled. |
| Output Dir | Directory where saved NSS EXR files are written. |
| History | History ownership: `Application` or `SDK` managed. |
| Scale | NSS upscale factor. |
| Quality | Active quality preset: `0` = Quality, `1` = Balanced, `2` = Performance. |
| Flags | NSS create-context flags. |
| Render | Render resolution. |
| Display | Display resolution. |
| LOD | Realtime mipmap LOD bias; dataset mode displays `N/A`. |

---


## NFRU Sample Details

The NFRU sample demonstrates how to use Neural Frame Rate Upscaling for frame generation.

### Overview

The sample supports two modes, selected automatically at startup:

**Realtime mode** (default, when `NFRU_DATASET_DIR` is not set):

1. The Sponza scene is rendered each frame, producing three attachments: **color** (R8G8B8A8_UNORM), **motion vectors** (R16G16_SFLOAT), and **depth** (D32_SFLOAT).
2. The camera rotates by `NFRU_ORBIT_SPEED` degrees per frame.
3. When `NFRU_SAVE_EXR=1`:
    - The scene is rendered with stride-2 motion vectors and frames alternate between generated and original, so each generated/original pair stays aligned.
    - Even frames call NFRU dispatch and save the generated image to `result/nfru/interpolated/` (relative to the project root). Each even frame N produces the interpolated frame N-1.
    - Odd frames skip NFRU dispatch, display the original rendered color, and save it to `result/nfru/ground_truth/` as the reference for that same odd frame.
4. When `NFRU_SAVE_EXR=0`, NFRU is dispatched every frame and the output is displayed directly.

**Dataset mode** (when `NFRU_DATASET_DIR` is provided):

1. Load dataset parameters from `<NFRU_DATASET_DIR>/<NFRU_DATASET_SEQUENCE>.json`.
2. For each step, load three EXR files from disk:
   - **Color** - tonemapped and converted to R8G8B8A8_UNORM.
   - **Depth** - loaded as R32F.
   - **Motion vectors** - converted to R16G16_SFLOAT.
3. Copy the data to GPU images, then call NFRU dispatch using the dataset camera parameters.
4. When `NFRU_SAVE_EXR=1`, interpolated frames are written to `<NFRU_DATASET_DIR>/result/nfru_x2/<NFRU_DATASET_SEQUENCE>/`.


### NFRU sample UI panel

| Element | Description |
|---|---|
| Mode | Displays `Realtime` or `Dataset`. |
| Path | Displays the selected execution path: `Fragment` or `Compute`. |
| Frame | Realtime: current frame number. Dataset: `current / total` frames. |
| Completed | Dataset mode only. Shown when all replay frames have been dispatched. |
| Frame Generated by NFRU | Shown in green when the displayed frame was produced by NFRU; yellow (`Original`) otherwise. |
| Output | Shown when EXR saving is enabled. Displays the interpolated-output directory. Realtime saving is disabled on Android. |
| GT | Shown when EXR saving is enabled in Realtime mode. Displays the ground-truth directory. Realtime saving is disabled on Android. |
| NFRU Debug View | Enables the NFRU debug visualization. |
| Orbit speed (deg/frame) | Realtime mode only. Slider (`0`-`10`) controlling `NFRU_ORBIT_SPEED`. |
