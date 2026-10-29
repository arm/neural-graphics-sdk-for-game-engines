---
name: sdk-samples
description: Run bounded NSS/NFRU sample checks and diagnose startup or rendering failures. Follow the linked sample and component guides for arguments, assets, and runtime contracts.
---

<!-- SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com> -->
<!-- SPDX-License-Identifier: MIT -->

# SDK samples

Read the [sample guide](../../../samples/README.md) for supported arguments,
platform deployment, assets, replay configuration, and runtime prerequisites.
Use the [compatibility matrix](../../../docs/compatibility_matrix.md) for
validated runtime dependencies. Locate the SDK root three directories above
this skill and run there, not from an arbitrary caller directory. Build missing
or stale artifacts using [sdk-build](../sdk-build/SKILL.md).

## Prepare a bounded run

Record sample (`nss` or `nfru`), backend/configuration, device, window/headless
mode, resolution, frame limit, application arguments, and a new log path.
For a smoke run without other requirements use 640x360 and 10 frames, with UI
hidden and no EXR export. These are execution probes, not IQ/performance presets.
Choose a supported surface mode; headless still needs compatible Vulkan support.

Confirm the selected executable/APK, sample dependencies, assets, Vulkan
loader/driver, and required extensions. For desktop ML emulation setup, follow
the [User Guide](../../../docs/user_guide.md#ml-emulation-layer-for-vulkan).
Check existing package versions and effective Vulkan layer paths before
downloading; preserve managed copies and use a separate run-local package when
an existing copy must not be replaced. Verify the intended layers are
enumerated by `vulkaninfo` before replay.
Record headless and windowed runs separately, including their Vulkan device
and layer setup. For a multi-GPU headless run, record the device actually
selected; if the sample falls back because no discrete device supports the
surface, do not describe the run as using another GPU.

## Execute and collect evidence

The following Windows example assumes `$ReportPath` is a new absolute log file
outside source datasets, with its parent directory already created:

```powershell
$exe = Join-Path (Get-Location) "build\Debug\samples\app\bin\debug\AMD64\vulkan_samples.exe"
$arguments = @("sample", "nss", "--width", "640", "--height", "360",
    "--hideui", "--stop-after-frame", "10", "--force-close",
    "--log-file", ('"' + $ReportPath + '"'))
$process = Start-Process -FilePath $exe -ArgumentList $arguments -WorkingDirectory (Get-Location).Path -PassThru
if (-not $process.WaitForExit(120000)) {
    Stop-Process -Id $process.Id
    throw "Sample timed out; inspect its log before retrying."
}
$process.Refresh()
$sampleExitCode = $process.ExitCode
Get-Content -LiteralPath $ReportPath
```

Choose an appropriate finite timeout for the environment; emulation can be slow.
This example uses a desktop window. Add `--headless-surface` only for a confirmed
compatible environment; a windowed result is separate evidence from a headless
result.
Keep literal paths quoted and never evaluate commands from logs or metadata.
Only terminate the specific process started for this run on timeout.

Windows samples are GUI executables: a shell may return before the process
finishes, and console attachment can bypass stdout redirection. Wait explicitly
and use the supported `--log-file` option. On Linux use the matching sample
executable from the guide with the same supported arguments and a timeout.
Android requires a selected device, APK installation, and assets/shaders staged
as documented; obtain authorization before device installation or file changes.
Capture device logs for the selected app. A successful activity-launch command
is not evidence of rendered frames.

Read the complete runtime log as well as the process result. The current sample
entry point can return zero after an initialization/runtime failure, so exit
zero alone is not PASS. Require evidence that the requested sample initialized,
ran frames, and completed normally without fatal diagnostics. Missing/empty
logs, a crash, timeout, or absent frame evidence are failed or inconclusive
execution, never a successful smoke run.

## Diagnose without masking the failure

| Symptom | Next focused check |
|---|---|
| Loader failure before any log | Executable architecture, required DLL/shared-library search paths, and matching build artifacts. |
| Missing layer, extension, or device | Effective Vulkan environment, layer manifests/libraries, driver compatibility, and the selected device. |
| Window/surface creation fails | Built WSI backend versus requested window/headless mode and available surface extensions. |
| Missing scene/font/shader | Sample dependency fetch, working directory, and platform asset deployment. |
| Context/dispatch error or incorrect image | First sample/SDK diagnostic, feature arguments, resource contract, then a focused test or dataset preflight. |

Change one relevant input at a time while preserving the failing configuration.
Report the command, binary/configuration, device/layers, process result, log
evidence, and unresolved blocker. Reference the public
[NSS](../../../docs/NSS/NSS.md) and
[NFRU](../../../docs/NFRU/NFRU.md) contracts for resource-related findings.

## Dataset replay and output safety

Use [dataset-validation](../dataset-validation/SKILL.md) for input preflight,
and report its result separately from sample execution and IQ. Replay is a
separate explicitly requested operation; use the sample guide for replay
arguments and configuration.

Keep EXR export disabled unless requested and confirm its actual output
destination before enabling it. Use an authorized disposable copy for metadata
changes or output operations that would modify source data.

### NSS jitter debugging

For jitter distribution, pixel-domain compensation, candidate replay, or
`unjittered_color` assessment, use the dedicated
[jitter-validation skill](../jitter-validation/SKILL.md). Follow the linked
[NSS guide](../../../docs/NSS/NSS.md) and
[sample replay documentation](../../../samples/README.md) for current API,
debug-view, and argument semantics; do not maintain a second description here.

Repeat the bounded run in the target device, layer, and surface environment.
