/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT
 */

#include "base.h"
#include "common.h"

#include <capture/file.h>

#include <cmath>
#include <utility>

static constexpr const char* ffxLogComponentName()
{
    return "Capture";
}

namespace arm::capture
{
    ControllerBase::ControllerBase(const char* platformPrefix, const char* component)
        : configInput_(platformPrefix, component)
        , armed_(configInput_.IsPollingEnabled())
    {
    }

    ffxReturnCode_t ControllerBase::Configure(const ffxApiConfigureDescCapture& desc)
    {
        if (desc.operation == FFX_API_CAPTURE_BEGIN)
        {
            if (execution_.GetState() != ExecutionPhase::Idle || configInput_.HasPendingApiConfig())
                return FFX_API_RETURN_ERROR_PARAMETER;
        }
        else if (desc.operation == FFX_API_CAPTURE_CANCEL)
        {
            if (execution_.GetState() != ExecutionPhase::Capturing)
                return FFX_API_RETURN_ERROR_PARAMETER;
        }

        const bool configured = ConfigureCapture(execution_, desc, [this](const std::optional<Config>& config, bool cancel) {
            if (config)
            {
                if (!configInput_.PushApiConfig(*config))
                {
                    ARM_CAPTURE_LOG_WARNING("CAPTURE", "rejected invalid API capture configuration");
                    return false;
                }
                armed_ = true;
                execution_.ResetResult();
                return true;
            }
            // CANCEL is an explicit API-mode operation. We intentionally switch the input
            // source to API here even when the caller does not identify which capture
            // request it is cancelling; the public API carries no request identity, so
            // distinguishing the first or a later cancel is not possible.
            configInput_.UseApi();
            return RequestCaptureDrain(cancel ? CaptureResult::Cancelled : CaptureResult::Success);
        });
        return configured ? FFX_API_RETURN_OK : FFX_API_RETURN_ERROR_PARAMETER;
    }

    void ControllerBase::GetStatus(FfxApiCaptureStatus& status, uint32_t capturedFrames)
    {
        TryFinalizeDrain();
        FillCaptureStatus(execution_, configInput_.GetActiveConfig(), status, capturedFrames);
    }

    bool ControllerBase::Shutdown()
    {
        OnCaptureStop();
        return execution_.DrainAndDestroyForShutdown(30000);
    }

    bool ControllerBase::StartCapture(const Config& config)
    {
        Config effectiveConfig = config;
        if (!EnsureUniqueOutputDirectory(effectiveConfig.outputDirectory))
        {
            ARM_CAPTURE_LOG_WARNING(effectiveConfig.component.c_str(), "failed to create output directory: %s", effectiveConfig.outputDirectory.c_str());
            execution_.MarkFailed();
            return false;
        }
        if (!execution_.StartCapture(effectiveConfig.component))
        {
            execution_.MarkFailed();
            return false;
        }

        configInput_.GetActiveConfig() = std::move(effectiveConfig);
        if (OnCaptureBegin(configInput_.GetActiveConfig()))
            return true;

        RequestCaptureDrain(CaptureResult::Failed);
        return false;
    }

    bool ControllerBase::RequestCaptureDrain(CaptureResult result)
    {
        if (result == CaptureResult::None)
            return false;

        OnCaptureStop();
        if (!execution_.RequestDrain(result))
            return false;
        TryFinalizeDrain();
        return true;
    }

    void ControllerBase::FailCapture(const char* reason)
    {
        ARM_CAPTURE_LOG_WARNING(configInput_.GetActiveConfig().component.c_str(), "capture draining with warning: %s", reason ? reason : "capture failed");
        RequestCaptureDrain(CaptureResult::Failed);
    }

    bool ControllerBase::TryFinalizeDrain()
    {
        if (!execution_.TryFinalizeDrain())
            return false;
        return true;
    }

}  // namespace arm::capture
