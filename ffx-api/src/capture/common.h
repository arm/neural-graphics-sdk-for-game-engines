/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <capture/config.h>
#include <capture/payload.h>
#include <capture/execution.h>
#include <ffx_api/ffx_api.h>
#include <ffx_provider.h>

#include <cstdint>
#include <cstdio>
#include <memory>
#include <optional>
#include <string>
#include <functional>
#include <vector>

namespace arm::capture
{
    inline constexpr const char* kCaptureResultsDirectory = "results/NGSDK";

    inline FfxApiCapturePhase ToApiCapturePhase(ExecutionPhase state)
    {
        switch (state)
        {
        case ExecutionPhase::Capturing:
            return FFX_API_CAPTURE_PHASE_CAPTURING;
        case ExecutionPhase::Draining:
            return FFX_API_CAPTURE_PHASE_DRAINING;
        case ExecutionPhase::Idle:
        default:
            return FFX_API_CAPTURE_PHASE_IDLE;
        }
    }

    inline FfxApiCaptureResult ToApiCaptureResult(CaptureResult result)
    {
        switch (result)
        {
        case CaptureResult::Success:
            return FFX_API_CAPTURE_RESULT_SUCCESS;
        case CaptureResult::Failed:
            return FFX_API_CAPTURE_RESULT_FAILED;
        case CaptureResult::Cancelled:
            return FFX_API_CAPTURE_RESULT_CANCELLED;
        case CaptureResult::None:
        default:
            return FFX_API_CAPTURE_RESULT_NONE;
        }
    }

    bool ConfigureCapture(Execution& execution, const ffxApiConfigureDescCapture& desc, const std::function<bool(const std::optional<Config>&, bool)>& apply);

    void FillCaptureStatus(Execution& execution, const Config& config, FfxApiCaptureStatus& status, uint32_t capturedFrames);

    bool TryBuildCapturePath(std::string& path, const std::string& outputDirectory, const char* directory, uint32_t index, const char* extension);

    std::string FormatRatioDirectory(float ratio);

    bool GetUpscaleRatio(uint32_t renderWidth, uint32_t renderHeight, uint32_t upscaleWidth, uint32_t upscaleHeight, float& ratio);

    int CalculateEmulatedFramerate(float frameTimeDelta, int minimum, int maximum);

    struct CaptureAttachmentInfo
    {
        const char* name = nullptr;
        ImageKind   kind = ImageKind::Color;
        std::string path;
    };

    struct PlannedCaptureAttachment
    {
        FfxResource           source{};
        CaptureAttachmentInfo info;
    };

    struct CaptureAttachmentPlan
    {
        std::vector<PlannedCaptureAttachment> attachments;
        // This plan covers only the explicit producer attachments. Internal
        // temporal state is not represented; supporting it requires extending
        // the plan, its byte/alignment requirements, and the Begin/Complete
        // attachment sequence together.
        size_t inputCount = 0;

        void Clear()
        {
            attachments.clear();
            inputCount = 0;
        }
    };

    struct CaptureHeaderSettings
    {
        uint32_t contextFlags           = 0;
        float    motionVectorUnormBias  = 0.5f;
        float    motionVectorUnormScale = 2.0f;
    };

    std::unique_ptr<Payload> CreateCapturePayload(const Config& config, const CaptureAttachmentInfo& attachment);

    bool BuildSlotRequirements(const CaptureAttachmentPlan& plan, SlotRequirements& requirements);

    bool AddCaptureAttachment(Execution& execution, const Config& config, const PlannedCaptureAttachment& attachment);

    bool AddCaptureAttachment(Execution& execution, const Config& config, const FfxResource& source, const CaptureAttachmentInfo& attachment);

    void WriteCaptureAttachmentJson(FILE* file, const CaptureAttachmentInfo& attachment, uint32_t format, uint32_t width, uint32_t height, bool first);

}  // namespace arm::capture
