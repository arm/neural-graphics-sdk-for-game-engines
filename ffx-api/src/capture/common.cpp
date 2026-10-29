/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT
 */

#include "common.h"

#include <capture/format.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <locale>
#include <sstream>

static constexpr const char* ffxLogComponentName()
{
    return "Capture";
}

namespace arm::capture
{
    std::string FormatRatioDirectory(float ratio)
    {
        std::ostringstream stream;
        stream.imbue(std::locale::classic());
        stream << std::fixed << std::setprecision(6) << ratio;
        std::string ratioText = stream.str();
        std::replace(ratioText.begin(), ratioText.end(), '.', '_');
        while (!ratioText.empty() && ratioText.back() == '0')
            ratioText.pop_back();
        if (!ratioText.empty() && ratioText.back() == '_')
            ratioText.pop_back();
        return "x" + ratioText;
    }

    bool GetUpscaleRatio(uint32_t renderWidth, uint32_t renderHeight, uint32_t upscaleWidth, uint32_t upscaleHeight, float& ratio)
    {
        if (renderWidth == 0 || renderHeight == 0 || upscaleWidth == 0 || upscaleHeight == 0)
            return false;

        const float widthRatio  = static_cast<float>(upscaleWidth) / renderWidth;
        const float heightRatio = static_cast<float>(upscaleHeight) / renderHeight;
        if (!std::isfinite(widthRatio) || !std::isfinite(heightRatio) || std::abs(widthRatio - heightRatio) > 0.001f)
            return false;

        ratio = widthRatio;
        return std::isfinite(ratio) && ratio > 0.0f;
    }

    bool ConfigureCapture(Execution& execution, const ffxApiConfigureDescCapture& desc, const std::function<bool(const std::optional<Config>&, bool)>& apply)
    {
        if (desc.operation != FFX_API_CAPTURE_BEGIN && desc.operation != FFX_API_CAPTURE_CANCEL)
            return false;

        if (desc.operation == FFX_API_CAPTURE_BEGIN)
        {
            if (!desc.outputDirectory)
                return false;
            std::size_t outputDirectoryLength = 0;
            while (outputDirectoryLength <= kMaxCaptureOutputDirectoryLength && desc.outputDirectory[outputDirectoryLength] != '\0')
                ++outputDirectoryLength;
            if (outputDirectoryLength == 0 || outputDirectoryLength > kMaxCaptureOutputDirectoryLength)
                return false;

            Config config{};
            config.outputDirectory        = desc.outputDirectory;
            config.startFrame             = desc.startDispatch;
            config.maxFrameCount          = desc.frameCount;
            config.motionVectorUnormBias  = desc.motionVectorUnormBias;
            config.motionVectorUnormScale = desc.motionVectorUnormScale;
            return apply(config, false);
        }
        return apply(std::nullopt, desc.operation == FFX_API_CAPTURE_CANCEL);
    }

    void FillCaptureStatus(Execution& execution, const Config& config, FfxApiCaptureStatus& status, uint32_t capturedFrames)
    {
        const ExecutionPhase phase  = execution.GetState();
        const CaptureResult  result = execution.GetResult();
        status.phase                = ToApiCapturePhase(phase);
        status.result               = ToApiCaptureResult(result);
        status.capturedFrames       = capturedFrames;
        status.requestedFrames      = config.maxFrameCount;
    }

    bool TryBuildCapturePath(std::string& path, const std::string& outputDirectory, const char* directory, uint32_t index, const char* extension)
    {
        if (!directory || !extension)
        {
            path.clear();
            return false;
        }
        char      buffer[640];
        const int length = std::snprintf(buffer, sizeof(buffer), "%s/%s/%04u%s", outputDirectory.c_str(), directory, index, extension);
        if (length < 0 || static_cast<size_t>(length) >= sizeof(buffer))
        {
            path.clear();
            return false;
        }
        path.assign(buffer, static_cast<size_t>(length));
        return true;
    }

    int CalculateEmulatedFramerate(float frameTimeDelta, int minimum, int maximum)
    {
        if (minimum > maximum)
            std::swap(minimum, maximum);

        const float defaultFramerate = std::clamp(60.0f, static_cast<float>(minimum), static_cast<float>(maximum));
        if (!std::isfinite(frameTimeDelta) || frameTimeDelta <= 0.0f)
            return static_cast<int>(defaultFramerate);

        const float framerate = 1000.0f / frameTimeDelta;
        if (!std::isfinite(framerate))
            return framerate > 0.0f ? maximum : static_cast<int>(defaultFramerate);

        const float rounded = std::round(framerate);
        return static_cast<int>(std::clamp(rounded, static_cast<float>(minimum), static_cast<float>(maximum)));
    }

    namespace
    {
        void WriteJsonString(FILE* file, const std::string& value)
        {
            std::fputc('"', file);
            for (const unsigned char character : value)
            {
                switch (character)
                {
                case '"':
                    std::fputs("\\\"", file);
                    break;
                case '\\':
                    std::fputs("\\\\", file);
                    break;
                case '\b':
                    std::fputs("\\b", file);
                    break;
                case '\f':
                    std::fputs("\\f", file);
                    break;
                case '\n':
                    std::fputs("\\n", file);
                    break;
                case '\r':
                    std::fputs("\\r", file);
                    break;
                case '\t':
                    std::fputs("\\t", file);
                    break;
                default:
                    if (character < 0x20)
                        std::fprintf(file, "\\u%04x", character);
                    else
                        std::fputc(character, file);
                    break;
                }
            }
            std::fputc('"', file);
        }
    }  // namespace

    std::unique_ptr<Payload> CreateCapturePayload(const Config& config, const CaptureAttachmentInfo& attachment)
    {
        if (config.imageOutputEncoding == ImageOutputEncoding::Raw)
            return std::make_unique<BinaryPayload>(attachment.name, attachment.path);

        switch (attachment.kind)
        {
        case ImageKind::Color:
        case ImageKind::GroundTruth:
        case ImageKind::Output:
        case ImageKind::Exposure:
            return std::make_unique<NormalPayload>(attachment.name, attachment.path);
        case ImageKind::Depth:
            return std::make_unique<DepthPayload>(attachment.name, attachment.path);
        case ImageKind::Motion:
            return std::make_unique<ScaledPayload>(attachment.name, attachment.path, ScaleAndBias{config.motionVectorUnormBias, config.motionVectorUnormScale});
        case ImageKind::OpticalFlow:
            return std::make_unique<ScaledPayload>(attachment.name, attachment.path, ScaleAndBias{0.0f, 1.0f});
        }
        return {};
    }

    bool BuildSlotRequirements(const CaptureAttachmentPlan& plan, SlotRequirements& requirements)
    {
        requirements.attachmentBytes.clear();
        requirements.attachmentAlignments.clear();
        if (plan.attachments.empty() || plan.inputCount > plan.attachments.size())
            return false;

        requirements.attachmentBytes.reserve(plan.attachments.size());
        requirements.attachmentAlignments.reserve(plan.attachments.size());
        for (const PlannedCaptureAttachment& attachment : plan.attachments)
        {
            uint64_t       bytes     = 0;
            const uint64_t alignment = GetResourceCopyAlignment(attachment.source);
            if (!AddResourceBytes(attachment.source, bytes) || alignment == 0)
            {
                requirements.attachmentBytes.clear();
                requirements.attachmentAlignments.clear();
                return false;
            }
            requirements.attachmentBytes.push_back(bytes);
            requirements.attachmentAlignments.push_back(alignment);
        }
        return true;
    }

    bool AddCaptureAttachment(Execution& execution, const Config& config, const PlannedCaptureAttachment& attachment)
    {
        return AddCaptureAttachment(execution, config, attachment.source, attachment.info);
    }

    bool AddCaptureAttachment(Execution& execution, const Config& config, const FfxResource& source, const CaptureAttachmentInfo& attachment)
    {
        return execution.AddAttachment(source, CreateCapturePayload(config, attachment));
    }

    void WriteCaptureAttachmentJson(FILE* file, const CaptureAttachmentInfo& attachment, uint32_t format, uint32_t width, uint32_t height, bool first)
    {
        std::string jsonPath = attachment.path;
        std::replace(jsonPath.begin(), jsonPath.end(), '\\', '/');

        std::fprintf(file, "%s", first ? "" : ",");
        WriteJsonString(file, attachment.name);
        std::fputs(":{\"path\":", file);
        WriteJsonString(file, jsonPath);
        std::fputs(",\"kind\":", file);
        WriteJsonString(file, GetImageKindName(attachment.kind));
        std::fprintf(file, ",\"format\":%u,\"width\":%u,\"height\":%u}", format, width, height);
    }

}  // namespace arm::capture
