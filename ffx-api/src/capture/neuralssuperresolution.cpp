/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT
 */

#include "neuralssuperresolution.h"
#include "common.h"

#include <ffx_provider.h>

#include <capture/config.h>
#include <capture/execution.h>
#include <capture/file.h>
#include <capture/format.h>
#include <capture/meta.h>
#include <capture/log.h>
#include <capture/payload.h>
#include <capture/json.hpp>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

static constexpr const char* ffxLogComponentName()
{
    return "NssCapture";
}

namespace arm::capture::neuralssuperresolution
{
    using namespace arm::capture;
    using CaptureFrameMeta = FrameMeta;

    struct FrameMeta final : CaptureFrameMeta
    {
        uint32_t upscaleWidth    = 0;
        uint32_t upscaleHeight   = 0;
        uint32_t exposureFormat  = 0;
        uint32_t exposureWidth   = 0;
        uint32_t exposureHeight  = 0;
        float    exposure        = 1.0f;
        float    upscaleRatio    = 1.0f;
        bool     exposureTexture = false;
    };

    class NssDatasetLayout
    {
    public:
        static std::string Metadata(const Config& config)
        {
            return config.outputDirectory + "/0000.json";
        }
        static bool GetCaptureInfo(const Config& config, float upscaleRatio, ImageKind kind, uint32_t index, CaptureAttachmentInfo& info)
        {
            const std::string ratioDirectory = FormatRatioDirectory(upscaleRatio);
            switch (kind)
            {
            case ImageKind::Color:
                info = {"color", kind, {}};
                return Image(config, ratioDirectory + "/color/0000", index, info.path);
            case ImageKind::Depth:
                info = {"depth", kind, {}};
                return Image(config, ratioDirectory + "/depth/0000", index, info.path);
            case ImageKind::Motion:
                info = {"motion", kind, {}};
                return Image(config, ratioDirectory + "/motion/0000", index, info.path);
            case ImageKind::Output:
                info = {"output", kind, {}};
                return Image(config, kCaptureResultsDirectory, index, info.path);
            case ImageKind::Exposure:
                info = {"exposure", kind, {}};
                return Image(config, "exposure/0000", index, info.path);
            case ImageKind::GroundTruth:
            case ImageKind::OpticalFlow:
                break;
            }
            info = {};
            return false;
        }

    private:
        static bool Image(const Config& config, const std::string& directory, uint32_t index, std::string& path)
        {
            const char* extension = config.imageOutputEncoding == ImageOutputEncoding::Raw ? ".bin" : ".exr";
            return TryBuildCapturePath(path, config.outputDirectory, directory.c_str(), index, extension);
        }
    };

    class NssMetadataWriter final : public JsonWriter
    {
    public:
        bool Open(const std::string& path, const FrameMeta& first, const CaptureHeaderSettings& settings)
        {
            if (!JsonWriter::Open(path))
            {
                ARM_CAPTURE_LOG_WARNING("NSS-CAPTURE", "failed to open metadata file: %s", path.c_str());
                return false;
            }

            const int emulatedFps = CalculateEmulatedFramerate(first.frameTimeDelta, 1, 240);
            Writef(
                "{\n  \"NearPlane\": %.8g,\n  \"FarPlane\": %.8g,\n  \"ReverseZ\": %s,\n  \"DepthInfinite\": %s,\n  "
                "\"MotionVectorsDilated\": false,\n  \"MotionVectorsJitterCancellation\": %s,\n  \"HighDynamicRange\": %s,\n  "
                "\"MotionVectorUnormBias\": %.16g,\n  \"MotionVectorUnormScale\": %.16g,\n  "
                "\"EmulatedFramerate\": %d,\n",
                first.cameraNear,
                first.cameraFar,
                (settings.contextFlags & FFX_API_NSS_CONTEXT_FLAG_DEPTH_INVERTED) != 0 ? "true" : "false",
                (settings.contextFlags & FFX_API_NSS_CONTEXT_FLAG_DEPTH_INFINITE) != 0 ? "true" : "false",
                (settings.contextFlags & FFX_API_NSS_CONTEXT_FLAG_ENABLE_MOTION_VECTORS_JITTER_CANCELLATION) != 0 ? "true" : "false",
                (settings.contextFlags & FFX_API_NSS_CONTEXT_FLAG_LOW_DYNAMIC_RANGE) == 0 ? "true" : "false",
                settings.motionVectorUnormBias,
                settings.motionVectorUnormScale,
                emulatedFps);
            Writef("  \"SDKVersion\": \"%s\",\n", GetCaptureBuildId().c_str());
            Writef("  \"TargetResolution\": { \"X\": %u, \"Y\": %u },\n", first.upscaleWidth, first.upscaleHeight);
            const std::string ratioDirectory = FormatRatioDirectory(first.upscaleRatio);
            Writef(
                "  \"Streams\": {\n"
                "    \"color\": { \"directory\": \"%s/color/0000\", \"kind\": \"color\", \"format\": %u, \"width\": %u, \"height\": %u },\n"
                "    \"depth\": { \"directory\": \"%s/depth/0000\", \"kind\": \"depth\", \"format\": %u, \"width\": %u, \"height\": %u },\n"
                "    \"motion\": { \"directory\": \"%s/motion/0000\", \"kind\": \"motion\", \"format\": %u, \"width\": %u, \"height\": %u },\n"
                "    \"output\": { \"directory\": \"%s\", \"kind\": \"output\", \"format\": %u, \"width\": %u, \"height\": %u }%s\n",
                ratioDirectory.c_str(),
                first.colorFormat,
                first.renderWidth,
                first.renderHeight,
                ratioDirectory.c_str(),
                first.depthFormat,
                first.renderWidth,
                first.renderHeight,
                ratioDirectory.c_str(),
                first.motionFormat,
                first.renderWidth,
                first.renderHeight,
                kCaptureResultsDirectory,
                first.outputFormat,
                first.upscaleWidth,
                first.upscaleHeight,
                first.exposureTexture ? ",\n    \"exposure\": { \"directory\": \"exposure/0000\", \"kind\": \"exposure\", \"format\": " : "");
            if (first.exposureTexture)
                Writef("%u, \"width\": %u, \"height\": %u }", first.exposureFormat, first.exposureWidth, first.exposureHeight);
            Writef(
                "\n  },\n"
                "  \"UpscalingRatiosIndices\": { \"%s_index\": 0 },\n"
                "  \"Samples\": { \"Sequence\": \"TiledHalton\", \"Count\": ",
                ratioDirectory.c_str());
            SetCountOffset();
            Writef("%10u, \"Quantized\": true },\n  \"Frames\": [\n", 0u);
            return !HasError();
        }

        bool WriteMetadata(const FrameMeta& frame)
        {
            const float aspect  = frame.renderHeight ? static_cast<float>(frame.renderWidth) / frame.renderHeight : 1.0f;
            const float fovX    = 2.0f * std::atan(std::tan(frame.cameraFovAngleVertical * 0.5f) * aspect);
            const float jitterX = frame.renderWidth ? frame.jitterX / frame.renderWidth : 0.0f;
            const float jitterY = frame.renderHeight ? frame.jitterY / frame.renderHeight : 0.0f;
            BeginEntry();
            Writef(
                "    { \"Frame\": %u, \"Reset\": %s, \"FrameTimeDelta\": %.9g, \"FovX\": %.16g, \"FovY\": %.16g, \"CameraNearPlane\": %.8g, "
                "\"CameraFarPlane\": %.8g, "
                "\"Exposure\": %.9g, \"MotionVectorScale\": [ %.16g, %.16g ], \"QuantisedPerRatioJitter\": [ { \"X\": %.16g, \"Y\": %.16g } ] }",
                frame.datasetIndex,
                frame.reset ? "true" : "false",
                frame.frameTimeDelta,
                fovX,
                frame.cameraFovAngleVertical,
                frame.cameraNear,
                frame.cameraFar,
                frame.exposure,
                frame.motionVectorScaleX,
                frame.motionVectorScaleY,
                jitterX,
                jitterY);
            CommitEntry();
            return !HasError();
        }

        bool Close()
        {
            if (!IsOpen())
                return true;
            Writef("\n  ]\n}\n");
            return BackfillCountAndClose();
        }
    };

    bool BuildNssAttachmentPlan(const Config&                config,
                                const ffxApiDispatchDescNss& dispatch,
                                float                        upscaleRatio,
                                uint32_t                     index,
                                bool                         includeExposureTexture,
                                CaptureAttachmentPlan&       plan)
    {
        plan.Clear();
        auto append = [&plan](const FfxResource& source, CaptureAttachmentInfo info) { plan.attachments.push_back({source, std::move(info)}); };

        CaptureAttachmentInfo colorInfo;
        CaptureAttachmentInfo depthInfo;
        CaptureAttachmentInfo motionInfo;
        CaptureAttachmentInfo exposureInfo;
        CaptureAttachmentInfo outputInfo;
        if (!NssDatasetLayout::GetCaptureInfo(config, upscaleRatio, ImageKind::Color, index, colorInfo) ||
            !NssDatasetLayout::GetCaptureInfo(config, upscaleRatio, ImageKind::Depth, index, depthInfo) ||
            !NssDatasetLayout::GetCaptureInfo(config, upscaleRatio, ImageKind::Motion, index, motionInfo) ||
            !NssDatasetLayout::GetCaptureInfo(config, {}, ImageKind::Output, index, outputInfo))
        {
            return false;
        }
        append(Convert(dispatch.color), std::move(colorInfo));
        append(Convert(dispatch.depth), std::move(depthInfo));
        append(Convert(dispatch.motionVectors), std::move(motionInfo));
        if (includeExposureTexture)
        {
            if (!NssDatasetLayout::GetCaptureInfo(config, upscaleRatio, ImageKind::Exposure, index, exposureInfo))
                return false;
            append(Convert(dispatch.exposureTexture), std::move(exposureInfo));
        }
        plan.inputCount = plan.attachments.size();
        append(Convert(dispatch.output), std::move(outputInfo));
        return true;
    }

    struct Controller::Recorder final : RecorderBase<Controller::Recorder, FrameMeta, NssMetadataWriter>
    {
        CaptureAttachmentPlan pendingPlan;

        explicit Recorder(uint32_t flags)
            : RecorderBase(flags)
        {
        }

        bool OpenWriters(const FrameMeta& frame)
        {
            // Capture validates producer metadata and fails closed before opening output files.
            const CaptureHeaderSettings headerSettings{contextFlags, config.motionVectorUnormBias, config.motionVectorUnormScale};
            if (!metadata.IsOpen() && !metadata.Open(NssDatasetLayout::Metadata(config), frame, headerSettings))
                return false;
            return true;
        }

        bool WriteMetadata(const FrameMeta& frame)
        {
            return metadata.WriteMetadata(frame);
        }

        void ResetExtra()
        {
            pendingPlan.Clear();
        }

        bool CloseWriters()
        {
            if (metadata.IsOpen())
            {
                const uint32_t count  = metadata.GetWrittenEntryCount();
                const bool     result = metadata.Close();
                ARM_CAPTURE_LOG_INFO("NSS-CAPTURE", "written metadata for %u frames", count);
                return result;
            }
            return true;
        }

        void OnFrameCommitted()
        {
        }
    };

    Controller::Controller(void* backendDevice, uint32_t contextFlags)
        : ControllerBase(kNssPlatformPrefix, "NSS-CAPTURE")
        , backendDevice_(backendDevice)
        , recorder_(std::make_shared<Recorder>(contextFlags))
    {
        ARM_CAPTURE_LOG_INFO("NSS-CAPTURE", "ready for capture API or environment source");
    }

    Controller::~Controller()
    {
        Shutdown();
    }

    bool Controller::OnCaptureBegin(const Config& config)
    {
        recorder_->Reset();
        recorder_->config = config;
        ARM_CAPTURE_LOG_INFO(
            "NSS-CAPTURE", "armed output=%s startFrame=%u frameCount=%u", config.outputDirectory.c_str(), config.startFrame, config.maxFrameCount);
        return true;
    }

    void Controller::Stop(const char* reason)
    {
        FailCapture(reason);
    }

    void Controller::GetStatus(FfxApiCaptureStatus& status)
    {
        ControllerBase::GetStatus(status, recorder_->GetCommittedFrameCount());
    }

    bool Controller::ShouldResetHistoryForDispatch() const
    {
        return execution_.GetState() == ExecutionPhase::Capturing && recorder_->GetCommittedFrameCount() == 0;
    }

    bool Controller::BeginDispatchCapture(const ffxApiDispatchDescNss& dispatch, FfxNssContext& context)
    {
        Recorder&  recorder   = *recorder_;
        const bool frameReady = AdvanceFrameThenCheck();
        if (!frameReady && execution_.GetState() != ExecutionPhase::Capturing)
            return false;

        if (execution_.GetState() != ExecutionPhase::Capturing)
            return false;

        const Config& config       = configInput_.GetActiveConfig();
        float         upscaleRatio = 1.0f;
        if (!GetUpscaleRatio(dispatch.renderSize.width, dispatch.renderSize.height, dispatch.upscaleSize.width, dispatch.upscaleSize.height, upscaleRatio))
        {
            Stop("invalid NSS render/upscale ratio");
            return false;
        }

        CaptureAttachmentPlan plan;
        const uint32_t        datasetIndex = recorder.GetCommittedFrameCount();
        if (!BuildNssAttachmentPlan(
                config, dispatch, upscaleRatio, datasetIndex, (recorder.contextFlags & FFX_API_NSS_CONTEXT_FLAG_EXPOSURE_TEXTURE) != 0, plan))
        {
            Stop("failed to construct capture attachment plan");
            return false;
        }
        SlotRequirements requirements;
        if (!BuildSlotRequirements(plan, requirements))
        {
            Stop("failed to determine capture slot size");
            return false;
        }
        SessionDescriptor sessionDesc{backendDevice_, requirements, recorder_};
        if (!execution_.EnsureSession(std::move(sessionDesc)))
        {
            if (!frameReady)
                Stop("failed to create warm-up capture session");
            else
                Stop("failed to create capture session");
            return false;
        }
        if (!frameReady)
            return false;

        auto       frameMeta        = std::make_unique<FrameMeta>();
        FrameMeta& meta             = *frameMeta;
        meta.datasetIndex           = datasetIndex;
        meta.jitterX                = dispatch.jitterOffset.x;
        meta.jitterY                = dispatch.jitterOffset.y;
        meta.cameraNear             = dispatch.cameraNear;
        meta.cameraFar              = dispatch.cameraFar;
        meta.cameraFovAngleVertical = dispatch.cameraFovAngleVertical;
        meta.motionVectorScaleX     = dispatch.motionVectorScale.x;
        meta.motionVectorScaleY     = dispatch.motionVectorScale.y;
        const bool exposureTexture  = (recorder.contextFlags & FFX_API_NSS_CONTEXT_FLAG_EXPOSURE_TEXTURE) != 0;
        meta.exposure               = exposureTexture ? 1.0f : dispatch.exposure;
        meta.exposureTexture        = exposureTexture;
        meta.frameTimeDelta         = dispatch.frameTimeDelta;
        meta.renderWidth            = dispatch.renderSize.width;
        meta.renderHeight           = dispatch.renderSize.height;
        meta.upscaleWidth           = dispatch.upscaleSize.width;
        meta.upscaleHeight          = dispatch.upscaleSize.height;
        meta.upscaleRatio           = upscaleRatio;
        meta.colorFormat            = static_cast<uint32_t>(dispatch.color.description.format);
        meta.depthFormat            = static_cast<uint32_t>(dispatch.depth.description.format);
        meta.motionFormat           = static_cast<uint32_t>(dispatch.motionVectors.description.format);
        meta.outputFormat           = static_cast<uint32_t>(dispatch.output.description.format);
        meta.reset                  = dispatch.reset || ShouldResetHistoryForDispatch();
        meta.exposureFormat         = static_cast<uint32_t>(dispatch.exposureTexture.description.format);
        meta.exposureWidth          = dispatch.exposureTexture.description.width;
        meta.exposureHeight         = dispatch.exposureTexture.description.height;
        recorder.pendingPlan        = std::move(plan);

        if (!execution_.BeginTransaction(requirements, dispatch.commandList, std::move(frameMeta)))
        {
            recorder.pendingPlan.Clear();
            Stop("capture acquisition unavailable: staging-memory budget or outstanding-slot cap reached");
            return false;
        }

        bool attached = true;
        for (size_t attachmentIndex = 0; attachmentIndex < recorder.pendingPlan.inputCount; ++attachmentIndex)
            attached = attached && AddCaptureAttachment(execution_, config, recorder.pendingPlan.attachments[attachmentIndex]);

        if (attached)
            return true;

        execution_.CancelTransaction();
        recorder.pendingPlan.Clear();
        Stop("failed to record input attachments");
        return false;
    }

    void Controller::AbortDispatchCapture(const char* reason)
    {
        execution_.DiscardTransaction();
        recorder_->pendingPlan.Clear();
        Stop(reason);
    }

    bool Controller::CompleteDispatchCapture(const ffxApiDispatchDescNss& dispatch, FfxNssContext&)
    {
        const std::optional<uint32_t> index = execution_.GetTransactionDatasetIndex();
        if (!index)
            return false;
        if (recorder_->pendingPlan.inputCount >= recorder_->pendingPlan.attachments.size())
        {
            AbortDispatchCapture("invalid capture attachment plan");
            return false;
        }

        Recorder& recorder                                                       = *recorder_;
        recorder.pendingPlan.attachments[recorder.pendingPlan.inputCount].source = Convert(dispatch.output);
        const Config&                   config                                   = configInput_.GetActiveConfig();
        const PlannedCaptureAttachment& outputAttachment                         = recorder.pendingPlan.attachments[recorder.pendingPlan.inputCount];
        const bool                      attached                                 = AddCaptureAttachment(execution_, config, outputAttachment);
        if (!attached || !execution_.CommitTransaction())
        {
            execution_.CancelTransaction();
            recorder.pendingPlan.Clear();
            Stop("failed to record or submit output attachment");
            return false;
        }
        recorder.pendingPlan.Clear();

        recorder.CommitFrame();
        ARM_CAPTURE_LOG_INFO("NSS-CAPTURE", "recorded frame %u/%u", recorder.GetCommittedFrameCount(), configInput_.GetActiveConfig().maxFrameCount);
        if (recorder.GetCommittedFrameCount() >= configInput_.GetActiveConfig().maxFrameCount)
            RequestCaptureDrain(CaptureResult::Success);
        return true;
    }

}  // namespace arm::capture::neuralssuperresolution
