/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT
 */

#include "framegenerations.h"
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

#include <cstdio>
#include <cstring>
#include <cmath>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

static constexpr const char* ffxLogComponentName()
{
    return "FrameGenerationCapture";
}

namespace arm::capture::framegenerations
{
    using namespace arm::capture;
    using CaptureFrameMeta = FrameMeta;
    namespace
    {
        struct FrameMeta final : CaptureFrameMeta
        {
            uint32_t displayWidth      = 0;
            uint32_t displayHeight     = 0;
            float    upscaleRatio      = 1.0f;
            uint32_t opticalFlowFormat = 0;
            uint32_t opticalFlowWidth  = 0;
            uint32_t opticalFlowHeight = 0;
        };

        class NfruDatasetLayout
        {
        public:
            static std::string Metadata(const Config& config)
            {
                return config.outputDirectory + "/0000.json";
            }
            static bool GetCaptureInfo(const Config& config, float upscaleRatio, ImageKind kind, uint32_t index, CaptureAttachmentInfo& info)
            {
                uint32_t inputFrame     = 0;
                uint32_t generatedFrame = 0;
                if (!GetFrameNumbers(index, inputFrame, generatedFrame))
                    return false;
                const std::string ratioDirectory = FormatRatioDirectory(upscaleRatio);
                switch (kind)
                {
                case ImageKind::GroundTruth:
                    info = {"groundTruth", kind, {}};
                    return Image(config, "ground_truth/0000", inputFrame, info.path);
                case ImageKind::Depth:
                    info = {"depth", kind, {}};
                    return Image(config, ratioDirectory + "/depth/0000", inputFrame, info.path);
                case ImageKind::Motion:
                    info = {"motion", kind, {}};
                    return Image(config, ratioDirectory + "/motion_m2/0000", inputFrame, info.path);
                case ImageKind::Output:
                    info = {"output", kind, {}};
                    return Image(config, kCaptureResultsDirectory, generatedFrame, info.path);
                case ImageKind::OpticalFlow:
                    info = {"opticalFlow", kind, {}};
                    return Image(config, "optical_flow_p2/0000", generatedFrame, info.path);
                case ImageKind::Exposure:
                case ImageKind::Color:
                    break;
                }
                info = {};
                return false;
            }
            static bool GetCaptureInfo(const Config& config, ImageKind kind, uint32_t index, CaptureAttachmentInfo& info)
            {
                return GetCaptureInfo(config, 1.0f, kind, index, info);
            }

        private:
            static bool GetFrameNumbers(uint32_t index, uint32_t& input, uint32_t& generated)
            {
                const uint64_t inputFrame = static_cast<uint64_t>(index) * 2;
                if (inputFrame > std::numeric_limits<uint32_t>::max())
                    return false;
                input     = static_cast<uint32_t>(inputFrame);
                generated = input == 0 ? 0 : input - 1;
                return true;
            }
            static bool Image(const Config& config, const std::string& directory, uint32_t number, std::string& path)
            {
                const char* extension = config.imageOutputEncoding == ImageOutputEncoding::Raw ? ".bin" : ".exr";
                return TryBuildCapturePath(path, config.outputDirectory, directory.c_str(), number, extension);
            }
        };

        class NfruMetadataWriter final : public JsonWriter
        {
        public:
            bool Open(const std::string& path, const FrameMeta& first, const CaptureHeaderSettings& settings)
            {
                if (!JsonWriter::Open(path))
                {
                    ARM_CAPTURE_LOG_WARNING("NFRU-CAPTURE", "failed to open metadata file: %s", path.c_str());
                    return false;
                }

                const int         emulatedFps    = CalculateEmulatedFramerate(first.frameTimeDelta, 30, 90);
                const std::string ratioDirectory = FormatRatioDirectory(first.upscaleRatio);
                Writef(
                    "{\n\t\"Version\": \"1.0.1\",\n\t\"MotionVectorsDilated\": false,\n\t\"UnjitteredSrcColor_Exported\": "
                    "false,\n\t\"OpaqueOnlyColor_Exported\": false,\n");
                Writef("\t\"SDKVersion\": \"%s\",\n", GetCaptureBuildId().c_str());
                Writef(
                    "\t\"EmulatedFramerate\": %d,\n\t\"GTColorSpace\": \"Linear\",\n\t\"ReverseZ\": %s,\n\t\"DepthInfinite\": %s,\n"
                    "\t\"MotionVectorsJitterCancellation\": %s,\n\t\"DisplayResolutionMotionVectors\": %s,\n"
                    "\t\"MotionVectorUnormBias\": %.16g,\n\t\"MotionVectorUnormScale\": %.16g,\n",
                    emulatedFps,
                    (settings.contextFlags & FFX_API_FG_CONTEXT_FLAG_ENABLE_DEPTH_INVERTED) != 0 ? "true" : "false",
                    (settings.contextFlags & FFX_API_FG_CONTEXT_FLAG_ENABLE_DEPTH_INFINITE) != 0 ? "true" : "false",
                    (settings.contextFlags & FFX_API_FG_CONTEXT_FLAG_ENABLE_MOTION_VECTORS_JITTER_CANCELLATION) != 0 ? "true" : "false",
                    (settings.contextFlags & FFX_API_FG_CONTEXT_FLAG_ENABLE_DISPLAY_RESOLUTION_MOTION_VECTORS) != 0 ? "true" : "false",
                    settings.motionVectorUnormBias,
                    settings.motionVectorUnormScale);
                Writef("\t\"ViewProjectionColumnMajor\": true,\n");
                Writef("\t\"TargetResolution\":\n\t{\n\t\t\"X\": %u,\n\t\t\"Y\": %u\n\t},\n", first.displayWidth, first.displayHeight);
                Writef(
                    "\t\"Streams\":\n\t{\n"
                    "\t\t\"groundTruth\": { \"directory\": \"ground_truth/0000\", \"kind\": \"groundTruth\", \"format\": %u, \"width\": %u, \"height\": %u },\n"
                    "\t\t\"depth\": { \"directory\": \"%s/depth/0000\", \"kind\": \"depth\", \"format\": %u, \"width\": %u, \"height\": %u },\n"
                    "\t\t\"motion\": { \"directory\": \"%s/motion_m2/0000\", \"kind\": \"motion\", \"format\": %u, \"width\": %u, \"height\": %u },\n"
                    "\t\t\"output\": { \"directory\": \"%s\", \"kind\": \"output\", \"format\": %u, \"width\": %u, \"height\": %u },\n"
                    "\t\t\"opticalFlow\": { \"directory\": \"optical_flow_p2/0000\", \"kind\": \"opticalFlow\", \"format\": %u, \"width\": %u, \"height\": %u "
                    "}\n"
                    "\t},\n",
                    first.colorFormat,
                    first.displayWidth,
                    first.displayHeight,
                    ratioDirectory.c_str(),
                    first.depthFormat,
                    first.renderWidth,
                    first.renderHeight,
                    ratioDirectory.c_str(),
                    first.motionFormat,
                    first.motionWidth,
                    first.motionHeight,
                    kCaptureResultsDirectory,
                    first.outputFormat,
                    first.displayWidth,
                    first.displayHeight,
                    first.opticalFlowFormat,
                    first.opticalFlowWidth,
                    first.opticalFlowHeight);
                Writef("\t\"UpscalingRatiosIndices\":\n\t{\n\t\t\"%s_index\": 0\n\t},\n", ratioDirectory.c_str());
                Writef("\t\"Samples\":\n\t{\n\t\t\"Sequence\": \"TiledHalton\",\n\t\t\"Count\": ");
                SetCountOffset();
                Writef("%10u,\n\t\t\"Quantized\": true\n\t},\n\t\"Frames\": [\n", 0u);
                return !HasError();
            }

            bool WriteFrame(const FrameMeta& frame, uint64_t frameNumber)
            {
                const float aspect  = frame.renderHeight ? static_cast<float>(frame.renderWidth) / frame.renderHeight : 1.0f;
                const float fovX    = 2.0f * atanf(tanf(frame.cameraFovAngleVertical * 0.5f) * aspect);
                const float jitterX = frame.renderWidth ? frame.jitterX / frame.renderWidth : 0.0f;
                const float jitterY = frame.renderHeight ? frame.jitterY / frame.renderHeight : 0.0f;
                BeginEntry();
                Writef(
                    "\t\t{\n\t\t\t\"Frame\": %llu,\n\t\t\t\"Reset\": %s,\n\t\t\t\"FrameTimeDelta\": %.9g,\n\t\t\t\"FovX\": %.16g,\n\t\t\t\"FovY\": "
                    "%.16g,\n\t\t\t\"CameraNearPlane\": "
                    "%.8g,\n\t\t\t\"CameraFarPlane\": %.8g,\n",
                    static_cast<unsigned long long>(frameNumber),
                    frame.reset ? "true" : "false",
                    frame.frameTimeDelta,
                    fovX,
                    frame.cameraFovAngleVertical,
                    frame.cameraNear,
                    frame.cameraFar);
                Writef("\t\t\t\"MotionVectorScale\": [ %.16g, %.16g ],\n\t\t\t\"ViewProjection\": [ ", frame.motionVectorScaleX, frame.motionVectorScaleY);
                for (int index = 0; index < 16; ++index)
                    Writef("%.16g%s", frame.viewProjection[index], index < 15 ? ", " : " ],\n");
                Writef(
                    "\t\t\t\"ViewSpaceToMetersFactor\": %.16g,\n\t\t\t\"Exposure\": 1.0,\n\t\t\t\"Jitter\":\n\t\t\t{\n\t\t\t\t\"X\": %.16g,\n\t\t\t\t\"Y\": "
                    "%.16g\n\t\t\t},\n\t\t\t\"NormalizedPerRatioJitter\": [\n",
                    frame.viewSpaceToMetersFactor,
                    frame.jitterX,
                    frame.jitterY);
                for (int ratio = 0; ratio < 4; ++ratio)
                    Writef("\t\t\t\t{\n\t\t\t\t\t\"X\": %.16g,\n\t\t\t\t\t\"Y\": %.16g\n\t\t\t\t}%s\n", jitterX, jitterY, ratio < 3 ? "," : "");
                Writef("\t\t\t]\n\t\t}");
                CommitEntry();
                return !HasError();
            }

            bool Close()
            {
                if (!IsOpen())
                    return true;
                Writef("\n\t]\n}\n");
                return BackfillCountAndClose();
            }
        };

        struct SelectedResources
        {
            FfxResource opticalFlow{};
        };

        FfxResource GetSharedResource(const Context& context, uint32_t identifier)
        {
            return context.backendInterface->fpGetResource(context.backendInterface, context.sharedResources[identifier]);
        }

        SelectedResources SelectResources(const ffxApiDispatchDescFrameGenerationPrepare&, const Context& context)
        {
            SelectedResources selected{};
            selected.opticalFlow = GetSharedResource(context, FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_OPTICAL_FLOW_VECTOR);
            return selected;
        }

        bool BuildNfruAttachmentPlan(const Config&                                   config,
                                     const ffxApiDispatchDescFrameGeneration&        dispatch,
                                     const ffxApiDispatchDescFrameGenerationPrepare& prepare,
                                     const SelectedResources&                        selected,
                                     float                                           upscaleRatio,
                                     uint32_t                                        index,
                                     CaptureAttachmentPlan&                          plan)
        {
            plan.Clear();
            auto append = [&plan](const FfxResource& source, CaptureAttachmentInfo info) { plan.attachments.push_back({source, std::move(info)}); };

            CaptureAttachmentInfo groundTruthInfo;
            CaptureAttachmentInfo depthInfo;
            CaptureAttachmentInfo motionInfo;
            if (!NfruDatasetLayout::GetCaptureInfo(config, upscaleRatio, ImageKind::GroundTruth, index, groundTruthInfo) ||
                !NfruDatasetLayout::GetCaptureInfo(config, upscaleRatio, ImageKind::Depth, index, depthInfo) ||
                !NfruDatasetLayout::GetCaptureInfo(config, upscaleRatio, ImageKind::Motion, index, motionInfo))
            {
                return false;
            }
            append(Convert(dispatch.presentColor), std::move(groundTruthInfo));
            append(Convert(prepare.depth), std::move(depthInfo));
            append(Convert(prepare.motionVectors), std::move(motionInfo));
            plan.inputCount = plan.attachments.size();

            if (index > 0)
            {
                CaptureAttachmentInfo outputInfo;
                CaptureAttachmentInfo opticalFlowInfo;
                if (!NfruDatasetLayout::GetCaptureInfo(config, ImageKind::Output, index, outputInfo) ||
                    !NfruDatasetLayout::GetCaptureInfo(config, ImageKind::OpticalFlow, index, opticalFlowInfo))
                {
                    return false;
                }
                append(Convert(dispatch.outputs[0]), std::move(outputInfo));
                append(selected.opticalFlow, std::move(opticalFlowInfo));
            }
            return true;
        }

    }  // namespace

    struct Controller::Recorder final : RecorderBase<Controller::Recorder, FrameMeta, NfruMetadataWriter>
    {
        bool                  preparePending = false;
        CaptureAttachmentPlan pendingPlan;

        explicit Recorder(uint32_t flags)
            : RecorderBase(flags)
        {
        }

        bool OpenWriters(const FrameMeta& frame)
        {
            const CaptureHeaderSettings headerSettings{contextFlags, config.motionVectorUnormBias, config.motionVectorUnormScale};
            if (!metadata.IsOpen() && !metadata.Open(NfruDatasetLayout::Metadata(config), frame, headerSettings))
                return false;
            return true;
        }

        bool WriteMetadata(const FrameMeta& frame)
        {
            const uint64_t inputFrame = static_cast<uint64_t>(frame.datasetIndex) * 2;
            if (frame.datasetIndex > 0 && !metadata.WriteFrame(frame, inputFrame - 1))
                return false;
            if (!metadata.WriteFrame(frame, inputFrame))
                return false;
            return true;
        }

        bool CloseWriters()
        {
            if (metadata.IsOpen())
            {
                const uint32_t count  = metadata.GetWrittenEntryCount();
                const bool     result = metadata.Close();
                ARM_CAPTURE_LOG_INFO("NFRU-CAPTURE", "written metadata for %u frames", count);
                return result;
            }
            return true;
        }

        void ResetExtra()
        {
            preparePending = false;
            pendingPlan.Clear();
        }

        void OnFrameCommitted()
        {
            preparePending = false;
        }
    };

    Controller::Controller(const CreateDesc& desc, const Context& context)
        : ControllerBase(kNfruPlatformPrefix, "NFRU-CAPTURE")
        , backendDevice_(desc.backendDevice)
        , context_(context)
        , recorder_(std::make_shared<Recorder>(desc.contextFlags))
    {
        ARM_CAPTURE_LOG_INFO("NFRU-CAPTURE", "ready for capture API or environment source");
    }

    Controller::~Controller()
    {
        Shutdown();
    }
    bool Controller::IsEnabled() const
    {
        return execution_.GetState() == ExecutionPhase::Capturing;
    }
    bool Controller::IsPreparePending() const
    {
        return IsEnabled() && recorder_->preparePending;
    }

    bool Controller::ShouldResetHistoryForDispatch() const
    {
        return IsEnabled() && recorder_->GetCommittedFrameCount() == 0;
    }

    bool Controller::OnCaptureBegin(const Config& config)
    {
        recorder_->Reset();
        recorder_->config = config;
        ARM_CAPTURE_LOG_INFO(
            "NFRU-CAPTURE", "armed output=%s startFrame=%u frameCount=%u", config.outputDirectory.c_str(), config.startFrame, config.maxFrameCount);
        return true;
    }

    void Controller::OnCaptureStop()
    {
        recorder_->preparePending = false;
    }

    void Controller::Stop(const char* reason)
    {
        recorder_->preparePending = false;
        FailCapture(reason);
    }

    void Controller::GetStatus(FfxApiCaptureStatus& status)
    {
        ControllerBase::GetStatus(status, recorder_->GetCommittedFrameCount());
    }

    void Controller::OnPrepare()
    {
        Recorder& recorder = *recorder_;
        // ControllerBase only reads environment settings while this capture is
        // idle; active and draining sessions use their fixed capture config.
        recorder.preparePending = AdvanceFrameThenCheck();
    }

    bool Controller::BeginDispatchCapture(const ffxApiDispatchDescFrameGeneration&        dispatch,
                                          const ffxApiDispatchDescFrameGenerationPrepare& prepare,
                                          bool                                            effectiveReset)
    {
        if (!IsPreparePending())
            return false;
        Recorder&               recorder     = *recorder_;
        const Config&           config       = configInput_.GetActiveConfig();
        const SelectedResources selected     = SelectResources(prepare, context_);
        float                   upscaleRatio = 1.0f;
        if (!GetUpscaleRatio(prepare.depth.description.width,
                             prepare.depth.description.height,
                             dispatch.presentColor.description.width,
                             dispatch.presentColor.description.height,
                             upscaleRatio))
        {
            Stop("invalid NFRU render/display ratio");
            return false;
        }
        const uint32_t        datasetIndex = recorder.GetCommittedFrameCount();
        CaptureAttachmentPlan plan;
        if (!BuildNfruAttachmentPlan(config, dispatch, prepare, selected, upscaleRatio, datasetIndex, plan))
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
            Stop("failed to create capture session");
            return false;
        }
        auto       frameMeta         = std::make_unique<FrameMeta>();
        FrameMeta& meta              = *frameMeta;
        meta.datasetIndex            = datasetIndex;
        meta.upscaleRatio            = upscaleRatio;
        meta.jitterX                 = prepare.jitterOffset.x;
        meta.jitterY                 = prepare.jitterOffset.y;
        meta.cameraNear              = prepare.cameraNear;
        meta.cameraFar               = prepare.cameraFar;
        meta.cameraFovAngleVertical  = prepare.cameraFovAngleVertical;
        meta.motionVectorScaleX      = prepare.motionVectorScale.x;
        meta.motionVectorScaleY      = prepare.motionVectorScale.y;
        meta.frameTimeDelta          = prepare.frameTimeDelta;
        meta.viewSpaceToMetersFactor = prepare.viewSpaceToMetersFactor;
        std::memcpy(meta.viewProjection, prepare.viewProjection, sizeof(meta.viewProjection));
        meta.hasViewProjection = true;
        meta.renderWidth       = prepare.depth.description.width;
        meta.renderHeight      = prepare.depth.description.height;
        meta.displayWidth      = dispatch.presentColor.description.width;
        meta.displayHeight     = dispatch.presentColor.description.height;
        meta.colorFormat       = static_cast<uint32_t>(dispatch.presentColor.description.format);
        meta.depthFormat       = static_cast<uint32_t>(prepare.depth.description.format);
        meta.motionFormat      = static_cast<uint32_t>(prepare.motionVectors.description.format);
        meta.motionWidth       = prepare.motionVectors.description.width;
        meta.motionHeight      = prepare.motionVectors.description.height;
        meta.outputFormat      = static_cast<uint32_t>(dispatch.outputs[0].description.format);
        meta.opticalFlowFormat = static_cast<uint32_t>(selected.opticalFlow.description.format);
        meta.opticalFlowWidth  = selected.opticalFlow.description.width;
        meta.opticalFlowHeight = selected.opticalFlow.description.height;
        meta.reset             = effectiveReset || ShouldResetHistoryForDispatch();
        recorder.pendingPlan   = std::move(plan);
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
        recorder.pendingPlan.Clear();
        execution_.CancelTransaction();
        Stop("failed to record input attachments");
        return false;
    }

    void Controller::AbortDispatchCapture(const char* reason)
    {
        execution_.DiscardTransaction();
        recorder_->pendingPlan.Clear();
        Stop(reason);
    }

    bool Controller::CompleteDispatchCapture(const FfxResource& output)
    {
        const std::optional<uint32_t> index = execution_.GetTransactionDatasetIndex();
        if (!index)
            return false;
        if (recorder_->pendingPlan.inputCount > recorder_->pendingPlan.attachments.size() ||
            (*index > 0 && recorder_->pendingPlan.inputCount == recorder_->pendingPlan.attachments.size()))
        {
            AbortDispatchCapture("invalid capture attachment plan");
            return false;
        }

        Recorder&     recorder = *recorder_;
        const Config& config   = configInput_.GetActiveConfig();
        if (*index > 0)
            recorder.pendingPlan.attachments[recorder.pendingPlan.inputCount].source = output;
        bool attached = true;
        for (size_t attachmentIndex = recorder.pendingPlan.inputCount; attachmentIndex < recorder.pendingPlan.attachments.size(); ++attachmentIndex)
        {
            attached = attached && AddCaptureAttachment(execution_, config, recorder.pendingPlan.attachments[attachmentIndex]);
        }
        // index one, even though index zero intentionally has no generated-frame
        // output file (its output index would be -1).
        if (!attached || !execution_.CommitTransaction())
        {
            execution_.CancelTransaction();
            recorder.pendingPlan.Clear();
            Stop("failed to record or submit output attachments");
            return false;
        }
        recorder.pendingPlan.Clear();
        recorder.CommitFrame();
        ARM_CAPTURE_LOG_INFO("NFRU-CAPTURE", "recorded frame %u/%u", recorder.GetCommittedFrameCount(), config.maxFrameCount);
        if (recorder.GetCommittedFrameCount() >= config.maxFrameCount)
            RequestCaptureDrain(CaptureResult::Success);
        return true;
    }

}  // namespace arm::capture::framegenerations
