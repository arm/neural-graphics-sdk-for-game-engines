/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT
 */

#include "../framework/test_framework.h"

#include <FidelityFX/host/ffx_frameinterpolation.h>
#include <FidelityFX/host/ffx_frameinterpolation_resolution.h>
#include <ffx_api/ffx_framegeneration.hpp>
#include "../../sdk/src/components/frameinterpolation/ffx_frameinterpolation_private.h"

#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace
{
    std::vector<std::string> messages;
    uint32_t                 registeredResources = 0;

    void MessageSink(FfxMsgType, const char* message)
    {
        messages.emplace_back(message);
    }

    void ApiMessageSink(uint32_t, const char* message)
    {
        messages.emplace_back(message);
    }

    bool SameSize(FfxDimensions2D first, FfxDimensions2D second)
    {
        return first.width == second.width && first.height == second.height;
    }

    FfxErrorCode RegisterResource(FfxInterface*, const FfxResource*, FfxUInt32, FfxResourceInternal* resource)
    {
        resource->internalIndex = ++registeredResources;
        return FFX_OK;
    }
}  // namespace

UNIT_TEST_CASE(nfru_resolution, resolution_lanes_follow_documented_boundaries)
{
    const struct
    {
        const char*     name;
        FfxDimensions2D display;
        FfxDimensions2D mvDepth;
        FfxDimensions2D ofInput;
        FfxDimensions2D lane;
        FfxDimensions2D flow;
        FfxDimensions2D tensor;
        FfxDimensions2D params;
        bool            allInputDownsample;
        bool            colourDownsample;
    } cases[] = {
        {"native", {640, 360}, {640, 360}, {640, 360}, {640, 360}, {160, 90}, {640, 360}, {320, 180}, false, false},
        {"uncapped colour-only", {1920, 1080}, {640, 360}, {1920, 1080}, {640, 360}, {480, 270}, {640, 360}, {320, 180}, false, true},
        {"pre-downsampled 540p", {3840, 2160}, {960, 540}, {1920, 1080}, {960, 540}, {480, 270}, {960, 544}, {480, 272}, false, true},
        {"below lane cap", {2160, 2160}, {1078, 539}, {1080, 1080}, {1078, 539}, {270, 270}, {1080, 544}, {540, 272}, false, true},
        {"at lane cap", {2160, 2160}, {1080, 540}, {1080, 1080}, {540, 540}, {270, 270}, {544, 544}, {272, 272}, false, true},
        {"below all-input threshold", {2160, 2160}, {809, 809}, {1080, 1080}, {540, 540}, {270, 270}, {544, 544}, {272, 272}, false, true},
        {"at all-input threshold", {2160, 2160}, {810, 810}, {1080, 1080}, {540, 540}, {270, 270}, {544, 544}, {272, 272}, true, false},
        {"above all-input threshold", {2160, 2160}, {1080, 1080}, {1080, 1080}, {540, 540}, {270, 270}, {544, 544}, {272, 272}, true, false},
    };

    std::string           failures;
    const FfxDimensions2D renderSize{1920, 1080};
    const auto            omittedInput  = ffxNfruResolveMvDepthSize(renderSize, {});
    const auto            explicitInput = ffxNfruResolveMvDepthSize(renderSize, renderSize);
    const auto            omittedPlan   = ffxNfruDeriveResolutionPlan({3840, 2160}, omittedInput, FFX_FRAMEINTERPOLATION_DATA_GRAPH_ALIGNMENT);
    const auto            explicitPlan  = ffxNfruDeriveResolutionPlan({3840, 2160}, explicitInput, FFX_FRAMEINTERPOLATION_DATA_GRAPH_ALIGNMENT);
    const auto            samePlan      = [](const FfxNfruResolutionPlan& first, const FfxNfruResolutionPlan& second) {
        return SameSize(first.fullResColour, second.fullResColour) && SameSize(first.mvDepthInput, second.mvDepthInput) &&
               SameSize(first.ofInputColour, second.ofInputColour) && SameSize(first.preprocessLane, second.preprocessLane) &&
               SameSize(first.mvDepthLane, second.mvDepthLane) && SameSize(first.flowLane, second.flowLane) &&
               SameSize(first.paddedTensorLane, second.paddedTensorLane) && SameSize(first.paramsLane, second.paramsLane) &&
               first.needsOfColourDownsample == second.needsOfColourDownsample && first.needsAllInputDownsample == second.needsAllInputDownsample &&
               first.needsColourDownsample == second.needsColourDownsample && first.noDownsample == second.noDownsample;
    };
    if (!SameSize(omittedInput, renderSize) || !SameSize(explicitInput, renderSize) || !samePlan(omittedPlan, explicitPlan))
    {
        failures += "omitted input: does not match explicit render-sized resolution plan; ";
    }
    for (const auto& testCase : cases)
    {
        const auto plan = ffxNfruDeriveResolutionPlan(testCase.display, testCase.mvDepth, FFX_FRAMEINTERPOLATION_DATA_GRAPH_ALIGNMENT);
        if (!SameSize(plan.ofInputColour, testCase.ofInput) || !SameSize(plan.mvDepthLane, testCase.lane) || !SameSize(plan.flowLane, testCase.flow) ||
            !SameSize(plan.paddedTensorLane, testCase.tensor) || !SameSize(plan.paramsLane, testCase.params))
        {
            failures += std::string(testCase.name) + ": incorrect resolution lanes; ";
        }
        if (plan.needsAllInputDownsample != testCase.allInputDownsample || plan.needsColourDownsample != testCase.colourDownsample ||
            plan.noDownsample != (!testCase.allInputDownsample && !testCase.colourDownsample) ||
            plan.needsOfColourDownsample != (testCase.display.height > testCase.ofInput.height))
        {
            failures += std::string(testCase.name) + ": incorrect downsample topology; ";
        }
        const auto historyUsage = testCase.allInputDownsample ? FFX_RESOURCE_USAGE_RENDERTARGET : FFX_RESOURCE_USAGE_UAV;
        if (ffxNfruGetDepthHistoryUsage(plan, false) != historyUsage || ffxNfruGetDepthHistoryUsage(plan, true) != FFX_RESOURCE_USAGE_RENDERTARGET)
        {
            failures += std::string(testCase.name) + ": incorrect depth history usage; ";
        }
    }
    if (!failures.empty())
    {
        UNIT_TEST_FAIL(failures);
    }
    return true;
}

UNIT_TEST_CASE(nfru_resolution, positive_rounding_rejects_unrepresentable_widths)
{
    const uint32_t maxWidth = std::numeric_limits<uint32_t>::max();
    UNIT_TEST_CHECK(ffxNfruRoundPositive(static_cast<double>(maxWidth) - 0.25) == maxWidth);
    UNIT_TEST_CHECK(ffxNfruRoundPositive(static_cast<double>(maxWidth) + 1.0) == 0);
    UNIT_TEST_CHECK(ffxNfruRoundPositive(std::numeric_limits<double>::infinity()) == 0);
    UNIT_TEST_CHECK(ffxNfruRoundPositive(std::numeric_limits<double>::quiet_NaN()) == 0);
    return true;
}

UNIT_TEST_CASE(nfru_resolution, context_creation_validates_resolution_at_both_entry_points)
{
    using Error                                      = FfxNfruResolutionValidationError;
    constexpr uint32_t maxShaderDimension            = static_cast<uint32_t>(std::numeric_limits<int32_t>::max());
    constexpr uint32_t firstUnrepresentableDimension = maxShaderDimension + 1u;
    const struct
    {
        const char*     name;
        FfxDimensions2D display;
        FfxDimensions2D render;
        FfxDimensions2D mvDepth;
        Error           expectedError;
        const char*     apiDiagnostic;
        const char*     hostDiagnostic;
    } cases[] = {
        {"omitted input", {3840, 2160}, {1920, 1080}, {}, Error::None, nullptr, "opticalFlowGridSize"},
        {"explicit render input", {3840, 2160}, {1920, 1080}, {1920, 1080}, Error::None, nullptr, "opticalFlowGridSize"},
        {"reduced input", {3840, 2160}, {1920, 1080}, {960, 540}, Error::None, nullptr, "opticalFlowGridSize"},
        {"zero input width", {3840, 2160}, {1920, 1080}, {0, 540}, Error::MvDepthWidth, "mvDepthSize.width", "mvDepthSize.width"},
        {"zero input height", {3840, 2160}, {1920, 1080}, {960, 0}, Error::MvDepthHeight, "mvDepthSize.height", "mvDepthSize.height"},
        {"undersized input", {3840, 2160}, {1920, 1080}, {2, 2}, Error::MvDepthWidth, "mvDepthSize.width", "mvDepthSize.width"},
        {"aspect ratio mismatch", {3840, 2160}, {1920, 1080}, {960, 541}, Error::MvDepthAspectRatio, "aspect ratio", "aspect ratio"},
        {"zero render width", {3840, 2160}, {0, 1080}, {960, 540}, Error::RenderWidth, "render width", "maxRenderSize.width"},
        {"zero display width", {0, 2160}, {1920, 1080}, {960, 540}, Error::DisplayWidth, "display width", "displaySize.width"},
        {"maximum signed dimension", {maxShaderDimension, 1080}, {maxShaderDimension, 540}, {}, Error::None, nullptr, "opticalFlowGridSize"},
        {"display width above shader range",
         {firstUnrepresentableDimension, 1080},
         {maxShaderDimension, 540},
         {},
         Error::ShaderDimensionOutOfRange,
         "shader range",
         "shader range"},
        {"display height above shader range",
         {1920, firstUnrepresentableDimension},
         {1920, 1080},
         {},
         Error::ShaderDimensionOutOfRange,
         "shader range",
         "shader range"},
        {"render width above shader range",
         {1920, 1080},
         {firstUnrepresentableDimension, 1080},
         {},
         Error::ShaderDimensionOutOfRange,
         "shader range",
         "shader range"},
        {"render height above shader range",
         {1920, 1080},
         {1920, firstUnrepresentableDimension},
         {},
         Error::ShaderDimensionOutOfRange,
         "shader range",
         "shader range"},
        {"mvDepthSize width above shader range",
         {1920, 1080},
         {1920, 1080},
         {firstUnrepresentableDimension, 1080},
         Error::ShaderDimensionOutOfRange,
         "shader range",
         "shader range"},
        {"mvDepthSize height above shader range",
         {1920, 1080},
         {1920, 1080},
         {1920, firstUnrepresentableDimension},
         Error::ShaderDimensionOutOfRange,
         "shader range",
         "shader range"},
        {"derived lane width exceeds signed range",
         {maxShaderDimension, 500},
         {128, 540},
         {},
         Error::DerivedDimensionOutOfRange,
         "derived NFRU dimensions",
         "derived NFRU dimensions"},
        {"aligned tensor width exceeds signed range",
         {maxShaderDimension, 8},
         {maxShaderDimension, 8},
         {},
         Error::DerivedDimensionOutOfRange,
         "derived NFRU dimensions",
         "derived NFRU dimensions"},
        {"derived lane width is not representable",
         {maxShaderDimension, 1},
         {128, 540},
         {},
         Error::DerivedWidth,
         "derived NFRU lane width",
         "derived lane widths"},
        // The aspect-ratio cross-products exceed 32 bits.
        {"large mismatched input", {3840, 2160}, {131072, 65536}, {65536, 65536}, Error::MvDepthAspectRatio, "aspect ratio", "aspect ratio"},
        {"large matching input", {3840, 2160}, {131072, 65536}, {65536, 32768}, Error::None, nullptr, "opticalFlowGridSize"},
    };

    auto                                    hostContext    = std::make_unique<FfxFrameInterpolationContext>();
    uint32_t                                backendCreates = 0;
    FfxFrameInterpolationContextDescription hostDescription{};
    hostDescription.fpMessage                               = MessageSink;
    hostDescription.backendInterface.device                 = &backendCreates;
    hostDescription.backendInterface.fpGetSDKVersion        = [](FfxInterface*) -> FfxVersionNumber { return 0; };
    hostDescription.backendInterface.fpCreateBackendContext = [](FfxInterface* backend, FfxEffect, FfxEffectBindlessConfig*, FfxUInt32*) -> FfxErrorCode {
        ++*static_cast<uint32_t*>(backend->device);
        return FFX_ERROR_BACKEND_API_ERROR;
    };
    hostDescription.backendInterface.fpDestroyBackendContext = [](FfxInterface*, FfxUInt32) -> FfxErrorCode { return FFX_OK; };

    std::string failures;
    for (const auto& testCase : cases)
    {
        if (ffxNfruValidateResolutionInputs(testCase.display, testCase.render, testCase.mvDepth) != testCase.expectedError)
        {
            failures += std::string(testCase.name) + ": incorrect resolution validation result; ";
        }

        ffx::CreateContextDescFrameGeneration description{};
        description.displaySize = {testCase.display.width, testCase.display.height};
        description.renderSize  = {testCase.render.width, testCase.render.height};
        if (testCase.mvDepth.width == 0 && testCase.mvDepth.height == 0)
        {
            if (description.mvDepthSize.width != 0 || description.mvDepthSize.height != 0)
            {
                failures += std::string(testCase.name) + ": descriptor default must omit the motion/depth size; ";
            }
        }
        else
        {
            description.mvDepthSize = {testCase.mvDepth.width, testCase.mvDepth.height};
        }
        description.backBufferFormat = FFX_API_SURFACE_FORMAT_R8G8B8A8_UNORM;
        description.fpMessage        = ApiMessageSink;
        ffxContext context           = nullptr;
        messages.clear();
        // Valid sizes reach the missing-backend error; invalid sizes must be rejected earlier.
        const auto apiResult = testCase.expectedError == Error::None ? FFX_API_RETURN_ERROR : FFX_API_RETURN_ERROR_PARAMETER;
        if (ffxCreateContext(&context, &description.header, nullptr) != apiResult || context != nullptr ||
            (testCase.apiDiagnostic && (messages.empty() || messages.front().find(testCase.apiDiagnostic) == std::string::npos)))
        {
            failures += std::string(testCase.name) + ": incorrect API creation result or diagnostic; ";
        }

        hostDescription.displaySize   = testCase.display;
        hostDescription.maxRenderSize = testCase.render;
        hostDescription.mvDepthSize   = testCase.mvDepth;
        messages.clear();
        backendCreates = 0;
        // An unset optical-flow grid stops valid sizes before any backend is created.
        if (ffxFrameInterpolationContextCreate(hostContext.get(), &hostDescription) != FFX_ERROR_INVALID_ARGUMENT || backendCreates != 0 || messages.empty() ||
            messages.front().find(testCase.hostDiagnostic) == std::string::npos)
        {
            failures += std::string(testCase.name) + ": incorrect host creation result or diagnostic; ";
        }
    }
    if (!failures.empty())
    {
        UNIT_TEST_FAIL(failures);
    }
    return true;
}

UNIT_TEST_CASE(nfru_resolution, prepare_uses_input_size_for_motion_and_validates_history)
{
    auto  context                                                 = std::make_unique<FfxFrameInterpolationContext>();
    auto* state                                                   = reinterpret_cast<FfxArmFrameInterpolationContext_Private*>(context.get());
    state->contextDescription.maxRenderSize                       = {1920, 1080};
    state->contextDescription.mvDepthSize                         = {960, 540};
    state->contextDescription.displaySize                         = {3840, 2160};
    state->contextDescription.fpMessage                           = MessageSink;
    state->contextDescription.backendInterface.fpRegisterResource = RegisterResource;
    state->resolutionPlan             = ffxNfruDeriveResolutionPlan({3840, 2160}, {960, 540}, FFX_FRAMEINTERPOLATION_DATA_GRAPH_ALIGNMENT);
    state->mvDepthLane                = state->resolutionPlan.mvDepthLane;
    state->constants._RenderSize[0]   = 1920;
    state->constants._RenderSize[1]   = 1080;
    state->constants._DisplaySize[0]  = 3840;
    state->constants._DisplaySize[1]  = 2160;
    state->lastFrameViewProjection[0] = state->lastFrameViewProjection[5] = state->lastFrameViewProjection[10] = state->lastFrameViewProjection[15] = 1.0f;

    FfxFrameInterpolationPrepareDescription description{};
    description.commandList            = context.get();
    description.cameraNear             = 0.1f;
    description.cameraFar              = 1000.0f;
    description.cameraFovAngleVertical = 1.2f;
    description.motionVectorScale      = {960.0f, 540.0f};
    description.viewProjection[0] = description.viewProjection[5] = description.viewProjection[10] = description.viewProjection[15] = 1.0f;
    description.depth.resource                                                                                                      = context.get();
    description.depth.description.width                                                                                             = 960;
    description.depth.description.height                                                                                            = 540;
    description.motionVectors                                                                                                       = description.depth;
    description.depthTm1                                                                                                            = description.depth;
    registeredResources                                                                                                             = 0;
    UNIT_TEST_CHECK(ffxFrameInterpolationPrepare(context.get(), &description) == FFX_OK);
    UNIT_TEST_CHECK(registeredResources > 0);
    UNIT_TEST_CHECK(state->constants._MotionVectorScale[0] == 1.0f && state->constants._MotionVectorScale[1] == 1.0f);
    UNIT_TEST_CHECK(state->constants._RenderSize[0] == 1920 && state->constants._RenderSize[1] == 1080);

    description.motionVectorScale = {1.0f, 1.0f};
    UNIT_TEST_CHECK(ffxFrameInterpolationPrepare(context.get(), &description) == FFX_OK);
    UNIT_TEST_CHECK(std::abs(state->constants._MotionVectorScale[0] - 1.0f / 960.0f) < 1e-7f);
    UNIT_TEST_CHECK(std::abs(state->constants._MotionVectorScale[1] - 1.0f / 540.0f) < 1e-7f);
    state->contextDescription.flags = FFX_FG_CONTEXT_FLAG_ENABLE_DISPLAY_RESOLUTION_MOTION_VECTORS;
    description.motionVectorScale   = {3840.0f, 2160.0f};
    UNIT_TEST_CHECK(ffxFrameInterpolationPrepare(context.get(), &description) == FFX_OK);
    UNIT_TEST_CHECK(state->constants._MotionVectorScale[0] == 1.0f && state->constants._MotionVectorScale[1] == 1.0f);

    state->contextDescription.flags = FFX_FG_CONTEXT_FLAG_MANAGE_PREVIOUS_DEPTH;
    description.depthTm1Next        = description.depth;
    UNIT_TEST_CHECK(ffxFrameInterpolationPrepare(context.get(), &description) == FFX_OK);
    description.viewProjection[12] = 5.0f;
    for (FfxResource* resource : {&description.depth, &description.motionVectors, &description.depthTm1, &description.depthTm1Next})
    {
        resource->description.width = 1920;
        registeredResources         = 0;
        messages.clear();
        UNIT_TEST_CHECK(ffxFrameInterpolationPrepare(context.get(), &description) == FFX_ERROR_INVALID_ARGUMENT);
        UNIT_TEST_CHECK(registeredResources == 0);
        UNIT_TEST_CHECK(state->lastFrameViewProjection[12] == 0.0f);
        UNIT_TEST_CHECK(!messages.empty() && messages.front().find("dimensions") != std::string::npos);
        resource->description.width = 960;
    }

    // All-input reduction still normalizes raw motion in input pixels, not in the reduced history lane.
    state->contextDescription.flags       = 0;
    state->contextDescription.mvDepthSize = {1920, 1080};
    state->resolutionPlan                 = ffxNfruDeriveResolutionPlan({3840, 2160}, {1920, 1080}, FFX_FRAMEINTERPOLATION_DATA_GRAPH_ALIGNMENT);
    description.depth.description.width   = 1920;
    description.depth.description.height  = 1080;
    description.motionVectors             = description.depth;
    description.motionVectorScale         = {1920.0f, 1080.0f};
    UNIT_TEST_CHECK(ffxFrameInterpolationPrepare(context.get(), &description) == FFX_OK);
    UNIT_TEST_CHECK(state->constants._MotionVectorScale[0] == 1.0f && state->constants._MotionVectorScale[1] == 1.0f);
    return true;
}
