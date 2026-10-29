/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <FidelityFX/host/ffx_types.h>

#include <cmath>
#include <limits>

constexpr uint32_t               FFX_FRAMEINTERPOLATION_DATA_GRAPH_ALIGNMENT = 8;
constexpr uint32_t               FFX_FRAMEINTERPOLATION_MIN_RENDER_SIZE      = (FFX_FRAMEINTERPOLATION_DATA_GRAPH_ALIGNMENT + 1u) / 2u;
constexpr uint32_t               FFX_NFRU_OF_INPUT_COLOUR_MAX_HEIGHT         = 1080;
constexpr uint32_t               FFX_NFRU_MV_DEPTH_LANE_MAX_HEIGHT           = 540;
constexpr FfxOpticalFlowGridSize FFX_OPTICAL_FLOW_DEFAULT_GRID_SIZE          = FFX_OPTICAL_FLOW_GRID_SIZE_4X4;

inline uint32_t ffxNfruRoundPositive(double value);

enum class FfxNfruResolutionValidationError
{
    None,
    DisplayWidth,
    DisplayHeight,
    RenderWidth,
    RenderHeight,
    MvDepthWidth,
    MvDepthHeight,
    MvDepthAspectRatio,
    DerivedWidth,
    ShaderDimensionOutOfRange,
    DerivedDimensionOutOfRange,
};

inline FfxDimensions2D ffxNfruResolveMvDepthSize(FfxDimensions2D renderSize, FfxDimensions2D mvDepthSize)
{
    return mvDepthSize.width == 0 && mvDepthSize.height == 0 ? renderSize : mvDepthSize;
}

inline FfxNfruResolutionValidationError ffxNfruValidateResolutionInputs(FfxDimensions2D fullResColour,
                                                                        FfxDimensions2D renderSize,
                                                                        FfxDimensions2D mvDepthSize = {})
{
    if (!fullResColour.width)
    {
        return FfxNfruResolutionValidationError::DisplayWidth;
    }
    if (!fullResColour.height)
    {
        return FfxNfruResolutionValidationError::DisplayHeight;
    }
    constexpr uint32_t maxShaderDimension = static_cast<uint32_t>(std::numeric_limits<int32_t>::max());
    if (fullResColour.width > maxShaderDimension || fullResColour.height > maxShaderDimension || renderSize.width > maxShaderDimension ||
        renderSize.height > maxShaderDimension)
    {
        return FfxNfruResolutionValidationError::ShaderDimensionOutOfRange;
    }
    if (renderSize.width < FFX_FRAMEINTERPOLATION_MIN_RENDER_SIZE)
    {
        return FfxNfruResolutionValidationError::RenderWidth;
    }
    if (renderSize.height < FFX_FRAMEINTERPOLATION_MIN_RENDER_SIZE)
    {
        return FfxNfruResolutionValidationError::RenderHeight;
    }

    const FfxDimensions2D mvDepthInput = ffxNfruResolveMvDepthSize(renderSize, mvDepthSize);
    if (mvDepthInput.width < FFX_FRAMEINTERPOLATION_MIN_RENDER_SIZE)
    {
        return FfxNfruResolutionValidationError::MvDepthWidth;
    }
    if (mvDepthInput.height < FFX_FRAMEINTERPOLATION_MIN_RENDER_SIZE)
    {
        return FfxNfruResolutionValidationError::MvDepthHeight;
    }
    if (mvDepthInput.width > maxShaderDimension || mvDepthInput.height > maxShaderDimension)
    {
        return FfxNfruResolutionValidationError::ShaderDimensionOutOfRange;
    }
    if (static_cast<uint64_t>(mvDepthInput.width) * renderSize.height != static_cast<uint64_t>(renderSize.width) * mvDepthInput.height)
    {
        return FfxNfruResolutionValidationError::MvDepthAspectRatio;
    }

    const double   aspectRatio   = static_cast<double>(fullResColour.width) / static_cast<double>(fullResColour.height);
    const uint32_t ofInputHeight = fullResColour.height > FFX_NFRU_OF_INPUT_COLOUR_MAX_HEIGHT ? FFX_NFRU_OF_INPUT_COLOUR_MAX_HEIGHT : fullResColour.height;
    const uint32_t ofInputWidth  = fullResColour.height > FFX_NFRU_OF_INPUT_COLOUR_MAX_HEIGHT
                                      ? ffxNfruRoundPositive(static_cast<double>(FFX_NFRU_OF_INPUT_COLOUR_MAX_HEIGHT) * aspectRatio)
                                      : fullResColour.width;
    if (!ofInputWidth)
    {
        return FfxNfruResolutionValidationError::DerivedWidth;
    }
    if (ofInputWidth > maxShaderDimension)
    {
        return FfxNfruResolutionValidationError::DerivedDimensionOutOfRange;
    }

    const bool     fixedScalePath = mvDepthInput.height >= FFX_NFRU_MV_DEPTH_LANE_MAX_HEIGHT;
    const uint32_t preprocessWidth =
        fixedScalePath ? ffxNfruRoundPositive(static_cast<double>(FFX_NFRU_MV_DEPTH_LANE_MAX_HEIGHT) * aspectRatio) : mvDepthInput.width;
    const uint32_t preprocessHeight = fixedScalePath ? FFX_NFRU_MV_DEPTH_LANE_MAX_HEIGHT : mvDepthInput.height;
    if (!preprocessWidth)
    {
        return FfxNfruResolutionValidationError::DerivedWidth;
    }
    if (preprocessWidth > maxShaderDimension)
    {
        return FfxNfruResolutionValidationError::DerivedDimensionOutOfRange;
    }

    const uint32_t scaledFlowHeight = ofInputHeight / static_cast<uint32_t>(FFX_OPTICAL_FLOW_DEFAULT_GRID_SIZE);
    const uint32_t flowHeight       = scaledFlowHeight > 0 ? scaledFlowHeight : 1u;
    const uint32_t flowWidth        = ffxNfruRoundPositive(static_cast<double>(flowHeight) * aspectRatio);
    if (!flowWidth)
    {
        return FfxNfruResolutionValidationError::DerivedWidth;
    }
    if (flowWidth > maxShaderDimension)
    {
        return FfxNfruResolutionValidationError::DerivedDimensionOutOfRange;
    }

    constexpr uint64_t alignment    = FFX_FRAMEINTERPOLATION_DATA_GRAPH_ALIGNMENT;
    const uint64_t     paddedWidth  = ((static_cast<uint64_t>(preprocessWidth) + alignment - 1u) / alignment) * alignment;
    const uint64_t     paddedHeight = ((static_cast<uint64_t>(preprocessHeight) + alignment - 1u) / alignment) * alignment;
    if (paddedWidth > maxShaderDimension || paddedHeight > maxShaderDimension)
    {
        return FfxNfruResolutionValidationError::DerivedDimensionOutOfRange;
    }
    return FfxNfruResolutionValidationError::None;
}

inline bool ffxNfruSupportsOpticalFlowGrid(FfxOpticalFlowGridSize gridSize)
{
    return gridSize == FFX_OPTICAL_FLOW_GRID_SIZE_1X1 || gridSize == FFX_OPTICAL_FLOW_GRID_SIZE_2X2 || gridSize == FFX_OPTICAL_FLOW_GRID_SIZE_4X4 ||
           gridSize == FFX_OPTICAL_FLOW_GRID_SIZE_8X8;
}

struct FfxNfruResolutionPlan
{
    FfxDimensions2D fullResColour;
    FfxDimensions2D mvDepthInput;
    FfxDimensions2D ofInputColour;
    FfxDimensions2D preprocessLane;
    FfxDimensions2D mvDepthLane;
    FfxDimensions2D flowLane;
    FfxDimensions2D paddedTensorLane;
    FfxDimensions2D paramsLane;
    bool            needsOfColourDownsample;
    bool            needsAllInputDownsample;
    bool            needsColourDownsample;
    bool            noDownsample;
};

inline uint32_t ffxNfruRoundPositive(double value)
{
    if (!(value >= 0.0) || value > static_cast<double>(std::numeric_limits<uint32_t>::max()))
    {
        return 0;
    }

    const double whole    = std::floor(value);
    const double fraction = value - whole;
    const bool   roundUp  = fraction > 0.5 || (fraction == 0.5 && (static_cast<uint64_t>(whole) & 1u));
    return static_cast<uint32_t>(whole + (roundUp ? 1.0 : 0.0));
}

inline FfxDimensions2D ffxNfruGetFlowLaneSize(FfxDimensions2D fullResColour, FfxDimensions2D ofInputColour)
{
    const uint32_t scaledHeight = ofInputColour.height / static_cast<uint32_t>(FFX_OPTICAL_FLOW_DEFAULT_GRID_SIZE);
    const uint32_t flowHeight   = scaledHeight > 0 ? scaledHeight : 1u;
    const double   aspectRatio  = static_cast<double>(fullResColour.width) / static_cast<double>(fullResColour.height);
    const uint32_t flowWidth    = ffxNfruRoundPositive(static_cast<double>(flowHeight) * aspectRatio);
    return {flowWidth, flowHeight};
}

inline uint32_t ffxNfruCeilDivide(uint32_t numerator, uint32_t denominator)
{
    return (numerator + denominator - 1) / denominator;
}

inline FfxDimensions2D ffxNfruGetOpticalFlowOutputSize(FfxDimensions2D ofInputColour, FfxOpticalFlowGridSize gridSize)
{
    const uint32_t gridDimension = static_cast<uint32_t>(gridSize);
    if (!ffxNfruSupportsOpticalFlowGrid(gridSize))
    {
        return {};
    }
    return {ffxNfruCeilDivide(ofInputColour.width, gridDimension), ffxNfruCeilDivide(ofInputColour.height, gridDimension)};
}

inline FfxNfruResolutionPlan ffxNfruDeriveResolutionPlan(FfxDimensions2D fullResColour, FfxDimensions2D mvDepthInput, uint32_t dataGraphAlignment)
{
    FfxNfruResolutionPlan plan = {};
    plan.fullResColour         = fullResColour;
    plan.mvDepthInput          = mvDepthInput;

    const double aspectRatio = static_cast<double>(fullResColour.width) / static_cast<double>(fullResColour.height);

    plan.needsOfColourDownsample = fullResColour.height > FFX_NFRU_OF_INPUT_COLOUR_MAX_HEIGHT;
    plan.ofInputColour           = plan.needsOfColourDownsample
                             ? FfxDimensions2D{ffxNfruRoundPositive(FFX_NFRU_OF_INPUT_COLOUR_MAX_HEIGHT * aspectRatio), FFX_NFRU_OF_INPUT_COLOUR_MAX_HEIGHT}
                             : fullResColour;

    const bool fixedScalePath = mvDepthInput.height >= FFX_NFRU_MV_DEPTH_LANE_MAX_HEIGHT;
    plan.preprocessLane       = fixedScalePath
                              ? FfxDimensions2D{ffxNfruRoundPositive(FFX_NFRU_MV_DEPTH_LANE_MAX_HEIGHT * aspectRatio), FFX_NFRU_MV_DEPTH_LANE_MAX_HEIGHT}
                              : mvDepthInput;
    plan.mvDepthLane = plan.preprocessLane;

    const float depthPerPreprocess    = static_cast<float>(mvDepthInput.height) / static_cast<float>(FFX_NFRU_MV_DEPTH_LANE_MAX_HEIGHT);
    const float preprocessColourScale = static_cast<float>(fullResColour.height) / static_cast<float>(plan.preprocessLane.height);
    plan.needsAllInputDownsample      = fixedScalePath && depthPerPreprocess >= 1.5f;
    plan.needsColourDownsample        = depthPerPreprocess < 1.5f && preprocessColourScale >= 1.5f;
    plan.noDownsample                 = depthPerPreprocess < 1.5f && preprocessColourScale < 1.5f;

    plan.flowLane = ffxNfruGetFlowLaneSize(plan.fullResColour, plan.ofInputColour);

    plan.paddedTensorLane.width  = ((plan.preprocessLane.width + dataGraphAlignment - 1) / dataGraphAlignment) * dataGraphAlignment;
    plan.paddedTensorLane.height = ((plan.preprocessLane.height + dataGraphAlignment - 1) / dataGraphAlignment) * dataGraphAlignment;
    plan.paramsLane.width        = plan.paddedTensorLane.width / 2;
    plan.paramsLane.height       = plan.paddedTensorLane.height / 2;

    return plan;
}

inline FfxResourceUsage ffxNfruGetDepthHistoryUsage(const FfxNfruResolutionPlan& resolution, bool warpUsesFragment)
{
    return resolution.needsAllInputDownsample || warpUsesFragment ? FFX_RESOURCE_USAGE_RENDERTARGET : FFX_RESOURCE_USAGE_UAV;
}
