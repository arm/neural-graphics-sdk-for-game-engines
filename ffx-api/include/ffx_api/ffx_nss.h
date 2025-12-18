// This file is part of the FidelityFX SDK.
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files(the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and /or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions :
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT

#pragma once
#include "ffx_api.h"
#include "ffx_api_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/// @ingroup ffxNss
enum FfxApiNssShaderQualityMode
{
    FFX_API_NSS_SHADER_QUALITY_MODE_QUALITY     = 1,  ///< Perform upscaling with a shader quality mode of 'Quality'
    FFX_API_NSS_SHADER_QUALITY_MODE_BALANCED    = 2,  ///< Perform upscaling with a shader quality mode of 'Balanced'
    FFX_API_NSS_SHADER_QUALITY_MODE_PERFORMANCE = 3,  ///< Perform upscaling with a shader quality mode of 'Performance'
};

/// @ingroup ffxNss
enum FfxApiCreateContextNssFlags
{
    FFX_API_NSS_CONTEXT_FLAG_QUANTIZED              = (1 << 0),  ///< Use a quantized data graph. Resources will be quantized to 8 bits.
    FFX_API_NSS_CONTEXT_FLAG_HIGH_DYNAMIC_RANGE     = (1 << 1),  ///< A bit indicating if the input color data provided is using a high-dynamic range.
    FFX_API_NSS_CONTEXT_FLAG_DEPTH_INVERTED         = (1 << 2),  ///< A bit indicating that the input depth buffer data provided is inverted [1..0].
    FFX_API_NSS_CONTEXT_FLAG_DEPTH_INFINITE         = (1 << 3),  ///< A bit indicating that the input depth buffer data provided is using an infinite far plane.
    FFX_API_NSS_CONTEXT_FLAG_RESAMPLE_BICUBIC       = (1 << 4),  ///< A bit indicating sample using Bicubic filtering
    FFX_API_NSS_CONTEXT_FLAG_READ_TENSORS_AS_IMAGES = (1 << 5),  ///< A bit indicating tensor image aliasing is enable.
    FFX_API_NSS_CONTEXT_FLAG_ALLOW_16BIT            = (1 << 6),  ///< A bit indicating that the runtime should allow 16bit resources to be used.
    FFX_API_NSS_CONTEXT_FLAG_DISABLE_PADDING        = (1 << 7),  ///< A bit indicating that the padding is disabled in sdk.
    FFX_API_NSS_CONTEXT_FLAG_ENABLE_DEBUG_CHECKING  = (1 << 8),  ///< A bit indicating that the runtime should check some API values and report issues.
};

/// @ingroup ffxNss
enum FfxApiDispatchNssFlags
{
    FFX_API_NSS_DISPATCH_FLAG_DRAW_DEBUG_VIEW = (1 << 0),  ///< A bit indicating that the output resource will contain debug views with relevant information.
};

/// @ingroup ffxNss
#define FFX_API_EFFECT_ID_NSS 0x000F0000u  ///< 0x000F0000u - chosen to avoid conflicts with the Upscale api - which is 0x00010000u

#define FFX_API_CREATE_CONTEXT_DESC_TYPE_NSS 0x000F0000u  ///< header type for <c><i>ffxApiCreateContextDescNss</i></c>.
/// @ingroup ffxNss
struct ffxApiCreateContextDescNss
{
    ffxCreateContextDescHeader header;
    uint32_t                   flags;           ///< Zero or a combination of values from FfxApiCreateContextNssFlags.
    struct FfxApiDimensions2D  maxRenderSize;   ///< The maximum size that rendering will be performed at.
    struct FfxApiDimensions2D  maxUpscaleSize;  ///< The size of the presentation resolution targeted by the upscaling process.

    ffxApiMessage              fpMessage;    ///< A pointer to a function that can receive messages from the runtime. May be null.
    FfxApiNssShaderQualityMode qualityMode;  ///< The desired quality mode for the upscaling process.
};

/// @ingroup ffxNss
#define FFX_API_DISPATCH_DESC_TYPE_NSS 0x000F0001u  ///< header type for <c><i>ffxApiDispatchDescNss</i></c>.
/// @ingroup ffxNss
struct ffxApiDispatchDescNss
{
    ffxDispatchDescHeader header;
    void*                 commandList;  ///< Command list to record upscaling rendering commands into.

    struct FfxApiResource color;     ///< A <c><i>FfxResource</i></c> containing the color buffer for the current frame (at render resolution).
    struct FfxApiResource depth;     ///< A <c><i>FfxResource</i></c> containing 32bit depth values for the current frame (at render resolution).
    struct FfxApiResource depthTm1;  ///< A <c><i>FfxResource</i></c> containing 32bit depth values for the previous frame (at render resolution).
    struct FfxApiResource
                          motionVectors;  ///< A <c><i>FfxResource</i></c> containing 2-dimensional motion vectors (at render resolution if <c><i>FFX_NSS_ENABLE_DISPLAY_RESOLUTION_MOTION_VECTORS</i></c> is not set).
    struct FfxApiResource outputTm1;  ///< A <c><i>FfxResource</i></c> containing the output color buffer for the previous frame (at presentation resolution).
    struct FfxApiResource output;     ///< A <c><i>FfxResource</i></c> containing the output color buffer for the current frame (at presentation resolution).

    struct FfxApiResource
        debugViews;  ///< A <c><i>FfxResource</i></c> containing the debug views if DRAW_DEBUG_VIEW dispatch flag is turned on. Can be left nullptr if not using it.
    ///< Necessary because we will pollute the colour history if we render debug views directly in the output buffer.
    ///< (We could also use an internally FFX managed colour history, but that would imply doing an extra blit every frame to copy the input colour into history!)

    struct FfxApiFloatCoords2D jitterOffset;            ///< The subpixel jitter offset applied to the camera.
    struct FfxApiDimensions2D  upscaleSize;             ///< The resolution that was used for rendering the output resources.
    struct FfxApiDimensions2D  renderSize;              ///< The resolution that was used for rendering the input resources.
    float                      cameraNear;              ///< The distance to the near plane of the camera.
    float                      cameraFar;               ///< The distance to the far plane of the camera.
    float                      cameraFovAngleVertical;  ///< The camera angle field of view in the vertical direction (expressed in radians).

    float                      exposure;           ///< The exposure value
    struct FfxApiFloatCoords2D motionVectorScale;  ///< The scale applied to the motion vectors.

    float frameTimeDelta;  ///< The time elapsed since the last frame (expressed in milliseconds).
    bool  reset;           ///< A boolean value which when set to true, indicates the camera has moved discontinuously.

    uint32_t flags;  ///< Zero or a combination of values from FfxApiDispatchNssFlags.
};

/// @ingroup ffxNss
#define FFX_API_QUERY_DESC_TYPE_NSS_GETJITTERPHASECOUNT 0x000F0004u  ///< header type for <c><i>ffxApiQueryDescNssGetJitterPhaseCount</i></c>.
/// @ingroup ffxNss
struct ffxApiQueryDescNssGetJitterPhaseCount
{
    ffxQueryDescHeader header;
    uint32_t           renderWidth;   ///< The render resolution width.
    uint32_t           displayWidth;  ///< The output resolution width.
    int32_t*
        pOutPhaseCount;  ///< A pointer to a <c>int32_t</c> which will hold the jitter phase count for the scaling factor between <c><i>renderWidth</i></c> and <c><i>displayWidth</i></c>.
};

/// @ingroup ffxNss
#define FFX_API_QUERY_DESC_TYPE_NSS_GETJITTEROFFSET 0x000F0005u  ///< header type for <c><i>ffxApiQueryDescNssGetJitterOffset</i></c>.
/// @ingroup ffxNss
struct ffxApiQueryDescNssGetJitterOffset
{
    ffxQueryDescHeader header;
    int32_t            index;       ///< The index within the jitter sequence.
    int32_t            phaseCount;  ///< The length of jitter phase. See <c><i>ffxQueryDescFsrGetJitterPhaseCount</i></c>.
    float*             pOutX;       ///< A pointer to a <c>float</c> which will contain the subpixel jitter offset for the x dimension.
    float*             pOutY;       ///< A pointer to a <c>float</c> which will contain the subpixel jitter offset for the y dimension.
};

#ifdef __cplusplus
}
#endif
