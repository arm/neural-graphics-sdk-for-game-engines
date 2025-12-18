// Copyright  © 2023 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT

#include <algorithm>  // for max used inside SPD CPU code.
#include <cfloat>     // for FLT_EPSILON
#include <cmath>      // for fabs, abs, sinf, sqrt, etc.
#include <string.h>   // for memset
#include <cstdint>

#include "FidelityFX/host/ffx_nss.h"
#define FFX_CPU

#include "FidelityFX/gpu/ffx_core.h"
#include "ffx_object_management.h"

#include "FidelityFX/host/ffx_util.h"

#ifdef __clang__
#pragma clang diagnostic ignored "-Wunused-variable"
#endif

#include "ffx_nss_private.h"
#include <tuple>
#include <cmath>

// max queued frames for descriptor management
static const uint32_t NSS_MAX_QUEUED_FRAMES = 16;

// threashold for whether we should use scale preset mode
static constexpr float SCALE_PRESET_MODE_THRESHOLD = 0.01;

// lists to map shader resource bindpoint name to resource identifier
typedef struct ResourceBinding
{
    uint32_t index;
    wchar_t  name[64];
} ResourceBinding;

static const ResourceBinding srvTextureBindingTable[] = {
    // For mirror padding
    {FFX_NSS_RESOURCE_IDENTIFIER_UNPADDED_COLOR, L"r_unpadded_color"},
    {FFX_NSS_RESOURCE_IDENTIFIER_UNPADDED_DEPTH, L"r_unpadded_depth"},
    {FFX_NSS_RESOURCE_IDENTIFIER_UNPADDED_DEPTH_TM1, L"r_unpadded_depth_tm1"},
    {FFX_NSS_RESOURCE_IDENTIFIER_UNPADDED_MOTION, L"r_unpadded_motion"},

    {FFX_NSS_RESOURCE_IDENTIFIER_INPUT_COLOR, L"r_input_color_jittered"},
    {FFX_NSS_RESOURCE_IDENTIFIER_INPUT_DEPTH, L"r_input_depth"},
    {FFX_NSS_RESOURCE_IDENTIFIER_INPUT_DEPTH_TM1, L"r_prev_depth"},
    {FFX_NSS_RESOURCE_IDENTIFIER_INPUT_MOTION_VECTORS, L"r_input_motion_vectors"},
    {FFX_NSS_RESOURCE_IDENTIFIER_LUMA_DERIV, L"r_prev_luma_deriv"},
    {FFX_NSS_RESOURCE_IDENTIFIER_HISTORY_UPSCALED_COLOR, L"r_prev_upscaled_color"},
    {FFX_NSS_RESOURCE_IDENTIFIER_NEAREST_DEPTH_COORD, L"r_input_nearest_depth_coord"},
    {FFX_NSS_RESOURCE_IDENTIFIER_NEAREST_DEPTH_COORD_TM1, L"r_input_nearest_depth_coord_tm1"},

    // Aliased tensors
    {FFX_NSS_RESOURCE_IDENTIFIER_FEEDBACK_TENSOR, L"r_prev_feedback_tensor"},
    {FFX_NSS_RESOURCE_IDENTIFIER_K0_TENSOR, L"r_coefficients_k0_tensor"},
    {FFX_NSS_RESOURCE_IDENTIFIER_K1_TENSOR, L"r_coefficients_k1_tensor"},
    {FFX_NSS_RESOURCE_IDENTIFIER_K2_TENSOR, L"r_coefficients_k2_tensor"},
    {FFX_NSS_RESOURCE_IDENTIFIER_K3_TENSOR, L"r_coefficients_k3_tensor"},
    {FFX_NSS_RESOURCE_IDENTIFIER_K4_TENSOR, L"r_coefficients_k4_tensor"},
    {FFX_NSS_RESOURCE_IDENTIFIER_PREPROCESS_INPUT_TENSOR, L"r_preprocessed_tensor"},
};

static const ResourceBinding nssUavTextureBindingTable[] = {
    {FFX_NSS_RESOURCE_IDENTIFIER_LUMA_DERIV, L"rw_luma_deriv"},
    {FFX_NSS_RESOURCE_IDENTIFIER_UPSCALED_OUTPUT, L"rw_upscaled_output"},
    {FFX_NSS_RESOURCE_IDENTIFIER_NEAREST_DEPTH_COORD, L"rw_nearest_depth_coord_out"},
    {FFX_NSS_RESOURCE_IDENTIFIER_DEBUG_VIEWS, L"rw_debug_views"},

    // For mirror padding
    {FFX_NSS_RESOURCE_IDENTIFIER_INPUT_COLOR, L"rw_input_color_jittered"},
    {FFX_NSS_RESOURCE_IDENTIFIER_INPUT_DEPTH, L"rw_input_depth"},
    {FFX_NSS_RESOURCE_IDENTIFIER_INPUT_DEPTH_TM1, L"rw_prev_depth"},
    {FFX_NSS_RESOURCE_IDENTIFIER_INPUT_MOTION_VECTORS, L"rw_input_motion_vectors"},
};

static const ResourceBinding srvTensorBindingTable[] = {
    {FFX_NSS_RESOURCE_IDENTIFIER_FEEDBACK_TENSOR, L"r_prev_feedback_tensor"},
    {FFX_NSS_RESOURCE_IDENTIFIER_K0_TENSOR, L"r_coefficients_k0_tensor"},
    {FFX_NSS_RESOURCE_IDENTIFIER_K1_TENSOR, L"r_coefficients_k1_tensor"},
    {FFX_NSS_RESOURCE_IDENTIFIER_K2_TENSOR, L"r_coefficients_k2_tensor"},
    {FFX_NSS_RESOURCE_IDENTIFIER_K3_TENSOR, L"r_coefficients_k3_tensor"},
    {FFX_NSS_RESOURCE_IDENTIFIER_K4_TENSOR, L"r_coefficients_k4_tensor"},
    {FFX_NSS_RESOURCE_IDENTIFIER_PREPROCESS_INPUT_TENSOR, L"r_preprocessed_tensor"},
};

static const ResourceBinding uavTensorBindingTable[] = {
    // Shader resources - taken from shader reflection information
    {FFX_NSS_RESOURCE_IDENTIFIER_PREPROCESS_INPUT_TENSOR, L"rw_preprocessed_tensor"},

    // Data graph resources - taken from data graph reflection information.
    {FFX_NSS_RESOURCE_IDENTIFIER_PREPROCESS_INPUT_TENSOR, L"Resource_0_input"},
    {FFX_NSS_RESOURCE_IDENTIFIER_FEEDBACK_TENSOR, L"Resource_1_output"},
    {FFX_NSS_RESOURCE_IDENTIFIER_K4_TENSOR, L"Resource_2_output"},
    {FFX_NSS_RESOURCE_IDENTIFIER_K3_TENSOR, L"Resource_3_output"},
    {FFX_NSS_RESOURCE_IDENTIFIER_K2_TENSOR, L"Resource_4_output"},
    {FFX_NSS_RESOURCE_IDENTIFIER_K1_TENSOR, L"Resource_5_output"},
    {FFX_NSS_RESOURCE_IDENTIFIER_K0_TENSOR, L"Resource_6_output"},
};

#define FFX_LENGTH(x, y) (sqrt((x) * (x) + (y) * (y)))

static void nssDebugCheckDispatch(FfxNssContext_Private* context, const FfxNssDispatchDescription* params)
{
    if (params->commandList == nullptr)
    {
        context->contextDescription.fpMessage(FFX_MESSAGE_TYPE_ERROR, L"commandList is null");
    }

    if (params->color.resource == nullptr)
    {
        context->contextDescription.fpMessage(FFX_MESSAGE_TYPE_ERROR, L"color resource is null");
    }

    if (params->depth.resource == nullptr)
    {
        context->contextDescription.fpMessage(FFX_MESSAGE_TYPE_ERROR, L"depth resource is null");
    }

    if (params->depthTm1.resource == nullptr)
    {
        context->contextDescription.fpMessage(FFX_MESSAGE_TYPE_ERROR, L"depthTm1 resource is null");
    }

    if (params->motionVectors.resource == nullptr)
    {
        context->contextDescription.fpMessage(FFX_MESSAGE_TYPE_ERROR, L"motionVectors resource is null");
    }

    if (params->output.resource == nullptr)
    {
        context->contextDescription.fpMessage(FFX_MESSAGE_TYPE_ERROR, L"output resource is null");
    }

    if (params->debugViews.resource == nullptr && (params->flags & FFX_NSS_DISPATCH_FLAG_DRAW_DEBUG_VIEW))
    {
        context->contextDescription.fpMessage(FFX_MESSAGE_TYPE_ERROR, L"debugViews resource is null");
    }

    if (fabs(params->jitterOffset.x) > 1.0f || fabs(params->jitterOffset.y) > 1.0f)
    {
        context->contextDescription.fpMessage(FFX_MESSAGE_TYPE_WARNING, L"jitterOffset contains value outside of expected range [-1.0, 1.0]");
    }

    if ((params->renderSize.width == 0) || (params->renderSize.height == 0))
    {
        context->contextDescription.fpMessage(FFX_MESSAGE_TYPE_WARNING, L"renderSize contains zero dimension");
    }

    if (params->upscaleSize.width / params->renderSize.width - 2.0f > FLT_EPSILON)
    {
        context->contextDescription.fpMessage(FFX_MESSAGE_TYPE_WARNING, L"It's recommanded to use upscale ratio less than x2.");
    }

    if ((params->renderSize.width != context->contextDescription.maxRenderSize.width) ||
        (params->renderSize.height != context->contextDescription.maxRenderSize.height))
    {
        context->contextDescription.fpMessage(FFX_MESSAGE_TYPE_WARNING, L"renderSize is different from context maxRenderSize");
    }

    if (fabs(params->motionVectorScale.x) > (float)context->contextDescription.maxRenderSize.width ||
        fabs(params->motionVectorScale.y) > (float)context->contextDescription.maxRenderSize.height)
    {
        context->contextDescription.fpMessage(FFX_MESSAGE_TYPE_WARNING, L"motionVectorScale is greater than context maxRenderSize");
    }

    if ((params->upscaleSize.width == 0) || (params->upscaleSize.height == 0))
    {
        context->contextDescription.fpMessage(FFX_MESSAGE_TYPE_WARNING, L"upscaleSize contains zero dimension");
    }

    if ((params->upscaleSize.width != context->contextDescription.maxUpscaleSize.width) ||
        (params->upscaleSize.height != context->contextDescription.maxUpscaleSize.height))
    {
        context->contextDescription.fpMessage(FFX_MESSAGE_TYPE_WARNING, L"upscaleSize is different from context maxUpscaleSize");
    }

    if (context->paddedInputWidth % FFX_NSS_RESOURCE_ALIGNMENT != 0 || context->paddedInputHeight % FFX_NSS_RESOURCE_ALIGNMENT != 0)
    {
        context->contextDescription.fpMessage(FFX_MESSAGE_TYPE_WARNING, L"padded input size is not aligned to required alignment");
    }
}

static FfxErrorCode patchResourceBindings(FfxPipelineState* inoutPipeline)
{
    for (uint32_t srvIndex = 0; srvIndex < inoutPipeline->srvTextureCount; ++srvIndex)
    {
        int32_t mapIndex = 0;
        for (mapIndex = 0; mapIndex < FFX_COUNTOF(srvTextureBindingTable); ++mapIndex)
        {
            if (0 == wcscmp(srvTextureBindingTable[mapIndex].name, inoutPipeline->srvTextureBindings[srvIndex].name))
                break;
        }
        if (mapIndex == FFX_COUNTOF(srvTextureBindingTable))
            return FFX_ERROR_INVALID_ARGUMENT;

        inoutPipeline->srvTextureBindings[srvIndex].resourceIdentifier = srvTextureBindingTable[mapIndex].index;
    }

    for (uint32_t uavIndex = 0; uavIndex < inoutPipeline->uavTextureCount; ++uavIndex)
    {
        int32_t mapIndex = 0;
        for (mapIndex = 0; mapIndex < FFX_COUNTOF(nssUavTextureBindingTable); ++mapIndex)
        {
            if (0 == wcscmp(nssUavTextureBindingTable[mapIndex].name, inoutPipeline->uavTextureBindings[uavIndex].name))
                break;
        }
        if (mapIndex == FFX_COUNTOF(nssUavTextureBindingTable))
            return FFX_ERROR_INVALID_ARGUMENT;

        inoutPipeline->uavTextureBindings[uavIndex].resourceIdentifier = nssUavTextureBindingTable[mapIndex].index;
    }

    for (uint32_t tensorIndex = 0; tensorIndex < inoutPipeline->srvTensorCount; ++tensorIndex)
    {
        int32_t mapIndex = 0;
        for (mapIndex = 0; mapIndex < FFX_COUNTOF(srvTensorBindingTable); ++mapIndex)
        {
            if (0 == wcscmp(srvTensorBindingTable[mapIndex].name, inoutPipeline->srvTensorBindings[tensorIndex].name))
                break;
        }
        if (mapIndex == FFX_COUNTOF(srvTensorBindingTable))
            return FFX_ERROR_INVALID_ARGUMENT;

        inoutPipeline->srvTensorBindings[tensorIndex].resourceIdentifier = srvTensorBindingTable[mapIndex].index;
    }

    for (uint32_t tensorIndex = 0; tensorIndex < inoutPipeline->uavTensorCount; ++tensorIndex)
    {
        int32_t mapIndex = 0;
        for (mapIndex = 0; mapIndex < FFX_COUNTOF(uavTensorBindingTable); ++mapIndex)
        {
            if (0 == wcscmp(uavTensorBindingTable[mapIndex].name, inoutPipeline->uavTensorBindings[tensorIndex].name))
                break;
        }
        if (mapIndex == FFX_COUNTOF(uavTensorBindingTable))
            return FFX_ERROR_INVALID_ARGUMENT;

        inoutPipeline->uavTensorBindings[tensorIndex].resourceIdentifier = uavTensorBindingTable[mapIndex].index;
    }

    return FFX_OK;
}

static uint32_t getPipelinePermutationFlags(FfxNssContext_Private* context, const float upscaleRatio)
{
    FFX_ASSERT(context);

    const uint32_t contextFlags  = context->contextDescription.flags;
    const bool     fp16Supported = context->deviceCapabilities.fp16Supported;

    uint32_t flags = 0;
    flags |= (contextFlags & FFX_NSS_CONTEXT_FLAG_QUANTIZED) ? NSS_SHADER_PERMUTATION_QUANTIZED : 0;
    flags |= (contextFlags & FFX_NSS_CONTEXT_FLAG_DEPTH_INVERTED) ? NSS_SHADER_PERMUTATION_REVERSE_Z : 0;
    flags |= (contextFlags & FFX_NSS_CONTEXT_FLAG_RESAMPLE_BICUBIC) ? NSS_SHADER_PERMUTATION_RESAMPLE_BICUBIC : 0;
    flags |= (contextFlags & FFX_NSS_CONTEXT_FLAG_READ_TENSORS_AS_IMAGES) ? NSS_SHADER_PERMUTATION_ALIAS_OUTPUT_TENSORS_AS_IMAGES : 0;

    const bool require16bit = (contextFlags & FFX_NSS_CONTEXT_FLAG_ALLOW_16BIT) != 0;
    if (require16bit)
    {
        if (fp16Supported)
        {
            flags |= NSS_SHADER_PERMUTATION_ALLOW_16BIT;
        }
        else
        {
            context->contextDescription.fpMessage(FFX_MESSAGE_TYPE_WARNING, L"16-bit resources requested but not supported by the device.");
        }
    }

    // If the upscale ratio matches our scale preset, we use preset mode for better performance
    if (fabs(upscaleRatio - 2.0f) < SCALE_PRESET_MODE_THRESHOLD)
    {
        // Only support preset upscale ratio 2.0x for now.
        // Other ratios will use general path.
        flags |= NSS_SHADER_PERMUTATION_SCALE_PRESET_MODE_X2;
    }

    return flags;
}

static FfxErrorCode createPipelineStates(FfxNssContext_Private* context)
{
    FFX_ASSERT(context);

    const uint32_t width        = context->paddedInputWidth;
    const uint32_t height       = context->paddedInputHeight;
    const float    upscaleRatio = static_cast<float>(context->paddedOutputWidth) / static_cast<float>(context->paddedInputWidth);

    FfxPipelineDescription pipelineDescription = {};
    pipelineDescription.contextFlags           = context->contextDescription.flags;

    // Samplers
    pipelineDescription.samplerCount      = 2;
    FfxSamplerDescription samplerDescs[2] = {
        {FFX_FILTER_TYPE_MINMAGMIP_POINT, FFX_ADDRESS_MODE_CLAMP, FFX_ADDRESS_MODE_CLAMP, FFX_ADDRESS_MODE_CLAMP, FFX_BIND_COMPUTE_SHADER_STAGE},
        {FFX_FILTER_TYPE_MINMAGMIP_LINEAR, FFX_ADDRESS_MODE_CLAMP, FFX_ADDRESS_MODE_CLAMP, FFX_ADDRESS_MODE_CLAMP, FFX_BIND_COMPUTE_SHADER_STAGE}};
    pipelineDescription.samplers = samplerDescs;

    // Set up pipeline descriptor (basically RootSignature and binding)
    uint32_t pipelineFlags = getPipelinePermutationFlags(context, upscaleRatio);

    // Root constants
    pipelineDescription.rootConstantBufferCount     = 1;
    FfxRootConstantDescription rootConstantDescs[1] = {{sizeof(NssConstants) / sizeof(uint32_t), FFX_BIND_COMPUTE_SHADER_STAGE}};
    pipelineDescription.rootConstants               = rootConstantDescs;

    const auto CreatePipeline = [&](FfxPass pass, const wchar_t* name, FfxPipelineState* pipeline) -> FfxErrorCode {
        ffxSafeReleasePipeline(&context->contextDescription.backendInterface, pipeline, context->effectContextId);

        wcscpy_s(pipelineDescription.name, name);

        FFX_VALIDATE(context->contextDescription.backendInterface.fpCreatePipeline(
            &context->contextDescription.backendInterface, FFX_EFFECT_NSS, pass, pipelineFlags, &pipelineDescription, context->effectContextId, pipeline));
        patchResourceBindings(pipeline);

        return FFX_OK;
    };

    if (context->hasPaddingPass)
    {
        FFX_VALIDATE(CreatePipeline(FFX_NSS_PASS_MIRROR_PADDING, L"NSS-MirrorPadding", &context->pipelineNssMirrorPadding));
    }
    FFX_VALIDATE(CreatePipeline(FFX_NSS_PASS_PREPROCESS, L"NSS-Preprocess", &context->pipelineNssPreprocess));
    FFX_VALIDATE(CreatePipeline(FFX_NSS_PASS_POSTPROCESS, L"NSS-Postprocess", &context->pipelineNssPostprocess));
    FFX_VALIDATE(CreatePipeline(FFX_NSS_PASS_DEBUG_VIEW, L"NSS-DebugView", &context->pipelineNssDebugView));

    // DATA GRAPH
    ffxSafeReleasePipeline(&context->contextDescription.backendInterface, &context->pipelineNssDataGraph, context->effectContextId);
    wcscpy(pipelineDescription.name, L"NSS-Graph");
    FFX_ASSERT_MESSAGE(width % FFX_NSS_RESOURCE_ALIGNMENT == 0 && height % FFX_NSS_RESOURCE_ALIGNMENT == 0,
                       "The NSS algorithm requires the input resolution must be 8 aligned!");
    FFX_VALIDATE(context->contextDescription.backendInterface.fpCreateDataGraphPipeline(&context->contextDescription.backendInterface,
                                                                                        FFX_EFFECT_NSS,
                                                                                        FFX_NSS_PASS_DATA_GRAPH,
                                                                                        pipelineFlags,
                                                                                        &pipelineDescription,
                                                                                        context->effectContextId,
                                                                                        width,
                                                                                        height,
                                                                                        &context->pipelineNssDataGraph));
    patchResourceBindings(&context->pipelineNssDataGraph);
    // END DATA GRAPH

    return FFX_OK;
}

static FfxErrorCode createResourceFromDescription(FfxNssContext_Private* context, const FfxInternalResourceDescription* resDesc)
{
    const FfxResourceType        resourceType        = resDesc->type;
    const FfxResourceDescription resourceDescription = {resourceType,
                                                        resDesc->format,
                                                        resDesc->width,
                                                        resDesc->height,
                                                        (resourceType == FFX_RESOURCE_TYPE_TENSOR) ? resDesc->channel : 1,
                                                        resDesc->mipCount,
                                                        resDesc->flags,
                                                        resDesc->usage,
                                                        resDesc->batchSize,
                                                        resDesc->shapeSize};
    const FfxResourceStates      initialState =
        (resDesc->usage == FFX_RESOURCE_USAGE_READ_ONLY) ? FFX_RESOURCE_STATE_COMPUTE_READ : FFX_RESOURCE_STATE_UNORDERED_ACCESS;
    const FfxCreateResourceDescription createResourceDescription = {
        FFX_HEAP_TYPE_DEFAULT, resourceDescription, initialState, resDesc->name, resDesc->id, resDesc->initData};
    return context->contextDescription.backendInterface.fpCreateResource(
        &context->contextDescription.backendInterface, &createResourceDescription, context->effectContextId, &context->srvResources[resDesc->id]);
}

/// Computes the padded input and output resolutions based on unpadded resolutions.
///
/// @param unpaddedInputWidth      Unpadded input width.
/// @param unpaddedInputHeight     Unpadded input height.
/// @param unpaddedOutputWidth     Unpadded output width.
/// @param unpaddedOutputHeight    Unpadded output height.
/// @param paddedInputWidth        [out] Padded input width.
/// @param paddedInputHeight       [out] Padded input height.
/// @param paddedOutputWidth       [out] Padded output width.
/// @param paddedOutputHeight      [out] Padded output height.
///
/// @return True if any of the padded dimensions differ from the unpadded dimensions, false otherwise.
static bool ComputePaddedResolution(const uint32_t unpaddedInputWidth,
                                    const uint32_t unpaddedInputHeight,
                                    const uint32_t unpaddedOutputWidth,
                                    const uint32_t unpaddedOutputHeight,
                                    uint32_t&      paddedInputWidth,
                                    uint32_t&      paddedInputHeight,
                                    uint32_t&      paddedOutputWidth,
                                    uint32_t&      paddedOutputHeight)
{
    paddedInputWidth           = FFX_ALIGN_UP(unpaddedInputWidth, FFX_NSS_RESOURCE_ALIGNMENT);
    paddedInputHeight          = FFX_ALIGN_UP(unpaddedInputHeight, FFX_NSS_RESOURCE_ALIGNMENT);
    const float upscaleFactorX = static_cast<float>(unpaddedOutputWidth) / static_cast<float>(unpaddedInputWidth);
    const float upscaleFactorY = static_cast<float>(unpaddedOutputHeight) / static_cast<float>(unpaddedInputHeight);

    // TODO: does the output require the alignment?
    paddedOutputWidth  = FFX_ALIGN_UP(static_cast<uint32_t>(paddedInputWidth * upscaleFactorX), FFX_NSS_RESOURCE_ALIGNMENT);
    paddedOutputHeight = FFX_ALIGN_UP(static_cast<uint32_t>(paddedInputHeight * upscaleFactorY), FFX_NSS_RESOURCE_ALIGNMENT);

    return paddedInputWidth != unpaddedInputWidth || paddedInputHeight != unpaddedInputHeight || paddedOutputWidth != unpaddedOutputWidth ||
           paddedOutputHeight != unpaddedOutputHeight;
}

static FfxErrorCode nssCreate(FfxNssContext_Private* context, const FfxNssContextDescription* contextDescription)
{
    FFX_ASSERT(context);
    FFX_ASSERT(contextDescription);

    // Setup the data for implementation.
    memset(context, 0, sizeof(FfxNssContext_Private));
    context->device = contextDescription->backendInterface.device;

    memcpy(&context->contextDescription, contextDescription, sizeof(FfxNssContextDescription));

    // Create the context.
    FfxErrorCode errorCode = context->contextDescription.backendInterface.fpCreateBackendContext(
        &context->contextDescription.backendInterface, FFX_EFFECT_NSS, nullptr, &context->effectContextId);
    FFX_RETURN_ON_ERROR(errorCode == FFX_OK, errorCode);

    // call out for device caps.
    errorCode =
        context->contextDescription.backendInterface.fpGetDeviceCapabilities(&context->contextDescription.backendInterface, &context->deviceCapabilities);
    FFX_RETURN_ON_ERROR(errorCode == FFX_OK, errorCode);

    const bool neuralGraphicsSupported = context->deviceCapabilities.tensorSupported && context->deviceCapabilities.dataGraphSupported;
    if (!neuralGraphicsSupported)
    {
        context->contextDescription.fpMessage(FFX_MESSAGE_TYPE_ERROR,
                                              L"NSS requires device with support for tensors and data graphs. "
                                              L"Please check device capabilities.");
        return FFX_ERROR_NULL_DEVICE;
    }

    // set defaults
    context->firstExecution     = true;
    context->resourceFrameIndex = 0;
    const bool needPaddingPass  = ComputePaddedResolution(contextDescription->maxRenderSize.width,
                                                         contextDescription->maxRenderSize.height,
                                                         contextDescription->maxUpscaleSize.width,
                                                         contextDescription->maxUpscaleSize.height,
                                                         context->paddedInputWidth,
                                                         context->paddedInputHeight,
                                                         context->paddedOutputWidth,
                                                         context->paddedOutputHeight);
    const bool hasPaddingFlag   = (context->contextDescription.flags & FFX_NSS_CONTEXT_FLAG_DISABLE_PADDING) == 0;
    context->hasPaddingPass     = hasPaddingFlag && needPaddingPass;

    // NOTE: This will not work for RHI-NNE Backend!
    FfxSurfaceFormat tensorFormatSingleChannel = ((contextDescription->flags & FFX_NSS_CONTEXT_FLAG_QUANTIZED) == FFX_NSS_CONTEXT_FLAG_QUANTIZED)
                                                     ? FFX_SURFACE_FORMAT_R8_SINT
                                                     : FFX_SURFACE_FORMAT_R32_FLOAT;

    // NOTE: This will not work for RHI-NNE Backend!
    FfxSurfaceFormat tensorFormatQuadChannel = ((contextDescription->flags & FFX_NSS_CONTEXT_FLAG_QUANTIZED) == FFX_NSS_CONTEXT_FLAG_QUANTIZED)
                                                   ? FFX_SURFACE_FORMAT_R8_SINT
                                                   : FFX_SURFACE_FORMAT_R32_FLOAT;

    const bool aliasTensorAsImage = (contextDescription->flags & FFX_NSS_CONTEXT_FLAG_READ_TENSORS_AS_IMAGES) == FFX_NSS_CONTEXT_FLAG_READ_TENSORS_AS_IMAGES;

    uint32_t           inputTensorChannel        = 12;
    constexpr uint32_t feedbackTensorChannel     = 4;
    constexpr uint32_t coefficientsTensorChannel = 4;
    // declare internal resources needed
    const FfxInternalResourceDescription internalSurfaceDesc[] = {
        {FFX_NSS_RESOURCE_IDENTIFIER_PREPROCESS_INPUT_TENSOR,
         L"NSS_PreprocessInputTensor",
         FFX_RESOURCE_TYPE_TENSOR,
         FFX_RESOURCE_USAGE_UAV,
         tensorFormatSingleChannel,
         context->paddedInputWidth,
         context->paddedInputHeight,
         1,
         FFX_RESOURCE_FLAGS_NONE,
         {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED},
         1,
         inputTensorChannel,
         4},

        {FFX_NSS_RESOURCE_IDENTIFIER_LUMA_DERIV_1,
         L"NSS_LumaDeriv_1",
         FFX_RESOURCE_TYPE_TEXTURE2D,
         (FfxResourceUsage)(FFX_RESOURCE_USAGE_RENDERTARGET | FFX_RESOURCE_USAGE_UAV),
         FFX_SURFACE_FORMAT_R8G8_UNORM,
         context->paddedInputWidth,
         context->paddedInputHeight,
         1,
         FFX_RESOURCE_FLAGS_NONE,
         {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED}},

        {FFX_NSS_RESOURCE_IDENTIFIER_LUMA_DERIV_2,
         L"NSS_LumaDeriv_2",
         FFX_RESOURCE_TYPE_TEXTURE2D,
         (FfxResourceUsage)(FFX_RESOURCE_USAGE_RENDERTARGET | FFX_RESOURCE_USAGE_UAV),
         FFX_SURFACE_FORMAT_R8G8_UNORM,
         context->paddedInputWidth,
         context->paddedInputHeight,
         1,
         FFX_RESOURCE_FLAGS_NONE,
         {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED}},

        {FFX_NSS_RESOURCE_IDENTIFIER_NEAREST_DEPTH_COORD_1,
         L"NSS_DepthOffset_1",
         FFX_RESOURCE_TYPE_TEXTURE2D,
         (FfxResourceUsage)(FFX_RESOURCE_USAGE_RENDERTARGET | FFX_RESOURCE_USAGE_UAV),
         FFX_SURFACE_FORMAT_R8_UNORM,
         context->paddedInputWidth,
         context->paddedInputHeight,
         1,
         FFX_RESOURCE_FLAGS_NONE,
         {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED}},

        {FFX_NSS_RESOURCE_IDENTIFIER_NEAREST_DEPTH_COORD_2,
         L"NSS_DepthOffset_2",
         FFX_RESOURCE_TYPE_TEXTURE2D,
         (FfxResourceUsage)(FFX_RESOURCE_USAGE_RENDERTARGET | FFX_RESOURCE_USAGE_UAV),
         FFX_SURFACE_FORMAT_R8_UNORM,
         context->paddedInputWidth,
         context->paddedInputHeight,
         1,
         FFX_RESOURCE_FLAGS_NONE,
         {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED}},

        {FFX_NSS_RESOURCE_IDENTIFIER_FEEDBACK_TENSOR_1,
         L"NSS_FeedbackTensor_1",
         FFX_RESOURCE_TYPE_TENSOR,
         FFX_RESOURCE_USAGE_UAV,
         tensorFormatSingleChannel,
         context->paddedInputWidth,
         context->paddedInputHeight,
         1,
         aliasTensorAsImage ? FFX_RESOURCE_FLAGS_IMAGE_ALIASED : FFX_RESOURCE_FLAGS_NONE,
         {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED},
         1,
         feedbackTensorChannel,
         4},

        {FFX_NSS_RESOURCE_IDENTIFIER_FEEDBACK_TENSOR_2,
         L"NSS_FeedbackTensor_2",
         FFX_RESOURCE_TYPE_TENSOR,
         FFX_RESOURCE_USAGE_UAV,
         tensorFormatSingleChannel,
         context->paddedInputWidth,
         context->paddedInputHeight,
         1,
         aliasTensorAsImage ? FFX_RESOURCE_FLAGS_IMAGE_ALIASED : FFX_RESOURCE_FLAGS_NONE,
         {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED},
         1,
         feedbackTensorChannel,
         4},

        {FFX_NSS_RESOURCE_IDENTIFIER_K0_TENSOR,
         L"NSS_CoefficientsTensor_0",
         FFX_RESOURCE_TYPE_TENSOR,
         FFX_RESOURCE_USAGE_UAV,
         tensorFormatQuadChannel,
         context->paddedInputWidth,
         context->paddedInputHeight,
         1,
         aliasTensorAsImage ? FFX_RESOURCE_FLAGS_IMAGE_ALIASED : FFX_RESOURCE_FLAGS_NONE,
         {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED},
         1,
         coefficientsTensorChannel,
         4},

        {FFX_NSS_RESOURCE_IDENTIFIER_K1_TENSOR,
         L"NSS_CoefficientsTensor_1",
         FFX_RESOURCE_TYPE_TENSOR,
         FFX_RESOURCE_USAGE_UAV,
         tensorFormatQuadChannel,
         context->paddedInputWidth,
         context->paddedInputHeight,
         1,
         aliasTensorAsImage ? FFX_RESOURCE_FLAGS_IMAGE_ALIASED : FFX_RESOURCE_FLAGS_NONE,
         {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED},
         1,
         coefficientsTensorChannel,
         4},

        {FFX_NSS_RESOURCE_IDENTIFIER_K2_TENSOR,
         L"NSS_CoefficientsTensor_2",
         FFX_RESOURCE_TYPE_TENSOR,
         FFX_RESOURCE_USAGE_UAV,
         tensorFormatQuadChannel,
         context->paddedInputWidth,
         context->paddedInputHeight,
         1,
         aliasTensorAsImage ? FFX_RESOURCE_FLAGS_IMAGE_ALIASED : FFX_RESOURCE_FLAGS_NONE,
         {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED},
         1,
         coefficientsTensorChannel,
         4},

        {FFX_NSS_RESOURCE_IDENTIFIER_K3_TENSOR,
         L"NSS_CoefficientsTensor_3",
         FFX_RESOURCE_TYPE_TENSOR,
         FFX_RESOURCE_USAGE_UAV,
         tensorFormatQuadChannel,
         context->paddedInputWidth,
         context->paddedInputHeight,
         1,
         aliasTensorAsImage ? FFX_RESOURCE_FLAGS_IMAGE_ALIASED : FFX_RESOURCE_FLAGS_NONE,
         {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED},
         1,
         coefficientsTensorChannel,
         4},

        {FFX_NSS_RESOURCE_IDENTIFIER_K4_TENSOR,
         L"NSS_CoefficientsTensor_4",
         FFX_RESOURCE_TYPE_TENSOR,
         FFX_RESOURCE_USAGE_UAV,
         tensorFormatQuadChannel,
         context->paddedInputWidth,
         context->paddedInputHeight,
         1,
         aliasTensorAsImage ? FFX_RESOURCE_FLAGS_IMAGE_ALIASED : FFX_RESOURCE_FLAGS_NONE,
         {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED},
         1,
         coefficientsTensorChannel,
         4},
    };

    // clear the SRV resources to NULL.
    memset(context->srvResources, 0, sizeof(context->srvResources));

    // Generally used resources by all presets
    for (int32_t currentSurfaceIndex = 0; currentSurfaceIndex < FFX_ARRAY_ELEMENTS(internalSurfaceDesc); ++currentSurfaceIndex)
    {
        FFX_VALIDATE(createResourceFromDescription(context, &internalSurfaceDesc[currentSurfaceIndex]));
    }

    if (context->hasPaddingPass)
    {
        const FfxInternalResourceDescription mirrorPaddingInternalSurfaceDesc[] = {
            {FFX_NSS_RESOURCE_IDENTIFIER_INPUT_COLOR,
             L"NSS_padded_input_color_jittered",
             FFX_RESOURCE_TYPE_TEXTURE2D,
             (FfxResourceUsage)(FFX_RESOURCE_USAGE_RENDERTARGET | FFX_RESOURCE_USAGE_UAV),
             FFX_SURFACE_FORMAT_R11G11B10_FLOAT,
             context->paddedInputWidth,
             context->paddedInputHeight,
             1,
             FFX_RESOURCE_FLAGS_NONE,
             {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED}},

            {FFX_NSS_RESOURCE_IDENTIFIER_INPUT_DEPTH,
             L"NSS_padded_input_depth",
             FFX_RESOURCE_TYPE_TEXTURE2D,
             (FfxResourceUsage)(FFX_RESOURCE_USAGE_RENDERTARGET | FFX_RESOURCE_USAGE_UAV),
             FFX_SURFACE_FORMAT_R32_FLOAT,
             context->paddedInputWidth,
             context->paddedInputHeight,
             1,
             FFX_RESOURCE_FLAGS_NONE,
             {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED}},

            {FFX_NSS_RESOURCE_IDENTIFIER_INPUT_DEPTH_TM1,
             L"NSS_padded_input_depth_tm1",
             FFX_RESOURCE_TYPE_TEXTURE2D,
             (FfxResourceUsage)(FFX_RESOURCE_USAGE_RENDERTARGET | FFX_RESOURCE_USAGE_UAV),
             FFX_SURFACE_FORMAT_R32_FLOAT,
             context->paddedInputWidth,
             context->paddedInputHeight,
             1,
             FFX_RESOURCE_FLAGS_NONE,
             {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED}},

            {FFX_NSS_RESOURCE_IDENTIFIER_INPUT_MOTION_VECTORS,
             L"NSS_padded_input_motion_vectors",
             FFX_RESOURCE_TYPE_TEXTURE2D,
             (FfxResourceUsage)(FFX_RESOURCE_USAGE_RENDERTARGET | FFX_RESOURCE_USAGE_UAV),
             FFX_SURFACE_FORMAT_R16G16_FLOAT,
             context->paddedInputWidth,
             context->paddedInputHeight,
             1,
             FFX_RESOURCE_FLAGS_NONE,
             {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED}},

            {FFX_NSS_RESOURCE_IDENTIFIER_PADDED_OUTPUT_1,
             L"NSS_padded_upscaled_color_1",
             FFX_RESOURCE_TYPE_TEXTURE2D,
             (FfxResourceUsage)(FFX_RESOURCE_USAGE_RENDERTARGET | FFX_RESOURCE_USAGE_UAV),
             FFX_SURFACE_FORMAT_R11G11B10_FLOAT,
             context->paddedOutputWidth,
             context->paddedOutputHeight,
             1,
             FFX_RESOURCE_FLAGS_NONE,
             {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED}},

            {FFX_NSS_RESOURCE_IDENTIFIER_PADDED_OUTPUT_2,
             L"NSS_padded_upscaled_color_2",
             FFX_RESOURCE_TYPE_TEXTURE2D,
             (FfxResourceUsage)(FFX_RESOURCE_USAGE_RENDERTARGET | FFX_RESOURCE_USAGE_UAV),
             FFX_SURFACE_FORMAT_R11G11B10_FLOAT,
             context->paddedOutputWidth,
             context->paddedOutputHeight,
             1,
             FFX_RESOURCE_FLAGS_NONE,
             {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED}},
        };

        for (int32_t currentSurfaceIndex = 0; currentSurfaceIndex < FFX_ARRAY_ELEMENTS(mirrorPaddingInternalSurfaceDesc); ++currentSurfaceIndex)
        {
            FFX_VALIDATE(createResourceFromDescription(context, &mirrorPaddingInternalSurfaceDesc[currentSurfaceIndex]));
        }
    }

    // copy resources to uavResrouces list
    memcpy(context->uavResources, context->srvResources, sizeof(context->srvResources));
    //memcpy(context->tensorResources, context->srvResources, sizeof(context->srvResources));

    // avoid compiling pipelines on first render
    {
        errorCode = createPipelineStates(context);
        FFX_RETURN_ON_ERROR(errorCode == FFX_OK, errorCode);
    }
    return FFX_OK;
}

static FfxErrorCode nssRelease(FfxNssContext_Private* context)
{
    FFX_ASSERT(context);

    if (context->hasPaddingPass)
    {
        ffxSafeReleasePipeline(&context->contextDescription.backendInterface, &context->pipelineNssMirrorPadding, context->effectContextId);
    }
    ffxSafeReleasePipeline(&context->contextDescription.backendInterface, &context->pipelineNssPreprocess, context->effectContextId);
    ffxSafeReleasePipeline(&context->contextDescription.backendInterface, &context->pipelineNssDataGraph, context->effectContextId);
    ffxSafeReleasePipeline(&context->contextDescription.backendInterface, &context->pipelineNssPostprocess, context->effectContextId);
    ffxSafeReleasePipeline(&context->contextDescription.backendInterface, &context->pipelineNssDebugView, context->effectContextId);

    // release internal resources
    for (int32_t currentResourceIndex = 0; currentResourceIndex < FFX_NSS_RESOURCE_IDENTIFIER_COUNT; ++currentResourceIndex)
    {
        ffxSafeReleaseResource(&context->contextDescription.backendInterface, context->srvResources[currentResourceIndex], context->effectContextId);
    }

    // unregister resources not created internally
    constexpr uint32_t external_resources[] = {
        FFX_NSS_RESOURCE_IDENTIFIER_UNPADDED_COLOR,
        FFX_NSS_RESOURCE_IDENTIFIER_UNPADDED_DEPTH,
        FFX_NSS_RESOURCE_IDENTIFIER_UNPADDED_DEPTH_TM1,
        FFX_NSS_RESOURCE_IDENTIFIER_UNPADDED_MOTION,
        FFX_NSS_RESOURCE_IDENTIFIER_UNPADDED_OUTPUT,
        FFX_NSS_RESOURCE_IDENTIFIER_DEBUG_VIEWS,
        // the following resources will be created internally if padding pass is used, otherwise they are created externally
        FFX_NSS_RESOURCE_IDENTIFIER_INPUT_COLOR,
        FFX_NSS_RESOURCE_IDENTIFIER_INPUT_DEPTH,
        FFX_NSS_RESOURCE_IDENTIFIER_INPUT_DEPTH_TM1,
        FFX_NSS_RESOURCE_IDENTIFIER_INPUT_MOTION_VECTORS,
        FFX_NSS_RESOURCE_IDENTIFIER_UPSCALED_OUTPUT,
    };
    for (uint32_t i = 0; i < FFX_COUNTOF(external_resources); ++i)
    {
        context->srvResources[external_resources[i]] = {FFX_NSS_RESOURCE_IDENTIFIER_NULL};
        context->uavResources[external_resources[i]] = {FFX_NSS_RESOURCE_IDENTIFIER_NULL};
    }

    // Destroy the context
    context->contextDescription.backendInterface.fpDestroyBackendContext(&context->contextDescription.backendInterface, context->effectContextId);

    return FFX_OK;
}

// Convert float32 to float16 (IEEE 754 half-precision)
static uint16_t packfloat32ToUint16(float value)
{
    FfxUInt32 bits;
    std::memcpy(&bits, &value, sizeof(bits));

    FfxUInt32 sign     = (bits >> 31) & 0x1;
    FfxInt32  exponent = ((bits >> 23) & 0xFF) - 127 + 15;
    FfxUInt32 mantissa = bits & 0x7FFFFF;

    if (exponent <= 0)
    {
        // Subnormal or zero
        return static_cast<FfxUInt16>(sign << 15);
    }
    else if (exponent >= 31)
    {
        // Inf or NaN
        return static_cast<FfxUInt16>((sign << 15) | (0x1F << 10));
    }

    // Round mantissa from 23 to 10 bits
    mantissa = mantissa >> 13;
    return static_cast<FfxUInt16>((sign << 15) | (exponent << 10) | mantissa);
}

// Pack two float32 values into one uint32 as float16
static FfxUInt32 packTwoFloatsTo32bit(FfxFloat32 a, FfxFloat32 b)
{
    FfxUInt16 a16 = packfloat32ToUint16(a);
    FfxUInt16 b16 = packfloat32ToUint16(b);
    return (static_cast<FfxUInt32>(b16) << 16) | a16;
}

static FfxUInt32 packTwoUintsTo32bit(FfxUInt32 a, FfxUInt32 b)
{
    FfxUInt16 a16 = static_cast<FfxUInt16>(a);
    FfxUInt16 b16 = static_cast<FfxUInt16>(b);
    return (static_cast<FfxUInt32>(b16) << 16) | a16;
}

static void setupDeviceDepthToViewSpaceDepthParams(FfxNssContext_Private* context, const FfxNssDispatchDescription* params)
{
    const bool bInverted = (context->contextDescription.flags & FFX_NSS_CONTEXT_FLAG_DEPTH_INVERTED) == FFX_NSS_CONTEXT_FLAG_DEPTH_INVERTED;
    const bool bInfinite = (context->contextDescription.flags & FFX_NSS_CONTEXT_FLAG_DEPTH_INFINITE) == FFX_NSS_CONTEXT_FLAG_DEPTH_INFINITE;

    // make sure it has no impact if near and far plane values are swapped in dispatch params
    // the flags "inverted" and "infinite" will decide what transform to use
    float fMin = FFX_MINIMUM(params->cameraNear, params->cameraFar);
    float fMax = FFX_MAXIMUM(params->cameraNear, params->cameraFar);

    if (bInverted)
    {
        float tmp = fMin;
        fMin      = fMax;
        fMax      = tmp;
    }

    // a 0 0 0   x
    // 0 b 0 0   y
    // 0 0 c d   z
    // 0 0 e 0   1

    const float fQ = fMax / (fMin - fMax);
    const float d  = -1.0f;  // for clarity

    const float matrix_elem_c[2][2] = {
        fQ,                   // non reversed, non infinite
        -1.0f - FLT_EPSILON,  // non reversed, infinite
        fQ,                   // reversed, non infinite
        0.0f + FLT_EPSILON    // reversed, infinite
    };

    const float matrix_elem_e[2][2] = {
        fQ * fMin,            // non reversed, non infinite
        -fMin - FLT_EPSILON,  // non reversed, infinite
        fQ * fMin,            // reversed, non infinite
        fMax,                 // reversed, infinite
    };

    context->constants._DeviceToViewDepth[0] = d * matrix_elem_c[bInverted][bInfinite];
    context->constants._DeviceToViewDepth[1] = matrix_elem_e[bInverted][bInfinite];

    // revert x and y coords
    const float aspect      = context->paddedInputWidth / float(context->paddedInputHeight);
    const float cotHalfFovY = cosf(0.5f * params->cameraFovAngleVertical) / sinf(0.5f * params->cameraFovAngleVertical);
    const float a           = cotHalfFovY / aspect;
    const float b           = cotHalfFovY;

    context->constants._DeviceToViewDepth[2] = (1.0f / a);
    context->constants._DeviceToViewDepth[3] = (1.0f / b);
}

static void computeJitterTileOffset(float jx, float jy, float sx, float sy, int mx, int my, FfxInt32x2& jitterTileOffset)
{
    // Project base and jittered LR pixel centers to HR index space
    int base_hr_x = static_cast<int>(std::floor(0.5f * sx));
    int base_hr_y = static_cast<int>(std::floor(0.5f * sy));

    int jittered_hr_x = static_cast<int>(std::floor((jx + 0.5f) * sx));
    int jittered_hr_y = static_cast<int>(std::floor((jy + 0.5f) * sy));

    int dx_offset = (jittered_hr_x - base_hr_x) % mx;
    if (dx_offset < 0)
        dx_offset += mx;
    int dy_offset = (jittered_hr_y - base_hr_y) % my;
    if (dy_offset < 0)
        dy_offset += my;

    jitterTileOffset[0] = dx_offset;
    jitterTileOffset[1] = dy_offset;
}

static bool NeedResetHistory(FfxNssContext_Private* context, const FfxNssDispatchDescription* params)
{
    /* Assume the application accounts for scenarios requiring a reset (eg. due to changes in render size,
	 * camera transitions, etc.), and correctly configures this through the reset parameter.
	 */
    return context->firstExecution || params->reset;
}

static void setupConstantBuffer(FfxNssContext_Private* context, const FfxNssDispatchDescription* params, bool use16bit)
{
    NssConstants& constants        = context->constants;
    const bool    needResetHistory = NeedResetHistory(context, params);
    if (needResetHistory)
    {
        memset(&constants, 0, sizeof(constants));
    }

    // compute params to enable device depth to view space depth computation in shader
    setupDeviceDepthToViewSpaceDepthParams(context, params);

    constants._InputDims[0]    = context->paddedInputWidth;
    constants._InputDims[1]    = context->paddedInputHeight;
    constants._InvInputDims[0] = 1.f / static_cast<float>(context->paddedInputWidth);
    constants._InvInputDims[1] = 1.f / static_cast<float>(context->paddedInputHeight);

    constants._OutputDims[0]    = context->paddedOutputWidth;
    constants._OutputDims[1]    = context->paddedOutputHeight;
    constants._InvOutputDims[0] = 1.f / static_cast<float>(context->paddedOutputWidth);
    constants._InvOutputDims[1] = 1.f / static_cast<float>(context->paddedOutputHeight);

    constants._UnpaddedInputDims[0] = params->renderSize.width;
    constants._UnpaddedInputDims[1] = params->renderSize.height;

    // The passed in jitter offset is in pixel space of unpadded render size
    const float jitterUvX = params->jitterOffset.x / static_cast<float>(params->renderSize.width);
    const float jitterUvY = params->jitterOffset.y / static_cast<float>(params->renderSize.height);
    // JitterOffset in pixels
    constants._JitterOffsetTm1[0] = constants._JitterOffset[0];
    constants._JitterOffsetTm1[1] = constants._JitterOffset[1];
    constants._JitterOffset[0]    = jitterUvX * static_cast<float>(context->paddedInputWidth);
    constants._JitterOffset[1]    = jitterUvY * static_cast<float>(context->paddedInputHeight);
    // JitterOffset in uv space
    constants._JitterOffsetTm1[2] = constants._JitterOffset[2];
    constants._JitterOffsetTm1[3] = constants._JitterOffset[3];
    constants._JitterOffset[2]    = jitterUvX;
    constants._JitterOffset[3]    = jitterUvY;

    // Setup scale factor (.xy scale, .zw inv_scale)
    constants._ScaleFactor[0] = constants._OutputDims[0] / FfxFloat32(constants._InputDims[0]);
    constants._ScaleFactor[1] = constants._OutputDims[1] / FfxFloat32(constants._InputDims[1]);
    constants._ScaleFactor[2] = constants._InputDims[0] / FfxFloat32(constants._OutputDims[0]);
    constants._ScaleFactor[3] = constants._InputDims[1] / FfxFloat32(constants._OutputDims[1]);

    // Setup motion vector scale. The passed in motion vectors are in pixel space of unpadded render size.
    constants._MotionVectorScale[0] = params->motionVectorScale.x / params->renderSize.width;
    constants._MotionVectorScale[1] = params->motionVectorScale.y / params->renderSize.height;

    // These quantize parameters are copied from the metadata of the using model.
    // Note that quantParamsSNORM.xy is not used because the input tensor’s shape cannot be aliased
    // through a single image. Therefore, we consistently use the int format for the input.
    constexpr FfxFloat32x4 quantParamsSNORM = {1.0f / 0.49803924513980746, -1.0078740157480315, 0.49999999813735485, -1.0f};
    constexpr FfxFloat32x4 quantParamsSINT  = {1.0f / 0.003921568859368563, -128.0f, 0.003937007859349251, -127.0f};
    const float            noHistoryReset   = needResetHistory ? 0.0f : 1.0f;
    const float            exposure         = (params->exposure <= 0.0f) ? 1.0 : params->exposure;
    const float            invExposure      = 1.f / exposure;

    // These value comes from arm-model-zoo/models/super_sampling/ClampNet_v5_1/scenarios/1920x1088/end_to_end/push_consts_generator.py
    // They are const or learned during training. Cannot be changed.
    const float       motionVectorThreshold       = 0.01f;
    const float       motionDisocclusionThreshold = 0.25f;
    const float       disocclusionScale           = 0.725223958492279f;
    const FfxUInt32x2 indexModulo                 = {2, 2};  // Hardcode to use 2x2 tile size.

    FfxInt32x2 jitterTileOffset;
    computeJitterTileOffset(constants._JitterOffset[0],
                            constants._JitterOffset[1],
                            constants._ScaleFactor[0],
                            constants._ScaleFactor[1],
                            indexModulo[0],
                            indexModulo[1],
                            jitterTileOffset);

    if (use16bit)
    {
        constants.dynamicPrecision._16bit._QuantParamsSNORM[0]   = packTwoFloatsTo32bit(quantParamsSNORM[0], quantParamsSNORM[1]);
        constants.dynamicPrecision._16bit._QuantParamsSNORM[1]   = packTwoFloatsTo32bit(quantParamsSNORM[2], quantParamsSNORM[3]);
        constants.dynamicPrecision._16bit._QuantParamsSINT[0]    = packTwoFloatsTo32bit(quantParamsSINT[0], quantParamsSINT[1]);
        constants.dynamicPrecision._16bit._QuantParamsSINT[1]    = packTwoFloatsTo32bit(quantParamsSINT[2], quantParamsSINT[3]);
        constants.dynamicPrecision._16bit._Exposure              = packTwoFloatsTo32bit(exposure, invExposure);
        constants.dynamicPrecision._16bit._MotionDisThreshPad[0] = packTwoFloatsTo32bit(motionVectorThreshold, motionDisocclusionThreshold);
        constants.dynamicPrecision._16bit._MotionDisThreshPad[1] = packTwoFloatsTo32bit(disocclusionScale, 0.0f);
        constants.dynamicPrecision._16bit._IndexModulo           = packTwoUintsTo32bit(indexModulo[0], indexModulo[1]);
        constants.dynamicPrecision._16bit._LutOffset             = packTwoUintsTo32bit(jitterTileOffset[0], jitterTileOffset[1]);
        constants.dynamicPrecision._16bit._NotHistoryReset       = packTwoFloatsTo32bit(noHistoryReset, 0.0f);
    }
    else
    {
        constants.dynamicPrecision._32bit._QuantParamsSNORM[0]   = quantParamsSNORM[0];
        constants.dynamicPrecision._32bit._QuantParamsSNORM[1]   = quantParamsSNORM[1];
        constants.dynamicPrecision._32bit._QuantParamsSNORM[2]   = quantParamsSNORM[2];
        constants.dynamicPrecision._32bit._QuantParamsSNORM[3]   = quantParamsSNORM[3];
        constants.dynamicPrecision._32bit._QuantParamsSINT[0]    = quantParamsSINT[0];
        constants.dynamicPrecision._32bit._QuantParamsSINT[1]    = quantParamsSINT[1];
        constants.dynamicPrecision._32bit._QuantParamsSINT[2]    = quantParamsSINT[2];
        constants.dynamicPrecision._32bit._QuantParamsSINT[3]    = quantParamsSINT[3];
        constants.dynamicPrecision._32bit._Exposure[0]           = exposure;
        constants.dynamicPrecision._32bit._Exposure[1]           = invExposure;
        constants.dynamicPrecision._32bit._MotionDisThreshPad[0] = motionVectorThreshold;
        constants.dynamicPrecision._32bit._MotionDisThreshPad[1] = motionDisocclusionThreshold;
        constants.dynamicPrecision._32bit._MotionDisThreshPad[2] = disocclusionScale;
        constants.dynamicPrecision._32bit._MotionDisThreshPad[3] = 0.0f;
        constants.dynamicPrecision._32bit._IndexModulo[0]        = indexModulo[0];
        constants.dynamicPrecision._32bit._IndexModulo[1]        = indexModulo[1];
        constants.dynamicPrecision._32bit._LutOffset[0]          = jitterTileOffset[0];
        constants.dynamicPrecision._32bit._LutOffset[1]          = jitterTileOffset[1];
        constants.dynamicPrecision._32bit._NotHistoryReset       = noHistoryReset;
    }

    // initialize constantBuffers data
    context->contextDescription.backendInterface.fpStageConstantBufferDataFunc(&context->contextDescription.backendInterface,
                                                                               &context->constants,
                                                                               sizeof(context->constants),
                                                                               &context->constantBuffers[FFX_NSS_CONSTANTBUFFER_IDENTIFIER_NSS]);
}

static void scheduleDispatch(FfxNssContext_Private*           context,
                             const FfxNssDispatchDescription* params,
                             const FfxPipelineState*          pipeline,
                             uint32_t                         dispatchX,
                             uint32_t                         dispatchY,
                             wchar_t*                         debugName)
{
    FfxGpuJobDescription dispatchJob = {FFX_GPU_JOB_COMPUTE};
    if (debugName != nullptr)
    {
        wcscpy(dispatchJob.jobLabel, debugName);
    }

    for (uint32_t currentShaderResourceViewIndex = 0; currentShaderResourceViewIndex < pipeline->srvTextureCount; ++currentShaderResourceViewIndex)
    {
        const uint32_t            currentResourceId = pipeline->srvTextureBindings[currentShaderResourceViewIndex].resourceIdentifier;
        const FfxResourceInternal currentResource   = context->srvResources[currentResourceId];
        dispatchJob.computeJobDescriptor.srvTextures[currentShaderResourceViewIndex].resource = currentResource;
#ifdef FFX_DEBUG
        wcscpy(dispatchJob.computeJobDescriptor.srvTextures[currentShaderResourceViewIndex].name,
               pipeline->srvTextureBindings[currentShaderResourceViewIndex].name);
#endif
    }

    for (uint32_t currentUnorderedAccessViewIndex = 0; currentUnorderedAccessViewIndex < pipeline->uavTextureCount; ++currentUnorderedAccessViewIndex)
    {
        const uint32_t currentResourceId = pipeline->uavTextureBindings[currentUnorderedAccessViewIndex].resourceIdentifier;
#ifdef FFX_DEBUG
        wcscpy(dispatchJob.computeJobDescriptor.uavTextureNames[currentUnorderedAccessViewIndex],
               pipeline->uavTextureBindings[currentUnorderedAccessViewIndex].name);
#endif

        {
            const FfxResourceInternal currentResource                                              = context->uavResources[currentResourceId];
            dispatchJob.computeJobDescriptor.uavTextures[currentUnorderedAccessViewIndex].resource = currentResource;
            dispatchJob.computeJobDescriptor.uavTextures[currentUnorderedAccessViewIndex].mip      = 0;
        }
    }

    for (uint32_t currentTensorIndex = 0; currentTensorIndex < pipeline->srvTensorCount; ++currentTensorIndex)
    {
        const uint32_t            currentResourceId                              = pipeline->srvTensorBindings[currentTensorIndex].resourceIdentifier;
        const FfxResourceInternal currentResource                                = context->srvResources[currentResourceId];
        dispatchJob.computeJobDescriptor.srvTensors[currentTensorIndex].resource = currentResource;
#ifdef FFX_DEBUG
        wcscpy(dispatchJob.computeJobDescriptor.srvTensors[currentTensorIndex].name, pipeline->srvTensorBindings[currentTensorIndex].name);
#endif
    }

    for (uint32_t currentTensorIndex = 0; currentTensorIndex < pipeline->uavTensorCount; ++currentTensorIndex)
    {
        const uint32_t            currentResourceId                              = pipeline->uavTensorBindings[currentTensorIndex].resourceIdentifier;
        const FfxResourceInternal currentResource                                = context->uavResources[currentResourceId];
        dispatchJob.computeJobDescriptor.uavTensors[currentTensorIndex].resource = currentResource;
#ifdef FFX_DEBUG
        wcscpy(dispatchJob.computeJobDescriptor.uavTensors[currentTensorIndex].name, pipeline->uavTensorBindings[currentTensorIndex].name);
#endif
    }

    dispatchJob.computeJobDescriptor.dimensions[0] = dispatchX;
    dispatchJob.computeJobDescriptor.dimensions[1] = dispatchY;
    dispatchJob.computeJobDescriptor.dimensions[2] = 1;
    dispatchJob.computeJobDescriptor.pipeline      = *pipeline;

    for (uint32_t currentRootConstantIndex = 0; currentRootConstantIndex < pipeline->constCount; ++currentRootConstantIndex)
    {
        wcscpy(dispatchJob.computeJobDescriptor.cbNames[currentRootConstantIndex], pipeline->constantBufferBindings[currentRootConstantIndex].name);
        dispatchJob.computeJobDescriptor.cbs[currentRootConstantIndex] =
            context->constantBuffers[pipeline->constantBufferBindings[currentRootConstantIndex].resourceIdentifier];
    }

    context->contextDescription.backendInterface.fpScheduleGpuJob(&context->contextDescription.backendInterface, &dispatchJob);
}

static void scheduleDataGraph(FfxNssContext_Private* context, const FfxNssDispatchDescription* params, FfxPipelineState* pipeline, wchar_t* debugName)
{
    FfxGpuJobDescription dataGraphJob = {FFX_GPU_JOB_DATA_GRAPH};
    if (debugName != nullptr)
    {
        wcscpy(dataGraphJob.jobLabel, debugName);
    }

    for (uint32_t currentTensorIndex = 0; currentTensorIndex < pipeline->srvTensorCount; ++currentTensorIndex)
    {
        const uint32_t            currentResourceId                                  = pipeline->srvTensorBindings[currentTensorIndex].resourceIdentifier;
        const FfxResourceInternal currentResource                                    = context->srvResources[currentResourceId];
        dataGraphJob.dataGraphJobDescription.srvTensors[currentTensorIndex].resource = currentResource;
#ifdef FFX_DEBUG
        wcscpy(dataGraphJob.dataGraphJobDescription.srvTensors[currentTensorIndex].name, pipeline->srvTensorBindings[currentTensorIndex].name);
#endif
    }

    for (uint32_t currentTensorIndex = 0; currentTensorIndex < pipeline->uavTensorCount; ++currentTensorIndex)
    {
        const uint32_t            currentResourceId                                  = pipeline->uavTensorBindings[currentTensorIndex].resourceIdentifier;
        const FfxResourceInternal currentResource                                    = context->uavResources[currentResourceId];
        dataGraphJob.dataGraphJobDescription.uavTensors[currentTensorIndex].resource = currentResource;
#ifdef FFX_DEBUG
        wcscpy(dataGraphJob.dataGraphJobDescription.uavTensors[currentTensorIndex].name, pipeline->uavTensorBindings[currentTensorIndex].name);
#endif
    }

    dataGraphJob.dataGraphJobDescription.pipeline = *pipeline;

    context->contextDescription.backendInterface.fpScheduleGpuJob(&context->contextDescription.backendInterface, &dataGraphJob);
}

static FfxErrorCode nssDispatch(FfxNssContext_Private* context, const FfxNssDispatchDescription* params)
{
    FFX_ASSERT(context);
    FFX_ASSERT(params);

    if ((context->contextDescription.flags & FFX_NSS_CONTEXT_FLAG_ENABLE_DEBUG_CHECKING) == FFX_NSS_CONTEXT_FLAG_ENABLE_DEBUG_CHECKING)
    {
        nssDebugCheckDispatch(context, params);
    }

    // take a short cut to the command list
    FfxCommandList commandList = params->commandList;

    FfxGpuJobDescription clearJob = {FFX_GPU_JOB_CLEAR_FLOAT};

    const float clearValuesToZeroFloat[]{0.f, 0.f, 0.f, 0.f};
    memcpy(clearJob.clearJobDescriptor.color, clearValuesToZeroFloat, 4 * sizeof(float));

    const bool resetAccumulation = NeedResetHistory(context, params);
    if (resetAccumulation)
    {
        constexpr uint32_t resources_to_clear[] = {FFX_NSS_RESOURCE_IDENTIFIER_LUMA_DERIV_1,
                                                   FFX_NSS_RESOURCE_IDENTIFIER_LUMA_DERIV_2,
                                                   FFX_NSS_RESOURCE_IDENTIFIER_NEAREST_DEPTH_COORD_1,
                                                   FFX_NSS_RESOURCE_IDENTIFIER_NEAREST_DEPTH_COORD_2,
                                                   FFX_NSS_RESOURCE_IDENTIFIER_FEEDBACK_TENSOR_1,
                                                   FFX_NSS_RESOURCE_IDENTIFIER_FEEDBACK_TENSOR_2};

        for (uint32_t i = 0; i < FFX_COUNTOF(resources_to_clear); ++i)
        {
            clearJob.clearJobDescriptor.target = context->srvResources[resources_to_clear[i]];
            context->contextDescription.backendInterface.fpScheduleGpuJob(&context->contextDescription.backendInterface, &clearJob);
        }

        if (context->hasPaddingPass)
        {
            clearJob.clearJobDescriptor.target = context->srvResources[FFX_NSS_RESOURCE_IDENTIFIER_PADDED_OUTPUT_1];
            context->contextDescription.backendInterface.fpScheduleGpuJob(&context->contextDescription.backendInterface, &clearJob);

            clearJob.clearJobDescriptor.target = context->srvResources[FFX_NSS_RESOURCE_IDENTIFIER_PADDED_OUTPUT_2];
            context->contextDescription.backendInterface.fpScheduleGpuJob(&context->contextDescription.backendInterface, &clearJob);
        }
    }

    // Prepare per frame descriptor tables
    const bool isOddFrame = !!(context->resourceFrameIndex & 1);

    const uint32_t lumaDerivSrvResourceIndex = isOddFrame ? FFX_NSS_RESOURCE_IDENTIFIER_LUMA_DERIV_2 : FFX_NSS_RESOURCE_IDENTIFIER_LUMA_DERIV_1;
    const uint32_t lumaDerivUavResourceIndex = isOddFrame ? FFX_NSS_RESOURCE_IDENTIFIER_LUMA_DERIV_1 : FFX_NSS_RESOURCE_IDENTIFIER_LUMA_DERIV_2;
    const uint32_t feedbackTm1ResourceIndex  = isOddFrame ? FFX_NSS_RESOURCE_IDENTIFIER_FEEDBACK_TENSOR_2 : FFX_NSS_RESOURCE_IDENTIFIER_FEEDBACK_TENSOR_1;
    const uint32_t feedbackResourceIndex     = isOddFrame ? FFX_NSS_RESOURCE_IDENTIFIER_FEEDBACK_TENSOR_1 : FFX_NSS_RESOURCE_IDENTIFIER_FEEDBACK_TENSOR_2;
    const uint32_t depthOffsetTm1ResourceIndex =
        isOddFrame ? FFX_NSS_RESOURCE_IDENTIFIER_NEAREST_DEPTH_COORD_2 : FFX_NSS_RESOURCE_IDENTIFIER_NEAREST_DEPTH_COORD_1;
    const uint32_t depthOffsetResourceIndex =
        isOddFrame ? FFX_NSS_RESOURCE_IDENTIFIER_NEAREST_DEPTH_COORD_1 : FFX_NSS_RESOURCE_IDENTIFIER_NEAREST_DEPTH_COORD_2;

    const uint32_t paddedOutputResourceIndex  = isOddFrame ? FFX_NSS_RESOURCE_IDENTIFIER_PADDED_OUTPUT_1 : FFX_NSS_RESOURCE_IDENTIFIER_PADDED_OUTPUT_2;
    const uint32_t paddedHistoryResourceIndex = isOddFrame ? FFX_NSS_RESOURCE_IDENTIFIER_PADDED_OUTPUT_2 : FFX_NSS_RESOURCE_IDENTIFIER_PADDED_OUTPUT_1;

    // Setup the resources for mirror padding and pre-process stage
    {
        // Input: Color/Depth/Depth tm1/Motion/History
        // If padding pass is needed, these inputs are register to unpadded resource ids used by mirror padding stage, and the
        // mirror padding stage will output to padded resource ids used by pre-process stage, which is already setup in nssCreate().
        // If padding pass is not needed, we register these inputs directly to padded resource ids used by pre-process stage.
        const uint32_t external_input_color_resource_id =
            context->hasPaddingPass ? FFX_NSS_RESOURCE_IDENTIFIER_UNPADDED_COLOR : FFX_NSS_RESOURCE_IDENTIFIER_INPUT_COLOR;
        const uint32_t external_input_depth_resource_id =
            context->hasPaddingPass ? FFX_NSS_RESOURCE_IDENTIFIER_UNPADDED_DEPTH : FFX_NSS_RESOURCE_IDENTIFIER_INPUT_DEPTH;
        const uint32_t external_input_depth_tm1_resource_id =
            context->hasPaddingPass ? FFX_NSS_RESOURCE_IDENTIFIER_UNPADDED_DEPTH_TM1 : FFX_NSS_RESOURCE_IDENTIFIER_INPUT_DEPTH_TM1;
        const uint32_t external_input_motion_resource_id =
            context->hasPaddingPass ? FFX_NSS_RESOURCE_IDENTIFIER_UNPADDED_MOTION : FFX_NSS_RESOURCE_IDENTIFIER_INPUT_MOTION_VECTORS;

        // Input: Depth tm1
        context->contextDescription.backendInterface.fpRegisterResource(&context->contextDescription.backendInterface,
                                                                        &params->depthTm1,
                                                                        context->effectContextId,
                                                                        &context->srvResources[external_input_depth_tm1_resource_id]);

        // Input: Depth
        context->contextDescription.backendInterface.fpRegisterResource(
            &context->contextDescription.backendInterface, &params->depth, context->effectContextId, &context->srvResources[external_input_depth_resource_id]);

        // Input: Color
        context->contextDescription.backendInterface.fpRegisterResource(
            &context->contextDescription.backendInterface, &params->color, context->effectContextId, &context->srvResources[external_input_color_resource_id]);

        // Input: Motion vector
        context->contextDescription.backendInterface.fpRegisterResource(&context->contextDescription.backendInterface,
                                                                        &params->motionVectors,
                                                                        context->effectContextId,
                                                                        &context->srvResources[external_input_motion_resource_id]);

        // Input: History
        // When there is padding pass, history will use the padded history generated by last frame.
        // When there is no padding pass, history is registered to the external history resource directly.
        if (context->hasPaddingPass)
        {
            FFX_ASSERT(context->uavResources[paddedHistoryResourceIndex].internalIndex == context->srvResources[paddedHistoryResourceIndex].internalIndex);
            context->srvResources[FFX_NSS_RESOURCE_IDENTIFIER_HISTORY_UPSCALED_COLOR] = context->srvResources[paddedHistoryResourceIndex];
        }
        else
        {
            context->contextDescription.backendInterface.fpRegisterResource(&context->contextDescription.backendInterface,
                                                                            &params->outputTm1,
                                                                            context->effectContextId,
                                                                            &context->srvResources[FFX_NSS_RESOURCE_IDENTIFIER_HISTORY_UPSCALED_COLOR]);
        }

        // Input: DepthOffset tm1
        context->srvResources[FFX_NSS_RESOURCE_IDENTIFIER_NEAREST_DEPTH_COORD_TM1] = context->srvResources[depthOffsetTm1ResourceIndex];

        // Input: Feedback tm1
        context->srvResources[FFX_NSS_RESOURCE_IDENTIFIER_FEEDBACK_TENSOR] = context->srvResources[feedbackTm1ResourceIndex];

        // Input: LumaDeriv tm1
        context->srvResources[FFX_NSS_RESOURCE_IDENTIFIER_LUMA_DERIV] = context->srvResources[lumaDerivSrvResourceIndex];

        // Output: LumaDeriv, consumed by next frame
        context->uavResources[FFX_NSS_RESOURCE_IDENTIFIER_LUMA_DERIV] = context->uavResources[lumaDerivUavResourceIndex];

        // Output: data graph input tensor
        // This is already setup in nssCreate(), nothing to do here.

        // Output: Depth offset, consumed by next frame and post-process stage
        context->uavResources[FFX_NSS_RESOURCE_IDENTIFIER_NEAREST_DEPTH_COORD] = context->uavResources[depthOffsetResourceIndex];
    }

    // Setup the resources for data graph stage
    {
        // Input: data graph input tensor, outputted by pre-process stage.
        // This is already setup in nssCreate(), nothing to do here.

        // Output: Feedback tensor, consumed by next frame.
        context->uavResources[FFX_NSS_RESOURCE_IDENTIFIER_FEEDBACK_TENSOR] = context->uavResources[feedbackResourceIndex];

        // Output: Coefficients tensors, consumed by post-process stage
        // This is already setup in nssCreate(), nothing to do here.
    }

    // Setup the resources for post-process stage
    {
        // Input: Coefficients tensors, outputted by data graph stage
        // This is already setup in nssCreate(), nothing to do here.

        // Input: Kernel LUT is baked in shader code, no work to do here.

        // Input: Depth offset, outputted by pre-process stage
        FFX_ASSERT(context->uavResources[depthOffsetResourceIndex].internalIndex == context->srvResources[depthOffsetResourceIndex].internalIndex);
        context->srvResources[FFX_NSS_RESOURCE_IDENTIFIER_NEAREST_DEPTH_COORD] = context->srvResources[depthOffsetResourceIndex];

        // Input: Color, already setup in pre-process stage, nothing to do here.
        // Input: Motion vector, already setup in pre-process stage, nothing to do here.
        // Input: History, already setup in pre-process stage, nothing to do here.

        // Output: Upscaled padded output
        // If padding pass is needed, upscaled output resource id is already setup in nssCreate(), we register the external output resource to unppaded output resource id.
        // If padding pass is not needed, we register the external output resource to upscaled output resource id.
        if (context->hasPaddingPass)
        {
            FFX_ASSERT(context->uavResources[paddedOutputResourceIndex].internalIndex == context->srvResources[paddedOutputResourceIndex].internalIndex);
            context->uavResources[FFX_NSS_RESOURCE_IDENTIFIER_UPSCALED_OUTPUT] = context->uavResources[paddedOutputResourceIndex];
            context->srvResources[FFX_NSS_RESOURCE_IDENTIFIER_UPSCALED_OUTPUT] = context->srvResources[paddedOutputResourceIndex];

            context->contextDescription.backendInterface.fpRegisterResource(&context->contextDescription.backendInterface,
                                                                            &params->output,
                                                                            context->effectContextId,
                                                                            &context->uavResources[FFX_NSS_RESOURCE_IDENTIFIER_UNPADDED_OUTPUT]);
        }
        else
        {
            context->contextDescription.backendInterface.fpRegisterResource(&context->contextDescription.backendInterface,
                                                                            &params->output,
                                                                            context->effectContextId,
                                                                            &context->uavResources[FFX_NSS_RESOURCE_IDENTIFIER_UPSCALED_OUTPUT]);
        }
    }

    if ((params->flags & FFX_NSS_DISPATCH_FLAG_DRAW_DEBUG_VIEW) == FFX_NSS_DISPATCH_FLAG_DRAW_DEBUG_VIEW)
    {
        context->contextDescription.backendInterface.fpRegisterResource(&context->contextDescription.backendInterface,
                                                                        &params->debugViews,
                                                                        context->effectContextId,
                                                                        &context->uavResources[FFX_NSS_RESOURCE_IDENTIFIER_DEBUG_VIEWS]);

        clearJob.clearJobDescriptor.target = context->uavResources[FFX_NSS_RESOURCE_IDENTIFIER_DEBUG_VIEWS];
        context->contextDescription.backendInterface.fpScheduleGpuJob(&context->contextDescription.backendInterface, &clearJob);
    }

    const int32_t dispatchSrcX = FFX_DIVIDE_ROUNDING_UP(context->paddedInputWidth, 16);
    const int32_t dispatchSrcY = FFX_DIVIDE_ROUNDING_UP(context->paddedInputHeight, 16);
    const int32_t dispatchDstX = FFX_DIVIDE_ROUNDING_UP(context->paddedOutputWidth, 16);
    const int32_t dispatchDstY = FFX_DIVIDE_ROUNDING_UP(context->paddedOutputHeight, 16);

    const bool require16bit = (context->contextDescription.flags & FFX_NSS_CONTEXT_FLAG_ALLOW_16BIT) == FFX_NSS_CONTEXT_FLAG_ALLOW_16BIT;
    const bool use16bit     = require16bit && context->deviceCapabilities.fp16Supported;
    setupConstantBuffer(context, params, use16bit);

    if (context->hasPaddingPass)
    {
        scheduleDispatch(context, params, &context->pipelineNssMirrorPadding, dispatchSrcX, dispatchSrcY, L"MirrorPadding");
    }

    scheduleDispatch(context, params, &context->pipelineNssPreprocess, dispatchSrcX, dispatchSrcY, L"Preprocess");

    scheduleDataGraph(context, params, &context->pipelineNssDataGraph, L"DataGraph");

    scheduleDispatch(context, params, &context->pipelineNssPostprocess, dispatchDstX, dispatchDstY, L"Postprocess");

    if (context->hasPaddingPass)
    {
        FfxGpuJobDescription copyJob = {FFX_GPU_JOB_COPY};

        copyJob.copyJobDescriptor.src      = context->srvResources[FFX_NSS_RESOURCE_IDENTIFIER_UPSCALED_OUTPUT];
        copyJob.copyJobDescriptor.dst      = context->uavResources[FFX_NSS_RESOURCE_IDENTIFIER_UNPADDED_OUTPUT];
        copyJob.copyJobDescriptor.copyMode = FFX_GPU_COPY_DST_EXTENT;  // We want to crop the padded output to the unpadded output.

        context->contextDescription.backendInterface.fpScheduleGpuJob(&context->contextDescription.backendInterface, &copyJob);
    }

    if ((params->flags & FFX_NSS_DISPATCH_FLAG_DRAW_DEBUG_VIEW) == FFX_NSS_DISPATCH_FLAG_DRAW_DEBUG_VIEW)
    {
        const int32_t dispatchSrcX = FFX_DIVIDE_ROUNDING_UP(params->renderSize.width, 16);
        const int32_t dispatchSrcY = FFX_DIVIDE_ROUNDING_UP(params->renderSize.height, 16);
        scheduleDispatch(context, params, &context->pipelineNssDebugView, dispatchSrcX, dispatchSrcY, L"DebugView");
    }

    context->resourceFrameIndex = (context->resourceFrameIndex + 1) % NSS_MAX_QUEUED_FRAMES;
    // NSS_MAX_QUEUED_FRAMES must be an even number.
    FFX_STATIC_ASSERT((NSS_MAX_QUEUED_FRAMES & 1) == 0);

    context->contextDescription.backendInterface.fpExecuteGpuJobs(&context->contextDescription.backendInterface, commandList, context->effectContextId);

    // release dynamic resources
    context->contextDescription.backendInterface.fpUnregisterResources(&context->contextDescription.backendInterface, commandList, context->effectContextId);

    context->firstExecution = false;
    return FFX_OK;
}

FfxErrorCode ffxNssContextCreate(FfxNssContext* context, const FfxNssContextDescription* contextDescription)
{
    // zero context memory
    memset(context, 0, sizeof(FfxNssContext));

    // check pointers are valid.
    FFX_RETURN_ON_ERROR(context, FFX_ERROR_INVALID_POINTER);
    FFX_RETURN_ON_ERROR(contextDescription, FFX_ERROR_INVALID_POINTER);

    // validate that all callbacks are set for the interface
    FFX_RETURN_ON_ERROR(contextDescription->backendInterface.fpGetDeviceCapabilities, FFX_ERROR_INCOMPLETE_INTERFACE);
    FFX_RETURN_ON_ERROR(contextDescription->backendInterface.fpCreateBackendContext, FFX_ERROR_INCOMPLETE_INTERFACE);
    FFX_RETURN_ON_ERROR(contextDescription->backendInterface.fpDestroyBackendContext, FFX_ERROR_INCOMPLETE_INTERFACE);

    // validate prarmeters
    FFX_RETURN_ON_ERROR(contextDescription->maxRenderSize.width, FFX_ERROR_INVALID_ALIGNMENT);
    FFX_RETURN_ON_ERROR(contextDescription->maxRenderSize.height, FFX_ERROR_INVALID_ALIGNMENT);
    FFX_RETURN_ON_ERROR(contextDescription->maxUpscaleSize.width, FFX_ERROR_INVALID_ALIGNMENT);
    FFX_RETURN_ON_ERROR(contextDescription->maxUpscaleSize.height, FFX_ERROR_INVALID_ALIGNMENT);

    // if a scratch buffer is declared, then we must have a size
    if (contextDescription->backendInterface.scratchBuffer)
    {
        FFX_RETURN_ON_ERROR(contextDescription->backendInterface.scratchBufferSize, FFX_ERROR_INCOMPLETE_INTERFACE);
    }

    // ensure the context is large enough for the internal context.
    FFX_STATIC_ASSERT(sizeof(FfxNssContext) >= sizeof(FfxNssContext_Private));

    // create the context.
    FfxNssContext_Private* contextPrivate = (FfxNssContext_Private*)(context);
    const FfxErrorCode     errorCode      = nssCreate(contextPrivate, contextDescription);

    return errorCode;
}

FfxErrorCode ffxNssContextDestroy(FfxNssContext* context)
{
    FFX_RETURN_ON_ERROR(context, FFX_ERROR_INVALID_POINTER);

    // destroy the context.
    FfxNssContext_Private* contextPrivate = (FfxNssContext_Private*)(context);
    const FfxErrorCode     errorCode      = nssRelease(contextPrivate);
    return errorCode;
}

FfxErrorCode ffxNssContextDispatch(FfxNssContext* context, const FfxNssDispatchDescription* dispatchParams)
{
    FFX_RETURN_ON_ERROR(context, FFX_ERROR_INVALID_POINTER);
    FFX_RETURN_ON_ERROR(dispatchParams, FFX_ERROR_INVALID_POINTER);

    FfxNssContext_Private* contextPrivate = (FfxNssContext_Private*)(context);
    FFX_ASSERT(contextPrivate);

    // validate zero sizes.
    FFX_RETURN_ON_ERROR(dispatchParams->renderSize.width, FFX_ERROR_INVALID_ARGUMENT);
    FFX_RETURN_ON_ERROR(dispatchParams->renderSize.height, FFX_ERROR_INVALID_ARGUMENT);
    FFX_RETURN_ON_ERROR(dispatchParams->upscaleSize.width, FFX_ERROR_INVALID_ARGUMENT);
    FFX_RETURN_ON_ERROR(dispatchParams->upscaleSize.height, FFX_ERROR_INVALID_ARGUMENT);

    // validate that renderSize/upscaleSize match the size declared at context creation.
    FFX_RETURN_ON_ERROR(dispatchParams->renderSize.width == contextPrivate->contextDescription.maxRenderSize.width, FFX_ERROR_OUT_OF_RANGE);
    FFX_RETURN_ON_ERROR(dispatchParams->renderSize.height == contextPrivate->contextDescription.maxRenderSize.height, FFX_ERROR_OUT_OF_RANGE);
    FFX_RETURN_ON_ERROR(dispatchParams->upscaleSize.width == contextPrivate->contextDescription.maxUpscaleSize.width, FFX_ERROR_OUT_OF_RANGE);
    FFX_RETURN_ON_ERROR(dispatchParams->upscaleSize.height == contextPrivate->contextDescription.maxUpscaleSize.height, FFX_ERROR_OUT_OF_RANGE);
    FFX_RETURN_ON_ERROR(contextPrivate->device, FFX_ERROR_NULL_DEVICE);

    FFX_RETURN_ON_ERROR(contextPrivate->device, FFX_ERROR_NULL_DEVICE);

    // dispatch the NSS passes.
    const FfxErrorCode errorCode = nssDispatch(contextPrivate, dispatchParams);
    return errorCode;
}

int32_t ffxNssGetJitterPhaseCount(int32_t renderWidth, int32_t displayWidth)
{
    const float   basePhaseCount   = 8.0f;
    const int32_t jitterPhaseCount = int32_t(basePhaseCount * pow((float(displayWidth) / renderWidth), 2.0f));
    return jitterPhaseCount;
}

// Calculate halton number for index and base.
static float halton(int32_t index, int32_t base)
{
    float f = 1.0f, result = 0.0f;

    for (int32_t currentIndex = index; currentIndex > 0;)
    {
        f /= (float)base;
        result       = result + f * (float)(currentIndex % base);
        currentIndex = (uint32_t)(floorf((float)(currentIndex) / (float)(base)));
    }

    return result;
}

FfxErrorCode ffxNssGetJitterOffset(float* outX, float* outY, int32_t index, int32_t phaseCount)
{
    FFX_RETURN_ON_ERROR(outX, FFX_ERROR_INVALID_POINTER);
    FFX_RETURN_ON_ERROR(outY, FFX_ERROR_INVALID_POINTER);
    FFX_RETURN_ON_ERROR(phaseCount > 0, FFX_ERROR_INVALID_ARGUMENT);

    const float x = halton((index % phaseCount) + 1, 2) - 0.5f;
    const float y = halton((index % phaseCount) + 1, 3) - 0.5f;

    *outX = x;
    *outY = y;
    return FFX_OK;
}

FFX_API bool ffxNssResourceIsNull(const FfxResource& resource)
{
    return resource.resource == NULL;
}
