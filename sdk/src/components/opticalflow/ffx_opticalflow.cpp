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

/*
 * SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT
 */

#include <algorithm>  // for max used inside SPD CPU code.
#include <cfloat>     // for FLT_EPSILON
#include <cmath>      // for fabs, abs, sinf, sqrt, etc.
#include <cstdio>
#include <limits>
#include <string.h>  // for memset

#include "FidelityFX/host/ffx_opticalflow.h"
#include "FidelityFX/host/ffx_frameinterpolation_resolution.h"
#define FFX_CPU

#include "FidelityFX/gpu/ffx_core.h"
#include "ffx_object_management.h"

#include "FidelityFX/host/ffx_util.h"

#include "ffx_logging.h"
#include "ffx_opticalflow_trace.h"

#ifdef __clang__
#pragma clang diagnostic ignored "-Wunused-variable"
#endif

#include "ffx_opticalflow_private.h"

static constexpr const char* ffxLogComponentName()
{
    return "OpticalFlow";
}

// lists to map shader resource bindpoint name to resource identifier
typedef struct ResourceBinding
{
    // sdk data
    uint32_t index;
    char     name[64];
} ResourceBinding;

constexpr int                srvTextureCount                         = 5;
static const ResourceBinding srvTextureBindingTable[srvTextureCount] = {
    {FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_COLOR, "r_color_tp1"},
    {FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_PREV_COLOR, "r_color_tm1"},
    {FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_DEPTH, "r_depth"},
    {FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_PREV_DEPTH, "r_depth_tm1"},
    {FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_COMPUTED_MOTION_VECTORS, "r_motion_vectors"},
};

constexpr int                uavTextureCount                         = 4;
static const ResourceBinding uavTextureBindingTable[uavTextureCount] = {
    {FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_RESULT, "rw_flow"},
    {FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_COMPUTED_MOTION_VECTORS, "rw_motion_vectors"},
    {FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_PREV_DEPTH, "rw_depth_tm1"},
    {FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_PREV_COLOR, "rw_color_tm1"},
};

static const ResourceBinding rtResourceBindingTable[] = {
    {FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_COMPUTED_MOTION_VECTORS, "rw_motion_vectors"},
};

static FfxErrorCode patchResourceBindings(FfxPipelineState* inoutPipeline)
{
    for (uint32_t srvIndex = 0; srvIndex < inoutPipeline->srvTextureCount; ++srvIndex)
    {
        int32_t mapIndex = 0;
        for (mapIndex = 0; mapIndex < FFX_COUNTOF(srvTextureBindingTable); ++mapIndex)
        {
            if (0 == strcmp(srvTextureBindingTable[mapIndex].name, inoutPipeline->srvTextureBindings[srvIndex].name))
                break;
        }
        if (mapIndex == FFX_COUNTOF(srvTextureBindingTable))
            return ffxLogError(FFX_ERROR_INVALID_ARGUMENT, "the shader SRV binding name is not recognized");

        inoutPipeline->srvTextureBindings[srvIndex].resourceIdentifier = srvTextureBindingTable[mapIndex].index;
    }

    for (uint32_t uavIndex = 0; uavIndex < inoutPipeline->uavTextureCount; ++uavIndex)
    {
        int32_t mapIndex = 0;
        for (mapIndex = 0; mapIndex < FFX_COUNTOF(uavTextureBindingTable); ++mapIndex)
        {
            if (0 == strcmp(uavTextureBindingTable[mapIndex].name, inoutPipeline->uavTextureBindings[uavIndex].name))
                break;
        }
        if (mapIndex == FFX_COUNTOF(uavTextureBindingTable))
            return ffxLogError(FFX_ERROR_INVALID_ARGUMENT, "the shader UAV binding name is not recognized");

        inoutPipeline->uavTextureBindings[uavIndex].resourceIdentifier = uavTextureBindingTable[mapIndex].index;
    }

    for (uint32_t rtIndex = 0; rtIndex < inoutPipeline->rtCount; ++rtIndex)
    {
        int32_t mapIndex = 0;
        for (mapIndex = 0; mapIndex < FFX_COUNTOF(rtResourceBindingTable); ++mapIndex)
        {
            if (0 == strcmp(rtResourceBindingTable[mapIndex].name, inoutPipeline->rtBindings[rtIndex].name))
                break;
        }
        if (mapIndex == FFX_COUNTOF(rtResourceBindingTable))
            return ffxLogError(FFX_ERROR_INVALID_ARGUMENT, "the shader render target binding name is not recognized");

        inoutPipeline->rtBindings[rtIndex].resourceIdentifier = rtResourceBindingTable[mapIndex].index;
    }

    return FFX_OK;
}

static FfxOpticalFlowDescription createOpticalFlowDescription(
    FfxSurfaceFormat colorFormat, FfxDimensions2D resolution, FfxOpticalFlowGridSize gridSize, FfxOpticalFlowPerformanceLevel performanceLevel, bool useMVHints)
{
    // Mimic the shader blob memory style for now.
    // TODO: We might want to dynamically allocate this and clean it up after we are done with it instead of allocating static memory.
    static const char*                  g_srv_names[]            = {"r_color_tp1", "r_color_tm1", "r_motion_vectors"};
    static uint32_t                     g_srv_texture_bindings[] = {0, 1, 2};
    static uint32_t                     g_srv_textureCount[]     = {1, 1, 1};
    static uint32_t                     g_srv_textureSpaces[]    = {0, 0, 0};
    static FfxSurfaceFormat             g_srv_formats[]          = {colorFormat, colorFormat, FFX_SURFACE_FORMAT_R16G16_FLOAT};
    static FfxOpticalFlowConnectionType g_srv_connection_types[] = {
        // Connection type mapping between SDK resource names and DDK optical flow roles:
        // The optical flow pipeline uses forward optical flow convention, where the TP1 (current frame)
        // image is the "search" frame and TM1 (previous frame) is the "template" frame.
        // In DDK terms:
        //   FFX_OPTICAL_FLOW_CONNECTION_REFERENCE -> "search" frame
        //   FFX_OPTICAL_FLOW_CONNECTION_INPUT     -> "template" frame
        // In SDK terms:
        //   "r_color_tp1"     = TP1 (current frame) color  -> "search" frame -> maps to FFX_OPTICAL_FLOW_CONNECTION_REFERENCE
        //   "r_color_tm1" = TM1 (previous frame) color -> "template" frame -> maps to FFX_OPTICAL_FLOW_CONNECTION_INPUT
        FFX_OPTICAL_FLOW_CONNECTION_REFERENCE,
        FFX_OPTICAL_FLOW_CONNECTION_INPUT,
        FFX_OPTICAL_FLOW_CONNECTION_HINT};

    static const char*                  g_uav_names[]            = {"rw_flow"};
    static uint32_t                     g_uav_texture_bindings[] = {3};
    static uint32_t                     g_uav_textureCount[]     = {1};
    static uint32_t                     g_uav_textureSpaces[]    = {0};
    static FfxOpticalFlowConnectionType g_uav_connection_types[] = {FFX_OPTICAL_FLOW_CONNECTION_FLOW_VECTOR};
    static FfxSurfaceFormat             g_uav_formats[]          = {FFX_SURFACE_FORMAT_R16G16_FLOAT};

    FfxOpticalFlowDescription ofDesc{};

    ofDesc.srvTextureCount               = useMVHints ? sizeof(g_srv_texture_bindings) / sizeof(g_srv_texture_bindings[0]) : 2;
    ofDesc.boundSRVTextureNames          = g_srv_names;
    ofDesc.boundSRVTextures              = g_srv_texture_bindings;
    ofDesc.boundSRVTextureCounts         = g_srv_textureCount;
    ofDesc.boundSRVTextureSpaces         = g_srv_textureSpaces;
    ofDesc.boundSRVTextureConnectionType = g_srv_connection_types;
    ofDesc.boundSRVTextureFormats        = g_srv_formats;
    // Input MVs are R16G16F, Color is in the passed in format

    ofDesc.uavTextureCount               = sizeof(g_uav_texture_bindings) / sizeof(g_uav_texture_bindings[0]);
    ofDesc.boundUAVTextureNames          = g_uav_names;
    ofDesc.boundUAVTextures              = g_uav_texture_bindings;
    ofDesc.boundUAVTextureCounts         = g_uav_textureCount;
    ofDesc.boundUAVTextureSpaces         = g_uav_textureSpaces;
    ofDesc.boundUAVTextureConnectionType = g_uav_connection_types;
    ofDesc.boundUAVTextureFormats        = g_uav_formats;

    // OF metadata
    ofDesc.dimensions       = resolution;
    ofDesc.gridSize         = gridSize;
    ofDesc.performanceLevel = performanceLevel;

    return ofDesc;
}

static uint32_t getPipelinePermutationOptions(uint32_t contextFlags)
{
    uint32_t permutationOptions = 0;

    // Check for depth inverted flag
    if (contextFlags & FFX_OPTICALFLOW_ENABLE_DEPTH_INVERTED)
    {
        permutationOptions |= OPTICALFLOW_SHADER_PERMUTATION_DEPTH_INVERTED;
    }

    // Check for MV hints compute flag
    if (contextFlags & FFX_OPTICALFLOW_ENABLE_MV_HINTS_COMPUTE)
    {
        permutationOptions |= OPTICALFLOW_SHADER_PERMUTATION_MV_HINTS_COMPUTE;
    }

    return permutationOptions;
}

static FfxErrorCode createPipelineStates(OpticalFlowContext_Private* context)
{
    FFX_ASSERT(context);

    ffxSafeReleasePipeline(&context->contextDescription.backendInterface, &context->pipelineOpticalflow, context->effectContextId);

    FfxOpticalFlowDescription ofDesc = createOpticalFlowDescription(context->contextDescription.backBufferFormat,
                                                                    context->contextDescription.resolution,
                                                                    context->contextDescription.gridSize,
                                                                    context->contextDescription.performanceLevel,
                                                                    context->useMVHints);

    // TODO: add more validation on the connection type with device caps like costSupported, hintSupported

    FFX_VALIDATE(context->contextDescription.backendInterface.fpCreateOpticalFlowPipeline(
        &context->contextDescription.backendInterface, "ARM-OpticalFlow", ofDesc, context->effectContextId, &context->pipelineOpticalflow));

    FFX_VALIDATE(patchResourceBindings(&context->pipelineOpticalflow));

    if (!context->useMVHints)
    {
        return FFX_OK;
    }

    context->useMVHintsCompute = !!(context->contextDescription.flags & FFX_OPTICALFLOW_ENABLE_MV_HINTS_COMPUTE);
    if (!context->useMVHintsCompute)
    {
        ffxSafeReleasePipeline(&context->contextDescription.backendInterface, &context->pipelineFragmentMVHints, context->effectContextId);

        FfxPipelineDescription pipelineDescription = {};
        pipelineDescription.stage                  = FFX_BIND_PIXEL_SHADER_STAGE;
        pipelineDescription.contextFlags           = context->contextDescription.flags;

        FfxRootConstantDescription rootConstantDescs;
        rootConstantDescs.size                      = sizeof(context->computeMVHintsConstants);
        rootConstantDescs.stage                     = FFX_BIND_PIXEL_SHADER_STAGE;
        pipelineDescription.rootConstantBufferCount = 1;
        pipelineDescription.rootConstants           = &rootConstantDescs;

        FfxSamplerDescription samplerDesc = {
            FFX_FILTER_TYPE_MINMAGMIP_POINT, FFX_ADDRESS_MODE_CLAMP, FFX_ADDRESS_MODE_CLAMP, FFX_ADDRESS_MODE_CLAMP, FFX_BIND_PIXEL_SHADER_STAGE, FFX_FALSE};
        pipelineDescription.samplerCount = 1;
        pipelineDescription.samplers     = &samplerDesc;

        strncpy(pipelineDescription.name, "ARM-OpticalFlow-ComputeMVHints", FFX_RESOURCE_NAME_SIZE - 1);
        pipelineDescription.name[FFX_RESOURCE_NAME_SIZE - 1] = '\0';

        uint32_t permutationOptions = getPipelinePermutationOptions(context->contextDescription.flags);

        FFX_VALIDATE(context->contextDescription.backendInterface.fpCreateGraphicsPipeline(&context->contextDescription.backendInterface,
                                                                                           ARM_EFFECT_OPTICALFLOW,
                                                                                           FFX_OPTICALFLOW_PASS_COMPUTE_MOTION_FROM_DEPTH,
                                                                                           permutationOptions,
                                                                                           &pipelineDescription,
                                                                                           context->effectContextId,
                                                                                           &context->pipelineFragmentMVHints));

        FFX_VALIDATE(patchResourceBindings(&context->pipelineFragmentMVHints));
    }
    else
    {
        ffxSafeReleasePipeline(&context->contextDescription.backendInterface, &context->pipelineComputeMVHints, context->effectContextId);

        FfxPipelineDescription pipelineDescription = {};
        pipelineDescription.stage                  = FFX_BIND_COMPUTE_SHADER_STAGE;
        pipelineDescription.contextFlags           = context->contextDescription.flags;

        FfxRootConstantDescription rootConstantDescs;
        rootConstantDescs.size                      = sizeof(context->computeMVHintsConstants);
        rootConstantDescs.stage                     = FFX_BIND_COMPUTE_SHADER_STAGE;
        pipelineDescription.rootConstantBufferCount = 1;
        pipelineDescription.rootConstants           = &rootConstantDescs;

        FfxSamplerDescription samplerDesc = {
            FFX_FILTER_TYPE_MINMAGMIP_POINT, FFX_ADDRESS_MODE_CLAMP, FFX_ADDRESS_MODE_CLAMP, FFX_ADDRESS_MODE_CLAMP, FFX_BIND_COMPUTE_SHADER_STAGE, FFX_FALSE};
        pipelineDescription.samplerCount = 1;
        pipelineDescription.samplers     = &samplerDesc;

        strncpy(pipelineDescription.name, "ARM-OpticalFlow-ComputeMVHints", FFX_RESOURCE_NAME_SIZE - 1);
        pipelineDescription.name[FFX_RESOURCE_NAME_SIZE - 1] = '\0';

        const uint32_t permutationOptions = getPipelinePermutationOptions(context->contextDescription.flags);

        FFX_VALIDATE(context->contextDescription.backendInterface.fpCreateComputePipeline(&context->contextDescription.backendInterface,
                                                                                          ARM_EFFECT_OPTICALFLOW,
                                                                                          FFX_OPTICALFLOW_PASS_COMPUTE_MOTION_FROM_DEPTH,
                                                                                          permutationOptions,
                                                                                          &pipelineDescription,
                                                                                          context->effectContextId,
                                                                                          &context->pipelineComputeMVHints));

        FFX_VALIDATE(patchResourceBindings(&context->pipelineComputeMVHints));
    }

    return FFX_OK;
}

static FfxErrorCode createResourceFromDescription(OpticalFlowContext_Private* context, const FfxInternalResourceDescription* resDesc)
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
    const FfxResourceStates initialState = (resDesc->usage == FFX_RESOURCE_USAGE_READ_ONLY) ? FFX_RESOURCE_STATE_COMPUTE_READ : FFX_RESOURCE_STATE_GENERIC_UAV;
    const FfxCreateResourceDescription createResourceDescription = {
        FFX_HEAP_TYPE_DEFAULT, resourceDescription, initialState, resDesc->name, resDesc->id, resDesc->initData};
    return context->contextDescription.backendInterface.fpCreateResource(
        &context->contextDescription.backendInterface, &createResourceDescription, context->effectContextId, &context->srvResources[resDesc->id]);
}

static FfxErrorCode opticalFlowVkCreate(OpticalFlowContext_Private* context, const FfxOpticalFlowContextDescription* contextDescription)
{
    FFX_ASSERT(context);
    FFX_ASSERT(contextDescription);

    // Setup the data for implementation.
    memset(context, 0, sizeof(OpticalFlowContext_Private));
    context->device = contextDescription->backendInterface.device;

    memcpy(&context->contextDescription, contextDescription, sizeof(FfxOpticalFlowContextDescription));

    // Create the context.
    FfxErrorCode errorCode = context->contextDescription.backendInterface.fpCreateBackendContext(
        &context->contextDescription.backendInterface, ARM_EFFECT_OPTICALFLOW, nullptr, &context->effectContextId);
    FFX_RETURN_ON_ERROR(errorCode == FFX_OK, errorCode);
    context->backendContextCreated = true;

    // call out for device caps.
    FfxDeviceCapabilities capabilities;
    FFX_VALIDATE(context->contextDescription.backendInterface.fpGetDeviceCapabilities(&context->contextDescription.backendInterface, &capabilities));
    if (!capabilities.dataGraphOFSupported)
    {
        return ffxLogError(FFX_ERROR_NULL_DEVICE, "the device does not support data graph optical flow");
    }

    const bool outputGridSizeSupported = (contextDescription->gridSize & capabilities.supportedOutputGridSizes) != 0;
    // TODO: add check on hint grid size when we start using hint.
    // const bool hintGridSizeSupported   = (contextDescription->gridSize & capabilities.supportedHintGridSizes) != 0;
    if (!outputGridSizeSupported)
    {
        return ffxLogError(FFX_ERROR_NULL_DEVICE, "the requested optical flow grid size is not supported");
    }

    if ((contextDescription->resolution.width < capabilities.dataGraphOFMinWidth) ||
        (contextDescription->resolution.height < capabilities.dataGraphOFMinHeight) ||
        (contextDescription->resolution.width > capabilities.dataGraphOFMaxWidth) ||
        (contextDescription->resolution.height > capabilities.dataGraphOFMaxHeight))
    {
        return ffxLogError(FFX_ERROR_NULL_DEVICE, "the requested optical flow resolution is outside the device limits");
    }

    // set defaults
    context->firstExecution     = true;
    context->resourceFrameIndex = 0;

    // set inital view projection matrix
    memcpy(context->lastFrameViewProjection, contextDescription->initialViewProjection, sizeof(context->lastFrameViewProjection));

    const FfxDimensions2D        displaySize = contextDescription->resolution;
    const FfxDimensions2D        renderSize  = contextDescription->maxRenderSize;
    const FfxOpticalFlowGridSize gridSize    = contextDescription->gridSize;
    context->opticalFlowSize                 = GetOpticalFlowTextureSizeFromBlockSize(displaySize, GetOpticalFlowBlockSize(gridSize));
    context->mvDepthInputSize                = ffxNfruResolveMvDepthSize(renderSize, contextDescription->mvDepthSize);
    FfxDimensions2D mvDepthLane              = contextDescription->mvDepthSize;
    if (mvDepthLane.width == 0 && mvDepthLane.height == 0)
    {
        mvDepthLane = ffxNfruDeriveResolutionPlan(displaySize, renderSize, FFX_FRAMEINTERPOLATION_DATA_GRAPH_ALIGNMENT).mvDepthLane;
    }
    context->contextDescription.mvDepthSize               = mvDepthLane;
    context->computeMVHintsConstants.mv_depth_dims[0]     = mvDepthLane.width;
    context->computeMVHintsConstants.mv_depth_dims[1]     = mvDepthLane.height;
    context->computeMVHintsConstants.inv_mv_depth_dims[0] = 1.0f / static_cast<float>(mvDepthLane.width);
    context->computeMVHintsConstants.inv_mv_depth_dims[1] = 1.0f / static_cast<float>(mvDepthLane.height);

    // clear the SRV resources to NULL.
    memset(context->srvResources, 0, sizeof(context->srvResources));

    context->useMVHints = !!(context->contextDescription.flags & FFX_OPTICALFLOW_ENABLE_MV_HINTS);
    if (context->useMVHints)
    {
        const FfxResourceUsage mvHintsUsage =
            context->contextDescription.flags & FFX_OPTICALFLOW_ENABLE_MV_HINTS_COMPUTE ? FFX_RESOURCE_USAGE_UAV : FFX_RESOURCE_USAGE_RENDERTARGET;
        const FfxInternalResourceDescription mvHintsSurfaceDesc = {FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_COMPUTED_MOTION_VECTORS,
                                                                   "ArmOF_Computed_Motion_Vectors",
                                                                   FFX_RESOURCE_TYPE_TEXTURE2D,
                                                                   FfxResourceUsage(mvHintsUsage),
                                                                   FFX_SURFACE_FORMAT_R16G16_FLOAT,
                                                                   context->opticalFlowSize.width,
                                                                   context->opticalFlowSize.height,
                                                                   1,
                                                                   FFX_RESOURCE_FLAGS_NONE,
                                                                   {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED}};
        FFX_VALIDATE(createResourceFromDescription(context, &mvHintsSurfaceDesc));
    }

    // copy resources to uavResrouces list
    memcpy(context->uavResources, context->srvResources, sizeof(context->srvResources));

    {
        errorCode = createPipelineStates(context);
        FFX_RETURN_ON_ERROR(errorCode == FFX_OK, errorCode);
    }
    return FFX_OK;
}

static FfxErrorCode opticalFlowVkRelease(OpticalFlowContext_Private* context)
{
    FFX_ASSERT(context);
    if (!context->backendContextCreated)
    {
        return FFX_OK;
    }

    ffxSafeReleasePipeline(&context->contextDescription.backendInterface, &context->pipelineOpticalflow, context->effectContextId);
    ffxSafeReleasePipeline(&context->contextDescription.backendInterface, &context->pipelineComputeMVHints, context->effectContextId);
    ffxSafeReleasePipeline(&context->contextDescription.backendInterface, &context->pipelineFragmentMVHints, context->effectContextId);

    // unregister resources not created internally
    context->srvResources[FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_COLOR]      = {FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_NULL};
    context->srvResources[FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_DEPTH]      = {FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_NULL};
    context->srvResources[FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_RESULT]     = {FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_NULL};
    context->srvResources[FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_PREV_COLOR] = {FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_NULL};
    context->srvResources[FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_PREV_DEPTH] = {FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_NULL};

    // release internal resources
    for (int32_t currentResourceIndex = 0; currentResourceIndex < FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_COUNT; ++currentResourceIndex)
    {
        ffxSafeReleaseResource(&context->contextDescription.backendInterface, context->srvResources[currentResourceIndex], context->effectContextId);
    }

    context->backendContextCreated = false;
    FFX_VALIDATE(context->contextDescription.backendInterface.fpDestroyBackendContext(&context->contextDescription.backendInterface, context->effectContextId));

    return FFX_OK;
}

static FfxErrorCode scheduleDataGraph(OpticalFlowContext_Private* context, FfxPipelineState* pipeline, bool reset)
{
    FfxGpuJobDescription dispatchJob = {FFX_GPU_JOB_DATA_GRAPH};

    for (uint32_t currentShaderResourceViewIndex = 0; currentShaderResourceViewIndex < pipeline->srvTextureCount; ++currentShaderResourceViewIndex)
    {
        const uint32_t            currentResourceId = pipeline->srvTextureBindings[currentShaderResourceViewIndex].resourceIdentifier;
        const FfxResourceInternal currentResource   = context->srvResources[currentResourceId];
        dispatchJob.dataGraphJobDescription.srvTextures[currentShaderResourceViewIndex].resource = currentResource;
#ifdef FFX_DEBUG
        strncpy(dispatchJob.dataGraphJobDescription.srvTextures[currentShaderResourceViewIndex].name,
                pipeline->srvTextureBindings[currentShaderResourceViewIndex].name,
                FFX_RESOURCE_NAME_SIZE - 1);
        dispatchJob.dataGraphJobDescription.srvTextures[currentShaderResourceViewIndex].name[FFX_RESOURCE_NAME_SIZE - 1] = '\0';
#endif
    }

    for (uint32_t currentUnorderedAccessViewIndex = 0; currentUnorderedAccessViewIndex < pipeline->uavTextureCount; ++currentUnorderedAccessViewIndex)
    {
        const uint32_t currentResourceId = pipeline->uavTextureBindings[currentUnorderedAccessViewIndex].resourceIdentifier;
#ifdef FFX_DEBUG
        strncpy(dispatchJob.dataGraphJobDescription.uavTextures[currentUnorderedAccessViewIndex].name,
                pipeline->uavTextureBindings[currentUnorderedAccessViewIndex].name,
                FFX_RESOURCE_NAME_SIZE - 1);
        dispatchJob.dataGraphJobDescription.uavTextures[currentUnorderedAccessViewIndex].name[FFX_RESOURCE_NAME_SIZE - 1] = '\0';
#endif

        {
            const FfxResourceInternal currentResource                                                 = context->uavResources[currentResourceId];
            dispatchJob.dataGraphJobDescription.uavTextures[currentUnorderedAccessViewIndex].resource = currentResource;
            dispatchJob.dataGraphJobDescription.uavTextures[currentUnorderedAccessViewIndex].mip      = 0;
        }
    }
    if (!context->firstExecution && !reset)
    {
        // dispatchJob.dataGraphJobDescription.opticalFlowExecuteFlags = FFX_DATA_GRAPH_OPTICAL_FLOW_EXECUTE_INPUT_IS_PREVIOUS_REFERENCE;
    }
    dispatchJob.dataGraphJobDescription.pipeline = *pipeline;

    return context->contextDescription.backendInterface.fpScheduleGpuJob(&context->contextDescription.backendInterface, &dispatchJob);
}

static FfxErrorCode scheduleMVHintsAndCopyResources(
    OpticalFlowContext_Private* context, const FfxPipelineState* pipeline, uint32_t dispatchX, uint32_t dispatchY, const char* debugName)
{
    FfxGpuJobDescription dispatchJob = {FFX_GPU_JOB_COMPUTE};
    if (debugName != nullptr)
    {
#ifdef FFX_DEBUG
        strncpy(dispatchJob.jobLabel, debugName, FFX_RESOURCE_NAME_SIZE - 1);
        dispatchJob.jobLabel[FFX_RESOURCE_NAME_SIZE - 1] = '\0';
#endif
    }

    for (uint32_t currentShaderResourceViewIndex = 0; currentShaderResourceViewIndex < pipeline->srvTextureCount; ++currentShaderResourceViewIndex)
    {
        const uint32_t            currentResourceId = pipeline->srvTextureBindings[currentShaderResourceViewIndex].resourceIdentifier;
        const FfxResourceInternal currentResource   = context->srvResources[currentResourceId];
        dispatchJob.computeJobDescriptor.srvTextures[currentShaderResourceViewIndex].resource = currentResource;
#ifdef FFX_DEBUG
        strncpy(dispatchJob.computeJobDescriptor.srvTextures[currentShaderResourceViewIndex].name,
                pipeline->srvTextureBindings[currentShaderResourceViewIndex].name,
                FFX_RESOURCE_NAME_SIZE - 1);
        dispatchJob.computeJobDescriptor.srvTextures[currentShaderResourceViewIndex].name[FFX_RESOURCE_NAME_SIZE - 1] = '\0';
#endif
    }

    for (uint32_t currentUnorderedAccessViewIndex = 0; currentUnorderedAccessViewIndex < pipeline->uavTextureCount; ++currentUnorderedAccessViewIndex)
    {
        const uint32_t currentResourceId = pipeline->uavTextureBindings[currentUnorderedAccessViewIndex].resourceIdentifier;
#ifdef FFX_DEBUG
        strncpy(dispatchJob.computeJobDescriptor.uavTextureNames[currentUnorderedAccessViewIndex],
                pipeline->uavTextureBindings[currentUnorderedAccessViewIndex].name,
                FFX_RESOURCE_NAME_SIZE - 1);
        dispatchJob.computeJobDescriptor.uavTextureNames[currentUnorderedAccessViewIndex][FFX_RESOURCE_NAME_SIZE - 1] = '\0';
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
        strncpy(dispatchJob.computeJobDescriptor.srvTensors[currentTensorIndex].name,
                pipeline->srvTensorBindings[currentTensorIndex].name,
                FFX_RESOURCE_NAME_SIZE - 1);
        dispatchJob.computeJobDescriptor.srvTensors[currentTensorIndex].name[FFX_RESOURCE_NAME_SIZE - 1] = '\0';
#endif
    }

    for (uint32_t currentTensorIndex = 0; currentTensorIndex < pipeline->uavTensorCount; ++currentTensorIndex)
    {
        const uint32_t            currentResourceId                              = pipeline->uavTensorBindings[currentTensorIndex].resourceIdentifier;
        const FfxResourceInternal currentResource                                = context->uavResources[currentResourceId];
        dispatchJob.computeJobDescriptor.uavTensors[currentTensorIndex].resource = currentResource;
#ifdef FFX_DEBUG
        strncpy(dispatchJob.computeJobDescriptor.uavTensors[currentTensorIndex].name,
                pipeline->uavTensorBindings[currentTensorIndex].name,
                FFX_RESOURCE_NAME_SIZE - 1);
        dispatchJob.computeJobDescriptor.uavTensors[currentTensorIndex].name[FFX_RESOURCE_NAME_SIZE - 1] = '\0';
#endif
    }

    dispatchJob.computeJobDescriptor.dimensions[0] = dispatchX;
    dispatchJob.computeJobDescriptor.dimensions[1] = dispatchY;
    dispatchJob.computeJobDescriptor.dimensions[2] = 1;
    dispatchJob.computeJobDescriptor.pipeline      = *pipeline;

    for (uint32_t currentRootConstantIndex = 0; currentRootConstantIndex < pipeline->constCount; ++currentRootConstantIndex)
    {
#ifdef FFX_DEBUG
        strncpy(dispatchJob.computeJobDescriptor.cbNames[currentRootConstantIndex],
                pipeline->constantBufferBindings[currentRootConstantIndex].name,
                FFX_RESOURCE_NAME_SIZE - 1);
        dispatchJob.computeJobDescriptor.cbNames[currentRootConstantIndex][FFX_RESOURCE_NAME_SIZE - 1] = '\0';
#endif
        dispatchJob.computeJobDescriptor.cbs[currentRootConstantIndex] =
            context->constantBuffers[pipeline->constantBufferBindings[currentRootConstantIndex].resourceIdentifier];
    }

    return context->contextDescription.backendInterface.fpScheduleGpuJob(&context->contextDescription.backendInterface, &dispatchJob);
}

static FfxErrorCode scheduleFragmentMVHints(
    OpticalFlowContext_Private* context, const FfxPipelineState* pipeline, uint32_t width, uint32_t height, const char* debugName)
{
    FfxFragmentJobDescription jobDescriptor = {};

    for (uint32_t currentShaderResourceViewIndex = 0; currentShaderResourceViewIndex < pipeline->srvTextureCount; ++currentShaderResourceViewIndex)
    {
        const uint32_t            currentResourceId                        = pipeline->srvTextureBindings[currentShaderResourceViewIndex].resourceIdentifier;
        const FfxResourceInternal currentResource                          = context->srvResources[currentResourceId];
        jobDescriptor.srvTextures[currentShaderResourceViewIndex].resource = currentResource;
#ifdef FFX_DEBUG
        strncpy(jobDescriptor.srvTextures[currentShaderResourceViewIndex].name,
                pipeline->srvTextureBindings[currentShaderResourceViewIndex].name,
                FFX_RESOURCE_NAME_SIZE - 1);
        jobDescriptor.srvTextures[currentShaderResourceViewIndex].name[FFX_RESOURCE_NAME_SIZE - 1] = '\0';
#endif
    }

    for (uint32_t currentUnorderedAccessViewIndex = 0; currentUnorderedAccessViewIndex < pipeline->uavTextureCount; ++currentUnorderedAccessViewIndex)
    {
        const uint32_t currentResourceId = pipeline->uavTextureBindings[currentUnorderedAccessViewIndex].resourceIdentifier;
#ifdef FFX_DEBUG
        strncpy(jobDescriptor.uavTextures[currentUnorderedAccessViewIndex].name,
                pipeline->uavTextureBindings[currentUnorderedAccessViewIndex].name,
                FFX_RESOURCE_NAME_SIZE - 1);
        jobDescriptor.uavTextures[currentUnorderedAccessViewIndex].name[FFX_RESOURCE_NAME_SIZE - 1] = '\0';
#endif

        const FfxResourceInternal currentResource                           = context->uavResources[currentResourceId];
        jobDescriptor.uavTextures[currentUnorderedAccessViewIndex].resource = currentResource;
        jobDescriptor.uavTextures[currentUnorderedAccessViewIndex].mip      = 0;
    }

    jobDescriptor.viewport[0] = width;
    jobDescriptor.viewport[1] = height;
    jobDescriptor.pipeline    = *pipeline;

    for (uint32_t currentRTIndex = 0; currentRTIndex < pipeline->rtCount; ++currentRTIndex)
    {
        const uint32_t            currentResourceId       = pipeline->rtBindings[currentRTIndex].resourceIdentifier;
        const FfxResourceInternal currentResource         = context->uavResources[currentResourceId];
        jobDescriptor.rtTextures[currentRTIndex].resource = currentResource;
#ifdef FFX_DEBUG
        strncpy(jobDescriptor.rtTextures[currentRTIndex].name, pipeline->rtBindings[currentRTIndex].name, FFX_RESOURCE_NAME_SIZE - 1);
        jobDescriptor.rtTextures[currentRTIndex].name[FFX_RESOURCE_NAME_SIZE - 1] = '\0';
#endif
    }

    for (uint32_t currentRootConstantIndex = 0; currentRootConstantIndex < pipeline->constCount; ++currentRootConstantIndex)
    {
#ifdef FFX_DEBUG
        strncpy(jobDescriptor.cbNames[currentRootConstantIndex], pipeline->constantBufferBindings[currentRootConstantIndex].name, FFX_RESOURCE_NAME_SIZE - 1);
        jobDescriptor.cbNames[currentRootConstantIndex][FFX_RESOURCE_NAME_SIZE - 1] = '\0';
#endif
        jobDescriptor.cbs[currentRootConstantIndex] = context->constantBuffers[pipeline->constantBufferBindings[currentRootConstantIndex].resourceIdentifier];
    }

    FfxGpuJobDescription dispatchJob = {FFX_GPU_JOB_FRAGMENT};
    if (debugName != nullptr)
    {
#ifdef FFX_DEBUG
        strncpy(dispatchJob.jobLabel, debugName, FFX_RESOURCE_NAME_SIZE - 1);
        dispatchJob.jobLabel[FFX_RESOURCE_NAME_SIZE - 1] = '\0';
#endif
    }
    dispatchJob.fragmentJobDescriptor = jobDescriptor;

    return context->contextDescription.backendInterface.fpScheduleGpuJob(&context->contextDescription.backendInterface, &dispatchJob);
}

static FfxErrorCode ArmOpticalFlowDispatch(OpticalFlowContext_Private* context, const FfxOpticalFlowDispatchDescription* params, FfxNfruOpticalFlowTrace* trace)
{
    const auto setTraceStage = [trace](FfxNfruOfStage stage) {
        if (trace)
        {
            trace->failedStage = stage;
        }
    };

    // take a short cut to the command list
    FfxCommandList commandList = params->commandList;

    bool       resourcesRegistered = false;
    const auto unregisterResources = [&]() -> FfxErrorCode {
        if (!resourcesRegistered)
        {
            return FFX_OK;
        }
        return context->contextDescription.backendInterface.fpUnregisterResources(
            &context->contextDescription.backendInterface, commandList, context->effectContextId);
    };
    const auto registerResource = [&](const FfxResource& resource, FfxResourceInternal* internalResource) -> FfxErrorCode {
        const FfxErrorCode result = context->contextDescription.backendInterface.fpRegisterResource(
            &context->contextDescription.backendInterface, &resource, context->effectContextId, internalResource);
        resourcesRegistered |= result == FFX_OK;
        return result;
    };
    const auto returnAfterCleanup = [&](FfxErrorCode result) -> FfxErrorCode {
        (void)unregisterResources();
        return result;
    };

    setTraceStage(FfxNfruOfStage::RegisterResources);
    FfxErrorCode errorCode = registerResource(params->color, &context->srvResources[FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_COLOR]);
    if (errorCode != FFX_OK)
    {
        return returnAfterCleanup(errorCode);
    }
    errorCode = registerResource(params->depth, &context->srvResources[FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_DEPTH]);
    if (errorCode != FFX_OK)
    {
        return returnAfterCleanup(errorCode);
    }
    errorCode = registerResource(params->depthTm1, &context->srvResources[FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_PREV_DEPTH]);
    if (errorCode != FFX_OK)
    {
        return returnAfterCleanup(errorCode);
    }
    errorCode = registerResource(params->colorTm1, &context->srvResources[FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_PREV_COLOR]);
    if (errorCode != FFX_OK)
    {
        return returnAfterCleanup(errorCode);
    }
    errorCode = registerResource(params->opticalFlowVector, &context->uavResources[FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_RESULT]);
    if (errorCode != FFX_OK)
    {
        return returnAfterCleanup(errorCode);
    }

    // On first execution or reset, clear previous depth buffers
    if (params->reset)
    {
        setTraceStage(FfxNfruOfStage::ClearHistory);
        FfxGpuJobDescription clearJob = {FFX_GPU_JOB_CLEAR_FLOAT};
        const float          clearValuesToZeroFloat[]{0.f, 0.f, 0.f, 0.f};
        memcpy(clearJob.clearJobDescriptor.color, clearValuesToZeroFloat, 4 * sizeof(float));

        constexpr uint32_t resources_to_clear[] = {FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_PREV_DEPTH, FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_PREV_COLOR};
        for (uint32_t i = 0; i < FFX_COUNTOF(resources_to_clear); ++i)
        {
            clearJob.clearJobDescriptor.target = context->srvResources[resources_to_clear[i]];
            errorCode = context->contextDescription.backendInterface.fpScheduleGpuJob(&context->contextDescription.backendInterface, &clearJob);
            if (errorCode != FFX_OK)
            {
                return returnAfterCleanup(errorCode);
            }
        }
    }

    // Compute MV hints from depth when enabled.
    if (context->useMVHints)
    {
        setTraceStage(FfxNfruOfStage::MotionVectorHints);
        // Set constant buffer for MV hints computation
        const FfxDimensions2D depthResolution{params->depthTm1.description.width, params->depthTm1.description.height};

        context->computeMVHintsConstants.output_dims[0]    = context->opticalFlowSize.width;
        context->computeMVHintsConstants.output_dims[1]    = context->opticalFlowSize.height;
        context->computeMVHintsConstants.depth_size[0]     = depthResolution.width;
        context->computeMVHintsConstants.depth_size[1]     = depthResolution.height;
        context->computeMVHintsConstants.depth_size_rcp[0] = 1.0f / (float)depthResolution.width;
        context->computeMVHintsConstants.depth_size_rcp[1] = 1.0f / (float)depthResolution.height;
        context->computeMVHintsConstants.color_size[0]     = params->color.description.width;
        context->computeMVHintsConstants.color_size[1]     = params->color.description.height;

        // Calculate reprojection matrix TM1 -> TP1 (previous to current)
        FfxFloat32x4x4 invViewProjTM1;
        MatrixInvert4x4(context->lastFrameViewProjection, invViewProjTM1);
        MatrixMul4x4(params->viewProjection, invViewProjTM1, context->computeMVHintsConstants.motion_matrix_m1p1);

        // Stage constant buffer
        errorCode =
            context->contextDescription.backendInterface.fpStageConstantBufferDataFunc(&context->contextDescription.backendInterface,
                                                                                       &context->computeMVHintsConstants,
                                                                                       sizeof(context->computeMVHintsConstants),
                                                                                       &context->constantBuffers[FFX_OPTICALFLOW_CONSTANTBUFFER_IDENTIFIER]);
        if (errorCode != FFX_OK)
        {
            return returnAfterCleanup(errorCode);
        }

        if (!context->useMVHintsCompute)
        {
            // Fragment path: dispatch at OF dims (the MV hints render target size)
            errorCode = scheduleFragmentMVHints(
                context, &context->pipelineFragmentMVHints, context->opticalFlowSize.width, context->opticalFlowSize.height, "ArmOF_MVHints");
        }
        else
        {
            // Compute path: dispatch at OF dims
            uint32_t dispatchX = (context->opticalFlowSize.width + 15) / 16;
            uint32_t dispatchY = (context->opticalFlowSize.height + 15) / 16;
            errorCode          = scheduleMVHintsAndCopyResources(context, &context->pipelineComputeMVHints, dispatchX, dispatchY, "ArmOF_MVHints");
        }
        if (errorCode != FFX_OK)
        {
            return returnAfterCleanup(errorCode);
        }
    }

    // Schedule optical flow data graph, with the MV hint input bound only when enabled.
    setTraceStage(FfxNfruOfStage::DataGraph);
    errorCode = scheduleDataGraph(context, &context->pipelineOpticalflow, params->reset);
    if (errorCode != FFX_OK)
    {
        return returnAfterCleanup(errorCode);
    }

    setTraceStage(FfxNfruOfStage::RecordCommands);
    errorCode =
        context->contextDescription.backendInterface.fpExecuteGpuJobs(&context->contextDescription.backendInterface, commandList, context->effectContextId);
    if (errorCode != FFX_OK)
    {
        return returnAfterCleanup(errorCode);
    }

    // Release dynamic resources
    setTraceStage(FfxNfruOfStage::Unregister);
    errorCode = unregisterResources();
    if (errorCode != FFX_OK)
    {
        return errorCode;
    }

    // Commit temporal state only after the backend accepted all recorded work.
    memcpy(context->lastFrameViewProjection, params->viewProjection, sizeof(context->lastFrameViewProjection));

    context->firstExecution = false;
    // Toggle between 0 and 1 to choose the correct internal resources
    context->resourceFrameIndex ^= 1;

    setTraceStage(FfxNfruOfStage::None);
    return FFX_OK;
}

static FfxErrorCode opticalFlowContextCreateInternal(FfxOpticalFlowContext*            context,
                                                     FfxOpticalFlowContextDescription* contextDescription,
                                                     const FfxDimensions2D*            mvDepthInputSize)
{
    // check pointers are valid.
    FFX_RETURN_ON_ERROR(context, FFX_ERROR_INVALID_POINTER);
    FFX_RETURN_ON_ERROR(contextDescription, FFX_ERROR_INVALID_POINTER);

    constexpr uint32_t maxShaderDimension = static_cast<uint32_t>(std::numeric_limits<int32_t>::max());
    if (mvDepthInputSize && (!mvDepthInputSize->width || !mvDepthInputSize->height || mvDepthInputSize->width > maxShaderDimension ||
                             mvDepthInputSize->height > maxShaderDimension))
    {
        return ffxLogError(FFX_ERROR_INVALID_ARGUMENT, "frame-generation depth input dimensions must fit the signed shader range");
    }

    // zero context memory
    memset(context, 0, sizeof(FfxOpticalFlowContext));

    // validate that all callbacks are set for the interface
    FFX_RETURN_ON_ERROR(contextDescription->backendInterface.fpCreateBackendContext, FFX_ERROR_INCOMPLETE_INTERFACE);
    FFX_RETURN_ON_ERROR(contextDescription->backendInterface.fpDestroyBackendContext, FFX_ERROR_INCOMPLETE_INTERFACE);
    FFX_RETURN_ON_ERROR(contextDescription->backendInterface.fpGetDeviceCapabilities, FFX_ERROR_INCOMPLETE_INTERFACE);
    const bool usesMvHints         = (contextDescription->flags & FFX_OPTICALFLOW_ENABLE_MV_HINTS) != 0;
    const bool usesComputeMvHints  = usesMvHints && (contextDescription->flags & FFX_OPTICALFLOW_ENABLE_MV_HINTS_COMPUTE) != 0;
    const bool usesFragmentMvHints = usesMvHints && !usesComputeMvHints;
    FFX_RETURN_ON_ERROR(!usesFragmentMvHints || contextDescription->backendInterface.fpCreateGraphicsPipeline, FFX_ERROR_INCOMPLETE_INTERFACE);
    FFX_RETURN_ON_ERROR(!usesComputeMvHints || contextDescription->backendInterface.fpCreateComputePipeline, FFX_ERROR_INCOMPLETE_INTERFACE);

    // if a scratch buffer is declared, then we must have a size
    if (contextDescription->backendInterface.scratchBuffer)
    {
        FFX_RETURN_ON_ERROR(contextDescription->backendInterface.scratchBufferSize, FFX_ERROR_INCOMPLETE_INTERFACE);
    }

    if (!contextDescription->resolution.width || !contextDescription->resolution.height || !contextDescription->maxRenderSize.width ||
        !contextDescription->maxRenderSize.height)
    {
        return ffxLogError(FFX_ERROR_INVALID_ARGUMENT, "context creation requires nonzero resolution and maxRenderSize dimensions");
    }

    if (contextDescription->resolution.width > maxShaderDimension || contextDescription->resolution.height > maxShaderDimension ||
        contextDescription->maxRenderSize.width > maxShaderDimension || contextDescription->maxRenderSize.height > maxShaderDimension)
    {
        return ffxLogError(FFX_ERROR_INVALID_ARGUMENT, "context creation requires resolution and maxRenderSize dimensions within the shader range");
    }

    const bool useDerivedMvDepthLane = contextDescription->mvDepthSize.width == 0 && contextDescription->mvDepthSize.height == 0;
    if (!useDerivedMvDepthLane && (!contextDescription->mvDepthSize.width || !contextDescription->mvDepthSize.height))
    {
        return ffxLogError(FFX_ERROR_INVALID_ARGUMENT, "mvDepthSize must be {0, 0} or contain two nonzero dimensions");
    }

    FfxDimensions2D mvDepthLane = contextDescription->mvDepthSize;
    if (useDerivedMvDepthLane)
    {
        mvDepthLane =
            ffxNfruDeriveResolutionPlan(contextDescription->resolution, contextDescription->maxRenderSize, FFX_FRAMEINTERPOLATION_DATA_GRAPH_ALIGNMENT)
                .mvDepthLane;
    }
    if (!mvDepthLane.width || !mvDepthLane.height || mvDepthLane.width > maxShaderDimension || mvDepthLane.height > maxShaderDimension)
    {
        return ffxLogError(FFX_ERROR_INVALID_ARGUMENT, "context creation requires positive mvDepthSize dimensions within the shader range");
    }

    // ensure the context is large enough for the internal context.
    FFX_STATIC_ASSERT(sizeof(FfxOpticalFlowContext) >= sizeof(OpticalFlowContext_Private));

    // create the context.
    OpticalFlowContext_Private* contextPrivate = (OpticalFlowContext_Private*)(context);
    const FfxErrorCode          errorCode      = opticalFlowVkCreate(contextPrivate, contextDescription);
    if (errorCode == FFX_OK && mvDepthInputSize)
    {
        contextPrivate->mvDepthInputSize = *mvDepthInputSize;
    }
    if (errorCode != FFX_OK)
    {
        ffxLogError(errorCode, "context creation");
        (void)opticalFlowVkRelease(contextPrivate);
    }

    return errorCode;
}

FfxErrorCode ffxOpticalFlowContextCreate(FfxOpticalFlowContext* context, FfxOpticalFlowContextDescription* contextDescription)
{
    return opticalFlowContextCreateInternal(context, contextDescription, nullptr);
}

FFX_API FfxErrorCode ffxOpticalFlowContextCreateForFrameGeneration(FfxOpticalFlowContext* context, FfxOpticalFlowContextDescription* contextDescription)
{
    FfxScopedLogSuppression suppression;
    return ffxOpticalFlowContextCreate(context, contextDescription);
}

FFX_API FfxErrorCode ffxOpticalFlowContextCreateForFrameGenerationWithDepthInput(FfxOpticalFlowContext*            context,
                                                                                 FfxOpticalFlowContextDescription* contextDescription,
                                                                                 FfxDimensions2D                   mvDepthInputSize)
{
    FfxScopedLogSuppression suppression;
    return opticalFlowContextCreateInternal(context, contextDescription, &mvDepthInputSize);
}

FfxErrorCode ffxOpticalFlowContextGetGpuMemoryUsage(FfxOpticalFlowContext* context, FfxEffectMemoryUsage* vramUsage)
{
    FFX_RETURN_ON_ERROR(context, FFX_ERROR_INVALID_POINTER);
    FFX_RETURN_ON_ERROR(vramUsage, FFX_ERROR_INVALID_POINTER);

    OpticalFlowContext_Private* contextPrivate = (OpticalFlowContext_Private*)(context);
    FFX_RETURN_ON_ERROR(contextPrivate->device, FFX_ERROR_NULL_DEVICE);
    FFX_RETURN_ON_ERROR(contextPrivate->contextDescription.backendInterface.fpGetEffectGpuMemoryUsage, FFX_ERROR_INCOMPLETE_INTERFACE);

    const FfxErrorCode errorCode = contextPrivate->contextDescription.backendInterface.fpGetEffectGpuMemoryUsage(
        &contextPrivate->contextDescription.backendInterface, contextPrivate->effectContextId, vramUsage);
    FFX_RETURN_ON_ERROR(errorCode == FFX_OK, errorCode);

    return FFX_OK;
}

FfxErrorCode ffxOpticalFlowContextDestroy(FfxOpticalFlowContext* context)
{
    FFX_RETURN_ON_ERROR(context, FFX_ERROR_INVALID_POINTER);

    // destroy the context.
    OpticalFlowContext_Private* contextPrivate = (OpticalFlowContext_Private*)(context);
    const FfxErrorCode          errorCode      = opticalFlowVkRelease(contextPrivate);
    FFX_RETURN_ON_ERROR(errorCode == FFX_OK, errorCode);
    return FFX_OK;
}

FFX_API FfxErrorCode ffxOpticalFlowContextDestroyForFrameGeneration(FfxOpticalFlowContext* context)
{
    FfxScopedLogSuppression suppression;
    return ffxOpticalFlowContextDestroy(context);
}

FfxOpticalFlowGridSize ffxGetDefaultDataGraphOpticalFlowGridSize(FfxInterface& backendInterface)
{
    FFX_ASSERT(backendInterface.devCapInitialized);

    FfxDeviceCapabilities capabilities;
    backendInterface.fpGetDeviceCapabilities(&backendInterface, &capabilities);
    const auto supportedSizes = capabilities.supportedOutputGridSizes;

    FfxOpticalFlowGridSize defaultGridSize = FFX_OPTICAL_FLOW_GRID_SIZE_UNKNOWN;
    if (supportedSizes & FFX_OPTICAL_FLOW_GRID_SIZE_4X4)
    {
        defaultGridSize = FFX_OPTICAL_FLOW_GRID_SIZE_4X4;
    }
    else if (supportedSizes & FFX_OPTICAL_FLOW_GRID_SIZE_8X8)
    {
        defaultGridSize = FFX_OPTICAL_FLOW_GRID_SIZE_8X8;
    }
    else if (supportedSizes & FFX_OPTICAL_FLOW_GRID_SIZE_2X2)
    {
        defaultGridSize = FFX_OPTICAL_FLOW_GRID_SIZE_2X2;
    }
    else
    {
        defaultGridSize = FFX_OPTICAL_FLOW_GRID_SIZE_1X1;
    }

    return defaultGridSize;
}

bool ffxOpticalFlowGridSizeSupported(FfxInterface& backendInterface, const FfxOpticalFlowGridSize gridSize)
{
    FFX_ASSERT(backendInterface.devCapInitialized);

    FfxDeviceCapabilities capabilities;
    backendInterface.fpGetDeviceCapabilities(&backendInterface, &capabilities);
    const auto supportedSizes = capabilities.supportedOutputGridSizes;

    return (gridSize & supportedSizes) != 0;
}

FfxOpticalFlowGridSize ffxOpticalFlowGetGridSize(FfxOpticalFlowContext* context)
{
    OpticalFlowContext_Private* contextPrivate = (OpticalFlowContext_Private*)(context);
    const auto                  gridSize       = contextPrivate ? contextPrivate->contextDescription.gridSize : FFX_OPTICAL_FLOW_GRID_SIZE_UNKNOWN;
    return gridSize;
}

FFX_API FfxErrorCode ffxOpticalFlowGetSharedResourceDescriptions(FfxOpticalFlowContext* context, FfxOpticalFlowSharedResourceDescriptions* SharedResources)
{
    FFX_RETURN_ON_ERROR(context, FFX_ERROR_INVALID_POINTER);
    FFX_RETURN_ON_ERROR(SharedResources, FFX_ERROR_INVALID_POINTER);

    OpticalFlowContext_Private* contextPrivate = (OpticalFlowContext_Private*)(context);
    FFX_ASSERT(contextPrivate);

    SharedResources->opticalFlowVector = {FFX_HEAP_TYPE_DEFAULT,
                                          {FFX_RESOURCE_TYPE_TEXTURE2D,
                                           FFX_SURFACE_FORMAT_R16G16_FLOAT,
                                           contextPrivate->opticalFlowSize.width,
                                           contextPrivate->opticalFlowSize.height,
                                           1,
                                           1,
                                           FFX_RESOURCE_FLAGS_NONE,
                                           FfxResourceUsage(FFX_RESOURCE_USAGE_UAV)},
                                          FFX_RESOURCE_STATE_GENERIC_UAV,
                                          "ARM_OpticalFlow_Result",
                                          0,
                                          {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED}};
    SharedResources->depthTm1          = {FFX_HEAP_TYPE_DEFAULT,
                                 {FFX_RESOURCE_TYPE_TEXTURE2D,
                                  FFX_SURFACE_FORMAT_R32_FLOAT,
                                  contextPrivate->contextDescription.mvDepthSize.width,
                                  contextPrivate->contextDescription.mvDepthSize.height,
                                  1,
                                  1,
                                  FFX_RESOURCE_FLAGS_NONE,
                                  FfxResourceUsage(FFX_RESOURCE_USAGE_RENDERTARGET)},
                                 FFX_RESOURCE_STATE_GENERIC_READ,
                                 "ARM_DepthTm1",
                                 0,
                                 {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED}};
    SharedResources->depthTm1Next      = SharedResources->depthTm1;
    SharedResources->depthTm1Next.name = "ARM_DepthTm1_Next";
    SharedResources->colorTm1          = {FFX_HEAP_TYPE_DEFAULT,
                                 {FFX_RESOURCE_TYPE_TEXTURE2D,
                                  contextPrivate->contextDescription.backBufferFormat,
                                  contextPrivate->contextDescription.resolution.width,
                                  contextPrivate->contextDescription.resolution.height,
                                  1,
                                  1,
                                  FFX_RESOURCE_FLAGS_NONE,
                                  FfxResourceUsage(FFX_RESOURCE_USAGE_UAV)},
                                 FFX_RESOURCE_STATE_GENERIC_UAV,
                                 "ARM_ColorTm1",
                                 0,
                                 {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED}};
    return FFX_OK;
}

FFX_API FfxErrorCode ffxOpticalFlowGetSharedResourceDescriptionsForFrameGeneration(FfxOpticalFlowContext*                    context,
                                                                                   FfxOpticalFlowSharedResourceDescriptions* sharedResources)
{
    FfxScopedLogSuppression suppression;
    return ffxOpticalFlowGetSharedResourceDescriptions(context, sharedResources);
}

FFX_API FfxErrorCode ffxOpticalFlowContextDispatchWithTrace(FfxOpticalFlowContext*             context,
                                                            FfxOpticalFlowDispatchDescription* dispatchParams,
                                                            FfxNfruOpticalFlowTrace*           trace)
{
    FfxScopedLogSuppression suppression(trace != nullptr);
    if (trace)
    {
        *trace = {FfxNfruOfStage::Validation};
    }
    FFX_RETURN_ON_ERROR(context, FFX_ERROR_INVALID_POINTER);
    FFX_RETURN_ON_ERROR(dispatchParams, FFX_ERROR_INVALID_POINTER);
    FFX_RETURN_ON_ERROR(dispatchParams->commandList, FFX_ERROR_INVALID_POINTER);
    FFX_RETURN_ON_ERROR(dispatchParams->color.resource != NULL, FFX_ERROR_INVALID_POINTER);
    FFX_RETURN_ON_ERROR(dispatchParams->depth.resource != NULL, FFX_ERROR_INVALID_POINTER);
    FFX_RETURN_ON_ERROR(dispatchParams->color.description.type == FFX_RESOURCE_TYPE_TEXTURE2D, FFX_ERROR_INVALID_ARGUMENT);
    FFX_RETURN_ON_ERROR(dispatchParams->opticalFlowVector.resource != NULL, FFX_ERROR_INVALID_POINTER);
    FFX_RETURN_ON_ERROR(dispatchParams->depthTm1.resource != NULL, FFX_ERROR_INVALID_POINTER);
    FFX_RETURN_ON_ERROR(dispatchParams->colorTm1.resource != NULL, FFX_ERROR_INVALID_POINTER);

    OpticalFlowContext_Private* contextPrivate = (OpticalFlowContext_Private*)(context);

    FFX_RETURN_ON_ERROR(contextPrivate->device, FFX_ERROR_NULL_DEVICE);
    FFX_RETURN_ON_ERROR(dispatchParams->color.description.width == contextPrivate->contextDescription.resolution.width, FFX_ERROR_INVALID_ARGUMENT);
    FFX_RETURN_ON_ERROR(dispatchParams->color.description.height == contextPrivate->contextDescription.resolution.height, FFX_ERROR_INVALID_ARGUMENT);

    if (contextPrivate->useMVHints)
    {
        const FfxDimensions2D inputSize    = contextPrivate->mvDepthInputSize;
        const FfxDimensions2D laneSize     = contextPrivate->contextDescription.mvDepthSize;
        const bool            depthMatches = dispatchParams->depth.description.type == FFX_RESOURCE_TYPE_TEXTURE2D &&
                                  dispatchParams->depth.description.width == inputSize.width && dispatchParams->depth.description.height == inputSize.height;
        const bool previousDepthMatches =
            dispatchParams->depthTm1.description.type == FFX_RESOURCE_TYPE_TEXTURE2D &&
            ((dispatchParams->depthTm1.description.width == inputSize.width && dispatchParams->depthTm1.description.height == inputSize.height) ||
             (dispatchParams->depthTm1.description.width == laneSize.width && dispatchParams->depthTm1.description.height == laneSize.height));
        if (!depthMatches || !previousDepthMatches)
        {
            return ffxLogError(FFX_ERROR_INVALID_ARGUMENT,
                               "depth must match the input size and depthTm1 must match the input or resolved lane when hints are enabled");
        }
    }

    const FfxErrorCode errorCode = ArmOpticalFlowDispatch(contextPrivate, dispatchParams, trace);
    if (errorCode != FFX_OK && !trace)
    {
        return ffxLogError(errorCode, "dispatch");
    }
    return errorCode;
}

FfxErrorCode ffxOpticalFlowContextDispatch(FfxOpticalFlowContext* context, FfxOpticalFlowDispatchDescription* dispatchParams)
{
    return ffxOpticalFlowContextDispatchWithTrace(context, dispatchParams, nullptr);
}

FfxDimensions2D ffxOpticalFlowGetSize(FfxOpticalFlowContext* context)
{
    OpticalFlowContext_Private* contextPrivate = (OpticalFlowContext_Private*)(context);
    return contextPrivate ? contextPrivate->opticalFlowSize : FfxDimensions2D{0, 0};
}