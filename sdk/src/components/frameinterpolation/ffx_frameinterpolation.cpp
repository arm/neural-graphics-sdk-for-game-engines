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

#include <algorithm>
#include <atomic>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
#include "FidelityFX/host/ffx_frameinterpolation.h"

#define FFX_CPU 1

#include "FidelityFX/gpu/ffx_core.h"
#include "ffx_object_management.h"

#include "ffx_logging.h"

#include "ffx_frameinterpolation_private.h"
#include "ffx_frameinterpolation_trace.h"

static constexpr const char* ffxLogComponentName()
{
    return "NFRU";
}

static std::atomic<bool> g_nfruVersionLogged{false};

FFX_API void ffxLogNfruVersion(FfxLogMessageCallback callback) noexcept
{
    constexpr size_t messageCapacity = 96;
    char             message[messageCapacity];
    std::snprintf(message,
                  sizeof(message),
                  "event=versions nfru_component_version=%d.%d.%d",
                  FFX_FRAMEINTERPOLATION_VERSION_MAJOR,
                  FFX_FRAMEINTERPOLATION_VERSION_MINOR,
                  FFX_FRAMEINTERPOLATION_VERSION_PATCH);
    ffxLogMessageOnceToSink(ffxLogComponentName(), FFX_MESSAGE_TYPE_INFO, message, g_nfruVersionLogged, callback);
}

using NfruCpuClock = std::chrono::steady_clock;

static double elapsedMicroseconds(NfruCpuClock::time_point begin, NfruCpuClock::time_point end)
{
    return std::chrono::duration<double, std::micro>(end - begin).count();
}

struct DynamicResourceCleanup
{
    FfxInterface*  backend;
    FfxCommandList commandList;
    FfxUInt32      effectContextId;
    bool           active;

    ~DynamicResourceCleanup()
    {
        if (active)
        {
            (void)backend->fpUnregisterResources(backend, commandList, effectContextId);
        }
    }
};

/* TODO: This is a chicken-and-egg problem: we need to create the data graph pipeline in order to know the alignment required for the corresponding session;
  but to create the data graph pipeline, we first need to know the alignment to determine the padded size and provide the size to spv optimizer.
*/
static constexpr float    MV_SIMILARITY_THRESHOLD_DEFAULT                                        = 0.25f;
static constexpr float    MV_SIMILARITY_NOISE_THRESHOLD_DEFAULT                                  = 0.125f;
static constexpr float    MV_SIMILARITY_RELATIVE_THRESHOLD_DEFAULT                               = 0.002f;
static constexpr uint32_t FFX_FRAMEINTERPOLATION_INPAINT_LEVEL_COUNT                             = 5;
static constexpr uint32_t INPAINT_LEVEL_RESOURCE_IDS[FFX_FRAMEINTERPOLATION_INPAINT_LEVEL_COUNT] = {
    FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_INPAINT_LEVEL_0,
    FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_INPAINT_LEVEL_1,
    FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_INPAINT_LEVEL_2,
    FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_INPAINT_LEVEL_3,
    FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_INPAINT_LEVEL_4,
};
static constexpr const char* INPAINT_LEVEL_RESOURCE_NAMES[FFX_FRAMEINTERPOLATION_INPAINT_LEVEL_COUNT] = {
    "ArmFI_Inpaint_Level_0",
    "ArmFI_Inpaint_Level_1",
    "ArmFI_Inpaint_Level_2",
    "ArmFI_Inpaint_Level_3",
    "ArmFI_Inpaint_Level_4",
};

// lists to map shader resource bindpoint name to resource identifier
typedef struct ResourceBinding
{
    uint32_t index;
    char     name[64];
} ResourceBinding;

static constexpr ResourceBinding srvResourceBindingTable[] = {
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_DEPTH, "r_depth_tp1"},
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_MOTION_VECTORS, "r_motion_tp1"},
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_OPTICAL_FLOW_VECTOR, "r_flow_tm1"},
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_DYNAMIC_MASK_TP1, "r_dynamic_mask_tp1"},
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_DYNAMIC_MASK_TM1, "r_dynamic_mask_tm1"},
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_PREVIOUS_INTERPOLATION_SOURCE, "r_previous_interpolation_source"},
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_CURRENT_INTERPOLATION_SOURCE, "r_current_interpolation_source"},
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_PRESENT_BACKBUFFER, "r_present_backbuffer"},
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_OUTPUT, "r_output"},
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_WARP_MOTION_QDATA_TP1, "r_warped_motion_qdata_tp1"},
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_WARP_FLOW_QDATA_TM1, "r_warped_flow_qdata_tm1"},
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_WARP_FILLED_FLOW_TM1, "r_warped_filled_flow_tm1"},
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_IN_TENSOR, "r_in_tensor"},
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_OUT_PARAMS_TENSOR, "r_out_params_tensor"},
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_OUT_PARAMS_GATE_TENSOR, "r_out_params_gate_tensor"},
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_INPAINT_SEED, "r_inpaint_seed"},
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_INPAINT_LEVEL_0, "r_inpaint_level_0"},
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_INPAINT_LEVEL_1, "r_inpaint_level_1"},
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_INPAINT_LEVEL_2, "r_inpaint_level_2"},
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_INPAINT_LEVEL_3, "r_inpaint_level_3"},
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_INPAINT_LEVEL_4, "r_inpaint_level_4"},
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_INPAINT_DOWNSAMPLE_SOURCE, "r_inpaint_source"},
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_INPAINT_RESULT, "r_inpaint_result"},
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_DEPTH_TM1, "r_depth_tm1"},
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_DEPTH_TM1_NEXT, "r_depth_tm1_next"},
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_OF_COLOUR_P1, "r_of_colour_p1"},
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_OF_COLOUR_M1, "r_of_colour_m1"},
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_PREPROCESS_COLOUR_P1, "r_preprocess_colour_p1"},
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_PREPROCESS_COLOUR_M1, "r_preprocess_colour_m1"},
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_REDUCED_MOTION_P1, "r_reduced_motion_p1"},
    // data graph duplicates
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_IN_TENSOR, "Resource_0_input"},
    // FRU duplicates, work around for spirv-reflect bug.
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_IN_TENSOR, "InputTensorBuffer_t"},
};

static constexpr ResourceBinding uavResourceBindingTable[] = {
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_OUTPUT, "rw_output"},
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_DYNAMIC_MASK_TP1, "rw_dynamic_mask_tp1"},
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_WARP_MOTION_QDATA_TP1, "rw_warped_motion_qdata_tp1"},
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_WARP_FLOW_QDATA_TM1, "rw_warped_flow_qdata_tm1"},
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_WARP_FILLED_FLOW_TM1, "rw_warped_filled_flow_tm1"},
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_IN_TENSOR, "rw_in_tensor"},
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_OUT_PARAMS_TENSOR, "rw_out_params_tensor"},
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_OUT_PARAMS_GATE_TENSOR, "rw_out_params_gate_tensor"},
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_DEPTH_TM1, "rw_depth_tm1"},
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_DEPTH_TM1_NEXT, "rw_depth_tm1_next"},
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_OF_COLOUR_P1, "rw_of_colour_p1"},
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_PREPROCESS_COLOUR_P1, "rw_preprocess_colour_p1"},
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_REDUCED_MOTION_P1, "rw_reduced_motion_p1"},
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_INPAINT_SEED, "rw_inpaint_seed"},
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_INPAINT_DOWNSAMPLE_DESTINATION, "rw_inpaint_destination"},
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_INPAINT_RESULT, "rw_inpaint_result"},
    // data graph duplicates
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_OUT_PARAMS_TENSOR, "Resource_1_output"},
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_OUT_PARAMS_GATE_TENSOR, "Resource_2_output"},
    // FRU duplicates, work around for spirv-reflect bug.
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_IN_TENSOR, "InputTensorBuffer_t"},
    // Async ping-pong backup written by the postprocess pass.
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_PREVIOUS_INTERPOLATION_SOURCE_BACKUP, "rw_previous_interpolation_source"},
};

static constexpr ResourceBinding rtResourceBindingTable[] = {
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_DYNAMIC_MASK_TP1, "rw_dynamic_mask_tp1"},
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_WARP_FILLED_FLOW_TM1, "rw_warped_filled_flow_tm1"},
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_OUTPUT, "rw_output"},
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_DEPTH_TM1, "rw_depth_tm1"},
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_DEPTH_TM1_NEXT, "rw_depth_tm1_next"},
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_OF_COLOUR_P1, "rw_of_colour_p1"},
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_PREPROCESS_COLOUR_P1, "rw_preprocess_colour_p1"},
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_REDUCED_MOTION_P1, "rw_reduced_motion_p1"},
    // Async ping-pong backup bound as a fragment-shader render target.
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_PREVIOUS_INTERPOLATION_SOURCE_BACKUP, "rw_previous_interpolation_source"}};

template <size_t Count>
constexpr bool HasResourceBinding(const ResourceBinding (&bindings)[Count], uint32_t resourceIdentifier)
{
    for (size_t index = 0; index < Count; ++index)
    {
        if (bindings[index].index == resourceIdentifier)
        {
            return true;
        }
    }
    return false;
}

static_assert(HasResourceBinding(srvResourceBindingTable, FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_OF_COLOUR_P1));
static_assert(HasResourceBinding(srvResourceBindingTable, FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_OF_COLOUR_M1));
static_assert(HasResourceBinding(srvResourceBindingTable, FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_PREPROCESS_COLOUR_P1));
static_assert(HasResourceBinding(srvResourceBindingTable, FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_PREPROCESS_COLOUR_M1));
static_assert(HasResourceBinding(srvResourceBindingTable, FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_REDUCED_MOTION_P1));
static_assert(HasResourceBinding(srvResourceBindingTable, FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_DEPTH_TM1_NEXT));
static_assert(HasResourceBinding(uavResourceBindingTable, FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_OF_COLOUR_P1));
static_assert(HasResourceBinding(uavResourceBindingTable, FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_PREPROCESS_COLOUR_P1));
static_assert(HasResourceBinding(uavResourceBindingTable, FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_REDUCED_MOTION_P1));
static_assert(HasResourceBinding(rtResourceBindingTable, FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_OF_COLOUR_P1));
static_assert(HasResourceBinding(rtResourceBindingTable, FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_PREPROCESS_COLOUR_P1));
static_assert(HasResourceBinding(rtResourceBindingTable, FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_REDUCED_MOTION_P1));

static const ResourceBinding cbResourceBindingTable[] = {
    {FFX_FRAMEINTERPOLATION_CONSTANTBUFFER_IDENTIFIER, "cbFI"}, {FFX_FRAMEINTERPOLATION_CONSTANTBUFFER_IDENTIFIER, "cbFI_t"},
    //{FFX_ARM_FRAMEINTERPOLATION_INPAINTING_PYRAMID_CONSTANTBUFFER_IDENTIFIER, "cbInpaintingPyramid"},
};

static const ResourceBinding srvTensorBindingTable[] = {
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_IN_TENSOR, "Resource_0_input"},
};

static const ResourceBinding uavTensorBindingTable[] = {
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_IN_TENSOR, "rw_in_tensor"},
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_OUT_PARAMS_TENSOR, "rw_out_params_tensor"},
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_OUT_PARAMS_GATE_TENSOR, "rw_out_params_gate_tensor"},
};

// Lanczos
static FfxFloat32 lanczos2(FfxFloat32 value)
{
    return abs(value) < FFX_EPSILON ? 1.f : (sinf(FFX_PI * value) / (FFX_PI * value)) * (sinf(0.5f * FFX_PI * value) / (0.5f * FFX_PI * value));
}

static void getIdentity(FfxFloat32x4x4& outMatrix)
{
    memset(&outMatrix, 0, sizeof(outMatrix));
    outMatrix[0] = outMatrix[5] = outMatrix[10] = outMatrix[15] = 1.0f;
}

// Code taken from MESA implementation of GLU
// Under terms of the SGI FREE SOFTWARE LICENSE B (Version 2.0, Sept. 18, 2008)
// https://cgit.freedesktop.org/mesa/glu/tree/src/libutil/project.c
static void matrixInvert(const FfxFloat32x4x4 m, FfxFloat32x4x4 out)
{
    FfxFloat32 inv[16], det;
    FfxInt32   i;

    inv[0]  = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] + m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];
    inv[4]  = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] - m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];
    inv[8]  = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] + m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];
    inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] - m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];
    inv[1]  = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] - m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];
    inv[5]  = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] + m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];
    inv[9]  = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] - m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];
    inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] + m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];
    inv[2]  = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] + m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];
    inv[6]  = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] - m[4] * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];
    inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] + m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];
    inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] - m[4] * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];
    inv[3]  = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] - m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];
    inv[7]  = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] + m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];
    inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] - m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];
    inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] + m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];
    det     = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];

    // FFX_ASSERT(det != 0);
    if (det == 0)
    {
        memset(out, 0, 16 * sizeof(float));
        return;
    }

    det = 1.0f / det;

    for (i = 0; i < 16; i++)
        out[i] = inv[i] * det;
}

static void matrixMul(const FfxFloat32x4x4 a, const FfxFloat32x4x4 b, FfxFloat32x4x4 out)
{
    // Column-major matrix multiplication: out = a * b
    // Each matrix is 4x4, indexed as m[col * 4 + row]
    for (uint32_t col = 0; col < 4; ++col)
    {
        for (uint32_t row = 0; row < 4; ++row)
        {
            float sum = 0.0f;
            for (uint32_t k = 0; k < 4; ++k)
            {
                // a(row,k) * b(k,col)
                sum += a[k * 4 + row] * b[col * 4 + k];
            }
            out[col * 4 + row] = sum;
        }
    }
}

static void printMessage(const FfxArmFrameInterpolationContext_Private* context, const FfxMsgType type, const char* message)
{
    ffxLogMessage(ffxLogComponentName(), type, message, context ? context->contextDescription.fpMessage : nullptr);
}

static void nfruDebugCheckCreate(FfxArmFrameInterpolationContext_Private* context)
{
    FFX_ASSERT(context);
    constexpr uint32_t MESSAGE_BUFFER_SIZE = 1024;
    char               message[MESSAGE_BUFFER_SIZE];

    if (context->resolutionPlan.preprocessLane.width % FFX_FRAMEINTERPOLATION_DATA_GRAPH_ALIGNMENT != 0 ||
        context->resolutionPlan.preprocessLane.height % FFX_FRAMEINTERPOLATION_DATA_GRAPH_ALIGNMENT != 0)
    {
        snprintf(message,
                 MESSAGE_BUFFER_SIZE,
                 "Preprocess size (%u x %u) is not aligned to required alignment (%u), SDK will align internally.\n",
                 context->resolutionPlan.preprocessLane.width,
                 context->resolutionPlan.preprocessLane.height,
                 FFX_FRAMEINTERPOLATION_DATA_GRAPH_ALIGNMENT);
        printMessage(context, FFX_MESSAGE_TYPE_WARNING, message);
    }
}

static FfxErrorCode createResourceFromDescription(FfxArmFrameInterpolationContext_Private* context, const FfxInternalResourceDescription* resDesc)
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

static FfxErrorCode patchResourceBindings(FfxPipelineState* inoutPipeline)
{
    for (uint32_t srvIndex = 0; srvIndex < inoutPipeline->srvTextureCount; ++srvIndex)
    {
        int32_t mapIndex = 0;
        for (mapIndex = 0; mapIndex < FFX_COUNTOF(srvResourceBindingTable); ++mapIndex)
        {
            if (0 == strcmp(srvResourceBindingTable[mapIndex].name, inoutPipeline->srvTextureBindings[srvIndex].name))
                break;
        }
        if (mapIndex == FFX_COUNTOF(srvResourceBindingTable))
            return ffxLogError(FFX_ERROR_INVALID_ARGUMENT, "the shader SRV binding name is not recognized");

        inoutPipeline->srvTextureBindings[srvIndex].resourceIdentifier = srvResourceBindingTable[mapIndex].index;
    }

    for (uint32_t tensorIndex = 0; tensorIndex < inoutPipeline->srvTensorCount; ++tensorIndex)
    {
        int32_t mapIndex = 0;
        for (mapIndex = 0; mapIndex < FFX_COUNTOF(srvResourceBindingTable); ++mapIndex)
        {
            if (0 == strcmp(srvResourceBindingTable[mapIndex].name, inoutPipeline->srvTensorBindings[tensorIndex].name))
                break;
        }
        if (mapIndex == FFX_COUNTOF(srvResourceBindingTable))
            return ffxLogError(FFX_ERROR_INVALID_ARGUMENT, "the shader tensor SRV binding name is not recognized");

        inoutPipeline->srvTensorBindings[tensorIndex].resourceIdentifier = srvResourceBindingTable[mapIndex].index;
    }

    for (uint32_t tensorIndex = 0; tensorIndex < inoutPipeline->uavTensorCount; ++tensorIndex)
    {
        int32_t mapIndex = 0;
        for (mapIndex = 0; mapIndex < FFX_COUNTOF(uavResourceBindingTable); ++mapIndex)
        {
            if (0 == strcmp(uavResourceBindingTable[mapIndex].name, inoutPipeline->uavTensorBindings[tensorIndex].name))
                break;
        }
        if (mapIndex == FFX_COUNTOF(uavResourceBindingTable))
            return ffxLogError(FFX_ERROR_INVALID_ARGUMENT, "the shader tensor UAV binding name is not recognized");

        inoutPipeline->uavTensorBindings[tensorIndex].resourceIdentifier = uavResourceBindingTable[mapIndex].index;
    }

    // check for UAVs where mip chains are to be bound
    for (uint32_t uavIndex = 0; uavIndex < inoutPipeline->uavTextureCount; ++uavIndex)
    {
        int32_t mapIndex = 0;
        for (mapIndex = 0; mapIndex < FFX_COUNTOF(uavResourceBindingTable); ++mapIndex)
        {
            if (0 == strcmp(uavResourceBindingTable[mapIndex].name, inoutPipeline->uavTextureBindings[uavIndex].name))
                break;
        }
        if (mapIndex == FFX_COUNTOF(uavResourceBindingTable))
            return ffxLogError(FFX_ERROR_INVALID_ARGUMENT, "the shader texture UAV binding name is not recognized");

        inoutPipeline->uavTextureBindings[uavIndex].resourceIdentifier = uavResourceBindingTable[mapIndex].index;
    }

    for (uint32_t uavIndex = 0; uavIndex < inoutPipeline->uavBufferCount; ++uavIndex)
    {
        int32_t mapIndex = 0;
        for (mapIndex = 0; mapIndex < FFX_COUNTOF(uavResourceBindingTable); ++mapIndex)
        {
            if (0 == strcmp(uavResourceBindingTable[mapIndex].name, inoutPipeline->uavBufferBindings[uavIndex].name))
                break;
        }
        if (mapIndex == FFX_COUNTOF(uavResourceBindingTable))
            return ffxLogError(FFX_ERROR_INVALID_ARGUMENT, "the shader buffer UAV binding name is not recognized");

        inoutPipeline->uavBufferBindings[uavIndex].resourceIdentifier = uavResourceBindingTable[mapIndex].index;
    }

    for (uint32_t cbIndex = 0; cbIndex < inoutPipeline->constCount; ++cbIndex)
    {
        int32_t mapIndex = 0;
        for (mapIndex = 0; mapIndex < FFX_COUNTOF(cbResourceBindingTable); ++mapIndex)
        {
            if (0 == strcmp(cbResourceBindingTable[mapIndex].name, inoutPipeline->constantBufferBindings[cbIndex].name))
                break;
        }
        if (mapIndex == FFX_COUNTOF(cbResourceBindingTable))
            return ffxLogError(FFX_ERROR_INVALID_ARGUMENT, "the shader constant buffer binding name is not recognized");

        inoutPipeline->constantBufferBindings[cbIndex].resourceIdentifier = cbResourceBindingTable[mapIndex].index;
    }

    for (uint32_t srvBufferIndex = 0; srvBufferIndex < inoutPipeline->srvBufferCount; ++srvBufferIndex)
    {
        int32_t mapIndex = 0;
        for (mapIndex = 0; mapIndex < FFX_COUNTOF(srvResourceBindingTable); ++mapIndex)
        {
            if (0 == strcmp(srvResourceBindingTable[mapIndex].name, inoutPipeline->srvBufferBindings[srvBufferIndex].name))
                break;
        }
        if (mapIndex == FFX_COUNTOF(srvResourceBindingTable))
            return ffxLogError(FFX_ERROR_INVALID_ARGUMENT, "the shader buffer SRV binding name is not recognized");

        inoutPipeline->srvBufferBindings[srvBufferIndex].resourceIdentifier = srvResourceBindingTable[mapIndex].index;
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

static uint32_t getPipelinePermutationFlags(FfxArmFrameInterpolationContext_Private* context, bool enableDataGraphFI)
{
    // work out what permutation to load.
    FFX_ASSERT(context);
    FFX_ASSERT(context->contextDescription.backendInterface.devCapInitialized);

    FfxDeviceCapabilities capabilities{};
    FFX_VALIDATE(context->contextDescription.backendInterface.fpGetDeviceCapabilities(&context->contextDescription.backendInterface, &capabilities));

    uint32_t   contextFlags          = context->contextDescription.flags;
    const bool fp16Supported         = capabilities.fp16Supported;
    const bool computeSupportTensor  = capabilities.computeSupportTensor;
    const bool fragmentSupportTensor = capabilities.fragmentSupportTensor;

    uint32_t flags = 0;
    flags |= (contextFlags & FFX_FG_CONTEXT_FLAG_ENABLE_DISPLAY_RESOLUTION_MOTION_VECTORS) ? 0 : FRAMEINTERPOLATION_SHADER_PERMUTATION_LOW_RES_MOTION_VECTORS;
    flags |= (contextFlags & FFX_FG_CONTEXT_FLAG_ENABLE_MOTION_VECTORS_JITTER_CANCELLATION) ? FRAMEINTERPOLATION_SHADER_PERMUTATION_JITTER_MOTION_VECTORS : 0;
    flags |= (contextFlags & FFX_FG_CONTEXT_FLAG_ENABLE_DEPTH_INVERTED) ? FRAMEINTERPOLATION_SHADER_PERMUTATION_DEPTH_INVERTED : 0;
    flags |= (fp16Supported) ? FRAMEINTERPOLATION_SHADER_PERMUTATION_ALLOW_FP16 : 0;
    flags |= (enableDataGraphFI) ? FRAMEINTERPOLATION_SHADER_PERMUTATION_ENABLE_DATA_GRAPH_FI : 0;

    flags |= (contextFlags & FFX_FG_CONTEXT_FLAG_WARP_COMPUTE) ? FRAMEINTERPOLATION_SHADER_PERMUTATION_WARP_COMPUTE : 0;
    flags |= (contextFlags & FFX_FG_CONTEXT_FLAG_FILL_HOLES_COMPUTE) ? FRAMEINTERPOLATION_SHADER_PERMUTATION_FILL_HOLES_COMPUTE : 0;
    flags |= (contextFlags & FFX_FG_CONTEXT_FLAG_PRE_PROCESS_COMPUTE) ? FRAMEINTERPOLATION_SHADER_PERMUTATION_PRE_PROCESS_COMPUTE : 0;
    flags |= (contextFlags & FFX_FG_CONTEXT_FLAG_POSTPROCESS_COMPUTE) ? FRAMEINTERPOLATION_SHADER_PERMUTATION_POSTPROCESS_COMPUTE : 0;
    flags |= (contextFlags & FFX_FG_CONTEXT_FLAG_DEBUG_VIEW_COMPUTE) ? FRAMEINTERPOLATION_SHADER_PERMUTATION_DEBUG_VIEW_COMPUTE : 0;
    flags |= (computeSupportTensor) ? FRAMEINTERPOLATION_SHADER_PERMUTATION_COMPUTE_SUPPORT_TENSOR : 0;
    flags |= (fragmentSupportTensor) ? FRAMEINTERPOLATION_SHADER_PERMUTATION_FRAGMENT_SUPPORT_TENSOR : 0;

    const bool usesInternalDepthHistory = (contextFlags & FFX_FG_CONTEXT_FLAG_MANAGE_PREVIOUS_DEPTH) != 0 || context->resolutionPlan.needsAllInputDownsample;
    const bool skipInternalDepthWrite   = !usesInternalDepthHistory || context->resolutionPlan.needsAllInputDownsample;
    const bool skipInternalColorWrite   = (contextFlags & FFX_FG_CONTEXT_FLAG_MANAGE_PREVIOUS_COLOR) == 0;
    flags |= skipInternalDepthWrite ? 0 : FRAMEINTERPOLATION_SHADER_PERMUTATION_MANAGE_PREVIOUS_DEPTH;
    flags |= skipInternalColorWrite ? 0 : FRAMEINTERPOLATION_SHADER_PERMUTATION_MANAGE_PREVIOUS_COLOR;

    return flags;
}

static FfxErrorCode buildDataGraphTensorInfo(const FfxInternalResourceDescription* internalSurfaceDesc,
                                             uint32_t                              internalSurfaceCount,
                                             FfxDataGraphTensorInfo*               outTensorInfo,
                                             uint32_t                              maxTensorInfoCount,
                                             uint32_t*                             outTensorInfoCount)
{
    FFX_ASSERT(internalSurfaceDesc != nullptr);
    FFX_ASSERT(outTensorInfo != nullptr);
    FFX_ASSERT(outTensorInfoCount != nullptr);

    uint32_t tensorInfoCount = 0;

    const auto appendTensorInfo = [&](const ResourceBinding& binding, bool bufferAliased) -> FfxErrorCode {
        for (uint32_t infoIndex = 0; infoIndex < tensorInfoCount; ++infoIndex)
        {
            if (0 == strcmp(outTensorInfo[infoIndex].resourceName, binding.name))
            {
                outTensorInfo[infoIndex].bufferAliased = bufferAliased;
                return FFX_OK;
            }
        }

        FFX_RETURN_ON_ERROR(tensorInfoCount < maxTensorInfoCount, FFX_ERROR_OUT_OF_RANGE);
        outTensorInfo[tensorInfoCount++] = {binding.name, bufferAliased};
        return FFX_OK;
    };

    for (uint32_t surfaceIndex = 0; surfaceIndex < internalSurfaceCount; ++surfaceIndex)
    {
        const FfxInternalResourceDescription& surfaceDesc = internalSurfaceDesc[surfaceIndex];
        if (surfaceDesc.type != FFX_RESOURCE_TYPE_TENSOR)
        {
            continue;
        }

        const bool bufferAliased = (surfaceDesc.flags & FFX_RESOURCE_FLAGS_BUFFER_ALIASED) == FFX_RESOURCE_FLAGS_BUFFER_ALIASED;
        for (uint32_t bindingIndex = 0; bindingIndex < FFX_COUNTOF(srvTensorBindingTable); ++bindingIndex)
        {
            if (srvTensorBindingTable[bindingIndex].index == surfaceDesc.id)
            {
                FFX_VALIDATE(appendTensorInfo(srvTensorBindingTable[bindingIndex], bufferAliased));
            }
        }

        for (uint32_t bindingIndex = 0; bindingIndex < FFX_COUNTOF(uavTensorBindingTable); ++bindingIndex)
        {
            if (uavTensorBindingTable[bindingIndex].index == surfaceDesc.id)
            {
                FFX_VALIDATE(appendTensorInfo(uavTensorBindingTable[bindingIndex], bufferAliased));
            }
        }
    }

    *outTensorInfoCount = tensorInfoCount;
    return FFX_OK;
}

static FfxErrorCode createPipelineStates(FfxArmFrameInterpolationContext_Private* context,
                                         const FfxDataGraphTensorInfo*            dataGraphTensorInfo,
                                         uint32_t                                 dataGraphTensorInfoCount)
{
    FFX_ASSERT(context);

    FfxPipelineDescription pipelineDescription = {};
    pipelineDescription.contextFlags           = context->contextDescription.flags;

    // Set up pipeline descriptor (basically RootSignature and binding)
    // Root constants
    pipelineDescription.rootConstantBufferCount = 1;
    auto CreatePipeline                         = [&](FfxPass pass, const char* name, FfxPipelineState* pipeline, bool useFragment) -> FfxErrorCode {
        ffxSafeReleasePipeline(&context->contextDescription.backendInterface, pipeline, context->effectContextId);
        strncpy(pipelineDescription.name, name, FFX_RESOURCE_NAME_SIZE - 1);
        pipelineDescription.name[FFX_RESOURCE_NAME_SIZE - 1] = '\0';

        FfxRootConstantDescription rootConstantDescs[2] = {
            {sizeof(FrameInterpolationConstants) / sizeof(uint32_t), useFragment ? FFX_BIND_PIXEL_SHADER_STAGE : FFX_BIND_COMPUTE_SHADER_STAGE}};
        pipelineDescription.rootConstants = rootConstantDescs;

        // Samplers
        constexpr size_t      samplerCount               = 4;
        FfxSamplerDescription samplerDescs[samplerCount] = {{FFX_FILTER_TYPE_MINMAGMIP_POINT,
                                                             FFX_ADDRESS_MODE_CLAMP,
                                                             FFX_ADDRESS_MODE_CLAMP,
                                                             FFX_ADDRESS_MODE_CLAMP,
                                                             useFragment ? FFX_BIND_PIXEL_SHADER_STAGE : FFX_BIND_COMPUTE_SHADER_STAGE,
                                                             FFX_FALSE},
                                                            {FFX_FILTER_TYPE_MINMAGMIP_LINEAR,
                                                             FFX_ADDRESS_MODE_CLAMP,
                                                             FFX_ADDRESS_MODE_CLAMP,
                                                             FFX_ADDRESS_MODE_CLAMP,
                                                             useFragment ? FFX_BIND_PIXEL_SHADER_STAGE : FFX_BIND_COMPUTE_SHADER_STAGE,
                                                             FFX_FALSE},
                                                            {FFX_FILTER_TYPE_MINMAGMIP_POINT,
                                                             FFX_ADDRESS_MODE_BORDER,
                                                             FFX_ADDRESS_MODE_BORDER,
                                                             FFX_ADDRESS_MODE_BORDER,
                                                             useFragment ? FFX_BIND_PIXEL_SHADER_STAGE : FFX_BIND_COMPUTE_SHADER_STAGE,
                                                             FFX_FALSE},
                                                            {FFX_FILTER_TYPE_MINMAGMIP_LINEAR,
                                                             FFX_ADDRESS_MODE_BORDER,
                                                             FFX_ADDRESS_MODE_BORDER,
                                                             FFX_ADDRESS_MODE_BORDER,
                                                             useFragment ? FFX_BIND_PIXEL_SHADER_STAGE : FFX_BIND_COMPUTE_SHADER_STAGE,
                                                             FFX_FALSE}};
        pipelineDescription.samplers                     = samplerDescs;
        pipelineDescription.samplerCount                 = samplerCount;
        pipelineDescription.stage                        = useFragment ? FFX_BIND_PIXEL_SHADER_STAGE : FFX_BIND_COMPUTE_SHADER_STAGE;

        if (useFragment)
        {
            FFX_VALIDATE(context->contextDescription.backendInterface.fpCreateGraphicsPipeline(&context->contextDescription.backendInterface,
                                                                                               ARM_EFFECT_FRAMEINTERPOLATION,
                                                                                               pass,
                                                                                               context->pipelineFlags,
                                                                                               &pipelineDescription,
                                                                                               context->effectContextId,
                                                                                               pipeline));
        }
        else
        {
            FFX_VALIDATE(context->contextDescription.backendInterface.fpCreateComputePipeline(&context->contextDescription.backendInterface,
                                                                                              ARM_EFFECT_FRAMEINTERPOLATION,
                                                                                              pass,
                                                                                              context->pipelineFlags,
                                                                                              &pipelineDescription,
                                                                                              context->effectContextId,
                                                                                              pipeline));
        }
        FFX_VALIDATE(patchResourceBindings(pipeline));

        return FFX_OK;
    };

    auto CreateDataGraphPipeline = [&](FfxPass pass, const char* name, FfxPipelineState* pipeline) -> FfxErrorCode {
        ffxSafeReleasePipeline(&context->contextDescription.backendInterface, pipeline, context->effectContextId);
        strncpy(pipelineDescription.name, name, FFX_RESOURCE_NAME_SIZE - 1);
        pipelineDescription.name[FFX_RESOURCE_NAME_SIZE - 1] = '\0';
        pipelineDescription.dataGraphTensorInfo              = dataGraphTensorInfo;
        pipelineDescription.dataGraphTensorInfoCount         = dataGraphTensorInfoCount;
        FFX_VALIDATE(context->contextDescription.backendInterface.fpCreateDataGraphPipeline(&context->contextDescription.backendInterface,
                                                                                            ARM_EFFECT_FRAMEINTERPOLATION,
                                                                                            pass,
                                                                                            context->pipelineFlags,
                                                                                            &pipelineDescription,
                                                                                            context->effectContextId,
                                                                                            context->dataGraphWidth,
                                                                                            context->dataGraphHeight,
                                                                                            pipeline));
        FFX_VALIDATE(patchResourceBindings(pipeline));

        return FFX_OK;
    };

    // Create Arm Frame Interpolation Pipelines
    const bool warpUseFragment        = (context->pipelineFlags & FRAMEINTERPOLATION_SHADER_PERMUTATION_WARP_COMPUTE) == 0;
    const bool fillHolesUseFragment   = (context->pipelineFlags & FRAMEINTERPOLATION_SHADER_PERMUTATION_FILL_HOLES_COMPUTE) == 0;
    const bool preProcessUseFragment  = (context->pipelineFlags & FRAMEINTERPOLATION_SHADER_PERMUTATION_PRE_PROCESS_COMPUTE) == 0;
    const bool postprocessUseFragment = (context->pipelineFlags & FRAMEINTERPOLATION_SHADER_PERMUTATION_POSTPROCESS_COMPUTE) == 0;
    const bool debugViewUseFragment   = (context->pipelineFlags & FRAMEINTERPOLATION_SHADER_PERMUTATION_DEBUG_VIEW_COMPUTE) == 0;
    // 0: Init warp (always compute)
    FFX_VALIDATE(CreatePipeline(FFX_FRAMEINTERPOLATION_PASS_INIT_WARP, "ARM_NFRU_INIT_WARP", &context->pipelineInitWarp, false));
    if (context->resolutionPlan.needsOfColourDownsample)
    {
        FFX_VALIDATE(
            CreatePipeline(FFX_FRAMEINTERPOLATION_PASS_DOWNSAMPLE_OF_COLOUR, "ARM_NFRU_DOWNSAMPLE_OF_COLOUR", &context->pipelineDownsampleOfColour, true));
    }
    if (context->resolutionPlan.needsAllInputDownsample)
    {
        FFX_VALIDATE(
            CreatePipeline(FFX_FRAMEINTERPOLATION_PASS_DOWNSAMPLE_ALL_INPUT, "ARM_NFRU_DOWNSAMPLE_ALL_INPUT", &context->pipelineDownsampleAllInput, true));
    }
    else if (context->resolutionPlan.needsColourDownsample)
    {
        FFX_VALIDATE(CreatePipeline(FFX_FRAMEINTERPOLATION_PASS_DOWNSAMPLE_COLOUR, "ARM_NFRU_DOWNSAMPLE_COLOUR", &context->pipelineDownsampleColour, true));
    }
    // 1: Motion vector warping
    FFX_VALIDATE(CreatePipeline(FFX_FRAMEINTERPOLATION_PASS_WARP_MV_TP1, "ARM_NFRU_WARP_MOTION_TP1", &context->pipelineWarpFlow, warpUseFragment));
    // 2: Optical flow warping
    FFX_VALIDATE(CreatePipeline(FFX_FRAMEINTERPOLATION_PASS_WARP_FLOW_TM1, "ARM_NFRU_WARP_FLOW_TM1", &context->pipelineWarpFlowTM1, warpUseFragment));
    // 3: Hole filling
    FFX_VALIDATE(CreatePipeline(FFX_FRAMEINTERPOLATION_PASS_FILL_HOLES_TM1, "ARM_NFRU_FILL_HOLES_TM1", &context->pipelineFillHolesTM1, fillHolesUseFragment));
    // 4: Neural network processing
    FFX_VALIDATE(CreatePipeline(FFX_FRAMEINTERPOLATION_PASS_PREPROCESS, "ARM_NFRU_PREPROCESS", &context->pipelinePreprocess, preProcessUseFragment));
    if (context->enableDataGraphFI)
    {
        // Neural Network Graph - Uses dispatch_graph instead of dispatch_compute
        FFX_VALIDATE(CreateDataGraphPipeline(FFX_FRAMEINTERPOLATION_PASS_NFRU_INTERPOLATION, "ARM_NFRU_INTERPOLATION", &context->pipelineNFRUInterpolation));
    }

    // Inpaint pyramid downsample (always compute; distinct resources preserve whole-image layouts)
    FFX_VALIDATE(CreatePipeline(
        FFX_FRAMEINTERPOLATION_PASS_DOWNSAMPLE_INPAINT_SEED, "ARM_NFRU_DOWNSAMPLE_INPAINT_SEED", &context->pipelineDownsampleInpaintSeed, false));
    FFX_VALIDATE(
        CreatePipeline(FFX_FRAMEINTERPOLATION_PASS_DOWNSAMPLE_INPAINT_MIP, "ARM_NFRU_DOWNSAMPLE_INPAINT_MIP", &context->pipelineDownsampleInpaintMip, false));
    FFX_VALIDATE(CreatePipeline(FFX_FRAMEINTERPOLATION_PASS_INPAINT, "ARM_NFRU_INPAINT", &context->pipelineInpaint, false));
    FFX_VALIDATE(CreatePipeline(FFX_FRAMEINTERPOLATION_PASS_POSTPROCESS, "ARM_NFRU_POSTPROCESS", &context->pipelinePostprocess, postprocessUseFragment));

    // Create utility pipelines
    FFX_VALIDATE(CreatePipeline(FFX_FRAMEINTERPOLATION_PASS_DEBUG_VIEW, "ARM_NFRU_DEBUG_VIEW", &context->pipelineDebugView, debugViewUseFragment));

    return FFX_OK;
}

static FfxErrorCode logErrorWithCallback(FfxErrorCode errorCode, const char* errorMessage, FfxLogMessageCallback callback, const char* funcName);

static FfxErrorCode frameinterpolationCreate(FfxArmFrameInterpolationContext_Private*       context,
                                             const FfxFrameInterpolationContextDescription* contextDescription)
{
    FFX_ASSERT(context);
    FFX_ASSERT(contextDescription);

    // Setup the data for implementation.
    memset(context, 0, sizeof(FfxArmFrameInterpolationContext_Private));
    context->device = contextDescription->backendInterface.device;

    memcpy(&context->contextDescription, contextDescription, sizeof(FfxFrameInterpolationContextDescription));

    // Check version info - make sure we are linked with the right backend version
    FfxVersionNumber version = context->contextDescription.backendInterface.fpGetSDKVersion(&context->contextDescription.backendInterface);
    FFX_RETURN_ON_ERROR(version == FFX_SDK_MAKE_VERSION(FFX_SDK_VERSION_MAJOR, FFX_SDK_VERSION_MINOR, FFX_SDK_VERSION_PATCH), FFX_ERROR_INVALID_VERSION);

    // Create the context.
    FfxErrorCode errorCode = context->contextDescription.backendInterface.fpCreateBackendContext(
        &context->contextDescription.backendInterface, ARM_EFFECT_FRAMEINTERPOLATION, nullptr, &context->effectContextId);
    FFX_RETURN_ON_ERROR(errorCode == FFX_OK, errorCode);
    context->backendContextCreated = true;

    // Register the message callback for backend
    errorCode = context->contextDescription.backendInterface.fpSetMessageCallback(&context->contextDescription.backendInterface, contextDescription->fpMessage);
    FFX_RETURN_ON_ERROR(errorCode == FFX_OK, errorCode);

    // Query device capabilities
    FfxDeviceCapabilities capabilities{};
    FFX_VALIDATE(context->contextDescription.backendInterface.fpGetDeviceCapabilities(&context->contextDescription.backendInterface, &capabilities));
    constexpr uint32_t fragmentAtomicComputeFlags = FFX_FG_CONTEXT_FLAG_WARP_COMPUTE | FFX_FG_CONTEXT_FLAG_PRE_PROCESS_COMPUTE;
    if ((contextDescription->flags & fragmentAtomicComputeFlags) != fragmentAtomicComputeFlags && !capabilities.fragmentStoresAndAtomicsSupported)
    {
        return logErrorWithCallback(
            FFX_ERROR_INVALID_ARGUMENT,
            "fragment warp and preprocess require fragmentStoresAndAtomics; enable the Vulkan feature or select compute for both stages",
            contextDescription->fpMessage,
            FFX_BUILTIN_FUNCTION);
    }

    // set initial view projection value.
    memcpy(context->lastFrameViewProjection, contextDescription->initialViewProjection, sizeof(context->lastFrameViewProjection));
    context->firstExecution = true;

    context->enableDataGraphFI = true;

    // Get dimensions.
    const FfxDimensions2D renderSize  = contextDescription->maxRenderSize;
    const FfxDimensions2D displaySize = contextDescription->displaySize;

    context->resolutionPlan = ffxNfruDeriveResolutionPlan(displaySize, contextDescription->mvDepthSize, FFX_FRAMEINTERPOLATION_DATA_GRAPH_ALIGNMENT);
    context->ofInputColour  = context->resolutionPlan.ofInputColour;
    context->mvDepthLane    = context->resolutionPlan.mvDepthLane;
    context->flowLane       = context->resolutionPlan.flowLane;
    context->tensorLane     = context->resolutionPlan.preprocessLane;
    context->paramsLane     = context->resolutionPlan.paramsLane;

    // Set constant buffer.
    context->constants._RenderSize[0]  = renderSize.width;
    context->constants._RenderSize[1]  = renderSize.height;
    context->constants._DisplaySize[0] = displaySize.width;
    context->constants._DisplaySize[1] = displaySize.height;

    context->constants._DisplaySizeRcp[0] = 1.0f / displaySize.width;
    context->constants._DisplaySizeRcp[1] = 1.0f / displaySize.height;

    context->constants._MvDepthLaneSize[0]    = context->mvDepthLane.width;
    context->constants._MvDepthLaneSize[1]    = context->mvDepthLane.height;
    context->constants._MvDepthLaneSizeRcp[0] = 1.0f / context->mvDepthLane.width;
    context->constants._MvDepthLaneSizeRcp[1] = 1.0f / context->mvDepthLane.height;
    context->constants._FlowLaneSize[0]       = context->flowLane.width;
    context->constants._FlowLaneSize[1]       = context->flowLane.height;
    context->constants._FlowLaneSizeRcp[0]    = 1.0f / context->flowLane.width;
    context->constants._FlowLaneSizeRcp[1]    = 1.0f / context->flowLane.height;
    context->constants._OfInputSizeRcp[0]     = 1.0f / context->ofInputColour.width;
    context->constants._OfInputSizeRcp[1]     = 1.0f / context->ofInputColour.height;

    // model specific
    // .xy = input quant (scale, zero); .zw = output dequant (scale, zero) = output tensor metadata.
    context->constants._QuantParamsSINT[0]  = 1 / 0.003921568859368563;  // input quant scale (x)
    context->constants._QuantParamsSINT[1]  = -128;                      // input quant zero
    context->constants._QuantParamsSINT[2]  = 0.4059530198574066;        // params dequant scale (getitem SINT)
    context->constants._QuantParamsSINT[3]  = 80;                        // params dequant zero
    context->constants._QuantParamsSNORM[0] = 1 / 0.49803924513980746;   // input quant scale
    context->constants._QuantParamsSNORM[1] = -1.0078740157480315;       // input quant zero
    context->constants._QuantParamsSNORM[2] = 51.55603352189064;         // params dequant scale (getitem SNORM)
    context->constants._QuantParamsSNORM[3] = 0.6299212598425197;        // params dequant zero
    // gate output (getitem_1) dequant params. .zw used by the postprocess gate.
    context->constants._QuantParamsGateSNORM[0] = 1 / 0.49803924513980746;  // gate quant scale
    context->constants._QuantParamsGateSNORM[1] = -1.0078740157480315;      // gate quant zero
    context->constants._QuantParamsGateSNORM[2] = 0.49803924513980746;      // gate dequant scale (getitem_1 SNORM)
    context->constants._QuantParamsGateSNORM[3] = -1.0078740157480315;      // gate dequant zero

    context->dataGraphWidth                = context->resolutionPlan.paddedTensorLane.width;
    context->dataGraphHeight               = context->resolutionPlan.paddedTensorLane.height;
    context->constants._InputTensorSize[0] = context->dataGraphWidth;
    context->constants._InputTensorSize[1] = context->dataGraphHeight;

    // graph parameter lane is the padded tensor lane / 2 (network outputs params + gate at half res).
    context->constants._ParamsLaneSizeRcp[0] = 1.0f / static_cast<float>(context->paramsLane.width);
    context->constants._ParamsLaneSizeRcp[1] = 1.0f / static_cast<float>(context->paramsLane.height);

    context->constants._Timestep              = 0.5;
    context->constants._RandomSeed            = 2357136044;
    context->constants._UseReducedMotionInput = context->resolutionPlan.needsAllInputDownsample;

    // Need to initialize the pipelineFlags early so we can check if fragment job will be enabled
    // for each stage and then setup the correct resource usage flag.
    context->pipelineFlags             = getPipelinePermutationFlags(context, context->enableDataGraphFI);
    const bool warpUseFragment         = (context->pipelineFlags & FRAMEINTERPOLATION_SHADER_PERMUTATION_WARP_COMPUTE) == 0;
    const bool fillHolesUseFragment    = (context->pipelineFlags & FRAMEINTERPOLATION_SHADER_PERMUTATION_FILL_HOLES_COMPUTE) == 0;
    const bool preprocessUseFragment   = (context->pipelineFlags & FRAMEINTERPOLATION_SHADER_PERMUTATION_PRE_PROCESS_COMPUTE) == 0;
    const bool preprocessSupportTensor = preprocessUseFragment ? capabilities.fragmentSupportTensor : capabilities.computeSupportTensor;
    const bool debugViewUseFragment    = (context->pipelineFlags & FRAMEINTERPOLATION_SHADER_PERMUTATION_DEBUG_VIEW_COMPUTE) == 0;
    const bool debugViewSupportTensor  = debugViewUseFragment ? capabilities.fragmentSupportTensor : capabilities.computeSupportTensor;
    const bool inTensorBufferAliased   = !preprocessSupportTensor || !debugViewSupportTensor;

    const FfxResourceFlags inputTensorFlags = inTensorBufferAliased ? FFX_RESOURCE_FLAGS_BUFFER_ALIASED : FFX_RESOURCE_FLAGS_NONE;
    // Each resource is written by exactly one stage, so use the corresponding stage's usage flag.
    const FfxResourceUsage warpOutputUsage      = warpUseFragment ? FFX_RESOURCE_USAGE_RENDERTARGET : FFX_RESOURCE_USAGE_UAV;
    const FfxResourceUsage fillHolesOutputUsage = fillHolesUseFragment ? FFX_RESOURCE_USAGE_RENDERTARGET : FFX_RESOURCE_USAGE_UAV;
    // inpaint usages: seed written by preprocess, five single-mip pyramid images built by pass 31,
    // and the resolved result written by pass 50.
    const FfxResourceUsage     inpaintSeedUsage   = FFX_RESOURCE_USAGE_UAV;
    const FfxResourceUsage     inpaintMipUsage    = FFX_RESOURCE_USAGE_UAV;
    const FfxResourceUsage     inpaintResultUsage = FFX_RESOURCE_USAGE_UAV;
    constexpr FfxResourceUsage tensorUsage        = FFX_RESOURCE_USAGE_UAV;

    // Debug checks
    nfruDebugCheckCreate(context);

    constexpr uint32_t inputTensorChannels  = 16;  // innermost dimension
    constexpr uint32_t outputTensorChannels = 4;   // innermost dimension
    constexpr uint32_t gateTensorChannels   = 1;   // second output (inpaint gate)
    constexpr uint32_t tensorShapeSize      = 4;

    // Declare internal resources needed
    const FfxDimensions2D                       mvDepthLane         = context->mvDepthLane;
    const FfxDimensions2D                       flowLane            = context->flowLane;
    std::vector<FfxInternalResourceDescription> internalSurfaceDesc = {
        {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_DYNAMIC_MASK_TM1,
         "ArmFI_DynamicMask_TM1",
         FFX_RESOURCE_TYPE_TEXTURE2D,
         warpOutputUsage,
         FFX_SURFACE_FORMAT_R8_UNORM,
         mvDepthLane.width,
         mvDepthLane.height,
         1,
         FFX_RESOURCE_FLAGS_NONE,
         {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED}},
        {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_DYNAMIC_MASK_TP1,
         "ArmFI_DynamicMask_TP1",
         FFX_RESOURCE_TYPE_TEXTURE2D,
         warpOutputUsage,
         FFX_SURFACE_FORMAT_R8_UNORM,
         mvDepthLane.width,
         mvDepthLane.height,
         1,
         FFX_RESOURCE_FLAGS_NONE,
         {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED}},
        {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_WARP_MOTION_QDATA_TP1,
         "ArmFI_WarpMotionQData_TP1",
         FFX_RESOURCE_TYPE_TEXTURE2D,
         FFX_RESOURCE_USAGE_UAV,
         FFX_SURFACE_FORMAT_R32_UINT,
         mvDepthLane.width,
         mvDepthLane.height,
         1,
         FFX_RESOURCE_FLAGS_NONE,
         {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED}},
        {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_WARP_FLOW_QDATA_TM1,
         "ArmFI_WarpFlowQData_TM1",
         FFX_RESOURCE_TYPE_TEXTURE2D,
         FFX_RESOURCE_USAGE_UAV,
         FFX_SURFACE_FORMAT_R32_UINT,
         flowLane.width,
         flowLane.height,
         1,
         FFX_RESOURCE_FLAGS_NONE,
         {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED}},
        {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_WARP_FILLED_FLOW_TM1,
         "ArmFI_FilledWarpFlowQData_TM1",
         FFX_RESOURCE_TYPE_TEXTURE2D,
         fillHolesOutputUsage,
         FFX_SURFACE_FORMAT_R16G16_FLOAT,
         flowLane.width,
         flowLane.height,
         1,
         FFX_RESOURCE_FLAGS_NONE,
         {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED}},
    };

    // inpaint resources: seed, five distinct logical levels, and the resolved result.
    internalSurfaceDesc.push_back({FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_INPAINT_SEED,
                                   "ArmFI_Inpaint_Seed",
                                   FFX_RESOURCE_TYPE_TEXTURE2D,
                                   inpaintSeedUsage,
                                   FFX_SURFACE_FORMAT_R8G8B8A8_UNORM,
                                   mvDepthLane.width,
                                   mvDepthLane.height,
                                   1,
                                   FFX_RESOURCE_FLAGS_NONE,
                                   {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED}});
    for (uint32_t level = 0; level < FFX_FRAMEINTERPOLATION_INPAINT_LEVEL_COUNT; ++level)
    {
        internalSurfaceDesc.push_back({INPAINT_LEVEL_RESOURCE_IDS[level],
                                       INPAINT_LEVEL_RESOURCE_NAMES[level],
                                       FFX_RESOURCE_TYPE_TEXTURE2D,
                                       inpaintMipUsage,
                                       FFX_SURFACE_FORMAT_R8G8B8A8_UNORM,
                                       std::max(1u, flowLane.width >> level),
                                       std::max(1u, flowLane.height >> level),
                                       1,
                                       FFX_RESOURCE_FLAGS_NONE,
                                       {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED}});
    }
    internalSurfaceDesc.push_back({FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_INPAINT_RESULT,
                                   "ArmFI_Inpaint_Result",
                                   FFX_RESOURCE_TYPE_TEXTURE2D,
                                   inpaintResultUsage,
                                   FFX_SURFACE_FORMAT_R8G8B8A8_UNORM,
                                   flowLane.width,
                                   flowLane.height,
                                   1,
                                   FFX_RESOURCE_FLAGS_NONE,
                                   {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED}});

    if (context->enableDataGraphFI)
    {
        internalSurfaceDesc.push_back({FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_IN_TENSOR,
                                       "ArmFI_IN_Tensor",
                                       FFX_RESOURCE_TYPE_TENSOR,
                                       tensorUsage,
                                       FFX_SURFACE_FORMAT_R8_SINT,
                                       context->dataGraphWidth,
                                       context->dataGraphHeight,
                                       1,
                                       inputTensorFlags,
                                       {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED},
                                       1,
                                       inputTensorChannels,
                                       tensorShapeSize});
        internalSurfaceDesc.push_back({FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_OUT_PARAMS_TENSOR,
                                       "ArmFI_Out_Params_Tensor",
                                       FFX_RESOURCE_TYPE_TENSOR,
                                       tensorUsage,
                                       FFX_SURFACE_FORMAT_R8_SINT,
                                       context->paramsLane.width,
                                       context->paramsLane.height,
                                       1,
                                       FFX_RESOURCE_FLAGS_IMAGE_ALIASED,
                                       {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED},
                                       1,
                                       outputTensorChannels,
                                       tensorShapeSize});
        // second graph output — per-pixel inpaint gate (1 channel) at params_lane.
        internalSurfaceDesc.push_back({FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_OUT_PARAMS_GATE_TENSOR,
                                       "ArmFI_Out_Params_Gate_Tensor",
                                       FFX_RESOURCE_TYPE_TENSOR,
                                       tensorUsage,
                                       FFX_SURFACE_FORMAT_R8_SINT,
                                       context->paramsLane.width,
                                       context->paramsLane.height,
                                       1,
                                       FFX_RESOURCE_FLAGS_IMAGE_ALIASED,
                                       {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED},
                                       1,
                                       gateTensorChannels,
                                       tensorShapeSize});
    }
    else
    {
        const uint32_t perTensorElementSize = inputTensorChannels * sizeof(int8_t);
        const uint32_t tensorDataSize       = context->dataGraphWidth * context->dataGraphHeight * perTensorElementSize;

        internalSurfaceDesc.push_back({FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_IN_TENSOR,
                                       "ArmFI_IN_Tensor",
                                       FFX_RESOURCE_TYPE_BUFFER,
                                       FFX_RESOURCE_USAGE_UAV,
                                       FFX_SURFACE_FORMAT_R8_SINT,
                                       tensorDataSize,        // buffersize
                                       perTensorElementSize,  // stride
                                       1,
                                       FFX_RESOURCE_FLAGS_NONE,
                                       {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED}});
        internalSurfaceDesc.push_back({FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_OUT_PARAMS_TENSOR,
                                       "ArmFI_Out_Params_Tensor",
                                       FFX_RESOURCE_TYPE_TEXTURE2D,
                                       FFX_RESOURCE_USAGE_UAV,
                                       FFX_SURFACE_FORMAT_R16G16B16A16_FLOAT,
                                       context->dataGraphWidth,
                                       context->dataGraphHeight,
                                       1,
                                       FFX_RESOURCE_FLAGS_NONE,
                                       {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED}});
    }

    if (context->resolutionPlan.needsOfColourDownsample)
    {
        internalSurfaceDesc.push_back({FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_OF_COLOUR_P1,
                                       "ArmFI_OfColourP1",
                                       FFX_RESOURCE_TYPE_TEXTURE2D,
                                       FFX_RESOURCE_USAGE_RENDERTARGET,
                                       contextDescription->backBufferFormat,
                                       context->resolutionPlan.ofInputColour.width,
                                       context->resolutionPlan.ofInputColour.height,
                                       1,
                                       FFX_RESOURCE_FLAGS_NONE,
                                       {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED}});
        internalSurfaceDesc.push_back({FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_OF_COLOUR_M1,
                                       "ArmFI_OfColourM1",
                                       FFX_RESOURCE_TYPE_TEXTURE2D,
                                       FFX_RESOURCE_USAGE_RENDERTARGET,
                                       contextDescription->backBufferFormat,
                                       context->resolutionPlan.ofInputColour.width,
                                       context->resolutionPlan.ofInputColour.height,
                                       1,
                                       FFX_RESOURCE_FLAGS_NONE,
                                       {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED}});
    }
    if (context->resolutionPlan.needsAllInputDownsample || context->resolutionPlan.needsColourDownsample)
    {
        internalSurfaceDesc.push_back({FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_PREPROCESS_COLOUR_P1,
                                       "ArmFI_PreprocessColourP1",
                                       FFX_RESOURCE_TYPE_TEXTURE2D,
                                       FFX_RESOURCE_USAGE_RENDERTARGET,
                                       contextDescription->backBufferFormat,
                                       context->resolutionPlan.preprocessLane.width,
                                       context->resolutionPlan.preprocessLane.height,
                                       1,
                                       FFX_RESOURCE_FLAGS_NONE,
                                       {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED}});
        internalSurfaceDesc.push_back({FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_PREPROCESS_COLOUR_M1,
                                       "ArmFI_PreprocessColourM1",
                                       FFX_RESOURCE_TYPE_TEXTURE2D,
                                       FFX_RESOURCE_USAGE_RENDERTARGET,
                                       contextDescription->backBufferFormat,
                                       context->resolutionPlan.preprocessLane.width,
                                       context->resolutionPlan.preprocessLane.height,
                                       1,
                                       FFX_RESOURCE_FLAGS_NONE,
                                       {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED}});
    }
    if (context->resolutionPlan.needsAllInputDownsample)
    {
        internalSurfaceDesc.push_back({FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_REDUCED_MOTION_P1,
                                       "ArmFI_ReducedMotionP1",
                                       FFX_RESOURCE_TYPE_TEXTURE2D,
                                       FFX_RESOURCE_USAGE_RENDERTARGET,
                                       FFX_SURFACE_FORMAT_R16G16_FLOAT,
                                       context->resolutionPlan.preprocessLane.width,
                                       context->resolutionPlan.preprocessLane.height,
                                       1,
                                       FFX_RESOURCE_FLAGS_NONE,
                                       {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED}});
    }

    // clear the SRV resources to NULL.
    memset(context->srvResources, 0, sizeof(context->srvResources));

    for (int32_t currentSurfaceIndex = 0; currentSurfaceIndex < internalSurfaceDesc.size(); ++currentSurfaceIndex)
    {
        FFX_VALIDATE(createResourceFromDescription(context, &internalSurfaceDesc[currentSurfaceIndex]));
    }

    // copy resources to uavResrouces list
    memcpy(context->uavResources, context->srvResources, sizeof(context->srvResources));

    context->dataGraphTensorInfoCount = 0;
    FFX_VALIDATE(buildDataGraphTensorInfo(internalSurfaceDesc.data(),
                                          internalSurfaceDesc.size(),
                                          context->dataGraphTensorInfo,
                                          FFX_COUNTOF(context->dataGraphTensorInfo),
                                          &context->dataGraphTensorInfoCount));

    // avoid compiling pipelines on first render
    {
        context->refreshPipelineStates = false;
        errorCode                      = createPipelineStates(context, context->dataGraphTensorInfo, context->dataGraphTensorInfoCount);
        FFX_RETURN_ON_ERROR(errorCode == FFX_OK, errorCode);
    }

    if (!capabilities.fp16Supported)
    {
        printMessage(context, FFX_MESSAGE_TYPE_WARNING, "NFRU precision=fp32 fp16_fallback=required_features_unsupported");
    }

    return FFX_OK;
}

static FfxErrorCode frameinterpolationRelease(FfxArmFrameInterpolationContext_Private* context)
{
    FFX_ASSERT(context);
    if (!context->backendContextCreated)
    {
        return FFX_OK;
    }

    // Release all of the algorithm's pipelines
    ffxSafeReleasePipeline(&context->contextDescription.backendInterface, &context->pipelineInitWarp, context->effectContextId);
    if (context->resolutionPlan.needsOfColourDownsample)
    {
        ffxSafeReleasePipeline(&context->contextDescription.backendInterface, &context->pipelineDownsampleOfColour, context->effectContextId);
    }
    if (context->resolutionPlan.needsAllInputDownsample)
    {
        ffxSafeReleasePipeline(&context->contextDescription.backendInterface, &context->pipelineDownsampleAllInput, context->effectContextId);
    }
    if (context->resolutionPlan.needsColourDownsample)
    {
        ffxSafeReleasePipeline(&context->contextDescription.backendInterface, &context->pipelineDownsampleColour, context->effectContextId);
    }
    ffxSafeReleasePipeline(&context->contextDescription.backendInterface, &context->pipelineWarpFlow, context->effectContextId);
    ffxSafeReleasePipeline(&context->contextDescription.backendInterface, &context->pipelineWarpFlowTM1, context->effectContextId);
    ffxSafeReleasePipeline(&context->contextDescription.backendInterface, &context->pipelineFillHolesTM1, context->effectContextId);

    ffxSafeReleasePipeline(&context->contextDescription.backendInterface, &context->pipelinePreprocess, context->effectContextId);
    if (context->enableDataGraphFI)
    {
        ffxSafeReleasePipeline(&context->contextDescription.backendInterface, &context->pipelineNFRUInterpolation, context->effectContextId);
    }
    ffxSafeReleasePipeline(&context->contextDescription.backendInterface, &context->pipelineInpaint, context->effectContextId);
    ffxSafeReleasePipeline(&context->contextDescription.backendInterface, &context->pipelineDownsampleInpaintSeed, context->effectContextId);
    ffxSafeReleasePipeline(&context->contextDescription.backendInterface, &context->pipelineDownsampleInpaintMip, context->effectContextId);
    ffxSafeReleasePipeline(&context->contextDescription.backendInterface, &context->pipelinePostprocess, context->effectContextId);
    // Release utility pipelines
    ffxSafeReleasePipeline(&context->contextDescription.backendInterface, &context->pipelineDebugView, context->effectContextId);

    // unregister resources not created internally
    // ARM_TODO: Release all resources
    context->srvResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_CURRENT_INTERPOLATION_SOURCE]         = {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_NULL};
    context->srvResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_OPTICAL_FLOW_VECTOR]                  = {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_NULL};
    context->srvResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_DEPTH]                                = {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_NULL};
    context->srvResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_DEPTH_TM1]                            = {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_NULL};
    context->srvResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_DEPTH_TM1_NEXT]                       = {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_NULL};
    context->srvResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_MOTION_VECTORS]                       = {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_NULL};
    context->srvResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_PREVIOUS_INTERPOLATION_SOURCE]        = {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_NULL};
    context->srvResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_OUTPUT]                               = {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_NULL};
    context->uavResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_OUTPUT]                               = {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_NULL};
    context->uavResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_PREVIOUS_INTERPOLATION_SOURCE_BACKUP] = {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_NULL};
    if (!context->resolutionPlan.needsOfColourDownsample)
    {
        context->srvResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_OF_COLOUR_P1] = {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_NULL};
    }
    if (!context->resolutionPlan.needsAllInputDownsample && !context->resolutionPlan.needsColourDownsample)
    {
        context->srvResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_PREPROCESS_COLOUR_P1] = {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_NULL};
        context->srvResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_PREPROCESS_COLOUR_M1] = {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_NULL};
    }

    // Release the copy resources for those that had init data
    // ARM_TODO: Needed?
    //ffxSafeReleaseCopyResource(&context->contextDescription.backendInterface, context->srvResources[FFX_ARM_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_COUNTERS], context->effectContextId);

    // release internal resources
    for (int32_t currentResourceIndex = 0; currentResourceIndex < FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_COUNT; ++currentResourceIndex)
    {
        ffxSafeReleaseResource(&context->contextDescription.backendInterface, context->srvResources[currentResourceIndex], context->effectContextId);
    }

    context->backendContextCreated = false;
    FFX_VALIDATE(context->contextDescription.backendInterface.fpDestroyBackendContext(&context->contextDescription.backendInterface, context->effectContextId));

    return FFX_OK;
}

static FfxErrorCode scheduleDispatch(FfxArmFrameInterpolationContext_Private* context,
                                     const FfxPipelineState*                  pipeline,
                                     uint32_t                                 dispatchX,
                                     uint32_t                                 dispatchY,
                                     const FfxResourceInternal*               srvTextureOverride = nullptr,
                                     const FfxResourceInternal*               uavTextureOverride = nullptr)
{
    FfxComputeJobDescription jobDescriptor = {};
    FFX_ASSERT(!srvTextureOverride || pipeline->srvTextureCount == 1);
    FFX_ASSERT(!uavTextureOverride || pipeline->uavTextureCount == 1);

    for (uint32_t currentShaderResourceViewIndex = 0; currentShaderResourceViewIndex < pipeline->srvTextureCount; ++currentShaderResourceViewIndex)
    {
        const uint32_t            currentResourceId = pipeline->srvTextureBindings[currentShaderResourceViewIndex].resourceIdentifier;
        const FfxResourceInternal currentResource   = srvTextureOverride ? *srvTextureOverride : context->srvResources[currentResourceId];
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

        const FfxResourceInternal currentResource = uavTextureOverride ? *uavTextureOverride : context->uavResources[currentResourceId];
        jobDescriptor.uavTextures[currentUnorderedAccessViewIndex].resource = currentResource;
        jobDescriptor.uavTextures[currentUnorderedAccessViewIndex].mip      = 0;
    }

    jobDescriptor.dimensions[0] = dispatchX;
    jobDescriptor.dimensions[1] = dispatchY;
    jobDescriptor.dimensions[2] = 1;
    jobDescriptor.pipeline      = *pipeline;

    for (uint32_t currentRootConstantIndex = 0; currentRootConstantIndex < pipeline->constCount; ++currentRootConstantIndex)
    {
#ifdef FFX_DEBUG
#ifdef FFX_DEBUG
        strncpy(jobDescriptor.cbNames[currentRootConstantIndex], pipeline->constantBufferBindings[currentRootConstantIndex].name, FFX_RESOURCE_NAME_SIZE - 1);
        jobDescriptor.cbNames[currentRootConstantIndex][FFX_RESOURCE_NAME_SIZE - 1] = '\0';
#endif
#endif
        jobDescriptor.cbs[currentRootConstantIndex] = context->constantBuffers[pipeline->constantBufferBindings[currentRootConstantIndex].resourceIdentifier];
    }

    for (uint32_t currentUnorderedAccessViewIndex = 0; currentUnorderedAccessViewIndex < pipeline->uavBufferCount; ++currentUnorderedAccessViewIndex)
    {
        const uint32_t currentResourceId                                   = pipeline->uavBufferBindings[currentUnorderedAccessViewIndex].resourceIdentifier;
        jobDescriptor.uavBuffers[currentUnorderedAccessViewIndex].resource = context->uavResources[currentResourceId];
#ifdef FFX_DEBUG
        strncpy(jobDescriptor.uavBuffers[currentUnorderedAccessViewIndex].name,
                pipeline->uavBufferBindings[currentUnorderedAccessViewIndex].name,
                FFX_RESOURCE_NAME_SIZE - 1);
        jobDescriptor.uavBuffers[currentUnorderedAccessViewIndex].name[FFX_RESOURCE_NAME_SIZE - 1] = '\0';
#endif
    }

    for (uint32_t currentShaderResourceViewIndex = 0; currentShaderResourceViewIndex < pipeline->srvBufferCount; ++currentShaderResourceViewIndex)
    {
        const uint32_t currentResourceId                                  = pipeline->srvBufferBindings[currentShaderResourceViewIndex].resourceIdentifier;
        jobDescriptor.srvBuffers[currentShaderResourceViewIndex].resource = context->srvResources[currentResourceId];
#ifdef FFX_DEBUG
        strncpy(jobDescriptor.srvBuffers[currentShaderResourceViewIndex].name,
                pipeline->srvBufferBindings[currentShaderResourceViewIndex].name,
                FFX_RESOURCE_NAME_SIZE - 1);
        jobDescriptor.srvBuffers[currentShaderResourceViewIndex].name[FFX_RESOURCE_NAME_SIZE - 1] = '\0';
#endif
    }

    for (uint32_t currentTensorIndex = 0; currentTensorIndex < pipeline->srvTensorCount; ++currentTensorIndex)
    {
        const uint32_t currentResourceId                      = pipeline->srvTensorBindings[currentTensorIndex].resourceIdentifier;
        jobDescriptor.srvTensors[currentTensorIndex].resource = context->srvResources[currentResourceId];
#ifdef FFX_DEBUG
        strncpy(jobDescriptor.srvTensors[currentTensorIndex].name, pipeline->srvTensorBindings[currentTensorIndex].name, FFX_RESOURCE_NAME_SIZE - 1);
        jobDescriptor.srvTensors[currentTensorIndex].name[FFX_RESOURCE_NAME_SIZE - 1] = '\0';
#endif
    }

    for (uint32_t currentTensorIndex = 0; currentTensorIndex < pipeline->uavTensorCount; ++currentTensorIndex)
    {
        const uint32_t currentResourceId                      = pipeline->uavTensorBindings[currentTensorIndex].resourceIdentifier;
        jobDescriptor.uavTensors[currentTensorIndex].resource = context->uavResources[currentResourceId];
#ifdef FFX_DEBUG
        strncpy(jobDescriptor.uavTensors[currentTensorIndex].name, pipeline->uavTensorBindings[currentTensorIndex].name, FFX_RESOURCE_NAME_SIZE - 1);
        jobDescriptor.uavTensors[currentTensorIndex].name[FFX_RESOURCE_NAME_SIZE - 1] = '\0';
#endif
    }

    FfxGpuJobDescription dispatchJob = {FFX_GPU_JOB_COMPUTE};
#ifdef FFX_DEBUG
    strncpy(dispatchJob.jobLabel, pipeline->name, FFX_RESOURCE_NAME_SIZE - 1);
    dispatchJob.jobLabel[FFX_RESOURCE_NAME_SIZE - 1] = '\0';
#endif
    dispatchJob.computeJobDescriptor = jobDescriptor;

    return context->contextDescription.backendInterface.fpScheduleGpuJob(&context->contextDescription.backendInterface, &dispatchJob);
}

static FfxErrorCode scheduleFragment(FfxArmFrameInterpolationContext_Private* context, const FfxPipelineState* pipeline, uint32_t width, uint32_t height)
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

    for (uint32_t currentShaderResourceViewIndex = 0; currentShaderResourceViewIndex < pipeline->rtCount; ++currentShaderResourceViewIndex)
    {
        const uint32_t            currentResourceId                       = pipeline->rtBindings[currentShaderResourceViewIndex].resourceIdentifier;
        const FfxResourceInternal currentResource                         = context->uavResources[currentResourceId];
        jobDescriptor.rtTextures[currentShaderResourceViewIndex].resource = currentResource;
#ifdef FFX_DEBUG
        strncpy(jobDescriptor.rtTextures[currentShaderResourceViewIndex].name,
                pipeline->rtBindings[currentShaderResourceViewIndex].name,
                FFX_RESOURCE_NAME_SIZE - 1);
        jobDescriptor.rtTextures[currentShaderResourceViewIndex].name[FFX_RESOURCE_NAME_SIZE - 1] = '\0';
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

    for (uint32_t currentUnorderedAccessViewIndex = 0; currentUnorderedAccessViewIndex < pipeline->uavBufferCount; ++currentUnorderedAccessViewIndex)
    {
        const uint32_t currentResourceId                                   = pipeline->uavBufferBindings[currentUnorderedAccessViewIndex].resourceIdentifier;
        jobDescriptor.uavBuffers[currentUnorderedAccessViewIndex].resource = context->uavResources[currentResourceId];
#ifdef FFX_DEBUG
        strncpy(jobDescriptor.uavBuffers[currentUnorderedAccessViewIndex].name,
                pipeline->uavBufferBindings[currentUnorderedAccessViewIndex].name,
                FFX_RESOURCE_NAME_SIZE - 1);
        jobDescriptor.uavBuffers[currentUnorderedAccessViewIndex].name[FFX_RESOURCE_NAME_SIZE - 1] = '\0';
#endif
    }

    for (uint32_t currentShaderResourceViewIndex = 0; currentShaderResourceViewIndex < pipeline->srvBufferCount; ++currentShaderResourceViewIndex)
    {
        const uint32_t currentResourceId                                  = pipeline->srvBufferBindings[currentShaderResourceViewIndex].resourceIdentifier;
        jobDescriptor.srvBuffers[currentShaderResourceViewIndex].resource = context->srvResources[currentResourceId];
#ifdef FFX_DEBUG
        strncpy(jobDescriptor.srvBuffers[currentShaderResourceViewIndex].name,
                pipeline->srvBufferBindings[currentShaderResourceViewIndex].name,
                FFX_RESOURCE_NAME_SIZE - 1);
        jobDescriptor.srvBuffers[currentShaderResourceViewIndex].name[FFX_RESOURCE_NAME_SIZE - 1] = '\0';
#endif
    }

    for (uint32_t currentTensorIndex = 0; currentTensorIndex < pipeline->srvTensorCount; ++currentTensorIndex)
    {
        const uint32_t currentResourceId                      = pipeline->srvTensorBindings[currentTensorIndex].resourceIdentifier;
        jobDescriptor.srvTensors[currentTensorIndex].resource = context->srvResources[currentResourceId];
#ifdef FFX_DEBUG
        strncpy(jobDescriptor.srvTensors[currentTensorIndex].name, pipeline->srvTensorBindings[currentTensorIndex].name, FFX_RESOURCE_NAME_SIZE - 1);
        jobDescriptor.srvTensors[currentTensorIndex].name[FFX_RESOURCE_NAME_SIZE - 1] = '\0';
#endif
    }

    for (uint32_t currentTensorIndex = 0; currentTensorIndex < pipeline->uavTensorCount; ++currentTensorIndex)
    {
        const uint32_t currentResourceId                      = pipeline->uavTensorBindings[currentTensorIndex].resourceIdentifier;
        jobDescriptor.uavTensors[currentTensorIndex].resource = context->uavResources[currentResourceId];
#ifdef FFX_DEBUG
        strncpy(jobDescriptor.uavTensors[currentTensorIndex].name, pipeline->uavTensorBindings[currentTensorIndex].name, FFX_RESOURCE_NAME_SIZE - 1);
        jobDescriptor.uavTensors[currentTensorIndex].name[FFX_RESOURCE_NAME_SIZE - 1] = '\0';
#endif
    }

    FfxGpuJobDescription dispatchJob = {FFX_GPU_JOB_FRAGMENT};
#ifdef FFX_DEBUG
    strncpy(dispatchJob.jobLabel, pipeline->name, FFX_RESOURCE_NAME_SIZE - 1);
    dispatchJob.jobLabel[FFX_RESOURCE_NAME_SIZE - 1] = '\0';
#endif
    dispatchJob.fragmentJobDescriptor = jobDescriptor;

    return context->contextDescription.backendInterface.fpScheduleGpuJob(&context->contextDescription.backendInterface, &dispatchJob);
}

static FfxErrorCode scheduleDataGraph(FfxArmFrameInterpolationContext_Private* context, const FfxPipelineState* pipeline)
{
    FfxGpuJobDescription dataGraphJob = {FFX_GPU_JOB_DATA_GRAPH};

    for (uint32_t currentTensorIndex = 0; currentTensorIndex < pipeline->srvTensorCount; ++currentTensorIndex)
    {
        const uint32_t            currentResourceId                                  = pipeline->srvTensorBindings[currentTensorIndex].resourceIdentifier;
        const FfxResourceInternal currentResource                                    = context->srvResources[currentResourceId];
        dataGraphJob.dataGraphJobDescription.srvTensors[currentTensorIndex].resource = currentResource;
#ifdef FFX_DEBUG
        strncpy(dataGraphJob.dataGraphJobDescription.srvTensors[currentTensorIndex].name,
                pipeline->srvTensorBindings[currentTensorIndex].name,
                FFX_RESOURCE_NAME_SIZE - 1);
        dataGraphJob.dataGraphJobDescription.srvTensors[currentTensorIndex].name[FFX_RESOURCE_NAME_SIZE - 1] = '\0';
#endif
    }

    for (uint32_t currentTensorIndex = 0; currentTensorIndex < pipeline->uavTensorCount; ++currentTensorIndex)
    {
        const uint32_t            currentResourceId                                  = pipeline->uavTensorBindings[currentTensorIndex].resourceIdentifier;
        const FfxResourceInternal currentResource                                    = context->uavResources[currentResourceId];
        dataGraphJob.dataGraphJobDescription.uavTensors[currentTensorIndex].resource = currentResource;
#ifdef FFX_DEBUG
        strncpy(dataGraphJob.dataGraphJobDescription.uavTensors[currentTensorIndex].name,
                pipeline->uavTensorBindings[currentTensorIndex].name,
                FFX_RESOURCE_NAME_SIZE - 1);
        dataGraphJob.dataGraphJobDescription.uavTensors[currentTensorIndex].name[FFX_RESOURCE_NAME_SIZE - 1] = '\0';
#endif
    }

#ifdef FFX_DEBUG
    strncpy(dataGraphJob.jobLabel, pipeline->name, FFX_RESOURCE_NAME_SIZE - 1);
    dataGraphJob.jobLabel[FFX_RESOURCE_NAME_SIZE - 1] = '\0';
#endif
    dataGraphJob.dataGraphJobDescription.pipeline = *pipeline;

    return context->contextDescription.backendInterface.fpScheduleGpuJob(&context->contextDescription.backendInterface, &dataGraphJob);
}

static FfxErrorCode logErrorWithCallback(FfxErrorCode          errorCode,
                                         const char*           errorMessage,
                                         FfxLogMessageCallback callback = nullptr,
                                         const char*           funcName = FFX_BUILTIN_FUNCTION)
{
    if (!ffxLogIsSuppressed() && callback && errorMessage)
    {
        callback(FFX_MESSAGE_TYPE_ERROR, errorMessage);
    }

    return ffxLogError(errorCode, errorMessage, funcName);
}

FfxErrorCode ffxFrameInterpolationContextCreate(FfxFrameInterpolationContext* context, FfxFrameInterpolationContextDescription* contextDescription)
{
    // zero context memory
    //memset(context, 0, sizeof(FfxFrameInterpolationContext));

    // check pointers are valid.
    if (!context)
    {
        return logErrorWithCallback(
            FFX_ERROR_INVALID_POINTER, "context creation requires a context", contextDescription ? contextDescription->fpMessage : nullptr);
    }
    if (!contextDescription)
    {
        return ffxLogError(FFX_ERROR_INVALID_POINTER, "context creation requires a description");
    }

    // validate that all callbacks are set for the interface
    if (!contextDescription->backendInterface.fpGetSDKVersion)
    {
        return logErrorWithCallback(FFX_ERROR_INCOMPLETE_INTERFACE, "context creation requires fpGetSDKVersion", contextDescription->fpMessage);
    }
    if (!contextDescription->backendInterface.fpCreateBackendContext)
    {
        return logErrorWithCallback(FFX_ERROR_INCOMPLETE_INTERFACE, "context creation requires fpCreateBackendContext", contextDescription->fpMessage);
    }
    if (!contextDescription->backendInterface.fpDestroyBackendContext)
    {
        return logErrorWithCallback(FFX_ERROR_INCOMPLETE_INTERFACE, "context creation requires fpDestroyBackendContext", contextDescription->fpMessage);
    }
    // if a scratch buffer is declared, then we must have a size
    if (contextDescription->backendInterface.scratchBuffer)
    {
        if (!contextDescription->backendInterface.scratchBufferSize)
        {
            return logErrorWithCallback(FFX_ERROR_INCOMPLETE_INTERFACE, "context creation requires scratchBufferSize", contextDescription->fpMessage);
        }
    }

    // ensure the context is large enough for the internal context.
    FFX_STATIC_ASSERT(sizeof(FfxFrameInterpolationContext) >= sizeof(FfxArmFrameInterpolationContext_Private));

    const FfxNfruResolutionValidationError resolutionError =
        ffxNfruValidateResolutionInputs(contextDescription->displaySize, contextDescription->maxRenderSize, contextDescription->mvDepthSize);
    if (resolutionError != FfxNfruResolutionValidationError::None)
    {
        const char* operation = "context creation validates resolution inputs";
        switch (resolutionError)
        {
        case FfxNfruResolutionValidationError::DisplayWidth:
            operation = "context creation validates displaySize.width";
            break;
        case FfxNfruResolutionValidationError::DisplayHeight:
            operation = "context creation validates displaySize.height";
            break;
        case FfxNfruResolutionValidationError::RenderWidth:
            operation = "context creation validates maxRenderSize.width";
            break;
        case FfxNfruResolutionValidationError::RenderHeight:
            operation = "context creation validates maxRenderSize.height";
            break;
        case FfxNfruResolutionValidationError::MvDepthWidth:
            operation = "context creation requires a valid mvDepthSize.width; only {0, 0} selects maxRenderSize";
            break;
        case FfxNfruResolutionValidationError::MvDepthHeight:
            operation = "context creation requires a valid mvDepthSize.height; only {0, 0} selects maxRenderSize";
            break;
        case FfxNfruResolutionValidationError::MvDepthAspectRatio:
            operation = "context creation requires mvDepthSize to have the same aspect ratio as maxRenderSize";
            break;
        case FfxNfruResolutionValidationError::DerivedWidth:
            operation = "context creation validates derived lane widths";
            break;
        case FfxNfruResolutionValidationError::ShaderDimensionOutOfRange:
            operation = "context creation requires displaySize, maxRenderSize, and mvDepthSize within the signed shader range";
            break;
        case FfxNfruResolutionValidationError::DerivedDimensionOutOfRange:
            operation = "context creation requires derived NFRU dimensions within the signed shader range";
            break;
        default:
            break;
        }
        return logErrorWithCallback(FFX_ERROR_INVALID_ARGUMENT, operation, contextDescription->fpMessage);
    }
    if (!ffxNfruSupportsOpticalFlowGrid(contextDescription->opticalFlowGridSize))
    {
        return logErrorWithCallback(FFX_ERROR_INVALID_ARGUMENT, "context creation validates opticalFlowGridSize", contextDescription->fpMessage);
    }

    FfxFrameInterpolationContextDescription resolvedDescription = *contextDescription;
    resolvedDescription.mvDepthSize                             = ffxNfruResolveMvDepthSize(contextDescription->maxRenderSize, contextDescription->mvDepthSize);
    const FfxNfruResolutionPlan resolutionPlan =
        ffxNfruDeriveResolutionPlan(resolvedDescription.displaySize, resolvedDescription.mvDepthSize, FFX_FRAMEINTERPOLATION_DATA_GRAPH_ALIGNMENT);
    const FfxDimensions2D expectedOpticalFlowSize = ffxNfruGetOpticalFlowOutputSize(resolutionPlan.ofInputColour, contextDescription->opticalFlowGridSize);
    if (contextDescription->opticalFlowSize.width != expectedOpticalFlowSize.width ||
        contextDescription->opticalFlowSize.height != expectedOpticalFlowSize.height)
    {
        return logErrorWithCallback(FFX_ERROR_INVALID_ARGUMENT, "context creation validates opticalFlowSize", contextDescription->fpMessage);
    }

    const bool usesSelectableFragmentPass = (contextDescription->flags & FFX_FG_CONTEXT_FLAG_ALL_STAGES_COMPUTE) != FFX_FG_CONTEXT_FLAG_ALL_STAGES_COMPUTE;
    const bool usesFixedFragmentPass = resolutionPlan.needsOfColourDownsample || resolutionPlan.needsAllInputDownsample || resolutionPlan.needsColourDownsample;
    if ((usesSelectableFragmentPass || usesFixedFragmentPass) && !contextDescription->backendInterface.fpCreateGraphicsPipeline)
    {
        return logErrorWithCallback(
            FFX_ERROR_INCOMPLETE_INTERFACE, "context creation requires fpCreateGraphicsPipeline for selected fragment passes", contextDescription->fpMessage);
    }
    if (!contextDescription->backendInterface.fpGetDeviceCapabilities)
    {
        return logErrorWithCallback(FFX_ERROR_INCOMPLETE_INTERFACE, "context creation requires fpGetDeviceCapabilities", contextDescription->fpMessage);
    }

    // create the context.
    ffxLogSdkVersion(contextDescription->fpMessage);
    ffxLogNfruVersion(contextDescription->fpMessage);
    FfxArmFrameInterpolationContext_Private* contextPrivate = (FfxArmFrameInterpolationContext_Private*)(context);
    FfxErrorCode                             errorCode      = frameinterpolationCreate(contextPrivate, &resolvedDescription);
    if (errorCode != FFX_OK)
    {
        logErrorWithCallback(errorCode, "NFRU context creation failed", contextDescription->fpMessage);
        (void)frameinterpolationRelease(contextPrivate);
    }

    return errorCode;
}

FFX_API FfxErrorCode ffxFrameInterpolationContextCreateForFrameGeneration(FfxFrameInterpolationContext*            context,
                                                                          FfxFrameInterpolationContextDescription* contextDescription)
{
    FfxScopedLogSuppression suppression;
    return ffxFrameInterpolationContextCreate(context, contextDescription);
}

FFX_API FfxErrorCode ffxFrameInterpolationContextGetGpuMemoryUsage(FfxFrameInterpolationContext* context, FfxEffectMemoryUsage* vramUsage)
{
    if (!context)
    {
        return ffxLogError(FFX_ERROR_INVALID_POINTER, "memory query requires a context");
    }
    FfxArmFrameInterpolationContext_Private* contextPrivate  = reinterpret_cast<FfxArmFrameInterpolationContext_Private*>(context);
    const auto                               userMsgCallback = contextPrivate ? contextPrivate->contextDescription.fpMessage : nullptr;
    if (!vramUsage)
    {
        return logErrorWithCallback(FFX_ERROR_INVALID_POINTER, "memory query requires vramUsage", userMsgCallback);
    }

    if (!contextPrivate->device)
    {
        return logErrorWithCallback(FFX_ERROR_NULL_DEVICE, "memory query requires a device", userMsgCallback);
    }
    if (!contextPrivate->contextDescription.backendInterface.fpGetEffectGpuMemoryUsage)
    {
        return logErrorWithCallback(FFX_ERROR_INCOMPLETE_INTERFACE, "memory query requires fpGetEffectGpuMemoryUsage", userMsgCallback);
    }

    FfxErrorCode errorCode = contextPrivate->contextDescription.backendInterface.fpGetEffectGpuMemoryUsage(
        &contextPrivate->contextDescription.backendInterface, contextPrivate->effectContextId, vramUsage);
    if (errorCode != FFX_OK)
    {
        return logErrorWithCallback(errorCode, "memory query", userMsgCallback);
    }

    return FFX_OK;
}

FfxErrorCode ffxFrameInterpolationContextDestroy(FfxFrameInterpolationContext* context)
{
    if (!context)
    {
        return ffxLogError(FFX_ERROR_INVALID_POINTER, "context destruction requires a context");
    }

    // destroy the context.
    FfxArmFrameInterpolationContext_Private* contextPrivate = (FfxArmFrameInterpolationContext_Private*)(context);
    const FfxErrorCode                       errorCode      = frameinterpolationRelease(contextPrivate);

    FFX_RETURN_ON_ERROR(errorCode == FFX_OK, errorCode);
    return FFX_OK;
}

FFX_API FfxErrorCode ffxFrameInterpolationContextDestroyForFrameGeneration(FfxFrameInterpolationContext* context)
{
    FfxScopedLogSuppression suppression;
    return ffxFrameInterpolationContextDestroy(context);
}

FfxErrorCode ffxArmFrameInterpolationContextEnqueueRefreshPipelineRequest(FfxFrameInterpolationContext* context)
{
    if (!context)
    {
        return ffxLogError(FFX_ERROR_INVALID_POINTER, "refresh request requires a context");
    }

    FfxArmFrameInterpolationContext_Private* contextPrivate = (FfxArmFrameInterpolationContext_Private*)context;
    contextPrivate->refreshPipelineStates                   = true;

    return FFX_OK;
}

static void setupDeviceDepthToViewSpaceDepthParams(FfxArmFrameInterpolationContext_Private*   context,
                                                   const FrameInterpolationRenderDescription* params,
                                                   FrameInterpolationConstants*               constants)
{
    const bool bInverted = (context->contextDescription.flags & FFX_FG_CONTEXT_FLAG_ENABLE_DEPTH_INVERTED) == FFX_FG_CONTEXT_FLAG_ENABLE_DEPTH_INVERTED;
    const bool bInfinite = (context->contextDescription.flags & FFX_FG_CONTEXT_FLAG_ENABLE_DEPTH_INFINITE) == FFX_FG_CONTEXT_FLAG_ENABLE_DEPTH_INFINITE;

    // An infinite far plane is represented by a separate flag. Do not use the
    // far-plane sentinel in the min/max ordering; the infinite projection
    // formula only needs the near plane.
    float fMin = params->cameraNear;
    float fMax = params->cameraNear;

    if (!bInfinite)
    {
        // Make sure it has no impact if near and far plane values are swapped
        // in dispatch params.
        fMin = FFX_MINIMUM(params->cameraNear, params->cameraFar);
        fMax = FFX_MAXIMUM(params->cameraNear, params->cameraFar);

        if (bInverted)
        {
            float tmp = fMin;
            fMin      = fMax;
            fMax      = tmp;
        }
    }

    // a 0 0 0   x
    // 0 b 0 0   y
    // 0 0 c d   z
    // 0 0 e 0   1

    const float fQ = bInfinite ? 0.0f : fMax / (fMin - fMax);
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

    constants->_DeviceToViewDepth[0] = d * matrix_elem_c[bInverted][bInfinite];
    constants->_DeviceToViewDepth[1] = matrix_elem_e[bInverted][bInfinite] * params->viewSpaceToMetersFactor;

    // Projection and the depth-clip invariants use the full-resolution colour lane.
    const float projectionWidth  = static_cast<float>(params->upscaleSize.width);
    const float projectionHeight = static_cast<float>(params->upscaleSize.height);
    const float aspect           = projectionWidth / projectionHeight;
    const float cotHalfFovY      = cosf(0.5f * params->cameraFovAngleVertical) / sinf(0.5f * params->cameraFovAngleVertical);
    const float a                = cotHalfFovY / aspect;
    const float b                = cotHalfFovY;

    constants->_DeviceToViewDepth[2] = (1.0f / a);
    constants->_DeviceToViewDepth[3] = (1.0f / b);

    const float depthDiagonal     = sqrtf(projectionWidth * projectionWidth + projectionHeight * projectionHeight);
    const float referenceDiagonal = sqrtf(1080.0f * 1080.0f + 1920.0f * 1920.0f);
    const float centerX           = static_cast<float>(static_cast<uint32_t>(projectionWidth * 0.5f));
    const float centerY           = static_cast<float>(static_cast<uint32_t>(projectionHeight * 0.5f));
    const float centerNdcX        = centerY / projectionHeight * 2.0f - 1.0f;
    const float centerNdcY        = centerX / projectionWidth * -2.0f + 1.0f;
    const float projectionX       = constants->_DeviceToViewDepth[2];
    const float projectionY       = constants->_DeviceToViewDepth[3];
    const float cornerLength      = sqrtf(projectionX * projectionX + projectionY * projectionY + 1.0f);
    const float centerLength      = sqrtf(projectionX * projectionX * centerNdcX * centerNdcX + projectionY * projectionY * centerNdcY * centerNdcY + 1.0f);

    constants->_DepthClipScale = 1.37e-05f * (cornerLength / centerLength) * depthDiagonal;
    constants->_DepthClipPower = 1.0f + 2.0f * FFX_MINIMUM(depthDiagonal / referenceDiagonal, 1.0f);
}

FFX_API bool ffxArmFrameInterpolationResourceIsNull(const FfxResource& resource)
{
    return resource.resource == NULL;
}

static const float debugBarColorSequence[] = {
    0.0f,  1.0f,  1.0f,  // teal
    1.0f,  0.42f, 0.0f,  // orange
    0.0f,  0.16f, 1.0f,  // blue
    0.74f, 1.0f,  0.0f,  // lime
    0.68f, 0.0f,  1.0f,  // purple
    0.0f,  1.0f,  0.1f,  // green
    1.0f,  1.0f,  0.48f  // bright yellow
};
const size_t debugBarColorSequenceLength = 7;

FFX_API FfxErrorCode ffxFrameInterpolationPrepare(FfxFrameInterpolationContext* context, const FfxFrameInterpolationPrepareDescription* params)
{
    if (!context)
    {
        return ffxLogError(FFX_ERROR_INVALID_POINTER, "prepare requires a context");
    }
    FfxArmFrameInterpolationContext_Private* contextPrivate  = (FfxArmFrameInterpolationContext_Private*)(context);
    const auto                               userMsgCallback = contextPrivate ? contextPrivate->contextDescription.fpMessage : nullptr;
    if (!params)
    {
        return logErrorWithCallback(FFX_ERROR_INVALID_POINTER, "prepare requires parameters", userMsgCallback);
    }
    if (!params->commandList)
    {
        return logErrorWithCallback(FFX_ERROR_INVALID_POINTER, "prepare requires a command list", userMsgCallback);
    }
    if (!std::isfinite(params->mvSimilarityRelativeThreshold))
    {
        return logErrorWithCallback(FFX_ERROR_INVALID_ARGUMENT, "prepare relative motion-vector similarity threshold must be finite", userMsgCallback);
    }

    const FfxDimensions2D mvDepthSize = contextPrivate->contextDescription.mvDepthSize;
    const bool            usesInternalDepthHistory =
        (contextPrivate->contextDescription.flags & FFX_FG_CONTEXT_FLAG_MANAGE_PREVIOUS_DEPTH) != 0 || contextPrivate->resolutionPlan.needsAllInputDownsample;
    const auto validateInputSize = [&](const FfxResource& resource, const char* name, FfxDimensions2D expectedSize) -> FfxErrorCode {
        if (!resource.resource)
        {
            char message[128];
            snprintf(message, sizeof(message), "prepare requires %s", name);
            return logErrorWithCallback(FFX_ERROR_INVALID_POINTER, message, userMsgCallback);
        }
        if (resource.description.width != expectedSize.width || resource.description.height != expectedSize.height)
        {
            char message[192];
            snprintf(message,
                     sizeof(message),
                     "prepare %s dimensions (%u x %u) do not match the configured MV/depth dimensions (%u x %u)",
                     name,
                     resource.description.width,
                     resource.description.height,
                     expectedSize.width,
                     expectedSize.height);
            return logErrorWithCallback(FFX_ERROR_INVALID_ARGUMENT, message, userMsgCallback);
        }
        return FFX_OK;
    };
    FFX_VALIDATE(validateInputSize(params->depth, "depth", mvDepthSize));
    // Keep accepting the legacy display-resolution motion-vector resource when its flag is set.
    const bool displayMotionResource = (contextPrivate->contextDescription.flags & FFX_FG_CONTEXT_FLAG_ENABLE_DISPLAY_RESOLUTION_MOTION_VECTORS) &&
                                       params->motionVectors.description.width == contextPrivate->contextDescription.displaySize.width &&
                                       params->motionVectors.description.height == contextPrivate->contextDescription.displaySize.height;
    FFX_VALIDATE(
        validateInputSize(params->motionVectors, "motionVectors", displayMotionResource ? contextPrivate->contextDescription.displaySize : mvDepthSize));
    // External history may already be reduced to the logical lane; both are sampled by normalized coordinates.
    const bool laneSizedHistory =
        params->depthTm1.description.width == contextPrivate->mvDepthLane.width && params->depthTm1.description.height == contextPrivate->mvDepthLane.height;
    FFX_VALIDATE(validateInputSize(params->depthTm1, "depthTm1", usesInternalDepthHistory || laneSizedHistory ? contextPrivate->mvDepthLane : mvDepthSize));
    if (usesInternalDepthHistory)
    {
        FFX_VALIDATE(validateInputSize(params->depthTm1Next, "depthTm1Next", contextPrivate->mvDepthLane));
    }

    // Set constant buffer.
    // Calculate reprojection matrix TM1 -> TP1
    FfxFloat32x4x4 invViewProjTM1;
    matrixInvert(contextPrivate->lastFrameViewProjection, invViewProjTM1);
    matrixMul(params->viewProjection, invViewProjTM1, contextPrivate->constants._MotionTM1ToTP1);
    // Calculate reprojection matrix TP1 -> TM1
    FfxFloat32x4x4 invViewProjTP1;
    matrixInvert(params->viewProjection, invViewProjTP1);
    matrixMul(contextPrivate->lastFrameViewProjection, invViewProjTP1, contextPrivate->constants._MotionTP1ToTM1);

    // Sample-position jitter is in render pixels, independent of motion-vector input resolution and polarity.
    contextPrivate->constants._JitterCancellation[0] =
        (params->jitterOffset.x - contextPrivate->constants._Jitter[0]) / contextPrivate->contextDescription.maxRenderSize.width;
    contextPrivate->constants._JitterCancellation[1] =
        (params->jitterOffset.y - contextPrivate->constants._Jitter[1]) / contextPrivate->contextDescription.maxRenderSize.height;
    contextPrivate->constants._Jitter[0] = params->jitterOffset.x;
    contextPrivate->constants._Jitter[1] = params->jitterOffset.y;

    const FfxDimensions2D motionVectorsTargetSize = (contextPrivate->contextDescription.flags & FFX_FG_CONTEXT_FLAG_ENABLE_DISPLAY_RESOLUTION_MOTION_VECTORS)
                                                        ? contextPrivate->contextDescription.displaySize
                                                        : contextPrivate->contextDescription.mvDepthSize;
    contextPrivate->constants._MotionVectorScale[0] = params->motionVectorScale.x / motionVectorsTargetSize.width;
    contextPrivate->constants._MotionVectorScale[1] = params->motionVectorScale.y / motionVectorsTargetSize.height;
    contextPrivate->constants._MvSimilarityThresholds[0] =
        params->mvSimilarityThreshold > 0.0f ? params->mvSimilarityThreshold : MV_SIMILARITY_THRESHOLD_DEFAULT;
    contextPrivate->constants._MvSimilarityThresholds[1] =
        params->mvSimilarityNoiseThreshold > 0.0f ? params->mvSimilarityNoiseThreshold : MV_SIMILARITY_NOISE_THRESHOLD_DEFAULT;
    contextPrivate->constants._MvSimilarityThresholds[2] =
        params->mvSimilarityRelativeThreshold > 0.0f ? params->mvSimilarityRelativeThreshold : MV_SIMILARITY_RELATIVE_THRESHOLD_DEFAULT;

    contextPrivate->renderDescription.cameraFar               = params->cameraFar;
    contextPrivate->renderDescription.cameraNear              = params->cameraNear;
    contextPrivate->renderDescription.viewSpaceToMetersFactor = (params->viewSpaceToMetersFactor > 0.0f) ? params->viewSpaceToMetersFactor : 1.0f;
    contextPrivate->renderDescription.cameraFovAngleVertical  = params->cameraFovAngleVertical;
    contextPrivate->renderDescription.renderSize  = {(uint32_t)contextPrivate->constants._RenderSize[0], (uint32_t)contextPrivate->constants._RenderSize[1]};
    contextPrivate->renderDescription.upscaleSize = {(uint32_t)contextPrivate->constants._DisplaySize[0], (uint32_t)contextPrivate->constants._DisplaySize[1]};
    setupDeviceDepthToViewSpaceDepthParams(contextPrivate, &contextPrivate->renderDescription, &contextPrivate->constants);

    FFX_ASSERT(!ffxArmFrameInterpolationResourceIsNull(params->depth));
    FFX_ASSERT(!ffxArmFrameInterpolationResourceIsNull(params->motionVectors));

    DynamicResourceCleanup cleanup{&contextPrivate->contextDescription.backendInterface, params->commandList, contextPrivate->effectContextId, true};
    FFX_VALIDATE(contextPrivate->contextDescription.backendInterface.fpRegisterResource(
        &contextPrivate->contextDescription.backendInterface,
        &params->depth,
        contextPrivate->effectContextId,
        &contextPrivate->srvResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_DEPTH]));

    FFX_VALIDATE(contextPrivate->contextDescription.backendInterface.fpRegisterResource(
        &contextPrivate->contextDescription.backendInterface,
        &params->motionVectors,
        contextPrivate->effectContextId,
        &contextPrivate->srvResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_MOTION_VECTORS]));

    FFX_VALIDATE(contextPrivate->contextDescription.backendInterface.fpRegisterResource(
        &contextPrivate->contextDescription.backendInterface,
        &params->depthTm1,
        contextPrivate->effectContextId,
        &contextPrivate->srvResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_DEPTH_TM1]));

    contextPrivate->uavResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_DEPTH_TM1] =
        contextPrivate->srvResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_DEPTH_TM1];

    if (params->depthTm1Next.resource != nullptr)
    {
        FFX_VALIDATE(contextPrivate->contextDescription.backendInterface.fpRegisterResource(
            &contextPrivate->contextDescription.backendInterface,
            &params->depthTm1Next,
            contextPrivate->effectContextId,
            &contextPrivate->srvResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_DEPTH_TM1_NEXT]));

        contextPrivate->uavResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_DEPTH_TM1_NEXT] =
            contextPrivate->srvResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_DEPTH_TM1_NEXT];
    }
    else
    {
        // External-history mode reads current P1 depth directly and must not retain a writable history binding.
        contextPrivate->srvResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_DEPTH_TM1_NEXT] =
            contextPrivate->srvResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_DEPTH];
        contextPrivate->uavResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_DEPTH_TM1_NEXT] = {};
    }

    FFX_VALIDATE(contextPrivate->contextDescription.backendInterface.fpRegisterResource(
        &contextPrivate->contextDescription.backendInterface,
        &params->colorTm1,
        contextPrivate->effectContextId,
        &contextPrivate->srvResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_PREVIOUS_INTERPOLATION_SOURCE]));

    // Register the caller-supplied backup of colorTm1 as the UAV / render-target the
    // postprocess pass writes into this frame. The SRV side still points at colorTm1 so
    // an async interpolation pass can keep reading the previous frame's source.
    FFX_VALIDATE(contextPrivate->contextDescription.backendInterface.fpRegisterResource(
        &contextPrivate->contextDescription.backendInterface,
        &params->colorTm1Backup,
        contextPrivate->effectContextId,
        &contextPrivate->uavResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_PREVIOUS_INTERPOLATION_SOURCE_BACKUP]));

    if (!contextPrivate->resolutionPlan.needsAllInputDownsample && !contextPrivate->resolutionPlan.needsColourDownsample)
    {
        contextPrivate->srvResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_PREPROCESS_COLOUR_M1] =
            contextPrivate->srvResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_PREVIOUS_INTERPOLATION_SOURCE];
    }
    memcpy(contextPrivate->pendingViewProjection, params->viewProjection, sizeof(params->viewProjection));
    cleanup.active = false;
    return FFX_OK;
}

FFX_API FfxErrorCode ffxFrameInterpolationPrepareForFrameGeneration(FfxFrameInterpolationContext*                  context,
                                                                    const FfxFrameInterpolationPrepareDescription* params)
{
    FfxScopedLogSuppression suppression;
    return ffxFrameInterpolationPrepare(context, params);
}

static FfxErrorCode dispatchFrameInterpolation(FfxFrameInterpolationContext*                   context,
                                               const FfxFrameInterpolationDispatchDescription* params,
                                               FfxNfruFrameInterpolationTrace*                 trace)
{
    FfxArmFrameInterpolationContext_Private*   contextPrivate = (FfxArmFrameInterpolationContext_Private*)(context);
    const FrameInterpolationRenderDescription* renderDesc     = &contextPrivate->renderDescription;
    FfxDimensions2D                            displaySize    = contextPrivate->contextDescription.displaySize;
    const FfxDimensions2D                      mvDepthLane    = contextPrivate->mvDepthLane;
    const FfxDimensions2D                      flowLane       = contextPrivate->flowLane;
    const bool jitterCancellation = (contextPrivate->contextDescription.flags & FFX_FG_CONTEXT_FLAG_ENABLE_MOTION_VECTORS_JITTER_CANCELLATION) != 0;

    const bool collectTimings = trace && trace->collectTimings;
    if (trace)
    {
        const bool requestedTimings = trace->collectTimings;
        *trace                      = {};
        trace->collectTimings       = requestedTimings;
    }
    const auto setTraceStage = [trace](FfxNfruFiStage stage) {
        if (trace)
        {
            trace->failedStage = stage;
        }
    };

    if (contextPrivate->refreshPipelineStates)
    {
        setTraceStage(FfxNfruFiStage::PipelineRefresh);
        FFX_VALIDATE(createPipelineStates(contextPrivate, contextPrivate->dataGraphTensorInfo, contextPrivate->dataGraphTensorInfoCount));
        contextPrivate->refreshPipelineStates = false;
    }

    const bool bReset = (contextPrivate->dispatchCount == 0) || params->reset;

    // Detect disjoint frameID values
    const bool bFrameID_Decreased = params->frameID < contextPrivate->previousFrameID;
    const bool bFrameID_Skipped   = !bFrameID_Decreased && (params->frameID - contextPrivate->previousFrameID) > 1;
    const bool bDisjointFrameID   = bFrameID_Decreased || bFrameID_Skipped;

    // Set constant buffer.
    contextPrivate->constants._Reset = bReset || bDisjointFrameID;
    if (!jitterCancellation || contextPrivate->constants._Reset)
    {
        contextPrivate->constants._JitterCancellation[0] = 0.0f;
        contextPrivate->constants._JitterCancellation[1] = 0.0f;
    }

    setTraceStage(FfxNfruFiStage::Preprocess);
    FFX_VALIDATE(contextPrivate->contextDescription.backendInterface.fpStageConstantBufferDataFunc(
        &contextPrivate->contextDescription.backendInterface,
        &contextPrivate->constants,
        sizeof(contextPrivate->constants),
        &contextPrivate->constantBuffers[FFX_FRAMEINTERPOLATION_CONSTANTBUFFER_IDENTIFIER]));

    DynamicResourceCleanup cleanup{&contextPrivate->contextDescription.backendInterface, params->commandList, contextPrivate->effectContextId, true};
    FFX_VALIDATE(contextPrivate->contextDescription.backendInterface.fpRegisterResource(
        &contextPrivate->contextDescription.backendInterface,
        &params->currentBackBuffer,
        contextPrivate->effectContextId,
        &contextPrivate->srvResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_CURRENT_INTERPOLATION_SOURCE]));
    contextPrivate->srvResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_PRESENT_BACKBUFFER] =
        contextPrivate->srvResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_CURRENT_INTERPOLATION_SOURCE];
    if (!contextPrivate->resolutionPlan.needsOfColourDownsample)
    {
        // No OF-colour downsample is required, so optical flow reads the full-resolution input directly.
        contextPrivate->srvResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_OF_COLOUR_P1] =
            contextPrivate->srvResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_CURRENT_INTERPOLATION_SOURCE];
    }
    if (!contextPrivate->resolutionPlan.needsAllInputDownsample && !contextPrivate->resolutionPlan.needsColourDownsample)
    {
        contextPrivate->srvResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_PREPROCESS_COLOUR_P1] =
            contextPrivate->srvResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_CURRENT_INTERPOLATION_SOURCE];
    }

    // Register output as SRV and UAV
    FFX_VALIDATE(contextPrivate->contextDescription.backendInterface.fpRegisterResource(
        &contextPrivate->contextDescription.backendInterface,
        &params->output,
        contextPrivate->effectContextId,
        &contextPrivate->uavResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_OUTPUT]));
    contextPrivate->srvResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_OUTPUT] =
        contextPrivate->uavResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_OUTPUT];

    // set optical flow buffers
    if (params->opticalFlowScale.x > 0)
    {
        FFX_VALIDATE(contextPrivate->contextDescription.backendInterface.fpRegisterResource(
            &contextPrivate->contextDescription.backendInterface,
            &params->opticalFlowVector,
            contextPrivate->effectContextId,
            &contextPrivate->srvResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_OPTICAL_FLOW_VECTOR]));
    }
    else
    {
        contextPrivate->srvResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_OPTICAL_FLOW_VECTOR] = {};
    }

    const bool bExecutePreparationPasses = (false == contextPrivate->constants._Reset);

    const bool warpUseFragment        = (contextPrivate->pipelineFlags & FRAMEINTERPOLATION_SHADER_PERMUTATION_WARP_COMPUTE) == 0;
    const bool fillHolesUseFragment   = (contextPrivate->pipelineFlags & FRAMEINTERPOLATION_SHADER_PERMUTATION_FILL_HOLES_COMPUTE) == 0;
    const bool preProcessUseFragment  = (contextPrivate->pipelineFlags & FRAMEINTERPOLATION_SHADER_PERMUTATION_PRE_PROCESS_COMPUTE) == 0;
    const bool postprocessUseFragment = (contextPrivate->pipelineFlags & FRAMEINTERPOLATION_SHADER_PERMUTATION_POSTPROCESS_COMPUTE) == 0;
    const bool debugViewUseFragment   = (contextPrivate->pipelineFlags & FRAMEINTERPOLATION_SHADER_PERMUTATION_DEBUG_VIEW_COMPUTE) == 0;
    const bool usesInternalDepthHistory =
        (contextPrivate->contextDescription.flags & FFX_FG_CONTEXT_FLAG_MANAGE_PREVIOUS_DEPTH) != 0 || contextPrivate->resolutionPlan.needsAllInputDownsample;

    const auto scheduleInitWarp = [&]() -> FfxErrorCode {
        uint32_t dispatchSizeX = FFX_DIVIDE_ROUNDING_UP(std::max(mvDepthLane.width, flowLane.width), FFX_FRAMEINTERPOLATION_THREAD_GROUP_WIDTH);
        uint32_t dispatchSizeY = FFX_DIVIDE_ROUNDING_UP(std::max(mvDepthLane.height, flowLane.height), FFX_FRAMEINTERPOLATION_THREAD_GROUP_HEIGHT);
        return scheduleDispatch(contextPrivate, &contextPrivate->pipelineInitWarp, dispatchSizeX, dispatchSizeY);
    };
    const auto schedulePreWarpDownsample = [&]() -> FfxErrorCode {
        if (contextPrivate->resolutionPlan.needsAllInputDownsample)
        {
            return scheduleFragment(contextPrivate, &contextPrivate->pipelineDownsampleAllInput, mvDepthLane.width, mvDepthLane.height);
        }
        if (contextPrivate->resolutionPlan.needsColourDownsample)
        {
            return scheduleFragment(contextPrivate, &contextPrivate->pipelineDownsampleColour, mvDepthLane.width, mvDepthLane.height);
        }
        return FFX_OK;
    };
    const auto bindWarpInputs = [&]() {
        if (contextPrivate->resolutionPlan.needsAllInputDownsample)
        {
            contextPrivate->srvResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_DEPTH] =
                contextPrivate->srvResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_DEPTH_TM1_NEXT];
            // The shader keeps the generic motion binding; the all-input path resolves it to the reduced resource here.
            contextPrivate->srvResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_MOTION_VECTORS] =
                contextPrivate->srvResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_REDUCED_MOTION_P1];
        }
    };
    ///////////////////
    // NFRU workloads //
    ///////////////////
    NfruCpuClock::time_point preScheduleStart{};
    NfruCpuClock::time_point postScheduleStart{};
    bool                     postTimingStarted = false;
    if (collectTimings)
    {
        preScheduleStart = NfruCpuClock::now();
    }
    {
        // When triggering the reset operation, the TP1 mask needs to be cleared, and frame generation is not required.
        if (!bExecutePreparationPasses)
        {
            // On reset, still run warp_flow to seed depth_tm1_next for the next frame's depth_tm1.
            // The init_warp pass zeroes QDATA/holes so warp's other outputs are safely initialized.
            FFX_VALIDATE(scheduleInitWarp());
            FFX_VALIDATE(schedulePreWarpDownsample());
            bindWarpInputs();
            // Warp seeds P1 depth history on topologies where it is the depth-history producer.
            // Its dynamic-mask output is intentionally discarded by the clear below.
            if (warpUseFragment)
            {
                FFX_VALIDATE(scheduleFragment(contextPrivate, &contextPrivate->pipelineWarpFlow, mvDepthLane.width, mvDepthLane.height));
            }
            else
            {
                uint32_t dispatchSizeX = FFX_DIVIDE_ROUNDING_UP(mvDepthLane.width, FFX_FRAMEINTERPOLATION_THREAD_GROUP_WIDTH);
                uint32_t dispatchSizeY = FFX_DIVIDE_ROUNDING_UP(mvDepthLane.height, FFX_FRAMEINTERPOLATION_THREAD_GROUP_HEIGHT);
                FFX_VALIDATE(scheduleDispatch(contextPrivate, &contextPrivate->pipelineWarpFlow, dispatchSizeX, dispatchSizeY));
            }
            FfxGpuJobDescription clearJobs    = {FFX_GPU_JOB_CLEAR_FLOAT};
            FfxResourceInternal  clearSources = contextPrivate->uavResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_DYNAMIC_MASK_TP1];

            const float clearValue[]{1.0f, 1.0f, 1.0f, 1.0f};
            memcpy(clearJobs.clearJobDescriptor.color, clearValue, 4 * sizeof(float));
#ifdef FFX_DEBUG
            strncpy(clearJobs.jobLabel, "Clear Dynamic Mask TP1 texture", FFX_RESOURCE_NAME_SIZE - 1);
            clearJobs.jobLabel[FFX_RESOURCE_NAME_SIZE - 1] = '\0';
#endif
            clearJobs.clearJobDescriptor.target = clearSources;
            FFX_VALIDATE(
                contextPrivate->contextDescription.backendInterface.fpScheduleGpuJob(&contextPrivate->contextDescription.backendInterface, &clearJobs));

            // copy current frame to interpolation output to avoid undefined output when resetting.
            FfxGpuJobDescription resetcopyJob  = {FFX_GPU_JOB_COPY};
            resetcopyJob.copyJobDescriptor.src = contextPrivate->srvResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_CURRENT_INTERPOLATION_SOURCE];
            resetcopyJob.copyJobDescriptor.dst = contextPrivate->uavResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_OUTPUT];
            FFX_VALIDATE(
                contextPrivate->contextDescription.backendInterface.fpScheduleGpuJob(&contextPrivate->contextDescription.backendInterface, &resetcopyJob));
        }
        else
        {
            // Zero-initialise QMVD and MV Holes before warp passes
            FFX_VALIDATE(scheduleInitWarp());
            FFX_VALIDATE(schedulePreWarpDownsample());
            bindWarpInputs();

            /////////////////
            // T+1 MV work //
            /////////////////
            // Generate warped filled motion using this frame's motion vectors
            // Warp motion vectors and encode with depth
            if (warpUseFragment)
            {
                FFX_VALIDATE(scheduleFragment(contextPrivate, &contextPrivate->pipelineWarpFlow, mvDepthLane.width, mvDepthLane.height));
            }
            else
            {
                uint32_t dispatchSizeX = FFX_DIVIDE_ROUNDING_UP(mvDepthLane.width, FFX_FRAMEINTERPOLATION_THREAD_GROUP_WIDTH);
                uint32_t dispatchSizeY = FFX_DIVIDE_ROUNDING_UP(mvDepthLane.height, FFX_FRAMEINTERPOLATION_THREAD_GROUP_HEIGHT);
                FFX_VALIDATE(scheduleDispatch(contextPrivate, &contextPrivate->pipelineWarpFlow, dispatchSizeX, dispatchSizeY));
            }
            /////////////////
            // T-1 OF work //
            /////////////////
            // Generated warped filled OF using the Optical Flow and last frame's depth
            // Warp Optical Flow and encode with last frame's depth
            if (warpUseFragment)
            {
                FFX_VALIDATE(scheduleFragment(contextPrivate, &contextPrivate->pipelineWarpFlowTM1, flowLane.width, flowLane.height));
            }
            else
            {
                uint32_t dispatchSizeX = FFX_DIVIDE_ROUNDING_UP(flowLane.width, FFX_FRAMEINTERPOLATION_WARP_FLOW_TM1_THREAD_GROUP_WIDTH);
                uint32_t dispatchSizeY = FFX_DIVIDE_ROUNDING_UP(flowLane.height, FFX_FRAMEINTERPOLATION_WARP_FLOW_TM1_THREAD_GROUP_HEIGHT);
                FFX_VALIDATE(scheduleDispatch(contextPrivate, &contextPrivate->pipelineWarpFlowTM1, dispatchSizeX, dispatchSizeY));
            }
            // Fill holes TM1
            if (fillHolesUseFragment)
            {
                FFX_VALIDATE(scheduleFragment(contextPrivate, &contextPrivate->pipelineFillHolesTM1, flowLane.width, flowLane.height));
            }
            else
            {
                uint32_t dispatchSizeX = FFX_DIVIDE_ROUNDING_UP(flowLane.width, FFX_FRAMEINTERPOLATION_THREAD_GROUP_WIDTH);
                uint32_t dispatchSizeY = FFX_DIVIDE_ROUNDING_UP(flowLane.height, FFX_FRAMEINTERPOLATION_THREAD_GROUP_HEIGHT);
                FFX_VALIDATE(scheduleDispatch(contextPrivate, &contextPrivate->pipelineFillHolesTM1, dispatchSizeX, dispatchSizeY));
            }

            ////////////////////
            // Generate frame //
            ////////////////////
            // Preprocessing
            if (preProcessUseFragment)
            {
                FFX_VALIDATE(
                    scheduleFragment(contextPrivate, &contextPrivate->pipelinePreprocess, contextPrivate->dataGraphWidth, contextPrivate->dataGraphHeight));
            }
            else
            {
                uint32_t dispatchSizeX = FFX_DIVIDE_ROUNDING_UP(contextPrivate->dataGraphWidth, FFX_FRAMEINTERPOLATION_PREPROCESS_THREAD_GROUP_WIDTH);
                uint32_t dispatchSizeY = FFX_DIVIDE_ROUNDING_UP(contextPrivate->dataGraphHeight, FFX_FRAMEINTERPOLATION_PREPROCESS_THREAD_GROUP_HEIGHT);
                FFX_VALIDATE(scheduleDispatch(contextPrivate, &contextPrivate->pipelinePreprocess, dispatchSizeX, dispatchSizeY));
            }

            // 31 seed -> level 0
            {
                const FfxResourceInternal seed        = contextPrivate->srvResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_INPAINT_SEED];
                const FfxResourceInternal destination = contextPrivate->uavResources[INPAINT_LEVEL_RESOURCE_IDS[0]];
                FFX_ASSERT(seed.internalIndex != destination.internalIndex);
                uint32_t dx = FFX_DIVIDE_ROUNDING_UP(flowLane.width, FFX_FRAMEINTERPOLATION_INPAINT_GROUP_WIDTH);
                uint32_t dy = FFX_DIVIDE_ROUNDING_UP(flowLane.height, FFX_FRAMEINTERPOLATION_INPAINT_GROUP_HEIGHT);
                FFX_VALIDATE(scheduleDispatch(contextPrivate, &contextPrivate->pipelineDownsampleInpaintSeed, dx, dy, nullptr, &destination));
            }
            // 31 level i-1 -> i. Per-job overrides make the source explicit and keep SRV/UAV images distinct.
            for (uint32_t level = 1; level < FFX_FRAMEINTERPOLATION_INPAINT_LEVEL_COUNT; ++level)
            {
                const FfxResourceInternal source      = contextPrivate->srvResources[INPAINT_LEVEL_RESOURCE_IDS[level - 1]];
                const FfxResourceInternal destination = contextPrivate->uavResources[INPAINT_LEVEL_RESOURCE_IDS[level]];
                FFX_ASSERT(source.internalIndex != destination.internalIndex);
                uint32_t levelWidth  = std::max(1u, flowLane.width >> level);
                uint32_t levelHeight = std::max(1u, flowLane.height >> level);
                uint32_t dx          = FFX_DIVIDE_ROUNDING_UP(levelWidth, FFX_FRAMEINTERPOLATION_INPAINT_GROUP_WIDTH);
                uint32_t dy          = FFX_DIVIDE_ROUNDING_UP(levelHeight, FFX_FRAMEINTERPOLATION_INPAINT_GROUP_HEIGHT);
                FFX_VALIDATE(scheduleDispatch(contextPrivate, &contextPrivate->pipelineDownsampleInpaintMip, dx, dy, &source, &destination));
            }

            if (contextPrivate->enableDataGraphFI)
            {
                // NFRU Interpolation
                if (collectTimings)
                {
                    trace->preScheduleUs = elapsedMicroseconds(preScheduleStart, NfruCpuClock::now());
                }
                setTraceStage(FfxNfruFiStage::DataGraph);
                NfruCpuClock::time_point graphScheduleStart{};
                if (collectTimings)
                {
                    graphScheduleStart = NfruCpuClock::now();
                }
                FFX_VALIDATE(scheduleDataGraph(contextPrivate, &contextPrivate->pipelineNFRUInterpolation));
                if (collectTimings)
                {
                    trace->graphScheduleUs = elapsedMicroseconds(graphScheduleStart, NfruCpuClock::now());
                    postScheduleStart      = NfruCpuClock::now();
                    postTimingStarted      = true;
                }
                setTraceStage(FfxNfruFiStage::Postprocess);
            }

            // 50 resolve the five logical levels -> inpaint result (flow lane)
            {
                uint32_t dx = FFX_DIVIDE_ROUNDING_UP(flowLane.width, FFX_FRAMEINTERPOLATION_INPAINT_GROUP_WIDTH);
                uint32_t dy = FFX_DIVIDE_ROUNDING_UP(flowLane.height, FFX_FRAMEINTERPOLATION_INPAINT_GROUP_HEIGHT);
                FFX_VALIDATE(scheduleDispatch(contextPrivate, &contextPrivate->pipelineInpaint, dx, dy));
            }
            // Postprocessing
            if (postprocessUseFragment)
            {
                FFX_VALIDATE(scheduleFragment(contextPrivate, &contextPrivate->pipelinePostprocess, displaySize.width, displaySize.height));
            }
            else
            {
                uint32_t dispatchSizeX = FFX_DIVIDE_ROUNDING_UP(displaySize.width, FFX_FRAMEINTERPOLATION_THREAD_GROUP_WIDTH);
                uint32_t dispatchSizeY = FFX_DIVIDE_ROUNDING_UP(displaySize.height, FFX_FRAMEINTERPOLATION_THREAD_GROUP_HEIGHT);
                FFX_VALIDATE(scheduleDispatch(contextPrivate, &contextPrivate->pipelinePostprocess, dispatchSizeX, dispatchSizeY));
            }
            if (params->flags & FFX_FRAMEINTERPOLATION_DISPATCH_DRAW_DEBUG_VIEW)
            {
                if (debugViewUseFragment)
                {
                    FFX_VALIDATE(scheduleFragment(contextPrivate, &contextPrivate->pipelineDebugView, displaySize.width, displaySize.height));
                }
                else
                {
                    uint32_t dispatchSizeX = FFX_DIVIDE_ROUNDING_UP(displaySize.width, FFX_FRAMEINTERPOLATION_THREAD_GROUP_WIDTH);
                    uint32_t dispatchSizeY = FFX_DIVIDE_ROUNDING_UP(displaySize.height, FFX_FRAMEINTERPOLATION_THREAD_GROUP_HEIGHT);
                    FFX_VALIDATE(scheduleDispatch(contextPrivate, &contextPrivate->pipelineDebugView, dispatchSizeX, dispatchSizeY));
                }
            }
        }

        // Store current buffer as the next frame's previous-source. Skipped when:
        //   * the user manages the previous colour externally (MANAGE flag not set), or
        //   * the preparation passes ran this frame and already populated the backup
        //     (PREVIOUS_INTERPOLATION_SOURCE_BACKUP) via the postprocess pass.
        if ((contextPrivate->contextDescription.flags & FFX_FG_CONTEXT_FLAG_MANAGE_PREVIOUS_COLOR) && !bExecutePreparationPasses)
        {
            FfxGpuJobDescription copyJobs  = {FFX_GPU_JOB_COPY};
            copyJobs.copyJobDescriptor.src = contextPrivate->srvResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_CURRENT_INTERPOLATION_SOURCE];
            copyJobs.copyJobDescriptor.dst = contextPrivate->uavResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_PREVIOUS_INTERPOLATION_SOURCE_BACKUP];
            FFX_VALIDATE(contextPrivate->contextDescription.backendInterface.fpScheduleGpuJob(&contextPrivate->contextDescription.backendInterface, &copyJobs));
        }

        if (collectTimings)
        {
            if (postTimingStarted)
            {
                trace->postScheduleUs = elapsedMicroseconds(postScheduleStart, NfruCpuClock::now());
            }
            else
            {
                trace->preScheduleUs = elapsedMicroseconds(preScheduleStart, NfruCpuClock::now());
            }
        }

        // schedule optical flow and frame interpolation
        setTraceStage(FfxNfruFiStage::RecordCommands);
        const FfxErrorCode executeResult = contextPrivate->contextDescription.backendInterface.fpExecuteGpuJobs(
            &contextPrivate->contextDescription.backendInterface, params->commandList, contextPrivate->effectContextId);
        if (executeResult != FFX_OK)
        {
            return executeResult;
        }
    }

    // release dynamic resources
    setTraceStage(FfxNfruFiStage::Unregister);
    const FfxErrorCode unregisterResult = contextPrivate->contextDescription.backendInterface.fpUnregisterResources(
        &contextPrivate->contextDescription.backendInterface, params->commandList, contextPrivate->effectContextId);
    cleanup.active = false;
    FFX_VALIDATE(unregisterResult);

    // Ping-pong the downsampled colour buffers: this frame's P1 becomes next frame's M1.
    // Runs unconditionally (including reset frames) so that M1 is seeded for the next frame.
    if (contextPrivate->resolutionPlan.needsOfColourDownsample)
    {
        std::swap(contextPrivate->srvResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_OF_COLOUR_P1],
                  contextPrivate->srvResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_OF_COLOUR_M1]);
        std::swap(contextPrivate->uavResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_OF_COLOUR_P1],
                  contextPrivate->uavResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_OF_COLOUR_M1]);
    }
    if (contextPrivate->resolutionPlan.needsAllInputDownsample || contextPrivate->resolutionPlan.needsColourDownsample)
    {
        std::swap(contextPrivate->srvResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_PREPROCESS_COLOUR_P1],
                  contextPrivate->srvResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_PREPROCESS_COLOUR_M1]);
        std::swap(contextPrivate->uavResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_PREPROCESS_COLOUR_P1],
                  contextPrivate->uavResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_PREPROCESS_COLOUR_M1]);
    }

    if (!contextPrivate->resolutionPlan.needsOfColourDownsample)
    {
        contextPrivate->srvResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_OF_COLOUR_P1] = {};
    }
    if (!contextPrivate->resolutionPlan.needsAllInputDownsample && !contextPrivate->resolutionPlan.needsColourDownsample)
    {
        contextPrivate->srvResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_PREPROCESS_COLOUR_P1] = {};
        contextPrivate->srvResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_PREPROCESS_COLOUR_M1] = {};
    }
    if (!usesInternalDepthHistory)
    {
        contextPrivate->srvResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_DEPTH_TM1_NEXT] = {};
    }

    // Ping-pong dynamic mask: this frame's TP1 becomes next frame's TM1.
    std::swap(contextPrivate->srvResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_DYNAMIC_MASK_TP1],
              contextPrivate->srvResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_DYNAMIC_MASK_TM1]);
    std::swap(contextPrivate->uavResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_DYNAMIC_MASK_TP1],
              contextPrivate->uavResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_DYNAMIC_MASK_TM1]);

    memcpy(contextPrivate->lastFrameViewProjection, contextPrivate->pendingViewProjection, sizeof(contextPrivate->lastFrameViewProjection));
    contextPrivate->previousFrameID = params->frameID;
    contextPrivate->dispatchCount++;
    contextPrivate->firstExecution = false;
    setTraceStage(FfxNfruFiStage::None);
    return FFX_OK;
}

FFX_API FfxErrorCode ffxFrameInterpolationDispatch(FfxFrameInterpolationContext* context, const FfxFrameInterpolationDispatchDescription* params)
{
    return ffxFrameInterpolationDispatchWithTrace(context, params, nullptr);
}

FFX_API FfxErrorCode ffxFrameInterpolationDispatchWithTrace(FfxFrameInterpolationContext*                   context,
                                                            const FfxFrameInterpolationDispatchDescription* params,
                                                            FfxNfruFrameInterpolationTrace*                 trace)
{
    FfxScopedLogSuppression suppression(trace != nullptr);
    if (trace)
    {
        const bool collectTimings = trace->collectTimings;
        *trace                    = {};
        trace->collectTimings     = collectTimings;
        trace->failedStage        = FfxNfruFiStage::Validation;
    }
    if (!context)
    {
        return ffxLogError(FFX_ERROR_INVALID_POINTER, "dispatch requires a context");
    }
    const auto* contextPrivate  = reinterpret_cast<const FfxArmFrameInterpolationContext_Private*>(context);
    const auto  userMsgCallback = contextPrivate ? contextPrivate->contextDescription.fpMessage : nullptr;
    if (!params)
    {
        return trace ? FFX_ERROR_INVALID_POINTER : logErrorWithCallback(FFX_ERROR_INVALID_POINTER, "dispatch requires parameters", userMsgCallback);
    }
    if (!params->commandList)
    {
        return trace ? FFX_ERROR_INVALID_POINTER : logErrorWithCallback(FFX_ERROR_INVALID_POINTER, "dispatch requires a command list", userMsgCallback);
    }

    const FfxErrorCode errorCode = dispatchFrameInterpolation(context, params, trace);
    if (errorCode != FFX_OK && !trace)
    {
        return logErrorWithCallback(errorCode, "dispatch", userMsgCallback, __func__);
    }
    return errorCode;
}

FFX_API FfxErrorCode ffxFrameInterpolationDispatchDownsampleOfColour(FfxFrameInterpolationContext* context,
                                                                     FfxCommandList                commandList,
                                                                     FfxResource                   currentColour)
{
    if (!context)
    {
        return ffxLogError(FFX_ERROR_INVALID_POINTER, "downsample dispatch requires a context");
    }

    FfxArmFrameInterpolationContext_Private* contextPrivate = reinterpret_cast<FfxArmFrameInterpolationContext_Private*>(context);

    // No-op when the resolution plan keeps optical-flow colour at full resolution.
    if (!contextPrivate->resolutionPlan.needsOfColourDownsample)
    {
        return FFX_OK;
    }

    // Register the input colour resource
    FFX_VALIDATE(contextPrivate->contextDescription.backendInterface.fpRegisterResource(
        &contextPrivate->contextDescription.backendInterface,
        &currentColour,
        contextPrivate->effectContextId,
        &contextPrivate->srvResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_CURRENT_INTERPOLATION_SOURCE]));

    // Set up the OF-colour output dimensions; DisplaySize remains the full-resolution input dimensions.
    // Save the original values — the main FI dispatch re-stages these without resetting them.
    const int32_t savedRenderSize[2] = {contextPrivate->constants._RenderSize[0], contextPrivate->constants._RenderSize[1]};

    contextPrivate->constants._RenderSize[0] = contextPrivate->ofInputColour.width;
    contextPrivate->constants._RenderSize[1] = contextPrivate->ofInputColour.height;

    FfxErrorCode errorCode = contextPrivate->contextDescription.backendInterface.fpStageConstantBufferDataFunc(
        &contextPrivate->contextDescription.backendInterface,
        &contextPrivate->constants,
        sizeof(contextPrivate->constants),
        &contextPrivate->constantBuffers[FFX_FRAMEINTERPOLATION_CONSTANTBUFFER_IDENTIFIER]);
    if (errorCode == FFX_OK)
    {
        errorCode = scheduleFragment(
            contextPrivate, &contextPrivate->pipelineDownsampleOfColour, contextPrivate->ofInputColour.width, contextPrivate->ofInputColour.height);
    }
    if (errorCode == FFX_OK)
    {
        errorCode = contextPrivate->contextDescription.backendInterface.fpExecuteGpuJobs(
            &contextPrivate->contextDescription.backendInterface, commandList, contextPrivate->effectContextId);
    }

    // NOTE: Do NOT call fpUnregisterResources here. This function shares the same effectContextId
    // as ffxFrameInterpolationPrepare and ffxFrameInterpolationDispatch. Calling unregister would
    // invalidate the dynamic resource slots (srvViewIndex = -1) set up by Prepare, causing a
    // SIGSEGV when subsequent compute passes try to bind those SRV textures.
    // The main ffxFrameInterpolationDispatch will unregister all dynamic resources at its end.
    // Unregister dynamic resources
    // FFX_VALIDATE(contextPrivate->contextDescription.backendInterface.fpUnregisterResources(
    // &contextPrivate->contextDescription.backendInterface, commandList, contextPrivate->effectContextId));

    // Restore original RenderSize so the main FI dispatch uses correct values
    contextPrivate->constants._RenderSize[0] = savedRenderSize[0];
    contextPrivate->constants._RenderSize[1] = savedRenderSize[1];

    return errorCode;
}

FFX_API FfxErrorCode ffxFrameInterpolationDispatchDownsampleOfColourForFrameGeneration(FfxFrameInterpolationContext* context,
                                                                                       FfxCommandList                commandList,
                                                                                       FfxResource                   currentColour)
{
    FfxScopedLogSuppression suppression;
    return ffxFrameInterpolationDispatchDownsampleOfColour(context, commandList, currentColour);
}

FFX_API FfxErrorCode ffxFrameInterpolationGetRequiresOfColourDownsample(FfxFrameInterpolationContext* context, bool* pRequired)
{
    if (!context)
    {
        return ffxLogError(FFX_ERROR_INVALID_POINTER, "downsample query requires a context");
    }

    if (!pRequired)
    {
        const FfxArmFrameInterpolationContext_Private* contextPrivate  = reinterpret_cast<const FfxArmFrameInterpolationContext_Private*>(context);
        const auto                                     userMsgCallback = contextPrivate ? contextPrivate->contextDescription.fpMessage : nullptr;
        return logErrorWithCallback(FFX_ERROR_INVALID_POINTER, "downsample query requires pRequired", userMsgCallback);
    }
    const FfxArmFrameInterpolationContext_Private* contextPrivate = reinterpret_cast<const FfxArmFrameInterpolationContext_Private*>(context);
    *pRequired                                                    = contextPrivate->resolutionPlan.needsOfColourDownsample;
    return FFX_OK;
}

FFX_API FfxErrorCode ffxFrameInterpolationGetDownsampledOfColourP1(FfxFrameInterpolationContext* context, FfxResource* pResource)
{
    if (!context)
    {
        return ffxLogError(FFX_ERROR_INVALID_POINTER, "downsampled colour query requires a context");
    }

    FfxArmFrameInterpolationContext_Private* contextPrivate = reinterpret_cast<FfxArmFrameInterpolationContext_Private*>(context);
    if (!pResource)
    {
        const auto userMsgCallback = contextPrivate ? contextPrivate->contextDescription.fpMessage : nullptr;
        return logErrorWithCallback(FFX_ERROR_INVALID_POINTER, "downsampled colour query requires pResource", userMsgCallback);
    }
    if (!contextPrivate->resolutionPlan.needsOfColourDownsample)
    {
        *pResource = {};
        return FFX_OK;
    }

    *pResource = contextPrivate->contextDescription.backendInterface.fpGetResource(
        &contextPrivate->contextDescription.backendInterface, contextPrivate->uavResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_OF_COLOUR_P1]);
    return FFX_OK;
}

FFX_API FfxErrorCode ffxFrameInterpolationGetDownsampledOfColourM1(FfxFrameInterpolationContext* context, FfxResource* pResource)
{
    if (!context)
    {
        return ffxLogError(FFX_ERROR_INVALID_POINTER, "downsampled colour query requires a context");
    }
    FfxArmFrameInterpolationContext_Private* contextPrivate = reinterpret_cast<FfxArmFrameInterpolationContext_Private*>(context);
    if (!pResource)
    {
        return logErrorWithCallback(FFX_ERROR_INVALID_POINTER, "downsampled colour query requires pResource", contextPrivate->contextDescription.fpMessage);
    }

    if (!contextPrivate->resolutionPlan.needsOfColourDownsample)
    {
        *pResource = {};
        return FFX_OK;
    }

    *pResource = contextPrivate->contextDescription.backendInterface.fpGetResource(
        &contextPrivate->contextDescription.backendInterface, contextPrivate->uavResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_OF_COLOUR_M1]);
    return FFX_OK;
}

FFX_API FfxVersionNumber ffxFrameInterpolationGetEffectVersion()
{
    return FFX_SDK_MAKE_VERSION(FFX_FRAMEINTERPOLATION_VERSION_MAJOR, FFX_FRAMEINTERPOLATION_VERSION_MINOR, FFX_FRAMEINTERPOLATION_VERSION_PATCH);
}
