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

#include "ffx_provider_nss.h"
#include "backends.h"
#include "validation.h"
#include <ffx_api/ffx_nss.hpp>
#ifdef FFX_BACKEND_VK
#include <ffx_api/vk/ffx_api_vk.h>
#endif  // #ifdef FFX_BACKEND_VK
#include <FidelityFX/gpu/nss/ffx_nss_resources.h>
#include <FidelityFX/host/ffx_nss.h>

#include <stdlib.h>

static uint32_t ConvertContextFlagsNss(uint32_t apiFlags)
{
    uint32_t outFlags = 0;
    if (apiFlags & FFX_API_NSS_CONTEXT_FLAG_QUANTIZED)
        outFlags |= FFX_NSS_CONTEXT_FLAG_QUANTIZED;
    if (apiFlags & FFX_API_NSS_CONTEXT_FLAG_HIGH_DYNAMIC_RANGE)
        outFlags |= FFX_NSS_CONTEXT_FLAG_HIGH_DYNAMIC_RANGE;
    if (apiFlags & FFX_API_NSS_CONTEXT_FLAG_DEPTH_INVERTED)
        outFlags |= FFX_NSS_CONTEXT_FLAG_DEPTH_INVERTED;
    if (apiFlags & FFX_API_NSS_CONTEXT_FLAG_DEPTH_INFINITE)
        outFlags |= FFX_NSS_CONTEXT_FLAG_DEPTH_INFINITE;
    if (apiFlags & FFX_API_NSS_CONTEXT_FLAG_RESAMPLE_BICUBIC)
        outFlags |= FFX_NSS_CONTEXT_FLAG_RESAMPLE_BICUBIC;
    if (apiFlags & FFX_API_NSS_CONTEXT_FLAG_READ_TENSORS_AS_IMAGES)
        outFlags |= FFX_NSS_CONTEXT_FLAG_READ_TENSORS_AS_IMAGES;
    if (apiFlags & FFX_API_NSS_CONTEXT_FLAG_ALLOW_16BIT)
        outFlags |= FFX_NSS_CONTEXT_FLAG_ALLOW_16BIT;
    if (apiFlags & FFX_API_NSS_CONTEXT_FLAG_ENABLE_DEBUG_CHECKING)
        outFlags |= FFX_NSS_CONTEXT_FLAG_ENABLE_DEBUG_CHECKING;
    return outFlags;
}

static uint32_t ConvertDispatchFlagsNss(uint32_t apiFlags)
{
    uint32_t outFlags = 0;
    if (apiFlags & FFX_API_NSS_DISPATCH_FLAG_DRAW_DEBUG_VIEW)
        outFlags |= FFX_NSS_DISPATCH_FLAG_DRAW_DEBUG_VIEW;
    return outFlags;
}

bool ffxProvider_Nss::CanProvide(uint64_t type) const
{
    return (type & FFX_API_EFFECT_MASK) == FFX_API_EFFECT_ID_NSS;
}

#define STRINGIFY_(X)                            #X
#define STRINGIFY(X)                             STRINGIFY_(X)
#define MAKE_VERSION_STRING(major, minor, patch) STRINGIFY major "." STRINGIFY minor "." STRINGIFY patch

uint64_t ffxProvider_Nss::GetId() const
{
    // NSS Scale, version from header
    return 0x2555'CA1Eull << 32 | (FFX_SDK_MAKE_VERSION(FFX_NSS_VERSION_MAJOR, FFX_NSS_VERSION_MINOR, FFX_NSS_VERSION_PATCH) & 0xFFFF'FFFF);
}

const char* ffxProvider_Nss::GetVersionName() const
{
    return MAKE_VERSION_STRING(FFX_NSS_VERSION_MAJOR, FFX_NSS_VERSION_MINOR, FFX_NSS_VERSION_PATCH);
}

struct InternalNssContext
{
    InternalContextHeader header;
    FfxInterface          backendInterface;
    FfxResourceInternal   sharedResources[FFX_NSS_RESOURCE_IDENTIFIER_COUNT];
    FfxNssContext         context;
    ffxApiMessage         fpMessage;
};

ffxReturnCode_t ffxProvider_Nss::CreateContext(ffxContext* context, ffxCreateContextDescHeader* header, Allocator& alloc) const
{
    VERIFY(context, FFX_API_RETURN_ERROR_PARAMETER);
    VERIFY(header, FFX_API_RETURN_ERROR_PARAMETER);

    if (auto desc = ffx::DynamicCast<ffxApiCreateContextDescNss>(header))
    {
        if (desc->fpMessage)
        {
#ifdef FFX_BACKEND_VK
            Validator{desc->fpMessage, header}.AcceptExtensions({FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK, FFX_API_DESC_TYPE_OVERRIDE_VERSION});
#endif  // FFX_BACKEND_VK
        }
        InternalNssContext* internal_context = alloc.construct<InternalNssContext>();
        VERIFY(internal_context, FFX_API_RETURN_ERROR_MEMORY);
        internal_context->header.provider = this;

        TRY(MustCreateBackend(header, &internal_context->backendInterface, 1, alloc));

        FfxNssContextDescription initializationParameters = {};
        initializationParameters.backendInterface         = internal_context->backendInterface;
        initializationParameters.maxRenderSize.width      = desc->maxRenderSize.width;
        initializationParameters.maxRenderSize.height     = desc->maxRenderSize.height;
        initializationParameters.maxUpscaleSize.width     = desc->maxUpscaleSize.width;
        initializationParameters.maxUpscaleSize.height    = desc->maxUpscaleSize.height;

        initializationParameters.displaySize.width  = desc->maxUpscaleSize.width;
        initializationParameters.displaySize.height = desc->maxUpscaleSize.height;

        initializationParameters.qualityMode = ConvertEnum<FfxNssShaderQualityMode>(desc->qualityMode);
        initializationParameters.flags       = ConvertContextFlagsNss(desc->flags);
        // Calling this casted function is undefined behaviour, but it's probably safe.
        initializationParameters.fpMessage = reinterpret_cast<FfxNssMessage>(desc->fpMessage);

        // Grab this fp for use in extensions later
        internal_context->fpMessage = desc->fpMessage;

        // Create the NSS context
        TRY2(ffxNssContextCreate(&internal_context->context, &initializationParameters));

        *context = internal_context;
        return FFX_API_RETURN_OK;
    }
    else
    {
        return FFX_API_RETURN_ERROR_UNKNOWN_DESCTYPE;
    }
}

ffxReturnCode_t ffxProvider_Nss::DestroyContext(ffxContext* context, Allocator& alloc) const
{
    VERIFY(context, FFX_API_RETURN_ERROR_PARAMETER);
    VERIFY(*context, FFX_API_RETURN_ERROR_PARAMETER);

    InternalNssContext* internal_context = reinterpret_cast<InternalNssContext*>(*context);

    for (FfxUInt32 i = 0; i < FFX_NSS_RESOURCE_IDENTIFIER_COUNT; i++)
    {
        TRY2(internal_context->backendInterface.fpDestroyResource(&internal_context->backendInterface, internal_context->sharedResources[i], 0));
    }

    TRY2(ffxNssContextDestroy(&internal_context->context));

    alloc.dealloc(internal_context->backendInterface.scratchBuffer);
    alloc.dealloc(internal_context);

    return FFX_API_RETURN_OK;
}

ffxReturnCode_t ffxProvider_Nss::Configure(ffxContext* context, const ffxConfigureDescHeader* header) const
{
    return FFX_API_RETURN_OK;
}

ffxReturnCode_t ffxProvider_Nss::Query(ffxContext* context, ffxQueryDescHeader* header) const
{
    VERIFY(header, FFX_API_RETURN_ERROR_PARAMETER);

    if (context)
    {
        InternalNssContext* internal_context = reinterpret_cast<InternalNssContext*>(*context);
        if (internal_context->fpMessage)
        {
            Validator{internal_context->fpMessage, header}.NoExtensions();
        }
    }

    switch (header->type)
    {
    case FFX_API_QUERY_DESC_TYPE_NSS_GETJITTEROFFSET:
    {
        auto  desc = reinterpret_cast<ffxApiQueryDescNssGetJitterOffset*>(header);
        float jitterX, jitterY;
        TRY2(ffxNssGetJitterOffset(&jitterX, &jitterY, desc->index, desc->phaseCount));
        if (desc->pOutX != nullptr)
        {
            *desc->pOutX = jitterX;
        }
        if (desc->pOutY != nullptr)
        {
            *desc->pOutY = jitterY;
        }
        break;
    }
    case FFX_API_QUERY_DESC_TYPE_NSS_GETJITTERPHASECOUNT:
    {
        auto          desc             = reinterpret_cast<ffxApiQueryDescNssGetJitterPhaseCount*>(header);
        const int32_t jitterPhaseCount = ffxNssGetJitterPhaseCount(desc->renderWidth, desc->displayWidth);

        if (desc->pOutPhaseCount != nullptr)
        {
            *desc->pOutPhaseCount = jitterPhaseCount;
        }
        break;
    }
    default:
        return FFX_API_RETURN_ERROR_UNKNOWN_DESCTYPE;
    }

    return FFX_API_RETURN_OK;
}

ffxReturnCode_t ffxProvider_Nss::Dispatch(ffxContext* context, const ffxDispatchDescHeader* header) const
{
    VERIFY(context, FFX_API_RETURN_ERROR_PARAMETER);
    VERIFY(*context, FFX_API_RETURN_ERROR_PARAMETER);
    VERIFY(header, FFX_API_RETURN_ERROR_PARAMETER);

    InternalNssContext* internal_context = reinterpret_cast<InternalNssContext*>(*context);
    if (internal_context->fpMessage)
    {
        Validator{internal_context->fpMessage, header}.NoExtensions();
    }

    switch (header->type)
    {
    case FFX_API_DISPATCH_DESC_TYPE_NSS:
    {
        auto desc = reinterpret_cast<const ffxApiDispatchDescNss*>(header);

        FfxNssDispatchDescription dispatchParameters = {};
        dispatchParameters.commandList               = desc->commandList;
        dispatchParameters.color                     = Convert(desc->color);
        dispatchParameters.depth                     = Convert(desc->depth);
        dispatchParameters.depthTm1                  = Convert(desc->depthTm1);
        dispatchParameters.motionVectors             = Convert(desc->motionVectors);
        dispatchParameters.outputTm1                 = Convert(desc->outputTm1);
        dispatchParameters.output                    = Convert(desc->output);
        dispatchParameters.debugViews                = Convert(desc->debugViews);
        dispatchParameters.jitterOffset.x            = desc->jitterOffset.x;
        dispatchParameters.jitterOffset.y            = desc->jitterOffset.y;
        dispatchParameters.cameraFar                 = desc->cameraFar;
        dispatchParameters.cameraNear                = desc->cameraNear;
        dispatchParameters.cameraFovAngleVertical    = desc->cameraFovAngleVertical;
        dispatchParameters.exposure                  = desc->exposure;
        dispatchParameters.motionVectorScale.x       = desc->motionVectorScale.x;
        dispatchParameters.motionVectorScale.y       = desc->motionVectorScale.y;
        dispatchParameters.reset                     = desc->reset;
        dispatchParameters.frameTimeDelta            = desc->frameTimeDelta;
        dispatchParameters.upscaleSize.width         = desc->upscaleSize.width;
        dispatchParameters.upscaleSize.height        = desc->upscaleSize.height;
        dispatchParameters.renderSize.width          = desc->renderSize.width;
        dispatchParameters.renderSize.height         = desc->renderSize.height;
        dispatchParameters.flags                     = ConvertDispatchFlagsNss(desc->flags);

        TRY2(ffxNssContextDispatch(&internal_context->context, &dispatchParameters));
        break;
    }
    default:
        return FFX_API_RETURN_ERROR_UNKNOWN_DESCTYPE;
    }

    return FFX_API_RETURN_OK;
}

ffxProvider_Nss ffxProvider_Nss::Instance;
