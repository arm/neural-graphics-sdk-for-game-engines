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

#include "ffx_provider.h"

#include "ffx_provider_external.h"

#if defined(FFX_NSS)
#include "ffx_provider_nss.h"
#endif

#include <array>
#include <optional>

static constexpr ffxProvider* providers[] = {
#if defined(FFX_NSS)
    &ffxProvider_Nss::Instance,
#endif
};
static constexpr size_t providerCount = sizeof(providers) / sizeof(ffxProvider*);

static std::array<std::optional<ffxProviderExternal>, 10> externalProviders = {};

const ffxProvider* GetffxProvider(ffxStructType_t descType, uint64_t overrideId, void* device)
{
    for (size_t i = 0; i < providerCount; ++i)
    {
        if (providers[i]->GetId() == overrideId || (overrideId == 0 && providers[i]->CanProvide(descType)))
            return providers[i];
    }

    return nullptr;
}

const ffxProvider* GetAssociatedProvider(ffxContext* context)
{
    const InternalContextHeader* hdr      = (const InternalContextHeader*)(*context);
    const ffxProvider*           provider = hdr->provider;
    return provider;
}

uint64_t GetProviderCount(ffxStructType_t descType, void* device)
{
    return GetProviderVersions(descType, device, UINT64_MAX, nullptr, nullptr);
}

uint64_t GetProviderVersions(ffxStructType_t descType, void* device, uint64_t capacity, uint64_t* versionIds, const char** versionNames)
{
    uint64_t count = 0;

    for (size_t i = 0; i < providerCount; ++i)
    {
        if (count >= capacity)
            break;
        if (providers[i]->CanProvide(descType))
        {
            auto index = count;
            count++;
            if (versionIds)
                versionIds[index] = providers[i]->GetId();
            if (versionNames)
                versionNames[index] = providers[i]->GetVersionName();
        }
    }

    return count;
}
