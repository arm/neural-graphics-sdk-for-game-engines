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

#include "ffx_api.hpp"
#include "ffx_nss.h"

// Helper types for header initialization. Api definition is in .h file.

namespace ffx
{
    template <>
    struct struct_type<ffxApiCreateContextDescNss> : std::integral_constant<uint64_t, FFX_API_CREATE_CONTEXT_DESC_TYPE_NSS>
    {
    };

    struct CreateContextDescNss : public InitHelper<ffxApiCreateContextDescNss>
    {
    };

    template <>
    struct struct_type<ffxApiDispatchDescNss> : std::integral_constant<uint64_t, FFX_API_DISPATCH_DESC_TYPE_NSS>
    {
    };

    struct DispatchDescNss : public InitHelper<ffxApiDispatchDescNss>
    {
    };

    template <>
    struct struct_type<ffxApiQueryDescNssGetJitterPhaseCount> : std::integral_constant<uint64_t, FFX_API_QUERY_DESC_TYPE_NSS_GETJITTERPHASECOUNT>
    {
    };

    struct QueryDescNssGetJitterPhaseCount : public InitHelper<ffxApiQueryDescNssGetJitterPhaseCount>
    {
    };

    template <>
    struct struct_type<ffxApiQueryDescNssGetJitterOffset> : std::integral_constant<uint64_t, FFX_API_QUERY_DESC_TYPE_NSS_GETJITTEROFFSET>
    {
    };

    struct QueryDescNssGetJitterOffset : public InitHelper<ffxApiQueryDescNssGetJitterOffset>
    {
    };

}  // namespace ffx
