/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <FidelityFX/host/ffx_interface.h>

#include <stdint.h>

namespace arm::capture
{
    constexpr uint64_t kMaxRawImageBytes = 1ull << 30;

    struct FormatInfo
    {
        uint32_t bytesPerPixel = 0;
    };

    // Describes every FfxApiSurfaceFormat with a known packed pixel size. A zero
    // value denotes an unknown or invalid format and must stop capture rather than
    // guessing a byte size.
    FormatInfo GetFormatInfo(uint32_t format);
    uint32_t   GetFormatBytesPerPixel(uint32_t format);
    bool       TryGetPackedImageSize(uint32_t format, uint32_t width, uint32_t height, uint64_t& size);
    uint64_t   GetResourceCopyAlignment(const FfxResource& resource);

    // Adds the exact packed size of a capture resource to total. Unknown formats,
    // incomplete descriptions, and integer overflow reject the whole plan.
    bool AddResourceBytes(const FfxResource& resource, uint64_t& total);

}  // namespace arm::capture
