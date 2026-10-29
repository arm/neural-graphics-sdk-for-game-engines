/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT
 */

#include "format.h"

#include <ffx_api/ffx_api_types.h>

#include <algorithm>
#include <limits>

namespace arm::capture
{
    FormatInfo GetFormatInfo(uint32_t format)
    {
        switch (static_cast<FfxApiSurfaceFormat>(format))
        {
        case FFX_API_SURFACE_FORMAT_R32G32B32A32_TYPELESS:
        case FFX_API_SURFACE_FORMAT_R32G32B32A32_UINT:
        case FFX_API_SURFACE_FORMAT_R32G32B32A32_FLOAT:
            return {16};
        case FFX_API_SURFACE_FORMAT_R32G32B32_FLOAT:
            return {12};
        case FFX_API_SURFACE_FORMAT_R16G16B16A16_FLOAT:
        case FFX_API_SURFACE_FORMAT_R16G16B16A16_TYPELESS:
        case FFX_API_SURFACE_FORMAT_R32G32_FLOAT:
        case FFX_API_SURFACE_FORMAT_R32G32_TYPELESS:
            return {8};
        case FFX_API_SURFACE_FORMAT_R11G11B10_FLOAT:
        case FFX_API_SURFACE_FORMAT_R10G10B10A2_UNORM:
        case FFX_API_SURFACE_FORMAT_R10G10B10A2_TYPELESS:
        case FFX_API_SURFACE_FORMAT_R9G9B9E5_SHAREDEXP:
        case FFX_API_SURFACE_FORMAT_R8G8B8A8_TYPELESS:
        case FFX_API_SURFACE_FORMAT_R8G8B8A8_UNORM:
        case FFX_API_SURFACE_FORMAT_R8G8B8A8_SNORM:
        case FFX_API_SURFACE_FORMAT_R8G8B8A8_SRGB:
        case FFX_API_SURFACE_FORMAT_B8G8R8A8_TYPELESS:
        case FFX_API_SURFACE_FORMAT_B8G8R8A8_UNORM:
        case FFX_API_SURFACE_FORMAT_B8G8R8A8_SRGB:
        case FFX_API_SURFACE_FORMAT_R32_UINT:
        case FFX_API_SURFACE_FORMAT_R32_FLOAT:
        case FFX_API_SURFACE_FORMAT_R32_TYPELESS:
        case FFX_API_SURFACE_FORMAT_R16G16_FLOAT:
        case FFX_API_SURFACE_FORMAT_R16G16_UINT:
        case FFX_API_SURFACE_FORMAT_R16G16_SINT:
        case FFX_API_SURFACE_FORMAT_R16G16_TYPELESS:
            return {4};
        case FFX_API_SURFACE_FORMAT_R16_FLOAT:
        case FFX_API_SURFACE_FORMAT_R16_UINT:
        case FFX_API_SURFACE_FORMAT_R16_UNORM:
        case FFX_API_SURFACE_FORMAT_R16_SNORM:
        case FFX_API_SURFACE_FORMAT_R16_TYPELESS:
        case FFX_API_SURFACE_FORMAT_R8G8_UNORM:
        case FFX_API_SURFACE_FORMAT_R8G8_UINT:
        case FFX_API_SURFACE_FORMAT_R8G8_SINT:
        case FFX_API_SURFACE_FORMAT_R8G8_TYPELESS:
            return {2};
        case FFX_API_SURFACE_FORMAT_R8_UINT:
        case FFX_API_SURFACE_FORMAT_R8_SINT:
        case FFX_API_SURFACE_FORMAT_R8_UNORM:
        case FFX_API_SURFACE_FORMAT_R8_SNORM:
        case FFX_API_SURFACE_FORMAT_R8_TYPELESS:
            return {1};
        case FFX_API_SURFACE_FORMAT_UNKNOWN:
        default:
            return {};
        }
    }

    uint32_t GetFormatBytesPerPixel(uint32_t format)
    {
        return GetFormatInfo(format).bytesPerPixel;
    }

    bool TryGetPackedImageSize(uint32_t format, uint32_t width, uint32_t height, uint64_t& size)
    {
        const uint32_t bytesPerPixel = GetFormatBytesPerPixel(format);
        if (bytesPerPixel == 0 || width == 0 || height == 0 || width > kMaxRawImageBytes / bytesPerPixel)
            return false;

        const uint64_t rowSize = static_cast<uint64_t>(width) * bytesPerPixel;
        if (height > kMaxRawImageBytes / rowSize)
            return false;

        size = rowSize * height;
        return true;
    }

    uint64_t GetResourceCopyAlignment(const FfxResource& resource)
    {
        if (resource.description.type == FFX_RESOURCE_TYPE_BUFFER)
            return 4;

        const uint32_t bytesPerPixel = GetFormatBytesPerPixel(static_cast<uint32_t>(resource.description.format));
        return bytesPerPixel == 0 ? 0 : std::max<uint64_t>(4, bytesPerPixel);
    }

    bool AddResourceBytes(const FfxResource& resource, uint64_t& total)
    {
        if (!resource.resource)
            return false;

        uint64_t bytes = resource.description.size;
        if (resource.description.type != FFX_RESOURCE_TYPE_BUFFER)
        {
            const uint32_t bytesPerPixel = GetFormatBytesPerPixel(static_cast<uint32_t>(resource.description.format));
            if (bytesPerPixel == 0 || resource.description.width == 0 || resource.description.height == 0)
            {
                return false;
            }
            const uint32_t channels = resource.description.type == FFX_RESOURCE_TYPE_TENSOR ? std::max(1u, resource.description.channel) : 1u;
            if (resource.description.width > std::numeric_limits<uint64_t>::max() / resource.description.height ||
                static_cast<uint64_t>(resource.description.width) * resource.description.height > std::numeric_limits<uint64_t>::max() / channels ||
                static_cast<uint64_t>(resource.description.width) * resource.description.height * channels >
                    std::numeric_limits<uint64_t>::max() / bytesPerPixel)
            {
                return false;
            }
            bytes = static_cast<uint64_t>(resource.description.width) * resource.description.height * channels * bytesPerPixel;
        }

        if (bytes == 0 || bytes > std::numeric_limits<uint64_t>::max() - total)
            return false;
        total += bytes;
        return true;
    }

}  // namespace arm::capture
