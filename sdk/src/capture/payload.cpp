/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT
 */

#include "payload.h"

#include "file.h"
#include "log.h"
#include "format.h"

#include <ffx_api/ffx_api_types.h>
#include <tinyexr/tinyexr.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <new>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace arm::capture
{
    constexpr uint64_t kMaxDecodedImageBytes = 1ull << 30;

    template <typename T>
    T Load(const uint8_t* source)
    {
        T value;
        std::memcpy(&value, source, sizeof(value));
        return value;
    }

    float DecodeFloat11(uint32_t value)
    {
        const uint32_t exponent = (value >> 6) & 0x1fu;
        const uint32_t mantissa = value & 0x3fu;
        if (exponent == 0)
            return std::ldexp(static_cast<float>(mantissa), -20);
        // An all-one exponent remains the IEEE infinity/NaN encoding; the
        // source mantissa is widened into the float NaN payload.
        const uint32_t floatBits = ((exponent == 31 ? 255u : exponent + 112u) << 23) | (mantissa << 17);
        float          result    = 0.0f;
        std::memcpy(&result, &floatBits, sizeof(result));
        return result;
    }

    float DecodeFloat10(uint32_t value)
    {
        const uint32_t exponent = (value >> 5) & 0x1fu;
        const uint32_t mantissa = value & 0x1fu;
        if (exponent == 0)
            return std::ldexp(static_cast<float>(mantissa), -19);
        // R11G11B10 has the same all-one exponent convention as IEEE floats.
        const uint32_t floatBits = ((exponent == 31 ? 255u : exponent + 112u) << 23) | (mantissa << 18);
        float          result    = 0.0f;
        std::memcpy(&result, &floatBits, sizeof(result));
        return result;
    }

    float DecodeSrgb(float value)
    {
        return value <= 0.04045f ? value / 12.92f : std::pow((value + 0.055f) / 1.055f, 2.4f);
    }

    float DecodeSNorm8(uint8_t value)
    {
        const int32_t signedValue = value < 128 ? static_cast<int32_t>(value) : static_cast<int32_t>(value) - 256;
        return std::max(-1.0f, static_cast<float>(signedValue) / 127.0f);
    }

    float DecodeSNorm16(uint16_t value)
    {
        const int32_t signedValue = value < 0x8000u ? static_cast<int32_t>(value) : static_cast<int32_t>(value) - 65536;
        return std::max(-1.0f, static_cast<float>(signedValue) / 32767.0f);
    }

    struct DecodedSurface
    {
        template <typename T>
        using ChannelStorage = std::array<std::vector<T>, 4>;

        using FloatChannels = ChannelStorage<float>;
        using HalfChannels  = ChannelStorage<uint16_t>;
        using UIntChannels  = ChannelStorage<uint32_t>;

    public:
        struct Status
        {
            bool        success;
            std::string reason;
        };

        DecodedSurface() = delete;

        explicit DecodedSurface(const PayloadView& raw)
            : DecodedSurface(raw, std::nullopt)
        {
        }

        DecodedSurface(const PayloadView& raw, ScaleAndBias scaleAndBias)
            : DecodedSurface(raw, std::optional<ScaleAndBias>{scaleAndBias})
        {
        }

        DecodedSurface(const PayloadView& raw, bool unpackD24)
            : DecodedSurface(raw, std::nullopt, unpackD24)
        {
        }

        Status ToExr(const std::string& path);

    private:
        DecodedSurface(const PayloadView& raw, std::optional<ScaleAndBias> scaleAndBias, bool unpackD24 = false)
            : channelCount_(0)
            , storage_(FloatChannels{})
            , format_(0)
            , extent_({})
            , pixelCount_(0)
            , status_{false, {}}
        {
            status_.success = scaleAndBias ? Decode(raw, scaleAndBias) : Decode(raw, std::nullopt, unpackD24);
        }

        bool Decode(const PayloadView& raw, const std::optional<ScaleAndBias>& scaleAndBias = std::nullopt, bool unpackD24 = false);
        bool Fail(std::string reason);

        template <typename StorageT, typename SourceT>
        bool DecodeInterleaved(const uint8_t* bytes, uint32_t sourceChannelCount, uint32_t pixelCount, uint32_t outputChannelCount = 0)
        {
            static_assert(std::is_same_v<StorageT, SourceT> ||
                          (std::is_same_v<StorageT, uint32_t> &&
                           (std::is_same_v<SourceT, uint8_t> || std::is_same_v<SourceT, uint16_t> || std::is_same_v<SourceT, uint32_t>)));

            auto& channels = Initialize<StorageT>(outputChannelCount ? outputChannelCount : sourceChannelCount, pixelCount);
            for (size_t i = 0; i < pixelCount; ++i)
            {
                for (uint32_t channel = 0; channel < sourceChannelCount; ++channel)
                {
                    const SourceT value  = Load<SourceT>(bytes + (i * static_cast<size_t>(sourceChannelCount) + channel) * sizeof(SourceT));
                    channels[channel][i] = static_cast<StorageT>(value);
                }
            }
            return true;
        }

        template <typename T>
        ChannelStorage<T>& Initialize(uint32_t channelCount, uint32_t count)
        {
            assert(channelCount <= 4);

            ChannelStorage<T> channels;
            for (uint32_t channel = 0; channel < channelCount; ++channel)
                channels[channel].resize(count);
            storage_      = std::move(channels);
            channelCount_ = channelCount;
            return Get<T>();
        }

        template <typename T>
        ChannelStorage<T>& Get()
        {
            ChannelStorage<T>* channels = std::get_if<ChannelStorage<T>>(&storage_);
            assert(channels);
            return *channels;
        }

        template <typename T>
        const ChannelStorage<T>& Get() const
        {
            const ChannelStorage<T>* channels = std::get_if<ChannelStorage<T>>(&storage_);
            assert(channels);
            return *channels;
        }

        uint32_t                                                channelCount_;
        std::variant<FloatChannels, HalfChannels, UIntChannels> storage_;
        uint32_t                                                format_;
        Extent2D                                                extent_;
        uint32_t                                                pixelCount_;
        Status                                                  status_;
    };

    bool DecodedSurface::Fail(std::string reason)
    {
        status_.success = false;
        status_.reason  = std::move(reason);
        return false;
    }

    void LogPayloadFailure(const char* role, const std::string& path, uint32_t format, const DecodedSurface::Status& status)
    {
        ARM_CAPTURE_LOG_WARNING("CAPTURE",
                                "failed to write payload role=%s path=%s format=%u: %s",
                                role ? role : "?",
                                path.c_str(),
                                format,
                                status.reason.empty() ? "unknown payload failure" : status.reason.c_str());
    }

    bool DecodedSurface::Decode(const PayloadView& view, const std::optional<ScaleAndBias>& scaleAndBias, bool unpackD24)
    {
        storage_        = FloatChannels{};
        channelCount_   = 0;
        format_         = view.format;
        extent_         = {};
        pixelCount_     = 0;
        status_.success = false;
        status_.reason.clear();

        if (view.extent.width == 0 || view.extent.height == 0 || view.extent.width > static_cast<uint32_t>(std::numeric_limits<int>::max()) ||
            view.extent.height > static_cast<uint32_t>(std::numeric_limits<int>::max()))
        {
            return Fail("image extent is invalid or too large for EXR");
        }
        const uint64_t pixels = static_cast<uint64_t>(view.extent.width) * view.extent.height;
        if (pixels > std::numeric_limits<uint32_t>::max())
        {
            return Fail("image extent is invalid or too large for EXR");
        }

        const uint32_t     count                    = static_cast<uint32_t>(pixels);
        constexpr uint64_t kMaxDecodedBytesPerPixel = 4ull * sizeof(float);
        if (static_cast<uint64_t>(count) > kMaxDecodedImageBytes / kMaxDecodedBytesPerPixel)
            return Fail("decoded image is too large for EXR conversion");
        if (!view.data)
        {
            return Fail("image view has no payload data");
        }
        const uint32_t bpp = GetFormatBytesPerPixel(view.format);
        if (bpp == 0)
            return Fail("unknown format cannot be converted to EXR");
        if (view.size < static_cast<uint64_t>(count) * bpp)
        {
            return Fail("image view does not contain its complete packed payload");
        }

        extent_              = view.extent;
        pixelCount_          = count;
        const uint8_t* bytes = static_cast<const uint8_t*>(view.data);
        if (scaleAndBias && view.format != FFX_API_SURFACE_FORMAT_R32G32_FLOAT && view.format != FFX_API_SURFACE_FORMAT_R16G16_FLOAT &&
            view.format != FFX_API_SURFACE_FORMAT_R8G8_UNORM)
        {
            return Fail("motion and optical flow require two floating-point or normalized channels; capture other data raw");
        }

        switch (static_cast<FfxApiSurfaceFormat>(view.format))
        {
        case FFX_API_SURFACE_FORMAT_R32G32B32A32_FLOAT:
            return DecodeInterleaved<float, float>(bytes, 4, count);
        case FFX_API_SURFACE_FORMAT_R32G32B32_FLOAT:
            return DecodeInterleaved<float, float>(bytes, 3, count);
        case FFX_API_SURFACE_FORMAT_R32G32_FLOAT:
            return DecodeInterleaved<float, float>(bytes, 2, count, scaleAndBias ? 3 : 2);
        case FFX_API_SURFACE_FORMAT_R32_FLOAT:
            return DecodeInterleaved<float, float>(bytes, 1, count);
        case FFX_API_SURFACE_FORMAT_R16G16B16A16_FLOAT:
            return DecodeInterleaved<uint16_t, uint16_t>(bytes, 4, count);
        case FFX_API_SURFACE_FORMAT_R16G16_FLOAT:
            return DecodeInterleaved<uint16_t, uint16_t>(bytes, 2, count, scaleAndBias ? 3 : 2);
        case FFX_API_SURFACE_FORMAT_R16_FLOAT:
            return DecodeInterleaved<uint16_t, uint16_t>(bytes, 1, count);
        case FFX_API_SURFACE_FORMAT_R11G11B10_FLOAT:
        {
            auto& channels = Initialize<float>(3, count);
            for (size_t i = 0; i < count; ++i)
            {
                const uint32_t pixel = Load<uint32_t>(bytes + i * sizeof(uint32_t));
                channels[0][i]       = DecodeFloat11(pixel & 0x7ffu);
                channels[1][i]       = DecodeFloat11((pixel >> 11) & 0x7ffu);
                channels[2][i]       = DecodeFloat10((pixel >> 22) & 0x3ffu);
            }
            return true;
        }
        case FFX_API_SURFACE_FORMAT_R9G9B9E5_SHAREDEXP:
        {
            auto& channels = Initialize<float>(3, count);
            for (size_t i = 0; i < count; ++i)
            {
                const uint32_t pixel = Load<uint32_t>(bytes + i * sizeof(uint32_t));
                const float    scale = std::ldexp(1.0f, static_cast<int>((pixel >> 27) & 0x1fu) - 24);
                channels[0][i]       = static_cast<float>(pixel & 0x1ffu) * scale;
                channels[1][i]       = static_cast<float>((pixel >> 9) & 0x1ffu) * scale;
                channels[2][i]       = static_cast<float>((pixel >> 18) & 0x1ffu) * scale;
            }
            return true;
        }
        case FFX_API_SURFACE_FORMAT_R10G10B10A2_UNORM:
        {
            auto& channels = Initialize<float>(4, count);
            for (size_t i = 0; i < count; ++i)
            {
                const uint32_t pixel = Load<uint32_t>(bytes + i * sizeof(uint32_t));
                channels[0][i]       = static_cast<float>(pixel & 0x3ffu) / 1023.0f;
                channels[1][i]       = static_cast<float>((pixel >> 10) & 0x3ffu) / 1023.0f;
                channels[2][i]       = static_cast<float>((pixel >> 20) & 0x3ffu) / 1023.0f;
                channels[3][i]       = static_cast<float>((pixel >> 30) & 0x3u) / 3.0f;
            }
            return true;
        }
        case FFX_API_SURFACE_FORMAT_R8G8B8A8_UNORM:
        case FFX_API_SURFACE_FORMAT_R8G8B8A8_SNORM:
        case FFX_API_SURFACE_FORMAT_R8G8B8A8_SRGB:
        {
            const bool srgb     = view.format == FFX_API_SURFACE_FORMAT_R8G8B8A8_SRGB;
            const bool snorm    = view.format == FFX_API_SURFACE_FORMAT_R8G8B8A8_SNORM;
            auto&      channels = Initialize<float>(4, count);
            for (size_t i = 0; i < count; ++i)
            {
                for (uint32_t channel = 0; channel < 4; ++channel)
                {
                    const float value    = snorm ? DecodeSNorm8(bytes[i * 4 + channel]) : bytes[i * 4 + channel] / 255.0f;
                    channels[channel][i] = srgb && channel < 3 ? DecodeSrgb(value) : value;
                }
            }
            return true;
        }
        case FFX_API_SURFACE_FORMAT_B8G8R8A8_UNORM:
        case FFX_API_SURFACE_FORMAT_B8G8R8A8_SRGB:
        {
            const bool srgb     = view.format == FFX_API_SURFACE_FORMAT_B8G8R8A8_SRGB;
            auto&      channels = Initialize<float>(4, count);
            for (size_t i = 0; i < count; ++i)
            {
                const float b  = bytes[i * 4] / 255.0f;
                const float g  = bytes[i * 4 + 1] / 255.0f;
                const float r  = bytes[i * 4 + 2] / 255.0f;
                channels[0][i] = srgb ? DecodeSrgb(r) : r;
                channels[1][i] = srgb ? DecodeSrgb(g) : g;
                channels[2][i] = srgb ? DecodeSrgb(b) : b;
                channels[3][i] = bytes[i * 4 + 3] / 255.0f;
            }
            return true;
        }
        case FFX_API_SURFACE_FORMAT_R16_UNORM:
        case FFX_API_SURFACE_FORMAT_R16_SNORM:
        {
            auto& channels = Initialize<float>(1, count);
            for (size_t i = 0; i < count; ++i)
            {
                const uint16_t value = Load<uint16_t>(bytes + i * sizeof(uint16_t));
                channels[0][i]       = view.format == FFX_API_SURFACE_FORMAT_R16_SNORM ? DecodeSNorm16(value) : static_cast<float>(value) / 65535.0f;
            }
            return true;
        }
        case FFX_API_SURFACE_FORMAT_R8_UNORM:
        case FFX_API_SURFACE_FORMAT_R8_SNORM:
        {
            auto& channels = Initialize<float>(1, count);
            for (size_t i = 0; i < count; ++i)
                channels[0][i] = view.format == FFX_API_SURFACE_FORMAT_R8_SNORM ? DecodeSNorm8(bytes[i]) : bytes[i] / 255.0f;
            return true;
        }
        case FFX_API_SURFACE_FORMAT_R8G8_UNORM:
        {
            auto& channels = Initialize<float>(scaleAndBias ? 3 : 2, count);
            for (size_t i = 0; i < count; ++i)
            {
                const float x = bytes[i * 2] / 255.0f;
                const float y = bytes[i * 2 + 1] / 255.0f;
                // Native float and half motion is already in its final unit;
                // scale/bias only decodes the packed UNORM representation.
                if (scaleAndBias)
                {
                    channels[0][i] = (x - scaleAndBias->bias) * scaleAndBias->scale;
                    channels[1][i] = (y - scaleAndBias->bias) * scaleAndBias->scale;
                    if (!std::isfinite(channels[0][i]) || !std::isfinite(channels[1][i]))
                        return Fail("motion-vector conversion produced a non-finite value");
                }
                else
                {
                    channels[0][i] = x;
                    channels[1][i] = y;
                }
            }
            return true;
        }
        case FFX_API_SURFACE_FORMAT_R32G32B32A32_UINT:
            return DecodeInterleaved<uint32_t, uint32_t>(bytes, 4, count);
        case FFX_API_SURFACE_FORMAT_R32_UINT:
            if (unpackD24)
            {
                auto& channels = Initialize<float>(1, count);
                for (size_t i = 0; i < count; ++i)
                    channels[0][i] = static_cast<float>(Load<uint32_t>(bytes + i * sizeof(uint32_t)) & 0x00ffffffu) / 16777215.0f;
                return true;
            }
            return DecodeInterleaved<uint32_t, uint32_t>(bytes, 1, count);
        case FFX_API_SURFACE_FORMAT_R16G16_UINT:
            return DecodeInterleaved<uint32_t, uint16_t>(bytes, 2, count);
        case FFX_API_SURFACE_FORMAT_R16_UINT:
            return DecodeInterleaved<uint32_t, uint16_t>(bytes, 1, count);
        case FFX_API_SURFACE_FORMAT_R8G8_UINT:
            return DecodeInterleaved<uint32_t, uint8_t>(bytes, 2, count);
        case FFX_API_SURFACE_FORMAT_R8_UINT:
            return DecodeInterleaved<uint32_t, uint8_t>(bytes, 1, count);
        case FFX_API_SURFACE_FORMAT_R8_SINT:
        case FFX_API_SURFACE_FORMAT_R8G8_SINT:
        case FFX_API_SURFACE_FORMAT_R16G16_SINT:
            return Fail("signed integer EXR is not lossless; capture its raw bytes instead");
        case FFX_API_SURFACE_FORMAT_R32G32B32A32_TYPELESS:
        case FFX_API_SURFACE_FORMAT_R8G8B8A8_TYPELESS:
        case FFX_API_SURFACE_FORMAT_B8G8R8A8_TYPELESS:
        case FFX_API_SURFACE_FORMAT_R16G16B16A16_TYPELESS:
        case FFX_API_SURFACE_FORMAT_R32G32_TYPELESS:
        case FFX_API_SURFACE_FORMAT_R10G10B10A2_TYPELESS:
        case FFX_API_SURFACE_FORMAT_R16G16_TYPELESS:
        case FFX_API_SURFACE_FORMAT_R16_TYPELESS:
        case FFX_API_SURFACE_FORMAT_R8_TYPELESS:
        case FFX_API_SURFACE_FORMAT_R8G8_TYPELESS:
        case FFX_API_SURFACE_FORMAT_R32_TYPELESS:
            return Fail("typeless format has no self-contained numeric interpretation; capture its raw bytes instead");
        case FFX_API_SURFACE_FORMAT_UNKNOWN:
        default:
            return Fail("unknown format cannot be converted to EXR");
        }
    }

    DecodedSurface::Status DecodedSurface::ToExr(const std::string& path)
    {
        if (!status_.success)
            return status_;
        if (channelCount_ == 0 || channelCount_ > 4 || extent_.width == 0 || extent_.height == 0)
        {
            Fail("decoded channel layout is incompatible with EXR output");
            return status_;
        }

        const char*          channelNames[] = {"R", "G", "B", "A"};
        const unsigned char* channels[4]    = {};
        const char*          names[4]       = {};
        int                  types[4]       = {};
        for (uint32_t channel = 0; channel < channelCount_; ++channel)
        {
            const uint32_t sourceChannel = channelCount_ - channel - 1;
            names[channel]               = channelNames[sourceChannel];
        }
        std::visit(
            [&](const auto& source) {
                using T = typename std::decay_t<decltype(source)>::value_type::value_type;
                constexpr int pixelType =
                    std::is_same_v<T, float> ? TINYEXR_PIXELTYPE_FLOAT : std::is_same_v<T, uint16_t> ? TINYEXR_PIXELTYPE_HALF : TINYEXR_PIXELTYPE_UINT;
                static_assert(std::is_same_v<T, float> || std::is_same_v<T, uint16_t> || std::is_same_v<T, uint32_t>);

                for (uint32_t channel = 0; channel < channelCount_; ++channel)
                    channels[channel] = reinterpret_cast<const unsigned char*>(source[channelCount_ - channel - 1].data());
                std::fill(types, types + channelCount_, pixelType);
            },
            storage_);

        EXRHeader header{};
        EXRImage  image{};
        InitEXRHeader(&header);
        InitEXRImage(&image);
        image.num_channels           = static_cast<int>(channelCount_);
        image.images                 = const_cast<unsigned char**>(channels);
        image.width                  = static_cast<int>(extent_.width);
        image.height                 = static_cast<int>(extent_.height);
        header.num_channels          = static_cast<int>(channelCount_);
        header.channels              = static_cast<EXRChannelInfo*>(std::calloc(channelCount_, sizeof(EXRChannelInfo)));
        header.pixel_types           = static_cast<int*>(std::malloc(channelCount_ * sizeof(int)));
        header.requested_pixel_types = static_cast<int*>(std::malloc(channelCount_ * sizeof(int)));
        if (!header.channels || !header.pixel_types || !header.requested_pixel_types)
        {
            std::free(header.channels);
            std::free(header.pixel_types);
            std::free(header.requested_pixel_types);
            Fail("failed to allocate EXR channel metadata");
            return status_;
        }
        for (uint32_t i = 0; i < channelCount_; ++i)
        {
            std::strncpy(header.channels[i].name, names[i], sizeof(header.channels[i].name) - 1);
            header.pixel_types[i]           = types[i];
            header.requested_pixel_types[i] = types[i];
        }
        header.compression_type = TINYEXR_COMPRESSIONTYPE_NONE;
        const char* error       = nullptr;
        const int   result      = SaveEXRImageToFile(&image, &header, path.c_str(), &error);
        std::free(header.channels);
        std::free(header.pixel_types);
        std::free(header.requested_pixel_types);
        if (result == TINYEXR_SUCCESS)
            return status_;
        Fail(std::string("EXR write failed: ") + (error ? error : "?"));
        if (error)
            FreeEXRErrorMessage(error);
        return status_;
    }

    DecodedSurface::Status WriteDepthExr(const PayloadView& view, const std::string& path, bool depthIsD24 = false)
    {
        switch (static_cast<FfxApiSurfaceFormat>(view.format))
        {
        case FFX_API_SURFACE_FORMAT_R32_FLOAT:
        case FFX_API_SURFACE_FORMAT_R16_FLOAT:
        case FFX_API_SURFACE_FORMAT_R16_UNORM:
        case FFX_API_SURFACE_FORMAT_R16_SNORM:
        case FFX_API_SURFACE_FORMAT_R8_UNORM:
        case FFX_API_SURFACE_FORMAT_R8_SNORM:
            break;
        case FFX_API_SURFACE_FORMAT_R32_UINT:
            if (depthIsD24)
                break;
            return {false, "ambiguous R32_UINT depth cannot be converted to EXR without original depth semantics; capture raw data instead"};
        default:
            return {false, "depth requires one supported floating-point or normalized channel; capture raw data instead"};
        }

        DecodedSurface surface(view, depthIsD24);
        return surface.ToExr(path);
    }

    NormalPayload::NormalPayload(std::string name, std::string path)
        : name_(std::move(name))
        , path_(std::move(path))
    {
    }
    const char* NormalPayload::GetName() const
    {
        return name_.c_str();
    }
    bool NormalPayload::Write(const PayloadView& view)
    {
        if (!EnsureParentDirectory(path_))
            return false;
        DecodedSurface               surface(view);
        const DecodedSurface::Status status = surface.ToExr(path_);

        if (status.success)
            ARM_CAPTURE_LOG_INFO("CAPTURE", "%s -> %s", name_.c_str(), path_.c_str());
        else
            LogPayloadFailure(name_.c_str(), path_, view.format, status);
        return status.success;
    }

    DepthPayload::DepthPayload(std::string name, std::string path)
        : name_(std::move(name))
        , path_(std::move(path))
    {
    }
    const char* DepthPayload::GetName() const
    {
        return name_.c_str();
    }
    bool DepthPayload::Write(const PayloadView& view)
    {
        if (!EnsureParentDirectory(path_))
            return false;
        const DecodedSurface::Status status = WriteDepthExr(view, path_);

        if (status.success)
            ARM_CAPTURE_LOG_INFO("CAPTURE", "%s -> %s", name_.c_str(), path_.c_str());
        else
            LogPayloadFailure(name_.c_str(), path_, view.format, status);
        return status.success;
    }

    ScaledPayload::ScaledPayload(std::string name, std::string path, ScaleAndBias scaleAndBias)
        : name_(std::move(name))
        , path_(std::move(path))
        , scaleAndBias_(scaleAndBias)
    {
    }
    const char* ScaledPayload::GetName() const
    {
        return name_.c_str();
    }
    bool ScaledPayload::Write(const PayloadView& view)
    {
        if (!EnsureParentDirectory(path_))
            return false;
        DecodedSurface               surface(view, scaleAndBias_);
        const DecodedSurface::Status status = surface.ToExr(path_);

        if (status.success)
            ARM_CAPTURE_LOG_INFO("CAPTURE", "%s -> %s", name_.c_str(), path_.c_str());
        else
            LogPayloadFailure(name_.c_str(), path_, view.format, status);
        return status.success;
    }

    BinaryPayload::BinaryPayload(std::string name, std::string path)
        : name_(std::move(name))
        , path_(std::move(path))
    {
    }
    const char* BinaryPayload::GetName() const
    {
        return name_.c_str();
    }
    bool BinaryPayload::Write(const PayloadView& view)
    {
        if (!view.data || view.size == 0)
            return false;
        if (!WriteBinaryFile(path_, view.data, view.size))
            return false;
        ARM_CAPTURE_LOG_INFO("CAPTURE", "%s -> %s", name_.c_str(), path_.c_str());
        return true;
    }

    bool ConvertRawImageToExr(const std::string& inputPath,
                              const std::string& outputPath,
                              uint32_t           format,
                              uint32_t           width,
                              uint32_t           height,
                              ImageKind          kind,
                              float              motionUnormBias,
                              float              motionUnormScale,
                              bool               depthIsD24)
    {
        if (width == 0 || height == 0)
            return false;
        if (kind == ImageKind::Motion && (!std::isfinite(motionUnormScale) || motionUnormScale == 0.0f))
            return false;
        uint64_t expectedSize = 0;
        if (!TryGetPackedImageSize(format, width, height, expectedSize))
            return false;

        std::ifstream input(inputPath, std::ios::binary | std::ios::ate);
        if (!input)
            return false;
        const std::streamsize inputSize = input.tellg();
        if (inputSize <= 0 || static_cast<uint64_t>(inputSize) < expectedSize || static_cast<uint64_t>(inputSize) > kMaxRawImageBytes)
            return false;
        input.seekg(0);
        std::vector<uint8_t> bytes(static_cast<size_t>(inputSize));
        if (!input.read(reinterpret_cast<char*>(bytes.data()), inputSize) || !EnsureParentDirectory(outputPath))
            return false;

        const PayloadView view{bytes.data(), static_cast<uint64_t>(bytes.size()), format, {width, height}};

        switch (kind)
        {
        case ImageKind::Color:
        case ImageKind::GroundTruth:
        case ImageKind::Output:
        case ImageKind::Exposure:
        {
            DecodedSurface surface(view);
            const auto     status = surface.ToExr(outputPath);
            if (!status.success)
                LogPayloadFailure(GetImageKindName(kind), outputPath, format, status);
            return status.success;
        }
        case ImageKind::Depth:
        {
            const auto status = WriteDepthExr(view, outputPath, depthIsD24);
            if (!status.success)
                LogPayloadFailure(GetImageKindName(kind), outputPath, format, status);
            return status.success;
        }
        case ImageKind::Motion:
        {
            DecodedSurface surface(view, {motionUnormBias, motionUnormScale});
            const auto     status = surface.ToExr(outputPath);
            if (!status.success)
                LogPayloadFailure(GetImageKindName(kind), outputPath, format, status);
            return status.success;
        }
        case ImageKind::OpticalFlow:
        {
            DecodedSurface surface(view, {0.0f, 1.0f});
            const auto     status = surface.ToExr(outputPath);
            if (!status.success)
                LogPayloadFailure(GetImageKindName(kind), outputPath, format, status);
            return status.success;
        }
        }
        return false;
    }

}  // namespace arm::capture
