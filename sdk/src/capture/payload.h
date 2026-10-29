/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "meta.h"

#include <cstdint>
#include <memory>
#include <string>

#if defined(_WIN32)
#define ARM_CAPTURE_CONVERSION_API __declspec(dllexport)
#else
#define ARM_CAPTURE_CONVERSION_API
#endif

namespace arm::capture
{
    namespace detail
    {
        class Worker;
    }

    struct Extent2D
    {
        uint32_t width  = 0;
        uint32_t height = 0;
    };

    struct PayloadView
    {
        const void* data   = nullptr;
        uint64_t    size   = 0;
        uint32_t    format = 0;
        Extent2D    extent{};
    };

    class Payload
    {
    public:
        Payload()          = default;
        virtual ~Payload() = default;

        Payload(const Payload&) = delete;
        Payload& operator=(const Payload&) = delete;

        virtual const char* GetName() const                = 0;
        virtual bool        Write(const PayloadView& view) = 0;

    private:
        friend class detail::Worker;

        uint64_t size_   = 0;
        uint32_t format_ = 0;
        Extent2D extent_{};
    };

    enum class ImageKind : uint8_t
    {
        Color,
        GroundTruth,
        Depth,
        Motion,
        Output,
        OpticalFlow,
        Exposure,
    };

    struct ScaleAndBias
    {
        float bias  = 0.5f;
        float scale = 2.0f;
    };

    class NormalPayload final : public Payload
    {
    public:
        NormalPayload(std::string name, std::string path);
        const char* GetName() const override;
        bool        Write(const PayloadView& view) override;

    private:
        std::string name_;
        std::string path_;
    };

    class DepthPayload final : public Payload
    {
    public:
        DepthPayload(std::string name, std::string path);
        const char* GetName() const override;
        bool        Write(const PayloadView& view) override;

    private:
        std::string name_;
        std::string path_;
    };

    class ScaledPayload final : public Payload
    {
    public:
        ScaledPayload(std::string name, std::string path, ScaleAndBias scaleAndBias);
        const char* GetName() const override;
        bool        Write(const PayloadView& view) override;

    private:
        std::string  name_;
        std::string  path_;
        ScaleAndBias scaleAndBias_;
    };

    class BinaryPayload final : public Payload
    {
    public:
        BinaryPayload(std::string name, std::string path);
        const char* GetName() const override;
        bool        Write(const PayloadView& view) override;

    private:
        std::string name_;
        std::string path_;
    };

    inline const char* GetImageKindName(ImageKind kind)
    {
        switch (kind)
        {
        case ImageKind::Color:
            return "color";
        case ImageKind::GroundTruth:
            return "groundTruth";
        case ImageKind::Depth:
            return "depth";
        case ImageKind::Motion:
            return "motion";
        case ImageKind::Output:
            return "output";
        case ImageKind::OpticalFlow:
            return "opticalFlow";
        case ImageKind::Exposure:
            return "exposure";
        }
        return "";
    }

    inline bool ParseImageKind(const std::string& value, ImageKind& kind)
    {
        if (value == "color")
            kind = ImageKind::Color;
        else if (value == "groundTruth")
            kind = ImageKind::GroundTruth;
        else if (value == "depth")
            kind = ImageKind::Depth;
        else if (value == "motion")
            kind = ImageKind::Motion;
        else if (value == "output")
            kind = ImageKind::Output;
        else if (value == "opticalFlow")
            kind = ImageKind::OpticalFlow;
        else if (value == "exposure")
            kind = ImageKind::Exposure;
        else
            return false;
        return true;
    }

    // Converts one provider-owned raw image into the same EXR representation used
    // by the capture payloads. Component-private artifacts are intentionally out
    // of scope for this conversion API.
    ARM_CAPTURE_CONVERSION_API bool ConvertRawImageToExr(const std::string& inputPath,
                                                         const std::string& outputPath,
                                                         uint32_t           format,
                                                         uint32_t           width,
                                                         uint32_t           height,
                                                         ImageKind          kind,
                                                         float              motionUnormBias  = 0.5f,
                                                         float              motionUnormScale = 2.0f,
                                                         bool               depthIsD24       = false);

}  // namespace arm::capture
