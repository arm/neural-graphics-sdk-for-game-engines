/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

#ifndef FFX_CAPTURE_ENABLED
#define FFX_CAPTURE_ENABLED 1
#endif

namespace arm::capture
{
    inline constexpr std::size_t kMaxCaptureConfigValueLength     = 4096;
    inline constexpr std::size_t kMaxCaptureOutputDirectoryLength = 512;
    inline constexpr const char* kAndroidNssPropertyPrefix        = "debug.ngsdk.nss.";
    inline constexpr const char* kDesktopNssEnvironmentPrefix     = "NSS_CAPTURE_";
    inline constexpr const char* kAndroidNfruPropertyPrefix       = "debug.ngsdk.nfru.";
    inline constexpr const char* kDesktopNfruEnvironmentPrefix    = "NFRU_CAPTURE_";

    inline constexpr const char* kAndroidPoolHostVisibleRatioProperty      = "debug.ngsdk.pool_ratio";
    inline constexpr const char* kDesktopPoolHostVisibleRatioEnvironment   = "NGSDK_CAPTURE_POOL_HOST_VISIBLE_RATIO";
    inline constexpr const char* kAndroidPerformanceProperty               = "debug.ngsdk.capture.perf";
    inline constexpr const char* kDesktopPerformanceEnvironment            = "NGSDK_CAPTURE_PERFORMANCE";
    inline constexpr const char* kConfigDirSuffix                          = "dir";
    inline constexpr const char* kConfigFrameCountSuffix                   = "frames";
    inline constexpr const char* kConfigRevisionSuffix                     = "revision";
    inline constexpr const char* kConfigStartFrameSuffix                   = "start";
    inline constexpr const char* kConfigMotionVectorScaleSuffix            = "mv_unorm_scale";
    inline constexpr const char* kConfigMotionVectorBiasSuffix             = "mv_unorm_bias";
    inline constexpr const char* kConfigImageOutputSuffix                  = "image_output";
    inline constexpr const char* kConfigDirEnvironmentSuffix               = "DIR";
    inline constexpr const char* kConfigFrameCountEnvironmentSuffix        = "FRAME_COUNT";
    inline constexpr const char* kConfigRevisionEnvironmentSuffix          = "CONFIG_REVISION";
    inline constexpr const char* kConfigStartFrameEnvironmentSuffix        = "START_FRAME";
    inline constexpr const char* kConfigMotionVectorScaleEnvironmentSuffix = "MV_UNORM_SCALE";
    inline constexpr const char* kConfigMotionVectorBiasEnvironmentSuffix  = "MV_UNORM_BIAS";
    inline constexpr const char* kConfigImageOutputEnvironmentSuffix       = "IMAGE_OUTPUT";

    inline constexpr const char* kConfigSuffixes[] = {
        kConfigDirSuffix,
        kConfigFrameCountSuffix,
        kConfigRevisionSuffix,
        kConfigStartFrameSuffix,
        kConfigMotionVectorScaleSuffix,
        kConfigMotionVectorBiasSuffix,
        kConfigImageOutputSuffix,
    };
    inline constexpr const char* kConfigEnvironmentSuffixes[] = {
        kConfigDirEnvironmentSuffix,
        kConfigFrameCountEnvironmentSuffix,
        kConfigRevisionEnvironmentSuffix,
        kConfigStartFrameEnvironmentSuffix,
        kConfigMotionVectorScaleEnvironmentSuffix,
        kConfigMotionVectorBiasEnvironmentSuffix,
        kConfigImageOutputEnvironmentSuffix,
    };

#if defined(__ANDROID__)
#define ARM_CAPTURE_PLATFORM_PREFIX(androidPropertyPrefix, environmentPrefix) androidPropertyPrefix
#else
#define ARM_CAPTURE_PLATFORM_PREFIX(androidPropertyPrefix, environmentPrefix) environmentPrefix
#endif

    inline constexpr const char* kNssPlatformPrefix  = ARM_CAPTURE_PLATFORM_PREFIX(kAndroidNssPropertyPrefix, kDesktopNssEnvironmentPrefix);
    inline constexpr const char* kNfruPlatformPrefix = ARM_CAPTURE_PLATFORM_PREFIX(kAndroidNfruPropertyPrefix, kDesktopNfruEnvironmentPrefix);

    enum class ImageOutputEncoding : uint8_t
    {
        Exr,
        Raw,
    };

    // Per-provider/per-context capture settings. This is intentionally a value
    // object: separate NFRU and NSS contexts must not share mutable configuration.
    struct Config
    {
        std::string         component;
        std::string         outputDirectory;
        uint32_t            startFrame             = 0;
        uint32_t            maxFrameCount          = 0;
        float               motionVectorUnormBias  = 0.5f;
        float               motionVectorUnormScale = 2.0f;
        ImageOutputEncoding imageOutputEncoding    = ImageOutputEncoding::Exr;
    };

    // Reads legacy property/environment capture values and de-duplicates
    // revisions, including absent or invalid configurations.
    class EnvironmentConfigReader
    {
    public:
        EnvironmentConfigReader(const char* platformPrefix, const char* component);

        // Update this value only after all other capture settings are written.
        // Returns a snapshot only when its revision is new and unchanged after parsing.
        std::optional<Config> ReadConfig();

    private:
        std::optional<uint64_t> ReadRevision() const;
        const char*             Read(const char* propertySuffix, const char* environmentSuffix) const;

        std::string             prefix_;
        std::string             component_;
        std::optional<uint64_t> lastRevision_;
        bool                    enabled_ = false;
    };

    // Immutable policy used when creating one shared StagingPool for a Vulkan
    // device. It is deliberately separate from Config because it is shared by
    // NFRU and NSS rather than owned by either provider context.
    struct StagingPoolConfig
    {
        float hostVisibleMemoryRatio = 0.125f;
    };

    StagingPoolConfig LoadDefaultStagingPoolConfig();

    bool ParseCapturePerformanceEnabled(const char* value);
    bool IsCapturePerformanceEnabled();

}  // namespace arm::capture
