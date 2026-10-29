/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT
 */

#include "config.h"
#include "log.h"

#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

#if defined(__ANDROID__)
#include <sys/system_properties.h>
#endif

namespace arm::capture
{
    namespace
    {
        constexpr char kOverlongConfigValue[] = "__FFX_CAPTURE_OVERLONG__";

        bool IsBoundedConfigValue(const char* value)
        {
            if (!value)
                return true;
            for (size_t length = 0; length <= kMaxCaptureConfigValueLength; ++length)
            {
                if (value[length] == '\0')
                    return true;
            }
            return false;
        }

        bool IsOverlongConfigValue(const char* value)
        {
            return value == kOverlongConfigValue;
        }

        bool ParseUint32(const char* value, uint32_t& output)
        {
            if (!value || !value[0])
                return false;
            for (const char* digit = value; *digit != '\0'; ++digit)
                if (*digit < '0' || *digit > '9')
                    return false;

            errno                      = 0;
            char*               end    = nullptr;
            const unsigned long parsed = std::strtoul(value, &end, 10);
            if (errno != 0 || end == value || *end != '\0' || parsed > std::numeric_limits<uint32_t>::max())
                return false;
            output = static_cast<uint32_t>(parsed);
            return true;
        }

        bool ParseUint64(const char* value, uint64_t& output)
        {
            if (!value || !value[0])
                return false;
            for (const char* digit = value; *digit != '\0'; ++digit)
                if (*digit < '0' || *digit > '9')
                    return false;

            errno                           = 0;
            char*                    end    = nullptr;
            const unsigned long long parsed = std::strtoull(value, &end, 10);
            if (errno != 0 || end == value || *end != '\0' || parsed > std::numeric_limits<uint64_t>::max())
                return false;
            output = static_cast<uint64_t>(parsed);
            return true;
        }

        bool ParseFiniteFloat(const char* value, float& output)
        {
            if (!value || !value[0])
                return false;

            errno              = 0;
            char*       end    = nullptr;
            const float parsed = std::strtof(value, &end);
            if (errno != 0 || end == value || *end != '\0' || !std::isfinite(parsed))
                return false;
            output = parsed;
            return true;
        }

        bool ParseNonZeroFiniteFloat(const char* value, float& output)
        {
            return ParseFiniteFloat(value, output) && output != 0.0f;
        }

        bool EqualsIgnoreCase(const char* value, const char* expected)
        {
            if (!value || !expected)
                return false;
            while (*value && *expected)
            {
                if (std::tolower(static_cast<unsigned char>(*value)) != std::tolower(static_cast<unsigned char>(*expected)))
                    return false;
                ++value;
                ++expected;
            }
            return *value == '\0' && *expected == '\0';
        }

        bool ParseImageOutputEncoding(const char* value, ImageOutputEncoding& output)
        {
            if (!value || !value[0])
            {
                output = ImageOutputEncoding::Exr;
                return true;
            }

            if (!IsBoundedConfigValue(value))
                return false;
            if (EqualsIgnoreCase(value, "exr"))
            {
                output = ImageOutputEncoding::Exr;
                return true;
            }
            if (EqualsIgnoreCase(value, "raw"))
            {
                output = ImageOutputEncoding::Raw;
                return true;
            }
            return false;
        }

        template <typename Value, typename Parser>
        bool ParseOptionalConfigValue(const char* value, const char* name, const char* component, Parser parser, Value& output)
        {
            if (!value || parser(value, output))
                return true;

            ARM_CAPTURE_LOG_ERROR(component, "invalid %s value: %s", name, value);
            return false;
        }

    }  // namespace

    EnvironmentConfigReader::EnvironmentConfigReader(const char* platformPrefix, const char* component)
        : prefix_(platformPrefix ? platformPrefix : "")
        , component_(component ? component : "")
        , enabled_(platformPrefix != nullptr)
    {
    }
    const char* EnvironmentConfigReader::Read(const char* propertySuffix, const char* environmentSuffix) const
    {
        if (!enabled_)
            return nullptr;

#if defined(__ANDROID__)
        // Android capture configuration is intentionally property-only. Android applications do
        // not provide the process environment contract used by desktop capture, so never fall
        // back to getenv() on this platform.
        static thread_local char propertyName[PROP_NAME_MAX];
        static thread_local char propertyValue[PROP_VALUE_MAX];
        if (!prefix_.empty())
        {
            const int propertyLength = std::snprintf(propertyName, sizeof(propertyName), "%s%s", prefix_.c_str(), propertySuffix);
            if (propertyLength > 0 && static_cast<size_t>(propertyLength) < sizeof(propertyName) && __system_property_get(propertyName, propertyValue) > 0)
                return IsBoundedConfigValue(propertyValue) ? propertyValue : kOverlongConfigValue;
        }
        return nullptr;
#else
        (void)propertySuffix;
        char      environmentName[128];
        const int environmentLength = std::snprintf(environmentName, sizeof(environmentName), "%s%s", prefix_.c_str(), environmentSuffix);
        if (environmentLength <= 0 || static_cast<size_t>(environmentLength) >= sizeof(environmentName))
            return nullptr;
        const char* value = std::getenv(environmentName);
        return !value ? nullptr : (IsBoundedConfigValue(value) ? value : kOverlongConfigValue);
#endif
    }

    std::optional<uint64_t> EnvironmentConfigReader::ReadRevision() const
    {
        const char* value = Read(kConfigRevisionSuffix, kConfigRevisionEnvironmentSuffix);
        if (!value)
            return std::nullopt;

        uint64_t revision = 0;
        if (!ParseUint64(value, revision))
        {
            ARM_CAPTURE_LOG_ERROR(component_.c_str(), "invalid CONFIG_REVISION value: %s", value);
            return std::nullopt;
        }
        return revision;
    }

    std::optional<Config> EnvironmentConfigReader::ReadConfig()
    {
        const std::optional<uint64_t> revision = ReadRevision();
        if (!revision)
            return std::nullopt;
        if (lastRevision_ && *revision <= *lastRevision_)
            return std::nullopt;

        const char* directory = Read(kConfigDirSuffix, kConfigDirEnvironmentSuffix);
        if (!directory || !directory[0])
            return std::nullopt;
        if (IsOverlongConfigValue(directory))
        {
            if (!lastRevision_ || *lastRevision_ != *revision)
                ARM_CAPTURE_LOG_ERROR(component_.c_str(), "capture configuration value exceeds the %zu-byte limit", kMaxCaptureConfigValueLength);
            lastRevision_ = *revision;
            return std::nullopt;
        }
        size_t directoryLength = 0;
        while (directoryLength <= kMaxCaptureOutputDirectoryLength && directory[directoryLength] != '\0')
            ++directoryLength;
        if (directoryLength > kMaxCaptureOutputDirectoryLength)
        {
            if (!lastRevision_ || *lastRevision_ != *revision)
                ARM_CAPTURE_LOG_ERROR(component_.c_str(), "capture output directory exceeds the %zu-byte limit", kMaxCaptureOutputDirectoryLength);
            lastRevision_ = *revision;
            return std::nullopt;
        }

        Config config{};
        config.component       = component_;
        config.outputDirectory = directory;
        uint32_t value         = 0;
        if (!ParseUint32(Read(kConfigFrameCountSuffix, kConfigFrameCountEnvironmentSuffix), value))
            return std::nullopt;
        config.maxFrameCount = value;
        if (config.maxFrameCount == 0)
            return std::nullopt;
        const char* startFrameValue = Read(kConfigStartFrameSuffix, kConfigStartFrameEnvironmentSuffix);
        if (startFrameValue && !ParseUint32(startFrameValue, value))
        {
            ARM_CAPTURE_LOG_ERROR(component_.c_str(), "invalid START_FRAME value: %s", startFrameValue);
            return std::nullopt;
        }
        if (startFrameValue)
            config.startFrame = value;
        const char* biasValue = Read(kConfigMotionVectorBiasSuffix, kConfigMotionVectorBiasEnvironmentSuffix);
        if (!ParseOptionalConfigValue(biasValue, "MV_UNORM_BIAS", component_.c_str(), ParseFiniteFloat, config.motionVectorUnormBias))
            return std::nullopt;
        const char* scaleValue = Read(kConfigMotionVectorScaleSuffix, kConfigMotionVectorScaleEnvironmentSuffix);
        if (!ParseOptionalConfigValue(scaleValue, "MV_UNORM_SCALE", component_.c_str(), ParseNonZeroFiniteFloat, config.motionVectorUnormScale))
            return std::nullopt;
        const char* imageOutputValue = Read(kConfigImageOutputSuffix, kConfigImageOutputEnvironmentSuffix);
        if (!ParseOptionalConfigValue(imageOutputValue, "IMAGE_OUTPUT", component_.c_str(), ParseImageOutputEncoding, config.imageOutputEncoding))
            return std::nullopt;
        const std::optional<uint64_t> finalRevision = ReadRevision();
        if (!finalRevision || *finalRevision != *revision)
            return std::nullopt;

        lastRevision_ = *revision;
        return config;
    }

    StagingPoolConfig LoadDefaultStagingPoolConfig()
    {
        StagingPoolConfig config{};
        float             ratio = 0.0f;
#if defined(__ANDROID__)
        // Keep the Android configuration contract property-only; getenv() is not a supported
        // source for capture settings on Android.
        static char propertyValue[PROP_VALUE_MAX];
        const char* ratioValue = __system_property_get(kAndroidPoolHostVisibleRatioProperty, propertyValue) > 0 ? propertyValue : nullptr;
#else
        const char* ratioValue = std::getenv(kDesktopPoolHostVisibleRatioEnvironment);
#endif
        if (!IsBoundedConfigValue(ratioValue))
            ratioValue = nullptr;
        if (ParseFiniteFloat(ratioValue, ratio) && ratio > 0.0f && ratio <= 1.0f)
        {
            config.hostVisibleMemoryRatio = ratio;
        }
        return config;
    }

    bool ParseCapturePerformanceEnabled(const char* value)
    {
        return value && (std::strcmp(value, "1") == 0 || EqualsIgnoreCase(value, "true") || EqualsIgnoreCase(value, "yes") || EqualsIgnoreCase(value, "on"));
    }

    bool IsCapturePerformanceEnabled()
    {
        static const bool enabled = [] {
#if defined(__ANDROID__)
            static char propertyValue[PROP_VALUE_MAX];
            const char* value = __system_property_get(kAndroidPerformanceProperty, propertyValue) > 0 ? propertyValue : nullptr;
#else
            const char* value = std::getenv(kDesktopPerformanceEnvironment);
#endif
            return ParseCapturePerformanceEnabled(value);
        }();
        return enabled;
    }

}  // namespace arm::capture
