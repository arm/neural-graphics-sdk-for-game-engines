/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstdio>
#include <stdint.h>
#include <memory>
#include <string>
#include <vector>

#include <FidelityFX/host/ffx_interface.h>

namespace arm::capture
{
    inline std::string GetCaptureBuildId()
    {
#ifdef ARM_CAPTURE_BUILD_ID
        return ARM_CAPTURE_BUILD_ID;
#else
        return std::to_string(FFX_SDK_VERSION_MAJOR) + "." + std::to_string(FFX_SDK_VERSION_MINOR) + "." + std::to_string(FFX_SDK_VERSION_PATCH);
#endif
    }

    // Per-frame metadata shared by image-based capture datasets.
    struct FrameMeta
    {
        virtual ~FrameMeta() = default;

        uint32_t datasetIndex = 0;

        float jitterX                 = 0.0f;
        float jitterY                 = 0.0f;
        float cameraNear              = 0.0f;
        float cameraFar               = 0.0f;
        float cameraFovAngleVertical  = 0.0f;
        float motionVectorScaleX      = 0.0f;
        float motionVectorScaleY      = 0.0f;
        float frameTimeDelta          = 0.0f;
        float viewSpaceToMetersFactor = 0.0f;
        float viewProjection[16]      = {};
        bool  hasViewProjection       = false;

        uint32_t renderWidth  = 0;
        uint32_t renderHeight = 0;
        uint32_t motionWidth  = 0;
        uint32_t motionHeight = 0;

        uint32_t colorFormat  = 0;
        uint32_t depthFormat  = 0;
        uint32_t motionFormat = 0;
        uint32_t outputFormat = 0;
        // Producer-requested temporal reset for this exact dispatch.
        bool reset = false;
    };

}  // namespace arm::capture
