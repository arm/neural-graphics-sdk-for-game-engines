/*
 * SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT
 */

#ifndef FFX_OPTICALFLOW_CALLBACKS_GLSL_H
#define FFX_OPTICALFLOW_CALLBACKS_GLSL_H

#if defined(FFX_GPU)
#include "ffx_core.h"

layout(set = 0, binding = 1000) uniform sampler s_PointClamp;
#endif  // #if defined(FFX_GPU)

#endif  // FFX_OPTICALFLOW_CALLBACKS_GLSL_H
