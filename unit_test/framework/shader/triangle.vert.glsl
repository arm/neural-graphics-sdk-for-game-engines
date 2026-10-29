/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT
 */

#version 450

vec2 positions[3] = vec2[](
    vec2(0.0, -0.7),
    vec2(0.7, 0.7),
    vec2(-0.7, 0.7)
);

void main()
{
    gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
}
