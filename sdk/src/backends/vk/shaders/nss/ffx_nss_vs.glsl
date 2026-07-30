// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT

#version 460

void main()
{
    vec2 uv      = vec2(gl_VertexIndex & 1, gl_VertexIndex >> 1) * 2.0;
    gl_Position  = vec4(uv * 2.0 - 1.0, 0.0, 1.0);
}
