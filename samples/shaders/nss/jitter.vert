#version 450
/* Copyright (c) 2019,2025-2026, Arm Limited and Contributors
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 the "License";
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

layout(location = 0) in vec3 position;
layout(location = 1) in vec2 texcoord_0;
layout(location = 2) in vec3 normal;

// Global uniforms (standard Vulkan Samples framework)
layout(set = 0, binding = 1) uniform GlobalUniform {
    mat4 model;
    mat4 view_proj;
    vec3 camera_position;
} global_uniform;

// TAA jitter settings (SHARED across all nodes in the scene)
layout(set = 0, binding = 2) uniform JitterSettings {
    vec2  jitter_offset;       // Sub-pixel offset in NDC space
    float jitter_scale;        // Scale multiplier (typically 1.0)
    float _padding;            // Padding for alignment
    mat4  jitter_view_proj;    // Pre-jittered view-projection matrix
    mat4  prev_view_proj;      // Previous frame view-projection (for motion vectors)
    mat4  current_view_proj;   // Current frame view-projection (without jitter)
} jitter_settings;

// Per-node velocity tracking (for motion vectors)
layout(set = 0, binding = 3) uniform VelocitySettings {
    mat4 prev_model;      // Previous frame model matrix
    mat4 current_model;   // Current frame model matrix
} velocity_settings;

layout (location = 0) out vec4 o_pos;
layout (location = 1) out vec2 o_uv;
layout (location = 2) out vec3 o_normal;
layout (location = 3) out vec4 o_current_pos;  // Current frame position in clip space
layout (location = 4) out vec4 o_previous_pos; // Previous frame position in clip space

void main(void)
{
    // ============================================================================
    // Transform vertex to world space
    // ============================================================================
    o_pos    = velocity_settings.current_model * vec4(position, 1.0);
    o_uv     = texcoord_0;
    o_normal = mat3(velocity_settings.current_model) * normal;

    // ============================================================================
    // Apply TAA Jitter (projection matrix method)
    // ============================================================================
    // Jitter is baked into projection matrix for better quality
    gl_Position = jitter_settings.jitter_view_proj * o_pos;

    // ============================================================================
    // Calculate Motion Vectors (for TAA history reprojection)
    // ============================================================================
    // Current frame clip space position (without jitter, for velocity calculation)
    o_current_pos  = jitter_settings.current_view_proj * velocity_settings.current_model * vec4(position, 1.0);

    // Previous frame clip space position
    o_previous_pos = jitter_settings.prev_view_proj * velocity_settings.prev_model * vec4(position, 1.0);

}
