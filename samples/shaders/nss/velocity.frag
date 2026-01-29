#version 450
/* Copyright (c) 2019-2026, Arm Limited and Contributors
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

precision highp float;

layout(set = 0, binding = 0) uniform sampler2D base_color_texture;

layout(location = 0) in vec4 in_pos;
layout(location = 1) in vec2 in_uv;
layout(location = 2) in vec3 in_normal;
layout(location = 3) in vec4 in_current_pos;  // Current frame position in clip space
layout(location = 4) in vec4 in_previous_pos; // Previous frame position in clip space

layout(location = 0) out vec4 o_color;
layout(location = 1) out vec2 o_velocity;

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

// Push constants come with a limitation in the size of data.
// The standard requires at least 128 bytes
layout(push_constant, std430) uniform PBRMaterialUniform {
	vec4  base_color_factor;
	float metallic_factor;
	float roughness_factor;
} pbr_material_uniform;

#include "lighting.h"

layout(set = 0, binding = 4) uniform LightsInfo {
	Light directional_lights[48];
	Light point_lights[48];
	Light spot_lights[48];
} lights_info;

// Specialization constants (set at pipeline creation time)
layout(constant_id = 0) const uint  DIRECTIONAL_LIGHT_COUNT = 0U;
layout(constant_id = 1) const uint  POINT_LIGHT_COUNT       = 0U;
layout(constant_id = 2) const uint  SPOT_LIGHT_COUNT        = 0U;
layout(constant_id = 3) const uint  DISPLAY_MODE            = 0U;
layout(constant_id = 4) const float MIPMAP_LOD_BIAS         = -1.0;  // Dynamic LOD bias based on render scale

void main(void)
{
	// ============================================================================
	// Motion Vector Calculation (for TAA history reprojection)
	// ============================================================================
	vec2 current_ndc  = in_current_pos.xy / in_current_pos.w;
	vec2 previous_ndc = in_previous_pos.xy / in_previous_pos.w;
	vec2 velocity     =  previous_ndc - current_ndc;

	// This matches NSS SDK's velocity range in [-1,1]
	o_velocity = velocity * 0.5;

	// ============================================================================
	// Standard PBR Lighting
	// ============================================================================
	vec3 normal = normalize(in_normal);

	vec3 light_contribution = vec3(0.0);

	for (uint i = 0U; i < DIRECTIONAL_LIGHT_COUNT; ++i)
	{
		light_contribution += apply_directional_light(lights_info.directional_lights[i], normal);
	}

	for (uint i = 0U; i < POINT_LIGHT_COUNT; ++i)
	{
		light_contribution += apply_point_light(lights_info.point_lights[i], in_pos.xyz, normal);
	}

	for (uint i = 0U; i < SPOT_LIGHT_COUNT; ++i)
	{
		light_contribution += apply_spot_light(lights_info.spot_lights[i], in_pos.xyz, normal);
	}

	// ============================================================================
	// Texture Sampling with Dynamic LOD Bias
	// ============================================================================
	// When rendering at low resolution
	// use negative LOD bias to select higher quality mipmaps, avoiding over-blurring
	vec4 base_color = texture(base_color_texture, in_uv, MIPMAP_LOD_BIAS);

	vec3 ambient_color = vec3(0.2) * base_color.xyz;

	o_color = vec4(ambient_color + light_contribution * base_color.xyz, base_color.w);
}
