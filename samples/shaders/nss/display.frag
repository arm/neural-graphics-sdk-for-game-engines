#version 450
/* Copyright (c) 2020-2026, Arm Limited and Contributors
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

// ============================================================================
// Input/Output
// ============================================================================
layout(location = 0) in  vec2 in_uv;
layout(location = 0) out vec4 o_color;

// ============================================================================
// Uniforms & Samplers
// ============================================================================
layout(set = 0, binding = 0) uniform PostprocessingUniform {
	vec2 near_far;  // x = far plane, y = near plane (reversed depth)
} postprocessing_uniform;

// Render target samplers
layout(set = 0, binding = 1) uniform sampler2D depth_sampler;        // Low-res depth buffer
layout(set = 0, binding = 2) uniform sampler2D color_sampler;        // Final output (NSS upscaled or low-res)
layout(set = 0, binding = 3) uniform sampler2D velocity_sampler;     // Low-res motion vectors
layout(set = 0, binding = 4) uniform sampler2D lowres_color_sampler; // Low-res scene color (before NSS)

// Display mode selection (push constant)
layout(push_constant) uniform DisplayMode {
    int mode;  // 0=final, 1=velocity, 2=depth, 3=nss_debug, 4=lowres
} display_mode;

// ============================================================================
// Utility Functions
// ============================================================================
float linearizeDepth(float depth, float near, float far) {
    return near * far / (far + depth * (near - far));
}


// ============================================================================
// Main Display Function
// ============================================================================
void main(void) {
	// ------------------------------------------------------------------------
	// Mode 1: Motion Vector Visualization
	// ------------------------------------------------------------------------
	// Visualizes motion vectors as colored direction + brightness for magnitude
	// R channel = horizontal motion, G channel = vertical motion
	if (display_mode.mode == 1) {
		vec2  velocity         = texture(velocity_sampler, in_uv).rg;
		float motion_magnitude = length(velocity);

		// Normalize direction and map [-1,1] to [0,1] color range
		vec2 normalized_dir = motion_magnitude > 0.0 ? (velocity / motion_magnitude) : vec2(0.0);
		vec3 direction_color = vec3(
			normalized_dir.x * 0.5 + 0.5,  // X: left=dark, right=bright
			normalized_dir.y * 0.5 + 0.5,  // Y: down=dark, up=bright
			0.0
		);

		// Use magnitude as brightness multiplier (amplify for visibility)
		float visible_magnitude = clamp(motion_magnitude * 50.0, 0.0, 1.0);
		o_color = vec4(direction_color * visible_magnitude, 1.0);
	}
	// ------------------------------------------------------------------------
	// Mode 2: Depth Visualization
	// ------------------------------------------------------------------------
	else if (display_mode.mode == 2) {
		float depth = texture(depth_sampler, in_uv).r;
		depth = linearizeDepth(depth, postprocessing_uniform.near_far.x, postprocessing_uniform.near_far.y);
		depth = (postprocessing_uniform.near_far.x - depth) / (postprocessing_uniform.near_far.x - postprocessing_uniform.near_far.y);
		o_color = vec4(vec3(depth), 1.0);
	}
	// ------------------------------------------------------------------------
	// Mode 3: NSS Debug View
	// ------------------------------------------------------------------------
	// When this mode is active, NSS SDK writes debug visualization directly to color_sampler
	// (controlled by FFX_NSS_DISPATCH_FLAG_DRAW_DEBUG_VIEW flag in C++ code)
	else if (display_mode.mode == 3) {
		o_color = texture(color_sampler, in_uv);
	}
	// ------------------------------------------------------------------------
	// Mode 4: Low-Res Color
	// ------------------------------------------------------------------------
	else if (display_mode.mode == 4) {
		o_color = texture(lowres_color_sampler, in_uv);
	}
	// ------------------------------------------------------------------------
	// Mode 0 (Default): Final Output (NSS upscaled result)
	// ------------------------------------------------------------------------
	else {
		o_color = texture(color_sampler, in_uv);
	}
}
