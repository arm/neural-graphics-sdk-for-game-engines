/* Copyright (c) 2026, Arm Limited and Contributors
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

#pragma once

#include "buffer_pool.h"
#include "common/glm_common.h"
#include "rendering/subpasses/forward_subpass.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <unordered_map>
#include <vector>

namespace vkb
{
namespace sg
{
class Node;
}

/**
 * @brief Forward subpass that renders the scene with a velocity/motion-vector attachment
 *        and optional sub-pixel jitter.  Used by NSS and NFRU samples.
 */
class TemporalForwardSubpass : public ForwardSubpass
{
  public:
	/**
	 * @brief Jitter settings stored in a uniform buffer
	 */
	struct alignas(16) JitterSettings
	{
		glm::mat4 jitter_view_proj;
		glm::mat4 prev_view_proj;
		glm::mat4 current_view_proj;
	};

	/**
	 * @brief Per‑node velocity settings stored in a uniform buffer
	 */
	struct alignas(16) VelocitySettings
	{
		glm::mat4 prev_model;
		glm::mat4 current_model;
	};

	/**
	 * @brief Constructs the subpass used for forward rendering with jitter
	 * @param render_context Render context
	 * @param vertex_shader Vertex shader source
	 * @param fragment_shader Fragment shader source
	 * @param scene Scene to render on this subpass
	 * @param camera Camera used to look at the scene
	 */
	TemporalForwardSubpass(RenderContext &render_context,
	                 ShaderSource &&vertex_shader,
	                 ShaderSource &&fragment_shader,
	                 sg::Scene &scene,
	                 sg::Camera &camera);

	void prepare() override;
	void draw(vkb::core::CommandBufferC &command_buffer) override;
	void update_uniform(vkb::core::CommandBufferC &command_buffer, vkb::sg::Node &node, size_t thread_index) override;

	// ---- Jitter configuration / accessors ----
	void  set_jitter_scale(float scale);
	float get_jitter_scale() const;
	glm::vec2 get_current_jitter_pixels() const;
	glm::vec2 get_current_jitter_pixels_scaled() const;

	// Override the subpass jitter with a caller-supplied pixel offset.
	// set_external_jitter() pins the jitter; clear_external_jitter() restores
	// the built-in Halton sequence. They are a matched pair.
	void set_external_jitter(const glm::vec2 &jitter_pixels);
	void clear_external_jitter();
	uint32_t peek_frame_index() const;

	// ---- View-projection access ----
	/// Returns the current unjittered view-projection matrix for the bound camera.
	/// This is computed on demand so callers never depend on stale draw-time state.
	glm::mat4 get_view_projection() const;

	// ---- Velocity stride (for NFRU stride-2) ----
	/// When set to 2, the velocity shader receives matrices from 2 frames ago
	/// as "prev", producing motion vectors that span two rendered frames.
	/// Default is 1 (standard single-frame velocity).
	void set_velocity_stride(uint32_t stride);

	// ---- Shader configuration ----
	void set_mipmap_lod_bias(float bias);


  private:
	struct NodeMotionData
	{
		glm::mat4 current_transform{1.0f};
		glm::mat4 previous_transform{1.0f};
		glm::mat4 prev_prev_transform{1.0f};
		uint32_t  last_update_frame{0};
		bool initialized{false};
	};

	static float create_halton_sequence(int32_t index, int32_t base);
	void         prepare_halton_sequence();
	void         update_jitter_settings_buffer();

	std::optional<glm::vec2> external_jitter_pixels{};
	glm::vec2               current_jitter_pixels{0.0f, 0.0f};
	float                   jitter_scale{1.0f};

	static constexpr uint32_t MAX_TAA_SAMPLES = 16;
	std::vector<glm::vec2>  halton_sequence{};

	bool                    first_frame{true};
	uint32_t                frame_index{0};

	glm::mat4               prev_view_proj_matrix{1.0f};
	glm::mat4               prev_prev_view_proj_matrix{1.0f};
	uint32_t                velocity_stride{1};

	vkb::BufferAllocationC   jitter_settings_allocation{};

	std::unordered_map<vkb::sg::Node *, NodeMotionData> node_motion_data{};

	float mipmap_lod_bias{0.0f};
};

}        // namespace vkb
