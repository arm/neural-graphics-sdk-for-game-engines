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

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "core/buffer.h"
#include "core/device.h"

struct NFRUFrameParams
{
	bool  reset{false};
	float view_projection[16]{};
	float jitter_x{0.0f};
	float jitter_y{0.0f};
	float camera_near{0.0f};
	float camera_far{0.0f};
	float camera_fov_angle_vertical{0.0f};
	float frame_time_delta_ms{0.0f};
	float motion_vector_scale_x{0.0f};
	float motion_vector_scale_y{0.0f};
	float view_space_to_meters_factor{1.0f};
};

// Loads sequence metadata and one color/depth/motion staging triplet per
// RenderFrame slot.
class NFRUDatasetLoader
{
  public:
	bool init(vkb::core::DeviceC &device,
	          const std::string &dataset_dir, const std::string &dataset_sequence,
	          uint32_t swapchain_count);

	uint64_t total_frames() const;
	int      interpolated_source_frame(int frame_index) const;

	bool get_frame_params(int frame_index, NFRUFrameParams &frame_params) const;

	bool load_frame(int frame_index, uint32_t swapchain_index, NFRUFrameParams &frame_params);

	VkBuffer get_color_staging_buffer(uint32_t swapchain_index) const;
	VkBuffer get_depth_staging_buffer(uint32_t swapchain_index) const;
	VkBuffer get_motion_vector_staging_buffer(uint32_t swapchain_index) const;

	VkExtent2D get_display_size() const;
	VkExtent2D get_render_size() const;
	bool       uses_inverted_depth() const;
	bool       uses_infinite_far() const;

  private:
	struct Config
	{
		bool  apply_tonemap             = true;
		bool  flip_depth                = false;
		bool  transpose_view_projection = true;
		bool  override_exposure         = false;
		float exposure                  = 1.0f;
	};

	struct FrameMetadata
	{
		bool  reset{false};
		float view_projection[16]{};
		float fov_y               = 0.75f;
		float camera_near         = 10.0f;
		float camera_far          = 10.0f;
		float jitter_x            = 0.0f;
		float jitter_y            = 0.0f;
		float exposure            = 1.0f;
		float frame_time_delta_ms = 0.0f;
		float motion_vector_scale_x = 0.0f;
		float motion_vector_scale_y = 0.0f;
		float view_space_to_meters_factor = 1.0f;
	};
	struct StagingBuffers
	{
		std::unique_ptr<vkb::core::BufferC> color;
		std::unique_ptr<vkb::core::BufferC> depth;
		std::unique_ptr<vkb::core::BufferC> motion_vectors;
	};

	std::vector<FrameMetadata> frames_;
	float                      frame_rate_hz_  = 60.0f;
	bool                       depth_inverted_ = false;
	bool                       depth_infinite_ = false;
	Config                     config_{};

	std::string color_dir_;
	std::string depth_dir_;
	std::string motion_dir_;

	VkExtent2D display_size_{};
	VkExtent2D render_size_{};

	std::vector<StagingBuffers> staging_buffers_;
};
