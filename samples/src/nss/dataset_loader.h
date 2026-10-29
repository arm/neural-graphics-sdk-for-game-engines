/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "core/buffer.h"
#include "core/device.h"

struct NSSFrameParams
{
	bool  reset{false};
	float jitter_x{0.0f};
	float jitter_y{0.0f};
	float camera_near{10.0f};
	float camera_far{5000.0f};
	float camera_fov_angle_vertical{0.75f};
	float frame_time_delta_ms{16.67f};
	float motion_vector_scale_x{1.0f};
	float motion_vector_scale_y{1.0f};
	float exposure{1.0f};
};

class NSSDatasetLoader
{
  public:
	bool init(vkb::core::DeviceC &device, const std::string &dataset_dir,
	          const std::string &dataset_sequence, float scale_factor, VkFormat color_format, uint32_t swapchain_count);

	bool load_frame(int frame_index, uint32_t swapchain_index, NSSFrameParams &frame_params);

	uint64_t total_frames() const;

	VkBuffer get_color_staging_buffer(uint32_t swapchain_index) const;
	VkBuffer get_depth_staging_buffer(uint32_t swapchain_index) const;
	VkBuffer get_motion_vector_staging_buffer(uint32_t swapchain_index) const;

	VkExtent2D         get_display_size() const;
	VkExtent2D         get_render_size() const;
	const std::string &get_ratio_dir() const;
	bool               uses_inverted_depth() const;
	bool               uses_infinite_far() const;

  private:
	struct Config
	{
		bool  flip_depth{false};
		bool  override_exposure{false};
		float exposure{1.0f};
	};

	struct FrameMetadata
	{
		bool  reset{false};
		float motion_vector_scale_x{0.0f};
		float motion_vector_scale_y{0.0f};
		float frame_time_delta_ms{0.0f};
		float jitter_x{0.0f};
		float jitter_y{0.0f};
		float camera_near{10.0f};
		float camera_far{5000.0f};
		float camera_fov_angle_vertical{0.75f};
		float exposure{1.0f};
	};
	struct StagingBuffers
	{
		std::unique_ptr<vkb::core::BufferC> color;
		std::unique_ptr<vkb::core::BufferC> depth;
		std::unique_ptr<vkb::core::BufferC> motion_vectors;
	};

	bool get_frame_params(int frame_index, NSSFrameParams &frame_params) const;

	VkFormat   color_format_{VK_FORMAT_B10G11R11_UFLOAT_PACK32};
	VkExtent2D display_size_{};
	VkExtent2D render_size_{};
	float      frame_rate_hz_{60.0f};
	bool       depth_inverted_{false};
	bool       depth_infinite_{false};
	Config     config_{};

	std::string ratio_dir_;
	std::string color_dir_;
	std::string depth_dir_;
	std::string motion_dir_;

	std::vector<FrameMetadata>  frames_;
	std::vector<StagingBuffers> staging_buffers_;
};