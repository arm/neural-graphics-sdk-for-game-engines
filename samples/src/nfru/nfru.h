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

/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <vulkan/vulkan.h>

#include "common/ffx_memory_usage.h"
#include "core/buffer.h"
#include "vulkan_sample.h"

// FidelityFX NFRU SDK headers
#ifndef FFX_CPU
#	define FFX_CPU
#endif
#include <ffx_api.hpp>
#include <ffx_framegeneration.hpp>
#include <vk/ffx_api_vk.hpp>

#include "dataset_loader.h"

namespace vkb
{
class TemporalForwardSubpass;
class RenderPipeline;
class PostProcessingPipeline;

namespace core
{
class Image;
class ImageView;
}        // namespace core

namespace sg
{
class PerspectiveCamera;
class Node;
}        // namespace sg
}        // namespace vkb

/**
 * @brief NFRU sample supporting both realtime scene rendering and
 *        dataset (EXR) input for NFRU frame interpolation.
 */
class NFRUSample : public vkb::VulkanSampleC
{
  public:
	NFRUSample();
	~NFRUSample() override;

	bool prepare(const vkb::ApplicationOptions &options) override;
	void update(float delta_time) override;
	void draw(vkb::core::CommandBufferC &command_buffer, vkb::RenderTarget &render_target) override;
	void draw_gui() override;

  private:
	enum class Mode
	{
		Realtime,
		Dataset
	};

	void request_gpu_features(vkb::core::PhysicalDeviceC &gpu) override;

	Mode mode = Mode::Realtime;
	// Number of scene frames rendered or loaded; reference frames advance this index.
	uint64_t scene_frame_index = 0;
	// Decisions for the current application frame, updated before GUI and draw.
	bool frame_is_generated{true};
	bool frame_should_save{false};

	// Frame inputs
	enum class Attachments
	{
		Depth    = 0,
		Color    = 1,
		Velocity = 2
	};
	struct ImageResource
	{
		std::unique_ptr<vkb::core::Image>     image;
		std::unique_ptr<vkb::core::ImageView> view;
		bool                                  initialized{false};
	};

	struct FrameInputs
	{
		ImageResource                      depth;
		ImageResource                      color;
		ImageResource                      motion_vectors;
		std::unique_ptr<vkb::RenderTarget> render_target;
	};

	// Indexed by nfru_frame_index; size is max(2, swapchain image count) to provide
	// distinct current and previous input slots.
	std::vector<FrameInputs> frame_inputs;
	// Keep Realtime reference renders out of the frame input ring.
	std::unique_ptr<FrameInputs> reference_frame_inputs;
	size_t                       resource_count{0};
	void                         create_frame_inputs(VkExtent2D color_extent, VkExtent2D depth_mv_extent, bool create_reference_frame);
	/// Transition scene inputs from their previous consumer layouts to attachment-write layouts.
	void prepare_render_inputs(vkb::core::CommandBufferC &command_buffer, FrameInputs &inputs);
	/// Publish attachment writes for NFRU/display or ground-truth readback.
	void finish_render_inputs(vkb::core::CommandBufferC &command_buffer, FrameInputs &inputs);
	/// Wait for previous reads, then transition all dataset inputs to transfer-destination layout.
	void prepare_upload_inputs(vkb::core::CommandBufferC &command_buffer, FrameInputs &inputs);
	/// Make transfer writes visible, then transition all dataset inputs to shader-read layout.
	void finish_upload_inputs(vkb::core::CommandBufferC &command_buffer, FrameInputs &inputs);

	// NFRU output is not history, so one image is sufficient.
	ImageResource nfru_output;
	void          create_nfru_output(VkExtent2D extent);
	/// Wait for previous consumers, then transition the output to the NFRU write layout.
	void prepare_output(vkb::core::CommandBufferC &command_buffer);
	/// Publish NFRU writes for display or generated-output readback.
	void finish_output(vkb::core::CommandBufferC &command_buffer);

	// NFRU context and dispatch
	ffx::Context nfru_context = nullptr;
	// Logical NFRU frame number passed to the SDK and used for current/previous selection.
	uint64_t                   nfru_frame_index    = 0;
	bool                       nfru_debug_view     = false;
	bool                       nfru_debug_checking = false;
	bool                       use_fragment_path   = true;
	bool                       enable_mv_hints     = false;
	bool                       nfru_manage_history = true;
	bool                       nfru_initialized{false};
	bool                       nfru_has_successful_dispatch{false};
	vkb::FfxMemoryUsageTracker memory_usage_tracker{};
	struct FragOrCompConfig
	{
		VkImageUsageFlags    output_usage_bit;           ///< STORAGE (compute UAV) / COLOR_ATTACHMENT (fragment RT)
		uint32_t             context_flags;              ///< FFX compute context flags (0 for fragment)
		VkPipelineStageFlags input_read_stage;           ///< stage that reads NFRU inputs
		VkPipelineStageFlags output_write_stage;         ///< stage that writes the NFRU output
		VkAccessFlags        output_write_access;        ///< access used when writing the NFRU output
		VkImageLayout        output_write_layout;        ///< image layout used while NFRU writes the output
		FfxApiResourceState  input_read_state;           ///< FFX resource state for NFRU inputs
		FfxApiResourceState  output_write_state;         ///< FFX resource state for the NFRU output
	};
	FragOrCompConfig frag_or_comp_config{};

	void create_nfru_context(VkExtent2D display_size, VkExtent2D render_size);
	void configure_nfru();
	void log_memory_usage(const char *event, bool force_log);
	/// Clear application-managed history once before the first NFRU read.
	void           initialize_external_history(vkb::core::CommandBufferC &command_buffer, FrameInputs &previous);
	FfxApiResource wrap_resource(const vkb::core::Image &image, FfxApiResourceState state,
	                             uint32_t additional_usages = 0);
	bool           dispatch_nfru(vkb::core::CommandBufferC &command_buffer, FrameInputs &current, FrameInputs &previous,
	                             const NFRUFrameParams &frame_params, bool reset);

	// Per-mode frame production, called from draw().
	const vkb::core::ImageView *render_frame_inputs(vkb::core::CommandBufferC &command_buffer, FrameInputs &inputs);
	const vkb::core::ImageView *draw_realtime(vkb::core::CommandBufferC &command_buffer, uint32_t swapchain_frame_index);
	const vkb::core::ImageView *draw_dataset(vkb::core::CommandBufferC &command_buffer, uint32_t swapchain_frame_index);

	// Display pipeline
	std::unique_ptr<vkb::PostProcessingPipeline> display_pipeline;

	// Realtime mode
	std::unique_ptr<vkb::RenderPipeline> scene_pipeline;
	vkb::TemporalForwardSubpass         *temporal_subpass        = nullptr;
	vkb::sg::PerspectiveCamera          *camera                  = nullptr;
	vkb::sg::Node                       *camera_node             = nullptr;
	float                                nfru_delta_time_seconds = 0.0f;
	VkExtent2D                           render_extent{};

	// Deterministic orbit camera
	float orbit_radius             = 12.0f;
	float orbit_height             = 150.0f;
	float camera_degrees_per_frame = 0.5f;

	// EXR readback and save
	struct ReadbackSlot
	{
		std::unique_ptr<vkb::core::BufferC> buffer;
		bool                                need_save{false};
		uint64_t                            frame_number{0};
		std::string                         output_dir;
	};

	// When saving, one readback slot is allocated per swapchain/RenderFrame slot; its fence
	// protects reuse of the in-flight image-to-buffer copy.
	std::vector<ReadbackSlot> readback_slots;

	bool        save_exr_enabled = false;
	std::string gt_output_dir{"result/nfru/ground_truth"};
	std::string interp_output_dir{"result/nfru/interpolated"};

	void create_readback_buffers(VkExtent2D extent, size_t count);
	void copy_to_readback(vkb::core::CommandBufferC  &command_buffer,
	                      const vkb::core::ImageView &source_view,
	                      vkb::core::BufferC         &destination);
	bool save_completed_readback(ReadbackSlot &slot, VkExtent2D extent);

	// Dataset mode. The loader owns one color/depth/motion staging triplet per
	// RenderFrame slot.
	std::unique_ptr<NFRUDatasetLoader> dataset_loader;
	VkExtent2D                         dataset_display_size{};
	VkExtent2D                         dataset_render_size{};
};

std::unique_ptr<vkb::VulkanSampleC> create_nfru();
