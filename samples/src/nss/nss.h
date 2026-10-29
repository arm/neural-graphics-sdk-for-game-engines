/* Copyright (c) 2025-2026, Arm Limited and Contributors
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
 * SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <vulkan/vulkan.h>

#include "common/ffx_memory_usage.h"
#include "core/buffer.h"
#include "vulkan_sample.h"

// FidelityFX NSS SDK headers
#ifndef FFX_CPU
#	define FFX_CPU
#endif

#include <ffx_api.hpp>
#include <ffx_api_types.h>
#include <ffx_nss.hpp>
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
}        // namespace sg
}        // namespace vkb

class NSSSample : public vkb::VulkanSampleC
{
  public:
	NSSSample();
	~NSSSample() override;

	bool prepare(const vkb::ApplicationOptions &options) override;
	void update(float delta_time) override;
	void draw_gui() override;
	void draw(vkb::core::CommandBufferC &command_buffer, vkb::RenderTarget &render_target) override;
	void request_gpu_features(vkb::core::PhysicalDeviceC &gpu) override;

  private:
	// Resolution configuration for rendering
	// Centralizes all resolution calculations to avoid redundancy
	struct ResolutionConfig
	{
		VkExtent2D display_extent;
		VkExtent2D render_extent;
		float      scale_factor;
		float      mipmap_lod_bias;

		// Calculate low-res extent from display extent and scale factor
		static ResolutionConfig create(const VkExtent2D &display, float scale)
		{
			ResolutionConfig config;
			config.display_extent = display;
			config.scale_factor   = scale;

			config.render_extent.width  = static_cast<uint32_t>(display.width / scale);
			config.render_extent.height = static_cast<uint32_t>(display.height / scale);

			config.mipmap_lod_bias = -std::log2(scale);

			return config;
		}
	};

	enum class DisplayMode : uint32_t
	{
		FinalOutput  = 0,
		NSSDebugView = 1
	};

	enum class Mode : uint32_t
	{
		Realtime,
		Dataset
	};
	vkb::sg::PerspectiveCamera                  *camera{nullptr};
	std::unique_ptr<vkb::RenderPipeline>         scene_pipeline{};
	std::unique_ptr<vkb::PostProcessingPipeline> postprocessing_pipeline{};
	vkb::TemporalForwardSubpass                 *temporal_subpass{nullptr};

	Mode             mode{Mode::Realtime};
	ResolutionConfig resolution_config{};
	DisplayMode      display_mode{DisplayMode::FinalOutput};

	/*******************************
	 * render targets and buffers
	 *******************************/
	/// Scene render target attachment indices.
	enum class Attachments : int
	{
		Depth    = 0,
		Color    = 1,
		Velocity = 2
	};

	struct FrameInputs
	{
		std::unique_ptr<vkb::core::Image>  depth;
		std::unique_ptr<vkb::core::Image>  color;
		std::unique_ptr<vkb::core::Image>  motion_vectors;
		std::unique_ptr<vkb::RenderTarget> render_target;
	};

	// Indexed by scene_frame_index; size is max(2, swapchain image count) to avoid
	// reusing an input image while an earlier submission may still read it.
	std::vector<FrameInputs>                        frame_inputs;
	size_t                                          resource_count{0};
	std::vector<std::unique_ptr<vkb::RenderTarget>> ldr_render_targets;

	void create_frame_inputs();
	void create_ldr_render_targets();
	// Barrier helpers
	/// Wait for previous reads, then transition all scene inputs to attachment-write layouts.
	void prepare_render_inputs(vkb::core::CommandBufferC &command_buffer, FrameInputs &inputs);
	/// Make attachment writes visible, then transition all scene inputs to shader-read layout.
	void finish_render_inputs(vkb::core::CommandBufferC &command_buffer, FrameInputs &inputs);
	/// Wait for previous reads, then transition all dataset inputs to transfer-destination layout.
	void prepare_upload_inputs(vkb::core::CommandBufferC &command_buffer, FrameInputs &inputs);
	/// Make transfer writes visible, then transition all dataset inputs to shader-read layout.
	void finish_upload_inputs(vkb::core::CommandBufferC &command_buffer, FrameInputs &inputs);

	/*******************************
	 * NSS
	 *******************************/
	bool use_compute_path = false;
	struct FragOrCompConfig
	{
		VkImageUsageFlags    output_usage_bit;           ///< STORAGE (compute UAV) / COLOR_ATTACHMENT (fragment RT)
		uint32_t             context_flags;              ///< FFX NSS compute context flags (0 for fragment)
		VkPipelineStageFlags input_read_stage;           ///< stage that reads NSS inputs
		VkPipelineStageFlags output_write_stage;         ///< stage that writes the NSS output
		VkAccessFlags        output_write_access;        ///< access used when writing the NSS output
		VkImageLayout        output_write_layout;        ///< image layout used while NSS writes the output
		FfxApiResourceState  input_read_state;           ///< FFX resource state for NSS inputs
		FfxApiResourceState  output_write_state;         ///< FFX resource state for the NSS output
	};
	FragOrCompConfig frag_or_comp_config{};

	// Command-line app arguments (forwarded by platform option: --app-arg KEY=VALUE)
	bool                       nss_force_disabled{false};
	bool                       nss_ldr_mode{false};
	bool                       nss_debug_checking{false};
	bool                       has_nss_flags_override{false};
	uint32_t                   nss_context_flags{0};
	bool                       nss_manage_history{false};
	float                      nss_scale_factor{2.0f};
	VkFormat                   nss_input_color_format{VK_FORMAT_B10G11R11_UFLOAT_PACK32};
	VkFormat                   nss_color_format{VK_FORMAT_B10G11R11_UFLOAT_PACK32};
	FfxApiNssShaderQualityMode nss_quality_mode{FFX_API_NSS_SHADER_QUALITY_MODE_BALANCED};
	uint32_t                   nss_debug_view_mode{0};
	uint32_t                   nss_jitter_phase_count{0};

	float last_delta_time_seconds{0.016f};

	ffxContext                                   nss_context{nullptr};
	bool                                         has_nss_context() const;
	void                                         create_nss_context(VkExtent2D display_size, VkExtent2D render_size);
	void                                         log_memory_usage(const char *event, bool force_log);
	vkb::FfxMemoryUsageTracker                   memory_usage_tracker{};
	bool                                         nss_has_successful_dispatch{false};
	std::unique_ptr<vkb::PostProcessingPipeline> ldr_postprocessing_pipeline;

	struct ImageResource
	{
		std::unique_ptr<vkb::core::Image>     image;
		std::unique_ptr<vkb::core::ImageView> view;
		bool                                  initialized{false};
	};

	// Indexed by scene_frame_index; the previous slot is passed as outputTm1 when
	// history is managed by the sample.
	std::vector<ImageResource> nss_outputs;
	// Debug output is not history, so one image is sufficient.
	ImageResource nss_debug_output;
	// Number of scene frames rendered or loaded; selects input and output slots.
	uint64_t scene_frame_index{0};

	void create_nss_output_images();

	// Barrier helpers
	/// Clear the first application-managed history image before NSS reads it.
	void initialize_external_history(vkb::core::CommandBufferC &command_buffer, ImageResource &previous);
	/// Wait for previous consumers, then transition this dispatch's outputs to the NSS write layout.
	void prepare_outputs(vkb::core::CommandBufferC &command_buffer, ImageResource &current, bool include_debug_output);
	/// Make NSS writes visible to the next readback or shader consumer.
	void finish_outputs(vkb::core::CommandBufferC &command_buffer, ImageResource &current, bool include_debug_output);

	// Wrap an image as FfxApiResource for NSS dispatch.
	FfxApiResource              wrap_resource(const vkb::core::Image &image, FfxApiResourceState state, uint32_t additional_usages = 0);
	const vkb::core::ImageView *dispatch_nss(vkb::core::CommandBufferC &command_buffer, FrameInputs &inputs,
	                                         const NSSFrameParams &frame_params, bool reset,
	                                         const vkb::core::Image *ldr_color_input = nullptr);
	const vkb::core::ImageView *draw_realtime(vkb::core::CommandBufferC &command_buffer);
	const vkb::core::ImageView *draw_dataset(vkb::core::CommandBufferC &command_buffer, uint32_t swapchain_frame_index);

	// Jitter sequence selection (for fallback testing)
	bool use_nss_jitter_sequence{true};        // True: NSS jitter; False: Halton sequence

	/*******************************
	 * dataset replay
	 *******************************/
	// Loader owns one color/depth/motion staging triplet per RenderFrame slot.
	std::unique_ptr<NSSDatasetLoader> dataset_loader{};
	std::string                       dataset_dir{};
	std::string                       dataset_sequence{"0000"};
	const vkb::core::ImageView       *last_dataset_display_view{nullptr};

	/*******************************
	 * EXR readback and save
	 *******************************/
	struct ReadbackSlot
	{
		std::unique_ptr<vkb::core::BufferC> buffer{};
		bool                                need_save{false};
		uint64_t                            frame_number{0};
	};

	void create_readback_buffers(VkExtent2D extent, size_t count);
	void copy_to_readback(vkb::core::CommandBufferC &command_buffer,
	                      ImageResource             &source,
	                      vkb::core::BufferC        &destination);
	bool save_completed_readback(ReadbackSlot &slot, VkExtent2D extent);

	// When saving, one readback slot is allocated per swapchain/RenderFrame slot; its fence
	// protects reuse of the in-flight image-to-buffer copy.
	std::vector<ReadbackSlot> readback_slots{};
	bool                      save_exr_enabled{false};
	std::string               save_exr_output_dir{};
};

std::unique_ptr<vkb::VulkanSampleC> create_nss();