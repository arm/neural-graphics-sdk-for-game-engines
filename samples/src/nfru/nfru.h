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

#include <vulkan/vulkan.h>

#include "vulkan_sample.h"

// FidelityFX NFRU SDK headers
#ifndef FFX_CPU
#define FFX_CPU
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
 * @brief NFRU sample — supports both realtime scene rendering and
 *        dataset (EXR) input for NFRU frame interpolation.
 */
class NFRUSample : public vkb::VulkanSampleC
{
  public:
	enum class Mode
	{
		Realtime,
		Dataset
	};

	NFRUSample();
	~NFRUSample() override;

	bool prepare(const vkb::ApplicationOptions &options) override;
	void update(float delta_time) override;
	void draw(vkb::core::CommandBufferC &command_buffer, vkb::RenderTarget &render_target) override;
	void draw_gui() override;

  private:
	void request_gpu_features(vkb::core::PhysicalDeviceC &gpu) override;

	Mode mode = Mode::Realtime;

	// ─── Per-swapchain input images ───
	// Depth: D32_SFLOAT  (DEPTH_STENCIL_ATTACHMENT | SAMPLED | TRANSFER_DST)
	// Color: R8G8B8A8    (COLOR_ATTACHMENT | SAMPLED | TRANSFER_DST)
	// MV:    R16G16      (COLOR_ATTACHMENT | SAMPLED | TRANSFER_DST)
	enum class Att
	{
		Depth = 0,
		Color = 1,
		Velocity = 2
	};
	std::vector<std::unique_ptr<vkb::core::Image>> depth_images, color_images, mv_images;
	void create_scene_render_images(VkExtent2D color_extent, VkExtent2D depth_mv_extent, size_t count);

	// ─── Per-swapchain NFRU output images ───
	std::vector<std::unique_ptr<vkb::core::Image>>     nfru_outputs;
	std::vector<std::unique_ptr<vkb::core::ImageView>> nfru_output_views;
	void create_nfru_output_images(VkExtent2D extent, size_t count);

	// ─── NFRU context + dispatch ───
	ffx::Context nfru_context    = nullptr;
	bool         nfru_initialized = false;
	uint64_t     ffx_frame_id    = 0;
	bool         nfru_debug_view = false;
	bool		 use_fragment_path = false;
	struct FragOrCompConfig
	{
		VkImageUsageFlags    output_usage_bit;     ///< STORAGE (compute UAV) / COLOR_ATTACHMENT (fragment RT)
		uint32_t             context_flags;        ///< FFX fragment context flags (0 for compute)
		VkPipelineStageFlags input_read_stage;     ///< stage that reads NFRU inputs
		VkPipelineStageFlags output_write_stage;   ///< stage that writes the NFRU output
		VkAccessFlags        output_write_access;  ///< access used when writing the NFRU output
		FfxApiResourceState  input_read_state;     ///< FFX resource state for NFRU inputs
	};
	FragOrCompConfig frag_or_comp_config{};

	void           create_nfru_context(VkExtent2D display_size, VkExtent2D render_size,
	                               const float *initial_vp);
	void           configure_nfru();
	FfxApiResource wrap_resource(const vkb::core::Image &img, FfxApiResourceState state,
	                             uint32_t additional_usages = 0);
	void           dispatch_nfru(vkb::core::CommandBufferC &cb, uint32_t swapchain_index,
	                                    const DatasetLoader::FrameParams &params, bool reset);

	// Per-mode frame production, called from draw().
	void           render_realtime(vkb::core::CommandBufferC &cb, uint32_t swapchain_index);
	void           replay_dataset(vkb::core::CommandBufferC &cb, uint32_t swapchain_index);

	// ─── Display pipeline ───
	std::unique_ptr<vkb::PostProcessingPipeline> display_pipeline;

	/// The image shown on screen this frame
	const vkb::core::ImageView *display_source = nullptr;

	/// Whether the frame being produced is the NFRU-generated frame
	bool showing_generated_frame() const;

	// ─── Realtime mode ───
	std::vector<std::unique_ptr<vkb::RenderTarget>> render_targets;
	std::unique_ptr<vkb::RenderPipeline>             scene_pipeline;
	vkb::TemporalForwardSubpass                      *temporal_subpass = nullptr;
	vkb::sg::PerspectiveCamera                      *camera      = nullptr;
	vkb::sg::Node                                   *camera_node = nullptr;
	uint64_t                                         frame_count  = 0;
	float                                            last_delta_time = 0.0f;
	VkExtent2D                                       render_extent{};

	// ─── Deterministic orbit camera ───
	float orbit_radius            = 12.0f;
	float orbit_height            = 150.0f;
	float camera_degrees_per_frame = 0.5f;

	// ─── EXR readback + save ───
	struct ReadbackBuffer
	{
		VkBuffer       buffer{VK_NULL_HANDLE};
		VkDeviceMemory memory{VK_NULL_HANDLE};
		VkDeviceSize   size{0};
		void          *mapped{nullptr};   ///< Persistently mapped pointer
	};

	ReadbackBuffer create_readback_buffer(VkDeviceSize size);

	struct ReadbackSlot
	{
		ReadbackBuffer buffer;
		bool           need_save{false};
		uint64_t       frame_number{0};
		std::string    output_dir;
	};

	std::vector<ReadbackSlot> readback_slots;   ///< One per swapchain image

	bool        save_enabled = false;
	std::string gt_output_dir{"result/nfru/ground_truth"};
	std::string interp_output_dir{"result/nfru/interpolated"};

	void create_readback_buffers(VkExtent2D extent, size_t count);
	void copy_to_readback(vkb::core::CommandBufferC &cmd,
	                      const vkb::core::ImageView &src_view, ReadbackBuffer &dst);
	void flush_pending_save(ReadbackSlot &slot, VkExtent2D extent);

	// ─── Dataset mode ───
	std::unique_ptr<DatasetLoader>  dataset_loader;
	int                             current_frame{0};
	VkExtent2D                      dataset_display_size{};
	VkExtent2D                      dataset_render_size{};
};

std::unique_ptr<vkb::VulkanSampleC> create_nfru();
