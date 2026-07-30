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

#pragma once

#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include <vulkan/vulkan.h>

#include "common/vk_common.h" // vkb::LoadStoreInfo
#include "vulkan_sample.h"

// FidelityFX NSS SDK headers
#ifndef FFX_CPU
#define FFX_CPU
#endif

#include <ffx_api_types.h>
#include <ffx_api.hpp>
#include <ffx_nss.hpp>
#include <vk/ffx_api_vk.hpp>

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
}

class NSSSample : public vkb::VulkanSampleC
{
  public:
	NSSSample();
	virtual ~NSSSample() override;

	virtual bool prepare(const vkb::ApplicationOptions &options) override;
	virtual void update(float delta_time) override;
	virtual void draw_gui() override;
	virtual void draw(vkb::core::CommandBufferC &command_buffer, vkb::RenderTarget &render_target) override;
	virtual void request_gpu_features(vkb::core::PhysicalDeviceC &gpu) override;

	/**
	 * @brief Resolution configuration for rendering
	 * Centralizes all resolution calculations to avoid redundancy
	 */
	struct ResolutionConfig
	{
		VkExtent2D display_extent;
		VkExtent2D render_extent;
		float      scale_factor;
		float      mipmap_lod_bias;

		/**
		 * @brief Calculate low-res extent from display extent and scale factor
		 */
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

	/**
	 * @brief Display mode enumeration for presentation output selection
	 */
	enum class DisplayMode : uint32_t
	{
		FinalOutput  = 0,
		NSSDebugView = 1
	};


  private:
	vkb::sg::PerspectiveCamera                  *camera{nullptr};
	std::unique_ptr<vkb::RenderPipeline>         scene_pipeline{};
	std::unique_ptr<vkb::PostProcessingPipeline> postprocessing_pipeline{};
	vkb::TemporalForwardSubpass                 *temporal_subpass{nullptr};

	ResolutionConfig resolution_config{};
	DisplayMode      display_mode{DisplayMode::FinalOutput};

	bool first_frame{true};

	// ========== Render Targets & Buffers ==========
	/// Scene render target attachment indices.
	enum class Attachments : int
	{
		Depth    = 0,
		Color    = 1,
		Velocity = 2
	};

	std::vector<vkb::LoadStoreInfo>                 scene_load_store{};
	std::vector<std::unique_ptr<vkb::RenderTarget>> low_res_render_targets;

	void create_low_res_render_targets();

	// ========== NSS (Neural Super Sampling) ==========
	bool nss_enabled{false};

	bool use_fragment_path = true;
	struct FragOrCompConfig
	{
		VkImageUsageFlags    output_usage_bit;     ///< STORAGE (compute UAV) / COLOR_ATTACHMENT (fragment RT)
		uint32_t             context_flags;        ///< FFX NSS fragment context flags (0 for compute)
		VkPipelineStageFlags input_read_stage;     ///< stage that reads NSS inputs
		VkPipelineStageFlags output_write_stage;   ///< stage that writes the NSS output
		VkAccessFlags        output_write_access;  ///< access used when writing the NSS output
		FfxApiResourceState  input_read_state;     ///< FFX resource state for NSS inputs
	};
	FragOrCompConfig frag_or_comp_config{};

	// Command-line app arguments (forwarded by platform option: --app-arg KEY=VALUE)
	bool                       nss_force_disabled{false};
	uint32_t                   nss_override_flags{0};
	float                      nss_scale_factor{2.0f};
	FfxApiNssShaderQualityMode nss_quality_mode{FFX_API_NSS_SHADER_QUALITY_MODE_BALANCED};
	uint32_t                   nss_debug_view_mode{0};
	uint32_t                   nss_jitter_phase_count{0};

	float last_delta_time{0.016f};

	ffxContext nss_context{nullptr};

	std::vector<std::unique_ptr<vkb::core::Image>>     nss_output_images;
	std::vector<std::unique_ptr<vkb::core::ImageView>> nss_output_views;

	std::vector<std::unique_ptr<vkb::core::Image>>     nss_debug_images;
	std::vector<std::unique_ptr<vkb::core::ImageView>> nss_debug_views;

	void create_nss_output_images();
	void initialize_nss_context(const VkExtent2D &low_res_extent, const VkExtent2D &display_extent);

	// Wrap an image as FfxApiResource for NSS dispatch.
	FfxApiResource wrap_resource(const vkb::core::Image &image, FfxApiResourceState state, uint32_t additional_usages = 0);

	// Jitter sequence selection (for fallback testing)
	bool use_nss_jitter_sequence{true};        // True: NSS SDK jitter sequence; False: Halton sequence
};

std::unique_ptr<vkb::VulkanSampleC> create_nss();