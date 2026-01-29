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

// Standard library headers
#include <cmath>
#include <memory>
#include <unordered_map>
#include <vector>

// Vulkan headers
#include <vulkan/vulkan.h>

// Vulkan Samples framework headers
#include "rendering/postprocessing_pipeline.h"
#include "rendering/render_pipeline.h"
#include "scene_graph/components/perspective_camera.h"
#include "vulkan_sample.h"

// FidelityFX NSS SDK headers
#ifndef FFX_CPU
#define FFX_CPU
#endif

#include <ffx_api_types.h>
#include <ffx_api.hpp>
#include <ffx_nss.hpp>
#include <vk/ffx_api_vk.hpp>

class NSSSample : public vkb::VulkanSampleC
{
  public:
	// ========== Public Interface ==========
	NSSSample();
	virtual ~NSSSample() override;

	// VulkanSample interface overrides
	virtual bool prepare(const vkb::ApplicationOptions &options) override;
	virtual void update(float delta_time) override;
	virtual void draw_gui() override;
	virtual void draw(vkb::core::CommandBufferC &command_buffer, vkb::RenderTarget &render_target) override;
	virtual void request_gpu_features(vkb::core::PhysicalDeviceC &gpu) override;

	// ========== Public Types ==========

	/**
	 * @brief Resolution configuration for NSS rendering
	 * Centralizes all resolution calculations to avoid redundancy
	 */
	struct ResolutionConfig
	{
		VkExtent2D display_extent;         // High-res (swapchain/screen resolution)
		VkExtent2D render_extent;          // Low-res (scene rendering resolution)
		float      scale_factor;           // NSS upscaling factor (e.g., 2.0)
		float      mipmap_lod_bias;        // Mipmap bias for texture sampling (-log2(scale_factor))

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

			// Calculate mipmap LOD bias: -log2(scale_factor)
			config.mipmap_lod_bias = -std::log2(scale);

			return config;
		}
	};

	/**
	 * @brief Display mode enumeration for debug visualization
	 */
	enum class DisplayMode : uint32_t
	{
		FinalOutput  = 0,        // Display final NSS output
		Velocity     = 1,        // Display motion vectors (color-coded)
		Depth        = 2,        // Display depth buffer (grayscale)
		NSSDebugView = 3,        // Display NSS debug visualization (SDK internal debug view)
		LowResColor  = 4         // Display low-resolution color
	};

	/**
	 * @brief TAA jitter settings uploaded to GPU as uniform buffer
	 */
	struct alignas(16) JitterSettings
	{
		glm::vec2 jitter_offset;            // 8 bytes
		float     jitter_scale;             // 4 bytes
		float     _padding;                 // 4 bytes (padding for alignment)
		glm::mat4 jitter_view_proj;         // 64 bytes - jittered view-projection matrix
		glm::mat4 prev_view_proj;           // 64 bytes - previous frame view-projection matrix
		glm::mat4 current_view_proj;        // 64 bytes - current frame view-projection matrix
	};

	/**
	 * @brief Per-node velocity settings for motion vector calculation
	 */
	struct alignas(16) VelocitySettings
	{
		glm::mat4 prev_model;           // 64 bytes - previous frame model matrix
		glm::mat4 current_model;        // 64 bytes - current frame model matrix
	};

  private:
	// ========== Custom Rendering Subpass ==========

	/**
	 * @brief Custom ForwardSubpass that handles TAA jitter and motion vectors
	 *
	 * As a nested class, TAAForwardSubpass has access to all private members of NSSSample.
	 * This allows it to manage jitter buffers and motion tracking without exposing internals.
	 */
	class TAAForwardSubpass : public vkb::ForwardSubpass
	{
	  public:
		TAAForwardSubpass(vkb::RenderContext &render_context, vkb::ShaderSource &&vertex_shader, vkb::ShaderSource &&fragment_shader, vkb::sg::Scene &scene, vkb::sg::Camera &camera, NSSSample *parent) :
		    vkb::ForwardSubpass(render_context, std::move(vertex_shader), std::move(fragment_shader), scene, camera), nss_sample(parent)
		{}

		void update_uniform(vkb::core::CommandBufferC &command_buffer, vkb::sg::Node &node, size_t thread_index) override;
		void draw(vkb::core::CommandBufferC &command_buffer) override;

	  private:
		NSSSample *nss_sample;
	};

	// ========== Core Components ==========
	vkb::sg::PerspectiveCamera                  *camera{nullptr};
	std::unique_ptr<vkb::RenderPipeline>         scene_pipeline{};
	std::unique_ptr<vkb::PostProcessingPipeline> postprocessing_pipeline{};

	ResolutionConfig resolution_config{};
	DisplayMode      display_mode{DisplayMode::FinalOutput};

	bool     first_frame{true};
	uint32_t total_frames{0};

	// ========== Jitter ===========

	glm::vec2 cur_jitter{0.0f, 0.0f};        // Current frame jitter offset (in pixels)
	float     jitter_scale{1.0f};            // user can modify jitter intensity

	static const int       MAX_TAA_SAMPLES = 16;        // Number of samples in Halton sequence
	std::vector<glm::vec2> halton_sequence;             // Pre-generated Halton jitter offsets
	void                   prepare_halton_sequence();

	JitterSettings                      jitter_settings_data{};
	std::unique_ptr<vkb::core::BufferC> jitter_settings_buffer{};
	void                                prepare_taa_settings_buffer();        // Create jitter uniform buffer
	void                                update_taa_settings_buffer();         // Update jitter data per frame

	// ========== Velocity ===========

	/**
	 * @brief Per-node motion vector tracking data
	 * Stores previous and current transforms to calculate per-object motion vectors
	 */
	struct NodeMotionData
	{
		glm::mat4 current_transform{1.0f};         // Current frame's model matrix
		glm::mat4 previous_transform{1.0f};        // Previous frame's model matrix
		uint32_t  last_update_frame{0};            // Frame number of last update (to detect multiple calls)
	};
	std::unordered_map<vkb::sg::Node *, NodeMotionData> node_motion_data;

	// Previous frame's view-projection matrix
	glm::mat4 prev_view_proj_matrix{1.0f};

	// ========== Render Targets & Buffers ==========

	/**
	 * @brief Scene render target attachment indices
	 */
	enum class Attachments : int
	{
		Depth    = 0,
		Color    = 1,
		Velocity = 2
	};

	std::vector<vkb::LoadStoreInfo>                 scene_load_store{};            // Load/store ops for low-res rendering
	std::vector<std::unique_ptr<vkb::RenderTarget>> low_res_render_targets;        // Low-res targets (one per swapchain image)

	void create_low_res_render_targets();        // Create low-res scene rendering targets

	// ========== NSS (Neural Super Sampling) ==========
	bool nss_enabled{false};        // Whether NSS was successfully initialized

	// Command-line app arguments (forwarded by platform option: --app-arg KEY=VALUE)
	bool     nss_force_disabled{false};        // Force disable NSS (NSS_ENABLE=0)
	uint32_t nss_override_flags{0};            // Override NSS context flags (NSS_FLAGS)
	float    nss_scale_factor{2.0f};           // NSS upscaling factor (NSS_SCALE_FACTOR)

	float last_delta_time{0.016f};

	// NSS context handle
	ffxContext nss_context{nullptr};

	// NSS output images
	std::vector<std::unique_ptr<vkb::core::Image>>     nss_output_images;
	std::vector<std::unique_ptr<vkb::core::ImageView>> nss_output_views;

	void create_nss_output_images();                                                                        // Create high-res NSS output images
	void initialize_nss_context(const VkExtent2D &low_res_extent, const VkExtent2D &display_extent);        // Initialize NSS context

	// Helper: Wrap an image as FfxApiResource for NSS dispatch.
	FfxApiResource wrap_resource(const vkb::core::Image &image, FfxApiResourceState state, uint32_t additional_usages = 0);

	// ========== Debug & Development ==========

	// Jitter sequence selection (for fallback testing)
	bool use_nss_jitter_sequence{true};        // True: NSS jitter; False: Halton sequence
};

std::unique_ptr<vkb::VulkanSampleC> create_nss();