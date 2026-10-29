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

/*
 * SPDX-FileCopyrightText: Copyright 2020-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT
 */

#include "nss.h"

#include <algorithm>
#include <cassert>
#include <filesystem>
#include <string>

#include <glm/gtc/packing.hpp>

#include "common/app_argument_reader.h"
#include "common/exr_image.h"
#include "common/strings.h"
#include "common/utils.h"
#include "core/image.h"
#include "core/image_view.h"

#include "rendering/render_pipeline.h"

#include "rendering/subpasses/temporal_forward_subpass.h"

#include "rendering/postprocessing_pipeline.h"

#include "scene_graph/components/perspective_camera.h"

#include "rendering/postprocessing_renderpass.h"

namespace
{
constexpr const char *kNssDebugViewModeLabels[] = {
    "Overview (4x4)",
    "1: history_color",
    "2: input_depth",
    "3: prev_depth",
    "4: nearest_offset",
    "5: low_res_color",
    "6: motion_vector",
    "7: luma_deriv_tm1",
    "8: temporal_feedback",
    "9: lr_warped_history_tensor",
    "10: disocclusion_mask",
    "11: luma_deriv_t",
    "12: depth_dilated",
    "13: unjittered_color_tensor",
    "14: motion_detector_tensor",
    "15: luma_instability_tensor",
    "16: warp_feedback_tensor",
};
}        // namespace

/*******************************
 * destructor and constructor
 *******************************/
NSSSample::~NSSSample()
{
	if (get_device().get_handle() != VK_NULL_HANDLE)
	{
		vkDeviceWaitIdle(get_device().get_handle());
		LOGI("GPU work completed, safe to destroy resources");
	}

	if (has_nss_context())
	{
		log_memory_usage("before_destroy", true);
		auto result = ffxDestroyContext(&nss_context, nullptr);
		if (result != FFX_API_RETURN_OK)
		{
			LOGW("NSS Context destroy returned error: {}", result);
		}
		else
		{
			LOGI("NSS Context destroyed");
		}

		nss_context = nullptr;
	}
}

NSSSample::NSSSample()
{
	set_api_version(VK_API_VERSION_1_4);

	add_device_extension("VK_KHR_dynamic_rendering", true);               // Required by VK_KHR_maintenance5
	add_device_extension("VK_KHR_maintenance5", true);                    // Required by VK_ARM_data_graph
	add_device_extension("VK_KHR_deferred_host_operations", true);        // Required by VK_ARM_data_graph

	// ARM ML extensions
	add_device_extension("VK_ARM_data_graph", true);
	add_device_extension("VK_ARM_tensors", true);
}

void NSSSample::request_gpu_features(vkb::core::PhysicalDeviceC &gpu)
{
	vkb::VulkanSampleC::request_gpu_features(gpu);

	// Quantized NSS shaders use 8-bit integer types for model operations.
	REQUEST_REQUIRED_FEATURE(gpu, VkPhysicalDeviceVulkan12Features, shaderInt8);
	// FP16 support is enabled by default unless FFX_API_NSS_CONTEXT_FLAG_DISABLE_16BIT is set.
	REQUEST_OPTIONAL_FEATURE(gpu, VkPhysicalDeviceVulkan12Features, shaderFloat16);
	REQUEST_OPTIONAL_FEATURE(gpu, VkPhysicalDevice16BitStorageFeatures, storageBuffer16BitAccess);
	REQUEST_OPTIONAL_FEATURE(gpu, VkPhysicalDevice16BitStorageFeatures, uniformAndStorageBuffer16BitAccess);
	if (gpu.get_features().shaderStorageImageExtendedFormats)
	{
		gpu.get_mutable_requested_features().shaderStorageImageExtendedFormats = true;
	}
	// Synchronization2: NSS uses vkCmdPipelineBarrier2 for pipeline barriers
	REQUEST_REQUIRED_FEATURE(gpu, VkPhysicalDeviceVulkan13Features, synchronization2);

	REQUEST_OPTIONAL_FEATURE(gpu, VkPhysicalDeviceTensorFeaturesARM, shaderTensorAccess);
	REQUEST_OPTIONAL_FEATURE(gpu, VkPhysicalDeviceTensorFeaturesARM, tensors);
	REQUEST_OPTIONAL_FEATURE(gpu, VkPhysicalDeviceDataGraphFeaturesARM, dataGraph);
}

/*******************************
 * prepare
 *******************************/

bool NSSSample::prepare(const vkb::ApplicationOptions &options)
{
	if (!vkb::VulkanSampleC::prepare(options))
	{
		return false;
	}

	// ------------------Command Line Configuration------------------------------
	// Example:
	//   vulkan_samples sample nss --app-arg NSS_ENABLE=1,NSS_SCALE_FACTOR=2.0,NSS_FLAGS=0x045,NSS_QUALITY=1,NSS_DEBUG_VIEW_MODE=0
	//   vulkan_samples sample nss --app-arg NSS_ENABLE=1,NSS_SCALE_FACTOR=2.0,NSS_DEBUG_CHECKING=1,NSS_QUALITY=1
	//
	// Supported keys:
	//   NSS_ENABLE       : 0 or 1 (default 1)
	//   NSS_SCALE_FACTOR : float in [1.0, 3.0] (default 2.0)
	//   NSS_FLAGS        : raw context flags; PRE/POST compute selectors must be paired in this sample
	//   NSS_DEBUG_CHECKING : 0 or 1 (default 0), enable runtime jitter diagnostics
	//   NSS_QUALITY      : 0 (Quality), 1 (Balanced), 2 (Performance), default 1
	//   NSS_DEBUG_VIEW_MODE : uint32 in [0, 16], default 0 (0=overview, 1..16=single tile)
	//   NSS_USE_COMPUTE : preprocess/postprocess path; 0 (fragment) or 1 (compute), default 0
	//   NSS_LDR_MODE : Experimental; 0 (HDR input) or 1 (LDR input), default 0
	//   NSS_COLOR_FORMAT : R11G11B10 (default), RGBA16F, or RGBA32F
	//   NSS_INPUT_COLOR_FORMAT : optional input override; takes precedence over NSS_COLOR_FORMAT
	//   NSS_OUTPUT_COLOR_FORMAT : optional output/history override; takes precedence over NSS_COLOR_FORMAT
	//   NSS_MANAGE_HISTORY : 0 (application provides outputTm1) or 1 (SDK manages history), default 0
	//   NSS_DATASET_DIR : dataset root; enables dataset replay when non-empty
	//   NSS_DATASET_SEQUENCE : dataset sequence ID, default 0000
	//   NSS_SAVE_EXR : save dataset NSS output as EXR, default 0
	//   NSS_OUTPUT_DIR : output root; defaults to result under the dataset directory
	vkb::AppArgumentReader args{options.app_arguments};

	if (args.has("NSS_ENABLE"))
	{
		nss_force_disabled = !args.get_bool("NSS_ENABLE", true);
		LOGI("NSS {} via NSS_ENABLE", nss_force_disabled ? "disabled" : "enabled");
	}

	nss_scale_factor = args.get_float("NSS_SCALE_FACTOR", nss_scale_factor, 1.0f, 3.0f);
	nss_ldr_mode     = args.get_bool("NSS_LDR_MODE", nss_ldr_mode);

	nss_debug_checking = args.get_bool("NSS_DEBUG_CHECKING", nss_debug_checking);
	if (nss_debug_checking)
	{
		LOGI("NSS runtime jitter checking enabled");
	}
	nss_quality_mode = static_cast<FfxApiNssShaderQualityMode>(
	    args.get_int("NSS_QUALITY", static_cast<int>(nss_quality_mode),
	                 FFX_API_NSS_SHADER_QUALITY_MODE_QUALITY, FFX_API_NSS_SHADER_QUALITY_MODE_PERFORMANCE));

	if (args.has("NSS_DEBUG_VIEW_MODE"))
	{
		nss_debug_view_mode = args.get_uint("NSS_DEBUG_VIEW_MODE", nss_debug_view_mode, 0, 16);
		display_mode        = DisplayMode::NSSDebugView;
	}

	const auto parse_color_format = [](const std::string &value, const char *name) {
		if (value == "RGBA16F")
			return VK_FORMAT_R16G16B16A16_SFLOAT;
		if (value == "RGBA32F")
			return VK_FORMAT_R32G32B32A32_SFLOAT;
		if (value != "R11G11B10")
			LOGW("Invalid {} '{}': expected R11G11B10, RGBA16F, or RGBA32F; using R11G11B10", name, value);
		return VK_FORMAT_B10G11R11_UFLOAT_PACK32;
	};
	const std::string color_format = args.get_string("NSS_COLOR_FORMAT", "R11G11B10");
	nss_color_format               = parse_color_format(color_format, "NSS_COLOR_FORMAT");
	nss_input_color_format         = nss_color_format;
	if (args.has("NSS_INPUT_COLOR_FORMAT"))
		nss_input_color_format = parse_color_format(args.get_string("NSS_INPUT_COLOR_FORMAT"), "NSS_INPUT_COLOR_FORMAT");
	if (args.has("NSS_OUTPUT_COLOR_FORMAT"))
		nss_color_format = parse_color_format(args.get_string("NSS_OUTPUT_COLOR_FORMAT"), "NSS_OUTPUT_COLOR_FORMAT");
	LOGI("NSS input color format: {}", vkb::to_string(nss_input_color_format));
	LOGI("NSS output/history color format: {}", vkb::to_string(nss_color_format));

	// dataset arguments
	const bool save_exr_requested = args.get_bool("NSS_SAVE_EXR", false);
	const std::string output_dir  = args.get_string("NSS_OUTPUT_DIR");
	dataset_dir                   = args.get_string("NSS_DATASET_DIR");
	dataset_sequence              = args.get_string("NSS_DATASET_SEQUENCE", dataset_sequence);
	mode             = dataset_dir.empty() ? Mode::Realtime : Mode::Dataset;
	save_exr_enabled = mode == Mode::Dataset && save_exr_requested;

	// flag arguments
	use_compute_path             = args.get_bool("NSS_USE_COMPUTE", use_compute_path);
	nss_manage_history           = args.get_bool("NSS_MANAGE_HISTORY", nss_manage_history);
	has_nss_flags_override       = args.has("NSS_FLAGS");
	const uint32_t compute_flags = FFX_API_NSS_CONTEXT_FLAG_PRE_PROCESS_COMPUTE |
	                               FFX_API_NSS_CONTEXT_FLAG_POST_PROCESS_COMPUTE;
	if (has_nss_flags_override)
	{
		nss_context_flags = args.get_uint("NSS_FLAGS", 0);
		if (nss_ldr_mode)
			nss_context_flags |= FFX_API_NSS_CONTEXT_FLAG_LOW_DYNAMIC_RANGE;
		nss_ldr_mode = (nss_context_flags & FFX_API_NSS_CONTEXT_FLAG_LOW_DYNAMIC_RANGE) != 0;

		// override flag config
		nss_manage_history = (nss_context_flags & FFX_API_NSS_CONTEXT_FLAG_MANAGE_HISTORY) != 0;

		const uint32_t selected_compute_flags = nss_context_flags & compute_flags;
		if (selected_compute_flags != 0 && selected_compute_flags != compute_flags)
		{
			LOGE("NSS_FLAGS must set PRE_PROCESS_COMPUTE and POST_PROCESS_COMPUTE together");
			return false;
		}
		use_compute_path = selected_compute_flags == compute_flags;
	}

	frag_or_comp_config = use_compute_path ? FragOrCompConfig{VK_IMAGE_USAGE_STORAGE_BIT,
	                                                          compute_flags,
	                                                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
	                                                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
	                                                          VK_ACCESS_SHADER_WRITE_BIT,
	                                                          VK_IMAGE_LAYOUT_GENERAL,
	                                                          FFX_API_RESOURCE_STATE_COMPUTE_READ,
	                                                          FFX_API_RESOURCE_STATE_UNORDERED_ACCESS} :
	                                         FragOrCompConfig{VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
	                                                          0u,
	                                                          VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
	                                                          VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
	                                                          VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
	                                                          VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
	                                                          FFX_API_RESOURCE_STATE_PIXEL_READ,
	                                                          FFX_API_RESOURCE_STATE_RENDER_TARGET};

	// ------------------Initial resolution------------------------------
	auto  &swapchain       = get_render_context().get_swapchain();
	size_t swapchain_count = swapchain.get_images().size();
	resource_count         = std::max<size_t>(2, swapchain_count);

	if (mode == Mode::Realtime)
	{
		load_scene("scenes/sponza/Sponza01.gltf");

		const VkExtent2D display_extent = get_render_context().get_surface_extent();
		resolution_config               = ResolutionConfig::create(display_extent, nss_scale_factor);
	}
	else
	{
		dataset_loader = std::make_unique<NSSDatasetLoader>();
		if (!dataset_loader->init(get_device(),
		                          dataset_dir,
		                          dataset_sequence,
		                          nss_scale_factor,
		                          nss_input_color_format,
		                          static_cast<uint32_t>(swapchain_count)))
		{
			close();
			return false;
		}

		resolution_config.display_extent = dataset_loader->get_display_size();
		resolution_config.render_extent  = dataset_loader->get_render_size();
		resolution_config.scale_factor   = static_cast<float>(resolution_config.display_extent.width) /
		                                 static_cast<float>(std::max(1u, resolution_config.render_extent.width));
		resolution_config.mipmap_lod_bias = 0.0f;
		use_nss_jitter_sequence           = false;

		const std::string     ratio_dir  = dataset_loader->get_ratio_dir();
		std::filesystem::path output_root = output_dir.empty() ? std::filesystem::path(dataset_dir) / "result" : std::filesystem::path(output_dir);
		if (!output_dir.empty() && output_root.is_relative())
		{
			output_root = std::filesystem::path(dataset_dir) / output_root;
		}
		save_exr_output_dir = (output_root / ("nss_" + ratio_dir) / dataset_sequence).string();
		if (save_exr_enabled)
		{
			if (!vkb::sample::exr::create_directories(save_exr_output_dir))
				return false;
		}
	}

	LOGI("=== Resolution Configuration ===");
	LOGI("  Display (high-res): {}x{}", resolution_config.display_extent.width, resolution_config.display_extent.height);
	LOGI("  Render (low-res):   {}x{}", resolution_config.render_extent.width, resolution_config.render_extent.height);
	LOGI("  Scale factor:       {:.2f}x", resolution_config.scale_factor);
	if (mode == Mode::Realtime)
	{
		LOGI("  Mipmap LOD bias:    {:.2f}", resolution_config.mipmap_lod_bias);
	}
	else
	{
		LOGI("  Mipmap LOD bias:    N/A (dataset replay)");
	}

	// ------------------NSS Context Creation------------------------------
	create_nss_context(resolution_config.display_extent, resolution_config.render_extent);

	// Adjust scale factor if NSS failed to initialize
	if (!has_nss_context() && mode == Mode::Realtime)
	{
		nss_scale_factor        = 1.0f;
		use_nss_jitter_sequence = false;        // Fall back to Halton sequence
		nss_jitter_phase_count  = 0;
		// Recalculate resolution config with scale factor = 1.0
		resolution_config = ResolutionConfig::create(get_render_context().get_surface_extent(), 1.0f);
		LOGI("NSS disabled: rendering at native resolution (scale factor = 1.0)");
	}
	else if (use_nss_jitter_sequence)
	{
		int32_t                              jitter_phase_count = 0;
		ffx::QueryDescNssGetJitterPhaseCount phase_query{};
		phase_query.renderWidth    = resolution_config.render_extent.width;
		phase_query.displayWidth   = resolution_config.display_extent.width;
		phase_query.pOutPhaseCount = &jitter_phase_count;

		ffx::ReturnCode result = ffx::Query(phase_query);
		if (result == ffx::ReturnCode::Ok && jitter_phase_count > 0)
		{
			nss_jitter_phase_count = static_cast<uint32_t>(jitter_phase_count);
		}
		else
		{
			use_nss_jitter_sequence = false;
			LOGW("Failed to query NSS jitter phase count (code {}), falling back to Halton sequence", static_cast<uint32_t>(result));
		}
	}

	// ------------------Create Render Targets and Images------------------------------
	// Create NSS input images (depth, color, motion vectors)
	create_frame_inputs();
	if (mode == Mode::Realtime && nss_ldr_mode && has_nss_context())
		create_ldr_render_targets();

	// Create NSS output images (only if NSS enabled)
	create_nss_output_images();

	// Create readback buffers for saving EXR output
	if (mode == Mode::Dataset && save_exr_enabled && !has_nss_context())
	{
		LOGW("NSS EXR saving disabled because NSS is not available");
		save_exr_enabled = false;
	}
	if (mode == Mode::Dataset && save_exr_enabled && has_nss_context())
	{
		create_readback_buffers(resolution_config.display_extent, swapchain_count);
	}

	// ------------------Scene Rendering Pipeline (Only Realtime Mode)------------------------------
	if (mode == Mode::Realtime)
	{
		auto &camera_node = vkb::add_free_camera(get_scene(), "main_camera", resolution_config.render_extent);
		camera            = dynamic_cast<vkb::sg::PerspectiveCamera *>(&camera_node.get_component<vkb::sg::Camera>());

		vkb::ShaderSource vert_shader("temporal/jitter.vert.spv");
		vkb::ShaderSource frag_shader("temporal/velocity.frag.spv");
		auto              scene_subpass = std::make_unique<vkb::TemporalForwardSubpass>(get_render_context(), std::move(vert_shader), std::move(frag_shader), get_scene(), *camera);
		temporal_subpass                = scene_subpass.get();
		scene_subpass->set_output_attachments({static_cast<int>(Attachments::Color), static_cast<int>(Attachments::Velocity)});
		scene_subpass->set_mipmap_lod_bias(resolution_config.mipmap_lod_bias);
		scene_subpass->set_jitter_scale(1.0f);
		if (!(use_nss_jitter_sequence && has_nss_context()))
		{
			scene_subpass->clear_external_jitter();
		}

		scene_pipeline = std::make_unique<vkb::RenderPipeline>();
		scene_pipeline->add_subpass(std::move(scene_subpass));
		scene_pipeline->set_load_store({
		    {VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE},
		    {VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE},
		    {VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE},
		});
		{
			VkClearValue depth_clear{};
			depth_clear.depthStencil = {0.0f, 0U};
			VkClearValue color_clear{};
			color_clear.color = {0.0f, 0.0f, 0.0f, 1.0f};
			VkClearValue vel_clear{};
			vel_clear.color = {0.0f, 0.0f, 0.0f, 0.0f};
			scene_pipeline->set_clear_value({depth_clear, color_clear, vel_clear});
		}
	}

	// ------------------Post-processing Pipeline------------------------------

	// Post-processing pass (display)
	vkb::ShaderSource postprocessing_vs("postprocessing.vert.spv");
	if (mode == Mode::Realtime && nss_ldr_mode && has_nss_context())
	{
		vkb::ShaderSource ldr_postprocessing_vs("postprocessing.vert.spv");
		ldr_postprocessing_pipeline = std::make_unique<vkb::PostProcessingPipeline>(get_render_context(), std::move(ldr_postprocessing_vs));
		ldr_postprocessing_pipeline->add_pass().set_debug_name("NSS LDR Preprocess").add_subpass(vkb::ShaderSource("display_ldr.frag.spv"));
	}
	postprocessing_pipeline = std::make_unique<vkb::PostProcessingPipeline>(get_render_context(), std::move(postprocessing_vs));
	postprocessing_pipeline->add_pass().add_subpass(vkb::ShaderSource("display.frag.spv"));

	get_stats().request_stats({vkb::StatIndex::frame_times});
	create_gui(*window, &get_stats());

	return true;
}

void NSSSample::update(float delta_time)
{
	last_delta_time_seconds = delta_time;

	VulkanSample::update(delta_time);
}

/*******************************
 * resource creation
 *******************************/
void NSSSample::create_frame_inputs()
{
	VkExtent3D input_extent = {
	    resolution_config.render_extent.width,
	    resolution_config.render_extent.height,
	    1};
	const bool realtime = mode == Mode::Realtime;

	// Realtime inputs are rendered as attachments; dataset inputs are transfer destinations.
	// All three become sampled resources for NSS after their producer completes.
	const VkImageUsageFlags depth_usage = VK_IMAGE_USAGE_SAMPLED_BIT |
	                                      (realtime ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT : VK_IMAGE_USAGE_TRANSFER_DST_BIT);
	const VkImageUsageFlags color_usage = VK_IMAGE_USAGE_SAMPLED_BIT |
	                                      (realtime ? VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT : VK_IMAGE_USAGE_TRANSFER_DST_BIT);
	const VkImageUsageFlags motion_vectors_usage = VK_IMAGE_USAGE_SAMPLED_BIT |
	                                               (realtime ? VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT : VK_IMAGE_USAGE_TRANSFER_DST_BIT);

	frame_inputs.resize(resource_count);
	for (size_t index = 0; index < resource_count; ++index)
	{
		auto &inputs       = frame_inputs[index];
		auto  create_input = [&](std::unique_ptr<vkb::core::Image> &image, VkFormat format, VkImageUsageFlags usage, const char *name) {
            image = std::make_unique<vkb::core::Image>(
                get_device(), input_extent, format, usage, VMA_MEMORY_USAGE_GPU_ONLY);
            image->set_debug_name(std::string{name} + " " + std::to_string(index));
		};

		create_input(inputs.depth, VK_FORMAT_D32_SFLOAT, depth_usage, realtime ? "NSS Input Depth" : "NSS Dataset Depth");
		create_input(inputs.color, nss_input_color_format, color_usage, realtime ? "NSS Input Color" : "NSS Dataset Color");
		create_input(inputs.motion_vectors, VK_FORMAT_R16G16_SFLOAT, motion_vectors_usage, realtime ? "NSS Input Motion Vectors" : "NSS Dataset Motion Vectors");

		std::vector<vkb::core::ImageView> views;
		views.emplace_back(*inputs.depth, VK_IMAGE_VIEW_TYPE_2D);
		views.emplace_back(*inputs.color, VK_IMAGE_VIEW_TYPE_2D);
		views.emplace_back(*inputs.motion_vectors, VK_IMAGE_VIEW_TYPE_2D);
		inputs.render_target = std::make_unique<vkb::RenderTarget>(std::move(views));
	}
}

void NSSSample::create_ldr_render_targets()
{
	const VkExtent3D extent = {
	    resolution_config.render_extent.width,
	    resolution_config.render_extent.height,
	    1};
	constexpr VkImageUsageFlags usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;

	ldr_render_targets.reserve(resource_count);
	for (size_t index = 0; index < resource_count; ++index)
	{
		std::vector<vkb::core::Image> images;
		images.emplace_back(get_device(), extent, nss_input_color_format, usage, VMA_MEMORY_USAGE_GPU_ONLY);
		images.back().set_debug_name("NSS LDR Input " + std::to_string(index));
		ldr_render_targets.push_back(std::make_unique<vkb::RenderTarget>(std::move(images)));
	}
}

void NSSSample::create_nss_output_images()
{
	if (!has_nss_context())
	{
		LOGI("NSS not enabled, skipping NSS output image creation");
		return;
	}

	// Outputs are sampled for display and application-managed history, then written
	// by the selected compute or fragment path. Application-managed history needs a
	// transfer clear before its first read; dataset saving needs transfer-source use.
	VkImageUsageFlags nss_output_usage = VK_IMAGE_USAGE_SAMPLED_BIT |
	                                     frag_or_comp_config.output_usage_bit;
	if (!nss_manage_history)
		nss_output_usage |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
	if (save_exr_enabled)
		nss_output_usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
	const VkImageUsageFlags debug_output_usage = VK_IMAGE_USAGE_SAMPLED_BIT |
	                                             frag_or_comp_config.output_usage_bit;

	VkExtent3D display_extent_3d = {
	    resolution_config.display_extent.width,
	    resolution_config.display_extent.height,
	    1};

	nss_outputs.resize(resource_count);
	for (size_t index = 0; index < resource_count; ++index)
	{
		auto &output = nss_outputs[index];
		output.image = std::make_unique<vkb::core::Image>(
		    get_device(),
		    display_extent_3d,
		    nss_color_format,
		    nss_output_usage,
		    VMA_MEMORY_USAGE_GPU_ONLY);
		output.image->set_debug_name("NSS Output " + std::to_string(index));
		output.view = std::make_unique<vkb::core::ImageView>(*output.image, VK_IMAGE_VIEW_TYPE_2D);
	}

	nss_debug_output.image = std::make_unique<vkb::core::Image>(
	    get_device(), display_extent_3d, nss_color_format, debug_output_usage, VMA_MEMORY_USAGE_GPU_ONLY);
	nss_debug_output.image->set_debug_name("NSS Debug Output");
	nss_debug_output.view = std::make_unique<vkb::core::ImageView>(*nss_debug_output.image, VK_IMAGE_VIEW_TYPE_2D);
}

/*******************************
 * draw
 *******************************/

const vkb::core::ImageView *NSSSample::draw_realtime(vkb::core::CommandBufferC &command_buffer)
{
	// Update jitter offset per frame
	if (use_nss_jitter_sequence && nss_jitter_phase_count > 0)
	{
		float                            jitter_x = 0.0f;
		float                            jitter_y = 0.0f;
		ffx::QueryDescNssGetJitterOffset query{};
		query.index      = static_cast<int32_t>(temporal_subpass->peek_frame_index());
		query.phaseCount = nss_jitter_phase_count;
		query.pOutX      = &jitter_x;
		query.pOutY      = &jitter_y;
		if (ffx::Query(query) == ffx::ReturnCode::Ok)
		{
			temporal_subpass->set_external_jitter({jitter_x, jitter_y});
		}
		else
		{
			use_nss_jitter_sequence = false;
			temporal_subpass->clear_external_jitter();
			LOGW("NSS jitter query failed, falling back to the sample jitter sequence");
		}
	}

	auto &inputs       = frame_inputs[scene_frame_index % resource_count];
	auto &scene_rt     = *inputs.render_target;
	auto &scene_views  = scene_rt.get_views();
	auto  scene_extent = scene_rt.get_extent();

	const auto color_attachment = static_cast<uint32_t>(Attachments::Color);
	// Transition scene inputs from UNDEFINED or SHADER_READ_ONLY_OPTIMAL to their
	// color/depth attachment layouts before rendering.
	prepare_render_inputs(command_buffer, inputs);

	VkViewport viewport{};
	viewport.width    = static_cast<float>(scene_extent.width);
	viewport.height   = static_cast<float>(scene_extent.height);
	viewport.minDepth = 0.0f;
	viewport.maxDepth = 1.0f;
	command_buffer.set_viewport(0, {viewport});
	VkRect2D scissor{};
	scissor.extent = {scene_extent.width, scene_extent.height};
	command_buffer.set_scissor(0, {scissor});

	scene_pipeline->draw(command_buffer, scene_rt);
	command_buffer.end_render_pass();

	// Transition the rendered attachments to SHADER_READ_ONLY_OPTIMAL for NSS or
	// native display sampling.
	finish_render_inputs(command_buffer, inputs);

	if (!has_nss_context())
	{
		++scene_frame_index;
		return &scene_views[color_attachment];
	}

	const vkb::core::Image *ldr_color_input = nullptr;
	if (nss_ldr_mode)
	{
		auto &ldr_render_target = *ldr_render_targets[scene_frame_index % resource_count];
		auto &ldr_pass          = ldr_postprocessing_pipeline->get_pass(0);
		auto &ldr_subpass       = ldr_pass.get_subpass(0);
		ldr_subpass.bind_sampled_image("color_sampler", vkb::core::SampledImage(scene_views[color_attachment]));
		ldr_pass.set_render_target(&ldr_render_target);
		ldr_postprocessing_pipeline->draw(command_buffer, ldr_render_target);
		command_buffer.end_render_pass();

		vkb::ImageMemoryBarrier barrier{};
		barrier.old_layout      = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
		barrier.new_layout      = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		barrier.src_access_mask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		barrier.dst_access_mask = VK_ACCESS_SHADER_READ_BIT;
		barrier.src_stage_mask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		barrier.dst_stage_mask  = frag_or_comp_config.input_read_stage;
		command_buffer.image_memory_barrier(ldr_render_target.get_views()[0], barrier);
		ldr_render_target.set_layout(0, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
		ldr_color_input = &ldr_render_target.get_views()[0].get_image();
	}

	NSSFrameParams  params{};
	const glm::vec2 jitter             = temporal_subpass->get_current_jitter_pixels_scaled();
	params.jitter_x                    = jitter.x;
	params.jitter_y                    = jitter.y;
	params.motion_vector_scale_x       = static_cast<float>(resolution_config.render_extent.width);
	params.motion_vector_scale_y       = static_cast<float>(resolution_config.render_extent.height);
	params.camera_fov_angle_vertical   = camera->get_field_of_view();
	params.camera_near                 = camera->get_near_plane();
	params.camera_far                  = camera->get_far_plane();
	params.frame_time_delta_ms         = last_delta_time_seconds * 1000.0f;
	params.exposure                    = 1.0f;
	const vkb::core::ImageView *output = dispatch_nss(command_buffer, inputs, params, scene_frame_index == 0, ldr_color_input);
	if (output)
		++scene_frame_index;
	return output;
}

const vkb::core::ImageView *NSSSample::draw_dataset(vkb::core::CommandBufferC &command_buffer, uint32_t swapchain_frame_index)
{
	if (scene_frame_index >= dataset_loader->total_frames())
		return last_dataset_display_view;

	const size_t current_index = scene_frame_index % resource_count;
	auto        &inputs        = frame_inputs[current_index];
	auto        &input_rt      = *inputs.render_target;
	auto        &input_views   = input_rt.get_views();

	NSSFrameParams params{};
	if (!dataset_loader->load_frame(static_cast<int>(scene_frame_index), swapchain_frame_index, params))
	{
		LOGE("Failed to load NSS dataset frame {}", scene_frame_index);
		close();
		return last_dataset_display_view;
	}

	// Transition dataset inputs from UNDEFINED or SHADER_READ_ONLY_OPTIMAL to
	// TRANSFER_DST_OPTIMAL before uploading the frame.
	prepare_upload_inputs(command_buffer, inputs);

	// Upload color, depth, and motion at the common render extent. Color and motion
	// use the color aspect; depth switches the same copy region to the depth aspect.
	VkBufferImageCopy region{};
	region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	region.imageSubresource.layerCount = 1;
	region.imageExtent                 = {resolution_config.render_extent.width, resolution_config.render_extent.height, 1};
	vkCmdCopyBufferToImage(command_buffer.get_handle(), dataset_loader->get_color_staging_buffer(swapchain_frame_index), inputs.color->get_handle(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
	region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
	vkCmdCopyBufferToImage(command_buffer.get_handle(), dataset_loader->get_depth_staging_buffer(swapchain_frame_index), inputs.depth->get_handle(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
	region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	vkCmdCopyBufferToImage(command_buffer.get_handle(), dataset_loader->get_motion_vector_staging_buffer(swapchain_frame_index), inputs.motion_vectors->get_handle(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

	// Transition all uploaded inputs from TRANSFER_DST_OPTIMAL to
	// SHADER_READ_ONLY_OPTIMAL for NSS or native display sampling.
	finish_upload_inputs(command_buffer, inputs);

	const vkb::core::ImageView *display_source = &input_views[static_cast<uint32_t>(Attachments::Color)];
	if (has_nss_context())
	{
		ImageResource &output = nss_outputs[scene_frame_index % resource_count];
		display_source        = dispatch_nss(command_buffer, inputs, params, params.reset);
		if (!display_source)
		{
			return last_dataset_display_view;
		}

		if (save_exr_enabled)
		{
			ReadbackSlot &slot = readback_slots[swapchain_frame_index];
			copy_to_readback(command_buffer, output, *slot.buffer);
			slot.need_save    = true;
			slot.frame_number = scene_frame_index;
		}
	}
	last_dataset_display_view = display_source;
	++scene_frame_index;
	return last_dataset_display_view;
}

void NSSSample::draw(vkb::core::CommandBufferC &command_buffer, vkb::RenderTarget &swapchain_render_target)
{
	const uint32_t swapchain_frame_index = get_render_context().get_active_frame_index();
	if (mode == Mode::Dataset && save_exr_enabled &&
	    !save_completed_readback(readback_slots[swapchain_frame_index], resolution_config.display_extent))
	{
		close();
		return;
	}

	auto  swapchain_extent = swapchain_render_target.get_extent();
	auto &swapchain_views  = swapchain_render_target.get_views();

	// Produce the image for presentation from realtime rendering or dataset replay.
	const vkb::core::ImageView *display_source = mode == Mode::Realtime ?
	                                                 draw_realtime(command_buffer) :
	                                                 draw_dataset(command_buffer, swapchain_frame_index);
	if (!display_source)
		return;

	// Acquire the swapchain image for postprocess and GUI color writes.
	{
		vkb::ImageMemoryBarrier barrier{};
		barrier.old_layout      = VK_IMAGE_LAYOUT_UNDEFINED;
		barrier.new_layout      = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
		barrier.dst_access_mask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		barrier.src_stage_mask  = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
		barrier.dst_stage_mask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		command_buffer.image_memory_barrier(swapchain_views[0], barrier);
		swapchain_render_target.set_layout(0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
	}

	// Display
	VkViewport viewport{};
	viewport.width  = static_cast<float>(swapchain_extent.width);
	viewport.height = static_cast<float>(swapchain_extent.height);
	command_buffer.set_viewport(0, {viewport});

	VkRect2D scissor{};
	scissor.extent = swapchain_extent;
	command_buffer.set_scissor(0, {scissor});

	auto &postprocessing_pass    = postprocessing_pipeline->get_pass(0);
	auto &postprocessing_subpass = postprocessing_pass.get_subpass(0);
	postprocessing_subpass.bind_sampled_image("color_sampler", vkb::core::SampledImage(*display_source));

	postprocessing_pipeline->draw(command_buffer, swapchain_render_target);

	// GUI
	if (has_gui())
		get_gui().draw(command_buffer);

	command_buffer.end_render_pass();

	// Postprocess/GUI color writes must finish before the presentation engine reads
	// the completed swapchain image.
	{
		vkb::ImageMemoryBarrier barrier{};
		barrier.old_layout      = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
		barrier.new_layout      = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
		barrier.src_access_mask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		barrier.src_stage_mask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		barrier.dst_stage_mask  = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
		command_buffer.image_memory_barrier(swapchain_views[0], barrier);
		swapchain_render_target.set_layout(0, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
	}
}

/*******************************
 * NSS API
 *******************************/

static void nss_message_callback(uint32_t type, const char *message)
{
	if (type == FFX_API_MESSAGE_TYPE_ERROR)
	{
		LOGE("[NSS] {}", message);
	}
	else if (type == FFX_API_MESSAGE_TYPE_WARNING)
	{
		LOGW("[NSS] {}", message);
	}
	else
	{
		LOGI("[NSS] {}", message);
	}
}

void NSSSample::create_nss_context(VkExtent2D display_size, VkExtent2D render_size)
{
	// Check if NSS is force-disabled by argument
	if (nss_force_disabled)
	{
		LOGI("NSS disabled by NSS_ENABLE=0, skipping context creation");
		return;
	}
#if defined(_HPP_VULKAN_LIBRARY)
	// Get Vulkan function pointers for FFX backend
	static vk::detail::DynamicLoader dl(_HPP_VULKAN_LIBRARY);
#else
	static vk::detail::DynamicLoader dl;
#endif
	PFN_vkGetInstanceProcAddr get_instance_proc_address =
	    dl.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");

	PFN_vkGetDeviceProcAddr get_device_proc_address =
	    reinterpret_cast<PFN_vkGetDeviceProcAddr>(
	        get_instance_proc_address(get_instance().get_handle(), "vkGetDeviceProcAddr"));

	// Configure NSS context flags
	if (!has_nss_flags_override)
	{
		nss_context_flags =
		    FFX_API_NSS_CONTEXT_FLAG_QUANTIZED |        // Use quantized model (required)
		    frag_or_comp_config.context_flags;        // PRE/POST process compute flags (0 for fragment)
		if (nss_ldr_mode)
			nss_context_flags |= FFX_API_NSS_CONTEXT_FLAG_LOW_DYNAMIC_RANGE;

		if (nss_manage_history)
			nss_context_flags |= FFX_API_NSS_CONTEXT_FLAG_MANAGE_HISTORY;

		if (mode == Mode::Realtime || (dataset_loader && dataset_loader->uses_inverted_depth()))
			nss_context_flags |= FFX_API_NSS_CONTEXT_FLAG_DEPTH_INVERTED;
		if (dataset_loader && dataset_loader->uses_infinite_far())
			nss_context_flags |= FFX_API_NSS_CONTEXT_FLAG_DEPTH_INFINITE;
	}

	// Create Vulkan backend + NSS context descriptor
	ffx::CreateBackendVKDesc backend_description{};
	backend_description.header.type           = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK;
	backend_description.vkInstance            = get_instance().get_handle();
	backend_description.vkDevice              = get_device().get_handle();
	backend_description.vkPhysicalDevice      = get_device().get_gpu().get_handle();
	backend_description.vkGetInstanceProcAddr = get_instance_proc_address;
	backend_description.vkDeviceProcAddr      = get_device_proc_address;

	ffx::CreateContextDescNss context_description{};
	context_description.header.type           = FFX_API_CREATE_CONTEXT_DESC_TYPE_NSS;
	context_description.maxRenderSize.width   = render_size.width;
	context_description.maxRenderSize.height  = render_size.height;
	context_description.maxUpscaleSize.width  = display_size.width;
	context_description.maxUpscaleSize.height = display_size.height;
	context_description.flags                 = nss_context_flags;
	context_description.fpMessage             = nss_message_callback;
	context_description.qualityMode           = nss_quality_mode;

	// Create NSS context
	ffx::ReturnCode result = ffx::CreateContext(nss_context, nullptr, context_description, backend_description);
	if (result != ffx::ReturnCode::Ok)
	{
		LOGW("Failed to create NSS context! Error code: {}, NSS features will be disabled", static_cast<uint32_t>(result));
		return;
	}

	LOGI("NSS Context created successfully!");
	log_memory_usage("context_created", true);
}

bool NSSSample::has_nss_context() const
{
	return nss_context != nullptr;
}

// Wrap a VkImage as an FfxApiResource for NSS input/output
FfxApiResource NSSSample::wrap_resource(const vkb::core::Image &image, FfxApiResourceState state, uint32_t additional_usages)
{
	VkImageCreateInfo create_info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
	create_info.imageType   = image.get_type();
	create_info.format      = image.get_format();
	create_info.extent      = image.get_extent();
	create_info.mipLevels   = image.get_subresource().mipLevel;
	create_info.arrayLayers = image.get_array_layer_count();
	create_info.usage       = image.get_usage();
	create_info.flags       = 0;

	const auto resource_description = ffxApiGetImageResourceDescriptionVK(image.get_handle(), create_info, additional_usages);
	return ffxApiGetResourceVK(reinterpret_cast<void *>(image.get_handle()), resource_description, static_cast<uint32_t>(state));
}

const vkb::core::ImageView *NSSSample::dispatch_nss(vkb::core::CommandBufferC &command_buffer, FrameInputs &inputs,
                                                    const NSSFrameParams &frame_params, bool reset,
                                                    const vkb::core::Image *ldr_color_input)
{
	const size_t current_index   = scene_frame_index % resource_count;
	const size_t previous_index  = (current_index + resource_count - 1) % resource_count;
	auto        &current         = nss_outputs[current_index];
	auto        &previous        = nss_outputs[previous_index];
	const bool   draw_debug_view = display_mode == DisplayMode::NSSDebugView;

	initialize_external_history(command_buffer, previous);
	// Transition this dispatch's output and optional debug output from their
	// current layouts to the compute or fragment output-write layout.
	prepare_outputs(command_buffer, current, draw_debug_view);

	ffxApiDispatchDescNss dispatch_description{};
	dispatch_description.header.type            = FFX_API_DISPATCH_DESC_TYPE_NSS;
	dispatch_description.commandList            = reinterpret_cast<void *>(command_buffer.get_handle());
	dispatch_description.color                  = wrap_resource(ldr_color_input ? *ldr_color_input : *inputs.color, frag_or_comp_config.input_read_state);
	dispatch_description.depth                  = wrap_resource(*inputs.depth, frag_or_comp_config.input_read_state);
	dispatch_description.motionVectors          = wrap_resource(*inputs.motion_vectors, frag_or_comp_config.input_read_state);
	dispatch_description.output                 = wrap_resource(*current.image, frag_or_comp_config.output_write_state);
	dispatch_description.jitterOffset           = {frame_params.jitter_x, frame_params.jitter_y};
	dispatch_description.renderSize             = {resolution_config.render_extent.width, resolution_config.render_extent.height};
	dispatch_description.upscaleSize            = {resolution_config.display_extent.width, resolution_config.display_extent.height};
	dispatch_description.motionVectorScale      = {frame_params.motion_vector_scale_x, frame_params.motion_vector_scale_y};
	dispatch_description.cameraFovAngleVertical = frame_params.camera_fov_angle_vertical;
	dispatch_description.cameraNear             = frame_params.camera_near;
	dispatch_description.cameraFar              = frame_params.camera_far;
	dispatch_description.frameTimeDelta         = frame_params.frame_time_delta_ms;
	dispatch_description.exposure               = frame_params.exposure;
	dispatch_description.reset                  = reset;
	dispatch_description.flags                  = nss_debug_checking ? FFX_API_NSS_DISPATCH_FLAG_ENABLE_DEBUG_CHECKING : 0;
	if (!nss_manage_history)
		dispatch_description.outputTm1 = wrap_resource(*previous.image, frag_or_comp_config.input_read_state);
	if (draw_debug_view)
	{
		dispatch_description.flags |= FFX_API_NSS_DISPATCH_FLAG_DRAW_DEBUG_VIEW;
		dispatch_description.debugViews    = wrap_resource(*nss_debug_output.image, frag_or_comp_config.output_write_state);
		dispatch_description.debugViewMode = nss_debug_view_mode;
	}

	const ffx::ReturnCode result = ffx::Dispatch(nss_context, dispatch_description);
	if (result != ffx::ReturnCode::Ok)
	{
		LOGE("NSS dispatch failed with error code: {}", static_cast<uint32_t>(result));
		close();
		return nullptr;
	}

	const bool first_dispatch = !nss_has_successful_dispatch;
	log_memory_usage(first_dispatch ? "first_dispatch" : "dispatch", first_dispatch);
	nss_has_successful_dispatch = true;

	// Publish the written output to readback or shader consumers.
	finish_outputs(command_buffer, current, draw_debug_view);

	return draw_debug_view ? nss_debug_output.view.get() : current.view.get();
}

void NSSSample::log_memory_usage(const char *event, bool force_log)
{
	FfxApiEffectMemoryUsage            vram_usage{};
	ffx::QueryDescNssGetGPUMemoryUsage query{};
	query.gpuMemoryUsageNss = &vram_usage;

	const auto query_result = ffx::Query(nss_context, query);
	if (query_result == ffx::ReturnCode::Ok)
	{
		if (memory_usage_tracker.update_gpu_usage(vram_usage, force_log))
		{
			LOGI("NSS_VRAM_USAGE(bytes) event={} total_usage={} texture_usage={} buffer_usage={} tensor_usage={} data_graph_usage={} aliasable_usage={}",
			     event,
			     vram_usage.totalUsageInBytes,
			     vram_usage.textureUsageInBytes,
			     vram_usage.bufferUsageInBytes,
			     vram_usage.tensorUsageInBytes,
			     vram_usage.dataGraphUsageInBytes,
			     vram_usage.aliasableUsageInBytes);
		}
	}
	else
	{
		LOGW("NSS VRAM query failed (event={}): {}", event, static_cast<uint32_t>(query_result));
	}

#ifdef FFX_ENABLE_CPU_HEAP_TRACKING
	FfxApiEffectCpuMemoryUsage         heap_usage{};
	ffx::QueryDescNssGetCPUMemoryUsage heap_query{};
	heap_query.cpuMemoryUsageNss = &heap_usage;

	const auto heap_query_result = ffx::Query(nss_context, heap_query);
	if (heap_query_result == ffx::ReturnCode::Ok)
	{
		if (memory_usage_tracker.update_cpu_usage(heap_usage, force_log))
		{
			LOGI("NSS_HEAP_USAGE(bytes) event={} current_usage={} peak_usage={}",
			     event,
			     heap_usage.currentUsageInBytes,
			     heap_usage.peakUsageInBytes);
		}
	}
	else
	{
		LOGW("NSS heap query failed (event={}): {}", event, static_cast<uint32_t>(heap_query_result));
	}
#endif
}

/*******************************
 * gui draw
 *******************************/
void NSSSample::draw_gui()
{
	get_gui().show_options_window(
	    /* body = */ [this]() {
		    ImGui::Text(" Neural Super Sampling Sample");
		    ImGui::Text("Mode: %s  NSS: %s  Path: %s",
		                mode == Mode::Realtime ? "Realtime" : "Dataset",
		                has_nss_context() ? "Enabled" : "Disabled",
		                use_compute_path ? "Compute" : "Fragment");
		    ImGui::Separator();
		    ImGui::TextUnformatted("Display:");
		    ImGui::SameLine();
		    int display_mode_int = static_cast<int>(display_mode);

		    ImGui::RadioButton("Final Output", &display_mode_int, static_cast<int>(DisplayMode::FinalOutput));
		    ImGui::SameLine();
		    ImGui::RadioButton("NSS Debug", &display_mode_int, static_cast<int>(DisplayMode::NSSDebugView));
		    display_mode = static_cast<DisplayMode>(display_mode_int);

		    if (display_mode == DisplayMode::NSSDebugView)
		    {
			    int debug_view_mode = static_cast<int>(nss_debug_view_mode);
			    ImGui::SetNextItemWidth(-FLT_MIN);
			    if (ImGui::Combo("Debug View Mode", &debug_view_mode, kNssDebugViewModeLabels, IM_ARRAYSIZE(kNssDebugViewModeLabels), IM_ARRAYSIZE(kNssDebugViewModeLabels)))
			    {
				    nss_debug_view_mode = static_cast<uint32_t>(debug_view_mode);
			    }
		    }

		    ImGui::Separator();
		    if (mode == Mode::Realtime && temporal_subpass)
		    {
			    float jitter_scale_ui = temporal_subpass->get_jitter_scale();
			    ImGui::SliderFloat("Jitter Scale", &jitter_scale_ui, 0.0f, 1.0f);
			    temporal_subpass->set_jitter_scale(jitter_scale_ui);

			    glm::vec2 jitter_pixels = temporal_subpass->get_current_jitter_pixels();
			    ImGui::Text("Current Jitter: (%.3f, %.3f) pixels", jitter_pixels.x, jitter_pixels.y);
		    }
		    else
		    {
			    const uint64_t total_frames = dataset_loader->total_frames();
			    ImGui::Text("Frame: %llu / %llu  Sequence: %s",
			                static_cast<unsigned long long>(scene_frame_index),
			                static_cast<unsigned long long>(total_frames),
			                dataset_sequence.c_str());
		    }
		    ImGui::Text("Save EXR: %s  History: %s-managed",
		                save_exr_enabled ? "Yes" : "No",
		                nss_manage_history ? "SDK" : "Application");
		    if (save_exr_enabled)
		    {
			    ImGui::TextWrapped("Output Dir: %s", save_exr_output_dir.c_str());
		    }
		    ImGui::Text("Scale: %.3fx  Quality: %d  Flags: 0x%08X",
		                nss_scale_factor, nss_quality_mode, nss_context_flags);
		    if (mode == Mode::Realtime)
		    {
			    ImGui::Text("Render: %ux%u  Display: %ux%u  LOD: %.2f",
			                resolution_config.render_extent.width, resolution_config.render_extent.height,
			                resolution_config.display_extent.width, resolution_config.display_extent.height,
			                resolution_config.mipmap_lod_bias);
		    }
		    else
		    {
			    ImGui::Text("Render: %ux%u  Display: %ux%u  LOD: N/A",
			                resolution_config.render_extent.width, resolution_config.render_extent.height,
			                resolution_config.display_extent.width, resolution_config.display_extent.height);
		    }
	    },
	    /* lines = */ 10);
}

/*******************************
 * barrier helpers
 *******************************/
void NSSSample::prepare_render_inputs(vkb::core::CommandBufferC &command_buffer, FrameInputs &inputs)
{
	// Prepare the scene inputs for rendering:
	// - Color and motion: UNDEFINED (first use) or SHADER_READ_ONLY_OPTIMAL (reuse)
	//   -> COLOR_ATTACHMENT_OPTIMAL.
	// - Depth: UNDEFINED (first use) or SHADER_READ_ONLY_OPTIMAL (reuse)
	//   -> DEPTH_STENCIL_ATTACHMENT_OPTIMAL.
	// On reuse, the previous consumer is NSS, or native display when NSS is disabled.
	const VkPipelineStageFlags input_read_stage = has_nss_context() ? frag_or_comp_config.input_read_stage :
	                                                                  VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
	vkb::ImageMemoryBarrier    barrier{};
	barrier.old_layout = inputs.render_target->get_layout(static_cast<uint32_t>(Attachments::Color));
	if (barrier.old_layout == VK_IMAGE_LAYOUT_UNDEFINED)
	{
		barrier.src_access_mask = 0;
		barrier.src_stage_mask  = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
	}
	else
	{
		barrier.old_layout      = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		barrier.src_access_mask = VK_ACCESS_SHADER_READ_BIT;
		barrier.src_stage_mask  = input_read_stage;
	}

	barrier.new_layout      = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
	barrier.dst_access_mask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
	barrier.dst_stage_mask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	command_buffer.image_memory_barrier(*inputs.render_target, static_cast<uint32_t>(Attachments::Color), barrier);
	command_buffer.image_memory_barrier(*inputs.render_target, static_cast<uint32_t>(Attachments::Velocity), barrier);

	barrier.new_layout      = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
	barrier.dst_access_mask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
	barrier.dst_stage_mask  = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
	command_buffer.image_memory_barrier(*inputs.render_target, static_cast<uint32_t>(Attachments::Depth), barrier);
}

void NSSSample::finish_render_inputs(vkb::core::CommandBufferC &command_buffer, FrameInputs &inputs)
{
	// Publish the rendered scene inputs:
	// - Color and motion: COLOR_ATTACHMENT_OPTIMAL -> SHADER_READ_ONLY_OPTIMAL.
	// - Depth: DEPTH_STENCIL_ATTACHMENT_OPTIMAL -> SHADER_READ_ONLY_OPTIMAL.
	// The producers are scene attachment writes. The next consumer is NSS, or native
	// display when NSS is disabled. Reuse reads were already ordered before rendering.
	const VkPipelineStageFlags input_read_stage = has_nss_context() ? frag_or_comp_config.input_read_stage :
	                                                                  VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;

	vkb::ImageMemoryBarrier barrier{};
	barrier.old_layout      = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
	barrier.new_layout      = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	barrier.src_access_mask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
	barrier.dst_access_mask = VK_ACCESS_SHADER_READ_BIT;
	barrier.src_stage_mask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	barrier.dst_stage_mask  = input_read_stage;
	command_buffer.image_memory_barrier(*inputs.render_target, static_cast<uint32_t>(Attachments::Color), barrier);
	command_buffer.image_memory_barrier(*inputs.render_target, static_cast<uint32_t>(Attachments::Velocity), barrier);

	barrier.old_layout      = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
	barrier.src_access_mask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
	barrier.src_stage_mask  = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
	command_buffer.image_memory_barrier(*inputs.render_target, static_cast<uint32_t>(Attachments::Depth), barrier);
}

void NSSSample::prepare_upload_inputs(vkb::core::CommandBufferC &command_buffer, FrameInputs &inputs)
{
	// Prepare all dataset inputs for upload:
	// UNDEFINED (first use) or SHADER_READ_ONLY_OPTIMAL (reuse)
	// -> TRANSFER_DST_OPTIMAL.
	// On reuse, the previous consumer is NSS, or native display when NSS is disabled.
	// The next producers are buffer-to-image writes for color, depth, and motion.
	const VkPipelineStageFlags input_read_stage = has_nss_context() ? frag_or_comp_config.input_read_stage :
	                                                                  VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
	vkb::ImageMemoryBarrier    barrier{};
	barrier.old_layout = inputs.render_target->get_layout(static_cast<uint32_t>(Attachments::Color));
	if (barrier.old_layout == VK_IMAGE_LAYOUT_UNDEFINED)
	{
		barrier.src_access_mask = 0;
		barrier.src_stage_mask  = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
	}
	else
	{
		barrier.old_layout      = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		barrier.src_access_mask = VK_ACCESS_SHADER_READ_BIT;
		barrier.src_stage_mask  = input_read_stage;
	}

	barrier.new_layout      = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
	barrier.dst_access_mask = VK_ACCESS_TRANSFER_WRITE_BIT;
	barrier.dst_stage_mask  = VK_PIPELINE_STAGE_TRANSFER_BIT;

	command_buffer.image_memory_barrier(*inputs.render_target, static_cast<uint32_t>(Attachments::Color), barrier);
	command_buffer.image_memory_barrier(*inputs.render_target, static_cast<uint32_t>(Attachments::Depth), barrier);
	command_buffer.image_memory_barrier(*inputs.render_target, static_cast<uint32_t>(Attachments::Velocity), barrier);
}

void NSSSample::finish_upload_inputs(vkb::core::CommandBufferC &command_buffer, FrameInputs &inputs)
{
	// Publish all uploaded dataset inputs:
	// TRANSFER_DST_OPTIMAL -> SHADER_READ_ONLY_OPTIMAL.
	// The producers are buffer-to-image transfer writes. The next consumer is NSS,
	// or native display when NSS is disabled.
	const VkPipelineStageFlags input_read_stage = has_nss_context() ? frag_or_comp_config.input_read_stage :
	                                                                  VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;

	vkb::ImageMemoryBarrier barrier{};
	barrier.old_layout      = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
	barrier.new_layout      = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	barrier.src_access_mask = VK_ACCESS_TRANSFER_WRITE_BIT;
	barrier.dst_access_mask = VK_ACCESS_SHADER_READ_BIT;
	barrier.src_stage_mask  = VK_PIPELINE_STAGE_TRANSFER_BIT;
	barrier.dst_stage_mask  = input_read_stage;

	command_buffer.image_memory_barrier(*inputs.render_target, static_cast<uint32_t>(Attachments::Color), barrier);
	command_buffer.image_memory_barrier(*inputs.render_target, static_cast<uint32_t>(Attachments::Depth), barrier);
	command_buffer.image_memory_barrier(*inputs.render_target, static_cast<uint32_t>(Attachments::Velocity), barrier);
}

void NSSSample::initialize_external_history(vkb::core::CommandBufferC &command_buffer, ImageResource &previous)
{
	if (nss_manage_history || scene_frame_index > 0)
		return;

	// Initialize the first application-managed history image through two transitions:
	// UNDEFINED -> TRANSFER_DST_OPTIMAL for the clear, then
	// TRANSFER_DST_OPTIMAL -> SHADER_READ_ONLY_OPTIMAL for the first NSS history
	// read. The reset dispatch still samples history before masking it invalid, so
	// a cleared value avoids propagating undefined floating-point data.
	VkImageSubresourceRange range{};
	range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	range.levelCount = 1;
	range.layerCount = 1;
	VkClearColorValue clear_color{};

	vkb::ImageMemoryBarrier barrier{};
	barrier.old_layout      = VK_IMAGE_LAYOUT_UNDEFINED;
	barrier.new_layout      = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
	barrier.src_access_mask = 0;
	barrier.dst_access_mask = VK_ACCESS_TRANSFER_WRITE_BIT;
	barrier.src_stage_mask  = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
	barrier.dst_stage_mask  = VK_PIPELINE_STAGE_TRANSFER_BIT;
	command_buffer.image_memory_barrier(*previous.view, barrier);
	vkCmdClearColorImage(command_buffer.get_handle(), previous.image->get_handle(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clear_color, 1, &range);

	barrier.old_layout      = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
	barrier.new_layout      = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	barrier.src_access_mask = VK_ACCESS_TRANSFER_WRITE_BIT;
	barrier.dst_access_mask = VK_ACCESS_SHADER_READ_BIT;
	barrier.src_stage_mask  = VK_PIPELINE_STAGE_TRANSFER_BIT;
	barrier.dst_stage_mask  = frag_or_comp_config.input_read_stage;
	command_buffer.image_memory_barrier(*previous.view, barrier);
	previous.initialized = true;
}

void NSSSample::prepare_outputs(vkb::core::CommandBufferC &command_buffer, ImageResource &current, bool include_debug_output)
{
	// Prepare each output written by this dispatch. An image starts in UNDEFINED;
	// after its first dispatch it is left in SHADER_READ_ONLY_OPTIMAL for history,
	// display, or optional readback.
	vkb::ImageMemoryBarrier barrier{};
	barrier.new_layout      = frag_or_comp_config.output_write_layout;
	barrier.dst_access_mask = frag_or_comp_config.output_write_access;
	barrier.dst_stage_mask  = frag_or_comp_config.output_write_stage;
	auto transition         = [&](ImageResource &output, bool is_history_output) {
        if (!output.initialized)
        {
            barrier.old_layout      = VK_IMAGE_LAYOUT_UNDEFINED;
            barrier.src_access_mask = 0;
            barrier.src_stage_mask  = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        }
        else
        {
            barrier.old_layout      = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            barrier.src_access_mask = VK_ACCESS_SHADER_READ_BIT;
            barrier.src_stage_mask  = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
            if (!nss_manage_history && is_history_output)
                barrier.src_stage_mask |= frag_or_comp_config.input_read_stage;
        }
        command_buffer.image_memory_barrier(*output.view, barrier);
	};

	transition(current, true);
	if (include_debug_output)
		transition(nss_debug_output, false);
}

void NSSSample::finish_outputs(vkb::core::CommandBufferC &command_buffer, ImageResource &current, bool include_debug_output)
{
	// Publish each output written by this dispatch. Dataset saving consumes the
	// regular output through a transfer first; all other paths consume outputs as
	// sampled images.
	vkb::ImageMemoryBarrier barrier{};
	barrier.old_layout      = frag_or_comp_config.output_write_layout;
	barrier.src_access_mask = frag_or_comp_config.output_write_access;
	barrier.src_stage_mask  = frag_or_comp_config.output_write_stage;
	if (save_exr_enabled)
	{
		barrier.new_layout      = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
		barrier.dst_access_mask = VK_ACCESS_TRANSFER_READ_BIT;
		barrier.dst_stage_mask  = VK_PIPELINE_STAGE_TRANSFER_BIT;
	}
	else
	{
		barrier.new_layout      = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		barrier.dst_access_mask = VK_ACCESS_SHADER_READ_BIT;
		barrier.dst_stage_mask  = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
		if (!nss_manage_history)
			barrier.dst_stage_mask |= frag_or_comp_config.input_read_stage;
	}
	command_buffer.image_memory_barrier(*current.view, barrier);
	current.initialized = true;
	if (include_debug_output)
	{
		barrier.new_layout      = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		barrier.dst_access_mask = VK_ACCESS_SHADER_READ_BIT;
		barrier.dst_stage_mask  = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
		command_buffer.image_memory_barrier(*nss_debug_output.view, barrier);
		nss_debug_output.initialized = true;
	}
}

/*******************************
 * readback buffers
 *******************************/
void NSSSample::create_readback_buffers(VkExtent2D extent, size_t count)
{
	size_t     buffer_size  = 0;
	const bool valid_format = vkb::sample::exr::vk_format_buffer_size(extent, nss_color_format, buffer_size);
	assert(valid_format);

	readback_slots.resize(count);
	vkb::core::BufferBuilderC builder(buffer_size);
	builder.with_usage(VK_BUFFER_USAGE_TRANSFER_DST_BIT)
	    .with_vma_usage(VMA_MEMORY_USAGE_GPU_TO_CPU)
	    .with_vma_flags(VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT);
	for (ReadbackSlot &slot : readback_slots)
		slot.buffer = builder.build_unique(get_device());

	LOGI("Created {} NSS EXR readback buffers ({} bytes each)", count, buffer_size);
}

void NSSSample::copy_to_readback(vkb::core::CommandBufferC &command_buffer,
                                 ImageResource             &source,
                                 vkb::core::BufferC        &destination)
{
	auto &source_image = *source.image;

	VkBufferImageCopy region{};
	region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	region.imageSubresource.layerCount = 1;
	region.imageExtent                 = source_image.get_extent();
	vkCmdCopyImageToBuffer(command_buffer.get_handle(), source_image.get_handle(),
	                       VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
	                       destination.get_handle(), 1, &region);

	vkb::BufferMemoryBarrier buffer_barrier{};
	buffer_barrier.src_access_mask = VK_ACCESS_TRANSFER_WRITE_BIT;
	buffer_barrier.dst_access_mask = VK_ACCESS_HOST_READ_BIT;
	buffer_barrier.src_stage_mask  = VK_PIPELINE_STAGE_TRANSFER_BIT;
	buffer_barrier.dst_stage_mask  = VK_PIPELINE_STAGE_HOST_BIT;
	command_buffer.buffer_memory_barrier(destination, 0, destination.get_size(), buffer_barrier);

	vkb::ImageMemoryBarrier barrier{};
	barrier.old_layout      = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
	barrier.new_layout      = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	barrier.src_access_mask = VK_ACCESS_TRANSFER_READ_BIT;
	barrier.dst_access_mask = VK_ACCESS_SHADER_READ_BIT;
	barrier.src_stage_mask  = VK_PIPELINE_STAGE_TRANSFER_BIT;
	barrier.dst_stage_mask  = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
	if (!nss_manage_history)
		barrier.dst_stage_mask |= frag_or_comp_config.input_read_stage;
	command_buffer.image_memory_barrier(*source.view, barrier);
}

bool NSSSample::save_completed_readback(ReadbackSlot &slot, VkExtent2D extent)
{
	if (!slot.need_save)
		return true;

	slot.need_save = false;

	char filename[32];
	snprintf(filename, sizeof(filename), "%04llu.exr", static_cast<unsigned long long>(slot.frame_number));
	const std::filesystem::path path = std::filesystem::path(save_exr_output_dir) / filename;
	slot.buffer->invalidate();
	return vkb::sample::exr::save_vk_format_to_exr(path.string().c_str(), nss_color_format, slot.buffer->get_data(),
	                                               static_cast<size_t>(slot.buffer->get_size()), extent);
}

std::unique_ptr<vkb::VulkanSampleC> create_nss()
{
	return std::make_unique<NSSSample>();
}
