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

#include "nfru.h"

#include <cassert>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <vector>

#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>

// tinyexr — header-only EXR file I/O library (declarations only).
// Function bodies are compiled inside the FFX backend library.
#include <tinyexr/tinyexr.h>

#include "common/app_argument_reader.h"
#include "core/image.h"
#include "core/image_view.h"

#include "rendering/render_pipeline.h"
#include "rendering/subpass.h"
#include "rendering/subpasses/temporal_forward_subpass.h"
#include "rendering/postprocessing_pipeline.h"
#include "rendering/postprocessing_renderpass.h"

#include "scene_graph/components/perspective_camera.h"
#include "scene_graph/components/transform.h"
#include "scene_graph/node.h"
#include "common/utils.h"

// ──────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ──────────────────────────────────────────────────────────────────────────────
static void OnNFRUMessage(uint32_t type, const char *message)
{
	if (type == FFX_API_MESSAGE_TYPE_ERROR)
	{
		LOGE("[NFRU] {}", message);
	}
	else if (type == FFX_API_MESSAGE_TYPE_WARNING)
	{
		LOGW("[NFRU] {}", message);
	}
	else
	{
		LOGI("[NFRU] {}", message);
	}
}

static void mkdirp(const std::string &path)
{
	std::error_code ec;
	std::filesystem::create_directories(path, ec);
	if (ec)
		LOGW("mkdirp({}): {}", path, ec.message());
}

/// Save R8G8B8A8_UNORM pixel buffer as floating-point EXR.
static void save_r8g8b8a8_as_exr(const void *data, VkExtent2D extent, const std::string &path)
{
	const uint32_t num_pixels = extent.width * extent.height;
	const uint8_t *src        = static_cast<const uint8_t *>(data);

	std::vector<float> rgba(num_pixels * 4);
	for (uint32_t i = 0; i < num_pixels * 4; ++i)
		rgba[i] = src[i] / 255.0f;

	const char *err = nullptr;
	int         ret = SaveEXR(rgba.data(),
	                          static_cast<int>(extent.width),
	                          static_cast<int>(extent.height),
	                          4, 0, path.c_str(), &err);
	if (ret != TINYEXR_SUCCESS)
	{
		LOGE("Failed to save EXR '{}': {}", path, err ? err : "unknown");
		FreeEXRErrorMessage(err);
	}
	else
	{
		LOGI("Saved: {}", path);
	}
}

// ──────────────────────────────────────────────────────────────────────────────
// Constructor / Destructor
// ──────────────────────────────────────────────────────────────────────────────

NFRUSample::NFRUSample()
{
	set_api_version(VK_API_VERSION_1_4);

	add_device_extension("VK_KHR_dynamic_rendering", true);
	add_device_extension("VK_KHR_maintenance5", true);
	add_device_extension("VK_KHR_deferred_host_operations", true);
	add_device_extension("VK_ARM_data_graph", true);
	add_device_extension("VK_ARM_tensors", true);
	add_device_extension("VK_ARM_data_graph_optical_flow", true);
}

NFRUSample::~NFRUSample()
{
	if (get_device().get_handle() != VK_NULL_HANDLE)
		vkDeviceWaitIdle(get_device().get_handle());

	if (nfru_initialized && nfru_context)
	{
		ffx::DestroyContext(nfru_context);
		nfru_context = nullptr;
	}

	auto destroy_rb = [&](ReadbackSlot &slot) {
		if (slot.buffer.mapped)
			vkUnmapMemory(get_device().get_handle(), slot.buffer.memory);
		if (slot.buffer.buffer != VK_NULL_HANDLE)
			vkDestroyBuffer(get_device().get_handle(), slot.buffer.buffer, nullptr);
		if (slot.buffer.memory != VK_NULL_HANDLE)
			vkFreeMemory(get_device().get_handle(), slot.buffer.memory, nullptr);
	};
	for (auto &s : readback_slots)
		destroy_rb(s);
}

void NFRUSample::request_gpu_features(vkb::core::PhysicalDeviceC &gpu)
{
	vkb::VulkanSampleC::request_gpu_features(gpu);

	REQUEST_REQUIRED_FEATURE(gpu, VkPhysicalDeviceVulkan12Features, shaderInt8);
	REQUEST_REQUIRED_FEATURE(gpu, VkPhysicalDeviceVulkan13Features, synchronization2);

	REQUEST_OPTIONAL_FEATURE(gpu, VkPhysicalDeviceTensorFeaturesARM, shaderTensorAccess);
	REQUEST_OPTIONAL_FEATURE(gpu, VkPhysicalDeviceTensorFeaturesARM, tensors);
	REQUEST_OPTIONAL_FEATURE(gpu, VkPhysicalDeviceDataGraphFeaturesARM, dataGraph);
	REQUEST_OPTIONAL_FEATURE(gpu, VkPhysicalDeviceDataGraphOpticalFlowFeaturesARM, dataGraphOpticalFlow);
}

// ──────────────────────────────────────────────────────────────────────────────
// prepare
// ──────────────────────────────────────────────────────────────────────────────

bool NFRUSample::prepare(const vkb::ApplicationOptions &options)
{
	if (!vkb::VulkanSampleC::prepare(options))
	{
		return false;
	}

	// ── Command-line configuration (see `--app-arg KEY=VALUE`) ──
	//   NFRU_DATASET_DIR      : path; selects Dataset mode when present (else Realtime)
	//   NFRU_DATASET_SEQUENCE : dataset sequence id (default "0000")
	//   NFRU_SAVE_EXR         : 0 or 1 — save output frames as EXR
	//   NFRU_ORBIT_SPEED      : float in [0, 10] degrees/frame (realtime camera)
	//   NFRU_USE_FRAGMENT     : 0 (compute path) or 1 (fragment shader path)
	//   NFRU_DEBUG_VIEW       : 0 or 1 — enable NFRU debug view
	vkb::AppArgumentReader args{options.app_arguments};

	const std::string dataset_dir_arg = args.get_string("NFRU_DATASET_DIR");
	mode = dataset_dir_arg.empty() ? Mode::Realtime : Mode::Dataset;

	save_enabled             = args.get_bool("NFRU_SAVE_EXR", save_enabled);
	camera_degrees_per_frame = args.get_float("NFRU_ORBIT_SPEED", camera_degrees_per_frame, 0.0f, 10.0f);
	use_fragment_path        = args.get_bool("NFRU_USE_FRAGMENT", use_fragment_path);
	nfru_debug_view          = args.get_bool("NFRU_DEBUG_VIEW", nfru_debug_view);

#if defined(__ANDROID__)
	if (mode == Mode::Realtime)
	{
		if (save_enabled)
			LOGW("Android realtime: force disable EXR save to avoid storage permission issues.");
		save_enabled = false;
	}
#endif

	frag_or_comp_config = use_fragment_path
	    ? FragOrCompConfig{VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
	                  FFX_API_FG_CONTEXT_FLAG_ALL_STAGES_FRAGMENT | FFX_API_FG_CONTEXT_FLAG_MV_HINTS_FRAGMENT,
	                  VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
	                  VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
	                  VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
	                  FFX_API_RESOURCE_STATE_PIXEL_READ}
	    : FragOrCompConfig{VK_IMAGE_USAGE_STORAGE_BIT,
	                  0u,
	                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
	                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
	                  VK_ACCESS_SHADER_WRITE_BIT,
	                  FFX_API_RESOURCE_STATE_COMPUTE_READ};

	size_t swapchain_count = get_render_context().get_swapchain().get_images().size();
	render_extent = get_render_context().get_surface_extent();

	if (mode == Mode::Dataset)
	{
		// ── Dataset initialization ──
		const std::string dataset_dir      = dataset_dir_arg;
		const std::string dataset_sequence = args.get_string("NFRU_DATASET_SEQUENCE", "0000");

		dataset_loader = std::make_unique<DatasetLoader>();

		if (!dataset_loader->init(get_device().get_handle(),
		                   get_device().get_gpu().get_handle(),
		                   dataset_dir, dataset_sequence,
		                   static_cast<uint32_t>(swapchain_count)))
		{
			LOGE("Failed to initialize DatasetLoader");
			return false;
		}

		dataset_display_size = dataset_loader->get_display_size();
		dataset_render_size = dataset_loader->get_render_size();

		LOGI("NFRU Dataset: display={}x{}, render={}x{}",
		     dataset_display_size.width, dataset_display_size.height,
		     dataset_render_size.width, dataset_render_size.height);

		// Create per-swapchain input images — color at display_size, depth/mv at render_size
		create_scene_render_images(dataset_display_size, dataset_render_size, swapchain_count);

		// Create NFRU context — load first frame to get initial VP
		DatasetLoader::FrameParams first_params{};
		if (!dataset_loader->get_frame_params(0, first_params))
		{
			LOGE("Failed to read dataset frame 0 parameters");
			return false;
		}

		create_nfru_context(dataset_display_size, dataset_render_size, first_params.viewProjection);
		if (!nfru_initialized)
		{
			LOGE("Failed to create NFRU context for dataset replay");
			return false;
		}

		create_nfru_output_images(dataset_display_size, swapchain_count);

		// Dataset output goes into the dataset directory
		interp_output_dir = dataset_dir + "/result/nfru_x2/" + dataset_sequence;

		// Readback buffers (used only when save_enabled)
		create_readback_buffers(dataset_display_size, swapchain_count);
		mkdirp(interp_output_dir);
	}
	else
	{
		// ── Realtime initialization ──
		load_scene("scenes/sponza/Sponza01.gltf");

		LOGI("NFRU: render extent = {}x{}", render_extent.width, render_extent.height);

		// Create per-swapchain input images
		create_scene_render_images(render_extent, render_extent, swapchain_count);

		// Camera
		auto &cam_node = vkb::add_free_camera(get_scene(), "main_camera", render_extent);
		camera      = dynamic_cast<vkb::sg::PerspectiveCamera *>(
		    &cam_node.get_component<vkb::sg::Camera>());
		camera_node = &cam_node;

		{
			glm::vec3 eye    = {orbit_radius, orbit_height, 0.0f};
			glm::vec3 center = {0.0f, orbit_height, 0.0f};
			glm::mat4 current_view = glm::lookAt(eye, center, glm::vec3{0.0f, 1.0f, 0.0f});
			camera_node->get_transform().set_matrix(glm::inverse(current_view));
		}

		// Scene render pipeline
		{
			vkb::ShaderSource vert("temporal/jitter.vert.spv");
			vkb::ShaderSource frag("temporal/velocity.frag.spv");
			auto subpass = std::make_unique<vkb::TemporalForwardSubpass>(
			    get_render_context(), std::move(vert), std::move(frag),
			    get_scene(), *camera);
			temporal_subpass = subpass.get();

			subpass->set_output_attachments({
			    static_cast<int>(Att::Color),
			    static_cast<int>(Att::Velocity)});
			subpass->set_mipmap_lod_bias(0.0f);
			subpass->set_jitter_scale(0.0f);
			// Stride depends on save_enabled: 2 when saving EXR, 1 when only displaying NFRU output.
			subpass->set_velocity_stride(save_enabled ? 2 : 1);

			scene_pipeline = std::make_unique<vkb::RenderPipeline>();
			scene_pipeline->add_subpass(std::move(subpass));

			scene_pipeline->set_load_store({
			    {VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE},        // Depth
			    {VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE},        // Color
			    {VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE},        // Velocity
			});
			{
				VkClearValue depth_clear{};
				depth_clear.depthStencil = {0.0f, 0U};        // reversed depth: 0.0 = far
				VkClearValue color_clear{};
				color_clear.color = {0.0f, 0.0f, 0.0f, 1.0f}; // black sky
				VkClearValue vel_clear{};
				vel_clear.color = {0.0f, 0.0f, 0.0f, 0.0f};   // no motion
				scene_pipeline->set_clear_value({depth_clear, color_clear, vel_clear});
			}
		}

		// Create NFRU context
		{
			glm::mat4 vp = temporal_subpass->get_view_projection();
			create_nfru_context(render_extent, render_extent, glm::value_ptr(vp));
		}
		if (nfru_initialized)
			create_nfru_output_images(render_extent, swapchain_count);

		// Readback buffers (used only when save_enabled)
		create_readback_buffers(render_extent, swapchain_count);
#if !defined(__ANDROID__)
        mkdirp(gt_output_dir);
        mkdirp(interp_output_dir);
#endif
	}

	// ── Display pipeline ──
	{
		vkb::ShaderSource pp_vs("postprocessing.vert.spv");
		display_pipeline = std::make_unique<vkb::PostProcessingPipeline>(
		    get_render_context(), std::move(pp_vs));
		display_pipeline->add_pass().add_subpass(
		    vkb::ShaderSource("display.frag.spv"));
	}

	get_stats().request_stats({vkb::StatIndex::frame_times});
	create_gui(*window, &get_stats());

	return true;
}

void NFRUSample::update(float delta_time)
{
	last_delta_time = delta_time;

	// The orbit is frame-driven so adjacent source frames stay evenly spaced
	if (mode == Mode::Realtime)
	{
		const float yaw_deg = static_cast<float>(frame_count) * camera_degrees_per_frame;
		const float yaw_rad = glm::radians(std::fmod(yaw_deg, 360.0f));
		glm::vec3   eye     = {cosf(yaw_rad) * orbit_radius, orbit_height, sinf(yaw_rad) * orbit_radius};
		glm::vec3   center  = {0.0f, orbit_height, 0.0f};
		glm::mat4 current_view = glm::lookAt(eye, center, glm::vec3{0.0f, 1.0f, 0.0f});
		camera_node->get_transform().set_matrix(glm::inverse(current_view));
	}

	VulkanSample::update(delta_time);
}

// ──────────────────────────────────────────────────────────────────────────────
// Draw
// ──────────────────────────────────────────────────────────────────────────────

bool NFRUSample::showing_generated_frame() const
{
	if (!nfru_initialized)
		return false;
	if (mode == Mode::Dataset)
		return true;
	return !save_enabled || (frame_count % 2 == 0);
}

void NFRUSample::render_realtime(vkb::core::CommandBufferC &command_buffer, uint32_t swapchain_index)
{
	auto &scene_rt    = *render_targets[swapchain_index];
	auto &scene_views = scene_rt.get_views();

	const bool show_generated = showing_generated_frame();

	temporal_subpass->set_velocity_stride(save_enabled ? 2 : 1);

	// Transition color + velocity to COLOR_ATTACHMENT
	{
		vkb::ImageMemoryBarrier barrier{};
		barrier.old_layout      = VK_IMAGE_LAYOUT_UNDEFINED;
		barrier.new_layout      = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
		barrier.src_access_mask = 0;
		barrier.dst_access_mask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		barrier.src_stage_mask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		barrier.dst_stage_mask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		command_buffer.image_memory_barrier(scene_views[static_cast<int>(Att::Color)], barrier);
		scene_rt.set_layout(static_cast<int>(Att::Color), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
		command_buffer.image_memory_barrier(scene_views[static_cast<int>(Att::Velocity)], barrier);
		scene_rt.set_layout(static_cast<int>(Att::Velocity), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
	}

	// Transition depth to DEPTH_STENCIL_ATTACHMENT
	{
		vkb::ImageMemoryBarrier barrier{};
		barrier.old_layout      = VK_IMAGE_LAYOUT_UNDEFINED;
		barrier.new_layout      = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
		barrier.src_access_mask = 0;
		barrier.dst_access_mask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
		                          VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
		barrier.src_stage_mask  = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
		barrier.dst_stage_mask  = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
		                          VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
		command_buffer.image_memory_barrier(scene_views[static_cast<int>(Att::Depth)], barrier);
		scene_rt.set_layout(static_cast<int>(Att::Depth), VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
	}

	// Set viewport/scissor to render extent
	{
		VkViewport vp{};
		vp.width    = static_cast<float>(render_extent.width);
		vp.height   = static_cast<float>(render_extent.height);
		vp.minDepth = 0.0f;
		vp.maxDepth = 1.0f;
		command_buffer.set_viewport(0, {vp});

		VkRect2D scissor{};
		scissor.extent = render_extent;
		command_buffer.set_scissor(0, {scissor});
	}

	// Render scene
	scene_pipeline->draw(command_buffer, scene_rt);
	command_buffer.end_render_pass();

	// Transition scene outputs to SHADER_READ_ONLY for NFRU / display
	{
		vkb::ImageMemoryBarrier barrier{};
		barrier.old_layout      = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
		barrier.new_layout      = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		barrier.src_access_mask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		barrier.dst_access_mask = VK_ACCESS_SHADER_READ_BIT;
		barrier.src_stage_mask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		barrier.dst_stage_mask  = frag_or_comp_config.input_read_stage;
		command_buffer.image_memory_barrier(scene_views[static_cast<int>(Att::Color)], barrier);
		scene_rt.set_layout(static_cast<int>(Att::Color), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

		command_buffer.image_memory_barrier(scene_views[static_cast<int>(Att::Velocity)], barrier);
		scene_rt.set_layout(static_cast<int>(Att::Velocity), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
	}
	// Transition depth to SHADER_READ_ONLY
	{
		vkb::ImageMemoryBarrier barrier{};
		barrier.old_layout      = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
		barrier.new_layout      = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		barrier.src_access_mask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
		barrier.dst_access_mask = VK_ACCESS_SHADER_READ_BIT;
		barrier.src_stage_mask  = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
		barrier.dst_stage_mask  = frag_or_comp_config.input_read_stage;
		command_buffer.image_memory_barrier(scene_views[static_cast<int>(Att::Depth)], barrier);
		scene_rt.set_layout(static_cast<int>(Att::Depth), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
	}

	if (show_generated)
	{
		// NFRU-generated frame: dispatch interpolation and show its output.
		DatasetLoader::FrameParams params{};
		glm::mat4 vp_mat = temporal_subpass->get_view_projection();
		std::memcpy(params.viewProjection, glm::value_ptr(vp_mat), sizeof(params.viewProjection));
		auto jitter = temporal_subpass->get_current_jitter_pixels_scaled();
		params.jitterX                = jitter.x;
		params.jitterY                = jitter.y;
		params.cameraNear             = camera->get_near_plane();
		params.cameraFar              = camera->get_far_plane();
		params.cameraFovAngleVertical = camera->get_field_of_view();
		params.frameTimeDelta         = last_delta_time * 1000.0f;
		params.motionVectorScaleX     = static_cast<float>(render_extent.width);
		params.motionVectorScaleY     = static_cast<float>(render_extent.height);

		dispatch_nfru(command_buffer, swapchain_index, params, /*reset=*/ffx_frame_id == 0);
		display_source = nfru_output_views[swapchain_index].get();
	}
	else
	{
		// Reference frame: show the original rendered scene.
		display_source = &scene_views[static_cast<int>(Att::Color)];
	}

	// EXR save: generated (even) frames go to interp, reference (odd) frames go
	// to ground truth. Frame 0 is skipped
	if (save_enabled && frame_count >= 1)
	{
		ReadbackSlot &slot = readback_slots[swapchain_index];
		slot.need_save = true;
		if (show_generated)
		{
			copy_to_readback(command_buffer, *nfru_output_views[swapchain_index], slot.buffer);
			slot.frame_number = frame_count - 1;
			slot.output_dir   = interp_output_dir;
		}
		else
		{
			copy_to_readback(command_buffer, scene_views[static_cast<int>(Att::Color)], slot.buffer);
			slot.frame_number = frame_count;
			slot.output_dir   = gt_output_dir;
		}
	}

	++frame_count;
}

void NFRUSample::replay_dataset(vkb::core::CommandBufferC &command_buffer, uint32_t swapchain_index)
{
	if (current_frame >= dataset_loader->total_frames())
		return;

	DatasetLoader::FrameParams frame_params{};
	if (!dataset_loader->load_frame(current_frame, swapchain_index, frame_params))
	{
		LOGE("Failed to load dataset frame {}", current_frame);
		close();
		return;
	}

	// Upload the loaded EXR data (color/depth/mv) into the GPU input images.
	{
		vkb::ImageMemoryBarrier barrier{};
		barrier.old_layout      = VK_IMAGE_LAYOUT_UNDEFINED;
		barrier.new_layout      = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
		barrier.src_access_mask = 0;
		barrier.dst_access_mask = VK_ACCESS_TRANSFER_WRITE_BIT;
		barrier.src_stage_mask  = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
		barrier.dst_stage_mask  = VK_PIPELINE_STAGE_TRANSFER_BIT;

		vkb::core::ImageView color_iv(*color_images[swapchain_index], VK_IMAGE_VIEW_TYPE_2D);
		command_buffer.image_memory_barrier(color_iv, barrier);
		vkb::core::ImageView mv_iv(*mv_images[swapchain_index], VK_IMAGE_VIEW_TYPE_2D);
		command_buffer.image_memory_barrier(mv_iv, barrier);
		vkb::core::ImageView depth_iv(*depth_images[swapchain_index], VK_IMAGE_VIEW_TYPE_2D);
		command_buffer.image_memory_barrier(depth_iv, barrier);
	}

	{
		VkBufferImageCopy region{};
		region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		region.imageSubresource.layerCount = 1;
		region.imageExtent                 = {dataset_display_size.width, dataset_display_size.height, 1};
		vkCmdCopyBufferToImage(command_buffer.get_handle(),
		    dataset_loader->get_color_staging(swapchain_index),
		    color_images[swapchain_index]->get_handle(),
		    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
	}
	{
		VkBufferImageCopy region{};
		region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
		region.imageSubresource.layerCount = 1;
		region.imageExtent                 = {dataset_render_size.width, dataset_render_size.height, 1};
		vkCmdCopyBufferToImage(command_buffer.get_handle(),
		    dataset_loader->get_depth_staging(swapchain_index),
		    depth_images[swapchain_index]->get_handle(),
		    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
	}
	{
		VkBufferImageCopy region{};
		region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		region.imageSubresource.layerCount = 1;
		region.imageExtent                 = {dataset_render_size.width, dataset_render_size.height, 1};
		vkCmdCopyBufferToImage(command_buffer.get_handle(),
		    dataset_loader->get_mv_staging(swapchain_index),
		    mv_images[swapchain_index]->get_handle(),
		    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
	}

	// Transition input images to SHADER_READ_ONLY for NFRU.
	{
		vkb::ImageMemoryBarrier barrier{};
		barrier.old_layout      = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
		barrier.new_layout      = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		barrier.src_access_mask = VK_ACCESS_TRANSFER_WRITE_BIT;
		barrier.dst_access_mask = VK_ACCESS_SHADER_READ_BIT;
		barrier.src_stage_mask  = VK_PIPELINE_STAGE_TRANSFER_BIT;
		barrier.dst_stage_mask  = frag_or_comp_config.input_read_stage;

		vkb::core::ImageView color_iv(*color_images[swapchain_index], VK_IMAGE_VIEW_TYPE_2D);
		command_buffer.image_memory_barrier(color_iv, barrier);
		vkb::core::ImageView depth_iv(*depth_images[swapchain_index], VK_IMAGE_VIEW_TYPE_2D);
		command_buffer.image_memory_barrier(depth_iv, barrier);
		vkb::core::ImageView mv_iv(*mv_images[swapchain_index], VK_IMAGE_VIEW_TYPE_2D);
		command_buffer.image_memory_barrier(mv_iv, barrier);
	}

	dispatch_nfru(command_buffer, swapchain_index, frame_params, /*reset=*/ffx_frame_id == 0);
	display_source = nfru_output_views[swapchain_index].get();

	// Save the interpolated output. Frame 0 is the reset frame (skipped).
	if (save_enabled && current_frame >= 1)
	{
		ReadbackSlot &slot = readback_slots[swapchain_index];
		copy_to_readback(command_buffer, *nfru_output_views[swapchain_index], slot.buffer);
		slot.need_save    = true;
		slot.frame_number = dataset_loader->interpolated_source_frame(current_frame);
		slot.output_dir   = interp_output_dir;
	}

	++current_frame;
}

void NFRUSample::draw(vkb::core::CommandBufferC &command_buffer,
                      vkb::RenderTarget         &swapchain_render_target)
{
	uint32_t swapchain_index = get_render_context().get_active_frame_index();
	auto    &swapchain_views = swapchain_render_target.get_views();
	auto     swapchain_extent = swapchain_render_target.get_extent();

	// ── Flush need_save EXR saves (fence guarantees GPU→buffer copy is done) ──
	{
		VkExtent2D save_extent = (mode == Mode::Dataset) ? dataset_display_size : render_extent;
		flush_pending_save(readback_slots[swapchain_index], save_extent);
	}

	// ── Phase 0: Transition swapchain to COLOR_ATTACHMENT ──
	{
		vkb::ImageMemoryBarrier barrier{};
		barrier.old_layout      = VK_IMAGE_LAYOUT_UNDEFINED;
		barrier.new_layout      = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
		barrier.src_access_mask = 0;
		barrier.dst_access_mask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		barrier.src_stage_mask  = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
		barrier.dst_stage_mask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		command_buffer.image_memory_barrier(swapchain_views[0], barrier);
		swapchain_render_target.set_layout(0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
	}

	// ── Phase 1: Produce this frame (scene render or dataset upload) + NFRU ──
	if (mode == Mode::Realtime)
		render_realtime(command_buffer, swapchain_index);
	else
		replay_dataset(command_buffer, swapchain_index);

	// ── Phase 2: Display ──
	{
		VkViewport vp{};
		vp.width    = static_cast<float>(swapchain_extent.width);
		vp.height   = static_cast<float>(swapchain_extent.height);
		vp.minDepth = 0.0f;
		vp.maxDepth = 1.0f;
		command_buffer.set_viewport(0, {vp});

		VkRect2D scissor{};
		scissor.extent = swapchain_extent;
		command_buffer.set_scissor(0, {scissor});

		auto &pp_pass    = display_pipeline->get_pass(0);
		auto &pp_subpass = pp_pass.get_subpass(0);

		if (display_source)
			pp_subpass.bind_sampled_image("color_sampler",
			    vkb::core::SampledImage(*display_source));

		display_pipeline->draw(command_buffer, swapchain_render_target);
	}

	// ── Phase 3: GUI ──
	if (has_gui())
	{
		get_gui().draw(command_buffer);
	}
	command_buffer.end_render_pass();

	// ── Phase 4: Transition swapchain to PRESENT ──
	{
		vkb::ImageMemoryBarrier barrier{};
		barrier.old_layout      = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
		barrier.new_layout      = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
		barrier.src_access_mask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		barrier.dst_access_mask = 0;
		barrier.src_stage_mask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		barrier.dst_stage_mask  = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
		command_buffer.image_memory_barrier(swapchain_views[0], barrier);
	}
}

// ──────────────────────────────────────────────────────────────────────────────
// GUI
// ──────────────────────────────────────────────────────────────────────────────

void NFRUSample::draw_gui()
{
	get_gui().show_options_window(
	    [this]() {
		    ImGui::Text("Mode: %s", mode == Mode::Realtime ? "Realtime" : "Dataset");
		    ImGui::Text("Path: %s", use_fragment_path ? "Fragment" : "Compute");
		    ImGui::Separator();

		    if (mode == Mode::Dataset)
		    {
			    int total = dataset_loader ? dataset_loader->total_frames() : 0;
			    ImGui::Text("Frame: %d / %d", current_frame, total);
			    if (current_frame >= total && total > 0)
				    ImGui::TextColored({0.4f, 1.0f, 0.4f, 1.0f}, "[Completed]");
		    }
		    else
		    {
			    ImGui::Text("Frame: %llu", static_cast<unsigned long long>(frame_count));
		    }

		    // Per-frame display label: which image is currently shown
		    if (showing_generated_frame())
			    ImGui::TextColored({0.4f, 1.0f, 0.4f, 1.0f}, "Frame Generated by NFRU");
		    else
			    ImGui::TextColored({1.0f, 0.85f, 0.4f, 1.0f}, "Original");

		    ImGui::Separator();

		    if (nfru_initialized)
		    {
			    bool debug_view_prev = nfru_debug_view;
			    ImGui::Checkbox("NFRU Debug View", &nfru_debug_view);
			    if (nfru_debug_view != debug_view_prev)
				    configure_nfru();
		    }
			else
		    {
			    ImGui::TextColored({1.0f, 0.4f, 0.4f, 1.0f}, "NFRU context not initialized");
		    }

		    if (save_enabled)
		    {
			    ImGui::Text("Output: %s", interp_output_dir.c_str());
			    if (mode == Mode::Realtime)
				    ImGui::Text("GT:     %s", gt_output_dir.c_str());
		    }

		    if (mode == Mode::Realtime)
		    {
			    ImGui::Separator();
			    ImGui::SliderFloat("Orbit speed (deg/frame)", &camera_degrees_per_frame, 0.0f, 10.0f);
		    }
	    },
	    10);
}

// ──────────────────────────────────────────────────────────────────────────────
// Resource helpers
// ──────────────────────────────────────────────────────────────────────────────

void NFRUSample::create_scene_render_images(VkExtent2D color_extent, VkExtent2D depth_mv_extent, size_t swapchain_count)
{
	VkExtent3D color_3d   = {color_extent.width, color_extent.height, 1};
	VkExtent3D depth_mv_3d = {depth_mv_extent.width, depth_mv_extent.height, 1};

	depth_images.reserve(swapchain_count);
	color_images.reserve(swapchain_count);
	mv_images.reserve(swapchain_count);
	render_targets.reserve(swapchain_count);

	for (size_t i = 0; i < swapchain_count; ++i)
	{
		// Depth: D32_SFLOAT at depth_mv extent
		auto depth = std::make_unique<vkb::core::Image>(
		    get_device(), depth_mv_3d, VK_FORMAT_D32_SFLOAT,
		    VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
		        VK_IMAGE_USAGE_SAMPLED_BIT |
		        VK_IMAGE_USAGE_TRANSFER_DST_BIT,
		    VMA_MEMORY_USAGE_GPU_ONLY);
		depth->set_debug_name("NFRU Depth " + std::to_string(i));

		// Color: R8G8B8A8 at color extent
		auto color = std::make_unique<vkb::core::Image>(
		    get_device(), color_3d, VK_FORMAT_R8G8B8A8_UNORM,
		    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
		        VK_IMAGE_USAGE_SAMPLED_BIT |
		        VK_IMAGE_USAGE_TRANSFER_DST_BIT |
		        VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
		    VMA_MEMORY_USAGE_GPU_ONLY);
		color->set_debug_name("NFRU Color " + std::to_string(i));

		// MV: R16G16_SFLOAT at depth_mv extent
		auto mv = std::make_unique<vkb::core::Image>(
		    get_device(), depth_mv_3d, VK_FORMAT_R16G16_SFLOAT,
		    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
		        VK_IMAGE_USAGE_SAMPLED_BIT |
		        VK_IMAGE_USAGE_TRANSFER_DST_BIT,
		    VMA_MEMORY_USAGE_GPU_ONLY);
		mv->set_debug_name("NFRU Velocity " + std::to_string(i));

		// RenderTarget only for realtime mode
		if (mode == Mode::Realtime)
		{
			std::vector<vkb::core::ImageView> views;
			views.emplace_back(*depth, VK_IMAGE_VIEW_TYPE_2D);
			views.emplace_back(*color, VK_IMAGE_VIEW_TYPE_2D);
			views.emplace_back(*mv, VK_IMAGE_VIEW_TYPE_2D);
			render_targets.push_back(std::make_unique<vkb::RenderTarget>(std::move(views)));
		}

		depth_images.push_back(std::move(depth));
		color_images.push_back(std::move(color));
		mv_images.push_back(std::move(mv));
	}
}

void NFRUSample::create_nfru_output_images(VkExtent2D extent, size_t count)
{
	VkExtent3D ext3d = {extent.width, extent.height, 1};
	VkImageUsageFlags usage = VK_IMAGE_USAGE_SAMPLED_BIT |
	                          VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
	                          VK_IMAGE_USAGE_TRANSFER_DST_BIT |
	                          frag_or_comp_config.output_usage_bit;

	nfru_outputs.reserve(count);
	nfru_output_views.reserve(count);

	for (size_t i = 0; i < count; ++i)
	{
		auto img = std::make_unique<vkb::core::Image>(
		    get_device(), ext3d, VK_FORMAT_R8G8B8A8_UNORM, usage, VMA_MEMORY_USAGE_GPU_ONLY);
		img->set_debug_name("NFRU FG Output " + std::to_string(i));
		auto view = std::make_unique<vkb::core::ImageView>(*img, VK_IMAGE_VIEW_TYPE_2D);
		nfru_outputs.push_back(std::move(img));
		nfru_output_views.push_back(std::move(view));
	}
}

void NFRUSample::create_nfru_context(VkExtent2D display_size, VkExtent2D render_size,
                                 const float *initial_vp)
{
	LOGI("Initializing NFRU FG context (display={}x{}, render={}x{})...",
	     display_size.width, display_size.height, render_size.width, render_size.height);

	auto vkGetInstanceProcAddr_fn = reinterpret_cast<PFN_vkGetInstanceProcAddr>(
	    vkGetInstanceProcAddr(get_instance().get_handle(), "vkGetInstanceProcAddr"));
	PFN_vkGetDeviceProcAddr vkGetDeviceProcAddr_fn =
	    reinterpret_cast<PFN_vkGetDeviceProcAddr>(
	        vkGetInstanceProcAddr_fn(get_instance().get_handle(), "vkGetDeviceProcAddr"));

	ffx::CreateBackendVKDesc backend{};
	backend.vkDevice              = get_device().get_handle();
	backend.vkPhysicalDevice      = get_device().get_gpu().get_handle();
	backend.vkInstance             = get_instance().get_handle();
	backend.vkGetInstanceProcAddr  = vkGetInstanceProcAddr_fn;
	backend.vkDeviceProcAddr       = vkGetDeviceProcAddr_fn;

	ffx::CreateContextDescFrameGeneration fg{};
	fg.displaySize      = {display_size.width, display_size.height};
	fg.renderSize        = {render_size.width, render_size.height};
	fg.backBufferFormat  = FFX_API_SURFACE_FORMAT_R8G8B8A8_UNORM;
	fg.fpMessage         = OnNFRUMessage;

	fg.flags = FFX_API_FG_CONTEXT_FLAG_ENABLE_DEPTH_INVERTED;
	fg.flags |= FFX_API_FG_CONTEXT_FLAG_MANAGE_PREVIOUS_COLOR;
	fg.flags |= FFX_API_FG_CONTEXT_FLAG_MANAGE_PREVIOUS_DEPTH;

	fg.flags |= frag_or_comp_config.context_flags;
	LOGI("NFRU: {} path", use_fragment_path ? "Fragment" : "Compute");

	std::memcpy(fg.initialViewProjection, initial_vp, sizeof(fg.initialViewProjection));

	auto result = ffx::CreateContext(nfru_context, nullptr, fg, backend);
	if (result != ffx::ReturnCode::Ok)
	{
		LOGW("ffx::CreateContext(FG) failed: {} - NFRU disabled", static_cast<uint32_t>(result));
		return;
	}

	nfru_initialized = true;
	LOGI("NFRU FG context created successfully.");

	configure_nfru();
}

void NFRUSample::configure_nfru()
{
	if (!nfru_initialized)
		return;

	ffx::ConfigureDescFrameGeneration config{};
	config.flags = FFX_API_FG_DISPATCH_FLAG_NO_SWAPCHAIN_CONTEXT_NOTIFY
	             | (nfru_debug_view ? FFX_API_FG_DISPATCH_FLAG_DRAW_DEBUG_VIEW : 0u);

	auto result = ffx::Configure(nfru_context, config);
	if (result != ffx::ReturnCode::Ok)
		LOGW("NFRU Configure failed: {}", static_cast<uint32_t>(result));
}

FfxApiResource NFRUSample::wrap_resource(const vkb::core::Image &image, FfxApiResourceState state,
                                         uint32_t additional_usages)
{
	VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
	ci.imageType   = image.get_type();
	ci.format      = image.get_format();
	ci.extent      = image.get_extent();
	ci.mipLevels   = image.get_subresource().mipLevel;
	ci.arrayLayers = image.get_array_layer_count();
	ci.usage       = image.get_usage();
	ci.flags       = 0;
	const auto desc = ffxApiGetImageResourceDescriptionVK(image.get_handle(), ci, additional_usages);
	return ffxApiGetResourceVK(reinterpret_cast<void *>(image.get_handle()),
	                           desc, static_cast<uint32_t>(state));
}

void NFRUSample::dispatch_nfru(vkb::core::CommandBufferC &cb, uint32_t swapchain_index,
                                      const DatasetLoader::FrameParams &params, bool reset)
{
	// ── Transition output to GENERAL for compute/fragment write ──
	{
		vkb::ImageMemoryBarrier barrier{};
		barrier.old_layout      = VK_IMAGE_LAYOUT_UNDEFINED;
		barrier.new_layout      = VK_IMAGE_LAYOUT_GENERAL;
		barrier.src_access_mask = 0;
		barrier.dst_access_mask = frag_or_comp_config.output_write_access;
		barrier.src_stage_mask  = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
		barrier.dst_stage_mask  = frag_or_comp_config.output_write_stage;
		cb.image_memory_barrier(*nfru_output_views[swapchain_index], barrier);
	}

	// ── Prepare ──
	{
		ffx::DispatchDescFrameGenerationPrepare prep{};
		prep.commandList             = reinterpret_cast<void *>(cb.get_handle());
		prep.frameID                 = ffx_frame_id;
		prep.jitterOffset            = {params.jitterX, params.jitterY};
		prep.motionVectorScale       = {params.motionVectorScaleX,
		                                params.motionVectorScaleY};
		prep.frameTimeDelta          = params.frameTimeDelta;
		prep.cameraNear              = params.cameraNear;
		prep.cameraFar               = params.cameraFar;
		prep.cameraFovAngleVertical  = params.cameraFovAngleVertical;
		prep.viewSpaceToMetersFactor = 0.0f;
		prep.depth                   = wrap_resource(*depth_images[swapchain_index],
		                                             frag_or_comp_config.input_read_state);
		prep.motionVectors           = wrap_resource(*mv_images[swapchain_index],
		                                             frag_or_comp_config.input_read_state);
		std::memcpy(prep.viewProjection, params.viewProjection, sizeof(prep.viewProjection));

		auto result = ffx::Dispatch(nfru_context, prep);
		if (result != ffx::ReturnCode::Ok)
			LOGW("NFRU Prepare failed (fid={}): {}", ffx_frame_id, static_cast<uint32_t>(result));
	}

	// ── Dispatch ──
	{
		ffx::DispatchDescFrameGeneration disp{};
		disp.commandList        = reinterpret_cast<void *>(cb.get_handle());
		disp.presentColor       = wrap_resource(*color_images[swapchain_index],
		                                        frag_or_comp_config.input_read_state);
		disp.outputs[0]         = wrap_resource(*nfru_outputs[swapchain_index],
		                                        FFX_API_RESOURCE_STATE_UNORDERED_ACCESS);
		disp.numGeneratedFrames = 1;
		disp.reset              = reset;
		disp.frameID            = ffx_frame_id;

		auto result = ffx::Dispatch(nfru_context, disp);
		if (result != ffx::ReturnCode::Ok)
			LOGW("NFRU Dispatch failed (fid={}): {}", ffx_frame_id, static_cast<uint32_t>(result));
	}

	++ffx_frame_id;

	// ── Transition output: GENERAL → SHADER_READ_ONLY ──
	{
		vkb::ImageMemoryBarrier barrier{};
		barrier.old_layout      = VK_IMAGE_LAYOUT_GENERAL;
		barrier.new_layout      = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		barrier.src_access_mask = frag_or_comp_config.output_write_access;
		barrier.dst_access_mask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
		barrier.src_stage_mask  = frag_or_comp_config.output_write_stage;
		barrier.dst_stage_mask  = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
		cb.image_memory_barrier(*nfru_output_views[swapchain_index], barrier);
	}
}

// ──────────────────────────────────────────────────────────────────────────────
// Readback buffer helpers
// ──────────────────────────────────────────────────────────────────────────────

NFRUSample::ReadbackBuffer NFRUSample::create_readback_buffer(VkDeviceSize size)
{
	ReadbackBuffer rb;
	rb.size = size;

	VkBufferCreateInfo buf_ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
	buf_ci.size        = size;
	buf_ci.usage       = VK_BUFFER_USAGE_TRANSFER_DST_BIT;  // GPU writes, CPU reads
	buf_ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	VK_CHECK(vkCreateBuffer(get_device().get_handle(), &buf_ci, nullptr, &rb.buffer));

	VkMemoryRequirements mem_req{};
	vkGetBufferMemoryRequirements(get_device().get_handle(), rb.buffer, &mem_req);

	VkPhysicalDeviceMemoryProperties mem_props{};
	vkGetPhysicalDeviceMemoryProperties(get_device().get_gpu().get_handle(), &mem_props);

	uint32_t mem_idx = UINT32_MAX;
	for (uint32_t j = 0; j < mem_props.memoryTypeCount; ++j)
	{
		const bool type_match = (mem_req.memoryTypeBits & (1u << j)) != 0;
		const bool flag_match = (mem_props.memoryTypes[j].propertyFlags &
		                         (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
		                        (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
		if (type_match && flag_match)
		{
			mem_idx = j;
			break;
		}
	}
	assert(mem_idx != UINT32_MAX && "No host-visible memory type found for readback buffer");

	VkMemoryAllocateInfo alloc_ci{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
	alloc_ci.allocationSize  = mem_req.size;
	alloc_ci.memoryTypeIndex = mem_idx;
	VK_CHECK(vkAllocateMemory(get_device().get_handle(), &alloc_ci, nullptr, &rb.memory));
	VK_CHECK(vkBindBufferMemory(get_device().get_handle(), rb.buffer, rb.memory, 0));
	VK_CHECK(vkMapMemory(get_device().get_handle(), rb.memory, 0, size, 0, &rb.mapped));

	return rb;
}

void NFRUSample::create_readback_buffers(VkExtent2D extent, size_t count)
{
	VkDeviceSize buf_size =
	    static_cast<VkDeviceSize>(extent.width) *
	    static_cast<VkDeviceSize>(extent.height) * 4;  // R8G8B8A8

	readback_slots.resize(count);

	for (size_t i = 0; i < count; ++i)
	{
		readback_slots[i].buffer = create_readback_buffer(buf_size);
	}

	LOGI("Created {} readback buffers ({} bytes each)", count, buf_size);
}

void NFRUSample::copy_to_readback(vkb::core::CommandBufferC &cmd,
                                 const vkb::core::ImageView &src_view, ReadbackBuffer &dst)
{
	auto &src_image = src_view.get_image();

	// SHADER_READ_ONLY → TRANSFER_SRC
	{
		vkb::ImageMemoryBarrier barrier{};
		barrier.old_layout      = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		barrier.new_layout      = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
		barrier.src_access_mask = VK_ACCESS_SHADER_READ_BIT;
		barrier.dst_access_mask = VK_ACCESS_TRANSFER_READ_BIT;
		barrier.src_stage_mask  = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
		barrier.dst_stage_mask  = VK_PIPELINE_STAGE_TRANSFER_BIT;
		cmd.image_memory_barrier(src_view, barrier);
	}

	VkBufferImageCopy region{};
	region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	region.imageSubresource.layerCount = 1;
	region.imageExtent                 = src_image.get_extent();
	vkCmdCopyImageToBuffer(cmd.get_handle(), src_image.get_handle(),
	                       VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
	                       dst.buffer, 1, &region);

	// TRANSFER_SRC → SHADER_READ_ONLY
	{
		vkb::ImageMemoryBarrier barrier{};
		barrier.old_layout      = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
		barrier.new_layout      = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		barrier.src_access_mask = VK_ACCESS_TRANSFER_READ_BIT;
		barrier.dst_access_mask = VK_ACCESS_SHADER_READ_BIT;
		barrier.src_stage_mask  = VK_PIPELINE_STAGE_TRANSFER_BIT;
		barrier.dst_stage_mask  = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
		cmd.image_memory_barrier(src_view, barrier);
	}
}

void NFRUSample::flush_pending_save(ReadbackSlot &slot, VkExtent2D extent)
{
	if (!slot.need_save)
		return;

	slot.need_save = false;

	char fname[32];
	snprintf(fname, sizeof(fname), "%04llu.exr", static_cast<unsigned long long>(slot.frame_number));
	save_r8g8b8a8_as_exr(slot.buffer.mapped, extent, slot.output_dir + "/" + fname);
}

// ──────────────────────────────────────────────────────────────────────────────
// Factory
// ──────────────────────────────────────────────────────────────────────────────

std::unique_ptr<vkb::VulkanSampleC> create_nfru()
{
	return std::make_unique<NFRUSample>();
}
