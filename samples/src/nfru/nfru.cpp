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

#include "nfru.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <vector>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "common/app_argument_reader.h"
#include "common/exr_image.h"
#include "core/image.h"
#include "core/image_view.h"

#include "rendering/postprocessing_pipeline.h"
#include "rendering/postprocessing_renderpass.h"
#include "rendering/render_pipeline.h"
#include "rendering/subpass.h"
#include "rendering/subpasses/temporal_forward_subpass.h"

#include "common/utils.h"
#include "scene_graph/components/perspective_camera.h"
#include "scene_graph/components/transform.h"
#include "scene_graph/node.h"

/*******************************
 * construction and destruction
 *******************************/

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

	if (nfru_context)
	{
		log_memory_usage("before_destroy", true);
		ffx::DestroyContext(nfru_context);
		nfru_context = nullptr;
	}
}

void NFRUSample::request_gpu_features(vkb::core::PhysicalDeviceC &gpu)
{
	vkb::VulkanSampleC::request_gpu_features(gpu);

	REQUEST_REQUIRED_FEATURE(gpu, VkPhysicalDeviceVulkan12Features, shaderInt8);

	REQUEST_OPTIONAL_FEATURE(gpu, VkPhysicalDeviceVulkan12Features, shaderFloat16);
	REQUEST_OPTIONAL_FEATURE(gpu, VkPhysicalDevice16BitStorageFeatures, storageBuffer16BitAccess);
	REQUEST_OPTIONAL_FEATURE(gpu, VkPhysicalDevice16BitStorageFeatures, uniformAndStorageBuffer16BitAccess);
	if (gpu.get_features().shaderStorageImageExtendedFormats)
	{
		gpu.get_mutable_requested_features().shaderStorageImageExtendedFormats = true;
	}

	REQUEST_REQUIRED_FEATURE(gpu, VkPhysicalDeviceVulkan13Features, synchronization2);

	REQUEST_OPTIONAL_FEATURE(gpu, VkPhysicalDeviceTensorFeaturesARM, shaderTensorAccess);
	REQUEST_OPTIONAL_FEATURE(gpu, VkPhysicalDeviceTensorFeaturesARM, tensors);
	REQUEST_OPTIONAL_FEATURE(gpu, VkPhysicalDeviceDataGraphFeaturesARM, dataGraph);
	REQUEST_OPTIONAL_FEATURE(gpu, VkPhysicalDeviceDataGraphOpticalFlowFeaturesARM, dataGraphOpticalFlow);
}

/*******************************
 * prepare
 *******************************/

bool NFRUSample::prepare(const vkb::ApplicationOptions &options)
{
	if (!vkb::VulkanSampleC::prepare(options))
	{
		return false;
	}

	// Command-line configuration (see `--app-arg KEY=VALUE`)
	//   NFRU_DATASET_DIR      : path; selects Dataset mode when present (else Realtime)
	//   NFRU_DATASET_SEQUENCE : dataset sequence id (default "0000")
	//   NFRU_SAVE_EXR         : 0 or 1, save output frames as EXR
	//   NFRU_MANAGE_HISTORY   : 0 (sample supplies previous color/depth) or 1 (SDK owns previous resources)
	//   NFRU_ORBIT_SPEED      : float in [0, 10] degrees/frame (realtime camera)
	//   NFRU_USE_FRAGMENT     : selectable-stage path; 0 (compute) or 1 (fragment)
	//   NFRU_ENABLE_MV_HINTS  : 0 or 1, use motion vector hints for optical flow
	//   NFRU_DEBUG_VIEW       : 0 or 1, enable NFRU debug view
	//   NFRU_DEBUG_CHECKING   : 0 or 1, enable verbose per-frame NFRU runtime logs
	vkb::AppArgumentReader args{options.app_arguments};

	const std::string dataset_dir_arg = args.get_string("NFRU_DATASET_DIR");
	mode                              = dataset_dir_arg.empty() ? Mode::Realtime : Mode::Dataset;

	save_exr_enabled         = args.get_bool("NFRU_SAVE_EXR", save_exr_enabled);
	nfru_manage_history      = args.get_bool("NFRU_MANAGE_HISTORY", nfru_manage_history);
	camera_degrees_per_frame = args.get_float("NFRU_ORBIT_SPEED", camera_degrees_per_frame, 0.0f, 10.0f);
	use_fragment_path        = args.get_bool("NFRU_USE_FRAGMENT", use_fragment_path);
	enable_mv_hints          = args.get_bool("NFRU_ENABLE_MV_HINTS", enable_mv_hints);
	nfru_debug_view          = args.get_bool("NFRU_DEBUG_VIEW", nfru_debug_view);
	nfru_debug_checking      = args.get_bool("NFRU_DEBUG_CHECKING", nfru_debug_checking);

#if defined(__ANDROID__)
	if (mode == Mode::Realtime)
	{
		if (save_exr_enabled)
			LOGW("Android realtime: force disable EXR save to avoid storage permission issues.");
		save_exr_enabled = false;
	}
#endif

	// Fixed downsample stages can still read inputs as fragments when all selectable stages use compute.
	frag_or_comp_config = use_fragment_path ? FragOrCompConfig{VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
	                                                           0u,
	                                                           VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
	                                                           VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
	                                                           VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
	                                                           VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
	                                                           FFX_API_RESOURCE_STATE_PIXEL_READ,
	                                                           FFX_API_RESOURCE_STATE_RENDER_TARGET} :
	                                          FragOrCompConfig{VK_IMAGE_USAGE_STORAGE_BIT,
	                                                           FFX_API_FG_CONTEXT_FLAG_ALL_STAGES_COMPUTE |
	                                                               (enable_mv_hints ? FFX_API_FG_CONTEXT_FLAG_MV_HINTS_COMPUTE : 0u),
	                                                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
	                                                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
	                                                           VK_ACCESS_SHADER_WRITE_BIT,
	                                                           VK_IMAGE_LAYOUT_GENERAL,
	                                                           FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ,
	                                                           FFX_API_RESOURCE_STATE_UNORDERED_ACCESS};

	size_t swapchain_count = get_render_context().get_swapchain().get_images().size();
	resource_count         = std::max<size_t>(2, swapchain_count);
	render_extent          = get_render_context().get_surface_extent();

	if (mode == Mode::Dataset)
	{
		// Dataset initialization
		const std::string dataset_dir      = dataset_dir_arg;
		const std::string dataset_sequence = args.get_string("NFRU_DATASET_SEQUENCE", "0000");

		dataset_loader = std::make_unique<NFRUDatasetLoader>();

		if (!dataset_loader->init(get_device(),
		                          dataset_dir, dataset_sequence,
		                          static_cast<uint32_t>(swapchain_count)))
		{
			return false;
		}

		dataset_display_size = dataset_loader->get_display_size();
		dataset_render_size  = dataset_loader->get_render_size();

		LOGI("NFRU Dataset: display={}x{}, render={}x{}",
		     dataset_display_size.width, dataset_display_size.height,
		     dataset_render_size.width, dataset_render_size.height);

		create_frame_inputs(dataset_display_size, dataset_render_size, false);

		create_nfru_context(dataset_display_size, dataset_render_size);
		if (!nfru_context)
			return false;

		create_nfru_output(dataset_display_size);

		// Dataset output goes into the dataset directory
		interp_output_dir = dataset_dir + "/result/nfru_x2/" + dataset_sequence;

		// Readback buffers (used only when save_exr_enabled)
		if (save_exr_enabled)
		{
			create_readback_buffers(dataset_display_size, swapchain_count);
			if (!vkb::sample::exr::create_directories(interp_output_dir))
				return false;
		}
	}
	else
	{
		// Realtime initialization
		load_scene("scenes/sponza/Sponza01.gltf");

		LOGI("NFRU: render extent = {}x{}", render_extent.width, render_extent.height);

		create_frame_inputs(render_extent, render_extent, save_exr_enabled);

		// Camera
		auto &camera_node_ref = vkb::add_free_camera(get_scene(), "main_camera", render_extent);
		camera                = dynamic_cast<vkb::sg::PerspectiveCamera *>(
            &camera_node_ref.get_component<vkb::sg::Camera>());
		camera_node = &camera_node_ref;

		{
			glm::vec3 eye          = {orbit_radius, orbit_height, 0.0f};
			glm::vec3 center       = {0.0f, orbit_height, 0.0f};
			glm::mat4 current_view = glm::lookAt(eye, center, glm::vec3{0.0f, 1.0f, 0.0f});
			camera_node->get_transform().set_matrix(glm::inverse(current_view));
		}

		// Scene render pipeline
		{
			vkb::ShaderSource vert("temporal/jitter.vert.spv");
			vkb::ShaderSource frag("temporal/velocity.frag.spv");
			auto              subpass = std::make_unique<vkb::TemporalForwardSubpass>(
                get_render_context(), std::move(vert), std::move(frag),
                get_scene(), *camera);
			temporal_subpass = subpass.get();

			subpass->set_output_attachments({static_cast<int>(Attachments::Color),
			                                 static_cast<int>(Attachments::Velocity)});
			subpass->set_mipmap_lod_bias(0.0f);
			subpass->set_jitter_scale(0.0f);
			// NFRU_SAVE_EXR also renders odd presentation frames as ground truth.
			subpass->set_velocity_stride(save_exr_enabled ? 2 : 1);

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
				color_clear.color = {0.0f, 0.0f, 0.0f, 1.0f};        // black sky
				VkClearValue vel_clear{};
				vel_clear.color = {0.0f, 0.0f, 0.0f, 0.0f};        // no motion
				scene_pipeline->set_clear_value({depth_clear, color_clear, vel_clear});
			}
		}

		create_nfru_context(render_extent, render_extent);

		if (!nfru_context)
			return false;
		create_nfru_output(render_extent);

		if (save_exr_enabled)
		{
			create_readback_buffers(render_extent, swapchain_count);
#if !defined(__ANDROID__)
			if (!vkb::sample::exr::create_directories(gt_output_dir))
				return false;
			if (!vkb::sample::exr::create_directories(interp_output_dir))
				return false;
#endif
		}
	}

	// Display pipeline
	{
		vkb::ShaderSource postprocessing_vertex_shader("postprocessing.vert.spv");
		display_pipeline = std::make_unique<vkb::PostProcessingPipeline>(
		    get_render_context(), std::move(postprocessing_vertex_shader));
		display_pipeline->add_pass().add_subpass(
		    vkb::ShaderSource("display.frag.spv"));
	}

	get_stats().request_stats({vkb::StatIndex::frame_times});
	create_gui(*window, &get_stats());

	return true;
}

void NFRUSample::update(float delta_time)
{
	frame_is_generated = mode == Mode::Dataset ||
	                     !save_exr_enabled ||
	                     (scene_frame_index % 2 == 0);
	frame_should_save = save_exr_enabled && scene_frame_index >= 1;

	// The orbit is frame-driven so adjacent source frames stay evenly spaced
	if (mode == Mode::Realtime)
	{
		nfru_delta_time_seconds += delta_time;
		const float yaw_degrees  = static_cast<float>(scene_frame_index) * camera_degrees_per_frame;
		const float yaw_radians  = glm::radians(std::fmod(yaw_degrees, 360.0f));
		glm::vec3   eye          = {cosf(yaw_radians) * orbit_radius, orbit_height, sinf(yaw_radians) * orbit_radius};
		glm::vec3   center       = {0.0f, orbit_height, 0.0f};
		glm::mat4   current_view = glm::lookAt(eye, center, glm::vec3{0.0f, 1.0f, 0.0f});
		camera_node->get_transform().set_matrix(glm::inverse(current_view));
	}

	VulkanSample::update(delta_time);
}

/*******************************
 * resource creation
 *******************************/

void NFRUSample::create_frame_inputs(VkExtent2D color_extent, VkExtent2D depth_mv_extent, bool create_reference_frame)
{
	VkExtent3D color_3d    = {color_extent.width, color_extent.height, 1};
	VkExtent3D depth_mv_3d = {depth_mv_extent.width, depth_mv_extent.height, 1};

	auto create_inputs = [&](FrameInputs &inputs, const std::string &name) {
		inputs.depth.image = std::make_unique<vkb::core::Image>(
		    get_device(), depth_mv_3d, VK_FORMAT_D32_SFLOAT,
		    VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
		        VK_IMAGE_USAGE_SAMPLED_BIT |
		        VK_IMAGE_USAGE_TRANSFER_DST_BIT,
		    VMA_MEMORY_USAGE_GPU_ONLY);
		inputs.depth.image->set_debug_name("NFRU Depth " + name);
		inputs.depth.view = std::make_unique<vkb::core::ImageView>(*inputs.depth.image, VK_IMAGE_VIEW_TYPE_2D);

		inputs.color.image = std::make_unique<vkb::core::Image>(
		    get_device(), color_3d, VK_FORMAT_R8G8B8A8_UNORM,
		    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
		        VK_IMAGE_USAGE_SAMPLED_BIT |
		        VK_IMAGE_USAGE_TRANSFER_DST_BIT |
		        VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
		    VMA_MEMORY_USAGE_GPU_ONLY);
		inputs.color.image->set_debug_name("NFRU Color " + name);
		inputs.color.view = std::make_unique<vkb::core::ImageView>(*inputs.color.image, VK_IMAGE_VIEW_TYPE_2D);

		inputs.motion_vectors.image = std::make_unique<vkb::core::Image>(
		    get_device(), depth_mv_3d, VK_FORMAT_R16G16_SFLOAT,
		    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
		        VK_IMAGE_USAGE_SAMPLED_BIT |
		        VK_IMAGE_USAGE_TRANSFER_DST_BIT,
		    VMA_MEMORY_USAGE_GPU_ONLY);
		inputs.motion_vectors.image->set_debug_name("NFRU Velocity " + name);
		inputs.motion_vectors.view = std::make_unique<vkb::core::ImageView>(*inputs.motion_vectors.image, VK_IMAGE_VIEW_TYPE_2D);

		if (mode == Mode::Realtime)
		{
			std::vector<vkb::core::ImageView> views;
			views.emplace_back(*inputs.depth.image, VK_IMAGE_VIEW_TYPE_2D);
			views.emplace_back(*inputs.color.image, VK_IMAGE_VIEW_TYPE_2D);
			views.emplace_back(*inputs.motion_vectors.image, VK_IMAGE_VIEW_TYPE_2D);
			inputs.render_target = std::make_unique<vkb::RenderTarget>(std::move(views));
		}
	};

	frame_inputs.resize(resource_count);
	for (size_t index = 0; index < resource_count; ++index)
		create_inputs(frame_inputs[index], std::to_string(index));
	if (create_reference_frame)
	{
		reference_frame_inputs = std::make_unique<FrameInputs>();
		create_inputs(*reference_frame_inputs, "Reference");
	}
}

void NFRUSample::create_nfru_output(VkExtent2D extent)
{
	VkExtent3D        ext3d = {extent.width, extent.height, 1};
	VkImageUsageFlags usage = VK_IMAGE_USAGE_SAMPLED_BIT |
	                          VK_IMAGE_USAGE_TRANSFER_DST_BIT |
	                          frag_or_comp_config.output_usage_bit;
	if (save_exr_enabled)
		usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;

	nfru_output.image = std::make_unique<vkb::core::Image>(
	    get_device(), ext3d, VK_FORMAT_R8G8B8A8_UNORM, usage, VMA_MEMORY_USAGE_GPU_ONLY);
	nfru_output.image->set_debug_name("NFRU Output");
	nfru_output.view = std::make_unique<vkb::core::ImageView>(*nfru_output.image, VK_IMAGE_VIEW_TYPE_2D);
}
/*******************************
 * draw
 *******************************/

const vkb::core::ImageView *NFRUSample::render_frame_inputs(vkb::core::CommandBufferC &command_buffer, FrameInputs &inputs)
{
	auto &render_target = *inputs.render_target;

	// Transition scene inputs from UNDEFINED or SHADER_READ_ONLY_OPTIMAL to their
	// color/depth attachment layouts before rendering.
	prepare_render_inputs(command_buffer, inputs);

	VkViewport viewport{};
	viewport.width    = static_cast<float>(render_extent.width);
	viewport.height   = static_cast<float>(render_extent.height);
	viewport.minDepth = 0.0f;
	viewport.maxDepth = 1.0f;
	command_buffer.set_viewport(0, {viewport});
	VkRect2D scissor{};
	scissor.extent = render_extent;
	command_buffer.set_scissor(0, {scissor});

	scene_pipeline->draw(command_buffer, render_target);
	command_buffer.end_render_pass();

	// Publish the rendered attachments for NFRU, display, or ground-truth readback.
	finish_render_inputs(command_buffer, inputs);

	return inputs.color.view.get();
}

const vkb::core::ImageView *NFRUSample::draw_realtime(vkb::core::CommandBufferC &command_buffer, uint32_t swapchain_frame_index)
{
	const size_t current_index  = nfru_frame_index % resource_count;
	const size_t previous_index = (current_index + resource_count - 1) % resource_count;
	auto        &current        = frame_inputs[current_index];
	auto        &previous       = frame_inputs[previous_index];

	FrameInputs                &render_inputs  = frame_is_generated ? current : *reference_frame_inputs;
	const vkb::core::ImageView *rendered_color = render_frame_inputs(command_buffer, render_inputs);
	const vkb::core::ImageView *display_source = rendered_color;

	if (frame_is_generated)
	{
		// NFRU-generated frame: dispatch interpolation and show its output.
		NFRUFrameParams params{};
		glm::mat4       view_projection = temporal_subpass->get_view_projection();
		std::memcpy(params.view_projection, glm::value_ptr(view_projection), sizeof(params.view_projection));
		auto jitter                      = temporal_subpass->get_current_jitter_pixels_scaled();
		params.jitter_x                  = jitter.x;
		params.jitter_y                  = jitter.y;
		params.camera_near               = camera->get_near_plane();
		params.camera_far                = camera->get_far_plane();
		params.camera_fov_angle_vertical = camera->get_field_of_view();
		params.frame_time_delta_ms       = std::fmax(nfru_delta_time_seconds * 1000.0f, 1.0f);
		params.motion_vector_scale_x     = static_cast<float>(render_extent.width);
		params.motion_vector_scale_y     = static_cast<float>(render_extent.height);

		if (!dispatch_nfru(command_buffer, current, previous, params, /*reset=*/nfru_frame_index == 0))
		{
			close();
			return nullptr;
		}
		nfru_delta_time_seconds = 0.0f;
		display_source          = nfru_output.view.get();
	}
	// EXR save: generated (even) frames go to interp, reference (odd) frames go
	// to ground truth. Frame 0 is skipped
	if (frame_should_save)
	{
		ReadbackSlot &slot = readback_slots[swapchain_frame_index];
		slot.need_save     = true;
		if (frame_is_generated)
		{
			copy_to_readback(command_buffer, *nfru_output.view, *slot.buffer);
			slot.frame_number = scene_frame_index - 1;
			slot.output_dir   = interp_output_dir;
		}
		else
		{
			copy_to_readback(command_buffer, *rendered_color, *slot.buffer);
			render_inputs.render_target->set_layout(static_cast<int>(Attachments::Color), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
			slot.frame_number = scene_frame_index;
			slot.output_dir   = gt_output_dir;
		}
	}

	++scene_frame_index;
	return display_source;
}

const vkb::core::ImageView *NFRUSample::draw_dataset(vkb::core::CommandBufferC &command_buffer, uint32_t swapchain_frame_index)
{
	if (scene_frame_index >= dataset_loader->total_frames())
		return nfru_output.view.get();

	NFRUFrameParams frame_params{};
	const size_t    current_index  = nfru_frame_index % resource_count;
	const size_t    previous_index = (current_index + resource_count - 1) % resource_count;
	auto           &current        = frame_inputs[current_index];
	auto           &previous       = frame_inputs[previous_index];
	if (!dataset_loader->load_frame(static_cast<int>(scene_frame_index), swapchain_frame_index, frame_params))
	{
		LOGE("Failed to load dataset frame {}", scene_frame_index);
		close();
		return nullptr;
	}
	frame_should_save = frame_should_save && !frame_params.reset;

	// Transition dataset inputs from UNDEFINED or SHADER_READ_ONLY_OPTIMAL to
	// TRANSFER_DST_OPTIMAL before uploading the frame.
	prepare_upload_inputs(command_buffer, current);

	// Upload color at display size and depth/motion at render size.
	VkBufferImageCopy region{};
	region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	region.imageSubresource.layerCount = 1;
	region.imageExtent                 = {dataset_display_size.width, dataset_display_size.height, 1};
	vkCmdCopyBufferToImage(command_buffer.get_handle(), dataset_loader->get_color_staging_buffer(swapchain_frame_index), current.color.image->get_handle(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
	region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
	region.imageExtent                 = {dataset_render_size.width, dataset_render_size.height, 1};
	vkCmdCopyBufferToImage(command_buffer.get_handle(), dataset_loader->get_depth_staging_buffer(swapchain_frame_index), current.depth.image->get_handle(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
	region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	vkCmdCopyBufferToImage(command_buffer.get_handle(), dataset_loader->get_motion_vector_staging_buffer(swapchain_frame_index), current.motion_vectors.image->get_handle(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

	// Transition all uploaded inputs from TRANSFER_DST_OPTIMAL to
	// SHADER_READ_ONLY_OPTIMAL for NFRU history and dispatch reads.
	finish_upload_inputs(command_buffer, current);

	if (!dispatch_nfru(command_buffer, current, previous, frame_params, frame_params.reset))
	{
		close();
		return nullptr;
	}

	// Save the interpolated output. Frame 0 is the reset frame (skipped).
	if (frame_should_save)
	{
		ReadbackSlot &slot = readback_slots[swapchain_frame_index];
		copy_to_readback(command_buffer, *nfru_output.view, *slot.buffer);
		slot.need_save    = true;
		slot.frame_number = dataset_loader->interpolated_source_frame(static_cast<int>(scene_frame_index));
		slot.output_dir   = interp_output_dir;
	}

	++scene_frame_index;
	return nfru_output.view.get();
}

void NFRUSample::draw(vkb::core::CommandBufferC &command_buffer,
                      vkb::RenderTarget         &swapchain_render_target)
{
	uint32_t swapchain_frame_index = get_render_context().get_active_frame_index();
	auto    &swapchain_views       = swapchain_render_target.get_views();
	auto     swapchain_extent      = swapchain_render_target.get_extent();

	// The frame fence guarantees the pending GPU-to-buffer copy is complete.
	{
		const VkExtent2D save_extent = mode == Mode::Dataset ? dataset_display_size : render_extent;
		if (save_exr_enabled && !save_completed_readback(readback_slots[swapchain_frame_index], save_extent))
		{
			close();
			return;
		}
	}

	// Acquire the swapchain image for postprocess and GUI color writes.
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

	// Produce this frame's display source from realtime rendering or dataset replay.
	const vkb::core::ImageView *display_source = mode == Mode::Realtime ?
	                                                 draw_realtime(command_buffer, swapchain_frame_index) :
	                                                 draw_dataset(command_buffer, swapchain_frame_index);
	if (!display_source)
		return;

	// Display
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

		auto &postprocessing_pass    = display_pipeline->get_pass(0);
		auto &postprocessing_subpass = postprocessing_pass.get_subpass(0);

		postprocessing_subpass.bind_sampled_image("color_sampler", vkb::core::SampledImage(*display_source));

		display_pipeline->draw(command_buffer, swapchain_render_target);
	}

	// GUI
	if (has_gui())
	{
		get_gui().draw(command_buffer);
	}
	command_buffer.end_render_pass();

	// Postprocess/GUI color writes must finish before the presentation engine reads
	// the completed swapchain image.
	{
		vkb::ImageMemoryBarrier barrier{};
		barrier.old_layout      = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
		barrier.new_layout      = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
		barrier.src_access_mask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		barrier.dst_access_mask = 0;
		barrier.src_stage_mask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		barrier.dst_stage_mask  = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
		command_buffer.image_memory_barrier(swapchain_views[0], barrier);
		swapchain_render_target.set_layout(0, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
	}
}

/*******************************
 * NFRU API
 *******************************/
static void nfru_message_callback(uint32_t type, const char *message)
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

void NFRUSample::create_nfru_context(VkExtent2D display_size, VkExtent2D render_size)
{
	float initial_view_projection_values[16]{};
	bool  depth_inverted = false;
	bool  depth_infinite = false;

	if (mode == Mode::Dataset)
	{
		NFRUFrameParams first_params{};
		if (!dataset_loader->get_frame_params(0, first_params))
		{
			LOGE("Failed to read dataset frame 0 parameters");
			return;
		}

		std::memcpy(initial_view_projection_values, first_params.view_projection, sizeof(initial_view_projection_values));
		depth_inverted = dataset_loader->uses_inverted_depth();
		depth_infinite = dataset_loader->uses_infinite_far();
	}
	else
	{
		const glm::mat4 initial_view_projection = temporal_subpass->get_view_projection();
		std::memcpy(initial_view_projection_values, glm::value_ptr(initial_view_projection), sizeof(initial_view_projection_values));
		depth_inverted = true;
	}

	LOGI("Initializing NFRU FG context (display={}x{}, render={}x{}, inputFlags=0x{:x})...",
	     display_size.width, display_size.height, render_size.width, render_size.height,
	     (depth_inverted ? FFX_API_FG_CONTEXT_FLAG_ENABLE_DEPTH_INVERTED : 0u) |
	         (depth_infinite ? FFX_API_FG_CONTEXT_FLAG_ENABLE_DEPTH_INFINITE : 0u));

	auto get_instance_proc_address = reinterpret_cast<PFN_vkGetInstanceProcAddr>(
	    vkGetInstanceProcAddr(get_instance().get_handle(), "vkGetInstanceProcAddr"));
	PFN_vkGetDeviceProcAddr get_device_proc_address =
	    reinterpret_cast<PFN_vkGetDeviceProcAddr>(
	        get_instance_proc_address(get_instance().get_handle(), "vkGetDeviceProcAddr"));

	ffx::CreateBackendVKDesc backend_description{};
	backend_description.vkDevice              = get_device().get_handle();
	backend_description.vkPhysicalDevice      = get_device().get_gpu().get_handle();
	backend_description.vkInstance            = get_instance().get_handle();
	backend_description.vkGetInstanceProcAddr = get_instance_proc_address;
	backend_description.vkDeviceProcAddr      = get_device_proc_address;

	ffx::CreateContextDescFrameGeneration context_description{};
	context_description.displaySize      = {display_size.width, display_size.height};
	context_description.renderSize       = {render_size.width, render_size.height};
	context_description.backBufferFormat = FFX_API_SURFACE_FORMAT_R8G8B8A8_UNORM;
	context_description.fpMessage        = nfru_message_callback;

	context_description.flags = (depth_inverted ? FFX_API_FG_CONTEXT_FLAG_ENABLE_DEPTH_INVERTED : 0u) |
	                            (depth_infinite ? FFX_API_FG_CONTEXT_FLAG_ENABLE_DEPTH_INFINITE : 0u);
	if (nfru_manage_history)
	{
		context_description.flags |= FFX_API_FG_CONTEXT_FLAG_MANAGE_PREVIOUS_COLOR;
		context_description.flags |= FFX_API_FG_CONTEXT_FLAG_MANAGE_PREVIOUS_DEPTH;
	}

	context_description.flags |= frag_or_comp_config.context_flags;
	if (enable_mv_hints)
		context_description.flags |= FFX_API_FG_CONTEXT_FLAG_ENABLE_MV_HINTS;

	LOGI("NFRU: selectable stages use {}, MV hints {}", use_fragment_path ? "fragment" : "compute", enable_mv_hints ? "enabled" : "disabled");
	LOGI("NFRU: {} previous color/depth history requested", nfru_manage_history ? "SDK-managed" : "sample-provided");
	if (!nfru_manage_history && display_size.height > 1080)
		LOGI("NFRU: internal previous depth is used while color downsampling is required");

	std::memcpy(context_description.initialViewProjection, initial_view_projection_values, sizeof(context_description.initialViewProjection));

	auto result = ffx::CreateContext(nfru_context, nullptr, context_description, backend_description);
	if (result != ffx::ReturnCode::Ok)
	{
		LOGE("ffx::CreateContext(FG) failed: {}", static_cast<uint32_t>(result));
		return;
	}

	nfru_initialized = true;
	LOGI("NFRU FG context created successfully.");

	configure_nfru();
	log_memory_usage("context_created", true);
}

void NFRUSample::configure_nfru()
{
	if (!nfru_initialized)
		return;

	ffx::ConfigureDescFrameGeneration configure_description{};
	configure_description.frameGenerationEnabled = true;
	configure_description.frameID                = nfru_frame_index;
	configure_description.flags                  = FFX_API_FG_DISPATCH_FLAG_NO_SWAPCHAIN_CONTEXT_NOTIFY |
	                              (nfru_debug_view ? FFX_API_FG_DISPATCH_FLAG_DRAW_DEBUG_VIEW : 0u) |
	                              (nfru_debug_checking ? FFX_API_FG_DISPATCH_FLAG_ENABLE_DEBUG_CHECKING : 0u);

	auto result = ffx::Configure(nfru_context, configure_description);
	if (result != ffx::ReturnCode::Ok)
		LOGW("NFRU Configure failed: {}", static_cast<uint32_t>(result));
}

void NFRUSample::log_memory_usage(const char *event, bool force_log)
{
	FfxApiEffectMemoryUsage                        vram_usage{};
	ffx::QueryDescFrameGenerationGetGPUMemoryUsage query{};
	query.gpuMemoryUsageFrameGeneration = &vram_usage;

	const auto query_result = ffx::Query(nfru_context, query);
	if (query_result == ffx::ReturnCode::Ok)
	{
		if (memory_usage_tracker.update_gpu_usage(vram_usage, force_log))
		{
			LOGI("NFRU_VRAM_USAGE(bytes) event={} total_usage={} texture_usage={} buffer_usage={} tensor_usage={} data_graph_usage={} aliasable_usage={}",
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
		LOGW("NFRU VRAM query failed (event={}): {}", event, static_cast<uint32_t>(query_result));
	}

#ifdef FFX_ENABLE_CPU_HEAP_TRACKING
	FfxApiEffectCpuMemoryUsage                     cpu_memory_usage{};
	ffx::QueryDescFrameGenerationGetCPUMemoryUsage cpu_memory_query{};
	cpu_memory_query.cpuMemoryUsageFrameGeneration = &cpu_memory_usage;

	const auto cpu_memory_query_result = ffx::Query(nfru_context, cpu_memory_query);
	if (cpu_memory_query_result == ffx::ReturnCode::Ok)
	{
		if (memory_usage_tracker.update_cpu_usage(cpu_memory_usage, force_log))
		{
			LOGI("NFRU_CPU_MEMORY_USAGE(bytes) event={} current_usage={} peak_usage={}",
			     event,
			     cpu_memory_usage.currentUsageInBytes,
			     cpu_memory_usage.peakUsageInBytes);
		}
	}
	else
	{
		LOGW("NFRU CPU memory query failed (event={}): {}", event, static_cast<uint32_t>(cpu_memory_query_result));
	}
#endif
}

FfxApiResource NFRUSample::wrap_resource(const vkb::core::Image &image, FfxApiResourceState state,
                                         uint32_t additional_usages)
{
	VkImageCreateInfo create_info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
	create_info.imageType           = image.get_type();
	create_info.format              = image.get_format();
	create_info.extent              = image.get_extent();
	create_info.mipLevels           = image.get_subresource().mipLevel;
	create_info.arrayLayers         = image.get_array_layer_count();
	create_info.usage               = image.get_usage();
	create_info.flags               = 0;
	const auto resource_description = ffxApiGetImageResourceDescriptionVK(image.get_handle(), create_info, additional_usages);
	return ffxApiGetResourceVK(reinterpret_cast<void *>(image.get_handle()),
	                           resource_description, static_cast<uint32_t>(state));
}

bool NFRUSample::dispatch_nfru(vkb::core::CommandBufferC &command_buffer, FrameInputs &current, FrameInputs &previous,
                               const NFRUFrameParams &frame_params, bool reset)
{
	initialize_external_history(command_buffer, previous);

	// Prepare NFRU inputs and frame metadata.
	{
		ffx::DispatchDescFrameGenerationPrepare prepare_description{};
		prepare_description.commandList                   = reinterpret_cast<void *>(command_buffer.get_handle());
		prepare_description.frameID                       = nfru_frame_index;
		prepare_description.jitterOffset                  = {frame_params.jitter_x, frame_params.jitter_y};
		prepare_description.motionVectorScale             = {frame_params.motion_vector_scale_x,
		                                                     frame_params.motion_vector_scale_y};
		prepare_description.mvSimilarityRelativeThreshold = 0.002f;
		prepare_description.frameTimeDelta                = frame_params.frame_time_delta_ms;
		prepare_description.cameraNear                    = frame_params.camera_near;
		prepare_description.cameraFar                     = frame_params.camera_far;
		prepare_description.cameraFovAngleVertical        = frame_params.camera_fov_angle_vertical;
		prepare_description.viewSpaceToMetersFactor       = frame_params.view_space_to_meters_factor;
		prepare_description.depth                         = wrap_resource(*current.depth.image,
		                                                                  frag_or_comp_config.input_read_state);
		prepare_description.motionVectors                 = wrap_resource(*current.motion_vectors.image,
		                                                                  frag_or_comp_config.input_read_state);
		if (!nfru_manage_history)
		{
			prepare_description.colorTm1 = wrap_resource(*previous.color.image, frag_or_comp_config.input_read_state);
			prepare_description.depthTm1 = wrap_resource(*previous.depth.image, frag_or_comp_config.input_read_state);
		}
		std::memcpy(prepare_description.viewProjection, frame_params.view_projection, sizeof(prepare_description.viewProjection));

		auto result = ffx::Dispatch(nfru_context, prepare_description);
		if (result != ffx::ReturnCode::Ok)
		{
			LOGE("NFRU Prepare failed (fid={}): {}", nfru_frame_index, static_cast<uint32_t>(result));
			return false;
		}
	}

	// Transition the output from SHADER_READ_ONLY_OPTIMAL to the selected compute
	// or fragment output-write layout.
	prepare_output(command_buffer);

	// Dispatch frame generation.
	{
		ffx::DispatchDescFrameGeneration dispatch_description{};
		dispatch_description.commandList        = reinterpret_cast<void *>(command_buffer.get_handle());
		dispatch_description.presentColor       = wrap_resource(*current.color.image,
		                                                        frag_or_comp_config.input_read_state);
		dispatch_description.outputs[0]         = wrap_resource(*nfru_output.image,
		                                                        frag_or_comp_config.output_write_state);
		dispatch_description.numGeneratedFrames = 1;
		dispatch_description.reset              = reset;
		dispatch_description.frameID            = nfru_frame_index;

		auto result = ffx::Dispatch(nfru_context, dispatch_description);
		if (result != ffx::ReturnCode::Ok)
		{
			LOGE("NFRU Dispatch failed (fid={}): {}", nfru_frame_index, static_cast<uint32_t>(result));
			return false;
		}

		const bool first_dispatch = !nfru_has_successful_dispatch;
		log_memory_usage(first_dispatch ? "first_dispatch" : "dispatch", first_dispatch);
		nfru_has_successful_dispatch = true;
	}

	finish_output(command_buffer);

	++nfru_frame_index;
	return true;
}
/*******************************
 * GUI
 *******************************/

void NFRUSample::draw_gui()
{
	get_gui().show_options_window(
	    [this]() {
		    ImGui::Text("Mode: %s", mode == Mode::Realtime ? "Realtime" : "Dataset");
		    ImGui::Text("Selectable stages: %s", use_fragment_path ? "Fragment" : "Compute");
		    ImGui::Separator();

		    if (mode == Mode::Dataset)
		    {
			    const uint64_t total = dataset_loader->total_frames();
			    ImGui::Text("Frame: %llu / %llu",
			                static_cast<unsigned long long>(scene_frame_index),
			                static_cast<unsigned long long>(total));
			    if (scene_frame_index >= total && total > 0)
				    ImGui::TextColored({0.4f, 1.0f, 0.4f, 1.0f}, "[Completed]");
		    }
		    else
		    {
			    ImGui::Text("Frame: %llu", static_cast<unsigned long long>(scene_frame_index));
		    }

		    // Per-frame display label: which image is currently shown
		    if (frame_is_generated)
			    ImGui::TextColored({0.4f, 1.0f, 0.4f, 1.0f}, "Frame Generated by NFRU");
		    else
			    ImGui::TextColored({1.0f, 0.85f, 0.4f, 1.0f}, "Original");

		    ImGui::Separator();

		    bool debug_view_prev = nfru_debug_view;
		    ImGui::Checkbox("NFRU Debug View", &nfru_debug_view);
		    if (nfru_debug_view != debug_view_prev)
			    configure_nfru();

		    bool debug_checking_prev = nfru_debug_checking;
		    ImGui::Checkbox("NFRU Debug Logging", &nfru_debug_checking);
		    if (nfru_debug_checking != debug_checking_prev)
			    configure_nfru();

		    if (save_exr_enabled)
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

/*******************************
 * barrier helpers
 *******************************/

void NFRUSample::prepare_render_inputs(vkb::core::CommandBufferC &command_buffer, FrameInputs &inputs)
{
	// Prepare scene inputs for attachment writes:
	// - Color and motion: UNDEFINED (first use) or SHADER_READ_ONLY_OPTIMAL (reuse)
	//   -> COLOR_ATTACHMENT_OPTIMAL.
	// - Depth: UNDEFINED (first use) or SHADER_READ_ONLY_OPTIMAL (reuse)
	//   -> DEPTH_STENCIL_ATTACHMENT_OPTIMAL.
	vkb::ImageMemoryBarrier barrier{};
	barrier.new_layout      = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
	barrier.dst_access_mask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
	barrier.dst_stage_mask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	auto transition         = [&](ImageResource &resource) {
        if (!resource.initialized)
        {
            barrier.old_layout      = VK_IMAGE_LAYOUT_UNDEFINED;
            barrier.src_access_mask = 0;
            barrier.src_stage_mask  = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        }
        else
        {
            barrier.old_layout      = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            barrier.src_access_mask = VK_ACCESS_SHADER_READ_BIT;
            barrier.src_stage_mask  = frag_or_comp_config.input_read_stage |
                                     VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        }
        command_buffer.image_memory_barrier(*resource.view, barrier);
	};
	transition(inputs.color);
	transition(inputs.motion_vectors);

	barrier.new_layout      = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
	barrier.dst_access_mask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
	barrier.dst_stage_mask  = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
	transition(inputs.depth);

	inputs.render_target->set_layout(static_cast<int>(Attachments::Color), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
	inputs.render_target->set_layout(static_cast<int>(Attachments::Velocity), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
	inputs.render_target->set_layout(static_cast<int>(Attachments::Depth), VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
}

void NFRUSample::finish_render_inputs(vkb::core::CommandBufferC &command_buffer, FrameInputs &inputs)
{
	// Publish the rendered scene inputs:
	// - Color: COLOR_ATTACHMENT_OPTIMAL -> TRANSFER_SRC_OPTIMAL for a ground-truth
	//   readback, otherwise SHADER_READ_ONLY_OPTIMAL.
	// - Motion: COLOR_ATTACHMENT_OPTIMAL -> SHADER_READ_ONLY_OPTIMAL.
	// - Depth: DEPTH_STENCIL_ATTACHMENT_OPTIMAL -> SHADER_READ_ONLY_OPTIMAL.
	const bool                 readback_color   = frame_should_save && !frame_is_generated;
	const VkPipelineStageFlags input_read_stage = frag_or_comp_config.input_read_stage |
	                                              VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;

	vkb::ImageMemoryBarrier barrier{};
	barrier.old_layout      = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
	barrier.src_access_mask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
	barrier.src_stage_mask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	if (readback_color)
	{
		barrier.new_layout      = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
		barrier.dst_access_mask = VK_ACCESS_TRANSFER_READ_BIT;
		barrier.dst_stage_mask  = VK_PIPELINE_STAGE_TRANSFER_BIT;
	}
	else
	{
		barrier.new_layout      = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		barrier.dst_access_mask = VK_ACCESS_SHADER_READ_BIT;
		barrier.dst_stage_mask  = input_read_stage;
	}
	command_buffer.image_memory_barrier(*inputs.color.view, barrier);

	barrier.new_layout      = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	barrier.dst_access_mask = VK_ACCESS_SHADER_READ_BIT;
	barrier.dst_stage_mask  = input_read_stage;
	command_buffer.image_memory_barrier(*inputs.motion_vectors.view, barrier);

	barrier.old_layout      = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
	barrier.src_access_mask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
	barrier.src_stage_mask  = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
	                         VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
	command_buffer.image_memory_barrier(*inputs.depth.view, barrier);

	inputs.color.initialized          = true;
	inputs.motion_vectors.initialized = true;
	inputs.depth.initialized          = true;

	inputs.render_target->set_layout(static_cast<int>(Attachments::Color), readback_color ? VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
	inputs.render_target->set_layout(static_cast<int>(Attachments::Velocity), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
	inputs.render_target->set_layout(static_cast<int>(Attachments::Depth), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}

void NFRUSample::prepare_upload_inputs(vkb::core::CommandBufferC &command_buffer, FrameInputs &inputs)
{
	// Prepare all dataset inputs for upload:
	// UNDEFINED (first use) or SHADER_READ_ONLY_OPTIMAL (reuse)
	// -> TRANSFER_DST_OPTIMAL.
	// On reuse, the previous consumers are NFRU/history sampling. The next producers
	// are buffer-to-image writes for color, depth, and motion.
	vkb::ImageMemoryBarrier barrier{};
	barrier.new_layout      = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
	barrier.dst_access_mask = VK_ACCESS_TRANSFER_WRITE_BIT;
	barrier.dst_stage_mask  = VK_PIPELINE_STAGE_TRANSFER_BIT;
	auto transition         = [&](ImageResource &resource) {
        if (!resource.initialized)
        {
            barrier.old_layout      = VK_IMAGE_LAYOUT_UNDEFINED;
            barrier.src_access_mask = 0;
            barrier.src_stage_mask  = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        }
        else
        {
            barrier.old_layout      = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            barrier.src_access_mask = VK_ACCESS_SHADER_READ_BIT;
            barrier.src_stage_mask  = frag_or_comp_config.input_read_stage;
        }
        command_buffer.image_memory_barrier(*resource.view, barrier);
	};
	transition(inputs.color);
	transition(inputs.depth);
	transition(inputs.motion_vectors);
}

void NFRUSample::finish_upload_inputs(vkb::core::CommandBufferC &command_buffer, FrameInputs &inputs)
{
	// Publish all uploaded dataset inputs:
	// TRANSFER_DST_OPTIMAL -> SHADER_READ_ONLY_OPTIMAL.
	// The producers are buffer-to-image transfer writes. The next consumers are
	// NFRU input and application-managed history sampling.
	// prepare_upload_inputs() synchronizes previous shader reads before upload, so
	// this source scope contains only the new transfer writes.
	vkb::ImageMemoryBarrier barrier{};
	barrier.old_layout      = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
	barrier.new_layout      = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	barrier.src_access_mask = VK_ACCESS_TRANSFER_WRITE_BIT;
	barrier.dst_access_mask = VK_ACCESS_SHADER_READ_BIT;
	barrier.src_stage_mask  = VK_PIPELINE_STAGE_TRANSFER_BIT;
	barrier.dst_stage_mask  = frag_or_comp_config.input_read_stage;
	command_buffer.image_memory_barrier(*inputs.color.view, barrier);
	command_buffer.image_memory_barrier(*inputs.depth.view, barrier);
	command_buffer.image_memory_barrier(*inputs.motion_vectors.view, barrier);

	inputs.color.initialized          = true;
	inputs.depth.initialized          = true;
	inputs.motion_vectors.initialized = true;
}

void NFRUSample::prepare_output(vkb::core::CommandBufferC &command_buffer)
{
	// Prepare the NFRU output for this dispatch:
	// UNDEFINED on first use, or SHADER_READ_ONLY_OPTIMAL after display/readback,
	// to the selected compute or fragment output-write layout.
	vkb::ImageMemoryBarrier barrier{};
	if (!nfru_output.initialized)
	{
		// No sample or SDK operation has used this output image yet.
		barrier.old_layout      = VK_IMAGE_LAYOUT_UNDEFINED;
		barrier.src_access_mask = 0;
		barrier.src_stage_mask  = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
	}
	else
	{
		barrier.old_layout      = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		barrier.src_access_mask = VK_ACCESS_SHADER_READ_BIT;
		barrier.src_stage_mask  = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
	}

	barrier.new_layout      = frag_or_comp_config.output_write_layout;
	barrier.dst_access_mask = frag_or_comp_config.output_write_access;
	barrier.dst_stage_mask  = frag_or_comp_config.output_write_stage;
	command_buffer.image_memory_barrier(*nfru_output.view, barrier);
}

void NFRUSample::finish_output(vkb::core::CommandBufferC &command_buffer)
{
	// Frame 0 initializes history and is not saved. Later saved outputs go directly
	// to transfer source; all other outputs become shader-readable for display.
	const bool              readback_output = frame_should_save;
	vkb::ImageMemoryBarrier barrier{};
	barrier.old_layout      = frag_or_comp_config.output_write_layout;
	barrier.src_access_mask = frag_or_comp_config.output_write_access;
	barrier.src_stage_mask  = frag_or_comp_config.output_write_stage;
	if (readback_output)
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
	}
	command_buffer.image_memory_barrier(*nfru_output.view, barrier);
	nfru_output.initialized = true;
}

void NFRUSample::initialize_external_history(vkb::core::CommandBufferC &command_buffer, FrameInputs &previous)
{
	if (nfru_manage_history || nfru_frame_index > 0)
		return;

	// Initialize application-managed history through two transitions:
	// previous color/depth UNDEFINED -> TRANSFER_DST_OPTIMAL for their clears, then
	// TRANSFER_DST_OPTIMAL -> SHADER_READ_ONLY_OPTIMAL for the first NFRU history
	// read. Motion vectors are current-frame inputs and are not part of NFRU
	// application-managed history.
	vkb::ImageMemoryBarrier barrier{};
	barrier.old_layout      = VK_IMAGE_LAYOUT_UNDEFINED;
	barrier.new_layout      = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
	barrier.src_access_mask = 0;
	barrier.dst_access_mask = VK_ACCESS_TRANSFER_WRITE_BIT;
	barrier.src_stage_mask  = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
	barrier.dst_stage_mask  = VK_PIPELINE_STAGE_TRANSFER_BIT;
	command_buffer.image_memory_barrier(*previous.color.view, barrier);
	command_buffer.image_memory_barrier(*previous.depth.view, barrier);

	VkImageSubresourceRange color_range{};
	color_range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	color_range.levelCount = 1;
	color_range.layerCount = 1;
	VkClearColorValue clear_color{};
	vkCmdClearColorImage(command_buffer.get_handle(), previous.color.image->get_handle(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clear_color, 1, &color_range);

	VkImageSubresourceRange depth_range{};
	depth_range.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
	depth_range.levelCount = 1;
	depth_range.layerCount = 1;
	bool depth_inverted    = false;
	if (mode == Mode::Dataset)
	{
		depth_inverted = dataset_loader->uses_inverted_depth();
	}
	else
	{
		depth_inverted = true;
	}
	VkClearDepthStencilValue clear_depth{depth_inverted ? 0.0f : 1.0f, 0};
	vkCmdClearDepthStencilImage(command_buffer.get_handle(), previous.depth.image->get_handle(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clear_depth, 1, &depth_range);

	barrier.old_layout      = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
	barrier.new_layout      = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	barrier.src_access_mask = VK_ACCESS_TRANSFER_WRITE_BIT;
	barrier.dst_access_mask = VK_ACCESS_SHADER_READ_BIT;
	barrier.src_stage_mask  = VK_PIPELINE_STAGE_TRANSFER_BIT;
	barrier.dst_stage_mask  = frag_or_comp_config.input_read_stage;
	command_buffer.image_memory_barrier(*previous.color.view, barrier);
	command_buffer.image_memory_barrier(*previous.depth.view, barrier);
	previous.color.initialized = true;
	previous.depth.initialized = true;
}

/*******************************
 * readback buffers
 *******************************/
void NFRUSample::create_readback_buffers(VkExtent2D extent, size_t count)
{
	size_t     buffer_size  = 0;
	const bool valid_format = vkb::sample::exr::vk_format_buffer_size(extent, VK_FORMAT_R8G8B8A8_UNORM, buffer_size);
	assert(valid_format);

	readback_slots.resize(count);
	vkb::core::BufferBuilderC builder(buffer_size);
	builder.with_usage(VK_BUFFER_USAGE_TRANSFER_DST_BIT)
	    .with_vma_usage(VMA_MEMORY_USAGE_GPU_TO_CPU)
	    .with_vma_flags(VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT);
	for (ReadbackSlot &slot : readback_slots)
		slot.buffer = builder.build_unique(get_device());

	LOGI("Created {} NFRU EXR readback buffers ({} bytes each)", count, buffer_size);
}

void NFRUSample::copy_to_readback(vkb::core::CommandBufferC  &command_buffer,
                                  const vkb::core::ImageView &source_view,
                                  vkb::core::BufferC         &destination)
{
	const auto       &source_image = source_view.get_image();
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
	command_buffer.image_memory_barrier(source_view, barrier);
}

bool NFRUSample::save_completed_readback(ReadbackSlot &slot, VkExtent2D extent)
{
	if (!slot.need_save)
		return true;

	slot.need_save = false;

	char filename[32];
	snprintf(filename, sizeof(filename), "%04llu.exr", static_cast<unsigned long long>(slot.frame_number));
	slot.buffer->invalidate();
	return vkb::sample::exr::save_vk_format_to_exr((slot.output_dir + "/" + filename).c_str(),
	                                               VK_FORMAT_R8G8B8A8_UNORM, slot.buffer->get_data(),
	                                               static_cast<size_t>(slot.buffer->get_size()), extent);
}

std::unique_ptr<vkb::VulkanSampleC> create_nfru()
{
	return std::make_unique<NFRUSample>();
}
