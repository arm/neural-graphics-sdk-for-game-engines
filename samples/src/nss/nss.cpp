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

#include "nss.h"

#include <string>

#include "common/app_argument_reader.h"
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
}

NSSSample::~NSSSample()
{
	if (get_device().get_handle() != VK_NULL_HANDLE)
	{
		vkDeviceWaitIdle(get_device().get_handle());
		LOGI("GPU work completed, safe to destroy resources");
	}

	if (nss_enabled && nss_context != nullptr)
	{
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

	// Automatic Resource Cleanup
}

NSSSample::NSSSample()
{
	// ========================================
	// Vulkan API Version Requirement
	// ========================================
	set_api_version(VK_API_VERSION_1_4);

	// ========================================
	// NSS SDK Required Extensions
	// https://docs.vulkan.org/refpages/latest/refpages/source/VK_ARM_data_graph.html
	// ========================================
	add_device_extension("VK_KHR_dynamic_rendering", true);               // Required by VK_KHR_maintenance5
	add_device_extension("VK_KHR_maintenance5", true);                    // Required by VK_ARM_data_graph
	add_device_extension("VK_KHR_deferred_host_operations", true);        // Required by VK_ARM_data_graph

	// ARM ML extensions
	add_device_extension("VK_ARM_data_graph", true);
	add_device_extension("VK_ARM_tensors", true);
}

void NSSSample::request_gpu_features(vkb::core::PhysicalDeviceC &gpu)
{
	// Call parent implementation first
	vkb::VulkanSampleC::request_gpu_features(gpu);

	// ========================================
	// NSS SDK Required GPU Features
	// ========================================

	// Int8 Support: NSS uses int8 for quantized model operations
	REQUEST_REQUIRED_FEATURE(gpu, VkPhysicalDeviceVulkan12Features, shaderInt8);

	// Synchronization2: NSS uses vkCmdPipelineBarrier2 for pipeline barriers
	REQUEST_REQUIRED_FEATURE(gpu, VkPhysicalDeviceVulkan13Features, synchronization2);

	// ========================================
	// ARM Vendor Extensions (Android/ARM Mali GPUs)
	// ========================================

	REQUEST_OPTIONAL_FEATURE(gpu, VkPhysicalDeviceTensorFeaturesARM, shaderTensorAccess);
	REQUEST_OPTIONAL_FEATURE(gpu, VkPhysicalDeviceTensorFeaturesARM, tensors);
	REQUEST_OPTIONAL_FEATURE(gpu, VkPhysicalDeviceDataGraphFeaturesARM, dataGraph);
}

static void OnNSSMessage(uint32_t type, const char *message)
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

void NSSSample::initialize_nss_context(const VkExtent2D &low_res_extent, const VkExtent2D &display_extent)
{
	// Check if NSS is force-disabled by argument
	if (nss_force_disabled)
	{
		LOGI("NSS disabled by NSS_ENABLE=0, skipping context creation");
		return;
	}

	// Get Vulkan function pointers for FFX backend
#if defined(_HPP_VULKAN_LIBRARY)
	static vk::detail::DynamicLoader dl(_HPP_VULKAN_LIBRARY);
#else
	static vk::detail::DynamicLoader dl;
#endif
	PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr =
	    dl.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");

	PFN_vkGetDeviceProcAddr vkGetDeviceProcAddr =
	    reinterpret_cast<PFN_vkGetDeviceProcAddr>(
	        vkGetInstanceProcAddr(get_instance().get_handle(), "vkGetDeviceProcAddr"));

	// Configure NSS context flags
	uint32_t nssFlags =
	    FFX_API_NSS_CONTEXT_FLAG_QUANTIZED |             // Use quantized model (required)
	    FFX_API_NSS_CONTEXT_FLAG_HIGH_DYNAMIC_RANGE |
	    FFX_API_NSS_CONTEXT_FLAG_DEPTH_INVERTED |        // Reversed depth (1.0 = near, 0.0 = far)
	    FFX_API_NSS_CONTEXT_FLAG_MANAGE_HISTORY |        // Let SDK manage history ping-pong
	    frag_or_comp_config.context_flags;               // PRE/POST process fragment flags (0 for compute)

	// Apply override flags from argument
	if (nss_override_flags != 0)
		nssFlags = nss_override_flags;

	// Create Vulkan backend + NSS context descriptor
	ffx::CreateBackendVKDesc backendDesc{};
	backendDesc.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK;
	backendDesc.vkInstance            = get_instance().get_handle();
	backendDesc.vkDevice              = get_device().get_handle();
	backendDesc.vkPhysicalDevice      = get_device().get_gpu().get_handle();
	backendDesc.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
	backendDesc.vkDeviceProcAddr      = vkGetDeviceProcAddr;

	ffx::CreateContextDescNss nss_desc{};
	nss_desc.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_NSS;
	nss_desc.maxRenderSize.width   = low_res_extent.width;
	nss_desc.maxRenderSize.height  = low_res_extent.height;
	nss_desc.maxUpscaleSize.width  = display_extent.width;
	nss_desc.maxUpscaleSize.height = display_extent.height;
	nss_desc.flags                 = nssFlags;
	nss_desc.fpMessage             = OnNSSMessage;
	nss_desc.qualityMode           = nss_quality_mode;

	// Create NSS context
	ffx::ReturnCode result = ffx::CreateContext(nss_context, nullptr, nss_desc, backendDesc);
	if (result != ffx::ReturnCode::Ok)
	{
		LOGW("Failed to create NSS context! Error code: {}, NSS features will be disabled", static_cast<uint32_t>(result));
		return;
	}

	LOGI("NSS Context created successfully!");
	nss_enabled = true;
}

bool NSSSample::prepare(const vkb::ApplicationOptions &options)
{
	if (!vkb::VulkanSampleC::prepare(options))
	{
		return false;
	}

	// ------------------Command Line Configuration------------------------------
	// Example:
	//   vulkan_samples sample nss --app-arg NSS_ENABLE=1,NSS_SCALE_FACTOR=2.0,NSS_FLAGS=0x1C7,NSS_QUALITY=1,NSS_DEBUG_VIEW_MODE=0
	//
	// Supported keys:
	//   NSS_ENABLE       : 0 or 1 (default 1)
	//   NSS_SCALE_FACTOR : float in [1.0, 3.0] (default 2.0)
	//   NSS_FLAGS        : uint32 (decimal or 0x-prefixed hex), default 0
	//   NSS_QUALITY      : 0 (Quality), 1 (Balanced), 2 (Performance), default 1
	//   NSS_DEBUG_VIEW_MODE : uint32 in [0, 16], default 0 (0=overview, 1..16=single tile)
	//   NSS_USE_FRAGMENT : 0 (compute path) or 1 (fragment shader path), default 1
	vkb::AppArgumentReader args{options.app_arguments};

	if (args.has("NSS_ENABLE"))
	{
		nss_force_disabled = !args.get_bool("NSS_ENABLE", true);
		LOGI("NSS {} via NSS_ENABLE", nss_force_disabled ? "disabled" : "enabled");
	}

	nss_scale_factor = args.get_float("NSS_SCALE_FACTOR", nss_scale_factor, 1.0f, 3.0f);

	nss_override_flags = args.get_uint("NSS_FLAGS", nss_override_flags);

	nss_quality_mode = static_cast<FfxApiNssShaderQualityMode>(
	    args.get_int("NSS_QUALITY", static_cast<int>(nss_quality_mode),
	                 FFX_API_NSS_SHADER_QUALITY_MODE_QUALITY, FFX_API_NSS_SHADER_QUALITY_MODE_PERFORMANCE));

	if (args.has("NSS_DEBUG_VIEW_MODE"))
	{
		nss_debug_view_mode = args.get_uint("NSS_DEBUG_VIEW_MODE", nss_debug_view_mode, 0, 16);
		display_mode        = DisplayMode::NSSDebugView;
	}

	use_fragment_path = args.get_bool("NSS_USE_FRAGMENT", use_fragment_path);

	frag_or_comp_config = use_fragment_path
	    ? FragOrCompConfig{VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
	                  FFX_API_NSS_CONTEXT_FLAG_PRE_PROCESS_FRAGMENT | FFX_API_NSS_CONTEXT_FLAG_POST_PROCESS_FRAGMENT,
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

	// ------------------Load scenes and initial resolution------------------------------
	load_scene("scenes/sponza/Sponza01.gltf");

	auto initial_extent = get_render_context().get_surface_extent();
	resolution_config   = ResolutionConfig::create(initial_extent, nss_scale_factor);

	LOGI("=== Resolution Configuration ===");
	LOGI("  Display (high-res): {}x{}", resolution_config.display_extent.width, resolution_config.display_extent.height);
	LOGI("  Render (low-res):   {}x{}", resolution_config.render_extent.width, resolution_config.render_extent.height);
	LOGI("  Scale factor:       {:.2f}x", resolution_config.scale_factor);
	LOGI("  Mipmap LOD bias:    {:.2f}", resolution_config.mipmap_lod_bias);

	// ------------------NSS Context Creation------------------------------
	initialize_nss_context(resolution_config.render_extent, resolution_config.display_extent);

	// Adjust scale factor if NSS failed to initialize
	if (!nss_enabled)
	{
		nss_scale_factor        = 1.0f;
		use_nss_jitter_sequence = false;        // Fall back to Halton sequence
		nss_jitter_phase_count  = 0;
		// Recalculate resolution config with scale factor = 1.0
		resolution_config = ResolutionConfig::create(initial_extent, 1.0f);
		LOGI("NSS disabled: rendering at native resolution (scale factor = 1.0)");
	}
	else if (use_nss_jitter_sequence)
	{
		int32_t jitter_phase_count = 0;
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

	// Create low-res render targets for scene rendering
	create_low_res_render_targets();

	// Create NSS output images (only if NSS enabled)
	create_nss_output_images();

	// ------------------Scene Camera Setup------------------------------

	auto &camera_node = vkb::add_free_camera(get_scene(), "main_camera", resolution_config.render_extent);
	camera            = dynamic_cast<vkb::sg::PerspectiveCamera *>(&camera_node.get_component<vkb::sg::Camera>());

	// ------------------Scene Rendering Pipeline------------------------------
	// Use jitter & velocity shader with custom subpass.
	vkb::ShaderSource vert_shader("temporal/jitter.vert.spv");
	vkb::ShaderSource frag_shader("temporal/velocity.frag.spv");
	auto              scene_subpass = std::make_unique<vkb::TemporalForwardSubpass>(get_render_context(), std::move(vert_shader), std::move(frag_shader), get_scene(), *camera);
	temporal_subpass           = scene_subpass.get();
	scene_subpass->set_output_attachments({static_cast<int>(Attachments::Color), static_cast<int>(Attachments::Velocity)});
	scene_subpass->set_mipmap_lod_bias(resolution_config.mipmap_lod_bias);
	scene_subpass->set_jitter_scale(1.0f);

	scene_pipeline = std::make_unique<vkb::RenderPipeline>();
	scene_pipeline->add_subpass(std::move(scene_subpass));
	scene_pipeline->set_load_store(scene_load_store);
	{
		VkClearValue depth_clear{};
		depth_clear.depthStencil = {0.0f, 0U};        // reversed depth: 0.0 = far
		VkClearValue color_clear{};
		color_clear.color = {0.0f, 0.0f, 0.0f, 1.0f}; // black sky
		VkClearValue vel_clear{};
		vel_clear.color = {0.0f, 0.0f, 0.0f, 0.0f};   // no motion
		scene_pipeline->set_clear_value({depth_clear, color_clear, vel_clear});
	}

	// ------------------Post-processing Pipeline------------------------------

	// Post-processing pass (display)
	vkb::ShaderSource postprocessing_vs("postprocessing.vert.spv");
	postprocessing_pipeline = std::make_unique<vkb::PostProcessingPipeline>(get_render_context(), std::move(postprocessing_vs));
	postprocessing_pipeline->add_pass().add_subpass(vkb::ShaderSource("display.frag.spv"));

	get_stats().request_stats({vkb::StatIndex::frame_times});
	create_gui(*window, &get_stats());

	return true;
}

void NSSSample::update(float delta_time)
{
	last_delta_time = delta_time;

	VulkanSample::update(delta_time);

	if (first_frame)
	{
		first_frame = false;
	}
}

/********************************
 * create buffer
 *********************************/
void NSSSample::create_low_res_render_targets()
{
	auto  &swapchain       = get_render_context().get_swapchain();
	size_t swapchain_count = swapchain.get_images().size();

	// Use centralized resolution configuration
	VkExtent3D low_res_extent = {
	    resolution_config.render_extent.width,
	    resolution_config.render_extent.height,
	    1};

	// Formats
	auto depth_format    = VK_FORMAT_D32_SFLOAT;
	auto color_format    = VK_FORMAT_B10G11R11_UFLOAT_PACK32;
	auto velocity_format = VK_FORMAT_R16G16_SFLOAT;

	low_res_render_targets.reserve(swapchain_count);

	// Configure load/store operations
	scene_load_store.clear();
	scene_load_store.push_back({VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE});        // Depth
	scene_load_store.push_back({VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE});        // Color
	scene_load_store.push_back({VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE});        // Velocity

	// Create one render target for each swapchain image
	for (size_t i = 0; i < swapchain_count; ++i)
	{
		std::vector<vkb::core::Image> images;

		// Attachment 0 - Depth (low res)
		VkImageUsageFlags depth_usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
		vkb::core::Image  depth_image{get_device(), low_res_extent, depth_format, depth_usage, VMA_MEMORY_USAGE_GPU_ONLY};
		depth_image.set_debug_name("Low-Res Depth " + std::to_string(i));
		images.push_back(std::move(depth_image));

		// Attachment 1 - Color (low res)
		VkImageUsageFlags color_usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
		vkb::core::Image  color_image{get_device(), low_res_extent, color_format, color_usage, VMA_MEMORY_USAGE_GPU_ONLY};
		color_image.set_debug_name("Low-Res Color " + std::to_string(i));
		images.push_back(std::move(color_image));

		// Attachment 2 - Velocity (low res)
		VkImageUsageFlags velocity_usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
		vkb::core::Image  velocity_image{get_device(), low_res_extent, velocity_format, velocity_usage, VMA_MEMORY_USAGE_GPU_ONLY};
		velocity_image.set_debug_name("Low-Res Velocity " + std::to_string(i));
		images.push_back(std::move(velocity_image));

		// Create render target
		auto render_target = std::make_unique<vkb::RenderTarget>(std::move(images));
		low_res_render_targets.push_back(std::move(render_target));
	}
}

void NSSSample::create_nss_output_images()
{
	if (!nss_enabled)
	{
		LOGI("NSS not enabled, skipping NSS output image creation");
		return;
	}

	auto  &swapchain       = get_render_context().get_swapchain();
	size_t swapchain_count = swapchain.get_images().size();

	// Format
	auto color_format = VK_FORMAT_B10G11R11_UFLOAT_PACK32;

	// Usage flags for NSS output
	VkImageUsageFlags nss_output_usage = VK_IMAGE_USAGE_SAMPLED_BIT |
	                                      VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
	                                      VK_IMAGE_USAGE_TRANSFER_DST_BIT |
	                                      frag_or_comp_config.output_usage_bit;

	nss_output_images.reserve(swapchain_count);
	nss_output_views.reserve(swapchain_count);
	nss_debug_images.reserve(swapchain_count);
	nss_debug_views.reserve(swapchain_count);

	// Create output images for each swapchain image
	VkExtent3D display_extent_3d = {
	    resolution_config.display_extent.width,
	    resolution_config.display_extent.height,
	    1};

	for (size_t i = 0; i < swapchain_count; ++i)
	{
		// NSS output image (high-res)
		auto nss_output_img = std::make_unique<vkb::core::Image>(
		    get_device(),
		    display_extent_3d,
		    color_format,
		    nss_output_usage,
		    VMA_MEMORY_USAGE_GPU_ONLY);
		nss_output_img->set_debug_name("NSS Output " + std::to_string(i));

		auto nss_output_view = std::make_unique<vkb::core::ImageView>(
		    *nss_output_img,
		    VK_IMAGE_VIEW_TYPE_2D);

		nss_output_images.push_back(std::move(nss_output_img));
		nss_output_views.push_back(std::move(nss_output_view));

		// NSS debug view image (same format/size as output; written when debug view is active)
		auto nss_debug_img = std::make_unique<vkb::core::Image>(
		    get_device(),
		    display_extent_3d,
		    color_format,
		    nss_output_usage,
		    VMA_MEMORY_USAGE_GPU_ONLY);
		nss_debug_img->set_debug_name("NSS Debug View " + std::to_string(i));

		auto nss_debug_view = std::make_unique<vkb::core::ImageView>(
		    *nss_debug_img,
		    VK_IMAGE_VIEW_TYPE_2D);

		nss_debug_images.push_back(std::move(nss_debug_img));
		nss_debug_views.push_back(std::move(nss_debug_view));
	}
}

/********************************
 * draw
 *********************************/
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

	const auto desc = ffxApiGetImageResourceDescriptionVK(image.get_handle(), create_info, additional_usages);
	return ffxApiGetResourceVK(reinterpret_cast<void *>(image.get_handle()), desc, static_cast<uint32_t>(state));
}

void NSSSample::draw(vkb::core::CommandBufferC &command_buffer, vkb::RenderTarget &swapchain_render_target)
{
	// Get current frame index for accessing per-frame resources
	uint32_t frame_index = get_render_context().get_active_frame_index();

	// Use NSS SDK jitter sequence via ffx-api query descriptors
	if (use_nss_jitter_sequence && nss_jitter_phase_count > 0)
	{

		float jitter_x = 0.0f;
		float jitter_y = 0.0f;

		ffx::QueryDescNssGetJitterOffset offset_query{};
		offset_query.index      = static_cast<int32_t>(temporal_subpass->peek_frame_index());
		offset_query.phaseCount = nss_jitter_phase_count;
		offset_query.pOutX      = &jitter_x;
		offset_query.pOutY      = &jitter_y;
		ffx::ReturnCode result  = ffx::Query(offset_query);

		if (result == ffx::ReturnCode::Ok)
		{
			temporal_subpass->set_external_jitter({jitter_x, jitter_y});
		}
		else
		{
			use_nss_jitter_sequence = false;
			LOGW("NSS jitter query failed with code {}, falling back to Halton sequence", static_cast<uint32_t>(result));
			temporal_subpass->clear_external_jitter();
		}
	}

	// Get low-res scene render target for this frame
	assert(frame_index < low_res_render_targets.size() && "Low-res render target not created for this frame");
	auto &scene_rt     = *low_res_render_targets[frame_index];
	auto &scene_views  = scene_rt.get_views();
	auto  scene_extent = scene_rt.get_extent();

	// Swapchain extent (high-res) for NSS output and postprocessing
	auto  swapchain_extent = swapchain_render_target.get_extent();
	auto &swapchain_views  = swapchain_render_target.get_views();

	// ============================================================================
	// PHASE 0: Transition swapchain image layout
	// ============================================================================
	// Swapchain image needs to be transitioned from PRESENT_SRC (from last frame)
	// to COLOR_ATTACHMENT_OPTIMAL for rendering
	{
		vkb::ImageMemoryBarrier swapchain_barrier{};
		swapchain_barrier.old_layout      = VK_IMAGE_LAYOUT_UNDEFINED;
		swapchain_barrier.new_layout      = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
		swapchain_barrier.src_access_mask = 0;
		swapchain_barrier.dst_access_mask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		swapchain_barrier.src_stage_mask  = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
		swapchain_barrier.dst_stage_mask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

		command_buffer.image_memory_barrier(swapchain_views[0], swapchain_barrier);
		swapchain_render_target.set_layout(0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
	}

	// ============================================================================
	// PHASE 1: Scene Rendering (Low Resolution)
	// ============================================================================

	// Transition scene render targets to optimal layouts
	{
		vkb::ImageMemoryBarrier memory_barrier{};
		memory_barrier.old_layout      = VK_IMAGE_LAYOUT_UNDEFINED;
		memory_barrier.new_layout      = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
		memory_barrier.src_access_mask = 0;
		memory_barrier.dst_access_mask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		memory_barrier.src_stage_mask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		memory_barrier.dst_stage_mask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

		// Low-res color and velocity attachments
		command_buffer.image_memory_barrier(scene_views[static_cast<int>(Attachments::Color)], memory_barrier);
		scene_rt.set_layout(static_cast<int>(Attachments::Color), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
		command_buffer.image_memory_barrier(scene_views[static_cast<int>(Attachments::Velocity)], memory_barrier);
		scene_rt.set_layout(static_cast<int>(Attachments::Velocity), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
	}

	// Depth attachment
	{
		vkb::ImageMemoryBarrier memory_barrier{};
		memory_barrier.old_layout      = VK_IMAGE_LAYOUT_UNDEFINED;
		memory_barrier.new_layout      = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
		memory_barrier.src_access_mask = 0;
		memory_barrier.dst_access_mask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
		memory_barrier.src_stage_mask  = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
		memory_barrier.dst_stage_mask  = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;

		command_buffer.image_memory_barrier(scene_views[static_cast<int>(Attachments::Depth)], memory_barrier);
		scene_rt.set_layout(static_cast<int>(Attachments::Depth), VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
	}

	// Set viewport to low-resolution (scene is rendered at low-res)
	VkExtent2D low_res_extent = {scene_extent.width, scene_extent.height};

	VkViewport viewport{};
	viewport.width    = static_cast<float>(low_res_extent.width);
	viewport.height   = static_cast<float>(low_res_extent.height);
	viewport.minDepth = 0.0f;
	viewport.maxDepth = 1.0f;
	command_buffer.set_viewport(0, {viewport});

	VkRect2D scissor{};
	scissor.extent = low_res_extent;
	command_buffer.set_scissor(0, {scissor});

	// Render low-res scene to scene_rt (NOT swapchain_render_target)
	scene_pipeline->draw(command_buffer, scene_rt);
	command_buffer.end_render_pass();

	// ============================================================================
	// PHASE 2: NSS Dispatch (Upscaling from Low-Res to High-Res)
	// ============================================================================

	if (nss_enabled && !nss_output_images.empty())
	{
		// Transition scene outputs to shader read for NSS input
		vkb::ImageMemoryBarrier barrier{};
		barrier.old_layout      = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
		barrier.new_layout      = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		barrier.src_access_mask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		barrier.dst_access_mask = VK_ACCESS_SHADER_READ_BIT;
		barrier.src_stage_mask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		barrier.dst_stage_mask  = frag_or_comp_config.input_read_stage;

		command_buffer.image_memory_barrier(scene_views[static_cast<int>(Attachments::Color)], barrier);
		scene_rt.set_layout(static_cast<int>(Attachments::Color), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
		command_buffer.image_memory_barrier(scene_views[static_cast<int>(Attachments::Velocity)], barrier);
		scene_rt.set_layout(static_cast<int>(Attachments::Velocity), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

		// Transition depth to shader read
		barrier.old_layout      = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
		barrier.new_layout      = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		barrier.src_access_mask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
		barrier.dst_access_mask = VK_ACCESS_SHADER_READ_BIT;
		barrier.src_stage_mask  = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
		barrier.dst_stage_mask  = frag_or_comp_config.input_read_stage;
		command_buffer.image_memory_barrier(scene_views[static_cast<int>(Attachments::Depth)], barrier);
		scene_rt.set_layout(static_cast<int>(Attachments::Depth), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

		// Transition NSS output to GENERAL for NSS write
		barrier.old_layout      = VK_IMAGE_LAYOUT_UNDEFINED;
		barrier.new_layout      = VK_IMAGE_LAYOUT_GENERAL;
		barrier.src_access_mask = 0;
		barrier.dst_access_mask = frag_or_comp_config.output_write_access;
		barrier.src_stage_mask  = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
		barrier.dst_stage_mask  = frag_or_comp_config.output_write_stage;

		command_buffer.image_memory_barrier(*nss_output_views[frame_index], barrier);

		// Transition NSS debug view image to GENERAL when debug view mode is active
		if (display_mode == DisplayMode::NSSDebugView && !nss_debug_views.empty())
		{
			command_buffer.image_memory_barrier(*nss_debug_views[frame_index], barrier);
		}

		// ============================================================================
		// NSS Dispatch: execute NSS using the current low-res inputs.
		// ============================================================================

		// Get current frame images from scene render target
		auto &color_image    = scene_views[static_cast<int>(Attachments::Color)].get_image();
		auto &depth_image    = scene_views[static_cast<int>(Attachments::Depth)].get_image();
		auto &velocity_image = scene_views[static_cast<int>(Attachments::Velocity)].get_image();

		// Get jitter (NSS expects pixel-space jitter)
		glm::vec2 jitter_pixels_scaled = temporal_subpass->get_current_jitter_pixels_scaled();

		// Prepare dispatch descriptor
		ffxApiDispatchDescNss desc{};
		std::memset(&desc, 0, sizeof(ffxApiDispatchDescNss));
		desc.header.type = FFX_API_DISPATCH_DESC_TYPE_NSS;

		// Command buffer
		desc.commandList = reinterpret_cast<void *>(command_buffer.get_handle()); // VkCommandBuffer stored in void* (see ffxApiDispatchDescNss::commandList)

		// Input resources
		desc.color         = wrap_resource(color_image, frag_or_comp_config.input_read_state);
		desc.depth         = wrap_resource(depth_image, frag_or_comp_config.input_read_state);
		desc.motionVectors = wrap_resource(velocity_image, frag_or_comp_config.input_read_state);

		// Output resource
		desc.output = wrap_resource(*nss_output_images[frame_index], FFX_API_RESOURCE_STATE_UNORDERED_ACCESS);

		// Dispatch parameters
		desc.jitterOffset.x         = jitter_pixels_scaled.x;
		desc.jitterOffset.y         = jitter_pixels_scaled.y;
		desc.renderSize.width       = resolution_config.render_extent.width;
		desc.renderSize.height      = resolution_config.render_extent.height;
		desc.upscaleSize.width      = resolution_config.display_extent.width;
		desc.upscaleSize.height     = resolution_config.display_extent.height;
		desc.motionVectorScale.x    = static_cast<float>(resolution_config.render_extent.width);
		desc.motionVectorScale.y    = static_cast<float>(resolution_config.render_extent.height);
		desc.cameraFovAngleVertical = camera->get_field_of_view();
		desc.cameraNear             = camera->get_near_plane();
		desc.cameraFar              = camera->get_far_plane();
		desc.frameTimeDelta         = last_delta_time * 1000.0f;
		desc.reset                  = first_frame;
		desc.exposure               = 1.0f;

		// When NSS debug presentation is active, ask the SDK to write the selected debug view.
		if (display_mode == DisplayMode::NSSDebugView && !nss_debug_images.empty())
		{
			desc.flags         = FFX_API_NSS_DISPATCH_FLAG_DRAW_DEBUG_VIEW;
			desc.debugViews    = wrap_resource(*nss_debug_images[frame_index], FFX_API_RESOURCE_STATE_UNORDERED_ACCESS);
			desc.debugViewMode = nss_debug_view_mode;
		}

		// Direct call to FFX API
		ffx::ReturnCode nssResult = ffx::Dispatch(nss_context, desc);
		if (nssResult != ffx::ReturnCode::Ok)
		{
			LOGE("NSS dispatch failed with error code: {}", static_cast<uint32_t>(nssResult));
		}

		// Transition NSS output and debug view to shader read for postprocessing.
		barrier.old_layout      = VK_IMAGE_LAYOUT_GENERAL;
		barrier.src_access_mask = frag_or_comp_config.output_write_access;
		barrier.src_stage_mask  = frag_or_comp_config.output_write_stage;
		barrier.new_layout      = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		barrier.dst_access_mask = VK_ACCESS_SHADER_READ_BIT;
		barrier.dst_stage_mask  = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;

		command_buffer.image_memory_barrier(*nss_output_views[frame_index], barrier);

		// Transition NSS debug view to shader read if debug view was written
		if (display_mode == DisplayMode::NSSDebugView && !nss_debug_views.empty())
		{
			command_buffer.image_memory_barrier(*nss_debug_views[frame_index], barrier);
		}
	}
	else
	{
		vkb::ImageMemoryBarrier barrier{};
		barrier.old_layout      = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
		barrier.new_layout      = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		barrier.src_access_mask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		barrier.dst_access_mask = VK_ACCESS_SHADER_READ_BIT;
		barrier.src_stage_mask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		barrier.dst_stage_mask  = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;

		command_buffer.image_memory_barrier(scene_views[static_cast<int>(Attachments::Color)], barrier);
		scene_rt.set_layout(static_cast<int>(Attachments::Color), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
		command_buffer.image_memory_barrier(scene_views[static_cast<int>(Attachments::Velocity)], barrier);
		scene_rt.set_layout(static_cast<int>(Attachments::Velocity), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

		barrier.old_layout      = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
		barrier.src_access_mask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
		barrier.src_stage_mask  = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
		command_buffer.image_memory_barrier(scene_views[static_cast<int>(Attachments::Depth)], barrier);
		scene_rt.set_layout(static_cast<int>(Attachments::Depth), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
	}

	// ============================================================================
	// PHASE 3: Postprocessing (Present final output or NSS debug view)
	// ============================================================================

	// Reset viewport to full resolution for postprocessing
	viewport.width  = static_cast<float>(swapchain_extent.width);
	viewport.height = static_cast<float>(swapchain_extent.height);
	command_buffer.set_viewport(0, {viewport});

	scissor.extent = swapchain_extent;
	command_buffer.set_scissor(0, {scissor});

	auto &postprocessing_pass = postprocessing_pipeline->get_pass(0);
	auto &postprocessing_subpass = postprocessing_pass.get_subpass(0);

	if (nss_enabled && !nss_output_views.empty())
	{
		// Present either the NSS debug view or the normal NSS output.
		if (display_mode == DisplayMode::NSSDebugView && !nss_debug_views.empty())
		{
			postprocessing_subpass.bind_sampled_image("color_sampler", vkb::core::SampledImage(*nss_debug_views[frame_index]));
		}
		else
		{
			postprocessing_subpass.bind_sampled_image("color_sampler", vkb::core::SampledImage(*nss_output_views[frame_index]));
		}
	}
	else
	{
		// NSS disabled - use low-res scene color as final output
		postprocessing_subpass.bind_sampled_image("color_sampler", vkb::core::SampledImage(scene_views[static_cast<int>(Attachments::Color)]));
	}

	postprocessing_pipeline->draw(command_buffer, swapchain_render_target);

	// ============================================================================
	// PHASE 4: GUI Rendering
	// ============================================================================

	if (has_gui())
	{
		get_gui().draw(command_buffer);
	}

	command_buffer.end_render_pass();

	// ============================================================================
	// PHASE 5: Prepare Swapchain for Presentation
	// ============================================================================

	{
		vkb::ImageMemoryBarrier memory_barrier{};
		memory_barrier.old_layout      = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
		memory_barrier.new_layout      = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
		memory_barrier.src_access_mask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		memory_barrier.src_stage_mask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		memory_barrier.dst_stage_mask  = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;

		command_buffer.image_memory_barrier(swapchain_views[0], memory_barrier);
		swapchain_render_target.set_layout(0, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
	}
}

/*******************************
 * gui draw
 *******************************/
void NSSSample::draw_gui()
{
	get_gui().show_options_window(
	    /* body = */ [this]() {
		    // Desktop layout
		    ImGui::Text(" Neural Super Sampling Sample");
		    ImGui::Separator();
		    ImGui::Text("Display Mode:");
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
		    ImGui::Text("NSS Settings:");
		    float jitter_scale_ui = temporal_subpass->get_jitter_scale();
		    ImGui::SliderFloat("Jitter Scale", &jitter_scale_ui, 0.0f, 1.0f);
		    temporal_subpass->set_jitter_scale(jitter_scale_ui);

		    glm::vec2 jitter_pixels = temporal_subpass->get_current_jitter_pixels();
		    ImGui::Text("Current Jitter: (%.3f, %.3f) pixels", jitter_pixels.x, jitter_pixels.y);
		    ImGui::Text("ScaleFactor: %.3f  Flags: 0x%08X", nss_scale_factor, nss_override_flags);
		    ImGui::Text("Render: %ux%u  Display: %ux%u  LOD Bias: %.2f",
		                resolution_config.render_extent.width, resolution_config.render_extent.height,
		                resolution_config.display_extent.width, resolution_config.display_extent.height,
		                resolution_config.mipmap_lod_bias);
		    ImGui::Text("NSS Quality Mode: %d", nss_quality_mode);
		    ImGui::Text("Path: %s", use_fragment_path ? "Fragment" : "Compute");
		    ImGui::Text("NSS Enabled: %s", nss_enabled ? "Yes" : "No");
	    },
	    /* lines = */ 14);
}

std::unique_ptr<vkb::VulkanSampleC> create_nss()
{
	return std::make_unique<NSSSample>();
}
