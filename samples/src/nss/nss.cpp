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

// Vulkan Samples framework - Core
#include "core/device.h"
#include "filesystem/legacy.h"
#include "gltf_loader.h"
#include "gui.h"
#include "stats/stats.h"

// Vulkan Samples framework - Rendering
#include "rendering/postprocessing_renderpass.h"
#include "rendering/subpasses/forward_subpass.h"

// Vulkan Samples framework - Scene graph
#include "scene_graph/components/camera.h"

// Standard library headers (implementation-specific)
#include <algorithm>
#include <cstring>

/********************************
 * destructor and constructor
 *********************************/
NSSSample::~NSSSample()
{
	if (get_device().get_handle() != VK_NULL_HANDLE)
	{
		vkDeviceWaitIdle(get_device().get_handle());
		LOGI("GPU work completed, safe to destroy resources");
	}

	// Destroy NSS Context
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
	// NSS SDK requires Vulkan 1.3 for ARM extensions (VK_ARM_tensors, VK_ARM_data_graph)
	// Without this, device creation will fail with:
	//   "Missing extension required by VK_ARM_tensors: VK_VERSION_1_3"
	set_api_version(VK_API_VERSION_1_3);

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

	// Configure display mode options (keyboard shortcuts for quick switching)
	// Press 0-4 to switch between different visualization modes
	auto &config = get_configuration();
	config.insert<vkb::IntSetting>(0, reinterpret_cast<int &>(display_mode), static_cast<int>(DisplayMode::FinalOutput));
	config.insert<vkb::IntSetting>(1, reinterpret_cast<int &>(display_mode), static_cast<int>(DisplayMode::Velocity));
	config.insert<vkb::IntSetting>(2, reinterpret_cast<int &>(display_mode), static_cast<int>(DisplayMode::Depth));
	config.insert<vkb::IntSetting>(3, reinterpret_cast<int &>(display_mode), static_cast<int>(DisplayMode::NSSDebugView));
	config.insert<vkb::IntSetting>(4, reinterpret_cast<int &>(display_mode), static_cast<int>(DisplayMode::LowResColor));
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

	// emulation layer supports these features on non-ARM platforms,
	REQUEST_OPTIONAL_FEATURE(gpu, VkPhysicalDeviceTensorFeaturesARM, shaderTensorAccess);

	// ARM Mali only
	REQUEST_OPTIONAL_FEATURE(gpu, VkPhysicalDeviceTensorFeaturesARM, tensors);

	// ARM Mali only
	REQUEST_OPTIONAL_FEATURE(gpu, VkPhysicalDeviceDataGraphFeaturesARM, dataGraph);
}

/**************************************
 * prepare
 *  - create NSS context
 *  - create pipelines and resources
 **************************************/
void OnNSSMessage(uint32_t type, const wchar_t *message)
{
	// Convert wide-char to narrow-char for cross-platform logging
	// TODO: This simple conversion works for ASCII subset; for full Unicode support,
	// consider using std::wstring_convert or platform-specific APIs
	std::wstring ws(message);
	std::string  narrow_msg(ws.begin(), ws.end());

	if (type == FFX_API_MESSAGE_TYPE_ERROR)
	{
		LOGE("{}", narrow_msg);
	}
	else if (type == FFX_API_MESSAGE_TYPE_WARNING)
	{
		LOGW("{}", narrow_msg);
	}
	else
	{
		LOGI("{}", narrow_msg);
	}
}

void NSSSample::initialize_nss_context(const VkExtent2D &low_res_extent, const VkExtent2D &display_extent)
{
	// Check if NSS is force-disabled by environment variable
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

	// Get vkGetDeviceProcAddr directly from instance loader (more reliable for device-level function pointers)
	// This ensures NSS SDK gets function pointers that properly understand ARM extensions
	PFN_vkGetDeviceProcAddr vkGetDeviceProcAddr =
	    reinterpret_cast<PFN_vkGetDeviceProcAddr>(
	        vkGetInstanceProcAddr(get_instance().get_handle(), "vkGetDeviceProcAddr"));

	// Configure NSS context flags
	uint32_t nssFlags =
	    FFX_API_NSS_CONTEXT_FLAG_QUANTIZED |             // Use quantized model (required)
	    FFX_API_NSS_CONTEXT_FLAG_DEPTH_INVERTED |        // Reversed depth (1.0 = near, 0.0 = far)
	    FFX_API_NSS_CONTEXT_FLAG_ENABLE_DEBUG_CHECKING;

	// Apply override flags from environment variable
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
	// Vulkan Samples CLI is extended via platform plugins. The `AppArgs` plugin
	// forwards repeatable `--app-arg KEY=VALUE` pairs into ApplicationOptions.
	//
	// Example:
	//   vulkan_samples sample nss --app-arg NSS_ENABLE=0,NSS_SCALE_FACTOR=2.0,NSS_FLAGS=0x60
	//
	// Supported keys (values are strings; parsing happens here):
	//   NSS_ENABLE       : 0 or 1 (default 1)
	//   NSS_SCALE_FACTOR : float in [1.0, 3.0] (default 2.0)
	//   NSS_FLAGS        : uint32 (decimal or 0x-prefixed hex), default 0
	auto get_arg = [&options](const char *key) -> std::string {
		auto it = options.app_arguments.find(key);
		if (it == options.app_arguments.end())
		{
			return {};
		}
		return it->second;
	};

	const std::string enable_str = get_arg("NSS_ENABLE");
	if (!enable_str.empty())
	{
		int enable = 1;
		bool parsed = false;
		try
		{
			size_t pos = 0;
			int    v   = std::stoi(enable_str, &pos, 10);
			if (pos != enable_str.size() || (v != 0 && v != 1))
			{
				throw std::invalid_argument("NSS_ENABLE must be 0 or 1");
			}
			enable = v;
			parsed = true;
		}
		catch (...)
		{
			LOGW("Invalid NSS_ENABLE value: {}, expected 0 or 1", enable_str);
		}

		if (parsed)
		{
			if (enable == 0)
			{
				nss_force_disabled = true;
				LOGI("NSS disabled via NSS_ENABLE=0");
			}
			else
			{
				LOGI("NSS enabled via NSS_ENABLE=1");
			}
		}
	}

	const std::string scale_str = get_arg("NSS_SCALE_FACTOR");
	if (!scale_str.empty())
	{
		float scale = nss_scale_factor;
		bool  parsed = false;
		try
		{
			size_t pos = 0;
			float  v   = std::stof(scale_str, &pos);
			if (pos != scale_str.size() || v < 1.0f || v > 3.0f)
			{
				throw std::out_of_range("NSS_SCALE_FACTOR out of range");
			}
			scale = v;
			parsed = true;
		}
		catch (...)
		{
			LOGW("Invalid NSS_SCALE_FACTOR: {}, expected float in [1.0, 3.0]", scale_str);
		}

		if (parsed)
		{
			nss_scale_factor = scale;
			LOGI("NSS scale factor set from command line: {:.2f}x", nss_scale_factor);
		}
	}

	const std::string flags_str = get_arg("NSS_FLAGS");
	if (!flags_str.empty())
	{
		uint32_t flags = 0;
		bool     parsed = false;
		try
		{
			size_t        pos = 0;
			unsigned long v   = std::stoul(flags_str, &pos, 0);
			if (pos != flags_str.size())
			{
				throw std::invalid_argument("NSS_FLAGS has trailing characters");
			}
			flags = static_cast<uint32_t>(v);
			parsed = true;
		}
		catch (...)
		{
			LOGW("Invalid NSS_FLAGS: {}, expected uint32 (decimal or 0xHEX)", flags_str);
		}

		if (parsed)
		{
			nss_override_flags = flags;
			LOGI("NSS override flags: 0x{:08X}", nss_override_flags);
		}
	}

	// ------------------Load scenes and initial resolution------------------------------
	load_scene("scenes/sponza/Sponza01.gltf");

	// Initialize resolution configuration (centralized resolution management)
	auto initial_extent = get_render_context().get_surface_extent();
	resolution_config   = ResolutionConfig::create(initial_extent, nss_scale_factor);

	LOGI("=== Resolution Configuration ===");
	LOGI("  Display (high-res): {}x{}", resolution_config.display_extent.width, resolution_config.display_extent.height);
	LOGI("  Render (low-res):   {}x{}", resolution_config.render_extent.width, resolution_config.render_extent.height);
	LOGI("  Scale factor:       {:.2f}x", resolution_config.scale_factor);
	LOGI("  Mipmap LOD bias:    {:.2f}", resolution_config.mipmap_lod_bias);

	// ------------------NSS Context Creation------------------------------
	// NSS context must be created before render targets because it determines
	// the rendering resolution (low-res if NSS enabled, full-res if disabled)
	initialize_nss_context(resolution_config.render_extent, resolution_config.display_extent);

	// Adjust scale factor if NSS failed to initialize
	if (!nss_enabled)
	{
		nss_scale_factor        = 1.0f;
		use_nss_jitter_sequence = false;        // Fall back to Halton sequence
		// Recalculate resolution config with scale factor = 1.0
		resolution_config = ResolutionConfig::create(initial_extent, 1.0f);
		LOGI("NSS disabled: rendering at native resolution (scale factor = 1.0)");
		LOGI("  Updated render resolution: {}x{}", resolution_config.render_extent.width, resolution_config.render_extent.height);
	}

	// ------------------Create Render Targets and Images------------------------------

	// Create low-res render targets for scene rendering (one per swapchain image)
	create_low_res_render_targets();

	// Create NSS output images (one per swapchain image, only if NSS enabled)
	create_nss_output_images();

	// ------------------Scene Camera Setup------------------------------

	auto &camera_node = vkb::add_free_camera(get_scene(), "main_camera", resolution_config.render_extent);
	camera            = dynamic_cast<vkb::sg::PerspectiveCamera *>(&camera_node.get_component<vkb::sg::Camera>());

	// ------------------Scene Rendering Pipeline------------------------------
	// Initialize TAA data
	total_frames = 0;
	jitter_scale = 1.0f;

	// Pre-generate Halton sequence (once at startup)
	prepare_halton_sequence();

	// Use jitter&velocity shader with custom subpass
	vkb::ShaderSource vert_shader("nss/jitter.vert.spv");
	vkb::ShaderSource frag_shader("nss/velocity.frag.spv");
	auto              scene_subpass = std::make_unique<TAAForwardSubpass>(get_render_context(), std::move(vert_shader), std::move(frag_shader), get_scene(), *camera, this);
	scene_subpass->set_output_attachments({static_cast<int>(Attachments::Color), static_cast<int>(Attachments::Velocity)});

	scene_pipeline = std::make_unique<vkb::RenderPipeline>();
	scene_pipeline->add_subpass(std::move(scene_subpass));
	scene_pipeline->set_load_store(scene_load_store);

	// Prepare TAA uniform buffer (allocate once)
	prepare_taa_settings_buffer();

	// ------------------Post-processing Pipeline------------------------------

	// Post-processing pass (display)
	vkb::ShaderSource postprocessing_vs("nss/postprocessing.vert.spv");
	postprocessing_pipeline = std::make_unique<vkb::PostProcessingPipeline>(get_render_context(), std::move(postprocessing_vs));
	postprocessing_pipeline->add_pass().add_subpass(vkb::ShaderSource("nss/display.frag.spv"));

	get_stats().request_stats({vkb::StatIndex::frame_times});
	create_gui(*window, &get_stats());

	return true;
}

void NSSSample::update(float delta_time)
{
	// Update per-frame state
	last_delta_time = delta_time;
	total_frames++;

	// Manual update flow for precise timing control
	// ---------------------------------------------
	// We cannot use the default VulkanSample::update() because we need to update
	// the TAA settings buffer AFTER the camera updates but BEFORE rendering begins.
	// This ensures the jitter projection-view matrix uses the current frame's camera state.
	//
	// Update sequence (following AsyncComputeSample pattern):
	//   1. Application::update()     -> FPS tracking
	//   2. update_scene()            -> Camera movement (FreeCamera::update())
	//   3. update_taa_settings_buffer() -> Jitter calculation with updated camera
	//   4. update_gui()              -> GUI state
	//   5. draw()                    -> Rendering with correct projection-view matrix

	vkb::Application::update(delta_time);
	update_scene(delta_time);
	update_taa_settings_buffer();
	update_gui(delta_time);

	// Record and submit rendering commands
	auto command_buffer = get_render_context().begin();
	update_stats(delta_time);
	command_buffer->begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
	get_stats().begin_sampling(*command_buffer);
	draw(*command_buffer, get_render_context().get_active_frame().get_render_target());
	get_stats().end_sampling(*command_buffer);
	command_buffer->end();
	get_render_context().submit(command_buffer);

	if (first_frame)
	{
		first_frame = false;
	}
}

/********************************
 * scene render pass: jitter + velocity
 *********************************/
// Halton sequence generation
float CreateHaltonSequence(unsigned int index, int base)
{
	float f       = 1;
	float r       = 0;
	int   current = index;
	do
	{
		f       = f / base;
		r       = r + f * (current % base);
		current = (int) glm::floor(current / base);
	} while (current > 0);

	return r;
}

void NSSSample::prepare_halton_sequence()
{
	// Pre-generate Halton sequence for TAA
	halton_sequence.clear();
	halton_sequence.reserve(MAX_TAA_SAMPLES);

	for (int i = 1; i <= MAX_TAA_SAMPLES; ++i)
	{
		glm::vec2 offset;

		// Generate Halton sequence values [0, 1]
		offset.x = CreateHaltonSequence(i, 2);        // Base 2 for X
		offset.y = CreateHaltonSequence(i, 3);        // Base 3 for Y

		// Convert to [-0.5, +0.5] range for sub-pixel jitter
		offset.x -= 0.5f;
		offset.y -= 0.5f;

		halton_sequence.push_back(offset);
	}
}

void NSSSample::TAAForwardSubpass::update_uniform(vkb::core::CommandBufferC &command_buffer, vkb::sg::Node &node, size_t thread_index)
{
	// Call parent to handle standard uniforms (GlobalUniform: model, view, projection)
	vkb::ForwardSubpass::update_uniform(command_buffer, node, thread_index);

	// Bind shared jitter settings buffer (camera jitter offset and view-proj matrices)
	if (nss_sample->jitter_settings_buffer)
	{
		command_buffer.bind_buffer(*nss_sample->jitter_settings_buffer, 0, sizeof(NSSSample::JitterSettings), 0, 2, 0);
	}

	// Update per-node motion tracking (previous and current transforms)
	auto &motion = nss_sample->node_motion_data[&node];
	if (motion.last_update_frame != nss_sample->total_frames)
	{
		// First update this frame - swap previous and current
		motion.previous_transform = motion.current_transform;
		motion.last_update_frame  = nss_sample->total_frames;
	}
	motion.current_transform = node.get_transform().get_world_matrix();

	// Allocate and bind per-node velocity buffer
	auto &render_frame        = get_render_context().get_active_frame();
	auto  velocity_allocation = render_frame.allocate_buffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
	                                                         sizeof(NSSSample::VelocitySettings),
	                                                         thread_index);

	NSSSample::VelocitySettings velocity_data;
	velocity_data.prev_model    = motion.previous_transform;
	velocity_data.current_model = motion.current_transform;

	velocity_allocation.update(velocity_data);
	command_buffer.bind_buffer(velocity_allocation.get_buffer(), velocity_allocation.get_offset(),
	                           velocity_allocation.get_size(), 0, 3, 0);
}

void NSSSample::TAAForwardSubpass::draw(vkb::core::CommandBufferC &command_buffer)
{
	// Set specialization constants before drawing
	// These affect shader compilation and must be set per draw call
	command_buffer.set_specialization_constant(3, static_cast<uint32_t>(nss_sample->display_mode));

	// Set mipmap LOD bias based on render scale (specialization constant 4)
	command_buffer.set_specialization_constant(4, nss_sample->resolution_config.mipmap_lod_bias);

	// Call parent draw which handles lighting setup and geometry rendering
	vkb::ForwardSubpass::draw(command_buffer);
}

void NSSSample::prepare_taa_settings_buffer()
{
	// Allocate jitter settings buffer once (shared across all nodes)
	jitter_settings_buffer = std::make_unique<vkb::core::BufferC>(
	    get_device(),
	    sizeof(JitterSettings),
	    VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
	    VMA_MEMORY_USAGE_CPU_TO_GPU);
}

void NSSSample::update_taa_settings_buffer()
{
	// Use centralized resolution configuration
	glm::vec2 render_extent = glm::vec2(
	    resolution_config.render_extent.width,
	    resolution_config.render_extent.height);

	// ---------------Calculate pixel-space jitter offset-----------------
	if (use_nss_jitter_sequence)
	{
		// Use NSS SDK jitter sequence via ffx-api query descriptors.
		int32_t jitter_phase_count = 0;
		ffx::QueryDescNssGetJitterPhaseCount phase_query{};
		phase_query.renderWidth    = resolution_config.render_extent.width;
		phase_query.displayWidth   = resolution_config.display_extent.width;
		phase_query.pOutPhaseCount = &jitter_phase_count;
		ffx::ReturnCode result = ffx::Query(phase_query);

		float jitter_x = 0.0f;
		float jitter_y = 0.0f;
		if (result == ffx::ReturnCode::Ok && jitter_phase_count > 0)
		{
			ffx::QueryDescNssGetJitterOffset offset_query{};
			offset_query.index      = static_cast<int32_t>(total_frames);
			offset_query.phaseCount = jitter_phase_count;
			offset_query.pOutX      = &jitter_x;
			offset_query.pOutY      = &jitter_y;
			result = ffx::Query(offset_query);
		}

		if(result == ffx::ReturnCode::Ok)
		{
			cur_jitter = {jitter_x, jitter_y};
		}else
		{
			use_nss_jitter_sequence = false;
			LOGW("NSS jitter query failed with code {}, falling back to Halton sequence", static_cast<uint32_t>(result));
		}
	}

	if (!use_nss_jitter_sequence)
	{
		// Fall back to Halton sequence (for compatibility testing)
		int jitter_index = total_frames % MAX_TAA_SAMPLES;
		cur_jitter       = halton_sequence[jitter_index];
	}

	// ----------------Convert pixel jitter to NDC jitter----------------
	// NDC jitter: multiply by 2.0 because NDC range is [-1, 1] (width of 2)
	glm::vec2 ndc_jitter_offset = glm::vec2(
	    cur_jitter.x * (2.0f / render_extent.x),
	    cur_jitter.y * (2.0f / render_extent.y));

	// ----------------Calculate view-projection matrices----------------
	glm::mat4 current_view_proj = camera->get_pre_rotation() *
	                              vkb::rendering::vulkan_style_projection(camera->get_projection()) *
	                              camera->get_view();

	// Handle first frame: initialize previous matrix
	if (first_frame)
	{
		prev_view_proj_matrix = current_view_proj;
	}

	// ----------------Apply jitter to projection matrix----------------
	// Modify projection matrix directly (most common TAA method)
	glm::mat4 jittered_projection = camera->get_projection();
#if defined(__ANDROID__)
	// No ideal why, but on Android the X jitter needs to be inverted, otherwise horizontal ghosting appears (vertical edges appear doubled).
	jittered_projection[2][0] -= ndc_jitter_offset.x * jitter_scale;
#else
	jittered_projection[2][0] += ndc_jitter_offset.x * jitter_scale;
#endif
	jittered_projection[2][1] -= ndc_jitter_offset.y * jitter_scale;        // Invert Y for Vulkan coordinate system

	glm::mat4 jitter_view_proj = camera->get_pre_rotation() *
	                             vkb::rendering::vulkan_style_projection(jittered_projection) *
	                             camera->get_view();

	// -----------------Update uniform buffer----------------------
	jitter_settings_data.jitter_offset     = ndc_jitter_offset;
	jitter_settings_data.jitter_scale      = jitter_scale;
	jitter_settings_data.jitter_view_proj  = jitter_view_proj;
	jitter_settings_data.prev_view_proj    = prev_view_proj_matrix;
	jitter_settings_data.current_view_proj = current_view_proj;

	jitter_settings_buffer->convert_and_update(jitter_settings_data);

	// Store current matrix for next frame
	prev_view_proj_matrix = current_view_proj;
}

/********************************
 * create buffer
 *********************************/
void NSSSample::create_low_res_render_targets()
{
	// Get swapchain info
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

	// Reserve space
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

	// Get swapchain info
	auto  &swapchain       = get_render_context().get_swapchain();
	size_t swapchain_count = swapchain.get_images().size();

	// Format
	auto color_format = VK_FORMAT_B10G11R11_UFLOAT_PACK32;

	// Usage flags for NSS output
	// Note: When debug view is enabled via FFX_NSS_DISPATCH_FLAG_DRAW_DEBUG_VIEW,
	// the same output texture will contain debug visualization instead of upscaled result
	VkImageUsageFlags nss_output_usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

	// Reserve space
	nss_output_images.reserve(swapchain_count);
	nss_output_views.reserve(swapchain_count);

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
		barrier.dst_stage_mask  = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

		command_buffer.image_memory_barrier(scene_views[static_cast<int>(Attachments::Color)], barrier);
		scene_rt.set_layout(static_cast<int>(Attachments::Color), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
		command_buffer.image_memory_barrier(scene_views[static_cast<int>(Attachments::Velocity)], barrier);
		scene_rt.set_layout(static_cast<int>(Attachments::Velocity), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

		// Transition depth to shader read
		barrier.old_layout      = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
		barrier.new_layout      = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		barrier.src_access_mask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
		barrier.dst_access_mask = VK_ACCESS_SHADER_READ_BIT;
		barrier.src_stage_mask  = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;        // Depth writes happen here
		barrier.dst_stage_mask  = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
		command_buffer.image_memory_barrier(scene_views[static_cast<int>(Attachments::Depth)], barrier);
		scene_rt.set_layout(static_cast<int>(Attachments::Depth), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

		// Transition NSS output to GENERAL for compute write
		barrier.old_layout      = VK_IMAGE_LAYOUT_UNDEFINED;
		barrier.new_layout      = VK_IMAGE_LAYOUT_GENERAL;
		barrier.src_access_mask = 0;
		barrier.dst_access_mask = VK_ACCESS_SHADER_WRITE_BIT;
		barrier.src_stage_mask  = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
		barrier.dst_stage_mask  = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

		command_buffer.image_memory_barrier(*nss_output_views[frame_index], barrier);

		// ============================================================================
		// NSS Dispatch: Execute neural network upscaling using Builder pattern
		// ============================================================================

		// Get current frame images from scene render target
		auto &color_image    = scene_views[static_cast<int>(Attachments::Color)].get_image();
		auto &depth_image    = scene_views[static_cast<int>(Attachments::Depth)].get_image();
		auto &velocity_image = scene_views[static_cast<int>(Attachments::Velocity)].get_image();

		// Calculate jitter (NSS expects pixel-space jitter)
		float jitterX = cur_jitter.x * jitter_scale;
		float jitterY = cur_jitter.y * jitter_scale;

		// Set dispatch flags based on display mode
		// When NSSDebugView mode is active, the SDK will write debug visualization to output instead of upscaled result
		uint32_t dispatch_flags = (display_mode == DisplayMode::NSSDebugView) ? FFX_API_NSS_DISPATCH_FLAG_DRAW_DEBUG_VIEW : 0;

		// Prepare dispatch descriptor
		ffxApiDispatchDescNss desc{};
		std::memset(&desc, 0, sizeof(ffxApiDispatchDescNss));
		desc.header.type = FFX_API_DISPATCH_DESC_TYPE_NSS;

		// Command buffer
		desc.commandList = reinterpret_cast<void *>(command_buffer.get_handle()); // VkCommandBuffer stored in void* (see ffxApiDispatchDescNss::commandList)

		// Input resources
		desc.color         = wrap_resource(color_image, FFX_API_RESOURCE_STATE_COMPUTE_READ);
		desc.depth         = wrap_resource(depth_image, FFX_API_RESOURCE_STATE_COMPUTE_READ);
		desc.motionVectors = wrap_resource(velocity_image, FFX_API_RESOURCE_STATE_COMPUTE_READ);

		// Output resource
		desc.output = wrap_resource(*nss_output_images[frame_index], FFX_API_RESOURCE_STATE_UNORDERED_ACCESS);

		// Dispatch parameters
		desc.jitterOffset.x         = jitterX;
		desc.jitterOffset.y         = jitterY;
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
		desc.flags                  = dispatch_flags;
		desc.exposure               = 1.0f;

		// Direct call to FFX API
		ffx::ReturnCode nssResult = ffx::Dispatch(nss_context, desc);
		if (nssResult != ffx::ReturnCode::Ok)
		{
			LOGE("NSS dispatch failed with error code: {}", static_cast<uint32_t>(nssResult));
		}

		// Transition NSS output to shader read for postprocessing
		barrier.old_layout      = VK_IMAGE_LAYOUT_GENERAL;
		barrier.new_layout      = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		barrier.src_access_mask = VK_ACCESS_SHADER_WRITE_BIT;
		barrier.dst_access_mask = VK_ACCESS_SHADER_READ_BIT;
		barrier.src_stage_mask  = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
		barrier.dst_stage_mask  = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;

		command_buffer.image_memory_barrier(*nss_output_views[frame_index], barrier);

		barrier.old_layout      = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		barrier.new_layout      = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		barrier.src_access_mask = VK_ACCESS_SHADER_READ_BIT;
		barrier.dst_access_mask = VK_ACCESS_SHADER_READ_BIT;
		barrier.src_stage_mask  = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
		barrier.dst_stage_mask  = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;

		command_buffer.image_memory_barrier(scene_views[static_cast<int>(Attachments::Color)], barrier);
		command_buffer.image_memory_barrier(scene_views[static_cast<int>(Attachments::Velocity)], barrier);
		command_buffer.image_memory_barrier(scene_views[static_cast<int>(Attachments::Depth)], barrier);
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
	// PHASE 3: Postprocessing (Display NSS Output or Scene Color)
	// ============================================================================

	// Reset viewport to full resolution for postprocessing
	viewport.width  = static_cast<float>(swapchain_extent.width);
	viewport.height = static_cast<float>(swapchain_extent.height);
	command_buffer.set_viewport(0, {viewport});

	scissor.extent = swapchain_extent;
	command_buffer.set_scissor(0, {scissor});

	auto &postprocessing_pass = postprocessing_pipeline->get_pass(0);

	glm::vec4 near_far = {camera->get_far_plane(), camera->get_near_plane(), -1.0f, -1.0f};
	postprocessing_pass.set_uniform_data(near_far);

	auto &postprocessing_subpass = postprocessing_pass.get_subpass(0);

	// Bind current frame inputs
	// Binding order MUST match shader layout:
	// binding = 1: depth_sampler (low-res depth)
	// binding = 2: color_sampler (final output - NSS upscaled or low-res scene if NSS disabled)
	// binding = 3: velocity_sampler (low-res motion vectors)
	// binding = 4: lowres_color_sampler (low-res scene color before NSS)
	//
	// Note: When display_mode == NSSDebugView, color_sampler contains debug visualization
	// instead of upscaled result (controlled by FFX_NSS_DISPATCH_FLAG_DRAW_DEBUG_VIEW)

	if (nss_enabled && !nss_output_views.empty())
	{
		postprocessing_subpass.bind_sampled_image("color_sampler", vkb::core::SampledImage(*nss_output_views[frame_index]));
	}
	else
	{
		// NSS disabled - use low-res scene color as final output
		postprocessing_subpass.bind_sampled_image("color_sampler", vkb::core::SampledImage(scene_views[static_cast<int>(Attachments::Color)]));
	}

	// Bind low-res buffers (depth, velocity, scene color before NSS)
	postprocessing_subpass.bind_sampled_image("depth_sampler", vkb::core::SampledImage(scene_views[static_cast<int>(Attachments::Depth)]));
	postprocessing_subpass.bind_sampled_image("velocity_sampler", vkb::core::SampledImage(scene_views[static_cast<int>(Attachments::Velocity)]));
	postprocessing_subpass.bind_sampled_image("lowres_color_sampler", vkb::core::SampledImage(scene_views[static_cast<int>(Attachments::Color)]));

	// Set display mode as push constant for shader
	postprocessing_subpass.set_push_constants(static_cast<uint32_t>(display_mode));

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
		    ImGui::RadioButton("Velocity", &display_mode_int, static_cast<int>(DisplayMode::Velocity));
		    ImGui::SameLine();
		    ImGui::RadioButton("Depth", &display_mode_int, static_cast<int>(DisplayMode::Depth));
		    ImGui::SameLine();
		    ImGui::RadioButton("NSS Debug", &display_mode_int, static_cast<int>(DisplayMode::NSSDebugView));
		    ImGui::SameLine();
		    ImGui::RadioButton("Low-Res Color", &display_mode_int, static_cast<int>(DisplayMode::LowResColor));
		    display_mode = static_cast<DisplayMode>(display_mode_int);

		    ImGui::Separator();
		    ImGui::Text("TAA Settings:");
		    ImGui::SliderFloat("Jitter Scale", &jitter_scale, 0.0f, 1.0f);

		    ImGui::Text("Current Jitter: (%.3f, %.3f) pixels", cur_jitter.x, cur_jitter.y);
		    ImGui::Text("ScaleFactor: %.3f  Flags: 0x%08X", nss_scale_factor, nss_override_flags);
		    ImGui::Text("Render: %ux%u  Display: %ux%u  LOD Bias: %.2f",
		                resolution_config.render_extent.width, resolution_config.render_extent.height,
		                resolution_config.display_extent.width, resolution_config.display_extent.height,
		                resolution_config.mipmap_lod_bias);
	    },
	    /* lines = */ 12);
}

std::unique_ptr<vkb::VulkanSampleC> create_nss()
{
	return std::make_unique<NSSSample>();
}
