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

#include "dataset_loader.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>

// nlohmann/json is available through tinygltf (framework -> third_party -> tinygltf -> json.hpp).
#include <json.hpp>
using json = nlohmann::json;

#include "common/exr_image.h"
#include "core/util/logging.hpp"

bool NFRUDatasetLoader::init(vkb::core::DeviceC &device,
                             const std::string &dataset_dir, const std::string &dataset_sequence,
                             uint32_t swapchain_count)
{
	const std::string config_path = dataset_dir + "/config.json";
	if (!std::filesystem::exists(config_path))
	{
		LOGI("NFRU dataset config not found; using sample defaults: {}", config_path);
	}
	else
	{
		std::ifstream config_stream(config_path);
		if (!config_stream.is_open())
		{
			LOGE("Cannot open NFRU dataset config: {}", config_path);
			return false;
		}
		try
		{
			const json root = json::parse(config_stream);
			if (root.contains("Tonemap"))
				config_.apply_tonemap = root.at("Tonemap").get<bool>();
			if (root.contains("FlipDepth"))
				config_.flip_depth = root.at("FlipDepth").get<bool>();
			if (root.contains("TransposeViewProjectionOnLoad"))
				config_.transpose_view_projection = root.at("TransposeViewProjectionOnLoad").get<bool>();
			if (root.contains("Exposure"))
			{
				config_.override_exposure   = true;
				config_.exposure            = root.at("Exposure").get<float>();
				if (config_.exposure <= 0.0f)
				{
					LOGE("NFRU dataset config Exposure must be greater than zero: {}", config_path);
					return false;
				}
			}
		}
		catch (const json::exception &exception)
		{
			LOGE("Failed to parse NFRU dataset config {}: {}", config_path, exception.what());
			return false;
		}
		LOGI("Loaded NFRU dataset overrides: {}", config_path);
	}
	std::string sequence_json = dataset_dir + "/" + dataset_sequence + ".json";
	color_dir_                = dataset_dir + "/ground_truth/" + dataset_sequence;
	depth_dir_                = dataset_dir + "/x2/depth/" + dataset_sequence;
	motion_dir_               = dataset_dir + "/x2/motion_m2/" + dataset_sequence;
	// Fall back to motion_t-2 when motion_m2 is absent.
	if (!std::filesystem::exists(motion_dir_))
		motion_dir_ = dataset_dir + "/x2/motion_t-2/" + dataset_sequence;

	if (!std::filesystem::is_directory(color_dir_))
	{
		LOGE("Missing NFRU dataset color directory: {}", color_dir_);
		return false;
	}
	if (!std::filesystem::is_directory(depth_dir_))
	{
		LOGE("Missing NFRU dataset depth directory: {}", depth_dir_);
		return false;
	}
	if (!std::filesystem::is_directory(motion_dir_))
	{
		LOGE("Missing NFRU dataset motion directory: {}", motion_dir_);
		return false;
	}

	LOGI("DatasetLoader: dir={}, sequence={}", dataset_dir, dataset_sequence);
	LOGI("  color_dir = {}", color_dir_);
	LOGI("  depth_dir = {}", depth_dir_);
	LOGI("  motion_dir = {}", motion_dir_);
	const std::string exposure_source = config_.override_exposure ? std::to_string(config_.exposure) : "sequence JSON";
	LOGI("  options: tonemap={}, exposure={}, flipDepth={}, transposeViewProjectionOnLoad={}",
	     config_.apply_tonemap, exposure_source, config_.flip_depth,
	     config_.transpose_view_projection);

	std::ifstream sequence_stream(sequence_json);
	if (!sequence_stream.is_open())
	{
		LOGE("Cannot open sequence JSON: {}", sequence_json);
		return false;
	}

	char path_buffer[512];
	snprintf(path_buffer, sizeof(path_buffer), "%s/%04d.exr", color_dir_.c_str(), 0);
	const bool display_extent_loaded = vkb::sample::exr::probe_extent(path_buffer, display_size_);
	snprintf(path_buffer, sizeof(path_buffer), "%s/%04d.exr", depth_dir_.c_str(), 0);
	const bool render_extent_loaded = vkb::sample::exr::probe_extent(path_buffer, render_size_);

	if (!display_extent_loaded || !render_extent_loaded || display_size_.width == 0 || render_size_.width == 0)
	{
		LOGE("Failed to probe EXR dimensions (color: {}x{}, depth: {}x{})",
		     display_size_.width, display_size_.height, render_size_.width, render_size_.height);
		return false;
	}

	try
	{
		const json root = json::parse(sequence_stream);
		frame_rate_hz_  = root.value("EmulatedFramerate", 60.0f);
		depth_inverted_ = root.value("ReverseZ", false) ^ config_.flip_depth;

		const float root_near = root.value("NearPlane", 10.0f);
		const float root_far  = root.value("FarPlane", -1.0f);

		if (!root.contains("Frames") || !root["Frames"].is_array() || root["Frames"].empty())
		{
			LOGE("NFRU dataset sequence has no Frames array: {}", sequence_json);
			return false;
		}
		const auto &json_frames = root["Frames"];
		frames_.resize(json_frames.size());
		const bool has_frame_indices = json_frames[0].contains("Frame");
		std::vector<bool> frame_seen(json_frames.size(), false);
		bool depth_convention_initialized = false;

		for (size_t frame_index = 0; frame_index < json_frames.size(); ++frame_index)
		{
			const auto &json_frame = json_frames[frame_index];
			if (json_frame.contains("Frame") != has_frame_indices || (has_frame_indices && !json_frame["Frame"].is_number_unsigned()))
			{
				LOGE("NFRU sequence has inconsistent Frame metadata at entry {}: {}", frame_index, sequence_json);
				return false;
			}
			const size_t metadata_frame_index = has_frame_indices ? json_frame["Frame"].get<size_t>() : frame_index;
			if (metadata_frame_index >= frames_.size() || frame_seen[metadata_frame_index])
			{
				LOGE("NFRU sequence has duplicate or out-of-range Frame index {}: {}", metadata_frame_index, sequence_json);
				return false;
			}
			frame_seen[metadata_frame_index] = true;
			auto       &frame_data = frames_[metadata_frame_index];
			const float default_frame_time_delta = (frame_rate_hz_ > 0) ? (1000.0f / frame_rate_hz_) : 1.0f;
			frame_data.frame_time_delta_ms       = (default_frame_time_delta > 1.0f) ? default_frame_time_delta : 1.0f;
			frame_data.motion_vector_scale_x     = -static_cast<float>(render_size_.width);
			frame_data.motion_vector_scale_y     = -static_cast<float>(render_size_.height);
			frame_data.view_space_to_meters_factor = 1.0f;
			frame_data.reset              = json_frame.value("Reset", metadata_frame_index == 0) || metadata_frame_index == 0;
			frame_data.fov_y                = json_frame.value("FovY", 0.75f);
			frame_data.camera_near          = json_frame.value("CameraNearPlane", root_near);
			const float camera_far          = json_frame.value("CameraFarPlane", root_far);
			frame_data.camera_far           = camera_far < 0.0f ? FLT_MAX : camera_far;
			const bool frame_depth_infinite = camera_far < 0.0f;
			if (!depth_convention_initialized)
			{
				depth_infinite_              = frame_depth_infinite;
				depth_convention_initialized = true;
			}
			else if (frame_depth_infinite != depth_infinite_)
			{
				LOGE("NFRU sequence mixes finite and infinite CameraFarPlane values at frame {}: {}", frame_index, sequence_json);
				return false;
			}
			frame_data.exposure = config_.override_exposure ? config_.exposure : json_frame.value("Exposure", 1.0f);
			if (json_frame.contains("FrameTimeDelta"))
			{
				frame_data.frame_time_delta_ms = json_frame["FrameTimeDelta"].get<float>();
				if (!std::isfinite(frame_data.frame_time_delta_ms) || frame_data.frame_time_delta_ms < 1.0f)
				{
					LOGE("NFRU frame {} has invalid FrameTimeDelta (must be finite and at least 1 ms): {}", metadata_frame_index, sequence_json);
					return false;
				}
			}
			if (json_frame.contains("ViewSpaceToMetersFactor"))
			{
				frame_data.view_space_to_meters_factor = json_frame["ViewSpaceToMetersFactor"].get<float>();
				if (!std::isfinite(frame_data.view_space_to_meters_factor))
				{
					LOGE("NFRU frame {} has invalid ViewSpaceToMetersFactor: {}", metadata_frame_index, sequence_json);
					return false;
				}
			}
			if (json_frame.contains("MotionVectorScale"))
			{
				const auto &motion_vector_scale = json_frame["MotionVectorScale"];
				if (!motion_vector_scale.is_array() || motion_vector_scale.size() != 2 ||
				    !motion_vector_scale[0].is_number() || !motion_vector_scale[1].is_number())
				{
					LOGE("NFRU sequence has invalid MotionVectorScale at frame {} (expected two finite numbers): {}", frame_index, sequence_json);
					return false;
				}
				frame_data.motion_vector_scale_x = motion_vector_scale[0].get<float>();
				frame_data.motion_vector_scale_y = motion_vector_scale[1].get<float>();
				if (!std::isfinite(frame_data.motion_vector_scale_x) || !std::isfinite(frame_data.motion_vector_scale_y))
				{
					LOGE("NFRU sequence has non-finite MotionVectorScale at frame {}: {}", frame_index, sequence_json);
					return false;
				}
			}
			if (frame_data.exposure <= 0.0f)
			{
				LOGE("NFRU sequence has invalid Exposure at frame {} (must be greater than zero): {}", frame_index, sequence_json);
				return false;
			}

			if (!json_frame.contains("ViewProjection") || !json_frame["ViewProjection"].is_array() || json_frame["ViewProjection"].size() != 16)
			{
				LOGE("NFRU frame {} has invalid ViewProjection in {}", frame_index, sequence_json);
				return false;
			}
			const auto &view_projection = json_frame["ViewProjection"];
			for (int index = 0; index < 16; ++index)
			{
				const int destination                   = config_.transpose_view_projection ? (index % 4) * 4 + index / 4 : index;
				frame_data.view_projection[destination] = view_projection[index].get<float>();
			}

			if (json_frame.contains("NormalizedPerRatioJitter") && !json_frame["NormalizedPerRatioJitter"].empty())
			{
				const auto &jitter  = json_frame["NormalizedPerRatioJitter"][0];
				frame_data.jitter_x = jitter.value("X", 0.0f);
				frame_data.jitter_y = jitter.value("Y", 0.0f);
			}
		}
	}
	catch (const json::exception &exception)
	{
		LOGE("Failed to parse NFRU sequence JSON '{}': {}", sequence_json, exception.what());
		return false;
	}

	LOGI("Parsed sequence: {} frames, {}fps, depthInverted={}, depthInfinite={}",
	     frames_.size(), frame_rate_hz_, depth_inverted_, depth_infinite_);
	LOGI("  camera planes: near={}, far={} (infinite far uses SDK encoding)",
	     frames_[0].camera_near, frames_[0].camera_far);

	LOGI("  display_size = {}x{}, render_size = {}x{}",
	     display_size_.width, display_size_.height, render_size_.width, render_size_.height);

	size_t color_bytes = 0;
	size_t depth_bytes = 0;
	size_t mv_bytes    = 0;
	if (!vkb::sample::exr::vk_format_buffer_size(display_size_, VK_FORMAT_R8G8B8A8_UNORM, color_bytes) ||
	    !vkb::sample::exr::vk_format_buffer_size(render_size_, VK_FORMAT_R32_SFLOAT, depth_bytes) ||
	    !vkb::sample::exr::vk_format_buffer_size(render_size_, VK_FORMAT_R16G16_SFLOAT, mv_bytes))
	{
		LOGE("Unsupported NFRU dataset image format");
		return false;
	}

	staging_buffers_.resize(swapchain_count);
	for (StagingBuffers &staging : staging_buffers_)
	{
		staging.color          = std::make_unique<vkb::core::BufferC>(vkb::core::BufferC::create_staging_buffer(device, color_bytes, nullptr));
		staging.depth          = std::make_unique<vkb::core::BufferC>(vkb::core::BufferC::create_staging_buffer(device, depth_bytes, nullptr));
		staging.motion_vectors = std::make_unique<vkb::core::BufferC>(vkb::core::BufferC::create_staging_buffer(device, mv_bytes, nullptr));
	}
	LOGI("Created {} NFRU dataset staging triplets (color={}, depth={}, mv={} bytes)",
	     swapchain_count, color_bytes, depth_bytes, mv_bytes);

	return true;
}

uint64_t NFRUDatasetLoader::total_frames() const
{
	return static_cast<uint64_t>((frames_.size() + 1) / 2);
}

// Returns the source-frame index represented by the interpolated output.
int NFRUDatasetLoader::interpolated_source_frame(int frame_index) const
{
	return frame_index * 2 - 1;
}

bool NFRUDatasetLoader::get_frame_params(int frame_index, NFRUFrameParams &frame_params) const
{
	const int source_frame_index = frame_index * 2;
	if (source_frame_index < 0 || source_frame_index >= static_cast<int>(frames_.size()))
		return false;

	const auto &metadata = frames_[source_frame_index];
	frame_params.reset    = metadata.reset;
	std::memcpy(frame_params.view_projection, metadata.view_projection, sizeof(frame_params.view_projection));
	frame_params.jitter_x                  = metadata.jitter_x * static_cast<float>(render_size_.width);
	frame_params.jitter_y                  = metadata.jitter_y * static_cast<float>(render_size_.height);
	frame_params.camera_near               = metadata.camera_near;
	frame_params.camera_far                = metadata.camera_far;
	frame_params.camera_fov_angle_vertical = metadata.fov_y;
	frame_params.frame_time_delta_ms       = metadata.frame_time_delta_ms;
	frame_params.motion_vector_scale_x     = metadata.motion_vector_scale_x;
	frame_params.motion_vector_scale_y     = metadata.motion_vector_scale_y;
	frame_params.view_space_to_meters_factor = metadata.view_space_to_meters_factor > 0.0f ?
	                                               metadata.view_space_to_meters_factor :
	                                               1.0f;
	return true;
}

bool NFRUDatasetLoader::load_frame(int frame_index, uint32_t swapchain_index, NFRUFrameParams &frame_params)
{
	const int source_frame_index = frame_index * 2;
	if (source_frame_index < 0 || source_frame_index >= static_cast<int>(frames_.size()))
		return false;

	auto &staging = staging_buffers_[swapchain_index];
	char  path_buffer[512];
	auto  load_image = [&](const char *kind, VkExtent2D expected_extent, VkFormat format,
                          const vkb::sample::exr::ExrLoadOptions &options,
                          void *destination, size_t destination_size) {
        VkExtent2D extent{};
        if (!vkb::sample::exr::load_exr_to_vk_format(path_buffer, format, options, destination, destination_size, extent))
            return false;
        if (extent.width == expected_extent.width && extent.height == expected_extent.height)
            return true;
        LOGE("NFRU {} EXR has size {}x{}, expected {}x{}: {}",
		      kind, extent.width, extent.height, expected_extent.width, expected_extent.height, path_buffer);
        return false;
	};

	// Pack display-resolution HDR color into R8G8B8A8_UNORM.
	{
		snprintf(path_buffer, sizeof(path_buffer), "%s/%04d.exr", color_dir_.c_str(), source_frame_index);

		vkb::sample::exr::ExrLoadOptions color_options{};
		color_options.tonemap             = config_.apply_tonemap;
		color_options.exposure            = frames_[source_frame_index].exposure;
		if (!load_image("color", display_size_, VK_FORMAT_R8G8B8A8_UNORM, color_options,
		                staging.color->map(), static_cast<size_t>(staging.color->get_size())))
			return false;
		staging.color->flush();
	}

	// Pack render-resolution depth into the R32_SFLOAT staging buffer.
	{
		snprintf(path_buffer, sizeof(path_buffer), "%s/%04d.exr", depth_dir_.c_str(), source_frame_index);

		vkb::sample::exr::ExrLoadOptions depth_options{};
		depth_options.flip_depth = config_.flip_depth;
		if (!load_image("depth", render_size_, VK_FORMAT_R32_SFLOAT, depth_options,
		                staging.depth->map(), static_cast<size_t>(staging.depth->get_size())))
			return false;
		staging.depth->flush();
	}

	// Pack render-resolution motion into the R16G16_SFLOAT staging buffer.
	{
		snprintf(path_buffer, sizeof(path_buffer), "%s/%04d.exr", motion_dir_.c_str(), source_frame_index);

		if (!load_image("motion", render_size_, VK_FORMAT_R16G16_SFLOAT, {},
		                staging.motion_vectors->map(), static_cast<size_t>(staging.motion_vectors->get_size())))
			return false;
		staging.motion_vectors->flush();
	}

	return get_frame_params(frame_index, frame_params);
}

VkBuffer NFRUDatasetLoader::get_color_staging_buffer(uint32_t swapchain_index) const
{
	return staging_buffers_[swapchain_index].color->get_handle();
}
VkBuffer NFRUDatasetLoader::get_depth_staging_buffer(uint32_t swapchain_index) const
{
	return staging_buffers_[swapchain_index].depth->get_handle();
}
VkBuffer NFRUDatasetLoader::get_motion_vector_staging_buffer(uint32_t swapchain_index) const
{
	return staging_buffers_[swapchain_index].motion_vectors->get_handle();
}

VkExtent2D NFRUDatasetLoader::get_display_size() const
{
	return display_size_;
}

VkExtent2D NFRUDatasetLoader::get_render_size() const
{
	return render_size_;
}

bool NFRUDatasetLoader::uses_inverted_depth() const
{
	return depth_inverted_;
}

bool NFRUDatasetLoader::uses_infinite_far() const
{
	return depth_infinite_;
}