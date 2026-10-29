/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT
 */

#include "dataset_loader.h"

#include <algorithm>
#include <cfloat>
#include <cmath>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>

#include <json.hpp>
using json = nlohmann::json;

#include "common/exr_image.h"
#include "core/util/logging.hpp"

namespace
{
// Dataset ratio directories encode the decimal scale with an underscore, for
// example 2.0 -> "x2" and 1.5 -> "x1_5".
std::string scale_factor_to_ratio_dir(float scale_factor)
{
	std::ostringstream stream;
	stream << std::fixed << std::setprecision(3) << scale_factor;
	std::string ratio = stream.str();
	while (!ratio.empty() && ratio.back() == '0')
		ratio.pop_back();
	if (!ratio.empty() && ratio.back() == '.')
		ratio.pop_back();
	std::replace(ratio.begin(), ratio.end(), '.', '_');
	return "x" + ratio;
}
}        // namespace

bool NSSDatasetLoader::init(vkb::core::DeviceC &device, const std::string &dataset_dir,
                            const std::string &dataset_sequence, float scale_factor, VkFormat color_format, uint32_t swapchain_count)
{
	color_format_ = color_format;

	const std::string config_path = dataset_dir + "/config.json";
	if (!std::filesystem::exists(config_path))
	{
		LOGI("NSS dataset config not found; using sample defaults: {}", config_path);
	}
	else
	{
		std::ifstream config_stream(config_path);
		if (!config_stream.is_open())
		{
			LOGE("Cannot open NSS dataset config: {}", config_path);
			return false;
		}
		try
		{
			const json root = json::parse(config_stream);
			if (root.contains("FlipDepth"))
				config_.flip_depth = root.at("FlipDepth").get<bool>();
			if (root.contains("Exposure"))
			{
				config_.override_exposure   = true;
				config_.exposure            = root.at("Exposure").get<float>();
				if (config_.exposure <= 0.0f)
				{
					LOGE("NSS dataset config Exposure must be greater than zero: {}", config_path);
					return false;
				}
			}
		}
		catch (const json::exception &exception)
		{
			LOGE("Failed to parse NSS dataset config {}: {}", config_path, exception.what());
			return false;
		}
		LOGI("Loaded NSS dataset overrides: {}", config_path);
	}
	ratio_dir_ = scale_factor_to_ratio_dir(scale_factor);
	LOGI("NSS dataset scale factor {} maps to ratio directory '{}'", scale_factor, ratio_dir_);

	const std::string sequence_json = dataset_dir + "/" + dataset_sequence + ".json";
	color_dir_                      = dataset_dir + "/" + ratio_dir_ + "/color/" + dataset_sequence;
	depth_dir_                      = dataset_dir + "/" + ratio_dir_ + "/depth/" + dataset_sequence;
	motion_dir_                     = dataset_dir + "/" + ratio_dir_ + "/motion/" + dataset_sequence;

	if (!std::filesystem::exists(color_dir_))
	{
		const std::string legacy_color_dir = dataset_dir + "/" + ratio_dir_ + "/colour/" + dataset_sequence;
		if (std::filesystem::exists(legacy_color_dir))
		{
			LOGW("Using legacy NSS dataset color directory: {}", legacy_color_dir);
			color_dir_ = legacy_color_dir;
		}
	}

	if (!std::filesystem::exists(color_dir_))
	{
		LOGE("Missing NSS dataset color directory: {}", color_dir_);
		return false;
	}
	if (!std::filesystem::exists(depth_dir_))
	{
		LOGE("Missing NSS dataset depth directory: {}", depth_dir_);
		return false;
	}
	if (!std::filesystem::exists(motion_dir_))
	{
		LOGE("Missing NSS dataset motion directory: {}", motion_dir_);
		return false;
	}

	LOGI("NSS DatasetLoader: dir={}, sequence={}, ratio={}", dataset_dir, dataset_sequence, ratio_dir_);
	LOGI("  color_dir  = {}", color_dir_);
	LOGI("  depth_dir  = {}", depth_dir_);
	LOGI("  motion_dir = {}", motion_dir_);
	const std::string exposure_source = config_.override_exposure ? std::to_string(config_.exposure) : "sequence JSON";
	LOGI("  options: exposure={}, flipDepth={}", exposure_source, config_.flip_depth);

	char path_buffer[512];
	snprintf(path_buffer, sizeof(path_buffer), "%s/%04d.exr", color_dir_.c_str(), 0);
	if (!vkb::sample::exr::probe_extent(path_buffer, render_size_))
	{
		LOGE("Failed to probe NSS dataset EXR dimensions");
		return false;
	}
	if (render_size_.width == 0 || render_size_.height == 0)
	{
		LOGE("Failed to probe NSS dataset EXR dimensions");
		return false;
	}

	std::ifstream stream(sequence_json);
	if (!stream.is_open())
	{
		LOGE("Cannot open NSS sequence JSON: {}", sequence_json);
		return false;
	}

	try
	{
		const json root = json::parse(stream);
		frame_rate_hz_  = root.value("EmulatedFramerate", 60.0f);
		depth_inverted_ = root.value("ReverseZ", false) ^ config_.flip_depth;
		if (root.contains("TargetResolution"))
		{
			display_size_.width  = root["TargetResolution"].value("X", 0u);
			display_size_.height = root["TargetResolution"].value("Y", 0u);
		}

		const std::string ratio_key = ratio_dir_ + "_index";
		if (!root.contains("UpscalingRatiosIndices") || !root["UpscalingRatiosIndices"].contains(ratio_key))
		{
			LOGE("NSS sequence JSON is missing UpscalingRatiosIndices.{}: {}", ratio_key, sequence_json);
			return false;
		}
		const int ratio_index = root["UpscalingRatiosIndices"][ratio_key].get<int>();
		if (ratio_index < 0)
		{
			LOGE("NSS sequence JSON has negative UpscalingRatiosIndices.{}: {}", ratio_key, sequence_json);
			return false;
		}

		const float default_near = root.value("NearPlane", 10.0f);
		const float far_value    = root.value("FarPlane", -1.0f);
		const float default_far  = (far_value < 0.0f) ? FLT_MAX : far_value;

		if (!root.contains("Frames") || !root["Frames"].is_array() || root["Frames"].empty())
		{
			LOGE("NSS dataset sequence has no Frames array: {}", sequence_json);
			return false;
		}
		const auto &json_frames = root["Frames"];
		frames_.resize(json_frames.size());
		const bool has_frame_indices = json_frames[0].contains("Frame");
		std::vector<bool> frame_seen(json_frames.size(), false);
		bool depth_convention_initialized = false;
		for (size_t index = 0; index < json_frames.size(); ++index)
		{
			const auto &json_frame = json_frames[index];
			if (json_frame.contains("Frame") != has_frame_indices || (has_frame_indices && !json_frame["Frame"].is_number_unsigned()))
			{
				LOGE("NSS sequence has inconsistent Frame metadata at entry {}: {}", index, sequence_json);
				return false;
			}
			const size_t frame_index = has_frame_indices ? json_frame["Frame"].get<size_t>() : index;
			if (frame_index >= frames_.size() || frame_seen[frame_index])
			{
				LOGE("NSS sequence has duplicate or out-of-range Frame index {}: {}", frame_index, sequence_json);
				return false;
			}
			frame_seen[frame_index] = true;
			auto       &frame_data = frames_[frame_index];
			frame_data.frame_time_delta_ms       = (frame_rate_hz_ > 0.0f) ? (1000.0f / frame_rate_hz_) : 16.67f;
			frame_data.motion_vector_scale_x     = -static_cast<float>(render_size_.width);
			frame_data.motion_vector_scale_y     = -static_cast<float>(render_size_.height);

			frame_data.camera_near          = json_frame.value("CameraNearPlane", default_near);
			const float camera_far          = json_frame.value("CameraFarPlane", default_far);
			frame_data.camera_far           = camera_far < 0.0f ? FLT_MAX : camera_far;
			const bool frame_depth_infinite = camera_far < 0.0f;
			if (!depth_convention_initialized)
			{
				depth_infinite_              = frame_depth_infinite;
				depth_convention_initialized = true;
			}
			else if (frame_depth_infinite != depth_infinite_)
			{
				LOGE("NSS sequence mixes finite and infinite CameraFarPlane values at frame {}: {}", index, sequence_json);
				return false;
			}
			frame_data.camera_fov_angle_vertical = json_frame.value("FovY", 0.75f);
			frame_data.exposure                  = config_.override_exposure ? config_.exposure : json_frame.value("Exposure", 1.0f);
			frame_data.reset                     = json_frame.value("Reset", frame_index == 0) || frame_index == 0;
			if (json_frame.contains("FrameTimeDelta"))
			{
				frame_data.frame_time_delta_ms = json_frame["FrameTimeDelta"].get<float>();
				if (!std::isfinite(frame_data.frame_time_delta_ms) || frame_data.frame_time_delta_ms < 0.0f)
				{
					LOGE("NSS sequence has non-finite FrameTimeDelta at frame {}: {}", index, sequence_json);
					return false;
				}
			}
			if (frame_data.exposure <= 0.0f)
			{
				LOGE("NSS sequence has invalid Exposure at frame {} (must be greater than zero): {}", index, sequence_json);
				return false;
			}

			if (json_frame.contains("MotionVectorScale"))
			{
				const auto &motion_vector_scale = json_frame["MotionVectorScale"];
				if (!motion_vector_scale.is_array() || motion_vector_scale.size() != 2 ||
				    !motion_vector_scale[0].is_number() || !motion_vector_scale[1].is_number())
				{
					LOGE("NSS sequence has invalid MotionVectorScale at frame {} (expected two finite numbers): {}", index, sequence_json);
					return false;
				}
				frame_data.motion_vector_scale_x = motion_vector_scale[0].get<float>();
				frame_data.motion_vector_scale_y = motion_vector_scale[1].get<float>();
				if (!std::isfinite(frame_data.motion_vector_scale_x) || !std::isfinite(frame_data.motion_vector_scale_y))
				{
					LOGE("NSS sequence has non-finite MotionVectorScale at frame {}: {}", index, sequence_json);
					return false;
				}
			}

			if (json_frame.contains("QuantisedPerRatioJitter") &&
			    json_frame["QuantisedPerRatioJitter"].is_array() &&
			    ratio_index < static_cast<int>(json_frame["QuantisedPerRatioJitter"].size()))
			{
				const auto &jitter  = json_frame["QuantisedPerRatioJitter"][ratio_index];
				frame_data.jitter_x = jitter.value("X", 0.0f);
				frame_data.jitter_y = jitter.value("Y", 0.0f);
			}
			else if (json_frame.contains("NormalizedPerRatioJitter") &&
			         json_frame["NormalizedPerRatioJitter"].is_array() &&
			         ratio_index < static_cast<int>(json_frame["NormalizedPerRatioJitter"].size()))
			{
				const auto &jitter  = json_frame["NormalizedPerRatioJitter"][ratio_index];
				frame_data.jitter_x = jitter.value("X", 0.0f);
				frame_data.jitter_y = jitter.value("Y", 0.0f);
			}
		}
	}
	catch (const json::exception &exception)
	{
		LOGE("Failed to parse NSS sequence JSON '{}': {}", sequence_json, exception.what());
		return false;
	}

	if (display_size_.width == 0 || display_size_.height == 0)
	{
		LOGE("NSS sequence JSON has invalid TargetResolution: {}", sequence_json);
		return false;
	}
	LOGI("  display_size = {}x{}, render_size = {}x{}",
	     display_size_.width, display_size_.height, render_size_.width, render_size_.height);

	size_t color_bytes = 0;
	size_t depth_bytes = 0;
	size_t mv_bytes    = 0;
	if (!vkb::sample::exr::vk_format_buffer_size(render_size_, color_format_, color_bytes) ||
	    !vkb::sample::exr::vk_format_buffer_size(render_size_, VK_FORMAT_R32_SFLOAT, depth_bytes) ||
	    !vkb::sample::exr::vk_format_buffer_size(render_size_, VK_FORMAT_R16G16_SFLOAT, mv_bytes))
	{
		LOGE("Unsupported NSS dataset format: {}", static_cast<int32_t>(color_format_));
		return false;
	}

	staging_buffers_.resize(swapchain_count);
	for (StagingBuffers &staging : staging_buffers_)
	{
		staging.color          = std::make_unique<vkb::core::BufferC>(vkb::core::BufferC::create_staging_buffer(device, color_bytes, nullptr));
		staging.depth          = std::make_unique<vkb::core::BufferC>(vkb::core::BufferC::create_staging_buffer(device, depth_bytes, nullptr));
		staging.motion_vectors = std::make_unique<vkb::core::BufferC>(vkb::core::BufferC::create_staging_buffer(device, mv_bytes, nullptr));
	}

	LOGI("Created {} NSS dataset staging triplets (color={}, depth={}, mv={} bytes)",
	     swapchain_count, color_bytes, depth_bytes, mv_bytes);
	return true;
}

bool NSSDatasetLoader::get_frame_params(int frame_index, NSSFrameParams &frame_params) const
{
	if (frame_index < 0 || frame_index >= static_cast<int>(frames_.size()))
		return false;

	const auto &metadata                   = frames_[frame_index];
	frame_params.reset                     = metadata.reset;
	frame_params.jitter_x                  = metadata.jitter_x * static_cast<float>(render_size_.width);
	frame_params.jitter_y                  = metadata.jitter_y * static_cast<float>(render_size_.height);
	frame_params.camera_near               = metadata.camera_near;
	frame_params.camera_far                = metadata.camera_far;
	frame_params.camera_fov_angle_vertical = metadata.camera_fov_angle_vertical;
	frame_params.exposure                  = metadata.exposure;
	frame_params.frame_time_delta_ms       = metadata.frame_time_delta_ms;
	frame_params.motion_vector_scale_x     = metadata.motion_vector_scale_x;
	frame_params.motion_vector_scale_y     = metadata.motion_vector_scale_y;
	return true;
}

bool NSSDatasetLoader::load_frame(int frame_index, uint32_t swapchain_index, NSSFrameParams &frame_params)
{
	if (!get_frame_params(frame_index, frame_params))
		return false;

	auto &staging = staging_buffers_[swapchain_index];
	char  path_buffer[512];
	auto  load_image = [&](const char *kind, VkFormat format,
                          const vkb::sample::exr::ExrLoadOptions &options,
                          void *destination, size_t destination_size) {
        VkExtent2D extent{};
        if (!vkb::sample::exr::load_exr_to_vk_format(path_buffer, format, options, destination, destination_size, extent))
            return false;
        if (extent.width == render_size_.width && extent.height == render_size_.height)
            return true;
        LOGE("NSS {} EXR has size {}x{}, expected {}x{}: {}",
		      kind, extent.width, extent.height, render_size_.width, render_size_.height, path_buffer);
        return false;
	};

	// Pack linear HDR color into the format consumed by NSS.
	snprintf(path_buffer, sizeof(path_buffer), "%s/%04d.exr", color_dir_.c_str(), frame_index);
	if (!load_image("color", color_format_, {}, staging.color->map(), static_cast<size_t>(staging.color->get_size())))
		return false;
	staging.color->flush();

	// Depth: R32F.
	snprintf(path_buffer, sizeof(path_buffer), "%s/%04d.exr", depth_dir_.c_str(), frame_index);
	vkb::sample::exr::ExrLoadOptions depth_options{};
	depth_options.flip_depth = config_.flip_depth;
	if (!load_image("depth", VK_FORMAT_R32_SFLOAT, depth_options, staging.depth->map(),
	                static_cast<size_t>(staging.depth->get_size())))
		return false;
	staging.depth->flush();

	// Motion: dataset is UV-space forward motion
	snprintf(path_buffer, sizeof(path_buffer), "%s/%04d.exr", motion_dir_.c_str(), frame_index);
	if (!load_image("motion", VK_FORMAT_R16G16_SFLOAT, {}, staging.motion_vectors->map(),
	                static_cast<size_t>(staging.motion_vectors->get_size())))
		return false;
	staging.motion_vectors->flush();

	return true;
}

uint64_t NSSDatasetLoader::total_frames() const
{
	return static_cast<uint64_t>(frames_.size());
}

VkBuffer NSSDatasetLoader::get_color_staging_buffer(uint32_t swapchain_index) const
{
	return staging_buffers_[swapchain_index].color->get_handle();
}

VkBuffer NSSDatasetLoader::get_depth_staging_buffer(uint32_t swapchain_index) const
{
	return staging_buffers_[swapchain_index].depth->get_handle();
}

VkBuffer NSSDatasetLoader::get_motion_vector_staging_buffer(uint32_t swapchain_index) const
{
	return staging_buffers_[swapchain_index].motion_vectors->get_handle();
}

VkExtent2D NSSDatasetLoader::get_display_size() const
{
	return display_size_;
}

VkExtent2D NSSDatasetLoader::get_render_size() const
{
	return render_size_;
}

const std::string &NSSDatasetLoader::get_ratio_dir() const
{
	return ratio_dir_;
}

bool NSSDatasetLoader::uses_inverted_depth() const
{
	return depth_inverted_;
}

bool NSSDatasetLoader::uses_infinite_far() const
{
	return depth_infinite_;
}