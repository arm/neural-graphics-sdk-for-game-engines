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
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>

#include <tinyexr/tinyexr.h>

// nlohmann/json — available via tinygltf (framework → third_party → tinygltf → json.hpp).
#include <json.hpp>
using json = nlohmann::json;

#include "core/util/logging.hpp"

// ──────────────────────────────────────────────────────────────────────────────
// float_to_half — IEEE 754 float32 → float16
// ──────────────────────────────────────────────────────────────────────────────

static uint16_t float_to_half(float f)
{
    uint32_t x;
    std::memcpy(&x, &f, sizeof(f));  // reinterpret bits as uint32_t

    uint32_t sign     = (x >> 31) & 0x1;
    int32_t  exp      = ((x >> 23) & 0xFF) - 127;  // float32 exponent
    uint32_t mantissa = x & 0x7FFFFF;              // float32 mantissa

    uint16_t h = 0;

    if (exp == 128)
    {  // Inf or NaN
        h = (sign << 15) | (0x1F << 10) | (mantissa ? 0x200 : 0);
    }
    else if (exp > 15)
    {  // Overflow → saturate to Inf
        h = (sign << 15) | (0x1F << 10);
    }
    else if (exp >= -14)
    {                     // Normalized half-float
        exp += 15;        // adjust bias from 127→15
        mantissa >>= 13;  // shift 23-bit mantissa to 10 bits
        h = (sign << 15) | (exp << 10) | mantissa;
    }
    else if (exp >= -24)
    {                          // Subnormal half-float
        mantissa |= 0x800000;  // restore implicit leading 1
        int shift = -14 - exp;
        mantissa >>= (13 + shift);
        h = (sign << 15) | mantissa;
    }
    else
    {  // Underflow → zero
        h = sign << 15;
    }

    return h;
}

// ──────────────────────────────────────────────────────────────────────────────
// EXR helpers
// ──────────────────────────────────────────────────────────────────────────────

static VkExtent2D exr_extent(const char *path)
{
	EXRVersion  version = {};
	EXRHeader   header  = {};
	const char *err     = nullptr;

	if (ParseEXRVersionFromFile(&version, path) != TINYEXR_SUCCESS)
		return {0, 0};
	if (ParseEXRHeaderFromFile(&header, &version, path, &err) != TINYEXR_SUCCESS)
	{
		FreeEXRErrorMessage(err);
		return {0, 0};
	}
	VkExtent2D ext{
	    static_cast<uint32_t>(header.data_window.max_x - header.data_window.min_x + 1),
	    static_cast<uint32_t>(header.data_window.max_y - header.data_window.min_y + 1)};
	FreeEXRHeader(&header);
	return ext;
}

// ──────────────────────────────────────────────────────────────────────────────
// Reinhard tonemapping
// ──────────────────────────────────────────────────────────────────────────────
static void apply_tonemap(float *pixels, int count)
{
	float exposure = std::exp(2.0f);

	for (int i = 0; i < count; i++)
	{
		float r = pixels[i * 4 + 0] * exposure;
		float g = pixels[i * 4 + 1] * exposure;
		float b = pixels[i * 4 + 2] * exposure;
		r = r / (1.0f + r);
		g = g / (1.0f + g);
		b = b / (1.0f + b);
		pixels[i * 4 + 0] = std::fmin(std::fmax(r, 0.0f), 1.0f);
		pixels[i * 4 + 1] = std::fmin(std::fmax(g, 0.0f), 1.0f);
		pixels[i * 4 + 2] = std::fmin(std::fmax(b, 0.0f), 1.0f);
		pixels[i * 4 + 3] = 1.0f;
	}
}

// ──────────────────────────────────────────────────────────────────────────────
// DatasetLoader implementation
// ──────────────────────────────────────────────────────────────────────────────

DatasetLoader::~DatasetLoader()
{
	for (auto &s : color_staging_) destroy_staging_buffer(s);
	for (auto &s : depth_staging_) destroy_staging_buffer(s);
	for (auto &s : mv_staging_)    destroy_staging_buffer(s);
}

void DatasetLoader::destroy_staging_buffer(StagingBuffer &sb)
{
	if (sb.mapped)
		vkUnmapMemory(device_, sb.memory);
	if (sb.buffer != VK_NULL_HANDLE)
		vkDestroyBuffer(device_, sb.buffer, nullptr);
	if (sb.memory != VK_NULL_HANDLE)
		vkFreeMemory(device_, sb.memory, nullptr);
	sb = {};
}

/// Create a host-visible, host-coherent staging buffer and persistently map it.
/// Used for CPU→GPU data transfer (EXR pixels → vkCmdCopyBufferToImage).
DatasetLoader::StagingBuffer DatasetLoader::create_staging_buffer(VkPhysicalDevice gpu, VkDeviceSize size)
{
	StagingBuffer sb{};

	VkBufferCreateInfo buf_ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
	buf_ci.size        = size;
	buf_ci.usage       = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;  // CPU writes, GPU reads
	buf_ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	if (vkCreateBuffer(device_, &buf_ci, nullptr, &sb.buffer) != VK_SUCCESS)
		assert(false && "vkCreateBuffer failed for staging buffer");

	VkMemoryRequirements mem_req{};
	vkGetBufferMemoryRequirements(device_, sb.buffer, &mem_req);

	VkPhysicalDeviceMemoryProperties mem_props{};
	vkGetPhysicalDeviceMemoryProperties(gpu, &mem_props);

	uint32_t mem_idx = UINT32_MAX;
	for (uint32_t j = 0; j < mem_props.memoryTypeCount; ++j)
	{
		const bool type_match = (mem_req.memoryTypeBits & (1u << j)) != 0;
		const bool flag_match = (mem_props.memoryTypes[j].propertyFlags &
		                         (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
		                        (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
		if (type_match && flag_match) { mem_idx = j; break; }
	}
	assert(mem_idx != UINT32_MAX && "No host-visible memory type for staging buffer");

	VkMemoryAllocateInfo alloc_ci{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
	alloc_ci.allocationSize  = mem_req.size;
	alloc_ci.memoryTypeIndex = mem_idx;
	vkAllocateMemory(device_, &alloc_ci, nullptr, &sb.memory);
	vkBindBufferMemory(device_, sb.buffer, sb.memory, 0);
	vkMapMemory(device_, sb.memory, 0, mem_req.size, 0, &sb.mapped);

	return sb;
}

bool DatasetLoader::init(VkDevice device, VkPhysicalDevice gpu,
                         const std::string &dataset_dir, const std::string &dataset_sequence,
                         uint32_t swapchain_count)
{
	device_ = device;

	std::string sequence_json = dataset_dir + "/" + dataset_sequence + ".json";
	color_dir_ = dataset_dir + "/ground_truth/" + dataset_sequence;
	depth_dir_ = dataset_dir + "/x2/depth/" + dataset_sequence;
	mv_dir_    = dataset_dir + "/x2/motion_m2/" + dataset_sequence;
	// Fall back to motion_t-2 when motion_m2 is absent.
	if (!std::filesystem::exists(mv_dir_))
		mv_dir_ = dataset_dir + "/x2/motion_t-2/" + dataset_sequence;

	LOGI("DatasetLoader: dir={}, sequence={}", dataset_dir, dataset_sequence);
	LOGI("  color_dir = {}", color_dir_);
	LOGI("  depth_dir = {}", depth_dir_);
	LOGI("  mv_dir    = {}", mv_dir_);

	std::ifstream ifs(sequence_json);
	if (!ifs.is_open())
	{
		LOGE("Cannot open sequence JSON: {}", sequence_json);
		return false;
	}

	json root = json::parse(ifs);
	fps_      = root.value("EmulatedFramerate", 60.0f);
	// The current dataset bundle always stores depth in the same pre-inverted
	// convention handled below, so ReverseZ does not currently change the load path. 
	reverse_z_ = root.value("ReverseZ", false);

	auto &json_frames = root["Frames"];
	frames_.resize(json_frames.size());

	for (size_t i = 0; i < json_frames.size(); i++)
	{
		auto &json_frame = json_frames[i];
		auto &frame_data = frames_[i];

		frame_data.fovY       = json_frame.value("FovY", 0.75f);
		frame_data.cameraNear = json_frame.value("CameraNearPlane", 10.0f);
		float farVal  = json_frame.value("CameraFarPlane", -1.0f);
		frame_data.cameraFar  = (farVal < 0) ? 5000.0f : farVal;

		if (json_frame.contains("NormalizedPerRatioJitter") && json_frame["NormalizedPerRatioJitter"].size() > 0)
		{
			frame_data.jitterX = json_frame["NormalizedPerRatioJitter"][0].value("X", 0.0f);
			frame_data.jitterY = json_frame["NormalizedPerRatioJitter"][0].value("Y", 0.0f);
		}

		auto &vp_arr = json_frame["ViewProjection"];

		// Column-major → row-major transpose
		frame_data.viewProjection[0]  = vp_arr[0].get<float>();
		frame_data.viewProjection[1]  = vp_arr[4].get<float>();
		frame_data.viewProjection[2]  = vp_arr[8].get<float>();
		frame_data.viewProjection[3]  = vp_arr[12].get<float>();
		frame_data.viewProjection[4]  = vp_arr[1].get<float>();
		frame_data.viewProjection[5]  = vp_arr[5].get<float>();
		frame_data.viewProjection[6]  = vp_arr[9].get<float>();
		frame_data.viewProjection[7]  = vp_arr[13].get<float>();
		frame_data.viewProjection[8]  = vp_arr[2].get<float>();
		frame_data.viewProjection[9]  = vp_arr[6].get<float>();
		frame_data.viewProjection[10] = vp_arr[10].get<float>();
		frame_data.viewProjection[11] = vp_arr[14].get<float>();
		frame_data.viewProjection[12] = vp_arr[3].get<float>();
		frame_data.viewProjection[13] = vp_arr[7].get<float>();
		frame_data.viewProjection[14] = vp_arr[11].get<float>();
		frame_data.viewProjection[15] = vp_arr[15].get<float>();

	}

	LOGI("Parsed sequence: {} frames, {}fps, reverseZ={}", frames_.size(), fps_, reverse_z_);

	char path_buf[512];
	snprintf(path_buf, sizeof(path_buf), "%s/%04d.exr", color_dir_.c_str(), 0);
	display_size_ = exr_extent(path_buf);
	snprintf(path_buf, sizeof(path_buf), "%s/%04d.exr", depth_dir_.c_str(), 0);
	render_size_ = exr_extent(path_buf);

	if (display_size_.width == 0 || render_size_.width == 0)
	{
		LOGE("Failed to probe EXR dimensions (color: {}x{}, depth: {}x{})",
		     display_size_.width, display_size_.height, render_size_.width, render_size_.height);
		return false;
	}

	LOGI("  display_size = {}x{}, render_size = {}x{}",
	     display_size_.width, display_size_.height, render_size_.width, render_size_.height);

	{
		VkDeviceSize color_bytes = static_cast<VkDeviceSize>(display_size_.width)
		                         * display_size_.height * 4;   // R8G8B8A8
		VkDeviceSize depth_bytes = static_cast<VkDeviceSize>(render_size_.width)
		                         * render_size_.height * 4;   // R32F
		VkDeviceSize mv_bytes    = static_cast<VkDeviceSize>(render_size_.width)
		                         * render_size_.height * 4;   // R16G16

		color_staging_.resize(swapchain_count);
		depth_staging_.resize(swapchain_count);
		mv_staging_.resize(swapchain_count);

		for (uint32_t i = 0; i < swapchain_count; ++i)
		{
			color_staging_[i] = create_staging_buffer(gpu, color_bytes);
			depth_staging_[i] = create_staging_buffer(gpu, depth_bytes);
			mv_staging_[i]    = create_staging_buffer(gpu, mv_bytes);
		}
		LOGI("Created {} x 3 staging buffers (color={}, depth={}, mv={} bytes)",
		     swapchain_count, color_bytes, depth_bytes, mv_bytes);
	}

	return true;
}

int DatasetLoader::total_frames() const
{
	return static_cast<int>((frames_.size() + 1) / 2);
}

bool DatasetLoader::get_frame_params(int frame, FrameParams &out) const
{
	int even_frame = frame * 2;
	if (even_frame < 0 || even_frame >= static_cast<int>(frames_.size()))
		return false;

	const auto &frame_data = frames_[even_frame];
	std::memcpy(out.viewProjection, frame_data.viewProjection, sizeof(out.viewProjection));
	out.jitterX                = frame_data.jitterX * static_cast<float>(render_size_.width);
	out.jitterY                = frame_data.jitterY * static_cast<float>(render_size_.height);
	out.cameraNear             = frame_data.cameraNear;
	out.cameraFar              = frame_data.cameraFar;
	out.cameraFovAngleVertical = frame_data.fovY;
	float deltaTimeMS = (fps_ > 0) ? (1000.0f / fps_) : 1.0f;
	out.frameTimeDelta         = (deltaTimeMS > 1.0f) ? deltaTimeMS : 1.0f;
	out.motionVectorScaleX     = 1.0f;   // Dataset MVs are pre-multiplied by extent
	out.motionVectorScaleY     = 1.0f;

	return true;
}

bool DatasetLoader::load_frame(int frame, uint32_t swapchain_index, FrameParams &out)
{
	int even_frame = frame * 2;
	if (even_frame < 0 || even_frame >= static_cast<int>(frames_.size()))
		return false;

	const int display_pixels = static_cast<int>(display_size_.width) * static_cast<int>(display_size_.height);
	const int render_pixels = static_cast<int>(render_size_.width) * static_cast<int>(render_size_.height);

	// ── 1. Load color (display resolution, float RGBA → tonemap → u8) ──
	{
		char path[512];
		snprintf(path, sizeof(path), "%s/%04d.exr", color_dir_.c_str(), even_frame);

		int         w, h;
		const char *err = nullptr;
		float      *tmp = nullptr;
		if (LoadEXR(&tmp, &w, &h, path, &err) != TINYEXR_SUCCESS)
		{
			LOGE("Failed to load color EXR: {} — {}", path, err ? err : "unknown");
			FreeEXRErrorMessage(err);
			return false;
		}
		apply_tonemap(tmp, display_pixels);

		// Convert f32 RGBA → u8 RGBA directly into staging buffer
		auto *dst = static_cast<uint8_t *>(color_staging_[swapchain_index].mapped);
		for (int i = 0; i < display_pixels; ++i)
		{
			dst[i * 4 + 0] = static_cast<uint8_t>(std::fmin(tmp[i * 4 + 0], 1.0f) * 255.0f + 0.5f);
			dst[i * 4 + 1] = static_cast<uint8_t>(std::fmin(tmp[i * 4 + 1], 1.0f) * 255.0f + 0.5f);
			dst[i * 4 + 2] = static_cast<uint8_t>(std::fmin(tmp[i * 4 + 2], 1.0f) * 255.0f + 0.5f);
			dst[i * 4 + 3] = static_cast<uint8_t>(std::fmin(tmp[i * 4 + 3], 1.0f) * 255.0f + 0.5f);
		}
		free(tmp);
	}

	// ── 2. Load depth (render resolution, R32F) ──
	{
		char path[512];
		snprintf(path, sizeof(path), "%s/%04d.exr", depth_dir_.c_str(), even_frame);

		int         w, h;
		const char *err = nullptr;
		float      *tmp = nullptr;
		if (LoadEXR(&tmp, &w, &h, path, &err) != TINYEXR_SUCCESS)
		{
			LOGE("Failed to load depth EXR: {} — {}", path, err ? err : "unknown");
			FreeEXRErrorMessage(err);
			return false;
		}
		auto *dst = static_cast<float *>(depth_staging_[swapchain_index].mapped);
		for (int i = 0; i < w * h; i++)
		{
			float d = tmp[i * 4 + 0];
			// The replay datasets store depth as (1 - z). NFRU is configured with
			// ENABLE_DEPTH_INVERTED and expects the usual reversed-Z depth value, so
			// convert the dataset back into that convention before uploading.
			d = 1.0f - d;
			dst[i] = d;
		}
		free(tmp);
	}

	// ── 3. Load motion vectors (render resolution, RG → half) ──
	//    LoadEXR always returns interleaved RGBA float32; channel 0 = R (X), channel 1 = G (Y).
	{
		char path[512];
		snprintf(path, sizeof(path), "%s/%04d.exr", mv_dir_.c_str(), even_frame);

		int         w, h;
		const char *err = nullptr;
		float      *tmp = nullptr;

		if (LoadEXR(&tmp, &w, &h, path, &err) == TINYEXR_SUCCESS)
		{
			auto       *dst = static_cast<uint16_t *>(mv_staging_[swapchain_index].mapped);
			const float wf  = static_cast<float>(render_size_.width);
			const float hf  = static_cast<float>(render_size_.height);

			for (int i = 0; i < render_pixels; i++)
			{
				dst[i * 2 + 0] = float_to_half(tmp[i * 4 + 0] * -1.0f * wf);
				dst[i * 2 + 1] = float_to_half(tmp[i * 4 + 1] * -1.0f * hf);
			}
			free(tmp);
		}
		else
		{
			LOGW("Failed to load MV EXR: {} — {} → zero-filling", path, err ? err : "unknown");
			FreeEXRErrorMessage(err);
			// No MV available for this frame → zero fill staging
			std::memset(mv_staging_[swapchain_index].mapped, 0,
			            static_cast<size_t>(render_pixels) * 4);
		}
	}

	// ── 4. Fill output params (camera / timing — from parsed JSON, no EXR I/O) ──
	get_frame_params(frame, out);

	return true;
}

VkBuffer DatasetLoader::get_color_staging(uint32_t swapchain_index) const { return color_staging_[swapchain_index].buffer; }
VkBuffer DatasetLoader::get_depth_staging(uint32_t swapchain_index) const { return depth_staging_[swapchain_index].buffer; }
VkBuffer DatasetLoader::get_mv_staging(uint32_t swapchain_index) const    { return mv_staging_[swapchain_index].buffer; }

VkExtent2D DatasetLoader::get_display_size() const
{
	return display_size_;
}

VkExtent2D DatasetLoader::get_render_size() const
{
	return render_size_;
}
