/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT
 */

#include "exr_image.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <new>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/packing.hpp>

#include <tinyexr/tinyexr.h>

#include "core/util/logging.hpp"

#include <system_error>

namespace
{
struct TinyExrData
{
	EXRHeader header{};
	EXRImage  image{};
	bool      image_owned{false};

	TinyExrData()
	{
		InitEXRHeader(&header);
		InitEXRImage(&image);
	}

	~TinyExrData()
	{
		if (image_owned)
			FreeEXRImage(&image);
		FreeEXRHeader(&header);
	}

	TinyExrData(const TinyExrData &)            = delete;
	TinyExrData &operator=(const TinyExrData &) = delete;
};

bool checked_pixel_count(VkExtent2D extent, size_t &pixel_count)
{
	if (extent.width == 0 || extent.height == 0)
		return false;

	const size_t width  = extent.width;
	const size_t height = extent.height;
	if (width > std::numeric_limits<size_t>::max() / height)
		return false;

	pixel_count = width * height;
	return true;
}

size_t bytes_per_pixel(VkFormat format)
{
	switch (format)
	{
		case VK_FORMAT_B10G11R11_UFLOAT_PACK32:
		case VK_FORMAT_R8G8B8A8_UNORM:
		case VK_FORMAT_R32_SFLOAT:
		case VK_FORMAT_R16G16_SFLOAT:
			return 4;
		case VK_FORMAT_R16G16B16A16_SFLOAT:
			return 8;
		case VK_FORMAT_R32G32B32A32_SFLOAT:
			return 16;
		default:
			return 0;
	}
}

bool checked_vk_format_buffer_size(VkExtent2D extent, VkFormat format, size_t &size)
{
	const size_t pixel_size  = bytes_per_pixel(format);
	size_t       pixel_count = 0;
	if (pixel_size == 0 || !checked_pixel_count(extent, pixel_count) ||
	    pixel_count > std::numeric_limits<size_t>::max() / pixel_size)
	{
		size = 0;
		return false;
	}

	size = pixel_count * pixel_size;
	return true;
}

bool is_color_format(VkFormat format)
{
	switch (format)
	{
		case VK_FORMAT_B10G11R11_UFLOAT_PACK32:
		case VK_FORMAT_R8G8B8A8_UNORM:
		case VK_FORMAT_R16G16B16A16_SFLOAT:
		case VK_FORMAT_R32G32B32A32_SFLOAT:
			return true;
		default:
			return false;
	}
}

bool is_depth_format(VkFormat format)
{
	return format == VK_FORMAT_R32_SFLOAT;
}

bool is_motion_format(VkFormat format)
{
	return format == VK_FORMAT_R16G16_SFLOAT;
}

int find_channel(const EXRHeader &header, const char *name)
{
	for (int channel = 0; channel < header.num_channels; ++channel)
	{
		if (std::strcmp(header.channels[channel].name, name) == 0)
			return channel;
	}
	return -1;
}

float read_channel_value(const EXRHeader &header, unsigned char *const *channels, int channel, size_t pixel)
{
	if (header.requested_pixel_types[channel] == TINYEXR_PIXELTYPE_UINT)
		return static_cast<float>(reinterpret_cast<const uint32_t *>(channels[channel])[pixel]);
	return reinterpret_cast<const float *>(channels[channel])[pixel];
}

uint32_t read_u32(const uint8_t *source)
{
	uint32_t value = 0;
	std::memcpy(&value, source, sizeof(value));
	return value;
}

void write_u32(uint8_t *destination, uint32_t value)
{
	std::memcpy(destination, &value, sizeof(value));
}

void write_float(uint8_t *destination, float value)
{
	std::memcpy(destination, &value, sizeof(value));
}

uint8_t encode_unorm8(float value)
{
	if (std::isnan(value) || value <= 0.0f)
		return 0;
	if (value >= 1.0f)
		return 255;
	return static_cast<uint8_t>(value * 255.0f + 0.5f);
}

uint32_t pack_unsigned_float(float value, uint32_t mantissa_bits)
{
	// Unsigned Vulkan float formats encode NaN with an all-one exponent and a nonzero mantissa.
	if (std::isnan(value))
		return (0x1fu << mantissa_bits) | 1u;
	if (!(value > 0.0f))
		return 0;
	if (!std::isfinite(value))
		return 0x1fu << mantissa_bits;

	constexpr int  exponent_bias = 15;
	const float    min_normal    = std::ldexp(1.0f, 1 - exponent_bias);
	const uint32_t mantissa_mask = (1u << mantissa_bits) - 1u;
	if (value < min_normal)
	{
		const float quantum = std::ldexp(1.0f, 1 - exponent_bias - static_cast<int>(mantissa_bits));
		return std::min(static_cast<uint32_t>(std::lround(value / quantum)), 1u << mantissa_bits);
	}

	int      exponent    = 0;
	float    significand = std::frexp(value, &exponent) * 2.0f;
	int      biased_exp  = exponent - 1 + exponent_bias;
	uint32_t mantissa    = static_cast<uint32_t>(std::lround((significand - 1.0f) * static_cast<float>(1u << mantissa_bits)));
	if (mantissa > mantissa_mask)
	{
		mantissa = 0;
		++biased_exp;
	}
	if (biased_exp >= 0x1f)
		return 0x1fu << mantissa_bits;

	return (static_cast<uint32_t>(biased_exp) << mantissa_bits) | mantissa;
}

uint32_t pack_r11g11b10(const glm::vec3 &color)
{
	return pack_unsigned_float(color.r, 6) |
	       (pack_unsigned_float(color.g, 6) << 11) |
	       (pack_unsigned_float(color.b, 5) << 22);
}

float unpack_unsigned_float(uint32_t value, uint32_t mantissa_bits)
{
	const uint32_t mantissa_mask = (1u << mantissa_bits) - 1u;
	const uint32_t mantissa      = value & mantissa_mask;
	const uint32_t exponent      = (value >> mantissa_bits) & 0x1fu;
	if (exponent == 0)
		return std::ldexp(static_cast<float>(mantissa), -14 - static_cast<int>(mantissa_bits));
	if (exponent == 0x1f)
		return mantissa == 0 ? std::numeric_limits<float>::infinity() : std::numeric_limits<float>::quiet_NaN();

	return std::ldexp(1.0f + static_cast<float>(mantissa) / static_cast<float>(1u << mantissa_bits),
	                  static_cast<int>(exponent) - 15);
}

glm::vec3 unpack_r11g11b10(uint32_t value)
{
	return {
	    unpack_unsigned_float(value, 6),
	    unpack_unsigned_float(value >> 11, 6),
	    unpack_unsigned_float(value >> 22, 5)};
}

void apply_tonemap(float &red, float &green, float &blue, float &alpha, float exposure)
{
	const float scale = exposure;
	red               = red * scale / (1.0f + red * scale);
	green             = green * scale / (1.0f + green * scale);
	blue              = blue * scale / (1.0f + blue * scale);
	red               = std::fmin(std::fmax(red, 0.0f), 1.0f);
	green             = std::fmin(std::fmax(green, 0.0f), 1.0f);
	blue              = std::fmin(std::fmax(blue, 0.0f), 1.0f);
	alpha             = 1.0f;
}

void write_vk_pixel(VkFormat format, uint8_t *destination, float red, float green, float blue, float alpha)
{
	switch (format)
	{
		case VK_FORMAT_B10G11R11_UFLOAT_PACK32:
			write_u32(destination, pack_r11g11b10(glm::vec3(red, green, blue)));
			return;
		case VK_FORMAT_R8G8B8A8_UNORM:
			destination[0] = encode_unorm8(red);
			destination[1] = encode_unorm8(green);
			destination[2] = encode_unorm8(blue);
			destination[3] = encode_unorm8(alpha);
			return;
		case VK_FORMAT_R16G16B16A16_SFLOAT:
			write_u32(destination, glm::packHalf2x16(glm::vec2(red, green)));
			write_u32(destination + sizeof(uint32_t), glm::packHalf2x16(glm::vec2(blue, alpha)));
			return;
		case VK_FORMAT_R32G32B32A32_SFLOAT:
			write_float(destination, red);
			write_float(destination + sizeof(float), green);
			write_float(destination + sizeof(float) * 2, blue);
			write_float(destination + sizeof(float) * 3, alpha);
			return;
		case VK_FORMAT_R32_SFLOAT:
			write_float(destination, red);
			return;
		case VK_FORMAT_R16G16_SFLOAT:
			write_u32(destination, glm::packHalf2x16(glm::vec2(red, green)));
			return;
		default:
			return;
	}
}

bool save_rgba_exr_impl(const char *path, const float *pixels, VkExtent2D extent)
{
	if (pixels == nullptr || extent.width > static_cast<uint32_t>(std::numeric_limits<int>::max()) ||
	    extent.height > static_cast<uint32_t>(std::numeric_limits<int>::max()))
	{
		LOGE("Cannot save invalid color EXR image: {}", path);
		return false;
	}

	const char *error  = nullptr;
	const int   result = SaveEXR(pixels, static_cast<int>(extent.width), static_cast<int>(extent.height), 4, 0, path, &error);
	if (result != TINYEXR_SUCCESS)
	{
		LOGE("Failed to save EXR '{}': {}", path, error ? error : "unknown");
		FreeEXRErrorMessage(error);
		return false;
	}

	LOGI("Saved: {}", path);
	return true;
}

bool decode_vk_format_to_rgba(VkFormat format, const void *source, size_t source_size, VkExtent2D extent,
                              std::vector<float> &rgba)
{
	size_t pixel_count   = 0;
	size_t required_size = 0;
	if (source == nullptr)
	{
		LOGE("Cannot convert null Vulkan image data to EXR");
		return false;
	}
	if (!checked_pixel_count(extent, pixel_count) || pixel_count > std::numeric_limits<size_t>::max() / 4 ||
	    !checked_vk_format_buffer_size(extent, format, required_size) || source_size < required_size)
	{
		LOGE("Cannot convert Vulkan image data to EXR: invalid extent, format, or source size");
		return false;
	}

	try
	{
		rgba.assign(pixel_count * 4, 0.0f);
	}
	catch (const std::bad_alloc &)
	{
		LOGE("Cannot allocate temporary RGBA data for EXR output");
		return false;
	}

	const auto *input = static_cast<const uint8_t *>(source);
	for (size_t index = 0; index < pixel_count; ++index)
	{
		float *pixel = rgba.data() + index * 4;
		switch (format)
		{
			case VK_FORMAT_B10G11R11_UFLOAT_PACK32:
			{
				const glm::vec3 color = unpack_r11g11b10(read_u32(input + index * sizeof(uint32_t)));
				pixel[0]              = color.r;
				pixel[1]              = color.g;
				pixel[2]              = color.b;
				pixel[3]              = 1.0f;
				break;
			}
			case VK_FORMAT_R8G8B8A8_UNORM:
				pixel[0] = input[index * 4 + 0] / 255.0f;
				pixel[1] = input[index * 4 + 1] / 255.0f;
				pixel[2] = input[index * 4 + 2] / 255.0f;
				pixel[3] = input[index * 4 + 3] / 255.0f;
				break;
			case VK_FORMAT_R16G16B16A16_SFLOAT:
			{
				const glm::vec2 rg = glm::unpackHalf2x16(read_u32(input + index * 8));
				const glm::vec2 ba = glm::unpackHalf2x16(read_u32(input + index * 8 + sizeof(uint32_t)));
				pixel[0]           = rg.x;
				pixel[1]           = rg.y;
				pixel[2]           = ba.x;
				pixel[3]           = ba.y;
				break;
			}
			case VK_FORMAT_R32G32B32A32_SFLOAT:
				std::memcpy(pixel, input + index * 16, 16);
				break;
			case VK_FORMAT_R32_SFLOAT:
				std::memcpy(pixel, input + index * sizeof(float), sizeof(float));
				pixel[1] = 0.0f;
				pixel[2] = 0.0f;
				pixel[3] = 1.0f;
				break;
			case VK_FORMAT_R16G16_SFLOAT:
			{
				const glm::vec2 rg = glm::unpackHalf2x16(read_u32(input + index * sizeof(uint32_t)));
				pixel[0]           = rg.x;
				pixel[1]           = rg.y;
				pixel[2]           = 0.0f;
				pixel[3]           = 1.0f;
				break;
			}
			default:
				LOGE("Cannot convert unsupported Vulkan format {} to EXR", static_cast<int32_t>(format));
				return false;
		}
	}
	return true;
}
}        // namespace

namespace vkb
{
namespace sample
{
namespace exr
{
bool probe_extent(const char *path, VkExtent2D &extent)
{
	extent = {};

	EXRVersion  version{};
	EXRHeader   header{};
	const char *error = nullptr;
	InitEXRHeader(&header);
	if (ParseEXRVersionFromFile(&version, path) != TINYEXR_SUCCESS)
	{
		LOGE("Failed to parse EXR version: {}", path);
		return false;
	}
	if (ParseEXRHeaderFromFile(&header, &version, path, &error) != TINYEXR_SUCCESS)
	{
		LOGE("Failed to parse EXR header '{}': {}", path, error ? error : "unknown");
		FreeEXRErrorMessage(error);
		FreeEXRHeader(&header);
		return false;
	}

	const int64_t width  = static_cast<int64_t>(header.data_window.max_x) - header.data_window.min_x + 1;
	const int64_t height = static_cast<int64_t>(header.data_window.max_y) - header.data_window.min_y + 1;
	FreeEXRHeader(&header);
	if (width <= 0 || height <= 0 || width > std::numeric_limits<uint32_t>::max() || height > std::numeric_limits<uint32_t>::max())
	{
		LOGE("Invalid EXR extent in {}: {}x{}", path, width, height);
		return false;
	}

	extent = {static_cast<uint32_t>(width), static_cast<uint32_t>(height)};
	return true;
}

bool load_exr_to_vk_format(const char *path, VkFormat format, const ExrLoadOptions &options,
                           void *destination, size_t destination_size, VkExtent2D &extent)
{
	extent = {};
	if (destination == nullptr || bytes_per_pixel(format) == 0)
	{
		LOGE("Invalid EXR load destination or Vulkan format: {}", static_cast<int32_t>(format));
		return false;
	}

	EXRVersion version{};
	if (ParseEXRVersionFromFile(&version, path) != TINYEXR_SUCCESS)
	{
		LOGE("Failed to parse EXR version: {}", path);
		return false;
	}
	if (version.multipart || version.non_image)
	{
		LOGE("Unsupported multipart or deep EXR: {}", path);
		return false;
	}

	TinyExrData source{};
	const char *error = nullptr;
	if (ParseEXRHeaderFromFile(&source.header, &version, path, &error) != TINYEXR_SUCCESS)
	{
		LOGE("Failed to parse EXR header '{}': {}", path, error ? error : "unknown");
		FreeEXRErrorMessage(error);
		return false;
	}
	const bool color_format   = is_color_format(format);
	const bool depth_format   = is_depth_format(format);
	const bool motion_format  = is_motion_format(format);
	const bool single_channel = source.header.num_channels == 1;
	const int  red            = find_channel(source.header, "R");
	const int  green          = find_channel(source.header, "G");
	const int  blue           = find_channel(source.header, "B");
	const int  alpha          = find_channel(source.header, "A");
	const bool valid_channels = color_format ? red >= 0 && green >= 0 && blue >= 0 :
	                            depth_format ? (single_channel || red >= 0) :
	                                           motion_format && red >= 0 && green >= 0;
	if (!valid_channels)
	{
		const char *expected_channels = color_format ? "R/G/B with optional A" :
		                                depth_format ? "one channel or an R channel" :
		                                               "R/G with optional extra channels";
		LOGE("Unsupported EXR channels in '{}': expected {}", path, expected_channels);
		return false;
	}

	for (int channel = 0; channel < source.header.num_channels; ++channel)
	{
		if (source.header.pixel_types[channel] == TINYEXR_PIXELTYPE_HALF)
			source.header.requested_pixel_types[channel] = TINYEXR_PIXELTYPE_FLOAT;
	}
	if (LoadEXRImageFromFile(&source.image, &source.header, path, &error) != TINYEXR_SUCCESS)
	{
		LOGE("Failed to load EXR '{}': {}", path, error ? error : "unknown");
		FreeEXRErrorMessage(error);
		return false;
	}
	source.image_owned = true;
	if (source.image.width <= 0 || source.image.height <= 0)
	{
		LOGE("Invalid EXR extent in {}: {}x{}", path, source.image.width, source.image.height);
		return false;
	}

	extent               = {static_cast<uint32_t>(source.image.width), static_cast<uint32_t>(source.image.height)};
	size_t required_size = 0;
	if (!checked_vk_format_buffer_size(extent, format, required_size) || destination_size < required_size)
	{
		LOGE("EXR destination buffer is too small for {}: {}", path, static_cast<int32_t>(format));
		return false;
	}

	const size_t pixel_size = bytes_per_pixel(format);
	auto         copy_pixel = [&](unsigned char *const *channel_planes, size_t source_pixel, size_t destination_pixel) {
        float red_value   = 0.0f;
        float green_value = 0.0f;
        float blue_value  = 0.0f;
        float alpha_value = 1.0f;
        if (depth_format)
        {
            red_value = read_channel_value(source.header, channel_planes, single_channel ? 0 : red, source_pixel);
            if (options.flip_depth)
                red_value = 1.0f - red_value;
        }
        else
        {
            red_value   = read_channel_value(source.header, channel_planes, red, source_pixel);
            green_value = read_channel_value(source.header, channel_planes, green, source_pixel);
            if (color_format)
            {
                blue_value  = read_channel_value(source.header, channel_planes, blue, source_pixel);
                alpha_value = alpha >= 0 ? read_channel_value(source.header, channel_planes, alpha, source_pixel) : 1.0f;
                if (options.tonemap)
                    apply_tonemap(red_value, green_value, blue_value, alpha_value, options.exposure);
            }
        }
        write_vk_pixel(format, static_cast<uint8_t *>(destination) + destination_pixel * pixel_size,
		                       red_value, green_value, blue_value, alpha_value);
	};

	if (source.header.tiled)
	{
		const size_t tile_stride = static_cast<size_t>(source.header.tile_size_x);
		const size_t tile_height = static_cast<size_t>(source.header.tile_size_y);
		for (int tile_index = 0; tile_index < source.image.num_tiles; ++tile_index)
		{
			const EXRTile &tile = source.image.tiles[tile_index];
			for (int y = 0; y < tile.height; ++y)
			{
				for (int x = 0; x < tile.width; ++x)
				{
					const size_t image_x = static_cast<size_t>(tile.offset_x) * tile_stride + static_cast<size_t>(x);
					const size_t image_y = static_cast<size_t>(tile.offset_y) * tile_height + static_cast<size_t>(y);
					if (image_x >= extent.width || image_y >= extent.height)
						continue;
					const size_t source_pixel = static_cast<size_t>(x) + static_cast<size_t>(y) * tile_stride;
					copy_pixel(tile.images, source_pixel, image_x + image_y * extent.width);
				}
			}
		}
	}
	else
	{
		const size_t pixel_count = static_cast<size_t>(extent.width) * extent.height;
		for (size_t pixel = 0; pixel < pixel_count; ++pixel)
			copy_pixel(source.image.images, pixel, pixel);
	}

	return true;
}

bool vk_format_buffer_size(VkExtent2D extent, VkFormat format, size_t &size)
{
	return checked_vk_format_buffer_size(extent, format, size);
}

bool save_vk_format_to_exr(const char *path, VkFormat format, const void *source, size_t source_size, VkExtent2D extent)
{
	std::vector<float> rgba;
	if (!decode_vk_format_to_rgba(format, source, source_size, extent, rgba))
		return false;
	return save_rgba_exr_impl(path, rgba.data(), extent);
}

bool create_directories(const std::filesystem::path &directory)
{
	std::error_code error;
	std::filesystem::create_directories(directory, error);
	if (error)
	{
		LOGE("Failed to create output directory '{}': {}", directory.string(), error.message());
		return false;
	}
	return true;
}

}        // namespace exr
}        // namespace sample
}        // namespace vkb
