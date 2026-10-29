/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstddef>

#include <filesystem>
#include <vulkan/vulkan.h>

namespace vkb
{
namespace sample
{
namespace exr
{
struct ExrLoadOptions
{
	bool  tonemap{false};
	float exposure{1.0f};
	bool  flip_depth{false};
};

bool probe_extent(const char *path, VkExtent2D &extent);

bool load_exr_to_vk_format(const char *path, VkFormat format, const ExrLoadOptions &options,
                           void *destination, size_t destination_size, VkExtent2D &extent);

/// Returns the byte size of tightly packed image data for the extent and Vulkan format.
bool vk_format_buffer_size(VkExtent2D extent, VkFormat format, size_t &size);

bool save_vk_format_to_exr(const char *path, VkFormat format, const void *source, size_t source_size, VkExtent2D extent);

bool create_directories(const std::filesystem::path &directory);
}        // namespace exr
}        // namespace sample
}        // namespace vkb