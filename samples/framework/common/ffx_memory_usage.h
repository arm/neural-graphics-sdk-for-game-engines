/* Copyright (c) 2026, Arm Limited and Contributors
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
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

#pragma once

#include <ffx_api_types.h>

namespace vkb
{
class FfxMemoryUsageTracker
{
  public:
	bool update_gpu_usage(const FfxApiEffectMemoryUsage &usage, bool force_log = false)
	{
		const bool changed = force_log || !has_gpu_usage_ || gpu_usage_changed(gpu_usage_, usage);
		gpu_usage_         = usage;
		has_gpu_usage_     = true;
		return changed;
	}

	bool update_cpu_usage(const FfxApiEffectCpuMemoryUsage &usage, bool force_log = false)
	{
		const bool changed = force_log || !has_cpu_usage_ || cpu_usage_changed(cpu_usage_, usage);
		cpu_usage_         = usage;
		has_cpu_usage_     = true;
		return changed;
	}

  private:
	static bool gpu_usage_changed(const FfxApiEffectMemoryUsage &lhs, const FfxApiEffectMemoryUsage &rhs)
	{
		return lhs.totalUsageInBytes != rhs.totalUsageInBytes ||
		       lhs.textureUsageInBytes != rhs.textureUsageInBytes ||
		       lhs.bufferUsageInBytes != rhs.bufferUsageInBytes ||
		       lhs.tensorUsageInBytes != rhs.tensorUsageInBytes ||
		       lhs.dataGraphUsageInBytes != rhs.dataGraphUsageInBytes ||
		       lhs.aliasableUsageInBytes != rhs.aliasableUsageInBytes;
	}

	static bool cpu_usage_changed(const FfxApiEffectCpuMemoryUsage &lhs, const FfxApiEffectCpuMemoryUsage &rhs)
	{
		return lhs.currentUsageInBytes != rhs.currentUsageInBytes ||
		       lhs.peakUsageInBytes != rhs.peakUsageInBytes;
	}

	FfxApiEffectMemoryUsage    gpu_usage_{};
	FfxApiEffectCpuMemoryUsage cpu_usage_{};
	bool                       has_gpu_usage_{false};
	bool                       has_cpu_usage_{false};
};
}        // namespace vkb
