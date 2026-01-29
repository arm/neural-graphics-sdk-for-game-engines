/* Copyright (c) 2025-2026, Arm Limited and Contributors
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

#pragma once

#include "platform/plugins/plugin_base.h"

namespace plugins
{
class AppArgs;

using AppArgsTags = vkb::PluginBase<AppArgs, vkb::tags::Passive>;

/**
 * @brief App Arguments
 *
 * Allows forwarding arbitrary key/value pairs to the selected application/sample.
 * This keeps the platform CLI stable and avoids adding per-sample CLI flags.
 *
 * Usage:
 *   vulkan_samples sample nss --app-arg NSS_ENABLE=0 --app-arg NSS_SCALE_FACTOR=2.0
 *   vulkan_samples sample nss --app-arg NSS_ENABLE=0,NSS_SCALE_FACTOR=2.0
 */
class AppArgs : public AppArgsTags
{
  public:
	AppArgs();
	virtual ~AppArgs() = default;

	bool handle_option(std::deque<std::string> &arguments) override;
};
}        // namespace plugins
