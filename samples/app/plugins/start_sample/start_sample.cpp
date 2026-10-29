/* Copyright (c) 2020-2026, Arm Limited and Contributors
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright 2020-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT
 */

#include "start_sample.h"

#include <map>
#include <string>
#include <vector>

#include "apps.h"

namespace plugins
{
StartSample::StartSample() :
    StartSampleTags("StartSample",
                    "A collection of flags to samples and apps.",
                    {},
                    {{"sample", "Run a specific sample"},
                     {"samples", "List available samples with descriptions"},
                     {"samples-oneline", "List available samples, one per line"}})
{
}

void StartSample::launch_sample(apps::SampleInfo const *sample) const
{
	vkb::Window::OptionalProperties properties;
	properties.title = "Vulkan Samples: " + sample->name;
	platform->set_window_properties(properties);
	platform->request_application(sample);
}

void StartSample::list_samples(bool one_per_line) const
{
	// Per-sample --app-arg documentation
	static const std::map<std::string, std::vector<std::string>> app_arg_docs = {
	    {"nss", {
	                "NSS_ENABLE=0|1          Enable/disable NSS upscaling (default: 1)",
	                "NSS_SCALE_FACTOR=float  Upscale factor [1.0, 3.0] (default: 2.0)",
	                "NSS_FLAGS=0xHEX         Raw flags override; pair PRE/POST compute flags in this sample",
	                "NSS_QUALITY=0|1|2       NSS quality mode: 0=Quality, 1=Balanced, 2=Performance (default: 1)",
	                "NSS_USE_COMPUTE=0|1     Pre/postprocess path: 0=fragment, 1=compute (default: 0)",
	                "NSS_LDR_MODE=0|1        [experimental] Use LDR preprocessing path for NSS input (default: 0)",
	                "NSS_COLOR_FORMAT=R11G11B10|RGBA16F|RGBA32F Input/output color format (default: R11G11B10)",
	                "NSS_DEBUG_VIEW_MODE=0..16  Debug view mode: 0=overview, 1..16=single tile (default: 0)",
	                "NSS_MANAGE_HISTORY=0|1  History: 0=application, 1=SDK managed (default: 0)",
	                "NSS_DATASET_DIR=path    Path to EXR dataset directory (enables dataset mode)",
	                "NSS_DATASET_SEQUENCE=N Dataset sequence ID (default: 0000)",
	                "NSS_SAVE_EXR=0|1        Save dataset NSS output to EXR files (default: 0)",
	                "NSS_OUTPUT_DIR=path     Output root, relative to dataset or absolute (default: result)",
	            }},
	    {"nfru", {
	                 "NFRU_DATASET_DIR=path   Path to EXR dataset directory (enables dataset mode)",
	                 "NFRU_DATASET_SEQUENCE=N Dataset sequence ID (default: 0000)",
	                 "NFRU_SAVE_EXR=0|1       Save generated frames to EXR files (default: 0)",
	                 "NFRU_ORBIT_SPEED=float  Camera rotation speed in degrees/frame (default: 0.5)",
	                 "NFRU_USE_FRAGMENT=0|1   Selectable-stage path: 0=compute, 1=fragment (default: 1)",
	                 "NFRU_ENABLE_MV_HINTS=0|1 Enable/disable optical flow MV hints (default: 0)",
	                 "NFRU_DEBUG_VIEW=0|1     Enable NFRU debug view at startup (default: 0)",
	                 "NFRU_DEBUG_CHECKING=0|1 Enable verbose per-frame NFRU runtime logs (default: 0)",
	                 "NFRU_MANAGE_HISTORY=0|1 History: 0=application, 1=SDK managed (default: 1)",
	             }},
	};

	auto samples = apps::get_samples();

	LOGI("");
	LOGI("Available Samples");
	LOGI("");

	for (auto *app : samples)
	{
		auto sample = reinterpret_cast<apps::SampleInfo *>(app);
		if (one_per_line)
		{
			LOGI("{}", sample->id.c_str());
		}
		else
		{
			LOGI("{}", sample->name.c_str());
			LOGI("\tid: {}", sample->id.c_str());
			LOGI("\tdescription: {}", sample->description.c_str());
			auto it = app_arg_docs.find(sample->id);
			if (it != app_arg_docs.end())
			{
				LOGI("\tapp args (--app-arg KEY=VALUE):");
				for (const auto &arg : it->second)
					LOGI("\t\t{}", arg.c_str());
			}
			LOGI("");
		}
	}

	platform->close();
}

bool StartSample::handle_command(std::deque<std::string> &arguments) const
{
	assert(!arguments.empty());
	if (arguments[0] == "sample")
	{
		if (arguments.size() < 2)
		{
			LOGE("Command \"sample\" is missing the actual sample_id to launch!");
			return false;
		}
		auto *sample = apps::get_sample(arguments[1]);
		if (!sample)
		{
			LOGE("Command \"sample\" is called with an unknown sample_id \"{}\"!", arguments[1]);
			return false;
		}
		launch_sample(sample);
		arguments.pop_front();
		arguments.pop_front();
		return true;
	}
	if ((arguments[0] == "samples") || (arguments[0] == "samples-oneline"))
	{
		list_samples(arguments[0] == "samples-oneline");
		arguments.pop_front();
		return true;
	}
	return false;
}
}        // namespace plugins