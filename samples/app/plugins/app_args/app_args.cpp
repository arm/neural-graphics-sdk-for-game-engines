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

#include "app_args.h"

#include "core/util/logging.hpp"

#include <cctype>
#include <vector>

namespace plugins
{
namespace
{
std::string trim_copy(const std::string &input)
{
	size_t start = 0;
	while (start < input.size() && std::isspace(static_cast<unsigned char>(input[start])))
	{
		start++;
	}

	size_t end = input.size();
	while (end > start && std::isspace(static_cast<unsigned char>(input[end - 1])))
	{
		end--;
	}

	return input.substr(start, end - start);
}

bool parse_key_value_segment(const std::string &segment, std::string &key, std::string &value)
{
	// Split on the first '='.
	size_t pos = segment.find('=');
	if (pos == std::string::npos || pos == 0)
	{
		return false;
	}

	key   = trim_copy(segment.substr(0, pos));
	value = trim_copy(segment.substr(pos + 1));
	return !key.empty();
}

std::vector<std::string> split_pairs(const std::string &input)
{
	// Split on ',' or ';'.
	std::vector<std::string> parts;
	std::string              current;
	current.reserve(input.size());
	for (char c : input)
	{
		if (c == ',' || c == ';')
		{
			parts.push_back(current);
			current.clear();
			continue;
		}
		current.push_back(c);
	}

	parts.push_back(current);
	return parts;
}
}        // namespace

AppArgs::AppArgs() :
    AppArgsTags("App Arguments",
                "Forward arbitrary KEY=VALUE arguments to the selected app/sample.",
                {},
                {},
                {{"app-arg", "Forward app arguments as KEY=VALUE. Repeatable. Also supports multiple pairs in one value: KEY1=VALUE1,KEY2=VALUE2."}})
{
}

bool AppArgs::handle_option(std::deque<std::string> &arguments)
{
	assert(!arguments.empty() && (arguments[0].substr(0, 2) == "--"));

	std::string option = arguments[0].substr(2);
	if (option != "app-arg")
	{
		return false;
	}

	if (arguments.size() < 2)
	{
		LOGE("Option \"--app-arg\" expects a value of the form KEY=VALUE");
		return false;
	}

	std::string key;
	std::string value;

	bool any = false;
	for (const auto &raw_part : split_pairs(arguments[1]))
	{
		auto part = trim_copy(raw_part);
		if (part.empty())
		{
			continue;
		}

		if (!parse_key_value_segment(part, key, value))
		{
			LOGE("Invalid --app-arg value segment: \"{}\" (expected KEY=VALUE). Full value: \"{}\"", part, arguments[1]);
			return false;
		}

		platform->set_app_argument(key, value);
		any = true;
	}

	if (!any)
	{
		LOGE("Invalid --app-arg value: \"{}\" (expected KEY=VALUE)", arguments[1]);
		return false;
	}

	arguments.pop_front();
	arguments.pop_front();
	return true;
}
}        // namespace plugins
