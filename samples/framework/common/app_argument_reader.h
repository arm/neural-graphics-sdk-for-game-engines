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

#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>

#include "core/util/logging.hpp"

namespace vkb
{
/**
 * @brief Typed accessors for application arguments.
 *
 * Missing keys return the fallback value. Invalid values log a warning and
 * also return the fallback.
 */
class AppArgumentReader
{
  public:
	explicit AppArgumentReader(const std::unordered_map<std::string, std::string> &args) :
	    args_{args}
	{
	}

	/// Returns true if @p key was supplied on the command line.
	bool has(const char *key) const
	{
		return args_.find(key) != args_.end();
	}

	/// Returns the raw string value, or @p fallback when the key is absent.
	std::string get_string(const char *key, const std::string &fallback = {}) const
	{
		auto it = args_.find(key);
		return (it != args_.end()) ? it->second : fallback;
	}

	/// Parses a boolean from a strict "0" or "1" value.
	bool get_bool(const char *key, bool fallback) const
	{
		auto it = args_.find(key);
		if (it == args_.end())
		{
			return fallback;
		}
		if (it->second == "0")
		{
			return false;
		}
		if (it->second == "1")
		{
			return true;
		}
		LOGW("Invalid {}: '{}', expected 0 or 1", key, it->second);
		return fallback;
	}

	/// Parses a signed integer constrained to [@p min_val, @p max_val].
	int get_int(const char *key, int fallback, int min_val, int max_val) const
	{
		auto it = args_.find(key);
		if (it == args_.end())
		{
			return fallback;
		}
		try
		{
			size_t pos = 0;
			int    v   = std::stoi(it->second, &pos, 10);
			if (pos == it->second.size() && v >= min_val && v <= max_val)
			{
				return v;
			}
		}
		catch (...)
		{
		}
		LOGW("Invalid {}: '{}', expected integer in [{}, {}]", key, it->second, min_val, max_val);
		return fallback;
	}

	/// Parses a float constrained to [@p min_val, @p max_val].
	float get_float(const char *key, float fallback, float min_val, float max_val) const
	{
		auto it = args_.find(key);
		if (it == args_.end())
		{
			return fallback;
		}
		try
		{
			size_t pos = 0;
			float  v   = std::stof(it->second, &pos);
			if (pos == it->second.size() && v >= min_val && v <= max_val)
			{
				return v;
			}
		}
		catch (...)
		{
		}
		LOGW("Invalid {}: '{}', expected float in [{}, {}]", key, it->second, min_val, max_val);
		return fallback;
	}

	/// Parses an unsigned integer, accepting decimal or 0x-prefixed hex.
	uint32_t get_uint(const char *key, uint32_t fallback,
	                  uint32_t min_val = 0, uint32_t max_val = UINT32_MAX) const
	{
		auto it = args_.find(key);
		if (it == args_.end())
		{
			return fallback;
		}
		try
		{
			size_t        pos = 0;
			unsigned long v   = std::stoul(it->second, &pos, 0);
			if (pos == it->second.size() && v >= min_val && v <= max_val)
			{
				return static_cast<uint32_t>(v);
			}
		}
		catch (...)
		{
		}
		LOGW("Invalid {}: '{}', expected unsigned integer in [{}, {}]", key, it->second, min_val, max_val);
		return fallback;
	}

  private:
	const std::unordered_map<std::string, std::string> &args_;
};

}        // namespace vkb