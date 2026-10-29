// This file is part of the FidelityFX SDK.
//
// Copyright (C) 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//
/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <atomic>
#include <cstdint>

namespace ffx
{
#ifdef FFX_ENABLE_CPU_HEAP_TRACKING
    /// Counts requested heap bytes allocated through one effect context's
    /// tracked Allocator. The context allocation bootstraps the initial usage.
    class EffectHeapTracker
    {
    public:
        explicit EffectHeapTracker(uint64_t initialUsageInBytes = 0) noexcept
            : currentUsageInBytes_(initialUsageInBytes)
            , peakUsageInBytes_(initialUsageInBytes)
        {
        }

        void Add(uint64_t sizeInBytes) noexcept
        {
            const uint64_t current = currentUsageInBytes_.fetch_add(sizeInBytes, std::memory_order_relaxed) + sizeInBytes;
            uint64_t       peak    = peakUsageInBytes_.load(std::memory_order_relaxed);
            while (peak < current && !peakUsageInBytes_.compare_exchange_weak(peak, current, std::memory_order_relaxed, std::memory_order_relaxed))
            {
            }
        }

        void Remove(uint64_t sizeInBytes) noexcept
        {
            currentUsageInBytes_.fetch_sub(sizeInBytes, std::memory_order_relaxed);
        }

        uint64_t GetCurrentUsageInBytes() const noexcept
        {
            return currentUsageInBytes_.load(std::memory_order_relaxed);
        }

        uint64_t GetPeakUsageInBytes() const noexcept
        {
            return peakUsageInBytes_.load(std::memory_order_relaxed);
        }

    private:
        std::atomic<uint64_t> currentUsageInBytes_;
        std::atomic<uint64_t> peakUsageInBytes_;
    };
#endif
}  // namespace ffx
