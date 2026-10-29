/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT
 */

#include "utils.h"

#include <cstddef>
#include <limits>
#include <numeric>

namespace arm::capture::detail
{
    namespace
    {
        constexpr uint64_t kStagingAllocationUnitBytes = 1024ull * 1024;
    }

    uint64_t AlignUp(uint64_t value, uint64_t alignment)
    {
        if (alignment <= 1)
            return value;
        const uint64_t remainder = value % alignment;
        if (remainder == 0)
            return value;
        const uint64_t padding = alignment - remainder;
        return value <= std::numeric_limits<uint64_t>::max() - padding ? value + padding : 0;
    }

    uint64_t LeastCommonMultiple(uint64_t first, uint64_t second)
    {
        if (first == 0 || second == 0)
            return 0;
        const uint64_t divisor = std::gcd(first, second);
        const uint64_t scaled  = first / divisor;
        return scaled <= std::numeric_limits<uint64_t>::max() / second ? scaled * second : 0;
    }

    uint64_t StagingAllocationUnitSize(uint64_t offsetAlignment)
    {
        if (offsetAlignment > 1 && (offsetAlignment & (offsetAlignment - 1)) != 0)
            return 0;
        return AlignUp(kStagingAllocationUnitBytes, offsetAlignment);
    }

    uint64_t SlotBytesForAttachments(const std::vector<uint64_t>& attachmentBytes,
                                     const std::vector<uint64_t>& attachmentAlignments,
                                     uint64_t                     offsetAlignment,
                                     uint64_t                     unitSize)
    {
        if (attachmentBytes.empty() || attachmentBytes.size() != attachmentAlignments.size() || offsetAlignment == 0 || unitSize == 0)
            return 0;

        uint64_t end = 0;
        for (std::size_t index = 0; index < attachmentBytes.size(); ++index)
        {
            const uint64_t bytes     = attachmentBytes[index];
            const uint64_t alignment = LeastCommonMultiple(offsetAlignment, attachmentAlignments[index]);
            if (alignment == 0)
                return 0;
            if (bytes == 0)
                return 0;
            const uint64_t offset = AlignUp(end, alignment);
            if ((offset == 0 && end != 0) || bytes > std::numeric_limits<uint64_t>::max() - offset)
                return 0;
            end = offset + bytes;
        }
        return AlignUp(end, unitSize);
    }

}  // namespace arm::capture::detail
