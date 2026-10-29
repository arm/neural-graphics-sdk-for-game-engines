/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstdint>
#include <vector>

namespace arm::capture::detail
{
    uint64_t AlignUp(uint64_t value, uint64_t alignment);
    uint64_t LeastCommonMultiple(uint64_t first, uint64_t second);
    uint64_t StagingAllocationUnitSize(uint64_t offsetAlignment);
    uint64_t SlotBytesForAttachments(const std::vector<uint64_t>& attachmentBytes,
                                     const std::vector<uint64_t>& attachmentAlignments,
                                     uint64_t                     offsetAlignment,
                                     uint64_t                     unitSize);

}  // namespace arm::capture::detail
