#pragma once

#include <cstdint>

struct NssJitterSample
{
    float x;
    float y;
};

struct NssJitterAnalysis
{
    uint32_t sampleCount;
    uint32_t uniqueCount;
    uint32_t expectedUniqueCount;
    uint32_t score;
    uint32_t failedChecks;
    uint32_t quadrantCount;
    float    centroidX;
    float    centroidY;
    float    spanX;
    float    spanY;
    float    inBoundsFraction;
    float    gridOccupancy;
    float    starDiscrepancy;
    float    spacingCv;
    float    xyCorrelation;
    float    uniqueness;
    bool     finite;
    bool     inBounds;
    bool     centered;
    bool     wideSpan;
    bool     quadrants;
    bool     grid;
    bool     discrepancy;
    bool     spacing;
    bool     correlation;
    bool     enoughUnique;
};

NssJitterAnalysis AnalyzeNssJitter(const NssJitterSample* samples, uint32_t sampleCount, uint32_t expectedUniqueCount);