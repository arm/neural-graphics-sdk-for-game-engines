#include "ffx_nss_jitter_analysis.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace
{
    constexpr float kBounds  = 0.500001f;
    constexpr float kEpsilon = 1e-8f;

    struct QuantizedSample
    {
        int64_t x;
        int64_t y;

        bool operator==(const QuantizedSample& other) const
        {
            return x == other.x && y == other.y;
        }
    };

    int64_t Quantize(float value)
    {
        return static_cast<int64_t>(std::llround(static_cast<double>(value) * 100000000.0));
    }

    bool Contains(const std::vector<QuantizedSample>& values, const QuantizedSample& value)
    {
        return std::find(values.begin(), values.end(), value) != values.end();
    }

    float Distance(const NssJitterSample& first, const NssJitterSample& second)
    {
        const float dx = first.x - second.x;
        const float dy = first.y - second.y;
        return std::sqrt(dx * dx + dy * dy);
    }

    float ApproximateStarDiscrepancy(const std::vector<NssJitterSample>& points)
    {
        float discrepancy = 0.0f;
        for (uint32_t yIndex = 0; yIndex <= 32; ++yIndex)
        {
            const float y = static_cast<float>(yIndex) / 32.0f;
            for (uint32_t xIndex = 0; xIndex <= 32; ++xIndex)
            {
                const float x     = static_cast<float>(xIndex) / 32.0f;
                uint32_t    count = 0;
                for (const NssJitterSample& point : points)
                {
                    const float unitX = std::clamp(point.x + 0.5f, 0.0f, 1.0f);
                    const float unitY = std::clamp(point.y + 0.5f, 0.0f, 1.0f);
                    count += unitX <= x && unitY <= y;
                }
                discrepancy = std::max(discrepancy, std::fabs(static_cast<float>(count) / static_cast<float>(points.size()) - x * y));
            }
        }
        return discrepancy;
    }

    float NearestNeighborCv(const std::vector<NssJitterSample>& points)
    {
        if (points.size() < 3)
            return std::numeric_limits<float>::infinity();

        std::vector<float> nearest(points.size(), std::numeric_limits<float>::infinity());
        for (size_t first = 0; first < points.size(); ++first)
        {
            for (size_t second = 0; second < points.size(); ++second)
            {
                if (first != second)
                    nearest[first] = std::min(nearest[first], Distance(points[first], points[second]));
            }
        }

        float mean = 0.0f;
        for (float value : nearest)
            mean += value;
        mean /= static_cast<float>(nearest.size());
        if (mean <= 0.0f)
            return std::numeric_limits<float>::infinity();

        float variance = 0.0f;
        for (float value : nearest)
            variance += (value - mean) * (value - mean);
        variance /= static_cast<float>(nearest.size());
        return std::sqrt(variance) / mean;
    }
}  // namespace

NssJitterAnalysis AnalyzeNssJitter(const NssJitterSample* samples, uint32_t sampleCount, uint32_t expectedUniqueCount)
{
    NssJitterAnalysis result{};
    result.sampleCount         = sampleCount;
    result.expectedUniqueCount = expectedUniqueCount;
    result.finite              = true;
    result.inBounds            = true;
    result.centered            = false;
    result.wideSpan            = false;
    result.quadrants           = false;
    result.grid                = false;
    result.discrepancy         = false;
    result.spacing             = false;
    result.correlation         = false;
    result.enoughUnique        = false;

    std::vector<NssJitterSample> uniquePoints;
    std::vector<QuantizedSample> quantizedPoints;
    uint32_t                     inBoundsCount = 0;
    if (samples != nullptr)
    {
        for (uint32_t index = 0; index < sampleCount; ++index)
        {
            const NssJitterSample sample       = samples[index];
            const bool            sampleFinite = std::isfinite(sample.x) && std::isfinite(sample.y);
            result.finite &= sampleFinite;
            if (sampleFinite && std::fabs(sample.x) <= kBounds && std::fabs(sample.y) <= kBounds)
                ++inBoundsCount;

            if (sampleFinite)
            {
                const QuantizedSample quantized{Quantize(sample.x), Quantize(sample.y)};
                if (!Contains(quantizedPoints, quantized))
                {
                    quantizedPoints.push_back(quantized);
                    uniquePoints.push_back(sample);
                }
            }
        }
    }

    result.uniqueCount      = static_cast<uint32_t>(uniquePoints.size());
    result.inBoundsFraction = sampleCount == 0 ? 0.0f : static_cast<float>(inBoundsCount) / static_cast<float>(sampleCount);
    result.inBounds         = result.inBoundsFraction >= 0.95f;
    if (!uniquePoints.empty())
    {
        float minX = uniquePoints[0].x;
        float maxX = uniquePoints[0].x;
        float minY = uniquePoints[0].y;
        float maxY = uniquePoints[0].y;
        for (const NssJitterSample& point : uniquePoints)
        {
            result.centroidX += point.x;
            result.centroidY += point.y;
            minX = std::min(minX, point.x);
            maxX = std::max(maxX, point.x);
            minY = std::min(minY, point.y);
            maxY = std::max(maxY, point.y);
        }
        result.centroidX /= static_cast<float>(uniquePoints.size());
        result.centroidY /= static_cast<float>(uniquePoints.size());
        result.spanX    = maxX - minX;
        result.spanY    = maxY - minY;
        result.centered = std::sqrt(result.centroidX * result.centroidX + result.centroidY * result.centroidY) <= 0.10f;
        result.wideSpan = result.spanX >= 0.70f && result.spanY >= 0.70f;
    }

    bool quadrant[4] = {};
    for (const NssJitterSample& point : uniquePoints)
    {
        if (std::fabs(point.x) > kEpsilon && std::fabs(point.y) > kEpsilon)
        {
            const uint32_t quadrantIndex = (point.x > 0.0f ? 1u : 0u) | (point.y > 0.0f ? 2u : 0u);
            quadrant[quadrantIndex]      = true;
        }
    }
    for (bool covered : quadrant)
        result.quadrantCount += covered;
    result.quadrants = result.quadrantCount == 4;

    const uint32_t gridSize     = std::max(2u, std::min(8u, static_cast<uint32_t>(std::ceil(std::sqrt(static_cast<float>(uniquePoints.size()))))));
    bool           occupied[64] = {};
    for (const NssJitterSample& point : uniquePoints)
    {
        const float    boundedX            = std::clamp(point.x + 0.5f, 0.0f, 1.0f - kEpsilon);
        const float    boundedY            = std::clamp(point.y + 0.5f, 0.0f, 1.0f - kEpsilon);
        const uint32_t cellX               = std::min(gridSize - 1, static_cast<uint32_t>(boundedX * gridSize));
        const uint32_t cellY               = std::min(gridSize - 1, static_cast<uint32_t>(boundedY * gridSize));
        occupied[cellY * gridSize + cellX] = true;
    }
    uint32_t occupiedCount = 0;
    for (uint32_t index = 0; index < gridSize * gridSize; ++index)
        occupiedCount += occupied[index];
    result.gridOccupancy   = static_cast<float>(occupiedCount) / static_cast<float>(gridSize * gridSize);
    result.grid            = result.gridOccupancy >= 0.45f;
    result.starDiscrepancy = uniquePoints.empty() ? std::numeric_limits<float>::infinity() : ApproximateStarDiscrepancy(uniquePoints);
    result.discrepancy     = result.starDiscrepancy <= 0.30f;
    result.spacingCv       = NearestNeighborCv(uniquePoints);
    result.spacing         = result.spacingCv <= 0.75f;

    if (uniquePoints.size() >= 3)
    {
        float meanX = 0.0f;
        float meanY = 0.0f;
        for (const NssJitterSample& point : uniquePoints)
        {
            meanX += point.x;
            meanY += point.y;
        }
        meanX /= static_cast<float>(uniquePoints.size());
        meanY /= static_cast<float>(uniquePoints.size());
        float covariance = 0.0f;
        float varianceX  = 0.0f;
        float varianceY  = 0.0f;
        for (const NssJitterSample& point : uniquePoints)
        {
            const float dx = point.x - meanX;
            const float dy = point.y - meanY;
            covariance += dx * dy;
            varianceX += dx * dx;
            varianceY += dy * dy;
        }
        result.xyCorrelation = varianceX > 0.0f && varianceY > 0.0f ? std::fabs(covariance / std::sqrt(varianceX * varianceY)) : 1.0f;
    }
    else
    {
        result.xyCorrelation = 1.0f;
    }
    result.correlation  = result.xyCorrelation <= 0.75f;
    result.uniqueness   = static_cast<float>(result.uniqueCount) / static_cast<float>(std::max(1u, expectedUniqueCount));
    result.enoughUnique = result.uniqueness >= 0.50f;

    const bool checks[] = {result.finite,
                           result.inBounds,
                           result.centered,
                           result.wideSpan,
                           result.quadrants,
                           result.grid,
                           result.discrepancy,
                           result.spacing,
                           result.correlation,
                           result.enoughUnique};
    for (bool passed : checks)
    {
        result.score += passed;
        result.failedChecks += !passed;
    }
    return result;
}