/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <capture/config.h>
#include <capture/execution.h>
#include "../../sdk/src/capture/log.h"
#include <ffx_api/ffx_api.h>
#include <ffx_provider.h>

#include <chrono>
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cmath>
#include <memory>
#include <limits>
#include <optional>
#include <string>
#include <new>
#include <utility>
#include <vector>

#define TRY2_CAPTURE(_expr)                                                 \
    do                                                                      \
    {                                                                       \
        FfxErrorCode _rc = (_expr);                                         \
        if (FFX_OK != _rc)                                                  \
        {                                                                   \
            ffxLogError("API provider", _rc, #_expr);                       \
            if (captureActive)                                              \
                internal_context->controller->AbortDispatchCapture(#_expr); \
            return FFX_API_RETURN_ERROR_RUNTIME_ERROR;                      \
        }                                                                   \
    } while (false)

// Provider failure paths discard the command list, so capture cancellation
// must release recorded staging/query state without recording a completion
// query into a command buffer that will never be submitted.

#if FFX_CAPTURE_ENABLED
#define ARM_LOG_DISPATCH_TIMESTAMP_IF_ENABLED(_component, _captureActive)                                                                               \
    do                                                                                                                                                  \
    {                                                                                                                                                   \
        if ((_captureActive) && ::arm::capture::IsCapturePerformanceEnabled())                                                                          \
        {                                                                                                                                               \
            const auto _timestamp = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count(); \
            ARM_CAPTURE_LOG_INFO(_component, "capture=1 dispatch_timestamp_us=%llu", static_cast<unsigned long long>(_timestamp));                      \
        }                                                                                                                                               \
    } while (false)
#else
#define ARM_LOG_DISPATCH_TIMESTAMP_IF_ENABLED(_component, _captureActive) \
    do                                                                    \
    {                                                                     \
        (void)(_component);                                               \
        (void)(_captureActive);                                           \
    } while (false)
#endif

namespace arm::capture
{
    template <typename ControllerT, typename... Args>
    ControllerT* MakeCaptureController(Allocator& allocator, Args&&... args)
    {
        void* storage = allocator.alloc(sizeof(ControllerT));
        if (!storage)
            return nullptr;

        return ::new (storage) ControllerT(std::forward<Args>(args)...);
    }

    template <typename ControllerT>
    void DestroyCaptureController(Allocator& allocator, ControllerT*& controller)
    {
        if (!controller)
            return;

        controller->~ControllerT();
        allocator.dealloc(controller, sizeof(ControllerT));
        controller = nullptr;
    }

    template <typename Derived, typename FrameT, typename MetadataWriterT>
    class RecorderBase : public FrameMetadataSink
    {
    public:
        explicit RecorderBase(uint32_t flags)
            : contextFlags(flags)
        {
        }

        bool CompleteFrame(std::unique_ptr<FrameMeta> completed) final
        {
            // Completion order is intentionally not part of the dataset contract.
            // Each frame carries its own dataset index and metadata; do not add a
            // reorder buffer merely to impose dispatch order on asynchronous results.
            std::unique_ptr<FrameT> frame(static_cast<FrameT*>(completed.release()));
            Derived&                derived = static_cast<Derived&>(*this);
            if (!metadata.IsOpen() && !derived.OpenWriters(*frame))
                return false;
            if (!derived.WriteMetadata(*frame))
                return false;
            return true;
        }

        bool FinalizeMetadata() final
        {
            return static_cast<Derived&>(*this).CloseWriters();
        }

        void Reset()
        {
            metadata.Reset();
            committedFrames_ = 0;
            static_cast<Derived&>(*this).ResetExtra();
        }

        uint32_t GetCommittedFrameCount() const
        {
            return committedFrames_;
        }

        void CommitFrame()
        {
            ++committedFrames_;
            static_cast<Derived&>(*this).OnFrameCommitted();
        }

        uint32_t contextFlags = 0;
        Config   config{};

    protected:
        MetadataWriterT metadata;

    private:
        uint32_t committedFrames_ = 0;
    };

    class ControllerBase
    {
    protected:
        struct ConfigInput
        {
            ConfigInput(const char* prefix, const char* component)
                : environment_(prefix, component)
                , component_(component ? component : "CAPTURE")
            {
                if (std::optional<Config> startupConfig = environment_.ReadConfig())
                {
                    if (QueueEnvironmentConfig(std::move(startupConfig)))
                        EnableEnvironmentPolling(0);
                }
            }

            bool IsPollingEnabled() const
            {
                return lastEnvironmentPollFrame_.has_value();
            }

            bool HasPendingConfig(uint32_t frameIndex = 0)
            {
                if (useApi_)
                    return HasPendingConfigWithoutPolling();

                // Runtime polling begins once the adapter observes a valid startup
                // configuration during controller construction.
                if (!lastEnvironmentPollFrame_)
                    return HasPendingConfigWithoutPolling();

                constexpr uint32_t kEnvironmentPollInterval = 30;
                if (lastEnvironmentPollFrame_ && frameIndex - *lastEnvironmentPollFrame_ < kEnvironmentPollInterval)
                    return HasPendingConfigWithoutPolling();
                lastEnvironmentPollFrame_ = frameIndex;
                QueueEnvironmentConfig(environment_.ReadConfig());
                return HasPendingConfigWithoutPolling();
            }

            bool HasPendingConfigWithoutPolling() const
            {
                return !configs_.empty();
            }

            void EnableEnvironmentPolling(uint32_t frameIndex)
            {
                lastEnvironmentPollFrame_ = frameIndex;
            }

            void UseApi()
            {
                if (!useApi_)
                {
                    // API cancellation permanently selects the API input mode. An unqualified
                    // CANCEL cannot be associated with a particular environment revision, and
                    // reverting to polling could accidentally consume a request that the caller
                    // intended to cancel.
                    useApi_ = true;
                    configs_.clear();
                }
            }

            bool HasPendingApiConfig() const
            {
                return useApi_ && !configs_.empty();
            }

            bool UsesApi() const
            {
                return useApi_;
            }

            bool PushApiConfig(const Config& config)
            {
                if (!IsConfigValid(config))
                    return false;

                Config apiConfig = config;
                if (apiConfig.component.empty())
                    apiConfig.component = component_;

                if (useApi_)
                {
                    configs_.push_back(std::move(apiConfig));
                }
                else
                {
                    std::vector<Config> replacement;
                    replacement.push_back(std::move(apiConfig));
                    configs_.swap(replacement);
                    useApi_ = true;
                }
                return true;
            }

            bool AdvanceConfig(uint32_t frameIndex)
            {
                if (configs_.empty() || frameIndex < configs_.front().startFrame)
                    return false;
                active_ = std::move(configs_.front());
                configs_.erase(configs_.begin());
                return true;
            }

            Config& GetActiveConfig()
            {
                return active_;
            }

            const Config& GetActiveConfig() const
            {
                return active_;
            }

        private:
            bool QueueEnvironmentConfig(std::optional<Config> config)
            {
                if (!config)
                    return false;
                if (!IsConfigValid(*config))
                {
                    ARM_CAPTURE_LOG_WARNING(component_.c_str(), "rejected capture environment revision: invalid configuration");
                    return false;
                }

                // ReadConfig filters already-observed revisions. Preserve each new
                // revision so distinct environment requests remain independent captures,
                // but bound the idle dispatch-path queue.
                constexpr size_t kMaxPendingEnvironmentConfigs = 16;
                if (configs_.size() == kMaxPendingEnvironmentConfigs)
                {
                    ARM_CAPTURE_LOG_WARNING(component_.c_str(), "rejected capture environment revision: pending request limit reached");
                    return false;
                }
                if (configs_.capacity() < kMaxPendingEnvironmentConfigs)
                    configs_.reserve(kMaxPendingEnvironmentConfigs);
                configs_.push_back(std::move(*config));
                return true;
            }

            bool IsConfigValid(const Config& config) const
            {
                const long double bias                = static_cast<long double>(config.motionVectorUnormBias);
                const long double scale               = static_cast<long double>(config.motionVectorUnormScale);
                const long double maxInputMagnitude   = std::max(std::abs(bias), std::abs(1.0L - bias));
                const bool        scaledInputIsFinite = maxInputMagnitude * std::abs(scale) <= static_cast<long double>(std::numeric_limits<float>::max());
                return !config.outputDirectory.empty() && config.outputDirectory.size() <= kMaxCaptureOutputDirectoryLength && config.maxFrameCount != 0 &&
                       std::isfinite(config.motionVectorUnormBias) && std::isfinite(config.motionVectorUnormScale) && config.motionVectorUnormScale != 0.0f &&
                       scaledInputIsFinite;
            }

            EnvironmentConfigReader environment_;
            std::string             component_;
            Config                  active_;
            std::vector<Config>     configs_;
            std::optional<uint32_t> lastEnvironmentPollFrame_;
            bool                    useApi_ = false;
        };

    public:
        ControllerBase(const char* prefix, const char* component);
        virtual ~ControllerBase() = default;

        ControllerBase(const ControllerBase&) = delete;
        ControllerBase& operator=(const ControllerBase&) = delete;

        ffxReturnCode_t Configure(const ffxApiConfigureDescCapture& desc);
        void            GetStatus(FfxApiCaptureStatus& status, uint32_t capturedFrames);
        // A timed-out capture is retained and this call returns false because its
        // Vulkan or file operation may still be in progress. The application
        // must keep the VkDevice and backend code valid and retry destruction
        // after submitted command buffers have completed.
        bool Shutdown();

    protected:
        virtual bool OnCaptureBegin(const Config& config) = 0;
        virtual void OnCaptureStop()
        {
        }

        bool StartCapture(const Config& config);
        bool RequestCaptureDrain(CaptureResult result);
        void FailCapture(const char* reason);
        bool AdvanceFrameThenCheck()
        {
            return AdvanceFrameThenCheck(dispatchFrameIndex_++);
        }

    protected:
        bool AdvanceFrameThenCheck(uint32_t frameIndex)
        {
            if (!armed_)
                return false;

            TryFinalizeDrain();
            const ExecutionPhase state = execution_.GetState();
            // Environment/property polling is enabled only when startup configuration
            // established that the adapter is in use.
            if (state == ExecutionPhase::Idle && configInput_.HasPendingConfig(frameIndex))
            {
                if (configInput_.AdvanceConfig(frameIndex))
                {
                    const Config& config = configInput_.GetActiveConfig();
                    // AdvanceConfig only succeeds after startFrame. Reject a queued
                    // configuration whose capture window has already ended.
                    if (frameIndex - config.startFrame >= config.maxFrameCount)
                    {
                        execution_.MarkFailed();
                        ARM_CAPTURE_LOG_WARNING(config.component.c_str(), "queued capture interval expired before capture started");
                    }
                    else if (!StartCapture(config))
                        ARM_CAPTURE_LOG_WARNING(configInput_.GetActiveConfig().component.c_str(), "failed to start queued capture configuration");
                }
            }
            if (execution_.GetState() != ExecutionPhase::Capturing)
                return false;

            const Config& activeConfig = configInput_.GetActiveConfig();
            if (frameIndex - activeConfig.startFrame >= activeConfig.maxFrameCount)
            {
                FailCapture("capture dispatch interval expired");
                return false;
            }
            return true;
        }

    private:
        bool TryFinalizeDrain();

    protected:
        ConfigInput configInput_;
        Execution   execution_;
        uint32_t    dispatchFrameIndex_ = 0;
        bool        armed_              = false;
    };

}  // namespace arm::capture
