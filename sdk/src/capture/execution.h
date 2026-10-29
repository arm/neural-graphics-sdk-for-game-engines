/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <capture/log.h>
#include <capture/session.h>
#include <capture/transaction.h>

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>

namespace arm::capture
{
    enum class ExecutionPhase : uint8_t
    {
        Idle,
        Capturing,
        Draining,
    };

    enum class CaptureResult : uint8_t
    {
        None,
        Success,
        Failed,
        Cancelled,
    };

    inline void SetCaptureResult(std::atomic<CaptureResult>& result, CaptureResult requested)
    {
        CaptureResult current = result.load(std::memory_order_acquire);
        while (current != CaptureResult::Failed)
        {
            if (result.compare_exchange_weak(current, requested, std::memory_order_acq_rel, std::memory_order_acquire))
                return;
        }
    }

    class Execution
    {
    public:
        ~Execution() = default;

        bool StartCapture(const std::string& component)
        {
            if (component.empty() || transaction_.active || HasSession() || state_ == ExecutionPhase::Capturing || state_ == ExecutionPhase::Draining)
                return false;

            component_ = component;
            state_     = ExecutionPhase::Capturing;
            result_->store(CaptureResult::None, std::memory_order_release);
            return true;
        }

        bool RequestDrain(CaptureResult result = CaptureResult::Success)
        {
            if (result == CaptureResult::None || state_ != ExecutionPhase::Capturing)
                return false;

            if (transaction_.active && !CancelTransaction())
                return false;

            SetCaptureResult(*result_, result);
            state_ = ExecutionPhase::Draining;
            if (HasSession())
            {
                CloseSession(session_);
                ARM_CAPTURE_LOG_INFO(component_.c_str(), "capture draining");
            }
            return true;
        }

        bool TryFinalizeDrain(uint32_t timeoutMilliseconds = 0)
        {
            const bool asynchronousFailure = result_->load(std::memory_order_acquire) == CaptureResult::Failed;
            if (state_ != ExecutionPhase::Draining && !asynchronousFailure)
                return false;

            if (asynchronousFailure)
            {
                if (transaction_.active && !CancelTransaction())
                    return false;
                state_ = ExecutionPhase::Draining;
                if (HasSession())
                    CloseSession(session_);
            }

            if (HasSession())
            {
                if (!WaitForSession(session_, timeoutMilliseconds))
                    return false;
                ReleaseSession();
                ARM_CAPTURE_LOG_INFO(component_.c_str(), "capture complete");
            }
            state_ = ExecutionPhase::Idle;
            return true;
        }

        ExecutionPhase GetState() const
        {
            return state_;
        }

        CaptureResult GetResult() const
        {
            return result_->load(std::memory_order_acquire);
        }

        void ResetResult()
        {
            result_->store(CaptureResult::None, std::memory_order_release);
        }

        void MarkFailed()
        {
            SetCaptureResult(*result_, CaptureResult::Failed);
        }

        bool EnsureSession(SessionDescriptor create)
        {
            if (HasSession())
                return true;

            auto shutdownDetached   = std::make_shared<std::atomic_bool>(false);
            create.onFailure        = CreateSessionFailureNotification();
            create.shutdownDetached = shutdownDetached;
            session_                = CreateSession(create);
            if (session_ != InvalidSession)
            {
                shutdownDetached_ = std::move(shutdownDetached);
                return true;
            }

            session_ = InvalidSession;
            return false;
        }

        bool BeginTransaction(const SlotRequirements& requirements, FfxCommandList commandList, std::unique_ptr<FrameMeta> frameMeta)
        {
            return GetState() == ExecutionPhase::Capturing && HasSession() &&
                   arm::capture::BeginTransaction(session_, requirements, commandList, std::move(frameMeta), transaction_);
        }

        bool BeginTransaction(uint64_t requiredSlotBytes, FfxCommandList commandList, std::unique_ptr<FrameMeta> frameMeta)
        {
            return BeginTransaction(SlotRequirements{{requiredSlotBytes}, {4}}, commandList, std::move(frameMeta));
        }

        std::optional<uint32_t> GetTransactionDatasetIndex() const
        {
            if (!transaction_.active || !transaction_.frameMeta)
                return std::nullopt;
            return transaction_.frameMeta->datasetIndex;
        }

        bool AddAttachment(const FfxResource& source, std::unique_ptr<Payload> payload)
        {
            return GetState() == ExecutionPhase::Capturing && arm::capture::AddAttachment(transaction_, source, std::move(payload));
        }

        bool CommitTransaction()
        {
            return GetState() == ExecutionPhase::Capturing && arm::capture::CommitTransaction(transaction_);
        }
        bool CancelTransaction()
        {
            return arm::capture::CancelTransaction(transaction_);
        }
        bool DiscardTransaction()
        {
            return arm::capture::DiscardTransaction(transaction_);
        }

        bool DrainAndDestroyForShutdown(uint32_t timeoutMilliseconds)
        {
            // RequestCaptureDrain may already have finalized an empty capture. In that case the
            // session is gone and shutdown is intentionally a no-op; the terminal result must
            // remain unchanged rather than being interpreted as a shutdown timeout.
            if (!HasSession())
                return true;

            // Do not append a completion query during context destruction. The
            // caller owns command-buffer submission, and even recording the
            // query may be blocked inside the Vulkan implementation.
            if (transaction_.active)
            {
                ARM_CAPTURE_LOG_ERROR(component_.c_str(), "capture shutdown discarded an active transaction");
                return false;
            }

            if (state_ == ExecutionPhase::Capturing)
                RequestDrain(CaptureResult::Cancelled);

            if (!TryFinalizeDrain(timeoutMilliseconds))
            {
                if (shutdownDetached_)
                    shutdownDetached_->store(true, std::memory_order_release);
                if (result_->load(std::memory_order_acquire) != CaptureResult::Cancelled)
                    result_->store(CaptureResult::Failed, std::memory_order_release);
                ARM_CAPTURE_LOG_ERROR(component_.c_str(), "capture shutdown timed out; retaining incomplete capture resources for retry");
                return false;
            }
            ARM_CAPTURE_LOG_INFO(component_.c_str(), "capture complete");
            state_ = ExecutionPhase::Idle;
            return true;
        }

    private:
        bool HasSession() const
        {
            return session_ != InvalidSession;
        }

        std::function<void()> CreateSessionFailureNotification()
        {
            return [result = result_]() { result->store(CaptureResult::Failed, std::memory_order_release); };
        }

        void ReleaseSession()
        {
            DestroySession(session_);
            session_ = InvalidSession;
            shutdownDetached_.reset();
        }

        SessionHandle  session_ = InvalidSession;
        Transaction    transaction_{};
        ExecutionPhase state_ = ExecutionPhase::Idle;
        std::string    component_;
        using AtomicResult                        = std::atomic<CaptureResult>;
        std::shared_ptr<AtomicResult>     result_ = std::make_shared<AtomicResult>(CaptureResult::None);
        std::shared_ptr<std::atomic_bool> shutdownDetached_;
    };

}  // namespace arm::capture
