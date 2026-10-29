/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <capture/config.h>
#include <capture/base.h>
#include <capture/execution.h>
#include <ffx_api/ffx_nss.h>
#include <FidelityFX/host/ffx_nss.h>

#include <memory>

namespace arm::capture::neuralssuperresolution
{
    class Controller : public ControllerBase
    {
    public:
        Controller(void* backendDevice, uint32_t contextFlags);
        ~Controller();
        Controller(const Controller&) = delete;
        Controller& operator=(const Controller&) = delete;
        // Capture failures are intentionally contained by the provider and never
        // make an NSS dispatch fail.
        void GetStatus(FfxApiCaptureStatus& status);
        bool ShouldResetHistoryForDispatch() const;
        bool BeginDispatchCapture(const ffxApiDispatchDescNss& dispatch, FfxNssContext& context);
        bool CompleteDispatchCapture(const ffxApiDispatchDescNss& dispatch, FfxNssContext& context);
        void AbortDispatchCapture(const char* reason);

    private:
        struct Recorder;
        void Stop(const char* reason);
        bool OnCaptureBegin(const Config& config) override;
        // Keep the backend opaque at the ffx-api boundary. This is a borrowed,
        // backend-owned device token, not a VkDevice or VkDeviceContext owned by
        // the controller. Vulkan handle extraction belongs to the SDK/backend
        // layer, which owns the device context contract and keeps the token valid
        // until controller shutdown. The controller must not dereference the
        // token as a Vulkan handle or extend its lifetime; storing VkDevice/
        // VkPhysicalDevice here would couple this layer to Vulkan and duplicate
        // backend validation.
        void* const               backendDevice_;
        std::shared_ptr<Recorder> recorder_;
    };
}  // namespace arm::capture::neuralssuperresolution
