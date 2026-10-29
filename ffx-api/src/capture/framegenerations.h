/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <ffx_api/ffx_framegeneration.h>
#include <FidelityFX/host/ffx_frameinterpolation.h>
#include <FidelityFX/gpu/frameinterpolation/ffx_frameinterpolation_resources.h>

#include <capture/config.h>
#include <capture/base.h>
#include <capture/execution.h>

#include <cstdint>
#include <memory>
#include <new>

namespace arm::capture::framegenerations
{
    struct CreateDesc
    {
        void*    backendDevice = nullptr;
        uint32_t contextFlags  = 0;
    };

    // The NFRU context spans the frame-interpolation effect and its shared
    // optical-flow/history resources. Capture keeps this context for the
    // lifetime of the controller, like NSS keeps FfxNssContext.
    struct Context
    {
        FfxInterface*                       backendInterface          = nullptr;
        FfxResourceInternal*                sharedResources           = nullptr;
        const FfxFrameInterpolationContext* frameInterpolationContext = nullptr;
    };

    class Controller : public ControllerBase
    {
    public:
        explicit Controller(const CreateDesc& desc, const Context& context);
        ~Controller();

        Controller(const Controller&) = delete;
        Controller& operator=(const Controller&) = delete;

        bool IsEnabled() const;
        bool IsPreparePending() const;
        bool ShouldResetHistoryForDispatch() const;
        void OnPrepare();
        void GetStatus(FfxApiCaptureStatus& status);

        // Starts the transaction and captures NFRU inputs. Returns false when
        // capture is unavailable; it never makes the provider dispatch fail.
        bool BeginDispatchCapture(const ffxApiDispatchDescFrameGeneration&        dispatch,
                                  const ffxApiDispatchDescFrameGenerationPrepare& prepare,
                                  bool                                            effectiveReset = false);
        bool CompleteDispatchCapture(const FfxResource& output);
        void AbortDispatchCapture(const char* reason);

    private:
        struct Recorder;
        void Stop(const char* reason);
        bool OnCaptureBegin(const Config& config) override;
        void OnCaptureStop() override;
        // Keep the backend opaque at the ffx-api boundary. This is a borrowed,
        // backend-owned device token, not a VkDevice or VkDeviceContext owned by
        // the controller. Vulkan handle extraction belongs to the SDK/backend
        // layer, which owns the device context contract and keeps the token valid
        // until controller shutdown. The controller must not dereference the
        // token as a Vulkan handle or extend its lifetime; storing VkDevice/
        // VkPhysicalDevice here would couple this layer to Vulkan and duplicate
        // backend validation.
        void* const               backendDevice_;
        const Context             context_;
        std::shared_ptr<Recorder> recorder_;
    };

}  // namespace arm::capture::framegenerations
