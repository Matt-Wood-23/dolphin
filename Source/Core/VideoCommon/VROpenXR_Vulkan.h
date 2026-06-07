// Copyright 2026 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// MHTriVR Phase 3a — OpenXR <-> Vulkan binding surface.
//
// Split out from VROpenXR.h because these declarations use Vulkan types: only
// the Vulkan backend (which already includes vulkan.h) includes this. The
// generic VROpenXR.h stays Vulkan-free for the rest of the engine.
//
// All functions are defined in VROpenXR.cpp. With the MHTRI_VR_OPENXR flag off
// they are inert (empty lists / VK_NULL_HANDLE / true), so VulkanContext can
// call them unconditionally without affecting the default build.
//
// Integration flow (vulkan_enable, KHR v1), interleaved into VulkanContext:
//   1. VROpenXR::Initialize() must have created the XR instance + system first.
//   2. Before vkCreateInstance: append GetRequiredVulkanInstanceExtensions().
//   3. After the VkInstance exists: GetVulkanGraphicsDevice() returns the one
//      VkPhysicalDevice the headset is on — Dolphin must select that GPU.
//   4. Before vkCreateDevice: append GetRequiredVulkanDeviceExtensions().
//   5. After the VkDevice + queue exist: SetVulkanBinding(...) hands the handles
//      to VROpenXR so it can xrCreateSession (done in a later step).

#pragma once

#include <string>
#include <vector>

#include <vulkan/vulkan.h>

#include "Common/CommonTypes.h"

class AbstractTexture;

namespace VROpenXR
{
// Records a blit of one layer of the (stereo) XFB texture into an OpenXR eye
// swapchain VkImage, on Dolphin's active command buffer (with layout barriers).
// Provided by the Vulkan backend (which has VKTexture/CommandBufferManager) and
// registered via SetBlitCallback, so VideoCommon needn't depend on the backend.
using VRBlitCallback = void (*)(VkImage dst, u32 dst_width, u32 dst_height,
                                const AbstractTexture* src_xfb, u32 src_layer);
void SetBlitCallback(VRBlitCallback cb);

// Extensions the OpenXR runtime requires on the Vulkan instance / device.
// Queried once during Initialize() and returned by reference to persistent
// storage, so the c_str() pointers stay valid while VulkanContext builds its
// extension list. Empty when OpenXR is inactive.
const std::vector<std::string>& GetRequiredVulkanInstanceExtensions();
const std::vector<std::string>& GetRequiredVulkanDeviceExtensions();

// Validate Dolphin's chosen Vulkan API version against the runtime's min/max.
// Returns true (no constraint) when inactive.
bool CheckVulkanGraphicsRequirements(u32 vk_api_version);

// The VkPhysicalDevice the HMD is attached to — Dolphin must use this GPU when
// OpenXR is active. VK_NULL_HANDLE when inactive (no constraint).
VkPhysicalDevice GetVulkanGraphicsDevice(VkInstance vk_instance);

// Hand Dolphin's finished Vulkan handles to VROpenXR for session creation
// (consumed in a later step). No-op when inactive.
void SetVulkanBinding(VkInstance instance, VkPhysicalDevice physical_device, VkDevice device,
                      u32 queue_family_index, u32 queue_index);
}  // namespace VROpenXR
