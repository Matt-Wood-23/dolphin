// Copyright 2016 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "VideoBackends/Vulkan/VideoBackend.h"

#include "Common/Logging/LogManager.h"
#include "Common/MsgHandler.h"

#include "VideoBackends/Vulkan/CommandBufferManager.h"
#include "VideoBackends/Vulkan/ObjectCache.h"
#include "VideoBackends/Vulkan/StateTracker.h"
#include "VideoBackends/Vulkan/VKBoundingBox.h"
#include "VideoBackends/Vulkan/VKGfx.h"
#include "VideoBackends/Vulkan/VKPerfQuery.h"
#include "VideoBackends/Vulkan/VKSwapChain.h"
#include "VideoBackends/Vulkan/VKTexture.h"
#include "VideoBackends/Vulkan/VKVertexManager.h"
#include "VideoBackends/Vulkan/VulkanContext.h"

#include <algorithm>

#include "VideoCommon/TextureCacheBase.h"
#include "VideoCommon/VROpenXR.h"
#include "VideoCommon/VROpenXR_Vulkan.h"
#include "VideoCommon/VideoConfig.h"

#if defined(VK_USE_PLATFORM_METAL_EXT)
#include <objc/message.h>
#endif

namespace Vulkan
{
void VideoBackend::InitBackendInfo(const WindowSystemInfo& wsi)
{
  VulkanContext::PopulateBackendInfo(&g_backend_info);

  if (LoadVulkanLibrary())
  {
    u32 vk_api_version = 0;
    VkInstance temp_instance = VulkanContext::CreateVulkanInstance(WindowSystemType::Headless,
                                                                   false, false, &vk_api_version);
    if (temp_instance)
    {
      if (LoadVulkanInstanceFunctions(temp_instance))
      {
        VulkanContext::GPUList gpu_list = VulkanContext::EnumerateGPUs(temp_instance);
        VulkanContext::PopulateBackendInfoAdapters(&g_backend_info, gpu_list);

        if (!gpu_list.empty())
        {
          // Use the selected adapter, or the first to fill features.
          size_t device_index = static_cast<size_t>(g_Config.iAdapter);
          if (device_index >= gpu_list.size())
            device_index = 0;

          VkPhysicalDevice gpu = gpu_list[device_index];
          VulkanContext::PhysicalDeviceInfo properties(gpu);
          VulkanContext::PopulateBackendInfoFeatures(&g_backend_info, gpu, properties);
          VulkanContext::PopulateBackendInfoMultisampleModes(&g_backend_info, gpu, properties);
        }
      }

      vkDestroyInstance(temp_instance, nullptr);
    }
    else
    {
      PanicAlertFmt("Failed to create Vulkan instance.");
    }

    UnloadVulkanLibrary();
  }
  else
  {
    PanicAlertFmt("Failed to load Vulkan library.");
  }
}

// Helper method to check whether the Host GPU logging category is enabled.
static bool IsHostGPULoggingEnabled()
{
  return Common::Log::LogManager::GetInstance()->IsEnabled(Common::Log::LogType::HOST_GPU,
                                                           Common::Log::LogLevel::LERROR);
}

// Helper method to determine whether to enable the debug utils extension.
static bool ShouldEnableDebugUtils(bool enable_validation_layers)
{
  // Enable debug utils if the Host GPU log option is checked, or validation layers are enabled.
  // The only issue here is that if Host GPU is not checked when the instance is created, the debug
  // report extension will not be enabled, requiring the game to be restarted before any reports
  // will be logged. Otherwise, we'd have to enable debug utils on every instance, when most
  // users will never check the Host GPU logging category.
  return enable_validation_layers || IsHostGPULoggingEnabled();
}

// MHTriVR Phase 3c: record a blit of one stereo XFB layer into an OpenXR eye
// swapchain image, on Dolphin's active command buffer. Registered with VROpenXR
// (which is backend-agnostic) so it can drive the copy without depending on the
// Vulkan backend. Runs on the video thread from Presenter::Present (RunFrame).
static void MHTriVRBlit(VkImage dst, u32 dst_width, u32 dst_height,
                        const AbstractTexture* src_xfb, u32 src_layer)
{
  if (dst == VK_NULL_HANDLE || src_xfb == nullptr)
    return;

  if (StateTracker::GetInstance()->InRenderPass())
    StateTracker::GetInstance()->EndRenderPass();

  const VkCommandBuffer cb = g_command_buffer_mgr->GetCurrentCommandBuffer();
  const auto* src = static_cast<const VKTexture*>(src_xfb);
  const u32 layer = std::min(src_layer, src->GetLayers() - 1);  // mono -> both eyes layer 0

  // Source XFB -> TRANSFER_SRC (whole image).
  src->TransitionToLayout(cb, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

  auto barrier = [&](VkImageLayout old_layout, VkImageLayout new_layout, VkAccessFlags src_access,
                     VkAccessFlags dst_access, VkPipelineStageFlags src_stage,
                     VkPipelineStageFlags dst_stage) {
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.srcAccessMask = src_access;
    b.dstAccessMask = dst_access;
    b.oldLayout = old_layout;
    b.newLayout = new_layout;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = dst;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cb, src_stage, dst_stage, 0, 0, nullptr, 0, nullptr, 1, &b);
  };

  // Eye image (contents irrelevant) -> TRANSFER_DST.
  barrier(VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0,
          VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
          VK_PIPELINE_STAGE_TRANSFER_BIT);

  VkImageBlit region{};
  region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, layer, 1};
  region.srcOffsets[0] = {0, 0, 0};
  region.srcOffsets[1] = {static_cast<int32_t>(src->GetWidth()),
                          static_cast<int32_t>(src->GetHeight()), 1};
  region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  region.dstOffsets[0] = {0, 0, 0};
  region.dstOffsets[1] = {static_cast<int32_t>(dst_width), static_cast<int32_t>(dst_height), 1};
  vkCmdBlitImage(cb, src->GetImage(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst,
                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region, VK_FILTER_LINEAR);

  // Eye image -> COLOR_ATTACHMENT_OPTIMAL (what the OpenXR runtime expects).
  barrier(VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
          VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
          VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
}

bool VideoBackend::Initialize(const WindowSystemInfo& wsi)
{
  if (!LoadVulkanLibrary())
  {
    PanicAlertFmt("Failed to load Vulkan library.");
    return false;
  }

  // Check for presence of the validation layers before trying to enable it
  bool enable_validation_layer = g_Config.bEnableValidationLayer;
  if (enable_validation_layer && !VulkanContext::CheckValidationLayerAvailablility())
  {
    WARN_LOG_FMT(VIDEO, "Validation layer requested but not available, disabling.");
    enable_validation_layer = false;
  }

  // Create Vulkan instance, needed before we can create a surface, or enumerate devices.
  // We use this instance to fill in backend info, then re-use it for the actual device.
  bool enable_surface = wsi.type != WindowSystemType::Headless;
  bool enable_debug_utils = ShouldEnableDebugUtils(enable_validation_layer);
  u32 vk_api_version = 0;

  // MHTriVR Phase 3a: bring up OpenXR BEFORE the Vulkan instance, so the runtime's
  // required instance/device extensions can be folded into Dolphin's own device
  // (direct OpenXR-Vulkan binding). The MHTRI_VR_OPENXR compile flag is the gate:
  // this is a no-op in the default build, and a graceful no-op if the headset/
  // runtime is unavailable. (We can't gate on game ID here — the ID isn't set yet
  // when the Vulkan backend initializes.)
  VROpenXR::Initialize();

  VkInstance instance = VulkanContext::CreateVulkanInstance(
      wsi.type, enable_debug_utils, enable_validation_layer, &vk_api_version);
  if (instance == VK_NULL_HANDLE)
  {
    PanicAlertFmt("Failed to create Vulkan instance.");
    UnloadVulkanLibrary();
    return false;
  }

  // Load instance function pointers.
  if (!LoadVulkanInstanceFunctions(instance))
  {
    PanicAlertFmt("Failed to load Vulkan instance functions.");
    vkDestroyInstance(instance, nullptr);
    UnloadVulkanLibrary();
    return false;
  }

  // Obtain a list of physical devices (GPUs) from the instance.
  // We'll re-use this list later when creating the device.
  VulkanContext::GPUList gpu_list = VulkanContext::EnumerateGPUs(instance);
  if (gpu_list.empty())
  {
    PanicAlertFmt("No Vulkan physical devices available.");
    vkDestroyInstance(instance, nullptr);
    UnloadVulkanLibrary();
    return false;
  }

  // Populate BackendInfo with as much information as we can at this point.
  VulkanContext::PopulateBackendInfo(&g_backend_info);
  VulkanContext::PopulateBackendInfoAdapters(&g_backend_info, gpu_list);

  // We need the surface before we can create a device, as some parameters depend on it.
  VkSurfaceKHR surface = VK_NULL_HANDLE;
  if (enable_surface)
  {
    surface = SwapChain::CreateVulkanSurface(instance, wsi);
    if (surface == VK_NULL_HANDLE)
    {
      PanicAlertFmt("Failed to create Vulkan surface.");
      vkDestroyInstance(instance, nullptr);
      UnloadVulkanLibrary();
      return false;
    }
  }

  // Since we haven't called InitializeShared yet, iAdapter may be out of range,
  // so we have to check it ourselves.
  size_t selected_adapter_index = static_cast<size_t>(g_Config.iAdapter);
  if (selected_adapter_index >= gpu_list.size())
  {
    WARN_LOG_FMT(VIDEO, "Vulkan adapter index out of range, selecting first adapter.");
    selected_adapter_index = 0;
  }

  // MHTriVR Phase 3a: OpenXR requires rendering on the GPU the HMD is attached to.
  // If active, override the adapter with the runtime's required physical device.
  // Returns VK_NULL_HANDLE (no override) in the default build / when inactive.
  VkPhysicalDevice selected_gpu = gpu_list[selected_adapter_index];
  if (VkPhysicalDevice xr_gpu = VROpenXR::GetVulkanGraphicsDevice(instance);
      xr_gpu != VK_NULL_HANDLE)
  {
    NOTICE_LOG_FMT(VIDEO, "MHTriVR/OpenXR: overriding Vulkan adapter with the HMD's GPU.");
    selected_gpu = xr_gpu;
  }

  // Now we can create the Vulkan device. VulkanContext takes ownership of the instance and surface.
  g_vulkan_context = VulkanContext::Create(instance, selected_gpu, surface, enable_debug_utils,
                                           enable_validation_layer, vk_api_version);
  if (!g_vulkan_context)
  {
    PanicAlertFmt("Failed to create Vulkan device");
    UnloadVulkanLibrary();
    return false;
  }

  // MHTriVR Phase 3a: hand Dolphin's finished Vulkan handles to OpenXR for session
  // creation (no-op in the default build / when inactive).
  VROpenXR::CheckVulkanGraphicsRequirements(vk_api_version);
  VROpenXR::SetBlitCallback(&MHTriVRBlit);
  VROpenXR::SetVulkanBinding(instance, g_vulkan_context->GetPhysicalDevice(),
                             g_vulkan_context->GetDevice(),
                             g_vulkan_context->GetGraphicsQueueFamilyIndex(), 0);

  // Since VulkanContext maintains a copy of the device features and properties, we can use this
  // to initialize the backend information, so that we don't need to enumerate everything again.
  VulkanContext::PopulateBackendInfoFeatures(&g_backend_info, g_vulkan_context->GetPhysicalDevice(),
                                             g_vulkan_context->GetDeviceInfo());
  VulkanContext::PopulateBackendInfoMultisampleModes(
      &g_backend_info, g_vulkan_context->GetPhysicalDevice(), g_vulkan_context->GetDeviceInfo());
  g_backend_info.bSupportsExclusiveFullscreen =
      enable_surface && g_vulkan_context->SupportsExclusiveFullscreen(wsi, surface);

  UpdateActiveConfig();

  // Remaining classes are also dependent on object cache.
  g_object_cache = std::make_unique<ObjectCache>();
  if (!g_object_cache->Initialize())
  {
    PanicAlertFmt("Failed to initialize Vulkan object cache.");
    Shutdown();
    return false;
  }

  // Create swap chain. This has to be done early so that the target size is correct for auto-scale.
  std::unique_ptr<SwapChain> swap_chain;
  if (surface != VK_NULL_HANDLE)
  {
    swap_chain = SwapChain::Create(wsi, surface, g_ActiveConfig.bVSyncActive);
    if (!swap_chain)
    {
      PanicAlertFmt("Failed to create Vulkan swap chain.");
      Shutdown();
      return false;
    }
  }

  // Create command buffers. We do this separately because the other classes depend on it.
  // MHTriVR Phase 3c: when OpenXR is active, force SINGLE-THREADED submission. The XR
  // compositor submits to our VkQueue during xrEndFrame on the video thread; Dolphin's
  // worker submit thread would race it on the same queue (VK_ERROR_DEVICE_LOST).
  const bool threaded_submission = g_Config.bBackendMultithreading && !VROpenXR::IsInitialized();
  g_command_buffer_mgr = std::make_unique<CommandBufferManager>(threaded_submission);
  size_t swapchain_image_count =
      surface != VK_NULL_HANDLE ? swap_chain->GetSwapChainImageCount() : 0;
  if (!g_command_buffer_mgr->Initialize(swapchain_image_count))
  {
    PanicAlertFmt("Failed to create Vulkan command buffers");
    Shutdown();
    return false;
  }

  if (!StateTracker::CreateInstance())
  {
    PanicAlertFmt("Failed to create state tracker");
    Shutdown();
    return false;
  }

  auto gfx = std::make_unique<VKGfx>(std::move(swap_chain), wsi.render_surface_scale);
  auto vertex_manager = std::make_unique<VertexManager>();
  auto perf_query = std::make_unique<PerfQuery>();
  auto bounding_box = std::make_unique<VKBoundingBox>();

  return InitializeShared(std::move(gfx), std::move(vertex_manager), std::move(perf_query),
                          std::move(bounding_box));
}

void VideoBackend::Shutdown()
{
  if (g_vulkan_context)
    vkDeviceWaitIdle(g_vulkan_context->GetDevice());

  if (g_object_cache)
    g_object_cache->Shutdown();

  ShutdownShared();

  g_object_cache.reset();
  StateTracker::DestroyInstance();
  g_command_buffer_mgr.reset();
  g_vulkan_context.reset();
  UnloadVulkanLibrary();
}

void VideoBackend::PrepareWindow(WindowSystemInfo& wsi)
{
#if defined(VK_USE_PLATFORM_METAL_EXT)
  // We only need to manually create the CAMetalLayer on macOS.
  if (wsi.type != WindowSystemType::MacOS)
    return;

  // This is kinda messy, but it avoids having to write Objective C++ just to create a metal layer.
  id view = reinterpret_cast<id>(wsi.render_surface);
  Class clsCAMetalLayer = objc_getClass("CAMetalLayer");
  if (!clsCAMetalLayer)
  {
    ERROR_LOG_FMT(VIDEO, "Failed to get CAMetalLayer class.");
    return;
  }

  // [CAMetalLayer layer]
  id layer = reinterpret_cast<id (*)(Class, SEL)>(objc_msgSend)(objc_getClass("CAMetalLayer"),
                                                                sel_getUid("layer"));
  if (!layer)
  {
    ERROR_LOG_FMT(VIDEO, "Failed to create Metal layer.");
    return;
  }

  // [view setWantsLayer:YES]
  reinterpret_cast<void (*)(id, SEL, BOOL)>(objc_msgSend)(view, sel_getUid("setWantsLayer:"), YES);

  // [view setLayer:layer]
  reinterpret_cast<void (*)(id, SEL, id)>(objc_msgSend)(view, sel_getUid("setLayer:"), layer);

  // NSScreen* screen = [NSScreen mainScreen]
  id screen = reinterpret_cast<id (*)(Class, SEL)>(objc_msgSend)(objc_getClass("NSScreen"),
                                                                 sel_getUid("mainScreen"));

  // CGFloat factor = [screen backingScaleFactor]
  double factor =
      reinterpret_cast<double (*)(id, SEL)>(objc_msgSend)(screen, sel_getUid("backingScaleFactor"));

  // layer.contentsScale = factor
  reinterpret_cast<void (*)(id, SEL, double)>(objc_msgSend)(layer, sel_getUid("setContentsScale:"),
                                                            factor);

  // Store the layer pointer, that way MoltenVK doesn't call [NSView layer] outside the main thread.
  wsi.render_surface = layer;
#endif
}
}  // namespace Vulkan
