// Copyright 2026 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "VideoCommon/VROpenXR.h"

#include "Common/CommonTypes.h"
#include "VideoCommon/VROpenXR_Vulkan.h"  // Vulkan-typed binding API (defined below)

// Phase 3 (HMD presentation) is gated behind MHTRI_VR_OPENXR. The default build
// does NOT define it, so this translation unit compiles to inert no-ops with no
// OpenXR loader dependency — the working monitor build (Phases 1-2) is
// unaffected. Build the VR path with: msbuild -p:MHTriVROpenXR=true (sets the
// define, adds Externals/OpenXR include + links openxr_loader.lib). See
// reports/vr_mod_plan.md Phase 3.

#ifdef MHTRI_VR_OPENXR

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

// openxr_platform.h exposes the Vulkan binding (XrGraphicsBindingVulkanKHR,
// xrGetVulkanGraphicsDeviceKHR, ...) only when XR_USE_GRAPHICS_API_VULKAN is set
// and vulkan.h is already included (pulled in via VROpenXR_Vulkan.h above).
#define XR_USE_GRAPHICS_API_VULKAN
#include <openxr/openxr.h>
#include <openxr/openxr_platform.h>

#include "Common/Logging/Log.h"

#include "VideoCommon/VRStereo.h"

namespace VROpenXR
{
namespace
{
XrInstance s_instance = XR_NULL_HANDLE;
XrSystemId s_system_id = XR_NULL_SYSTEM_ID;
XrSessionState s_session_state = XR_SESSION_STATE_UNKNOWN;
std::atomic<bool> s_session_running{false};  // true between xrBeginSession/xrEndSession

// Latest head orientation (game-space yaw/pitch, radians). Written on the video
// thread in RunFrame, read on the CPU thread in GetHeadPose -> atomics.
std::atomic<float> s_head_yaw{0.0f};
std::atomic<float> s_head_pitch{0.0f};
std::atomic<bool> s_pose_valid{false};
std::atomic<float> s_fov_scale{1.0f};

// Dolphin's Vulkan handles, captured via SetVulkanBinding for session creation.
VkInstance s_vk_instance = VK_NULL_HANDLE;
VkPhysicalDevice s_vk_physical_device = VK_NULL_HANDLE;
VkDevice s_vk_device = VK_NULL_HANDLE;
u32 s_vk_queue_family = 0;
u32 s_vk_queue_index = 0;

// Persistent storage for the runtime-required Vulkan extensions (queried once in
// Initialize). VulkanContext stores raw c_str() pointers, so these must outlive
// instance/device creation — function-level statics here do exactly that.
std::vector<std::string> s_vk_instance_exts;
std::vector<std::string> s_vk_device_exts;

// Session + per-eye swapchains (created once the Vulkan binding is captured).
constexpr u32 kEyeCount = 2;  // XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO
XrSession s_session = XR_NULL_HANDLE;
XrSpace s_space = XR_NULL_HANDLE;        // LOCAL (world)
XrSpace s_view_space = XR_NULL_HANDLE;   // VIEW (head-locked, for the quad)
XrSwapchain s_swapchains[kEyeCount] = {XR_NULL_HANDLE, XR_NULL_HANDLE};
std::vector<VkImage> s_swapchain_images[kEyeCount];
u32 s_eye_width[kEyeCount] = {0, 0};
u32 s_eye_height[kEyeCount] = {0, 0};
int64_t s_swapchain_format = 0;

// Per-frame submit state (RunFrame -> EndFrame, video thread only).
VRBlitCallback s_blit_cb = nullptr;
bool s_frame_begun = false;
bool s_should_submit = false;
bool s_acquired[kEyeCount] = {false, false};
XrTime s_predicted_display_time = 0;
u32 s_frame_counter = 0;
int s_last_should_render = -1;
XrResult s_last_endframe = XR_SUCCESS;
XrCompositionLayerProjectionView s_proj_views[kEyeCount] = {
    {XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW}, {XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW}};

// KHR_vulkan_enable entry points (loaded via xrGetInstanceProcAddr).
PFN_xrGetVulkanGraphicsRequirementsKHR s_xrGetVulkanGraphicsRequirementsKHR = nullptr;
PFN_xrGetVulkanInstanceExtensionsKHR s_xrGetVulkanInstanceExtensionsKHR = nullptr;
PFN_xrGetVulkanDeviceExtensionsKHR s_xrGetVulkanDeviceExtensionsKHR = nullptr;
PFN_xrGetVulkanGraphicsDeviceKHR s_xrGetVulkanGraphicsDeviceKHR = nullptr;

template <typename Fn>
bool LoadXrFn(const char* name, Fn& out)
{
  return XR_SUCCEEDED(
      xrGetInstanceProcAddr(s_instance, name, reinterpret_cast<PFN_xrVoidFunction*>(&out)));
}

std::vector<std::string> SplitSpaceList(const std::string& s)
{
  std::vector<std::string> out;
  std::string cur;
  for (const char c : s)
  {
    if (c == ' ' || c == '\0')
    {
      if (!cur.empty())
      {
        out.push_back(cur);
        cur.clear();
      }
    }
    else
    {
      cur += c;
    }
  }
  if (!cur.empty())
    out.push_back(cur);
  return out;
}

std::vector<std::string> QueryExtList(PFN_xrGetVulkanInstanceExtensionsKHR fn)
{
  if (fn == nullptr || s_system_id == XR_NULL_SYSTEM_ID)
    return {};
  u32 cap = 0;
  if (XR_FAILED(fn(s_instance, s_system_id, 0, &cap, nullptr)) || cap == 0)
    return {};
  std::string buf(cap, '\0');
  if (XR_FAILED(fn(s_instance, s_system_id, cap, &cap, buf.data())))
    return {};
  return SplitSpaceList(buf);
}

// TEMP (Phase-3 smoke test): mirror the OpenXR init outcome to a file so we can
// confirm HMD detection without depending on Dolphin's log config. Remove once
// the real session path is verified.
void SmokeResult(const char* msg)
{
  if (FILE* f = std::fopen("D:/Matt/Games/dolphin-fork/mhtri_openxr_smoketest.txt", "w"))
  {
    std::fputs(msg, f);
    std::fputc('\n', f);
    std::fclose(f);
  }
}

// Separate file for the Vulkan-binding outcome so the game-boot Initialize()
// retry can't overwrite it. TEMP — remove with the smoke test.
void BindResult(const char* msg)
{
  if (FILE* f = std::fopen("D:/Matt/Games/dolphin-fork/mhtri_openxr_bind.txt", "w"))
  {
    std::fputs(msg, f);
    std::fputc('\n', f);
    std::fclose(f);
  }
}

// Session-creation outcome (kept distinct from BIND OK). TEMP.
void SessionResult(const char* msg)
{
  if (FILE* f = std::fopen("D:/Matt/Games/dolphin-fork/mhtri_openxr_session.txt", "w"))
  {
    std::fputs(msg, f);
    std::fputc('\n', f);
    std::fclose(f);
  }
}

// Per-frame loop diagnostic (overwritten periodically). TEMP.
void FrameResult(const char* msg)
{
  if (FILE* f = std::fopen("D:/Matt/Games/dolphin-fork/mhtri_openxr_frame.txt", "w"))
  {
    std::fputs(msg, f);
    std::fputc('\n', f);
    std::fclose(f);
  }
}

// Live VR tuning: re-read a small key=value config file so values can be tuned
// without relaunching (env vars don't inherit reliably through the launcher).
// Keys: fov_scale, depth, convergence. TEMP (fold into a real UI later).
void ReadVRConfig()
{
  FILE* f = std::fopen("D:/Matt/Games/dolphin-fork/mhtri_vr_config.txt", "r");
  if (f == nullptr)
    return;
  char line[128];
  while (std::fgets(line, sizeof(line), f) != nullptr)
  {
    float v = 0.0f;
    if (std::sscanf(line, "fov_scale=%f", &v) == 1)
      SetFovScale(v);
    else if (std::sscanf(line, "depth=%f", &v) == 1)
      VRStereo::SetDepth(v);
    else if (std::sscanf(line, "convergence=%f", &v) == 1)
      VRStereo::SetConvergence(v);
  }
  std::fclose(f);
}

// Create the XR session bound to Dolphin's Vulkan device, plus the per-eye
// swapchains and a reference space. Does NOT begin the session or submit frames
// yet (that's the per-frame state machine in a later step). Called once the
// Vulkan binding is captured.
bool CreateSessionInternal()
{
  if (s_session != XR_NULL_HANDLE)
    return true;
  if (s_system_id == XR_NULL_SYSTEM_ID || s_vk_device == VK_NULL_HANDLE)
    return false;

  XrGraphicsBindingVulkanKHR binding{XR_TYPE_GRAPHICS_BINDING_VULKAN_KHR};
  binding.instance = s_vk_instance;
  binding.physicalDevice = s_vk_physical_device;
  binding.device = s_vk_device;
  binding.queueFamilyIndex = s_vk_queue_family;
  binding.queueIndex = s_vk_queue_index;

  XrSessionCreateInfo session_info{XR_TYPE_SESSION_CREATE_INFO};
  session_info.next = &binding;
  session_info.systemId = s_system_id;
  if (XrResult r = xrCreateSession(s_instance, &session_info, &s_session); XR_FAILED(r))
  {
    char buf[128];
    std::snprintf(buf, sizeof(buf), "SESSION FAIL: xrCreateSession failed (%d)",
                  static_cast<int>(r));
    SessionResult(buf);
    s_session = XR_NULL_HANDLE;
    return false;
  }

  // Per-eye recommended render resolution.
  u32 view_count = 0;
  xrEnumerateViewConfigurationViews(s_instance, s_system_id,
                                    XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO, 0, &view_count,
                                    nullptr);
  std::vector<XrViewConfigurationView> views(view_count, {XR_TYPE_VIEW_CONFIGURATION_VIEW});
  xrEnumerateViewConfigurationViews(s_instance, s_system_id,
                                    XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO, view_count,
                                    &view_count, views.data());

  // Pick an sRGB 8-bit swapchain format if offered (matches Dolphin's XFB).
  u32 fmt_count = 0;
  xrEnumerateSwapchainFormats(s_session, 0, &fmt_count, nullptr);
  std::vector<int64_t> formats(fmt_count);
  xrEnumerateSwapchainFormats(s_session, fmt_count, &fmt_count, formats.data());
  s_swapchain_format = formats.empty() ? static_cast<int64_t>(VK_FORMAT_R8G8B8A8_SRGB) : formats[0];
  for (const int64_t f : formats)
  {
    if (f == VK_FORMAT_R8G8B8A8_SRGB || f == VK_FORMAT_B8G8R8A8_SRGB)
    {
      s_swapchain_format = f;
      break;
    }
  }

  const u32 eyes = std::min<u32>(view_count, kEyeCount);
  for (u32 e = 0; e < eyes; ++e)
  {
    s_eye_width[e] = views[e].recommendedImageRectWidth;
    s_eye_height[e] = views[e].recommendedImageRectHeight;

    XrSwapchainCreateInfo sc{XR_TYPE_SWAPCHAIN_CREATE_INFO};
    sc.usageFlags = XR_SWAPCHAIN_USAGE_TRANSFER_DST_BIT | XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT;
    sc.format = s_swapchain_format;
    sc.sampleCount = 1;
    sc.width = s_eye_width[e];
    sc.height = s_eye_height[e];
    sc.faceCount = 1;
    sc.arraySize = 1;
    sc.mipCount = 1;
    if (XR_FAILED(xrCreateSwapchain(s_session, &sc, &s_swapchains[e])))
    {
      char buf[128];
      std::snprintf(buf, sizeof(buf), "SESSION FAIL: xrCreateSwapchain eye %u failed", e);
      SessionResult(buf);
      return false;
    }

    u32 img_count = 0;
    xrEnumerateSwapchainImages(s_swapchains[e], 0, &img_count, nullptr);
    std::vector<XrSwapchainImageVulkanKHR> imgs(img_count, {XR_TYPE_SWAPCHAIN_IMAGE_VULKAN_KHR});
    xrEnumerateSwapchainImages(s_swapchains[e], img_count, &img_count,
                               reinterpret_cast<XrSwapchainImageBaseHeader*>(imgs.data()));
    s_swapchain_images[e].clear();
    for (const auto& im : imgs)
      s_swapchain_images[e].push_back(im.image);
  }

  // Seated/standing reference space (identity pose).
  XrReferenceSpaceCreateInfo space_info{XR_TYPE_REFERENCE_SPACE_CREATE_INFO};
  space_info.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_LOCAL;
  space_info.poseInReferenceSpace.orientation.w = 1.0f;
  xrCreateReferenceSpace(s_session, &space_info, &s_space);

  // Head-locked space for the virtual-screen (quad) layer: the screen stays in
  // front of the user regardless of where they look.
  XrReferenceSpaceCreateInfo view_space_info{XR_TYPE_REFERENCE_SPACE_CREATE_INFO};
  view_space_info.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_VIEW;
  view_space_info.poseInReferenceSpace.orientation.w = 1.0f;
  xrCreateReferenceSpace(s_session, &view_space_info, &s_view_space);

  NOTICE_LOG_FMT(VIDEO,
                 "MHTriVR/OpenXR: SESSION CREATED. views={} eye0={}x{} eye1={}x{} fmt={} "
                 "imagesL={} imagesR={}",
                 view_count, s_eye_width[0], s_eye_height[0], s_eye_width[1], s_eye_height[1],
                 s_swapchain_format, s_swapchain_images[0].size(), s_swapchain_images[1].size());

  char buf[384];
  std::snprintf(buf, sizeof(buf),
                "SESSION CREATED: views=%u eye0=%ux%u eye1=%ux%u fmt=%lld imgsL=%zu imgsR=%zu. "
                "(not begun/submitting yet — that's the next step)",
                view_count, s_eye_width[0], s_eye_height[0], s_eye_width[1], s_eye_height[1],
                static_cast<long long>(s_swapchain_format), s_swapchain_images[0].size(),
                s_swapchain_images[1].size());
  SessionResult(buf);
  return true;
}
}  // namespace

bool Initialize()
{
  if (s_instance != XR_NULL_HANDLE)
    return true;  // already up

  // Create the instance WITH the Vulkan binding extension so the
  // xrGetVulkanGraphics*KHR entry points are available for the session.
  const char* const enabled_exts[] = {XR_KHR_VULKAN_ENABLE_EXTENSION_NAME};

  XrInstanceCreateInfo create_info{XR_TYPE_INSTANCE_CREATE_INFO};
  std::strncpy(create_info.applicationInfo.applicationName, "Dolphin MHTriVR",
               XR_MAX_APPLICATION_NAME_SIZE - 1);
  create_info.applicationInfo.applicationVersion = 1;
  std::strncpy(create_info.applicationInfo.engineName, "Dolphin", XR_MAX_ENGINE_NAME_SIZE - 1);
  create_info.applicationInfo.engineVersion = 1;
  create_info.applicationInfo.apiVersion = XR_CURRENT_API_VERSION;
  create_info.enabledExtensionCount = 1;
  create_info.enabledExtensionNames = enabled_exts;

  XrResult result = xrCreateInstance(&create_info, &s_instance);
  if (XR_FAILED(result))
  {
    ERROR_LOG_FMT(VIDEO,
                  "MHTriVR/OpenXR: xrCreateInstance failed ({}). Is an OpenXR runtime active "
                  "(Meta/Oculus via Quest Link) with Vulkan support?",
                  static_cast<int>(result));
    char buf[160];
    std::snprintf(buf, sizeof(buf),
                  "FAIL: xrCreateInstance(vulkan_enable) failed (%d) — runtime/vulkan support?",
                  static_cast<int>(result));
    SmokeResult(buf);
    s_instance = XR_NULL_HANDLE;
    return false;
  }

  XrInstanceProperties inst_props{XR_TYPE_INSTANCE_PROPERTIES};
  xrGetInstanceProperties(s_instance, &inst_props);

  XrSystemGetInfo system_info{XR_TYPE_SYSTEM_GET_INFO};
  system_info.formFactor = XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY;

  // The HMD form factor is often transiently unavailable while the runtime/headset
  // spins up (XR_ERROR_FORM_FACTOR_UNAVAILABLE). The spec says to retry. We do, for
  // up to ~4s, because this runs at emulation start before the Vulkan device — the
  // session needs the system NOW, not a second later. A real error breaks early.
  constexpr int kMaxTries = 40;
  for (int attempt = 0; attempt < kMaxTries; ++attempt)
  {
    result = xrGetSystem(s_instance, &system_info, &s_system_id);
    if (XR_SUCCEEDED(result) || result != XR_ERROR_FORM_FACTOR_UNAVAILABLE)
      break;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  if (XR_FAILED(result))
  {
    ERROR_LOG_FMT(VIDEO,
                  "MHTriVR/OpenXR: xrGetSystem failed ({}). Runtime '{}' up but no HMD — headset "
                  "on? Quest Link connected?",
                  static_cast<int>(result), inst_props.runtimeName);
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "PARTIAL: runtime='%s' reached, xrGetSystem failed (%d) — HMD not available",
                  inst_props.runtimeName, static_cast<int>(result));
    SmokeResult(buf);
    xrDestroyInstance(s_instance);
    s_instance = XR_NULL_HANDLE;
    s_system_id = XR_NULL_SYSTEM_ID;
    return false;
  }

  XrSystemProperties sys_props{XR_TYPE_SYSTEM_PROPERTIES};
  xrGetSystemProperties(s_instance, s_system_id, &sys_props);

  // Load the Vulkan-binding entry points now that the instance + extension exist.
  LoadXrFn("xrGetVulkanGraphicsRequirementsKHR", s_xrGetVulkanGraphicsRequirementsKHR);
  LoadXrFn("xrGetVulkanInstanceExtensionsKHR", s_xrGetVulkanInstanceExtensionsKHR);
  LoadXrFn("xrGetVulkanDeviceExtensionsKHR", s_xrGetVulkanDeviceExtensionsKHR);
  LoadXrFn("xrGetVulkanGraphicsDeviceKHR", s_xrGetVulkanGraphicsDeviceKHR);
  const bool vk_fns_ok = s_xrGetVulkanGraphicsRequirementsKHR &&
                         s_xrGetVulkanInstanceExtensionsKHR && s_xrGetVulkanDeviceExtensionsKHR &&
                         s_xrGetVulkanGraphicsDeviceKHR;

  // Query the runtime-required Vulkan extensions once, into persistent storage.
  s_vk_instance_exts = QueryExtList(s_xrGetVulkanInstanceExtensionsKHR);
  s_vk_device_exts = QueryExtList(s_xrGetVulkanDeviceExtensionsKHR);
  NOTICE_LOG_FMT(VIDEO, "MHTriVR/OpenXR: runtime requires {} instance + {} device Vulkan extensions",
                 s_vk_instance_exts.size(), s_vk_device_exts.size());

  NOTICE_LOG_FMT(VIDEO,
                 "MHTriVR/OpenXR: HMD DETECTED. runtime='{}' system='{}' maxSwapchain={}x{} "
                 "vulkan_fns={}. (instance+system OK; session not yet created)",
                 inst_props.runtimeName, sys_props.systemName,
                 sys_props.graphicsProperties.maxSwapchainImageWidth,
                 sys_props.graphicsProperties.maxSwapchainImageHeight, vk_fns_ok);

  char buf[384];
  std::snprintf(buf, sizeof(buf),
                "PASS: HMD DETECTED. runtime='%s' system='%s' maxSwapchain=%ux%u vulkan_fns=%d",
                inst_props.runtimeName, sys_props.systemName,
                sys_props.graphicsProperties.maxSwapchainImageWidth,
                sys_props.graphicsProperties.maxSwapchainImageHeight, vk_fns_ok ? 1 : 0);
  SmokeResult(buf);

  return true;
}

void Shutdown()
{
  for (u32 e = 0; e < kEyeCount; ++e)
  {
    if (s_swapchains[e] != XR_NULL_HANDLE)
    {
      xrDestroySwapchain(s_swapchains[e]);
      s_swapchains[e] = XR_NULL_HANDLE;
    }
    s_swapchain_images[e].clear();
  }
  if (s_space != XR_NULL_HANDLE)
  {
    xrDestroySpace(s_space);
    s_space = XR_NULL_HANDLE;
  }
  if (s_view_space != XR_NULL_HANDLE)
  {
    xrDestroySpace(s_view_space);
    s_view_space = XR_NULL_HANDLE;
  }
  if (s_session != XR_NULL_HANDLE)
  {
    xrDestroySession(s_session);
    s_session = XR_NULL_HANDLE;
  }
  if (s_instance != XR_NULL_HANDLE)
  {
    xrDestroyInstance(s_instance);
    s_instance = XR_NULL_HANDLE;
  }
  s_system_id = XR_NULL_SYSTEM_ID;
  s_session_running = false;
}

bool IsInitialized()
{
  return s_instance != XR_NULL_HANDLE && s_system_id != XR_NULL_SYSTEM_ID;
}

bool IsActive()
{
  return s_session_running.load();
}

void SetFovScale(float scale)
{
  if (scale > 0.05f && scale < 20.0f)
    s_fov_scale.store(scale);
}

float GetFovScale()
{
  return s_fov_scale.load();
}

namespace
{
// Pump the session lifecycle events. Begins/ends the session on READY/STOPPING.
void PollEvents()
{
  XrEventDataBuffer ev{XR_TYPE_EVENT_DATA_BUFFER};
  while (xrPollEvent(s_instance, &ev) == XR_SUCCESS)
  {
    if (ev.type == XR_TYPE_EVENT_DATA_SESSION_STATE_CHANGED)
    {
      const auto* ssc = reinterpret_cast<const XrEventDataSessionStateChanged*>(&ev);
      s_session_state = ssc->state;
      switch (ssc->state)
      {
      case XR_SESSION_STATE_READY:
      {
        XrSessionBeginInfo begin_info{XR_TYPE_SESSION_BEGIN_INFO};
        begin_info.primaryViewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
        if (XR_SUCCEEDED(xrBeginSession(s_session, &begin_info)))
        {
          s_session_running.store(true);
          NOTICE_LOG_FMT(VIDEO, "MHTriVR/OpenXR: session RUNNING (head tracking live)");
          SessionResult("SESSION RUNNING: head tracking live (3b). Frame submit = next step.");
        }
        break;
      }
      case XR_SESSION_STATE_STOPPING:
        s_session_running.store(false);
        xrEndSession(s_session);
        break;
      case XR_SESSION_STATE_EXITING:
      case XR_SESSION_STATE_LOSS_PENDING:
        s_session_running.store(false);
        break;
      default:
        break;
      }
    }
    ev = {XR_TYPE_EVENT_DATA_BUFFER};
  }
}

// Convert an OpenXR view orientation (-Z forward, +Y up, +X right) into the
// game's yaw (look right = +) / pitch (look up = +) by rotating the forward
// vector and reading its heading/elevation.
void OrientationToYawPitch(const XrQuaternionf& q, float& yaw, float& pitch)
{
  const float fx = -2.0f * (q.x * q.z + q.w * q.y);
  const float fy = -2.0f * (q.y * q.z - q.w * q.x);
  const float fz = -(1.0f - 2.0f * (q.x * q.x + q.y * q.y));
  yaw = std::atan2(fx, -fz);
  pitch = std::asin(std::clamp(fy, -1.0f, 1.0f));
}
}  // namespace

void RunFrame(const AbstractTexture* xfb_stereo_array)
{
  if (s_session == XR_NULL_HANDLE)
    return;

  // Live tuning: re-read the config file ~twice a second.
  static u32 s_cfg_tick = 0;
  if ((s_cfg_tick++ % 45) == 0)
    ReadVRConfig();

  PollEvents();
  if (!s_session_running.load())
    return;

  XrFrameWaitInfo wait_info{XR_TYPE_FRAME_WAIT_INFO};
  XrFrameState frame_state{XR_TYPE_FRAME_STATE};
  if (XR_FAILED(xrWaitFrame(s_session, &wait_info, &frame_state)))
    return;

  XrFrameBeginInfo begin_info{XR_TYPE_FRAME_BEGIN_INFO};
  if (XR_FAILED(xrBeginFrame(s_session, &begin_info)))
    return;

  s_frame_begun = true;
  s_should_submit = false;
  s_acquired[0] = s_acquired[1] = false;
  s_predicted_display_time = frame_state.predictedDisplayTime;
  s_last_should_render = (frame_state.shouldRender == XR_TRUE) ? 1 : 0;

  if (frame_state.shouldRender != XR_TRUE)
    return;  // EndFrame will submit 0 layers

  // Locate views: head pose (for the camera) + per-eye pose/fov (for the layer).
  XrViewLocateInfo locate_info{XR_TYPE_VIEW_LOCATE_INFO};
  locate_info.viewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
  locate_info.displayTime = frame_state.predictedDisplayTime;
  locate_info.space = s_space;
  XrViewState view_state{XR_TYPE_VIEW_STATE};
  XrView views[kEyeCount] = {{XR_TYPE_VIEW}, {XR_TYPE_VIEW}};
  u32 located = 0;
  const bool views_ok =
      XR_SUCCEEDED(
          xrLocateViews(s_session, &locate_info, &view_state, kEyeCount, &located, views)) &&
      (view_state.viewStateFlags & XR_VIEW_STATE_ORIENTATION_VALID_BIT);
  if (views_ok)
  {
    float yaw = 0.0f;
    float pitch = 0.0f;
    OrientationToYawPitch(views[0].pose.orientation, yaw, pitch);
    s_head_yaw.store(yaw);
    s_head_pitch.store(pitch);
    s_pose_valid.store(true);
  }

  // Submit only when we have a source + a backend blit.
  if (xfb_stereo_array == nullptr || s_blit_cb == nullptr)
    return;

  // Virtual-screen (stereo quad) approach: blit each eye's render (XFB array
  // layer 0 = left, layer 1 = right) into its own swapchain, then EndFrame shows
  // two quad layers (LEFT/RIGHT) at the SAME screen pose/size. Because both quads
  // sit at one spot, the eyes fuse into one flat screen (no unfusable doubling),
  // while the per-eye parallax from Dolphin's GS stereo gives real depth back.
  for (u32 e = 0; e < kEyeCount; ++e)
  {
    u32 image_index = 0;
    XrSwapchainImageAcquireInfo acquire_info{XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO};
    if (XR_FAILED(xrAcquireSwapchainImage(s_swapchains[e], &acquire_info, &image_index)))
      return;

    // Finite timeout (100ms) so a stuck image can never hang the video thread.
    XrSwapchainImageWaitInfo swap_wait{XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO};
    swap_wait.timeout = 100'000'000;  // ns
    if (xrWaitSwapchainImage(s_swapchains[e], &swap_wait) != XR_SUCCESS)
    {
      XrSwapchainImageReleaseInfo rel{XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO};
      xrReleaseSwapchainImage(s_swapchains[e], &rel);
      return;
    }
    s_acquired[e] = true;

    // src_layer = e: layer 0 -> left eye, layer 1 -> right eye.
    s_blit_cb(s_swapchain_images[e][image_index], s_eye_width[e], s_eye_height[e],
              xfb_stereo_array, e);
  }
  s_should_submit = true;
}

void EndFrame()
{
  if (s_session == XR_NULL_HANDLE || !s_frame_begun)
    return;

  // Release the images we acquired (their blits were just submitted to the queue
  // by PresentBackbuffer), then submit the two quads.
  for (u32 e = 0; e < kEyeCount; ++e)
  {
    if (s_acquired[e])
    {
      XrSwapchainImageReleaseInfo rel{XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO};
      xrReleaseSwapchainImage(s_swapchains[e], &rel);
      s_acquired[e] = false;
    }
  }

  // Flat virtual screen, head-locked (always in front), ~2.4m wide at 2m. Two
  // quads at the SAME pose/size, one per eye, each fed its own eye's render: the
  // eyes fuse into one screen while keeping the per-eye parallax (stereo depth).
  // 16:9 quad so the XFB (stretched to fill the square swapchain) reads correctly.
  static const XrEyeVisibility kEyeVis[kEyeCount] = {XR_EYE_VISIBILITY_LEFT,
                                                     XR_EYE_VISIBILITY_RIGHT};
  XrCompositionLayerQuad quads[kEyeCount];
  const XrCompositionLayerBaseHeader* layers[kEyeCount];
  for (u32 e = 0; e < kEyeCount; ++e)
  {
    XrCompositionLayerQuad& quad = quads[e];
    quad = XrCompositionLayerQuad{XR_TYPE_COMPOSITION_LAYER_QUAD};
    quad.layerFlags = 0;
    quad.space = s_view_space;
    quad.eyeVisibility = kEyeVis[e];
    quad.subImage.swapchain = s_swapchains[e];
    quad.subImage.imageRect.offset = {0, 0};
    quad.subImage.imageRect.extent = {static_cast<int32_t>(s_eye_width[e]),
                                      static_cast<int32_t>(s_eye_height[e])};
    quad.subImage.imageArrayIndex = 0;
    quad.pose.orientation = {0.0f, 0.0f, 0.0f, 1.0f};
    quad.pose.position = {0.0f, 0.0f, -2.0f};
    quad.size = {2.4f, 2.4f * 9.0f / 16.0f};
    layers[e] = reinterpret_cast<const XrCompositionLayerBaseHeader*>(&quad);
  }

  XrFrameEndInfo end_info{XR_TYPE_FRAME_END_INFO};
  end_info.displayTime = s_predicted_display_time;
  end_info.environmentBlendMode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE;
  end_info.layerCount = s_should_submit ? kEyeCount : 0u;
  end_info.layers = s_should_submit ? layers : nullptr;
  s_last_endframe = xrEndFrame(s_session, &end_info);

  s_frame_begun = false;

  // Periodic loop diagnostic: if the frame counter advances, the loop is NOT
  // hung; state shows whether we've reached VISIBLE/FOCUSED; endFrame shows if
  // submission is rejected.
  if ((++s_frame_counter % 90) == 0)
  {
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "FRAME #%u: sessionState=%d running=%d shouldRender=%d submitted=%d endFrame=%d",
                  s_frame_counter, static_cast<int>(s_session_state), s_session_running.load() ? 1 : 0,
                  s_last_should_render, s_should_submit ? 1 : 0, static_cast<int>(s_last_endframe));
    FrameResult(buf);
  }
}

void SetBlitCallback(VRBlitCallback cb)
{
  s_blit_cb = cb;
}

bool GetHeadPose(float& yaw_rad, float& pitch_rad)
{
  if (!s_session_running.load() || !s_pose_valid.load())
    return false;
  yaw_rad = s_head_yaw.load();
  pitch_rad = s_head_pitch.load();
  return true;
}

// ---- OpenXR <-> Vulkan binding queries (VROpenXR_Vulkan.h) -------------------

const std::vector<std::string>& GetRequiredVulkanInstanceExtensions()
{
  return s_vk_instance_exts;
}

const std::vector<std::string>& GetRequiredVulkanDeviceExtensions()
{
  return s_vk_device_exts;
}

bool CheckVulkanGraphicsRequirements(u32 vk_api_version)
{
  if (s_xrGetVulkanGraphicsRequirementsKHR == nullptr || s_system_id == XR_NULL_SYSTEM_ID)
    return true;

  XrGraphicsRequirementsVulkanKHR req{XR_TYPE_GRAPHICS_REQUIREMENTS_VULKAN_KHR};
  if (XR_FAILED(s_xrGetVulkanGraphicsRequirementsKHR(s_instance, s_system_id, &req)))
    return false;

  const XrVersion have = XR_MAKE_VERSION(VK_VERSION_MAJOR(vk_api_version),
                                         VK_VERSION_MINOR(vk_api_version), 0);
  const bool ok = have >= req.minApiVersionSupported && have <= req.maxApiVersionSupported;
  NOTICE_LOG_FMT(VIDEO,
                 "MHTriVR/OpenXR: Vulkan req min={}.{} max={}.{}, Dolphin={}.{} -> {}",
                 XR_VERSION_MAJOR(req.minApiVersionSupported),
                 XR_VERSION_MINOR(req.minApiVersionSupported),
                 XR_VERSION_MAJOR(req.maxApiVersionSupported),
                 XR_VERSION_MINOR(req.maxApiVersionSupported), VK_VERSION_MAJOR(vk_api_version),
                 VK_VERSION_MINOR(vk_api_version), ok ? "OK" : "OUT-OF-RANGE");
  return ok;
}

VkPhysicalDevice GetVulkanGraphicsDevice(VkInstance vk_instance)
{
  if (s_xrGetVulkanGraphicsDeviceKHR == nullptr || s_system_id == XR_NULL_SYSTEM_ID)
    return VK_NULL_HANDLE;

  VkPhysicalDevice pd = VK_NULL_HANDLE;
  if (XR_FAILED(s_xrGetVulkanGraphicsDeviceKHR(s_instance, s_system_id, vk_instance, &pd)))
  {
    ERROR_LOG_FMT(VIDEO, "MHTriVR/OpenXR: xrGetVulkanGraphicsDeviceKHR failed");
    return VK_NULL_HANDLE;
  }
  return pd;
}

void SetVulkanBinding(VkInstance instance, VkPhysicalDevice physical_device, VkDevice device,
                      u32 queue_family_index, u32 queue_index)
{
  s_vk_instance = instance;
  s_vk_physical_device = physical_device;
  s_vk_device = device;
  s_vk_queue_family = queue_family_index;
  s_vk_queue_index = queue_index;
  // Session creation consumes these in a later step.

  if (s_system_id != XR_NULL_SYSTEM_ID)
  {
    NOTICE_LOG_FMT(VIDEO,
                   "MHTriVR/OpenXR: Vulkan binding captured (qfamily={}, qindex={}). Dolphin's "
                   "Vulkan device is now OpenXR-compatible; session creation is the next step.",
                   queue_family_index, queue_index);
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "BIND OK: Dolphin Vulkan device bound to OpenXR (instExts=%zu devExts=%zu, "
                  "qfamily=%u). Game still rendering = Step 2 success. Session next.",
                  s_vk_instance_exts.size(), s_vk_device_exts.size(), queue_family_index);
    BindResult(buf);

    // Now that the device is bound, create the session + swapchains.
    CreateSessionInternal();
  }
  else
  {
    BindResult("BIND SKIPPED: OpenXR was not active when the Vulkan device was created "
               "(headset/runtime not ready at emulation start). Retry should fix this.");
  }
}
}  // namespace VROpenXR

#else  // !MHTRI_VR_OPENXR — default build: inert facade, no OpenXR dependency.

namespace VROpenXR
{
bool Initialize()
{
  return false;
}
void Shutdown()
{
}
bool IsInitialized()
{
  return false;
}
bool IsActive()
{
  return false;
}
void SetFovScale(float /*scale*/)
{
}
float GetFovScale()
{
  return 1.0f;
}
void RunFrame(const AbstractTexture* /*xfb_stereo_array*/)
{
}
void EndFrame()
{
}
bool GetHeadPose(float& /*yaw_rad*/, float& /*pitch_rad*/)
{
  return false;
}
void SetBlitCallback(VRBlitCallback /*cb*/)
{
}

const std::vector<std::string>& GetRequiredVulkanInstanceExtensions()
{
  static const std::vector<std::string> empty;
  return empty;
}
const std::vector<std::string>& GetRequiredVulkanDeviceExtensions()
{
  static const std::vector<std::string> empty;
  return empty;
}
bool CheckVulkanGraphicsRequirements(u32 /*vk_api_version*/)
{
  return true;
}
VkPhysicalDevice GetVulkanGraphicsDevice(VkInstance /*vk_instance*/)
{
  return VK_NULL_HANDLE;
}
void SetVulkanBinding(VkInstance /*instance*/, VkPhysicalDevice /*physical_device*/,
                      VkDevice /*device*/, u32 /*queue_family_index*/, u32 /*queue_index*/)
{
}
}  // namespace VROpenXR

#endif  // MHTRI_VR_OPENXR
