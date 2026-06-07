// Copyright 2026 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "Core/HW/MHTriVR.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdlib>

#include "Common/CommonTypes.h"
#include "Common/Logging/Log.h"

#include "Core/ConfigManager.h"
#include "Core/HLE/HLE.h"
#include "Core/PowerPC/MMU.h"
#include "Core/PowerPC/PowerPC.h"
#include "Core/System.h"

#include "VideoCommon/VROpenXR.h"
#include "VideoCommon/VRStereo.h"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#endif

namespace MHTriVR
{
namespace
{
constexpr float kPi = 3.14159265358979323846f;

// Hook site (NTSC-U / RMHE08). FUN_802be0f4 is the camera COMMIT: it pushes
// eye+target into the nw4r g3d Camera (SetPosture/SetPerspective) that the
// renderer reads. Writing g_cam_work directly is too late because the commit
// already happened. At entry: r4 = eye vec3 ptr, r5 = target vec3 ptr (args).
// See facts/01_memory_map.md.
constexpr u32 kHookAddr = 0x802be0f4;   // cam commit (SetPosture/SetPerspective)
constexpr u32 kEyePtrReg = 4;           // r4 = eye vec3 pointer
constexpr u32 kTargetPtrReg = 5;        // r5 = target (look-at) vec3 pointer

std::atomic<bool> s_enabled{true};
std::atomic<float> s_yaw{0.0f};
std::atomic<float> s_pitch{0.0f};
std::atomic<bool> s_have_external_pose{false};  // OpenXR/SetHeadPose authoritative
std::atomic<bool> s_mouse{true};   // Win32 relative-mouse look (hold Left Alt)
std::atomic<bool> s_demo{false};   // env-gated yaw oscillator (no-input smoke test)
u32 s_frames = 0;  // oscillator phase (hook thread only)

// Self-contained mouse-look: while the engage key (Left Alt) is held, relative
// mouse motion accumulates into the head yaw/pitch and the cursor is recentered
// each frame so it never clips a screen edge. A stand-in for the eventual
// OpenXR head pose; proves live 1:1 control of the camera. Windows-only (the VR
// target runtime). Updates s_yaw/s_pitch in place — read by CamPostHook.
void PollMouseLook()
{
#ifdef _WIN32
  static bool s_was_engaged = false;
  static POINT s_anchor{};

  const bool engaged = (GetAsyncKeyState(VK_MENU) & 0x8000) != 0;  // Left/Right Alt
  if (engaged)
  {
    POINT p{};
    if (!s_was_engaged)
      GetCursorPos(&s_anchor);  // latch anchor at engage; measure deltas from it
    GetCursorPos(&p);
    const int dx = p.x - s_anchor.x;
    const int dy = p.y - s_anchor.y;
    if (dx != 0 || dy != 0)
    {
      constexpr float kSens = 0.0025f;  // rad per pixel
      float yaw = s_yaw.load() + static_cast<float>(dx) * kSens;
      float pitch = s_pitch.load() - static_cast<float>(dy) * kSens;  // up = look up
      if (yaw > kPi)
        yaw -= 2.0f * kPi;
      else if (yaw < -kPi)
        yaw += 2.0f * kPi;
      pitch = std::clamp(pitch, -1.4f, 1.4f);  // ~+/-80 deg
      s_yaw.store(yaw);
      s_pitch.store(pitch);
      s_have_external_pose.store(true);
      SetCursorPos(s_anchor.x, s_anchor.y);  // recenter for the next frame's delta
    }
  }
  s_was_engaged = engaged;
#endif
}
}  // namespace

void SetEnabled(bool enabled)
{
  s_enabled.store(enabled);
}

bool IsEnabled()
{
  return s_enabled.load();
}

void SetHeadPose(float yaw_rad, float pitch_rad)
{
  s_yaw.store(yaw_rad);
  s_pitch.store(pitch_rad);
  s_have_external_pose.store(true);
}

void CamPostHook(const Core::CPUThreadGuard& guard)
{
  if (!s_enabled.load())
    return;

  // Refresh the head pose from the live input driver (mouse) before applying it.
  // An external OpenXR pose, if set, still wins below.
  if (s_mouse.load())
    PollMouseLook();

  auto& system = guard.GetSystem();
  const auto& ppc_state = system.GetPPCState();

  // Args at the commit: r4 = eye ptr, r5 = target ptr.
  const u32 eye_ptr = ppc_state.gpr[kEyePtrReg];
  const u32 tgt_ptr = ppc_state.gpr[kTargetPtrReg];
  const auto in_mem1 = [](u32 a) { return a >= 0x80000000 && a < 0x81800000; };
  if (!in_mem1(eye_ptr) || !in_mem1(tgt_ptr))
    return;

  // Determine the head-look offset. Priority: live OpenXR HMD tracking > mouse-look
  // / external pose (s_have_external_pose) > the env-gated oscillator > identity.
  float yaw, pitch;
  if (VROpenXR::GetHeadPose(yaw, pitch))
  {
    // Real headset orientation drives the in-game camera (Phase 3b).
  }
  else if (s_have_external_pose.load())
  {
    yaw = s_yaw.load();
    pitch = s_pitch.load();
  }
  else if (s_demo.load())
  {
    ++s_frames;
    yaw = 0.5f * std::sin(static_cast<float>(s_frames) * 0.02f);  // +/-0.5 rad pan
    pitch = 0.0f;
  }
  else
  {
    return;  // safe default: leave the game camera untouched
  }
  if (yaw == 0.0f && pitch == 0.0f)
    return;

  const auto rd = [&guard](u32 addr) {
    return std::bit_cast<float>(PowerPC::MMU::HostRead<u32>(guard, addr));
  };
  const auto wr = [&guard](u32 addr, float v) {
    PowerPC::MMU::HostWrite<u32>(guard, std::bit_cast<u32>(v), addr);
  };

  // Eye/target being committed (big-endian f32 vec3s).
  const float ex = rd(eye_ptr + 0);
  const float ey = rd(eye_ptr + 4);
  const float ez = rd(eye_ptr + 8);
  const float tx = rd(tgt_ptr + 0);
  const float ty = rd(tgt_ptr + 4);
  const float tz = rd(tgt_ptr + 8);

  // Rotate the look vector (target - eye) about the eye: yaw about world Y,
  // pitch about the horizontal right axis. Eye stays put => head-look (Mode A).
  // Recomputed from the game's fresh eye/target each frame, so no drift.
  float dx = tx - ex;
  float dy = ty - ey;
  float dz = tz - ez;

  const float cy = std::cos(yaw);
  const float sy = std::sin(yaw);
  const float rx = dx * cy - dz * sy;
  const float rz = dx * sy + dz * cy;
  dx = rx;
  dz = rz;

  const float horiz = std::sqrt(dx * dx + dz * dz);
  if (horiz > 1.0e-4f)
  {
    const float cp = std::cos(pitch);
    const float sp = std::sin(pitch);
    const float new_horiz = horiz * cp - dy * sp;
    const float new_y = horiz * sp + dy * cp;
    const float scale = new_horiz / horiz;
    dx *= scale;
    dz *= scale;
    dy = new_y;
  }

  wr(tgt_ptr + 0, ex + dx);
  wr(tgt_ptr + 4, ey + dy);
  wr(tgt_ptr + 8, ez + dz);
}

void TryInstall(Core::System& system)
{
  const std::string game_id = SConfig::GetInstance().GetGameID();
  if (game_id != "RMHE08")
    return;

  // Opt-out: MHTRI_VR=0 skips installation entirely.
  if (const char* off = std::getenv("MHTRI_VR"); off != nullptr && off[0] == '0')
    return;

  // Live driver: Win32 relative-mouse look (hold Left Alt). Default ON; disable
  // with MHTRI_VR_MOUSE=0 (e.g. once an OpenXR pose source drives SetHeadPose()).
  if (const char* m = std::getenv("MHTRI_VR_MOUSE"); m != nullptr && m[0] == '0')
    s_mouse.store(false);

  // No-input smoke test: a self-panning yaw oscillator. Opt-in via
  // MHTRI_VR_DEMO=1 (only used when no external pose has been set).
  if (const char* demo = std::getenv("MHTRI_VR_DEMO"); demo != nullptr && demo[0] == '1')
    s_demo.store(true);

  // Phase 2 (stereo): take ownership of the per-eye shear params. Inert until a
  // Dolphin stereo mode (e.g. Side-by-Side) is enabled. Default ON; opt out with
  // MHTRI_VR_STEREO=0. Optional tuning via MHTRI_VR_DEPTH / MHTRI_VR_CONVERGENCE.
  const char* stereo_off = std::getenv("MHTRI_VR_STEREO");
  const bool stereo_on = !(stereo_off != nullptr && stereo_off[0] == '0');
  VRStereo::SetEnabled(stereo_on);
  if (const char* d = std::getenv("MHTRI_VR_DEPTH"); d != nullptr)
  {
    if (const float v = std::strtof(d, nullptr); v > 0.0f)
      VRStereo::SetDepth(v);
  }
  if (const char* c = std::getenv("MHTRI_VR_CONVERGENCE"); c != nullptr)
  {
    if (const float v = std::strtof(c, nullptr); v > 0.0f)
      VRStereo::SetConvergence(v);
  }

  // VR FOV widen (combats the "zoomed in" HMD look). <1 widens; tune live.
  if (const char* fov = std::getenv("MHTRI_VR_FOV_SCALE"); fov != nullptr)
  {
    if (const float v = std::strtof(fov, nullptr); v > 0.0f)
      VROpenXR::SetFovScale(v);
  }

  // Fixed-address Start hook: runs CamPostHook, then the original instruction at
  // kHookAddr (li r0,0xff) executes normally and the function continues.
  // Phase 3 (OpenXR): inert no-op in the default build; in a MHTriVROpenXR build
  // this brings up the OpenXR instance and logs whether the HMD is detected.
  VROpenXR::Initialize();

  HLE::Patch(system, kHookAddr, "MHTriVRCamPost");
  INFO_LOG_FMT(CORE,
               "MHTriVR: installed camera VR hook at {:#010x} (RMHE08) "
               "[mouse={}, demo={}, stereo={} depth={} conv={}]",
               kHookAddr, s_mouse.load(), s_demo.load(), VRStereo::IsEnabled(),
               VRStereo::GetDepth(), VRStereo::GetConvergence());
}
}  // namespace MHTriVR
