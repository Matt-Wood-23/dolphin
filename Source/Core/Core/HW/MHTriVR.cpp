// Copyright 2026 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "Core/HW/MHTriVR.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "Common/CommonTypes.h"
#include "Common/FileUtil.h"
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

// --- First-person (Mode B) rig -------------------------------------------------
// Instead of head-look on the 3rd-person follow cam, place the rendered eye AT the
// player's head and aim it with (body facing + HMD head yaw/pitch). This makes the
// game first-person — the prerequisite for true wrap-around VR immersion. We don't
// need to neuter cam_follow_update: we just overwrite the eye/target the commit is
// about to push. Tunables live in mhtri_vr_config.txt (re-read live; no rebuild).
//
//   PLW (player_work) ptr  @ 0x806BBC74 (g_aggregate_p1_ptr, cached by the cam)
//   PLW+0x48 = pos vec3 {X, Y(up), Z}   (live-verified, facts/01)
//   PLW+0xD8 = s16 facing yaw, BAMS     (deg = val*360/65536)
std::atomic<bool> s_fp{false};            // first-person mode on/off
std::atomic<float> s_fp_eye_height{150.0f};  // world units added to player Y for the eye
std::atomic<float> s_fp_eye_forward{0.0f};   // push eye forward along view (out of the model)
std::atomic<float> s_fp_look_dist{300.0f};   // eye->target length (direction only matters)
std::atomic<float> s_fp_yaw_offset{0.0f};    // radians added to body facing (tune forward)
std::atomic<float> s_fp_yaw_sign{1.0f};      // +1/-1 flips facing rotation sense
// Snap-recenter (Right Ctrl): a runtime yaw offset that re-aims the world-locked
// view at the hunter's CURRENT facing. Kept separate from s_fp_yaw_offset so the
// live config reload can't clobber it. Calibration-free: we snap to the game's own
// follow-cam azimuth (read at hook entry), which is in our world coordinates.
std::atomic<float> s_fp_recenter{0.0f};
// If the follow-cam direction points opposite the way you want to face, flip the
// recenter/follow-base target 180° (live config `recenter_flip=1`).
std::atomic<bool> s_fp_recenter_flip{false};
// Follow-base mode (live `fp_follow_base=1`): the FP view YAW tracks the game's
// follow-cam azimuth + head, instead of a fixed world offset. The follow-cam yaw is
// driven by the RIGHT STICK (and is independent of body facing), so the right stick
// turns the view — VR-style camera turn — with no spin. Supersedes recenter when on.
std::atomic<bool> s_fp_follow_base{false};
std::atomic<float> s_fp_head_yaw_sign{-1.0f};  // +1/-1 flips HMD yaw sense (Mode-A match)
std::atomic<float> s_fp_head_pitch_sign{1.0f};  // +1/-1 flips HMD pitch sense
constexpr u32 kPlayerWorkPtr = 0x806BBC74;   // -> player_work base
constexpr u32 kPlwPosOff = 0x48;             // vec3 current position
constexpr u32 kPlwFacingOff = 0xD8;          // s16 facing yaw (BAMS)

// Live config (shared with VROpenXR's reader): mhtri_vr_config.txt next to the
// Dolphin executable. Re-read every N frames from the CPU thread so eye-height /
// look direction can be tuned in-headset on the fly.
const std::string& ConfigPath()
{
  static const std::string path = File::GetExeDirectory() + "/mhtri_vr_config.txt";
  return path;
}

void ReadFPConfig()
{
  std::FILE* f = std::fopen(ConfigPath().c_str(), "rb");
  if (f == nullptr)
    return;
  char line[256];
  while (std::fgets(line, sizeof(line), f) != nullptr)
  {
    float v;
    int iv;
    if (std::sscanf(line, "firstperson=%d", &iv) == 1)
      s_fp.store(iv != 0);
    else if (std::sscanf(line, "eye_height=%f", &v) == 1)
      s_fp_eye_height.store(v);
    else if (std::sscanf(line, "eye_forward=%f", &v) == 1)
      s_fp_eye_forward.store(v);
    else if (std::sscanf(line, "look_dist=%f", &v) == 1)
      s_fp_look_dist.store(v);
    else if (std::sscanf(line, "fp_yaw_offset=%f", &v) == 1)
      s_fp_yaw_offset.store(v);
    else if (std::sscanf(line, "fp_yaw_sign=%f", &v) == 1)
      s_fp_yaw_sign.store(v);
    else if (std::sscanf(line, "fp_head_yaw_sign=%f", &v) == 1)
      s_fp_head_yaw_sign.store(v);
    else if (std::sscanf(line, "fp_head_pitch_sign=%f", &v) == 1)
      s_fp_head_pitch_sign.store(v);
    else if (std::sscanf(line, "recenter_flip=%d", &iv) == 1)
      s_fp_recenter_flip.store(iv != 0);
    else if (std::sscanf(line, "fp_follow_base=%d", &iv) == 1)
      s_fp_follow_base.store(iv != 0);
  }
  std::fclose(f);
}

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
  // In first-person mode we still want to place the eye even when the head is
  // centered, so a zero pose is valid there (have_pose just gates Mode-A below).
  float yaw = 0.0f, pitch = 0.0f;
  bool have_pose = false;
  if (VROpenXR::GetHeadPose(yaw, pitch))
  {
    have_pose = true;  // Real headset orientation drives the camera (Phase 3b).
  }
  else if (s_have_external_pose.load())
  {
    yaw = s_yaw.load();
    pitch = s_pitch.load();
    have_pose = true;
  }
  else if (s_demo.load())
  {
    ++s_frames;
    yaw = 0.5f * std::sin(static_cast<float>(s_frames) * 0.02f);  // +/-0.5 rad pan
    pitch = 0.0f;
    have_pose = true;
  }

  const auto rd = [&guard](u32 addr) {
    return std::bit_cast<float>(PowerPC::MMU::HostRead<u32>(guard, addr));
  };
  const auto wr = [&guard](u32 addr, float v) {
    PowerPC::MMU::HostWrite<u32>(guard, std::bit_cast<u32>(v), addr);
  };

  // First-person (Mode B): live-tunable, re-read config every ~48 frames. Eye is
  // placed at the player head; look direction = body facing + HMD head yaw/pitch.
  if ((s_frames++ & 0x2F) == 0)
    ReadFPConfig();
  if (s_fp.load())
  {
    // player_work lives in MEM2 (0x90xxxxxx) on this title, so accept either RAM.
    const auto in_ram = [](u32 a) {
      return (a >= 0x80000000 && a < 0x81800000) || (a >= 0x90000000 && a < 0x94000000);
    };
    const u32 plw = PowerPC::MMU::HostRead<u32>(guard, kPlayerWorkPtr);
    if (in_ram(plw))
    {
      const float px = rd(plw + kPlwPosOff + 0);
      const float py = rd(plw + kPlwPosOff + 4);
      const float pz = rd(plw + kPlwPosOff + 8);
      const auto facing_bams =
          static_cast<s16>(PowerPC::MMU::HostRead<u16>(guard, plw + kPlwFacingOff));
      const float facing = static_cast<float>(facing_bams) * (2.0f * kPi / 65536.0f);
      // HMD yaw sense must match the working Mode-A head-look (azimuth decreases
      // with +head-yaw), hence the per-axis head sign knobs.
      const float head_yaw = yaw * s_fp_head_yaw_sign.load();
      const float head_pitch = pitch * s_fp_head_pitch_sign.load();

      // The follow-cam's look direction = the ORIGINAL eye->target, still in
      // g_cam_work at hook entry (cam_follow_update wrote it just before this
      // commit). Its azimuth is driven by the RIGHT STICK, independent of body
      // facing -> a clean, spin-free heading for both recenter and follow-base.
      const float oex = rd(eye_ptr + 0), oez = rd(eye_ptr + 8);
      const float otx = rd(tgt_ptr + 0), otz = rd(tgt_ptr + 8);
      float az_follow = std::atan2(otx - oex, otz - oez);
      if (s_fp_recenter_flip.load())
        az_follow += kPi;  // face the opposite of the follow-cam direction

      // Snap-recenter (Right Ctrl, rising edge): re-aim the world-locked view at the
      // hunter's current facing. (Unused in follow-base mode, which auto-tracks.)
#ifdef _WIN32
      static bool s_was_recenter = false;
      const bool recenter = (GetAsyncKeyState(VK_RCONTROL) & 0x8000) != 0;
      if (recenter && !s_was_recenter)
        s_fp_recenter.store(az_follow - facing * s_fp_yaw_sign.load() -
                            s_fp_yaw_offset.load() - head_yaw);
      s_was_recenter = recenter;
#endif

      // Follow-base: view yaw tracks the right-stick-driven follow azimuth + head
      // (VR camera turn). Otherwise: fixed world offset + snap-recenter.
      const float total_yaw =
          s_fp_follow_base.load() ?
              (az_follow + s_fp_yaw_offset.load() + head_yaw) :
              (facing * s_fp_yaw_sign.load() + s_fp_yaw_offset.load() +
               s_fp_recenter.load() + head_yaw);
      const float cp = std::cos(head_pitch);
      const float fx = std::sin(total_yaw) * cp;
      const float fy = std::sin(head_pitch);
      const float fz = std::cos(total_yaw) * cp;
      // Push the eye forward along the horizontal view direction so it leaves the
      // player model (otherwise you're rendering from inside the hunter's head).
      const float ef = s_fp_eye_forward.load();
      const float ex = px + std::sin(total_yaw) * ef;
      const float ey = py + s_fp_eye_height.load();
      const float ez = pz + std::cos(total_yaw) * ef;
      const float d = s_fp_look_dist.load();
      wr(eye_ptr + 0, ex);
      wr(eye_ptr + 4, ey);
      wr(eye_ptr + 8, ez);
      wr(tgt_ptr + 0, ex + fx * d);
      wr(tgt_ptr + 4, ey + fy * d);
      wr(tgt_ptr + 8, ez + fz * d);
    }
    return;
  }

  // Mode A (head-look on the follow cam): only when a head pose is present.
  if (!have_pose || (yaw == 0.0f && pitch == 0.0f))
    return;

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

  // First-person (Mode B): MHTRI_VR_FP=1 starts in first person; otherwise it's
  // driven live by `firstperson=` in mhtri_vr_config.txt (re-read each frame).
  if (const char* fp = std::getenv("MHTRI_VR_FP"); fp != nullptr && fp[0] == '1')
    s_fp.store(true);
  // Prime the live config once at install so eye-height etc. are set before frame 0.
  ReadFPConfig();

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
