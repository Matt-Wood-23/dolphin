// Copyright 2026 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// VR camera injection — the game-specific half of the VR mod.
//
// Despite the historical name, this module is multi-game: it drives whatever
// title has a profile in Core/HW/VRGameProfiles.h. It rewrites the camera's
// eye/look-target each frame so head movement (and, in first-person mode, the
// player's own position/facing) steers the in-game view. We hook the engine's
// camera COMMIT — the function that pushes eye+target into the GPU/scene-graph
// camera the renderer reads — and overwrite the two vec3s it is about to commit
// (hooking the commit, not the camera update, lets the game's camera code run
// untouched). The hook address and the GPRs holding the eye/target pointers are
// per-game and come from the active profile.
//
// Reference title MH Tri (RMHE08): hook 0x802be0f4 = cam_commit_to_g3d, r4 = eye
// vec3 ptr, r5 = target vec3 ptr. See facts/01_memory_map.md "Camera update
// pipeline & VR injection point" and reports/vr_mod_plan.md in mhtri_reversing,
// and docs/ADDING_A_GAME.md in the mhtri-vr repo for porting to a new game.
//
// This is the camera half of the VR plan; stereo rendering and OpenXR
// presentation are separate phases. The pose source is pluggable: an OpenXR
// tracker (future) calls SetHeadPose(); until then a self-contained Win32
// relative-mouse poll (hold Left Alt) drives the look live, with an optional
// yaw oscillator (MHTRI_VR_DEMO=1) as a no-input smoke test.

#pragma once

namespace Core
{
class CPUThreadGuard;
class System;
}  // namespace Core

namespace MHTriVR
{
// Master enable. Also gated on game ID (RMHE08) at install time.
void SetEnabled(bool enabled);
bool IsEnabled();

// Absolute head-look offset, radians. yaw = look left(-)/right(+), pitch =
// down(-)/up(+). Calling this marks an external pose source as authoritative,
// taking priority over the built-in mouse/oscillator drivers.
void SetHeadPose(float yaw_rad, float pitch_rad);

// HLE hook body. Installed as a Fixed+Start hook at 0x802be0f4.
void CamPostHook(const Core::CPUThreadGuard& guard);

// Install the fixed-address camera hook iff the running game is MH Tri (RMHE08).
// No-op for any other title. Called from HLE::PatchFixedFunctions.
void TryInstall(Core::System& system);
}  // namespace MHTriVR
