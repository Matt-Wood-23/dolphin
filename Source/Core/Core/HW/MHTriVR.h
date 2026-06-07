// Copyright 2026 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// Monster Hunter Tri (RMHE08) VR camera injection — Phase 1 (head-look).
//
// Rewrites the camera's look target each frame so head movement steers the
// in-game view. Hooked at 0x802be0f4 — cam_commit_to_g3d, the camera COMMIT
// that pushes eye+target into the nw4r g3d Camera (SetPosture/SetPerspective)
// that the renderer actually reads. At entry r4 = eye vec3 ptr, r5 = target
// vec3 ptr; the hook rotates *target about *eye by the current head yaw/pitch.
// (Writing g_cam_work is a frame too late — the commit already happened.)
// See facts/01_memory_map.md "Camera update pipeline & VR injection point" and
// reports/vr_mod_plan.md in the mhtri_reversing project.
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
