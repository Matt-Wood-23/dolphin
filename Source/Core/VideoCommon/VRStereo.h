// Copyright 2026 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// Monster Hunter Tri VR — Phase 2 (stereo) override.
//
// A tiny, self-contained holder for VR-owned stereoscopy parameters. When
// enabled, GeometryShaderManager::SetConstants drives the per-eye horizontal
// shear (eye separation + convergence) from these values instead of the GUI
// sliders, so the stereo strength is owned by the VR layer and can be tuned /
// later driven from the headset IPD and scene scale.
//
// Lives in VideoCommon (which sits below Core) so the shader-constant code can
// read it without depending on Core; the Core-side MHTriVR module sets it.
//
// Values are in the SAME final units the shader consumes (see
// VideoConfig::stereo_depth / stereo_convergence): depth ~0.02 typical,
// convergence ~20.

#pragma once

namespace VRStereo
{
// Master enable for the VR stereo override. Inert unless a Dolphin stereo mode
// (e.g. Side-by-Side) is also active — SetConstants only touches stereoparams
// when stereo_mode != Off.
void SetEnabled(bool enabled);
bool IsEnabled();

// Eye-separation magnitude (becomes ±stereoparams.x/.y). Typical ~0.02.
void SetDepth(float depth);
float GetDepth();

// Convergence distance in view-depth (w) units (becomes stereoparams.z).
// Objects at this depth sit on the screen plane (zero parallax). Typical ~20.
void SetConvergence(float convergence);
float GetConvergence();
}  // namespace VRStereo
