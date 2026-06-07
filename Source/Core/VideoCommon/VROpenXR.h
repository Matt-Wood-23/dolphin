// Copyright 2026 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// Monster Hunter Tri VR — Phase 3 (HMD presentation) facade.
//
// Thin, backend-agnostic surface the rest of the engine talks to. The real
// OpenXR session + Vulkan swapchain interop is gated behind the MHTRI_VR_OPENXR
// compile flag, which the DEFAULT BUILD DOES NOT DEFINE. With the flag off this
// compiles to inert no-ops with no OpenXR loader dependency, so the working
// monitor build (Phases 1-2: mouse head-look + VRStereo) is untouched.
//
// Frame model (real path, future): at frame start BeginFrame() runs
// xrWaitFrame/xrBeginFrame and polls the predicted head pose, feeding
// MHTriVR::SetHeadPose() and the IPD into VRStereo. At frame end SubmitStereo()
// takes the final XFB — a 2-layer texture array (layer 0 = left eye, layer 1 =
// right eye) produced by Dolphin's geometry-shader stereo — and copies each
// layer into the per-eye OpenXR swapchain images, then xrEndFrame submits a
// projection layer. Hook site for SubmitStereo is Presenter::Present().
//
// Turning the real path on additionally requires vendoring the OpenXR loader
// into Externals and adding it to the build. See reports/vr_mod_plan.md Phase 3.

#pragma once

class AbstractTexture;

namespace VROpenXR
{
// Bring up the OpenXR instance and detect the HMD (logs runtime + system name).
// Smoke-test milestone: this does NOT yet create a session/swapchains, so
// IsActive() stays false and the render path remains inert. Returns true if an
// HMD was found. No-op returning false in the default (flag-off) build.
bool Initialize();

// Tear down the OpenXR instance/session. No-op in the default build.
void Shutdown();

// True once the OpenXR instance + HMD system are up (after Initialize), even
// before a session is running. Used to make backend setup decisions (e.g. force
// single-threaded Vulkan submission to avoid racing the XR compositor on the
// queue). Always false in the default build.
bool IsInitialized();

// True only while a real OpenXR session is running. Always false in the default
// (flag-off) build, so all call sites stay inert.
bool IsActive();

// VR FOV scale applied to the game's perspective projection so the rendered view
// roughly fills the HMD's wide FOV (values < 1 widen the FOV / reduce the
// "zoomed in" look). 1.0 = unchanged. Set from MHTRI_VR_FOV_SCALE; read by
// VertexShaderManager::LoadProjectionMatrix while a session is active.
void SetFovScale(float scale);
float GetFovScale();

// Drive the per-frame OpenXR loop: poll the session state machine, wait/begin the
// frame, locate the head pose (stored for GetHeadPose), acquire the eye swapchain
// images and record the XFB blit into them. Called once per frame from
// Presenter::Present BEFORE PresentBackbuffer (so the blit lands on the command
// buffer Dolphin is about to submit). No-op in the default build / no session.
void RunFrame(const AbstractTexture* xfb_stereo_array);

// Finish the OpenXR frame: release the eye swapchain images and xrEndFrame with
// the stereo projection layer. Called from Presenter::Present AFTER
// PresentBackbuffer, so the recorded blit has been submitted to the queue.
void EndFrame();

// Latest HMD head orientation as game-space yaw/pitch (radians). Returns true and
// fills the args while a VR session is delivering tracking; false otherwise (so
// callers can fall back to mouse). Called from the camera hook on the CPU thread.
bool GetHeadPose(float& yaw_rad, float& pitch_rad);
}  // namespace VROpenXR
