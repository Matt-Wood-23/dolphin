// Copyright 2026 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "VideoCommon/VRStereo.h"

#include <atomic>

namespace VRStereo
{
namespace
{
std::atomic<bool> s_enabled{false};
std::atomic<float> s_depth{0.02f};       // matches Dolphin's default stereo_depth
std::atomic<float> s_convergence{20.0f};  // matches Dolphin's default convergence
}  // namespace

void SetEnabled(bool enabled)
{
  s_enabled.store(enabled, std::memory_order_relaxed);
}

bool IsEnabled()
{
  return s_enabled.load(std::memory_order_relaxed);
}

void SetDepth(float depth)
{
  s_depth.store(depth, std::memory_order_relaxed);
}

float GetDepth()
{
  return s_depth.load(std::memory_order_relaxed);
}

void SetConvergence(float convergence)
{
  s_convergence.store(convergence, std::memory_order_relaxed);
}

float GetConvergence()
{
  return s_convergence.load(std::memory_order_relaxed);
}
}  // namespace VRStereo
