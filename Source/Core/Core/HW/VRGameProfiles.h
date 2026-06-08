// Copyright 2026 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// VR game profiles — the per-title data the generic VR camera injector needs.
//
// The VR plumbing is game-agnostic: OpenXR session sharing, per-eye stereo, the
// compositor quad, head tracking, and Touch->Classic controller input all live
// in VideoCommon/VROpenXR and know nothing about any particular game. The ONLY
// game-specific knowledge is:
//   1. where the engine commits the final camera (so we can overwrite eye+target)
//   2. how to read the player's world position and facing (for the first-person rig)
// Both of those, for every supported title, live in the table below.
//
// To add a game you do NOT touch any .cpp. You:
//   * find its camera-commit hook (the function that pushes the final eye + look-at
//     into the GPU/scene-graph camera each frame) and which GPRs hold the eye and
//     target vec3 pointers at its entry, and
//   * find its player-state layout: a fixed RAM word that points at the live player
//     struct, plus the offsets of the position vec3 and the facing yaw inside it,
// then add one GameProfile row here and rebuild.
//
// Full procedure (how to locate these with Ghidra + the Dolphin RE MCP) is in
// docs/ADDING_A_GAME.md in the mhtri-vr repo.

#pragma once

#include <string_view>

#include "Common/CommonTypes.h"

namespace VR
{
struct GameProfile
{
  std::string_view game_id;  // 6-char Game ID from the disc header, e.g. "RMHE08"
  std::string_view name;     // human-readable, for logs only

  // --- Camera commit hook -----------------------------------------------------
  // Address of the engine function that pushes the final eye + look-at into the
  // renderer's camera each frame. We hook its entry (HookType::Start) and rewrite
  // the two vec3s it is about to commit. Hooking the commit (rather than the
  // camera-update logic) means we don't have to neuter the game's own camera code.
  u32 hook_addr;
  u8 eye_ptr_reg;     // GPR index holding the eye    vec3 pointer at hook entry
  u8 target_ptr_reg;  // GPR index holding the target vec3 pointer at hook entry

  // --- Player state (first-person rig) ----------------------------------------
  // player_work_ptr is a FIXED RAM address whose contents are a pointer to the
  // live player struct (may resolve into MEM1 0x80xxxxxx or MEM2 0x90xxxxxx). From
  // that struct base: pos_off is a big-endian {X, Y(up), Z} f32 vec3, and
  // facing_off is a signed 16-bit BAMS yaw (degrees = value * 360 / 65536).
  u32 player_work_ptr;
  u32 pos_off;
  u32 facing_off;
};

// The supported-title table. MH Tri is the reference profile; add new games as
// additional rows. Order doesn't matter — lookup is by Game ID.
inline constexpr GameProfile kProfiles[] = {
    {
        "RMHE08",
        "Monster Hunter Tri (NTSC-U)",
        0x802be0f4,  // cam_commit_to_g3d (SetPosture/SetPerspective)
        4,           // r4 = eye    vec3 ptr
        5,           // r5 = target vec3 ptr
        0x806bbc74,  // g_aggregate_p1_ptr -> player_work
        0x48,        // player_work + 0x48 = pos vec3 {X,Y,Z}
        0xd8,        // player_work + 0xD8 = s16 facing yaw (BAMS)
    },
};

// Returns the profile for a Game ID, or nullptr if the title isn't supported.
inline const GameProfile* FindProfile(std::string_view game_id)
{
  for (const auto& p : kProfiles)
  {
    if (p.game_id == game_id)
      return &p;
  }
  return nullptr;
}
}  // namespace VR
