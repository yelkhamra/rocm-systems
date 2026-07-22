// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file narrow_isa_registry_fixture.cpp
/// @brief Exported observations of a gfx1250-model-only component registry.

#include "rocjitsu/base/rj_compiler.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/target_registry.h"
#include "rocjitsu_gfx1250_model_registry.h"

#include <cstddef>
#include <cstdint>
#include <memory>

extern "C" RJ_API_EXPORT size_t rj_test_narrow_target_count() {
  return rocjitsu::rj_get_gfx1250_model_targets().targets().size();
}

extern "C" RJ_API_EXPORT const char *rj_test_narrow_target_id(size_t index) {
  const auto targets = rocjitsu::rj_get_gfx1250_model_targets().targets();
  return index < targets.size() ? targets[index].id.c_str() : nullptr;
}

extern "C" RJ_API_EXPORT bool rj_test_narrow_has_target(const char *id) {
  return id != nullptr && rocjitsu::rj_get_gfx1250_model_targets().find(id) != nullptr;
}

extern "C" RJ_API_EXPORT bool rj_test_narrow_has_execution() {
  const auto targets = rocjitsu::rj_get_gfx1250_model_targets().targets();
  return targets.size() == 1 && rocjitsu::has_capability(targets[0].capabilities,
                                                         rocjitsu::IsaTargetCapability::Execution);
}

extern "C" RJ_API_EXPORT bool rj_test_narrow_decode_has_execute() {
  constexpr uint32_t kSNop = 0xBF800000u;
  auto decoder = rocjitsu::Decoder::create(rocjitsu::rj_get_gfx1250_model_targets(), "gfx1250");
  if (decoder == nullptr)
    return false;
  std::unique_ptr<rocjitsu::Instruction> instruction(decoder->decode(&kSNop));
  return instruction != nullptr && instruction->execute != nullptr;
}
