// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file target_registry_composition.cpp
/// @brief Checked-in implementation shared by static registry compositions.

#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/target_registry.h"

#ifndef RJ_ISA_TARGET_HEADERS
#error "RJ_ISA_TARGET_HEADERS must name a generated provider-header list"
#endif

#ifndef RJ_ISA_TARGET_REGISTRY_ACCESSOR
#error "RJ_ISA_TARGET_REGISTRY_ACCESSOR must name the registry accessor"
#endif

#include RJ_ISA_TARGET_HEADERS

namespace rocjitsu {

const IsaTargetRegistry &RJ_ISA_TARGET_REGISTRY_ACCESSOR() {
  static const IsaTargetRegistry registry = [] {
    IsaTargetRegistry registry;
#define ROCJITSU_GET_ISA_TARGET_REGISTRATION(provider) provider(registry);
#include RJ_ISA_TARGET_HEADERS
#undef ROCJITSU_GET_ISA_TARGET_REGISTRATION
    registry.freeze();
    return registry;
  }();
  return registry;
}

#ifdef RJ_ISA_TARGET_REGISTRY_DEFAULT
const IsaTargetRegistry &default_isa_target_registry() { return RJ_ISA_TARGET_REGISTRY_ACCESSOR(); }

std::unique_ptr<Decoder> Decoder::create(rj_code_arch_t arch) {
  return create(default_isa_target_registry(), arch);
}
#endif

} // namespace rocjitsu
