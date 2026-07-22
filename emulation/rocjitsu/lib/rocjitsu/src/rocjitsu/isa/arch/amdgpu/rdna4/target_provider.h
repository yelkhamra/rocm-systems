// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_RDNA4_TARGET_PROVIDER_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_RDNA4_TARGET_PROVIDER_H_

#include "rocjitsu/isa/target_registry.h"

#include <array>

namespace rocjitsu::rdna4 {

inline constexpr std::array<std::string_view, 2> target_aliases{"gfx1200", "gfx1201"};
inline constexpr std::array<rj_code_arch_t, 1> target_architecture_ids{ROCJITSU_CODE_ARCH_RDNA4};
inline constexpr std::array<rj_code_target_id_t, 2> target_gpu_target_ids{
    ROCJITSU_CODE_TARGET_GFX1200, ROCJITSU_CODE_TARGET_GFX1201};
inline constexpr IsaTargetDescription target_description{
    .id = "rdna4",
    .aliases = target_aliases,
    .architecture_ids = target_architecture_ids,
    .gpu_target_ids = target_gpu_target_ids,
    .capabilities = IsaTargetCapability::Model | IsaTargetCapability::Execution,
};

void register_target(IsaTargetRegistry &registry);

} // namespace rocjitsu::rdna4

#endif // ROCJITSU_ISA_ARCH_AMDGPU_RDNA4_TARGET_PROVIDER_H_

#ifdef ROCJITSU_GET_ISA_TARGET_REGISTRATION
ROCJITSU_GET_ISA_TARGET_REGISTRATION(rocjitsu::rdna4::register_target)
#endif
