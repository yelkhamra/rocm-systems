// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_CDNA3_TARGET_PROVIDER_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_CDNA3_TARGET_PROVIDER_H_

#include "rocjitsu/isa/target_registry.h"

#include <array>

namespace rocjitsu::cdna3 {

inline constexpr std::array<std::string_view, 1> target_aliases{"gfx942"};
inline constexpr std::array<rj_code_arch_t, 1> target_architecture_ids{ROCJITSU_CODE_ARCH_CDNA3};
inline constexpr std::array<rj_code_target_id_t, 1> target_gpu_target_ids{
    ROCJITSU_CODE_TARGET_GFX942};
inline constexpr IsaTargetDescription target_description{
    .id = "cdna3",
    .aliases = target_aliases,
    .architecture_ids = target_architecture_ids,
    .gpu_target_ids = target_gpu_target_ids,
    .capabilities = IsaTargetCapability::Model | IsaTargetCapability::Execution,
};

void register_target(IsaTargetRegistry &registry);

} // namespace rocjitsu::cdna3

#endif // ROCJITSU_ISA_ARCH_AMDGPU_CDNA3_TARGET_PROVIDER_H_

#ifdef ROCJITSU_GET_ISA_TARGET_REGISTRATION
ROCJITSU_GET_ISA_TARGET_REGISTRATION(rocjitsu::cdna3::register_target)
#endif
