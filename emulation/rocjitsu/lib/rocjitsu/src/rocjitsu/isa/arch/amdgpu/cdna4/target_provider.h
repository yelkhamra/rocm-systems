// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_CDNA4_TARGET_PROVIDER_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_CDNA4_TARGET_PROVIDER_H_

#include "rocjitsu/isa/target_registry.h"

#include <array>

namespace rocjitsu::cdna4 {

inline constexpr std::array<std::string_view, 1> target_aliases{"gfx950"};
inline constexpr std::array<rj_code_arch_t, 1> target_architecture_ids{ROCJITSU_CODE_ARCH_CDNA4};
inline constexpr std::array<rj_code_target_id_t, 1> target_gpu_target_ids{
    ROCJITSU_CODE_TARGET_GFX950};
inline constexpr IsaTargetDescription target_description{
    .id = "cdna4",
    .aliases = target_aliases,
    .architecture_ids = target_architecture_ids,
    .gpu_target_ids = target_gpu_target_ids,
    .capabilities = IsaTargetCapability::Model | IsaTargetCapability::Execution,
};

void register_target(IsaTargetRegistry &registry);

} // namespace rocjitsu::cdna4

#endif // ROCJITSU_ISA_ARCH_AMDGPU_CDNA4_TARGET_PROVIDER_H_

#ifdef ROCJITSU_GET_ISA_TARGET_REGISTRATION
ROCJITSU_GET_ISA_TARGET_REGISTRATION(rocjitsu::cdna4::register_target)
#endif
