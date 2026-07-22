// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_GFX1250_TARGET_PROVIDER_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_GFX1250_TARGET_PROVIDER_H_

#include "rocjitsu/isa/target_registry.h"

#include <array>

namespace rocjitsu::gfx1250 {

inline constexpr std::array<rj_code_arch_t, 1> target_architecture_ids{ROCJITSU_CODE_ARCH_GFX1250};
inline constexpr std::array<rj_code_target_id_t, 1> target_gpu_target_ids{
    ROCJITSU_CODE_TARGET_GFX1250};
/// Full execution alternative; do not combine it with the model-only provider
/// in the same registry.
inline constexpr IsaTargetDescription target_description{
    .id = "gfx1250",
    .architecture_ids = target_architecture_ids,
    .gpu_target_ids = target_gpu_target_ids,
    .capabilities = IsaTargetCapability::Model | IsaTargetCapability::Execution,
};

void register_target(IsaTargetRegistry &registry);

} // namespace rocjitsu::gfx1250

#endif // ROCJITSU_ISA_ARCH_AMDGPU_GFX1250_TARGET_PROVIDER_H_

#ifdef ROCJITSU_GET_ISA_TARGET_REGISTRATION
ROCJITSU_GET_ISA_TARGET_REGISTRATION(rocjitsu::gfx1250::register_target)
#endif
