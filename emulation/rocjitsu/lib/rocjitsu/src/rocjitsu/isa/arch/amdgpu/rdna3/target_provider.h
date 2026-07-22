// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_RDNA3_TARGET_PROVIDER_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_RDNA3_TARGET_PROVIDER_H_

#include "rocjitsu/isa/target_registry.h"

#include <array>

namespace rocjitsu::rdna3 {

inline constexpr std::array<rj_code_arch_t, 1> target_architecture_ids{ROCJITSU_CODE_ARCH_RDNA3};
inline constexpr IsaTargetDescription target_description{
    .id = "rdna3",
    .architecture_ids = target_architecture_ids,
    .capabilities = IsaTargetCapability::Model | IsaTargetCapability::Execution,
};

void register_target(IsaTargetRegistry &registry);

} // namespace rocjitsu::rdna3

#endif // ROCJITSU_ISA_ARCH_AMDGPU_RDNA3_TARGET_PROVIDER_H_

#ifdef ROCJITSU_GET_ISA_TARGET_REGISTRATION
ROCJITSU_GET_ISA_TARGET_REGISTRATION(rocjitsu::rdna3::register_target)
#endif
