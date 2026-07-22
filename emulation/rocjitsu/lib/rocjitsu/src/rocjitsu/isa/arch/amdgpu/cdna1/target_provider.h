// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_CDNA1_TARGET_PROVIDER_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_CDNA1_TARGET_PROVIDER_H_

#include "rocjitsu/isa/target_registry.h"

#include <array>

namespace rocjitsu::cdna1 {

inline constexpr std::array<rj_code_arch_t, 1> target_architecture_ids{ROCJITSU_CODE_ARCH_CDNA1};
inline constexpr IsaTargetDescription target_description{
    .id = "cdna1",
    .architecture_ids = target_architecture_ids,
    .capabilities = IsaTargetCapability::Model | IsaTargetCapability::Execution,
};

void register_target(IsaTargetRegistry &registry);

} // namespace rocjitsu::cdna1

#endif // ROCJITSU_ISA_ARCH_AMDGPU_CDNA1_TARGET_PROVIDER_H_

#ifdef ROCJITSU_GET_ISA_TARGET_REGISTRATION
ROCJITSU_GET_ISA_TARGET_REGISTRATION(rocjitsu::cdna1::register_target)
#endif
