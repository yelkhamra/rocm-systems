// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_TESTS_FIXTURES_DOWNSTREAM_ISA_TARGET_PROVIDER_H_
#define ROCJITSU_TESTS_FIXTURES_DOWNSTREAM_ISA_TARGET_PROVIDER_H_

#include "rocjitsu/isa/target_registry.h"

#include <array>

namespace rocjitsu::test {

inline constexpr std::array<rj_code_arch_t, 1> downstream_architecture_ids{
    ROCJITSU_CODE_ARCH_RESERVED_0};
inline constexpr std::array<rj_code_target_id_t, 1> downstream_gpu_target_ids{
    ROCJITSU_CODE_TARGET_RESERVED_0};
inline constexpr IsaTargetDescription downstream_target_description{
    .id = "vendor-downstream-test",
    .architecture_ids = downstream_architecture_ids,
    .gpu_target_ids = downstream_gpu_target_ids,
};

void register_downstream_target(IsaTargetRegistry &registry);

} // namespace rocjitsu::test

#endif // ROCJITSU_TESTS_FIXTURES_DOWNSTREAM_ISA_TARGET_PROVIDER_H_

#ifdef ROCJITSU_GET_ISA_TARGET_REGISTRATION
ROCJITSU_GET_ISA_TARGET_REGISTRATION(rocjitsu::test::register_downstream_target)
#endif
