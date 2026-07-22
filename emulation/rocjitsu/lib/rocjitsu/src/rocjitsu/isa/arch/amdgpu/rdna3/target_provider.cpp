// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/arch/amdgpu/rdna3/target_provider.h"

#include "rocjitsu/isa/arch/amdgpu/rdna3/isa.h"
#include "rocjitsu/isa/target_provider.h"

namespace rocjitsu::rdna3 {

void register_target(IsaTargetRegistry &registry) {
  add_isa_target<Isa>(registry, target_description);
}

} // namespace rocjitsu::rdna3
