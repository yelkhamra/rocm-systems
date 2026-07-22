# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

rj_add_object_library(rocjitsu_isa
    decoder.cpp
    execution_backend.cpp
    operand.cpp
    register_set.cpp
    rj_decode.cpp
    target_registry.cpp
)

add_subdirectory(arch/amdgpu/cdna1)
add_subdirectory(arch/amdgpu/cdna2)
add_subdirectory(arch/amdgpu/cdna3)
add_subdirectory(arch/amdgpu/cdna4)
add_subdirectory(arch/amdgpu/rdna1)
add_subdirectory(arch/amdgpu/rdna2)
add_subdirectory(arch/amdgpu/rdna3)
add_subdirectory(arch/amdgpu/rdna3_5)
add_subdirectory(arch/amdgpu/rdna4)
add_subdirectory(arch/amdgpu/gfx1250)

# Keep an AMDGPU-only composition for DBT tools and hooks that intentionally do
# not link the RISC-V VM. Each final image still owns exactly one DEFAULT
# registry and therefore one legacy Decoder::create adapter.
get_property(
    _rj_builtin_amdgpu_isa_providers
    GLOBAL
    PROPERTY RJ_BUILTIN_ISA_PROVIDERS
)
rj_add_isa_target_registry(
    rocjitsu_amdgpu_isa_registry
    ACCESSOR rj_get_all_builtin_amdgpu_isa_targets
    PROVIDERS ${_rj_builtin_amdgpu_isa_providers}
    DEFAULT
)

add_subdirectory(arch/risc_v)

# Each built-in subdirectory contributes its provider target to this property.
# There is no second central provider/include/switch list to update. The full
# simulator composition includes AMDGPU and RISC-V; narrow consumers select
# only the providers they require.
get_property(_rj_builtin_isa_providers GLOBAL PROPERTY RJ_BUILTIN_ISA_PROVIDERS)
rj_add_isa_target_registry(
    rocjitsu_isa_registry
    ACCESSOR rj_get_all_builtin_isa_targets
    PROVIDERS ${_rj_builtin_isa_providers}
    DEFAULT
)
set(RJ_BUILTIN_ISA_PROVIDERS ${_rj_builtin_isa_providers} PARENT_SCOPE)
unset(_rj_builtin_isa_providers)
unset(_rj_builtin_amdgpu_isa_providers)
