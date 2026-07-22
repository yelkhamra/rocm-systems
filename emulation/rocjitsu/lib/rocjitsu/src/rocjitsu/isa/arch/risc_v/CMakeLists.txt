# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

rj_add_object_library(rocjitsu_isa_risc_v
    decoder.cpp
    operand.cpp
    encodings.cpp
    base_i.cpp
    m_ext.cpp
    a_ext.cpp
    f_ext.cpp
    d_ext.cpp
    zicsr.cpp
    zifencei.cpp
    privileged.cpp
)

rj_add_isa_target_provider(
    rocjitsu_isa_risc_v_provider
    SOURCE target_provider.cpp
    HEADER rocjitsu/isa/arch/risc_v/target_provider.h
    IMPLEMENTATION rocjitsu_isa_risc_v
    BUILTIN
)
