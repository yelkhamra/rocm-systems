# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

rj_add_object_library(rocjitsu_isa_cdna2
    addr_calc.cpp
    decoder.cpp
    ds.cpp
    encodings.cpp
    flat.cpp
    mimg.cpp
    mtbuf.cpp
    mubuf.cpp
    operand.cpp
    smem.cpp
    sop1.cpp
    sop2.cpp
    sopc.cpp
    sopk.cpp
    sopp.cpp
    vop1.cpp
    vop2.cpp
    vop3.cpp
    vop3p.cpp
    vopc.cpp
)

rj_add_isa_target_provider(
    rocjitsu_isa_cdna2_provider
    SOURCE target_provider.cpp
    HEADER rocjitsu/isa/arch/amdgpu/cdna2/target_provider.h
    IMPLEMENTATION rocjitsu_isa_cdna2
    BUILTIN
)
