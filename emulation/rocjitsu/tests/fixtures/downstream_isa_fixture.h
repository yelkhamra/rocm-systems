// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file downstream_isa_fixture.h
/// @brief Minimal out-of-tree-style ISA used to test static provider selection.

#ifndef ROCJITSU_TESTS_FIXTURES_DOWNSTREAM_ISA_FIXTURE_H_
#define ROCJITSU_TESTS_FIXTURES_DOWNSTREAM_ISA_FIXTURE_H_

#include "rocjitsu/isa/instruction.h"

#include <memory>

namespace rocjitsu::test {

struct DownstreamIsa {
  struct Decoder {
    static std::unique_ptr<Instruction> decode(const rj_code_binary_inst_t *) { return nullptr; }
  };
};

} // namespace rocjitsu::test

#endif // ROCJITSU_TESTS_FIXTURES_DOWNSTREAM_ISA_FIXTURE_H_
