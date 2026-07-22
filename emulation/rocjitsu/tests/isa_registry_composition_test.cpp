// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file isa_registry_composition_test.cpp
/// @brief Multi-component static ISA registry composition tests.

#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/target_registry.h"
#include "rocjitsu_downstream_registry_fixture.h"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>

extern "C" size_t rj_test_narrow_target_count();
extern "C" const char *rj_test_narrow_target_id(size_t index);
extern "C" bool rj_test_narrow_has_target(const char *id);
extern "C" bool rj_test_narrow_has_execution();
extern "C" bool rj_test_narrow_decode_has_execute();

namespace {

TEST(IsaRegistryCompositionTest, ComponentsKeepIndependentStaticSubsets) {
  using namespace rocjitsu;

  constexpr std::array<std::string_view, 11> kBuiltinIds = {
      "cdna1", "cdna2", "cdna3",   "cdna4", "gfx1250", "rdna1",
      "rdna2", "rdna3", "rdna3_5", "rdna4", "risc-v",
  };
  const IsaTargetRegistry &full = default_isa_target_registry();
  ASSERT_EQ(full.targets().size(), kBuiltinIds.size());
  for (size_t i = 0; i < kBuiltinIds.size(); ++i)
    EXPECT_EQ(full.targets()[i].id, kBuiltinIds[i]);

  const IsaTargetDescriptor *full_gfx1250 = full.find("gfx1250");
  ASSERT_NE(full_gfx1250, nullptr);
  EXPECT_TRUE(has_capability(full_gfx1250->capabilities, IsaTargetCapability::Execution));
  EXPECT_NE(full_gfx1250->execution_backend, nullptr);

  constexpr uint32_t kSNop = 0xBF800000u;
  auto full_decoder = Decoder::create(full, "gfx1250");
  ASSERT_NE(full_decoder, nullptr);
  std::unique_ptr<Instruction> full_instruction(full_decoder->decode(&kSNop));
  ASSERT_NE(full_instruction, nullptr);
  EXPECT_NE(full_instruction->execute, nullptr);

  ASSERT_EQ(rj_test_narrow_target_count(), 1u);
  ASSERT_NE(rj_test_narrow_target_id(0), nullptr);
  EXPECT_STREQ(rj_test_narrow_target_id(0), "gfx1250");
  EXPECT_TRUE(rj_test_narrow_has_target("gfx1250"));
  EXPECT_FALSE(rj_test_narrow_has_target("rdna4"));
  EXPECT_FALSE(rj_test_narrow_has_execution());
  EXPECT_FALSE(rj_test_narrow_decode_has_execute());

  const IsaTargetRegistry &downstream = rj_get_downstream_fixture_targets();
  ASSERT_EQ(downstream.targets().size(), 1u);
  EXPECT_EQ(downstream.targets()[0].id, "vendor-downstream-test");
  EXPECT_EQ(downstream.find(ROCJITSU_CODE_ARCH_RESERVED_0), &downstream.targets()[0]);
  EXPECT_EQ(downstream.find(ROCJITSU_CODE_TARGET_RESERVED_0), &downstream.targets()[0]);
  EXPECT_NE(Decoder::create(downstream, "vendor-downstream-test"), nullptr);
  EXPECT_EQ(full.find("vendor-downstream-test"), nullptr);
  EXPECT_FALSE(rj_test_narrow_has_target("vendor-downstream-test"));

  // Observing either other component never mutates or unions these registries.
  EXPECT_EQ(full.targets().size(), kBuiltinIds.size());
  EXPECT_EQ(rj_test_narrow_target_count(), 1u);
}

} // namespace
