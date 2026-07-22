// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/arch/amdgpu/cdna1/target_provider.h"
#include "rocjitsu/isa/arch/amdgpu/cdna2/target_provider.h"
#include "rocjitsu/isa/arch/amdgpu/rdna4/target_provider.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/target_registry.h"

#include <gtest/gtest.h>

#include <array>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

namespace rocjitsu {
namespace {

class FixtureDecoder final : public Decoder {
public:
  Instruction *decode(const rj_code_binary_inst_t *) override { return nullptr; }
};

std::unique_ptr<Decoder> create_fixture_decoder(const IsaExecutionBackend *) {
  return std::make_unique<FixtureDecoder>();
}

IsaTargetDescriptor fixture_target(std::string id) {
  IsaTargetDescriptor descriptor;
  descriptor.id = std::move(id);
  descriptor.decoder_factory = &create_fixture_decoder;
  return descriptor;
}

void register_downstream_target(IsaTargetRegistry &registry) {
  IsaTargetDescriptor descriptor = fixture_target("gfx9999");
  descriptor.aliases = {"vendor-next"};
  descriptor.architecture_ids = {ROCJITSU_CODE_ARCH_RESERVED_0};
  descriptor.gpu_target_ids = {ROCJITSU_CODE_TARGET_RESERVED_0};
  registry.add(std::move(descriptor));
}

void first_execute(Instruction &, void *) {}
void second_execute(Instruction &, void *) {}

static_assert(std::is_move_constructible_v<IsaTargetRegistry>);
static_assert(!std::is_move_assignable_v<IsaTargetRegistry>);
static_assert(cdna1::target_description.aliases.empty());
static_assert(cdna2::target_description.id == "cdna2");
static_assert(cdna2::target_description.aliases.size() == 1);
static_assert(cdna2::target_description.aliases.front() == "gfx90a");
static_assert(cdna2::target_description.architecture_ids.front() == ROCJITSU_CODE_ARCH_CDNA2);
static_assert(cdna2::target_description.gpu_target_ids.front() == ROCJITSU_CODE_TARGET_GFX90A);
static_assert(rdna4::target_description.aliases.size() == 2);
static_assert(rdna4::target_description.aliases[0] == "gfx1200");
static_assert(rdna4::target_description.aliases[1] == "gfx1201");
static_assert(static_cast<int>(ROCJITSU_CODE_TARGET_GFX1250) == 5);
static_assert(static_cast<int>(ROCJITSU_CODE_TARGET_INVALID) == 6);
static_assert(static_cast<int>(ROCJITSU_CODE_TARGET_RESERVED_0) == 0x100);

TEST(IsaTargetRegistryTest, FreezesIntoDeterministicCanonicalOrder) {
  IsaTargetRegistry registry;
  registry.add(fixture_target("zeta"));
  registry.add(fixture_target("alpha"));
  registry.freeze();

  ASSERT_EQ(registry.targets().size(), 2u);
  EXPECT_EQ(registry.targets()[0].id, "alpha");
  EXPECT_EQ(registry.targets()[1].id, "zeta");
  EXPECT_EQ(registry.find("missing"), nullptr);
  EXPECT_THROW(registry.add(fixture_target("late")), std::logic_error);
  EXPECT_THROW(registry.freeze(), std::logic_error);
}

TEST(IsaTargetRegistryTest, RejectsConflictingOpenIdentities) {
  IsaTargetRegistry registry;
  IsaTargetDescriptor first = fixture_target("first");
  first.aliases = {"shared"};
  registry.add(std::move(first));

  IsaTargetDescriptor duplicate_id = fixture_target("shared");
  EXPECT_THROW(registry.add(std::move(duplicate_id)), std::invalid_argument);

  IsaTargetDescriptor duplicate_local_alias = fixture_target("second");
  duplicate_local_alias.aliases = {"twice", "twice"};
  EXPECT_THROW(registry.add(std::move(duplicate_local_alias)), std::invalid_argument);
}

TEST(IsaTargetRegistryTest, RejectsConflictingPublicEnumKeys) {
  IsaTargetRegistry registry;
  IsaTargetDescriptor first = fixture_target("first");
  first.architecture_ids = {ROCJITSU_CODE_ARCH_RESERVED_0};
  first.gpu_target_ids = {ROCJITSU_CODE_TARGET_RESERVED_0};
  registry.add(std::move(first));

  IsaTargetDescriptor duplicate_architecture = fixture_target("duplicate-architecture");
  duplicate_architecture.architecture_ids = {ROCJITSU_CODE_ARCH_RESERVED_0};
  EXPECT_THROW(registry.add(std::move(duplicate_architecture)), std::invalid_argument);

  IsaTargetDescriptor duplicate_gpu_target = fixture_target("duplicate-gpu-target");
  duplicate_gpu_target.gpu_target_ids = {ROCJITSU_CODE_TARGET_RESERVED_0};
  EXPECT_THROW(registry.add(std::move(duplicate_gpu_target)), std::invalid_argument);
}

TEST(IsaTargetRegistryTest, RejectsInvalidTargetDescriptors) {
  IsaTargetRegistry registry;
  EXPECT_THROW(registry.add(fixture_target("")), std::invalid_argument);

  IsaTargetDescriptor missing_factory = fixture_target("missing-factory");
  missing_factory.decoder_factory = nullptr;
  EXPECT_THROW(registry.add(std::move(missing_factory)), std::invalid_argument);

  IsaTargetDescriptor missing_model = fixture_target("missing-model");
  missing_model.capabilities = static_cast<IsaTargetCapability>(0);
  EXPECT_THROW(registry.add(std::move(missing_model)), std::invalid_argument);

  IsaTargetDescriptor empty_alias = fixture_target("empty-alias");
  empty_alias.aliases = {""};
  EXPECT_THROW(registry.add(std::move(empty_alias)), std::invalid_argument);

  IsaTargetDescriptor invalid_enums = fixture_target("invalid-enums");
  invalid_enums.architecture_ids = {ROCJITSU_CODE_ARCH_INVALID};
  EXPECT_THROW(registry.add(std::move(invalid_enums)), std::invalid_argument);

  IsaTargetDescriptor unallocated_architecture = fixture_target("unallocated-architecture");
  unallocated_architecture.architecture_ids = {static_cast<rj_code_arch_t>(42)};
  EXPECT_THROW(registry.add(std::move(unallocated_architecture)), std::invalid_argument);

  IsaTargetDescriptor unallocated_gpu_target = fixture_target("unallocated-gpu-target");
  unallocated_gpu_target.gpu_target_ids = {static_cast<rj_code_target_id_t>(42)};
  EXPECT_THROW(registry.add(std::move(unallocated_gpu_target)), std::invalid_argument);

  IsaTargetDescriptor duplicate_enums = fixture_target("duplicate-enums");
  duplicate_enums.gpu_target_ids = {ROCJITSU_CODE_TARGET_RESERVED_1,
                                    ROCJITSU_CODE_TARGET_RESERVED_1};
  EXPECT_THROW(registry.add(std::move(duplicate_enums)), std::invalid_argument);
}

TEST(IsaTargetRegistryTest, RejectsLookupBeforeFreezeAndNullProviders) {
  IsaTargetRegistry registry;
  registry.add(fixture_target("target"));
  EXPECT_THROW(static_cast<void>(registry.targets()), std::logic_error);
  EXPECT_THROW(static_cast<void>(registry.find("target")), std::logic_error);
  EXPECT_THROW(static_cast<void>(registry.find(ROCJITSU_CODE_ARCH_CDNA1)), std::logic_error);
  EXPECT_THROW(static_cast<void>(registry.find(ROCJITSU_CODE_TARGET_GFX90A)), std::logic_error);

  constexpr std::array<IsaTargetProvider, 1> providers = {nullptr};
  EXPECT_THROW(static_cast<void>(make_isa_target_registry(providers)), std::invalid_argument);
}

TEST(IsaTargetRegistryTest, MergesOnlyFrozenScopedRegistries) {
  IsaTargetRegistry source;
  source.add(fixture_target("source"));
  IsaTargetRegistry destination;
  EXPECT_THROW(destination.merge(source), std::logic_error);

  source.freeze();
  destination.add(fixture_target("destination"));
  destination.merge(source);
  destination.freeze();
  EXPECT_NE(destination.find("source"), nullptr);
  EXPECT_NE(destination.find("destination"), nullptr);
}

TEST(IsaTargetRegistryTest, RejectsConflictsIntroducedByMerge) {
  IsaTargetRegistry source;
  source.add(fixture_target("shared"));
  source.freeze();

  IsaTargetRegistry destination;
  destination.add(fixture_target("shared"));
  EXPECT_THROW(destination.merge(source), std::invalid_argument);
}

TEST(IsaTargetRegistryTest, ExecutionBackendScopesNestAndRestore) {
  constexpr std::array<Instruction::ExecuteFn, 1> outer_callbacks = {&first_execute};
  constexpr std::array<Instruction::ExecuteFn, 1> inner_callbacks = {&second_execute};
  constexpr int outer_operand_backend = 1;
  constexpr int inner_operand_backend = 2;
  const IsaExecutionBackend outer{
      .instruction_callbacks = outer_callbacks.data(),
      .instruction_callback_count = outer_callbacks.size(),
      .operand_backend = &outer_operand_backend,
  };
  const IsaExecutionBackend inner{
      .instruction_callbacks = inner_callbacks.data(),
      .instruction_callback_count = inner_callbacks.size(),
      .operand_backend = &inner_operand_backend,
  };

  EXPECT_EQ(current_instruction_execute(0), nullptr);
  EXPECT_EQ(current_isa_operand_backend(), nullptr);
  {
    ScopedIsaExecutionBackend outer_scope(&outer);
    EXPECT_EQ(current_instruction_execute(0), &first_execute);
    EXPECT_EQ(current_instruction_execute(1), nullptr);
    EXPECT_EQ(current_isa_operand_backend(), &outer_operand_backend);
    {
      ScopedIsaExecutionBackend inner_scope(&inner);
      EXPECT_EQ(current_instruction_execute(0), &second_execute);
      EXPECT_EQ(current_isa_operand_backend(), &inner_operand_backend);
    }
    EXPECT_EQ(current_instruction_execute(0), &first_execute);
    EXPECT_EQ(current_isa_operand_backend(), &outer_operand_backend);
  }
  EXPECT_EQ(current_instruction_execute(0), nullptr);
  EXPECT_EQ(current_isa_operand_backend(), nullptr);
}

TEST(IsaTargetRegistryTest, DownstreamProviderBindsReservedPublicEnumSlots) {
  constexpr IsaTargetProvider providers[] = {&register_downstream_target};
  IsaTargetRegistry downstream = make_isa_target_registry(providers);

  EXPECT_NE(downstream.find("gfx9999"), nullptr);
  EXPECT_NE(downstream.find("vendor-next"), nullptr);
  const IsaTargetDescriptor *reserved_arch = downstream.find(ROCJITSU_CODE_ARCH_RESERVED_0);
  ASSERT_NE(reserved_arch, nullptr);
  EXPECT_EQ(reserved_arch->id, "gfx9999");
  const IsaTargetDescriptor *reserved_gpu = downstream.find(ROCJITSU_CODE_TARGET_RESERVED_0);
  ASSERT_NE(reserved_gpu, nullptr);
  EXPECT_EQ(reserved_gpu->id, "gfx9999");
  EXPECT_NE(Decoder::create(downstream, "gfx9999"), nullptr);
  EXPECT_NE(Decoder::create(downstream, ROCJITSU_CODE_ARCH_RESERVED_0), nullptr);

  IsaTargetRegistry unrelated;
  unrelated.add(fixture_target("unrelated"));
  unrelated.freeze();
  EXPECT_EQ(unrelated.find("gfx9999"), nullptr);
  EXPECT_EQ(unrelated.find(ROCJITSU_CODE_ARCH_RESERVED_0), nullptr);
}

TEST(IsaTargetRegistryTest, BuiltinRegistryUsesProviderOwnedPublicEnumBindings) {
  const IsaTargetRegistry &registry = default_isa_target_registry();
  const std::vector<std::string> expected = {
      "cdna1", "cdna2", "cdna3",   "cdna4", "gfx1250", "rdna1",
      "rdna2", "rdna3", "rdna3_5", "rdna4", "risc-v",
  };
  std::vector<std::string> actual;
  for (const IsaTargetDescriptor &target : registry.targets())
    actual.push_back(target.id);
  EXPECT_EQ(actual, expected);

  const IsaTargetDescriptor *gfx1201 = registry.find("gfx1201");
  ASSERT_NE(gfx1201, nullptr);
  EXPECT_EQ(gfx1201->id, "rdna4");
  const IsaTargetDescriptor *rv64i = registry.find("rv64i");
  ASSERT_NE(rv64i, nullptr);
  EXPECT_EQ(rv64i->id, "risc-v");
  const IsaTargetDescriptor *cdna3 = registry.find(ROCJITSU_CODE_ARCH_CDNA3);
  ASSERT_NE(cdna3, nullptr);
  EXPECT_EQ(cdna3->id, "cdna3");
  const IsaTargetDescriptor *gfx1201_enum = registry.find(ROCJITSU_CODE_TARGET_GFX1201);
  ASSERT_NE(gfx1201_enum, nullptr);
  EXPECT_EQ(gfx1201_enum->id, "rdna4");
  EXPECT_NE(Decoder::create(registry, "gfx942"), nullptr);
  EXPECT_NE(Decoder::create(registry, ROCJITSU_CODE_ARCH_CDNA3), nullptr);
  auto risc_v_decoder = Decoder::create(registry, ROCJITSU_CODE_ARCH_RV32I);
  ASSERT_NE(risc_v_decoder, nullptr);
  constexpr rj_code_binary_inst_t kAddiX1X0One = 0x00100093;
  std::unique_ptr<Instruction> risc_v_instruction(risc_v_decoder->decode(&kAddiX1X0One));
  ASSERT_NE(risc_v_instruction, nullptr);
  EXPECT_NE(risc_v_instruction->execute, nullptr);
  EXPECT_NE(Decoder::create(registry, ROCJITSU_CODE_ARCH_RV64I), nullptr);
}

TEST(IsaTargetRegistryTest, PublicCEntryPointAcceptsOpenTargetIds) {
  rj_code_decoder_t *decoder = nullptr;
  EXPECT_EQ(rj_code_decoder_create_for_target("gfx942", &decoder), ROCJITSU_STATUS_SUCCESS);
  ASSERT_NE(decoder, nullptr);
  rj_code_decoder_destroy(decoder);

  decoder = nullptr;
  EXPECT_EQ(rj_code_decoder_create_for_target("rv64i", &decoder), ROCJITSU_STATUS_SUCCESS);
  ASSERT_NE(decoder, nullptr);
  rj_code_decoder_destroy(decoder);

  decoder = nullptr;
  EXPECT_EQ(rj_code_decoder_create_for_target("vendor-not-linked", &decoder),
            ROCJITSU_STATUS_ERROR);
  EXPECT_EQ(decoder, nullptr);
  EXPECT_EQ(rj_code_decoder_create_for_target("", &decoder), ROCJITSU_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(rj_code_decoder_create_for_target(nullptr, &decoder), ROCJITSU_STATUS_INVALID_ARGUMENT);

  EXPECT_EQ(rj_code_decoder_create(ROCJITSU_CODE_ARCH_RESERVED_0, &decoder), ROCJITSU_STATUS_ERROR);
  EXPECT_EQ(decoder, nullptr);
}

} // namespace
} // namespace rocjitsu
