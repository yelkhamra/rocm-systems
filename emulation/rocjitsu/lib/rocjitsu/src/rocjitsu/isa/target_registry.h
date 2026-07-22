// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file target_registry.h
/// @brief Scoped registry for statically selected ISA target providers.

#ifndef ROCJITSU_ISA_TARGET_REGISTRY_H_
#define ROCJITSU_ISA_TARGET_REGISTRY_H_

#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/execution_backend.h"

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace rocjitsu {

class Decoder;

/// @brief Capabilities present in a statically selected target contribution.
enum class IsaTargetCapability : uint32_t {
  Model = 1u << 0,
  Execution = 1u << 1,
};

constexpr IsaTargetCapability operator|(IsaTargetCapability lhs, IsaTargetCapability rhs) {
  return static_cast<IsaTargetCapability>(static_cast<uint32_t>(lhs) | static_cast<uint32_t>(rhs));
}

constexpr bool has_capability(IsaTargetCapability capabilities, IsaTargetCapability capability) {
  return (static_cast<uint32_t>(capabilities) & static_cast<uint32_t>(capability)) != 0;
}

/// @brief Non-owning, compile-time description published by an ISA target.
struct IsaTargetDescription {
  std::string_view id;
  std::span<const std::string_view> aliases = {};
  /// Public enum keys bound by this provider in its final component.
  std::span<const rj_code_arch_t> architecture_ids = {};
  std::span<const rj_code_target_id_t> gpu_target_ids = {};
  IsaTargetCapability capabilities = IsaTargetCapability::Model;
};

/// @brief Immutable description contributed by one ISA provider.
struct IsaTargetDescriptor {
  using DecoderFactory = std::unique_ptr<Decoder> (*)(const IsaExecutionBackend *backend);

  /// Canonical, open-ended target identity (for example ``gfx1250``).
  std::string id;
  /// Additional string identities accepted by lookup.
  std::vector<std::string> aliases = {};
  /// Public architecture enum keys accepted by lookup.
  std::vector<rj_code_arch_t> architecture_ids = {};
  /// Public GPU target enum keys accepted by lookup.
  std::vector<rj_code_target_id_t> gpu_target_ids = {};
  DecoderFactory decoder_factory = nullptr;
  IsaTargetCapability capabilities = IsaTargetCapability::Model;
  /// Null for model-only targets and for unsplit ISAs with inline execution.
  const IsaExecutionBackend *execution_backend = nullptr;
};

/// @brief Ordinary provider function selected explicitly by a consumer's build.
using IsaTargetProvider = void (*)(class IsaTargetRegistry &registry);

/// @brief A consumer-owned registry that becomes immutable before use.
///
/// There is intentionally no singleton or dynamic-loading entry point. A final
/// tool or shared object constructs its own instance from the exact provider
/// list selected by its build, then freezes it before lookup.
class IsaTargetRegistry final {
public:
  IsaTargetRegistry() = default;
  IsaTargetRegistry(IsaTargetRegistry &&) noexcept = default;
  IsaTargetRegistry &operator=(IsaTargetRegistry &&) = delete;
  IsaTargetRegistry(const IsaTargetRegistry &) = delete;
  IsaTargetRegistry &operator=(const IsaTargetRegistry &) = delete;

  /// @brief Add one provider contribution during initialization.
  /// @throws std::invalid_argument for invalid or conflicting metadata.
  /// @throws std::logic_error after freeze().
  void add(IsaTargetDescriptor descriptor);

  /// @brief Copy the contents of an already-frozen registry into this one.
  void merge(const IsaTargetRegistry &other);

  /// @brief Sort deterministically and make the registry read-only.
  void freeze();
  bool frozen() const noexcept { return frozen_; }

  /// @brief Enumerate targets by canonical ID. Requires a frozen registry.
  std::span<const IsaTargetDescriptor> targets() const;

  /// @brief Look up a canonical ID or string alias. Requires freeze().
  const IsaTargetDescriptor *find(std::string_view id) const;
  /// @brief Look up a provider-bound public architecture key. Requires freeze().
  const IsaTargetDescriptor *find(rj_code_arch_t architecture_id) const;
  /// @brief Look up a provider-bound public GPU target key. Requires freeze().
  const IsaTargetDescriptor *find(rj_code_target_id_t gpu_target_id) const;

private:
  void require_frozen() const;
  void require_mutable() const;

  std::vector<IsaTargetDescriptor> targets_;
  bool frozen_ = false;
};

/// @brief Construct and freeze a scoped registry from explicit provider calls.
IsaTargetRegistry make_isa_target_registry(std::span<const IsaTargetProvider> providers);

/// @brief Registry selected for a component's public enum and C entry points.
///
/// This function is defined only by a static composition marked ``DEFAULT``.
/// Its function-local registry is owned by that final linked image; it is not a
/// process-wide registry and is not shared with independently linked DSOs.
const IsaTargetRegistry &default_isa_target_registry();

} // namespace rocjitsu

#endif // ROCJITSU_ISA_TARGET_REGISTRY_H_
