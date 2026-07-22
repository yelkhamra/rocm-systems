// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/target_registry.h"

#include <algorithm>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>

namespace rocjitsu {
namespace {

std::string duplicate_message(std::string_view kind, std::string_view value) {
  return "duplicate ISA target " + std::string(kind) + " '" + std::string(value) + "'";
}

template <typename Key> constexpr auto enum_value(Key key) {
  return static_cast<std::underlying_type_t<Key>>(key);
}

constexpr bool is_public_architecture_key(rj_code_arch_t key) {
  const auto value = enum_value(key);
  return value < enum_value(ROCJITSU_CODE_ARCH_NUM_ARCHS) ||
         (value >= enum_value(ROCJITSU_CODE_ARCH_RESERVED_0) &&
          value <= enum_value(ROCJITSU_CODE_ARCH_RESERVED_7));
}

constexpr bool is_public_gpu_target_key(rj_code_target_id_t key) {
  const auto value = enum_value(key);
  return value < enum_value(ROCJITSU_CODE_TARGET_INVALID) ||
         (value >= enum_value(ROCJITSU_CODE_TARGET_RESERVED_0) &&
          value <= enum_value(ROCJITSU_CODE_TARGET_RESERVED_7));
}

template <typename Range, typename Predicate>
void validate_enum_keys(const Range &keys, Predicate is_public_key, std::string_view kind) {
  for (auto it = keys.begin(); it != keys.end(); ++it) {
    if (!is_public_key(*it))
      throw std::invalid_argument("ISA target contains an unallocated " + std::string(kind) +
                                  " enum value");
    if (std::find(keys.begin(), it, *it) != it)
      throw std::invalid_argument(duplicate_message(kind, std::to_string(enum_value(*it))));
  }
}

} // namespace

void IsaTargetRegistry::require_frozen() const {
  if (!frozen_)
    throw std::logic_error("ISA target registry must be frozen before lookup");
}

void IsaTargetRegistry::require_mutable() const {
  if (frozen_)
    throw std::logic_error("cannot modify a frozen ISA target registry");
}

void IsaTargetRegistry::add(IsaTargetDescriptor descriptor) {
  require_mutable();
  if (descriptor.id.empty())
    throw std::invalid_argument("ISA target canonical ID must not be empty");
  if (descriptor.decoder_factory == nullptr)
    throw std::invalid_argument("ISA target '" + descriptor.id + "' has no decoder factory");
  if (!has_capability(descriptor.capabilities, IsaTargetCapability::Model))
    throw std::invalid_argument("ISA target '" + descriptor.id + "' has no model capability");

  std::unordered_map<std::string_view, bool> new_ids;
  new_ids.emplace(descriptor.id, true);
  for (const std::string &alias : descriptor.aliases) {
    if (alias.empty())
      throw std::invalid_argument("ISA target '" + descriptor.id + "' has an empty alias");
    if (!new_ids.emplace(alias, true).second)
      throw std::invalid_argument(duplicate_message("ID", alias));
  }
  validate_enum_keys(descriptor.architecture_ids, is_public_architecture_key, "architecture");
  validate_enum_keys(descriptor.gpu_target_ids, is_public_gpu_target_key, "GPU target");

  for (const IsaTargetDescriptor &existing : targets_) {
    auto conflicts_with = [&](std::string_view id) {
      if (new_ids.contains(id))
        throw std::invalid_argument(duplicate_message("ID", id));
    };
    conflicts_with(existing.id);
    for (const std::string &alias : existing.aliases)
      conflicts_with(alias);

    for (rj_code_arch_t architecture_id : descriptor.architecture_ids) {
      if (std::find(existing.architecture_ids.begin(), existing.architecture_ids.end(),
                    architecture_id) != existing.architecture_ids.end())
        throw std::invalid_argument(duplicate_message(
            "architecture",
            std::to_string(static_cast<std::underlying_type_t<rj_code_arch_t>>(architecture_id))));
    }
    for (rj_code_target_id_t gpu_target_id : descriptor.gpu_target_ids) {
      if (std::find(existing.gpu_target_ids.begin(), existing.gpu_target_ids.end(),
                    gpu_target_id) != existing.gpu_target_ids.end())
        throw std::invalid_argument(duplicate_message(
            "GPU target", std::to_string(static_cast<std::underlying_type_t<rj_code_target_id_t>>(
                              gpu_target_id))));
    }
  }

  targets_.push_back(std::move(descriptor));
}

void IsaTargetRegistry::merge(const IsaTargetRegistry &other) {
  require_mutable();
  other.require_frozen();
  for (const IsaTargetDescriptor &descriptor : other.targets_)
    add(descriptor);
}

void IsaTargetRegistry::freeze() {
  require_mutable();
  std::sort(targets_.begin(), targets_.end(),
            [](const IsaTargetDescriptor &lhs, const IsaTargetDescriptor &rhs) {
              return lhs.id < rhs.id;
            });
  frozen_ = true;
}

std::span<const IsaTargetDescriptor> IsaTargetRegistry::targets() const {
  require_frozen();
  return targets_;
}

const IsaTargetDescriptor *IsaTargetRegistry::find(std::string_view id) const {
  require_frozen();
  for (const IsaTargetDescriptor &target : targets_) {
    if (target.id == id ||
        std::find(target.aliases.begin(), target.aliases.end(), id) != target.aliases.end())
      return &target;
  }
  return nullptr;
}

const IsaTargetDescriptor *IsaTargetRegistry::find(rj_code_arch_t architecture_id) const {
  require_frozen();
  for (const IsaTargetDescriptor &target : targets_) {
    if (std::find(target.architecture_ids.begin(), target.architecture_ids.end(),
                  architecture_id) != target.architecture_ids.end())
      return &target;
  }
  return nullptr;
}

const IsaTargetDescriptor *IsaTargetRegistry::find(rj_code_target_id_t gpu_target_id) const {
  require_frozen();
  for (const IsaTargetDescriptor &target : targets_) {
    if (std::find(target.gpu_target_ids.begin(), target.gpu_target_ids.end(), gpu_target_id) !=
        target.gpu_target_ids.end())
      return &target;
  }
  return nullptr;
}

IsaTargetRegistry make_isa_target_registry(std::span<const IsaTargetProvider> providers) {
  IsaTargetRegistry registry;
  for (IsaTargetProvider provider : providers) {
    if (provider == nullptr)
      throw std::invalid_argument("ISA target provider must not be null");
    provider(registry);
  }
  registry.freeze();
  return registry;
}

} // namespace rocjitsu
