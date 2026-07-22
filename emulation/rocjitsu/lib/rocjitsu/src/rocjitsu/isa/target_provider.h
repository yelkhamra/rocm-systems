// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file target_provider.h
/// @brief Helper for target-owned static ISA provider definitions.

#ifndef ROCJITSU_ISA_TARGET_PROVIDER_H_
#define ROCJITSU_ISA_TARGET_PROVIDER_H_

#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/target_registry.h"

#include <memory>
#include <utility>

namespace rocjitsu {

/// @brief Add a target descriptor using the decoder for @p Isa.
template <typename Isa>
void add_isa_target(IsaTargetRegistry &registry, const IsaTargetDescription &description,
                    const IsaExecutionBackend *execution_backend = nullptr) {
  std::vector<std::string> aliases;
  aliases.reserve(description.aliases.size());
  for (std::string_view alias : description.aliases)
    aliases.emplace_back(alias);

  std::vector<rj_code_arch_t> architecture_ids(description.architecture_ids.begin(),
                                               description.architecture_ids.end());
  std::vector<rj_code_target_id_t> gpu_target_ids(description.gpu_target_ids.begin(),
                                                  description.gpu_target_ids.end());

  registry.add({
      .id = std::string(description.id),
      .aliases = std::move(aliases),
      .architecture_ids = std::move(architecture_ids),
      .gpu_target_ids = std::move(gpu_target_ids),
      .decoder_factory = +[](const IsaExecutionBackend *backend) -> std::unique_ptr<Decoder> {
        return std::make_unique<IsaDecoder<Isa>>(backend);
      },
      .capabilities = description.capabilities,
      .execution_backend = execution_backend,
  });
}

} // namespace rocjitsu

#endif // ROCJITSU_ISA_TARGET_PROVIDER_H_
