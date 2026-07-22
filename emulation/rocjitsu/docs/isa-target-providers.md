# ISA Target Providers

RocJITsu ISA targets are selected statically for each final executable or
shared object. A consumer owns a frozen `IsaTargetRegistry`; it does not consult
a process-global registry and it cannot discover or register targets after
initialization.

This lets different components in one process have different target sets. For
example, a model-only errata patcher can contain only gfx1250 while the
simulator contains every built-in target. Linking both does not union their
registries or pull the simulator execution implementation into the patcher.

## Target contract

Each target header publishes a constexpr `IsaTargetDescription` containing:

- a canonical, open-ended string ID;
- optional string aliases;
- optional public architecture and GPU-target enum keys;
- model and optional execution capabilities.

Registration converts that high-level description into an owning
`IsaTargetDescriptor`, adding the target's decoder factory and optional
immutable execution backend.

The canonical ID is not limited by either public C enum. Named built-in values
and opaque `RESERVED_*` slots are lookup keys bound by the selected provider
inside each component's registry. A downstream integration selects a reserved
slot without adding its final target name, ISA structs, or registration details
to the public C header.

A provider header is usable as an ordinary header. Its declarations and
description are guarded, while an MLIR-style opt-in section after the guard
emits its registration entry when requested:

```cpp
#ifndef VENDOR_ISA_TARGET_PROVIDER_H_
#define VENDOR_ISA_TARGET_PROVIDER_H_

#include "rocjitsu/isa/target_registry.h"

#include <array>

namespace vendor {
inline constexpr std::array<rj_code_arch_t, 1> architecture_ids{
    ROCJITSU_CODE_ARCH_RESERVED_0,
};
inline constexpr std::array<rj_code_target_id_t, 1> gpu_target_ids{
    ROCJITSU_CODE_TARGET_RESERVED_0,
};
inline constexpr rocjitsu::IsaTargetDescription target_description{
    .id = "vendor-next",
    .architecture_ids = architecture_ids,
    .gpu_target_ids = gpu_target_ids,
};
void register_target(rocjitsu::IsaTargetRegistry &registry);
} // namespace vendor

#endif // VENDOR_ISA_TARGET_PROVIDER_H_

#ifdef ROCJITSU_GET_ISA_TARGET_REGISTRATION
ROCJITSU_GET_ISA_TARGET_REGISTRATION(vendor::register_target)
#endif
```

Its provider source owns the decoder type and implementation-only details:

```cpp
void vendor::register_target(rocjitsu::IsaTargetRegistry &registry) {
  rocjitsu::add_isa_target<Isa>(registry, target_description);
}
```

`add_isa_target` is the common decoder path, not a registration ABI. A target
with a bespoke decoder factory can construct an `IsaTargetDescriptor` and call
`IsaTargetRegistry::add` directly.

Providers run only while a consumer registry is being constructed. `freeze()`
sorts canonical IDs for deterministic enumeration and makes the registry
read-only. Empty IDs, missing decoder factories, duplicate canonical IDs or
aliases, unallocated public enum keys, lookup before freeze, and mutation after
freeze are rejected.

## Declaring a provider in CMake

`rj_add_isa_target_provider` packages those target-owned files with the
target implementation. The build system does not contain or generate target
IDs, aliases, C++ types, capabilities, or backend expressions:

```cmake
add_library(vendor_isa_model INTERFACE)
target_include_directories(vendor_isa_model INTERFACE include)

rj_add_isa_target_provider(
    vendor_isa_provider
    SOURCE target_provider.cpp
    HEADER vendor/target_provider.h
    IMPLEMENTATION vendor_isa_model
)
```

The provider header declares the target's capabilities. For a split ISA, the
provider source wires one immutable `IsaExecutionBackend`. gfx1250 uses a
generated dense callback table: decoding selects one direct
`Instruction::execute` callback, while model-only decoding selects null. The
model library therefore has no VM, SimDojo, address calculation, or instruction
execution symbols.

The opaque operand pointer in `IsaExecutionBackend` is owned by the provider
and must remain valid for the lifetime of every decoder and instruction using
it. A split ISA captures that pointer in operands created during decode and
re-enters the same backend scope when execution constructs temporary operands.

The provider function must be unique in a linked image. `IMPLEMENTATION` may be
an object, static, or interface target; its include requirements are also
available while the provider source is compiled.

## Selecting a subset per consumer

`rj_add_isa_target_registry` includes exactly the provider contributions listed
by one consumer:

```cmake
# Model-only errata component: exactly one target.
rj_add_isa_target_registry(
    errata_isa_registry
    ACCESSOR errata_isa_targets
    PROVIDERS rocjitsu_isa_gfx1250_model_provider
)

# Full simulator: all providers collected from the built-in ISA directories.
rj_add_isa_target_registry(
    simulator_isa_registry
    ACCESSOR simulator_isa_targets
    PROVIDERS ${RJ_BUILTIN_ISA_PROVIDERS}
)

# A downstream target can be selected for one consumer only.
rj_add_isa_target_registry(
    vendor_tool_isa_registry
    ACCESSOR vendor_tool_isa_targets
    PROVIDERS vendor_isa_provider
)
```

Each accessor owns a function-local, immutable registry. CMake generates a
target-list header whose entire contents are one `#include` directive per
selected provider. Checked-in composition C++ includes the list normally for
declarations, then re-includes it with
`ROCJITSU_GET_ISA_TARGET_REGISTRATION` defined to emit direct calls. The macro
is immediately undefined. This follows the same selectable-section pattern as
MLIR generated headers while keeping ordinary includes side-effect free.

The direct calls retain selected providers without static constructors,
`--whole-archive`, or a hand-maintained decoder switch. Removing a provider
from `PROVIDERS` removes both its registration call and its linked
implementation from that consumer.

The helper also generates `<registry-target>.h` in the target's binary
directory and publishes that directory as an interface include path. Consumers
include this header instead of redeclaring the accessor:

```cpp
#include "errata_isa_registry.h"

auto decoder = rocjitsu::Decoder::create(errata_isa_targets(), "gfx1250");
```

`DEFAULT` additionally defines that linked image's enum-based
`Decoder::create(rj_code_arch_t)` and public C API registry. Use it for the
component that owns those enum entry points; scoped C++ consumers should call
the explicit registry overload.

The C API equivalent for a component's default registry is
`rj_code_decoder_create_for_target("gfx1250", &decoder)`.
Enum-based C entry points resolve both named and reserved keys through the same
component-owned registry; an unbound reserved slot is rejected.
Providers that bind a `gpu_target_ids` value for basic-block APIs must bind
exactly one `architecture_ids` value so that the component can select its
decoder unambiguously.

## Lifecycle and composition

Registries are move-constructible but neither assignable nor copyable. They can
be assembled directly from a provider span with `make_isa_target_registry()` or
merged from another frozen registry before freezing. Once frozen, concurrent
readers need no locking.

Two components may contain overlapping IDs because conflicts are checked only
within one registry. Their ownership remains independent:

| Component | Selected providers | Visible targets |
|---|---|---|
| gfx1250 errata shared object | `rocjitsu_isa_gfx1250_model_provider` | `gfx1250` model only |
| simulator executable | `${RJ_BUILTIN_ISA_PROVIDERS}` | all AMDGPU and RISC-V built-ins, with execution |
| downstream tool | `vendor_isa_provider` | `vendor-next` |

There is intentionally no `dlopen` provider ABI, plugin search path, runtime
discovery, late registration API, custom RTTI system, or global target union.
