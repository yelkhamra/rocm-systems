# AGENTS.md

Guidance for coding agents working on hipFile.

## Project Overview

hipFile is an early-access AMD Infinity Storage library for direct-to-GPU file
IO. The primary C API is in `include/hipfile.h`; AMD implementation code lives
under `src/amd_detail`, NVIDIA compatibility code lives under
`src/nvidia_detail`, and Python bindings live under `python/`.

This project uses CMake for builds and CTest for tests. Prefer the existing
CMake helpers in `cmake/` and the established source layout over introducing
new build patterns.

## Source Layout

- `include/`: public C API headers.
- `src/amd_detail/`: AMD implementation.
- `src/nvidia_detail/`: NVIDIA/cuFile compatibility implementation.
- `src/common/`: common implementation pieces.
- `shared/`: shared compatibility and warning headers.
- `test/`: CTest/GTest tests, split by unit, system, platform, and legacy tests.
- `examples/`: example applications, including `aiscp`.
- `tools/`: installable helper tools such as `ais-check` and `ais-stats`.
- `python/`: experimental Python bindings built with Cython and
  scikit-build-core.
- `util/`: project helper scripts for formatting, coverage, and CI checks.

## Build

Use `build/` as the default local build directory.

Before configuring a build, ask whether running system tests is desired. If they
are, ask the user for an AIS-capable directory and pass it as
`-DAIS_CAPABLE_DIR=<path>`.

```bash
cmake -S . -B build \
  -DCMAKE_C_COMPILER=amdclang \
  -DCMAKE_CXX_COMPILER=amdclang++ \
  -DCMAKE_HIP_COMPILER=amdclang++ \
  '-DCMAKE_HIP_ARCHITECTURES=gfx950;gfx1201;gfx1200;gfx1101;gfx1100;gfx1030;gfx942;gfx90a;gfx908' \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
  -DHIPFILE_ROCPROFILER_REGISTER=OFF
cmake --build build --parallel
```

Keep the `CMAKE_HIP_ARCHITECTURES` value quoted in shell commands because the
semicolon-separated architecture list is a single CMake argument.

The default HIP platform is AMD. To be explicit, add:

```bash
-DCMAKE_HIP_PLATFORM=amd
```

For NVIDIA/cuFile compatibility builds, use `build-nvidia/` as the build
directory:

```bash
cmake -S . -B build-nvidia \
  -DCMAKE_C_COMPILER=gcc \
  -DCMAKE_CXX_COMPILER=g++ \
  -DCMAKE_HIP_COMPILER=nvcc \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
  -DCMAKE_HIP_PLATFORM=nvidia
cmake --build build-nvidia --parallel
```

Useful CMake cache options:

- `CMAKE_HIP_PLATFORM=amd|nvidia`: target HIP platform.
- `CMAKE_C_COMPILER=amdclang`: preferred C compiler for AMD builds.
- `CMAKE_CXX_COMPILER=amdclang++`: preferred C++ compiler for AMD builds.
- `CMAKE_HIP_COMPILER=<path>`: use when CMake cannot find `hipcc` or `nvcc`.
- `CMAKE_HIP_ARCHITECTURES=<list>`: target GPU architectures; local AMD
  builds should use `gfx950;gfx1201;gfx1200;gfx1101;gfx1100;gfx1030;gfx942;gfx90a;gfx908`.
- `CMAKE_EXPORT_COMPILE_COMMANDS=1`: write `compile_commands.json`.
- `ROCM_PATH=<path>`: ROCm install path; defaults to `/opt/rocm` or
  `/opt/rocm-${ROCM_VERSION}`.
- `ROCM_VERSION=<version>`: ROCm version hint.
- `AIS_CXX_STANDARD=17|20`: C++ standard; C++17 is the normal baseline.
- `BUILD_TESTING=ON|OFF`: enable or disable test targets.
- `AIS_CAPABLE_DIR=<path>`: directory used by end-to-end/system tests.
- `AIS_BUILD_DOCS=ON|OFF`: build Doxygen API documentation.
- `AIS_INSTALL_EXAMPLES=ON|OFF`: include example programs.
- `AIS_INSTALL_TOOLS=ON|OFF`: include helper tools.
- `AIS_USE_CLANG_TIDY=ON|OFF`: run `clang-tidy` with clang builds.
- `AIS_USE_IWYU=ON|OFF`: run include-what-you-use.
- `AIS_USE_CODE_COVERAGE=ON|OFF`: build with LLVM coverage instrumentation.
- `AIS_USE_SANITIZERS=ON|OFF`: enable sanitizer set for clang builds.
- `AIS_USE_THREAD_SANITIZER=ON|OFF`: enable thread sanitizer; do not combine
  with `AIS_USE_SANITIZERS`.

Do not assume system, stress, or end-to-end tests are safe to run on every
machine. hipFile system tests can require a configured ROCm stack, a HIP-capable
GPU, and an AIS-supported filesystem/device.

## Test

Run tests from the build directory, or use `--test-dir build`.
Run CTest outside the filesystem sandbox. Some unit tests exercise process,
descriptor, and filesystem behavior that can fail spuriously under sandboxing.

Unit tests should only use StrictMocks, instead of more permissive mocks.

```bash
ctest --test-dir build -V -L unit
```

CTest labels in current use include:

- `unit`: tests that should not require a HIP-capable GPU.
- `system`: tests that call into the GPU driver or exercise hardware behavior.
- `stress`: concurrency/stress tests that may be hardware- or timing-sensitive.
- `hipfile`: public hipFile API tests.
- `internal`: internal implementation tests.

Multiple `-L` arguments are combined as logical AND. A single label expression
can use `|` for OR.

Examples:

```bash
ctest --test-dir build -V -L unit
ctest --test-dir build -V -L internal -L unit
ctest --test-dir build -V -L 'system|unit'
ctest --test-dir build -V -R HipFileBuffer.get_buffer_makes_temporary_buffer
```

System tests that need an AIS-capable directory require configuring with
`-DAIS_CAPABLE_DIR=<path>`. The path must point to an AIS-capable directory;
do not hard-code paths in tests. For example:

```bash
-DAIS_CAPABLE_DIR=/mnt/ais/ext4/jpatters
```

## Formatting and Linting

Format C and C++ files with the project wrapper:

```bash
./util/format-source.sh
```

To use a specific clang-format major version:

```bash
./util/format-source.sh 18
```

The repository also carries configuration for `cmakelint`, `codespell`,
`clang-format`, `clang-tidy`, IWYU, ShellCheck, pylint, and black. Use the
existing config files and CMake options instead of creating parallel tooling.

Shell scripts should pass ShellCheck and use:

```bash
#!/usr/bin/env bash
```

## Coverage

For coverage, configure with clang and `AIS_USE_CODE_COVERAGE=ON`, then build
and run the relevant tests.

```bash
cmake -S . -B build -DCMAKE_CXX_COMPILER=amdclang++ -DAIS_USE_CODE_COVERAGE=ON
cmake --build build --parallel
ctest --test-dir build -V -L unit
./util/llvm-coverage.sh
```

The coverage script defaults to `build/` and writes `coverage-report.txt` and
`coverage-lines.txt` there. Use `./util/llvm-coverage.sh -b <build-dir>` for a
non-default build directory.

## Python Bindings

The Python package is experimental and lives in `python/`. It uses Cython and
scikit-build-core and requires Python 3.10 or newer.

Prefer an existing `.venv` if present:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e python
```

If editable install cannot find the C library or HIP headers, pass explicit
CMake definitions from the repository root:

```bash
pip install -e python \
  -Ccmake.define.HIPFILE_INCLUDE_DIR=../include \
  -Ccmake.define.HIPFILE_LIBRARY=../build/src/amd_detail \
  -Ccmake.define.HIP_INCLUDE_DIR=/opt/rocm/include
```

Use pylint and black for Python linting/formatting according to the existing
project configuration and CI expectations.

## Coding Conventions

- C++17 and C11 are supported. Avoid GNU extensions.
- Follow modern CMake 3.x patterns; do not introduce legacy CMake style.
- Use hyphens instead of underscores in new file and directory names.
- Public headers require Doxygen markup for public symbols.
- Private headers should use Doxygen markup with `@internal` where useful.
- Use `#pragma once` instead of include guards.
- Place local quoted includes before system bracket includes, with each block
  sorted alphabetically.
- Keep changes scoped. Avoid unrelated refactors, generated-output churn, and
  broad formatting changes unless the task specifically calls for them.
- New functionality needs automated tests. Bug fixes should include a
  proof-of-concept test that fails before the fix and passes after.
- Production code should not use AIS_TESTING to ifdef testing-only functions. If necessary, dependency injection should be used.

## Safety Notes

- hipFile is early-access software; avoid recommending production use.
- Local NVMe, ext4/xfs mount options, ROCm version, kernel P2PDMA support, and
  amdgpu setup can affect test results.
- Do not run destructive storage setup commands, reformat devices, alter kernel
  module configuration, or run broad system/stress tests unless explicitly asked.
- Preserve unrelated user changes in the working tree.
