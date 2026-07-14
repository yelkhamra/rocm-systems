# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# PMC hipFile Collector Sources (header-only)
set(pmc_hipfile_sources ${CMAKE_CURRENT_LIST_DIR}/collector.hpp)

# Add to parent variable (used by collectors CMakeLists.txt)
set(pmc_hipfile_sources ${pmc_hipfile_sources} PARENT_SCOPE)
