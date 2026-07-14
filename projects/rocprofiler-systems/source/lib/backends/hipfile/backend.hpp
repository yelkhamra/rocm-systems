// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// hipFile's public stats API (hipFileStatsLevel3_t + hipFileGetStatsL3). The
// backend links directly against libhipfile (via the hip::hipfile imported
// target), matching how the profiler consumes amd_smi and other ROCm libraries.
// hipFile's stats query reads the calling process' own shared stats region, so
// this backend must run inside the profiled process (which the injected
// profiler already does); the dynamic linker guarantees a single libhipfile
// instance per process, so the profiler and the target application observe the
// same stats.
#include <hipfile.h>

namespace rocprofsys::backends::hipfile
{

/**
 * @brief Thin wrapper around hipFile's in-process Level-3 stats query.
 */
class backend
{
public:
    /**
     * @brief Snapshot the Level-3 (per-GPU) hipFile stats.
     * @param out [out] populated on success (zero-initialized first).
     * @return true if the stats were read successfully. Returns false when the
     *         target has not initialized hipFile stats (e.g. it never performed
     *         hipFile I/O, or HIPFILE_STATS_LEVEL is disabled).
     */
    bool get_stats(hipFileStatsLevel3_t& out) noexcept
    {
        out          = hipFileStatsLevel3_t{};
        auto _status = hipFileGetStatsL3(&out);
        return _status.err == hipFileSuccess;
    }
};

}  // namespace rocprofsys::backends::hipfile
