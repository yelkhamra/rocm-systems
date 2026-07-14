// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "backends/hipfile/backend.hpp"
#include "common/env_vars.hpp"
#include "core/agent.hpp"
#include "core/categories.hpp"
#include "core/common.hpp"
#include "core/config.hpp"
#include "core/trace_cache/cache_manager.hpp"
#include "core/trace_cache/cacheable.hpp"
#include "core/trace_cache/metadata_registry.hpp"
#include "core/trace_cache/sample_type.hpp"
#include "logger/debug.hpp"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <set>
#include <string>

namespace rocprofsys::pmc::collectors::hipfile
{

/**
 * @brief Periodic sampler for hipFile GPU-direct storage I/O statistics.
 *
 * Unlike the AMD-SMI GPU collector (which enumerates devices and queries each
 * one), hipFile exposes a single process-wide call, hipFileGetStatsL3(), that
 * returns per-GPU stats for the whole process. Each sample snapshots that call
 * and emits per-GPU counters (and two process-global registration counters)
 * through the generic pmc_event_with_sample path, so they land in both the
 * perfetto counter tracks and the rocpd PMC tables with no new sample type.
 *
 * hipFile's counters are cumulative over the process lifetime. This collector
 * reports the per-interval DELTA between consecutive samples (not the raw
 * cumulative totals), so the counter tracks reflect activity in each sampling
 * window. Bandwidth is recomputed per interval as delta(bytes)/delta(duration).
 *
 * Duck-typed to satisfy pmc::collectors::collector_slice:
 * setup/config/sample/post_process/shutdown/pause.
 */
class collector
{
public:
    /// @brief Per-GPU counter descriptor: display suffix, unit, and the L3 field.
    /// Each is reported as a per-interval delta.
    struct metric_desc
    {
        const char*   suffix;
        const char*   unit;
        std::uint64_t hipFilePerGpuStats_t::*field;
    };

    static constexpr std::array<metric_desc, 12> per_gpu_counters() noexcept
    {
        return {
            { { "Read Bytes", "bytes", &hipFilePerGpuStats_t::read_bytes },
              { "Write Bytes", "bytes", &hipFilePerGpuStats_t::writes_bytes },
              { "Read Ops", "count", &hipFilePerGpuStats_t::n_total_reads },
              { "Write Ops", "count", &hipFilePerGpuStats_t::n_total_writes },
              { "Fastpath Reads", "count", &hipFilePerGpuStats_t::n_nvfs_reads },
              { "Fastpath Writes", "count", &hipFilePerGpuStats_t::n_nvfs_writes },
              { "Fallback Reads", "count", &hipFilePerGpuStats_t::n_posix_reads },
              { "Fallback Writes", "count", &hipFilePerGpuStats_t::n_posix_writes },
              { "Unaligned Reads", "count", &hipFilePerGpuStats_t::n_unaligned_reads },
              { "Unaligned Writes", "count", &hipFilePerGpuStats_t::n_unaligned_writes },
              { "Read Errors", "count", &hipFilePerGpuStats_t::n_reads_err },
              { "Write Errors", "count", &hipFilePerGpuStats_t::n_writes_err } }
        };
    }

    void setup()
    {
        m_enabled = config::get_use_hipfile();
        if(!m_enabled)
        {
            return;
        }

        // Ask libhipfile to enable its stats server. hipFile reads this the first
        // time it initializes; do not override a level the user set explicitly.
        setenv(env_vars::HIPFILE_STATS_LEVEL, "1", /*overwrite=*/0);

        trace_cache::get_metadata_registry().add_string(
            trait::name<category::hipfile>::value);

        LOG_INFO("hipFile I/O stats sampling enabled (per-interval deltas)");
    }

    void config()
    {
        if(!m_enabled)
        {
            return;
        }
        trace_cache::get_metadata_registry().add_string(
            trait::name<category::hipfile>::value);
    }

    void sample(std::int64_t timestamp)
    {
        if(!m_enabled)
        {
            return;
        }

        hipFileStatsLevel3_t _stats{};
        if(!m_backend.get_stats(_stats))
        {
            return;
        }

        const auto _ts = static_cast<std::uint64_t>(timestamp);

        for(std::uint32_t _gpu = 0; _gpu < HIPFILE_MAX_GPUS; ++_gpu)
        {
            const auto& _cur = _stats.per_gpu_stats[_gpu];
            const auto& _prv = m_prev.per_gpu_stats[_gpu];
            if(!gpu_has_activity(_cur) && _gpu != 0)
            {
                continue;
            }

            ensure_registered(_gpu);

            // Cumulative counters -> per-interval deltas.
            for(const auto& _m : per_gpu_counters())
            {
                emit(_gpu, _m.suffix,
                     static_cast<double>(delta(_cur.*(_m.field), _prv.*(_m.field))), _ts);
            }

            // Bandwidth: recomputed per interval as delta(bytes) / delta(duration).
            emit(_gpu, "Read Bandwidth",
                 bandwidth(delta(_cur.read_bytes, _prv.read_bytes),
                           delta(_cur.read_duration_us, _prv.read_duration_us)),
                 _ts);
            emit(_gpu, "Write Bandwidth",
                 bandwidth(delta(_cur.writes_bytes, _prv.writes_bytes),
                           delta(_cur.write_duration_us, _prv.write_duration_us)),
                 _ts);

            // Registration counters are process-global; report them on GPU 0.
            if(_gpu == 0)
            {
                emit(
                    0, "File Registrations",
                    static_cast<double>(delta(_stats.detailed.basic.hdl_register_ops.ok,
                                              m_prev.detailed.basic.hdl_register_ops.ok)),
                    _ts);
                emit(
                    0, "Buffer Registrations",
                    static_cast<double>(delta(_stats.detailed.basic.buf_register_ops.ok,
                                              m_prev.detailed.basic.buf_register_ops.ok)),
                    _ts);
            }
        }

        m_prev = _stats;
    }

    void pause(std::int64_t timestamp)
    {
        if(!m_enabled)
        {
            return;
        }
        const auto _ts = static_cast<std::uint64_t>(timestamp);
        for(auto _gpu : m_registered_gpus)
        {
            for(const auto& _m : per_gpu_counters())
            {
                emit(_gpu, _m.suffix, 0.0, _ts);
            }
            emit(_gpu, "Read Bandwidth", 0.0, _ts);
            emit(_gpu, "Write Bandwidth", 0.0, _ts);
            if(_gpu == 0)
            {
                emit(0, "File Registrations", 0.0, _ts);
                emit(0, "Buffer Registrations", 0.0, _ts);
            }
        }
    }

    void post_process()
    {
        if(m_enabled && m_registered_gpus.empty())
        {
            LOG_WARNING(
                "hipFile stats sampling was enabled but no hipFile activity was observed "
                "(is libhipfile loaded and HIPFILE_STATS_LEVEL set in the target?)");
        }
    }

    void shutdown() {}

private:
    /// @brief Per-interval delta of a cumulative counter, guarding against resets.
    static std::uint64_t delta(std::uint64_t cur, std::uint64_t prev) noexcept
    {
        return (cur >= prev) ? (cur - prev) : 0;
    }

    /// @brief Bandwidth (bytes/sec) from an interval's byte and duration deltas.
    static double bandwidth(std::uint64_t bytes, std::uint64_t duration_us) noexcept
    {
        return (duration_us > 0)
                   ? (static_cast<double>(bytes) * 1e6 / static_cast<double>(duration_us))
                   : 0.0;
    }

    static bool gpu_has_activity(const hipFilePerGpuStats_t& _g) noexcept
    {
        return (_g.read_bytes | _g.writes_bytes | _g.n_total_reads | _g.n_total_writes |
                _g.n_reads_err | _g.n_writes_err | _g.n_mmap) != 0;
    }

    static std::string track_name(std::uint32_t gpu_id, const char* suffix)
    {
        return "hipFile GPU" + std::to_string(gpu_id) + " " + suffix;
    }

    void ensure_registered(std::uint32_t gpu_id)
    {
        if(m_registered_gpus.count(gpu_id) > 0)
        {
            return;
        }
        m_registered_gpus.insert(gpu_id);

        auto _register = [&](const char* suffix, const char* unit) {
            const auto _name = track_name(gpu_id, suffix);
            trace_cache::get_metadata_registry().add_track({ _name, std::nullopt, "{}" });
            trace_cache::get_metadata_registry().add_pmc_info(
                { agent_type::GPU, gpu_id, "GPU", 0, 0, _name, suffix,
                  trait::name<category::hipfile>::description, "", "", unit,
                  rocprofsys::trace_cache::ABSOLUTE, "", "", 0, 0, "{}" });
        };

        for(const auto& _m : per_gpu_counters())
        {
            _register(_m.suffix, _m.unit);
        }
        _register("Read Bandwidth", "bytes/s");
        _register("Write Bandwidth", "bytes/s");
        if(gpu_id == 0)
        {
            _register("File Registrations", "count");
            _register("Buffer Registrations", "count");
        }
    }

    void emit(std::uint32_t gpu_id, const char* suffix, double value, std::uint64_t ts)
    {
        const auto _name = track_name(gpu_id, suffix);
        trace_cache::get_buffer_storage().store(trace_cache::pmc_event_with_sample{
            static_cast<size_t>(category_enum_id<category::hipfile>::value), _name, ts,
            "{}", 0, 0, 0, "{}", "{}", gpu_id, static_cast<std::uint8_t>(agent_type::GPU),
            _name, value, std::nullopt });
    }

    backends::hipfile::backend m_backend{};
    bool                       m_enabled{ false };
    std::set<std::uint32_t>    m_registered_gpus{};
    hipFileStatsLevel3_t       m_prev{};  ///< Previous snapshot for delta computation.
};

}  // namespace rocprofsys::pmc::collectors::hipfile
