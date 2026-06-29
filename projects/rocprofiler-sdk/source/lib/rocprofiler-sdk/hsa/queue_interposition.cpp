// MIT License
//
// Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// SDK-level HSA queue interposition: wraps hsa_queue_*_write_index_* and
// hsa_signal_store_* to virtualize the queue write pointer. Producer threads
// advance QueueState::virtual_wptr; the real write_dispatch_id only advances
// at doorbell time after process_doorbell_impl runs the WriteInterceptor chain.
// Tracing-only; the gate in registration.cpp forces the legacy
// hsa_amd_queue_intercept_create path whenever a context registers
// dispatch_counter_collection, dispatch_thread_trace, or pc_sampler.
// See queue_interposition.hpp for the API.

#include "lib/rocprofiler-sdk/hsa/queue_interposition.hpp"
#include "lib/common/container/pool.hpp"
#include "lib/common/container/pool_object.hpp"
#include "lib/common/environment.hpp"
#include "lib/common/logging.hpp"
#include "lib/common/static_object.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/code_object/code_object.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/hsa/hsa.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_controller.hpp"
#include "lib/rocprofiler-sdk/hsa/signal_pool.hpp"
#include "lib/rocprofiler-sdk/internal_threading.hpp"
#include "lib/rocprofiler-sdk/kernel_dispatch/tracing.hpp"
#include "lib/rocprofiler-sdk/registration.hpp"
#include "lib/rocprofiler-sdk/tracing/tracing.hpp"

#include <rocprofiler-sdk/cxx/operators.hpp>

#include <fmt/format.h>
#include <hsa/amd_hsa_queue.h>
#include <hsa/amd_hsa_signal.h>
#include <hsa/hsa.h>
#include <hsa/hsa_api_trace.h>
#include <pthread.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace rocprofiler
{
namespace hsa
{
namespace queue_interposition
{
namespace
{
// NOTE:
//  - "installed" is for checking whether HSA functions have been passed
//  - "active" is for controlling whether wrappers are intercepting or passing through
//  - "dynamic" is for whether to allow dynamic discovery of queues whose creation was not
//      observed/intercepted. E.g., during attachment, we want to toggle this on.
auto s_intercept_installed = std::atomic<bool>{false};  // installed (may not be active)
auto s_intercept_active    = std::atomic<bool>{false};  // actively intercepting
auto s_intercept_dynamic   = std::atomic<bool>{false};  // dynamically add queue states

bool
should_bypass_inline_intercept()
{
    return (!s_intercept_installed.load(std::memory_order_acquire) ||
            !s_intercept_active.load(std::memory_order_acquire) ||
            registration::get_fini_status() != 0 ||
            // TODO: debug and enable queue interposition for attachment
            registration::supports_attachment());
}

auto*&
get_original_table()
{
    static CoreApiTable* _v = nullptr;
    return _v;
}

// Saved next-in-chain function pointers (tracing functors or raw HSA, depending on
// when install_intercept is called). Our wrappers chain through these for untracked
// queues and for the final doorbell ring on tracked queues.
auto*
get_next_table()
{
    static auto*& _v = common::static_object<CoreApiTable>::construct();
    return _v;
}

auto s_queue_interposition_debug_event_id = std::atomic<uint64_t>{0};
auto s_async_waiter_debug_id              = std::atomic<uint64_t>{0};

uint64_t
next_queue_interposition_debug_event_id()
{
    return s_queue_interposition_debug_event_id.fetch_add(1, std::memory_order_relaxed) + 1;
}

uint64_t
next_async_waiter_debug_id()
{
    return s_async_waiter_debug_id.fetch_add(1, std::memory_order_relaxed) + 1;
}

const void*
debug_ptr(const volatile void* ptr)
{
    return const_cast<const void*>(ptr);
}

std::string
debug_state_summary(const QueueState* state)
{
    if(!state) return "state=null";

    auto real_wdid = (state->real_wdid) ? __atomic_load_n(state->real_wdid, __ATOMIC_ACQUIRE) : 0;
    auto real_rdid = (state->real_rdid) ? __atomic_load_n(state->real_rdid, __ATOMIC_ACQUIRE) : 0;
    auto virtual_wptr = state->virtual_wptr.load(std::memory_order_acquire);

    return fmt::format(
        "state={} queue={} ring_buf={} ring_size={} ring_mask={} pkt_size={} virtual_wptr={} "
        "real_wdid_ptr={} real_wdid={} real_rdid_ptr={} real_rdid={} next_scan_pos={} "
        "next_submit_pos={} doorbell={}",
        fmt::ptr(state),
        fmt::ptr(static_cast<const void*>(state->hsa_queue)),
        fmt::ptr(static_cast<const void*>(state->ring_buf)),
        state->ring_size,
        state->ring_mask,
        state->pkt_size,
        virtual_wptr,
        fmt::ptr(debug_ptr(state->real_wdid)),
        real_wdid,
        fmt::ptr(debug_ptr(state->real_rdid)),
        real_rdid,
        state->next_scan_pos,
        state->next_submit_pos,
        state->doorbell_signal.handle);
}

void
queue_interposition_debug_log(const std::string& msg)
{
    ROCP_WARNING << fmt::format("[QI-DEBUG tid={}] {}", common::get_tid(), msg);
}

template <typename ClockPointT>
int64_t
debug_elapsed_us(ClockPointT start)
{
    return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() -
                                                                 start)
        .count();
}
}  // namespace

queue_registry_t&
get_queue_registry()
{
    static auto*& _v = common::static_object<queue_registry_t>::construct();
    return *_v;
}

queue_state_ptr_t
lookup_queue_state(const hsa_queue_t* queue, bool create_if_missing)
{
    queue_interposition_debug_log(
        fmt::format("lookup_queue_state begin queue={} create_if_missing={}",
                    fmt::ptr(static_cast<const void*>(queue)),
                    create_if_missing));

    auto _state = get_queue_registry().rlock([&](const auto& registry) -> queue_state_ptr_t {
        if(auto it = registry.find(queue); it != registry.end()) return it->second;
        return queue_state_ptr_t{};
    });

    // if create_if_missing is true, create a new state. this is for dynamic discovery of queues.
    if(!_state && create_if_missing)
    {
        queue_interposition_debug_log(
            fmt::format("lookup_queue_state creating dynamic state queue={}",
                        fmt::ptr(static_cast<const void*>(queue))));
        return create_queue_state(queue, true);
    }

    queue_interposition_debug_log(fmt::format("lookup_queue_state end queue={} found={} {}",
                                              fmt::ptr(static_cast<const void*>(queue)),
                                              static_cast<bool>(_state),
                                              debug_state_summary(_state.get())));

    return _state;
}

queue_state_ptr_t
lookup_queue_state_by_doorbell(hsa_signal_t signal, bool create_if_missing)
{
    queue_interposition_debug_log(
        fmt::format("lookup_queue_state_by_doorbell begin signal={} create_if_missing={}",
                    signal.handle,
                    create_if_missing));

    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    const auto* _amd_signal = reinterpret_cast<amd_signal_t*>(signal.handle);

    if(!_amd_signal)
    {
        queue_interposition_debug_log("lookup_queue_state_by_doorbell no amd signal");
        return queue_state_ptr_t{};
    }

    // Only doorbell-kind signals carry a valid queue_ptr (it aliases reserved2 otherwise).
    if(_amd_signal->kind != AMD_SIGNAL_KIND_DOORBELL &&
       _amd_signal->kind != AMD_SIGNAL_KIND_LEGACY_DOORBELL)
    {
        queue_interposition_debug_log(
            fmt::format("lookup_queue_state_by_doorbell non-doorbell signal={} kind={}",
                        signal.handle,
                        _amd_signal->kind));
        return queue_state_ptr_t{};
    }

    if(_amd_signal->queue_ptr)
    {
        queue_interposition_debug_log(
            fmt::format("lookup_queue_state_by_doorbell signal={} queue_ptr={}",
                        signal.handle,
                        fmt::ptr(reinterpret_cast<const void*>(_amd_signal->queue_ptr))));
        return lookup_queue_state(reinterpret_cast<const hsa_queue_t*>(_amd_signal->queue_ptr),
                                  create_if_missing);
    }

    queue_interposition_debug_log(
        fmt::format("lookup_queue_state_by_doorbell signal={} has no queue_ptr", signal.handle));
    return queue_state_ptr_t{};
}

uint64_t
add_write_index_impl(QueueState* state, uint64_t value, std::memory_order order)
{
    auto prev = state->virtual_wptr.fetch_add(value, order);
    queue_interposition_debug_log(
        fmt::format("add_write_index state={} value={} previous={} next={} order={}",
                    fmt::ptr(state),
                    value,
                    prev,
                    prev + value,
                    static_cast<int>(order)));
    return prev;
}

void
store_write_index_impl(QueueState* state, uint64_t value, std::memory_order order)
{
    auto previous = state->virtual_wptr.load(std::memory_order_acquire);
    state->virtual_wptr.store(value, order);
    queue_interposition_debug_log(
        fmt::format("store_write_index state={} previous={} value={} order={}",
                    fmt::ptr(state),
                    previous,
                    value,
                    static_cast<int>(order)));
}

uint64_t
cas_write_index_impl(QueueState* state, uint64_t expected, uint64_t value, std::memory_order order)
{
    uint64_t prev      = expected;
    auto     exchanged = state->virtual_wptr.compare_exchange_strong(prev, value, order);
    queue_interposition_debug_log(fmt::format(
        "cas_write_index state={} expected={} desired={} observed={} exchanged={} order={}",
        fmt::ptr(state),
        expected,
        value,
        prev,
        exchanged,
        static_cast<int>(order)));
    return prev;
}

uint64_t
load_write_index_impl(const QueueState* state, std::memory_order order)
{
    auto value = state->virtual_wptr.load(order);
    queue_interposition_debug_log(fmt::format("load_write_index state={} value={} order={}",
                                              fmt::ptr(state),
                                              value,
                                              static_cast<int>(order)));
    return value;
}

namespace
{
// CPU pause hint for short spin-waits (cheaper than yield/sleep, no added latency).
inline void
cpu_relax()
{
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__)
    asm volatile("yield" ::: "memory");
#else
    std::this_thread::yield();
#endif
}

// Per-thread handoff from process_doorbell_impl() to ring_buffer_writer().
struct doorbell_tls_t
{
    QueueState*          state                     = nullptr;
    uint64_t             submit_pos                = 0;
    uint32_t             pkt_size                  = 64;
    const doorbell_fn_t* ring_doorbell             = nullptr;
    uint64_t             last_published_submit_pos = 0;
};

doorbell_tls_t&
get_doorbell_tls()
{
    static thread_local auto _v = doorbell_tls_t{};
    return _v;
}

inline void
publish_submitted_packets(QueueState* state, uint64_t submit_pos)
{
    auto& tls = get_doorbell_tls();
    if(!tls.ring_doorbell || submit_pos == 0)
    {
        queue_interposition_debug_log(
            fmt::format("publish_submitted_packets skipped missing-doorbell state={} submit_pos={} "
                        "last_published_submit_pos={} ring_doorbell={}",
                        fmt::ptr(state),
                        submit_pos,
                        tls.last_published_submit_pos,
                        fmt::ptr(static_cast<const void*>(tls.ring_doorbell))));
        return;
    }

    if(submit_pos <= tls.last_published_submit_pos) return;

    // submit_pos must never regress below what we already published (corruption); fatal in CI.
    ROCP_CI_LOG_IF(WARNING, submit_pos < tls.last_published_submit_pos)
        << "publish_submitted_packets: submit_pos (" << submit_pos
        << ") regressed below last_published_submit_pos (" << tls.last_published_submit_pos << ")";

    queue_interposition_debug_log(fmt::format(
        "publish_submitted_packets begin state={} submit_pos={} last_published_submit_pos={} "
        "doorbell_value={} {}",
        fmt::ptr(state),
        submit_pos,
        tls.last_published_submit_pos,
        submit_pos - 1,
        debug_state_summary(state)));

    __atomic_store_n(state->real_wdid, submit_pos, __ATOMIC_RELEASE);
    (*tls.ring_doorbell)(state->doorbell_signal, static_cast<hsa_signal_value_t>(submit_pos - 1));
    tls.last_published_submit_pos = submit_pos;

    queue_interposition_debug_log(fmt::format(
        "publish_submitted_packets end state={} submit_pos={} last_published_submit_pos={} {}",
        fmt::ptr(state),
        submit_pos,
        tls.last_published_submit_pos,
        debug_state_summary(state)));
}

inline void
wait_for_free_slot(QueueState* state, uint64_t submit_pos)
{
    auto spin_count = uint64_t{0};
    auto spin_start = std::chrono::steady_clock::now();

    while(true)
    {
        auto real_rdid = __atomic_load_n(state->real_rdid, __ATOMIC_ACQUIRE);
        auto ring_used = submit_pos - real_rdid;
        if(ring_used < state->ring_size)
        {
            if(spin_count > 0)
            {
                queue_interposition_debug_log(fmt::format(
                    "wait_for_free_slot released state={} submit_pos={} real_rdid={} ring_used={} "
                    "ring_size={} spins={} elapsed_us={}",
                    fmt::ptr(state),
                    submit_pos,
                    real_rdid,
                    ring_used,
                    state->ring_size,
                    spin_count,
                    debug_elapsed_us(spin_start)));
            }
            return;
        }

        if(spin_count == 0)
        {
            queue_interposition_debug_log(fmt::format(
                "wait_for_free_slot full-ring begin state={} submit_pos={} real_rdid={} "
                "ring_used={} ring_size={} {}",
                fmt::ptr(state),
                submit_pos,
                real_rdid,
                ring_used,
                state->ring_size,
                debug_state_summary(state)));
        }
        else if(spin_count % (1UL << 20) == 0)
        {
            queue_interposition_debug_log(fmt::format(
                "wait_for_free_slot still full state={} submit_pos={} real_rdid={} ring_used={} "
                "ring_size={} spins={} elapsed_us={}",
                fmt::ptr(state),
                submit_pos,
                real_rdid,
                ring_used,
                state->ring_size,
                spin_count,
                debug_elapsed_us(spin_start)));
        }

        // If the producer is blocked on a full ring and has already written
        // packets beyond the last visible write index, publish progress so the
        // consumer can observe and drain them.
        publish_submitted_packets(state, submit_pos);
        ++spin_count;
        cpu_relax();
    }
}

void
ring_buffer_writer(const void* pkts, uint64_t pkt_count)
{
    auto&       tls      = get_doorbell_tls();
    auto*       state    = tls.state;
    auto        pkt_size = tls.pkt_size;
    const auto* src      = static_cast<const char*>(pkts);

    queue_interposition_debug_log(fmt::format(
        "ring_buffer_writer begin pkts={} pkt_count={} state={} submit_start={} pkt_size={} {}",
        fmt::ptr(pkts),
        pkt_count,
        fmt::ptr(state),
        tls.submit_pos,
        pkt_size,
        debug_state_summary(state)));

    for(uint64_t i = 0; i < pkt_count; i++)
    {
        wait_for_free_slot(state, tls.submit_pos);
        auto        slot   = tls.submit_pos & state->ring_mask;
        auto*       dst    = static_cast<char*>(state->ring_buf) + (slot * pkt_size);
        const auto* s      = src + i * pkt_size;
        auto        header = uint16_t{0};
        if(pkt_size >= sizeof(header)) ::memcpy(&header, s, sizeof(header));

        if(i < 4 || i + 1 == pkt_count || i % 1024 == 0)
        {
            queue_interposition_debug_log(fmt::format(
                "ring_buffer_writer packet index={} pkt_count={} submit_pos={} slot={} src={} "
                "dst={} header=0x{:04x}",
                i,
                pkt_count,
                tls.submit_pos,
                slot,
                fmt::ptr(static_cast<const void*>(s)),
                fmt::ptr(static_cast<const void*>(dst)),
                header));
        }

        if(dst != s)
        {
            constexpr auto header_size = sizeof(uint16_t);
            if(pkt_size > header_size)
            {
                ::memcpy(dst + header_size, s + header_size, pkt_size - header_size);
                uint16_t header = 0;
                ::memcpy(&header, s, header_size);
                __atomic_store_n(reinterpret_cast<uint16_t*>(dst), header, __ATOMIC_RELEASE);
            }
            else
            {
                ::memcpy(dst, s, pkt_size);
            }
        }
        tls.submit_pos++;
    }

    queue_interposition_debug_log(
        fmt::format("ring_buffer_writer end state={} pkt_count={} submit_end={} {}",
                    fmt::ptr(state),
                    pkt_count,
                    tls.submit_pos,
                    debug_state_summary(state)));
}

auto
async_signal_handler_exists()
{
    return common::static_object<internal_threading::task_group_t>::get();
}

internal_threading::task_group_t*
get_async_signal_handler()
{
    using task_group_t           = internal_threading::task_group_t;
    using create_task_group_fn_t = task_group_t* (*) (void*, size_t);

    const auto gpu_thread_count = common::get_env("GPU_MAX_HW_QUEUES", static_cast<int64_t>(4));
    const auto requested_thread_count =
        common::get_env("ROCPROFILER_ASYNC_SIGNAL_HANDLER_THREADS", gpu_thread_count);

    queue_interposition_debug_log(
        fmt::format("get_async_signal_handler begin GPU_MAX_HW_QUEUES_resolved={} "
                    "ROCPROFILER_ASYNC_SIGNAL_HANDLER_THREADS_resolved={}",
                    gpu_thread_count,
                    requested_thread_count));

    // default to 4 threads if neither GPU_MAX_HW_QUEUES or ROCPROFILER_ASYNC_SIGNAL_HANDLER_THREADS
    // is set, since the async signal handler is primarily intended for handling queue completion
    // signals and a typical GPU may have on the order of 4 hardware queues. Note: GPU_MAX_HW_QUEUES
    // is a ROCr/HSA environment variable. If GPU_MAX_HW_QUEUES is set but
    // ROCPROFILER_ASYNC_SIGNAL_HANDLER_THREADS is not set, we will use the value of
    // GPU_MAX_HW_QUEUES to determine the number of threads for the async signal handler. If
    // ROCPROFILER_ASYNC_SIGNAL_HANDLER_THREADS is set, it will take precedence over
    // GPU_MAX_HW_QUEUES.
    static auto*& _v =
        common::static_object<internal_threading::task_group_t>::construct_via_function(
            static_cast<create_task_group_fn_t>(&internal_threading::create_task_group),
            requested_thread_count);

    queue_interposition_debug_log(
        fmt::format("get_async_signal_handler end task_group={}", fmt::ptr(_v)));
    return _v;
}

bool
context_filter(const context::context* ctx)
{
    return (ctx->is_tracing_one_of(ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH,
                                   ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH));
}

template <typename Integral>
Integral
bit_extract(Integral x, int first, int last)
{
    static_assert(std::is_integral<Integral>::value, "Integral type required");

    auto&& bit_mask = [](int _first, int _last) {
        ROCP_FATAL_IF(!(_last >= _first)) << fmt::format(
            "[queue::bit_extract::bit_mask] -> invalid argument. last (={}) is not >= first (={})",
            _last,
            _first);

        size_t num_bits = _last - _first + 1;
        return ((num_bits >= sizeof(Integral) * 8) ? ~Integral{0}
                                                   /* num_bits exceed the size of Integral */
                                                   : ((Integral{1} << num_bits) - 1))
               << _first;
    };

    return (x >> first) & bit_mask(0, last - first);
}

void
async_signal_handler(hsa_signal_t                            completion_signal,
                     hsa_signal_value_t                      starting_value,
                     std::shared_ptr<queue_info_session_t>&& session,
                     uint64_t                                debug_waiter_id)
{
    constexpr auto timeout_hint =
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::microseconds{10});

    auto signal_value = starting_value;
    auto niterations  = uint64_t{0};
    auto wait_start   = std::chrono::steady_clock::now();
    auto session_queue =
        session ? static_cast<const void*>(session->queue.intercept_queue()) : nullptr;

    queue_interposition_debug_log(fmt::format(
        "async_signal_handler begin waiter_id={} signal={} starting_value={} session={} "
        "session_packets={} queue_id={} queue={} tid={}",
        debug_waiter_id,
        completion_signal.handle,
        starting_value,
        fmt::ptr(session.get()),
        session ? session->packet_data.size() : 0,
        session ? session->queue.get_id().handle : 0,
        fmt::ptr(session_queue),
        session ? session->tid : 0));

    // Stop only on completion or finalization; never run cleanup while the kernel is live.
    while(true)
    {
        signal_value = get_core_table()->hsa_signal_wait_relaxed_fn(completion_signal,
                                                                    HSA_SIGNAL_CONDITION_LT,
                                                                    starting_value,
                                                                    timeout_hint.count(),
                                                                    HSA_WAIT_STATE_ACTIVE);

        if(signal_value < starting_value) break;         // kernel completed
        if(registration::get_fini_status() != 0) break;  // tearing down: run cleanup path
        ++niterations;

        // Surface long-running waits for diagnostics without giving up the wait.
        constexpr auto warn_interval = (1UL << 20);
        if(niterations <= 4 || niterations % warn_interval == 0)
            ROCP_WARNING << fmt::format(
                "[QI-DEBUG tid={}] async_signal_handler waiting waiter_id={} signal={{.handle={}}} "
                "iterations={} value={} starting_value={} elapsed_us={} fini_status={}",
                common::get_tid(),
                debug_waiter_id,
                completion_signal.handle,
                niterations,
                signal_value,
                starting_value,
                debug_elapsed_us(wait_start),
                registration::get_fini_status());
    }

    ROCP_WARNING << fmt::format(
        "[QI-DEBUG tid={}] async_signal_handler wait-finished waiter_id={} signal={{.handle={}}} "
        "value={} starting_value={} iterations={} elapsed_us={} fini_status={}",
        common::get_tid(),
        debug_waiter_id,
        completion_signal.handle,
        signal_value,
        starting_value,
        niterations,
        debug_elapsed_us(wait_start),
        registration::get_fini_status());

    if(auto delay_us = common::get_env("ROCPROFILER_TEST_INLINE_ASYNC_DELAY_US", 0); delay_us > 0)
    {
        queue_interposition_debug_log(fmt::format(
            "async_signal_handler test-delay waiter_id={} delay_us={}", debug_waiter_id, delay_us));
        std::this_thread::sleep_for(std::chrono::microseconds{delay_us});
    }

    for(size_t packet_idx = 0; packet_idx < session->packet_data.size(); ++packet_idx)
    {
        auto& packet = session->packet_data.at(packet_idx);
        queue_interposition_debug_log(
            fmt::format("async_signal_handler completion-callback begin waiter_id={} packet_idx={} "
                        "dispatch_id={} completion_signal={} pooled_signal={}",
                        debug_waiter_id,
                        packet_idx,
                        packet.callback_record.dispatch_info.dispatch_id,
                        packet.completion_signal.handle,
                        fmt::ptr(packet.pooled_signal)));

        auto dispatch_time = kernel_dispatch::get_dispatch_time(*session, packet);
        kernel_dispatch::dispatch_complete(*session, packet, dispatch_time);

        // if the completion signal was from the pool, we just release it back to the pool for
        // reuse.
        if(packet.pooled_signal)
        {
            Queue::release_signal(packet.pooled_signal);
        }
        else
        {
            // if the signal was not from the pool, we need to decrement the signal value to clean
            // up the signal for the application
            get_core_table()->hsa_signal_subtract_relaxed_fn(packet.completion_signal, 1);
        }

        // we need to decrement this reference count at the end of the functions
        auto* _corr_id = session->correlation_id;
        if(_corr_id)
        {
            ROCP_FATAL_IF(_corr_id->get_ref_count() == 0)
                << "reference counter for correlation id " << _corr_id->internal << " from thread "
                << _corr_id->thread_idx << " has no reference count";
            _corr_id->sub_kern_count();
            _corr_id->sub_ref_count();
        }

        queue_interposition_debug_log(
            fmt::format("async_signal_handler completion-callback end waiter_id={} packet_idx={} "
                        "dispatch_id={}",
                        debug_waiter_id,
                        packet_idx,
                        packet.callback_record.dispatch_info.dispatch_id));
    }

    queue_interposition_debug_log(
        fmt::format("async_signal_handler end waiter_id={} signal={} total_elapsed_us={}",
                    debug_waiter_id,
                    completion_signal.handle,
                    debug_elapsed_us(wait_start)));
}

// Local kernel-dispatch tracing path: swaps in pooled completion signals,
// runs KERNEL_DISPATCH_ENQUEUE tracer hooks, and enqueues a completion-signal
// waiter on the async signal handler pool. Strict 1:1 packet forwarding; does
// not insert PM4 packets. Distinct from Queue::WriteInterceptor (legacy path).
void
write_interceptor(Queue*                                queue,
                  const void*                           packets,
                  uint64_t                              pkt_count,
                  hsa_amd_queue_intercept_packet_writer writer)
{
    using callback_record_t = packet_data_t::callback_record_t;
    using packet_vector_t   = common::container::small_vector<rocprofiler_packet, 512>;

    if(registration::get_fini_status() > 0)
    {
        queue_interposition_debug_log(
            fmt::format("write_interceptor bypass fini_status={} queue={} packets={} pkt_count={}",
                        registration::get_fini_status(),
                        fmt::ptr(queue),
                        fmt::ptr(packets),
                        pkt_count));
        writer(packets, pkt_count);
        return;
    }

    queue_interposition_debug_log(fmt::format(
        "write_interceptor begin queue={} queue_id={} packets={} pkt_count={} fini_status={}",
        fmt::ptr(queue),
        queue ? queue->get_id().handle : 0,
        fmt::ptr(packets),
        pkt_count,
        registration::get_fini_status()));

    auto _contexts = context::get_active_contexts(context_filter);
    queue_interposition_debug_log(
        fmt::format("write_interceptor active-contexts queue={} pkt_count={} context_count={}",
                    fmt::ptr(queue),
                    pkt_count,
                    _contexts.size()));

    // We have no packets or no one who needs to be notified, do nothing.
    if(pkt_count == 0 || _contexts.empty())
    {
        queue_interposition_debug_log(fmt::format(
            "write_interceptor direct-writer no-work queue={} pkt_count={} context_count={}",
            fmt::ptr(queue),
            pkt_count,
            _contexts.size()));
        writer(packets, pkt_count);
        return;
    }

    // unique sequence id for the dispatch (global across all queues, matches SDK contract)
    static auto sequence_counter = std::atomic<rocprofiler_dispatch_id_t>{0};

    const auto* packets_arr          = static_cast<const rocprofiler_packet*>(packets);
    auto        num_dispatch_packets = size_t{0};
    for(size_t i = 0; i < pkt_count; ++i)
    {
        const auto& original_packet = packets_arr[i].kernel_dispatch;
        auto        packet_type     = bit_extract(original_packet.header,
                                       HSA_PACKET_HEADER_TYPE,
                                       HSA_PACKET_HEADER_TYPE + HSA_PACKET_HEADER_WIDTH_TYPE - 1);
        if(packet_type == HSA_PACKET_TYPE_KERNEL_DISPATCH)
        {
            ++num_dispatch_packets;
        }
    }

    queue_interposition_debug_log(
        fmt::format("write_interceptor packet-scan queue={} pkt_count={} dispatch_packets={}",
                    fmt::ptr(queue),
                    pkt_count,
                    num_dispatch_packets));

    if(num_dispatch_packets == 0)
    {
        queue_interposition_debug_log(
            fmt::format("write_interceptor direct-writer no-dispatch queue={} pkt_count={}",
                        fmt::ptr(queue),
                        pkt_count));
        writer(packets, pkt_count);
        return;
    }

    auto tracing_data_v = tracing::tracing_data{};
    tracing::populate_contexts(ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH,
                               ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH,
                               tracing_data_v);

    // all packets should have the same correlation id so we can just look at the first one to
    // get the correlation id for the entire batch of packets
    auto*                    corr_id      = context::get_latest_correlation_id();
    context::correlation_id* _corr_id_pop = nullptr;

    // Allocate a correlation id if we have at least one dispatch packet and we don't have a
    // correlation id already. There will not be a correlation id if there is no API tracing but
    // it was requested by tools to always provide one.
    if(!corr_id)
    {
        constexpr auto ref_count = 1;
        corr_id                  = context::correlation_tracing_service::construct(ref_count);
        _corr_id_pop             = corr_id;
        queue_interposition_debug_log(fmt::format("write_interceptor constructed-correlation-id "
                                                  "queue={} corr_id={} thread_idx={} internal={}",
                                                  fmt::ptr(queue),
                                                  fmt::ptr(corr_id),
                                                  corr_id ? corr_id->thread_idx : 0,
                                                  corr_id ? corr_id->internal : 0));
    }

    // During finalization, correlation tracing service will not construct a correlation id so
    // just write packet through without tracing
    if(!corr_id)
    {
        queue_interposition_debug_log(
            fmt::format("write_interceptor direct-writer no-correlation queue={} pkt_count={}",
                        fmt::ptr(queue),
                        pkt_count));
        writer(packets, pkt_count);
        return;
    }

    // if we constructed a correlation id, this decrements the reference count after the
    // underlying function returns
    auto _corr_id_dtor = common::scope_destructor{[_corr_id_pop]() {
        if(_corr_id_pop)
        {
            context::pop_latest_correlation_id(_corr_id_pop);
            _corr_id_pop->sub_ref_count();
        }
    }};

    using packet_writer_fn_t = std::function<void(packet_vector_t &&)>;

    auto process_packet_batch = [&queue, &corr_id, tracing_data_v](
                                    const rocprofiler_packet* _packets,
                                    uint64_t                  _num_packets,
                                    const packet_writer_fn_t& _writer) {
        static constexpr auto null_signal    = hsa_signal_t{.handle = 0};
        const auto            batch_debug_id = next_queue_interposition_debug_event_id();

        queue_interposition_debug_log(fmt::format(
            "process_packet_batch begin batch_id={} queue={} queue_id={} packets={} num_packets={} "
            "corr_id={}",
            batch_debug_id,
            fmt::ptr(queue),
            queue ? queue->get_id().handle : 0,
            fmt::ptr(_packets),
            _num_packets,
            fmt::ptr(corr_id)));

        auto transformed_packets = packet_vector_t{};

        auto thr_id           = (corr_id) ? corr_id->thread_idx : common::get_tid();
        auto internal_corr_id = (corr_id) ? corr_id->internal : 0;
        auto ancestor_corr_id = (corr_id) ? corr_id->ancestor : 0;

        using packet_data_array_t = queue_info_session_t::packet_data_array_t;

        auto _info_session = queue_info_session_t{.queue          = *queue,
                                                  .tid            = thr_id,
                                                  .enqueue_ts     = common::timestamp_ns(),
                                                  .correlation_id = corr_id,
                                                  .packet_data    = packet_data_array_t{}};

        // Searching across all the packets given during this write
        for(size_t i = 0; i < _num_packets; ++i)
        {
            const auto& original_packet = _packets[i].kernel_dispatch;
            auto        packet_type =
                bit_extract(original_packet.header,
                            HSA_PACKET_HEADER_TYPE,
                            HSA_PACKET_HEADER_TYPE + HSA_PACKET_HEADER_WIDTH_TYPE - 1);
            if(packet_type != HSA_PACKET_TYPE_KERNEL_DISPATCH)
            {
                if(i < 4 || i + 1 == _num_packets)
                    queue_interposition_debug_log(fmt::format(
                        "process_packet_batch passthrough batch_id={} index={} packet_type={}",
                        batch_debug_id,
                        i,
                        packet_type));
                transformed_packets.emplace_back(_packets[i]);
                continue;
            }

            queue_interposition_debug_log(fmt::format(
                "process_packet_batch dispatch begin batch_id={} index={} kernel_object=0x{:x} "
                "completion_signal={} header=0x{:04x}",
                batch_debug_id,
                i,
                original_packet.kernel_object,
                original_packet.completion_signal.handle,
                original_packet.header));

            // increase the reference count to denote that this correlation id is being used in a
            // kernel
            corr_id->add_ref_count();
            corr_id->add_kern_count();

            auto _packet_data = packet_data_t{};

            // make a copy of the tracing data
            _packet_data.tracing_data = tracing_data_v;

            tracing::populate_external_correlation_ids(
                _packet_data.tracing_data.external_correlation_ids,
                thr_id,
                ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_KERNEL_DISPATCH,
                ROCPROFILER_KERNEL_DISPATCH_ENQUEUE,
                internal_corr_id);

            const uint64_t kernel_id = code_object::get_kernel_id(original_packet.kernel_object);
            const auto     original_completion_signal = original_packet.completion_signal;
            const auto     existing_completion_signal = (original_completion_signal != null_signal);

            queue_interposition_debug_log(fmt::format(
                "process_packet_batch dispatch metadata batch_id={} index={} kernel_id={} "
                "existing_completion_signal={} corr_internal={} corr_ancestor={} thr_id={}",
                batch_debug_id,
                i,
                kernel_id,
                existing_completion_signal,
                internal_corr_id,
                ancestor_corr_id,
                thr_id));

            // Copy kernel pkt, copy is to allow for signal to be modified
            _packet_data.kernel_packet = _packets[i];
            // create a reference for short hand access
            auto& kernel_packet     = _packet_data.kernel_packet;
            auto& completion_signal = _packet_data.kernel_packet.kernel_dispatch.completion_signal;

            auto create_signal = [](auto* signal) -> common::container::pool_object<signal_t>* {
                if(auto* pool = get_signal_pool(); pool && signal->handle == 0)
                {
                    auto& _signal = pool->acquire(construct_hsa_signal, 0, 0, nullptr, 0);
                    ROCP_FATAL_IF(!_signal.in_use())
                        << "Acquired signal from pool that is not in use";
                    ROCP_FATAL_IF(_signal.get().value == null_signal)
                        << "Acquired signal from pool that has invalid handle";
                    *CHECK_NOTNULL(signal) = _signal.get().value;
                    return &_signal;
                }
                return nullptr;
            };

            // No barrier packet: borrow a pooled signal if needed, then bump value by 1.
            if(!existing_completion_signal)
                _packet_data.pooled_signal = create_signal(&completion_signal);

            get_core_table()->hsa_signal_add_scacq_screl_fn(completion_signal, 1);
            auto signal_value_after_add =
                get_core_table()->hsa_signal_load_scacquire_fn(completion_signal);
            queue_interposition_debug_log(fmt::format(
                "process_packet_batch signal-prepared batch_id={} index={} completion_signal={} "
                "pooled_signal={} value_after_add={}",
                batch_debug_id,
                i,
                completion_signal.handle,
                fmt::ptr(_packet_data.pooled_signal),
                signal_value_after_add));

            // set the completion signal to the kernel packet
            _packet_data.completion_signal = completion_signal;

            // computes the "size" based on the offset of reserved_padding field
            constexpr auto kernel_dispatch_info_rt_size =
                common::compute_runtime_sizeof<rocprofiler_kernel_dispatch_info_t>();

            static_assert(kernel_dispatch_info_rt_size < sizeof(rocprofiler_kernel_dispatch_info_t),
                          "failed to compute size field based on offset of reserved_padding field");

            auto dispatch_id             = ++sequence_counter;
            _packet_data.callback_record = callback_record_t{
                sizeof(callback_record_t),
                rocprofiler_timestamp_t{0},
                rocprofiler_timestamp_t{0},
                rocprofiler_kernel_dispatch_info_t{
                    .size                 = kernel_dispatch_info_rt_size,
                    .agent_id             = queue->get_agent().get_rocp_agent()->id,
                    .queue_id             = queue->get_id(),
                    .kernel_id            = kernel_id,
                    .dispatch_id          = dispatch_id,
                    .private_segment_size = kernel_packet.kernel_dispatch.private_segment_size,
                    .group_segment_size   = kernel_packet.kernel_dispatch.group_segment_size,
                    .workgroup_size =
                        rocprofiler_dim3_t{kernel_packet.kernel_dispatch.workgroup_size_x,
                                           kernel_packet.kernel_dispatch.workgroup_size_y,
                                           kernel_packet.kernel_dispatch.workgroup_size_z},
                    .grid_size = rocprofiler_dim3_t{kernel_packet.kernel_dispatch.grid_size_x,
                                                    kernel_packet.kernel_dispatch.grid_size_y,
                                                    kernel_packet.kernel_dispatch.grid_size_z},
                    .reserved_padding = {0}}};

            queue_interposition_debug_log(fmt::format(
                "process_packet_batch dispatch-record batch_id={} index={} dispatch_id={} "
                "queue_id={} agent_id_handle={}",
                batch_debug_id,
                i,
                dispatch_id,
                queue->get_id().handle,
                queue->get_agent().get_rocp_agent()->id.handle));

            {
                auto tracer_data = _packet_data.callback_record;
                queue_interposition_debug_log(fmt::format(
                    "process_packet_batch enqueue-enter-callbacks begin batch_id={} index={} "
                    "dispatch_id={}",
                    batch_debug_id,
                    i,
                    dispatch_id));
                tracing::execute_phase_enter_callbacks(
                    _packet_data.tracing_data.callback_contexts,
                    thr_id,
                    internal_corr_id,
                    _packet_data.tracing_data.external_correlation_ids,
                    ancestor_corr_id,
                    ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH,
                    ROCPROFILER_KERNEL_DISPATCH_ENQUEUE,
                    tracer_data);
                queue_interposition_debug_log(fmt::format(
                    "process_packet_batch enqueue-enter-callbacks end batch_id={} index={} "
                    "dispatch_id={}",
                    batch_debug_id,
                    i,
                    dispatch_id));
            }

            // map all the external correlation ids (after enqueue enter phase) for all the contexts
            // captured by the info session
            tracing::update_external_correlation_ids(
                _packet_data.tracing_data.external_correlation_ids,
                thr_id,
                ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_KERNEL_DISPATCH);

            // Stores the instrumentation pkt (i.e. AQL packets for counter collection)
            // along with an ID of the client we got the packet from (this will be returned via
            // completed_cb_t)

            // emplace the kernel packet
            transformed_packets.emplace_back(kernel_packet);

            ROCP_FATAL_IF(packet_type != HSA_PACKET_TYPE_KERNEL_DISPATCH)
                << "get_kernel_id below might need to be updated";

            {
                auto tracer_data = _packet_data.callback_record;
                queue_interposition_debug_log(fmt::format(
                    "process_packet_batch enqueue-exit-callbacks begin batch_id={} index={} "
                    "dispatch_id={}",
                    batch_debug_id,
                    i,
                    dispatch_id));
                tracing::execute_phase_exit_callbacks(
                    _packet_data.tracing_data.callback_contexts,
                    _packet_data.tracing_data.external_correlation_ids,
                    ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH,
                    ROCPROFILER_KERNEL_DISPATCH_ENQUEUE,
                    tracer_data);
                queue_interposition_debug_log(fmt::format(
                    "process_packet_batch enqueue-exit-callbacks end batch_id={} index={} "
                    "dispatch_id={}",
                    batch_debug_id,
                    i,
                    dispatch_id));
            }

            _info_session.packet_data.emplace_back(std::move(_packet_data));
            queue_interposition_debug_log(
                fmt::format("process_packet_batch dispatch end batch_id={} index={} dispatch_id={} "
                            "session_packet_count={}",
                            batch_debug_id,
                            i,
                            dispatch_id,
                            _info_session.packet_data.size()));
        }

        if(!_info_session.packet_data.empty())
        {
            auto last_completion_signal = _info_session.packet_data.back().completion_signal;

            ROCP_FATAL_IF(last_completion_signal == null_signal)
                << "invalid completion signal in the last packet of the batch";

            auto current_signal_value =
                get_core_table()->hsa_signal_load_scacquire_fn(last_completion_signal);

            const auto debug_waiter_id = next_async_waiter_debug_id();
            queue_interposition_debug_log(
                fmt::format("process_packet_batch waiter-ready batch_id={} waiter_id={} signal={} "
                            "current_signal_value={} session_packets={} transformed_packets={}",
                            batch_debug_id,
                            debug_waiter_id,
                            last_completion_signal.handle,
                            current_signal_value,
                            _info_session.packet_data.size(),
                            transformed_packets.size()));

            auto _shared_info_session =
                std::make_shared<queue_info_session_t>(std::move(_info_session));
            auto async_schedule_start = std::chrono::steady_clock::now();
            queue_interposition_debug_log(
                fmt::format("process_packet_batch async-handler-get begin batch_id={} waiter_id={}",
                            batch_debug_id,
                            debug_waiter_id));
            auto* async_handler = get_async_signal_handler();
            queue_interposition_debug_log(fmt::format(
                "process_packet_batch async-handler-get end batch_id={} waiter_id={} handler={} "
                "elapsed_us={}",
                batch_debug_id,
                debug_waiter_id,
                fmt::ptr(async_handler),
                debug_elapsed_us(async_schedule_start)));

            queue_interposition_debug_log(fmt::format(
                "process_packet_batch async-schedule begin batch_id={} waiter_id={} handler={} "
                "signal={} expected_value={}",
                batch_debug_id,
                debug_waiter_id,
                fmt::ptr(async_handler),
                last_completion_signal.handle,
                current_signal_value));
            async_handler->async([_signal_v          = last_completion_signal,
                                  _expected_signal_v = current_signal_value,
                                  _session_v         = std::move(_shared_info_session),
                                  debug_waiter_id]() mutable {
                async_signal_handler(
                    _signal_v, _expected_signal_v, std::move(_session_v), debug_waiter_id);
            });
            queue_interposition_debug_log(fmt::format(
                "process_packet_batch async-schedule end batch_id={} waiter_id={} elapsed_us={}",
                batch_debug_id,
                debug_waiter_id,
                debug_elapsed_us(async_schedule_start)));
        }

        auto writer_start = std::chrono::steady_clock::now();
        queue_interposition_debug_log(fmt::format(
            "process_packet_batch writer begin batch_id={} transformed_packets={} writer={}",
            batch_debug_id,
            transformed_packets.size(),
            fmt::ptr(&_writer)));
        _writer(std::move(transformed_packets));
        queue_interposition_debug_log(
            fmt::format("process_packet_batch writer end batch_id={} elapsed_us={}",
                        batch_debug_id,
                        debug_elapsed_us(writer_start)));
    };

    ROCP_TRACE_IF(pkt_count > 1) << fmt::format(
        "[{}] Batching packets. Number of packets = {}", __FUNCTION__, pkt_count);

    process_packet_batch(packets_arr, pkt_count, [&writer](packet_vector_t&& _packets) {
        queue_interposition_debug_log(
            fmt::format("write_interceptor final-writer lambda begin packets={} count={}",
                        fmt::ptr(_packets.data()),
                        _packets.size()));
        writer(_packets.data(), _packets.size());
        queue_interposition_debug_log(
            fmt::format("write_interceptor final-writer lambda end packets={} count={}",
                        fmt::ptr(_packets.data()),
                        _packets.size()));
    });

    queue_interposition_debug_log(
        fmt::format("write_interceptor end queue={} pkt_count={}", fmt::ptr(queue), pkt_count));
}
}  // namespace

void
process_doorbell_impl(const queue_state_ptr_t& state,
                      hsa_signal_value_t       value,
                      const doorbell_fn_t&     ring_doorbell)
{
    const auto doorbell_debug_id = next_queue_interposition_debug_event_id();
    queue_interposition_debug_log(fmt::format(
        "process_doorbell_impl enter doorbell_id={} state_shared={} value={} ring_doorbell={}",
        doorbell_debug_id,
        fmt::ptr(state.get()),
        value,
        fmt::ptr(static_cast<const void*>(&ring_doorbell))));

    if(!state)
    {
        queue_interposition_debug_log(
            fmt::format("process_doorbell_impl null-state doorbell_id={}", doorbell_debug_id));
        return;
    }

    auto* state_ptr = state.get();

    // gate_lock serializes doorbell processing; producers never take it, so no deadlock.
    auto lock_start = std::chrono::steady_clock::now();
    queue_interposition_debug_log(
        fmt::format("process_doorbell_impl gate-lock wait begin doorbell_id={} {}",
                    doorbell_debug_id,
                    debug_state_summary(state_ptr)));
    std::unique_lock<std::mutex> lock{state_ptr->gate_lock};
    queue_interposition_debug_log(
        fmt::format("process_doorbell_impl gate-lock acquired doorbell_id={} elapsed_us={} {}",
                    doorbell_debug_id,
                    debug_elapsed_us(lock_start),
                    debug_state_summary(state_ptr)));

    const uint64_t scan_pos = state_ptr->next_scan_pos;

    const uint64_t wptr_end = state_ptr->virtual_wptr.load(std::memory_order_acquire);
    queue_interposition_debug_log(fmt::format(
        "process_doorbell_impl scan-window doorbell_id={} scan_pos={} wptr_end={} pending={} {}",
        doorbell_debug_id,
        scan_pos,
        wptr_end,
        (wptr_end >= scan_pos) ? (wptr_end - scan_pos) : 0,
        debug_state_summary(state_ptr)));

    if(scan_pos >= wptr_end)
    {
        queue_interposition_debug_log(fmt::format(
            "process_doorbell_impl no-pending-packets doorbell_id={} ringing-original value={}",
            doorbell_debug_id,
            value));
        ring_doorbell(state_ptr->doorbell_signal, value);
        queue_interposition_debug_log(fmt::format(
            "process_doorbell_impl no-pending-packets done doorbell_id={}", doorbell_debug_id));
        return;
    }

    static thread_local auto snapshot_storage = std::vector<char>{};
    const uint64_t           max_bytes        = (wptr_end - scan_pos) * state_ptr->pkt_size;
    if(snapshot_storage.size() < max_bytes) snapshot_storage.resize(max_bytes);
    char* const source_snapshot = snapshot_storage.data();

    queue_interposition_debug_log(fmt::format(
        "process_doorbell_impl snapshot-ready doorbell_id={} max_bytes={} snapshot_size={} "
        "source_snapshot={}",
        doorbell_debug_id,
        max_bytes,
        snapshot_storage.size(),
        fmt::ptr(source_snapshot)));

    uint64_t drained = 0;
    for(uint64_t pos = scan_pos; pos < wptr_end; ++pos)
    {
        const auto  ring_slot = pos & state_ptr->ring_mask;
        char* const slot_base =
            static_cast<char*>(state_ptr->ring_buf) + (ring_slot * state_ptr->pkt_size);
        auto* const hdr_ptr = reinterpret_cast<volatile uint16_t*>(slot_base);
        auto        header  = __atomic_load_n(hdr_ptr, __ATOMIC_ACQUIRE);

        if((header & 0xFFu) == static_cast<unsigned>(HSA_PACKET_TYPE_INVALID))
        {
            queue_interposition_debug_log(fmt::format(
                "process_doorbell_impl drain-stop-invalid doorbell_id={} pos={} ring_slot={} "
                "header=0x{:04x} drained={} wptr_end={}",
                doorbell_debug_id,
                pos,
                ring_slot,
                header,
                drained,
                wptr_end));
            break;
        }

        if(drained < 4 || pos + 1 == wptr_end || drained % 1024 == 0)
        {
            queue_interposition_debug_log(
                fmt::format("process_doorbell_impl drain-copy doorbell_id={} pos={} ring_slot={} "
                            "header=0x{:04x} "
                            "drained_before={} slot_base={}",
                            doorbell_debug_id,
                            pos,
                            ring_slot,
                            header,
                            drained,
                            fmt::ptr(static_cast<const void*>(slot_base))));
        }

        ::memcpy(source_snapshot + (drained * state_ptr->pkt_size), slot_base, state_ptr->pkt_size);
        __atomic_store_n(hdr_ptr, static_cast<uint16_t>(HSA_PACKET_TYPE_INVALID), __ATOMIC_RELEASE);
        ++drained;
    }

    if(drained == 0)
    {
        queue_interposition_debug_log(fmt::format(
            "process_doorbell_impl no-drained-packets doorbell_id={} ringing-original value={} {}",
            doorbell_debug_id,
            value,
            debug_state_summary(state_ptr)));
        ring_doorbell(state_ptr->doorbell_signal, value);
        queue_interposition_debug_log(fmt::format(
            "process_doorbell_impl no-drained-packets done doorbell_id={}", doorbell_debug_id));
        return;
    }

    const uint64_t pkt_count = drained;
    const uint64_t scan_end  = scan_pos + drained;

    ROCP_INFO << fmt::format("{} :: pkt_count={} (scan_pos={}, scan_end={})",
                             __FUNCTION__,
                             pkt_count,
                             scan_pos,
                             scan_end);
    queue_interposition_debug_log(fmt::format(
        "process_doorbell_impl drained doorbell_id={} pkt_count={} scan_pos={} scan_end={}",
        doorbell_debug_id,
        pkt_count,
        scan_pos,
        scan_end));

    auto& tls                     = get_doorbell_tls();
    tls.state                     = state_ptr;
    tls.submit_pos                = state_ptr->next_submit_pos;
    tls.pkt_size                  = state_ptr->pkt_size;
    tls.ring_doorbell             = &ring_doorbell;
    tls.last_published_submit_pos = state_ptr->next_submit_pos;
    uint64_t start_submit_pos     = tls.submit_pos;

    queue_interposition_debug_log(fmt::format(
        "process_doorbell_impl tls-ready doorbell_id={} start_submit_pos={} last_published={} "
        "pkt_size={} {}",
        doorbell_debug_id,
        start_submit_pos,
        tls.last_published_submit_pos,
        tls.pkt_size,
        debug_state_summary(state_ptr)));

    auto*        qc = get_queue_controller();
    const Queue* queue =
        (qc && state_ptr->hsa_queue) ? qc->get_queue(*state_ptr->hsa_queue) : nullptr;
    queue_interposition_debug_log(fmt::format(
        "process_doorbell_impl queue-controller doorbell_id={} controller={} hsa_queue={} queue={} "
        "queue_id={}",
        doorbell_debug_id,
        fmt::ptr(qc),
        fmt::ptr(static_cast<const void*>(state_ptr->hsa_queue)),
        fmt::ptr(queue),
        queue ? queue->get_id().handle : 0));

    auto write_start = std::chrono::steady_clock::now();
    if(queue)
    {
        // call local write_interceptor directly instead of heavyweight
        // Queue::invoke_write_interceptor
        queue_interposition_debug_log(fmt::format(
            "process_doorbell_impl write_interceptor begin doorbell_id={} pkt_count={} queue={}",
            doorbell_debug_id,
            pkt_count,
            fmt::ptr(queue)));
        write_interceptor(
            const_cast<Queue*>(queue), source_snapshot, pkt_count, ring_buffer_writer);
        queue_interposition_debug_log(fmt::format(
            "process_doorbell_impl write_interceptor end doorbell_id={} elapsed_us={} queue={}",
            doorbell_debug_id,
            debug_elapsed_us(write_start),
            fmt::ptr(queue)));
    }
    else
    {
        queue_interposition_debug_log(fmt::format(
            "process_doorbell_impl ring_buffer_writer-direct begin doorbell_id={} pkt_count={}",
            doorbell_debug_id,
            pkt_count));
        ring_buffer_writer(source_snapshot, pkt_count);
        queue_interposition_debug_log(fmt::format(
            "process_doorbell_impl ring_buffer_writer-direct end doorbell_id={} elapsed_us={}",
            doorbell_debug_id,
            debug_elapsed_us(write_start)));
    }

    uint64_t written = tls.submit_pos - start_submit_pos;
    queue_interposition_debug_log(fmt::format(
        "process_doorbell_impl write-count doorbell_id={} input_pkt_count={} written={} "
        "start_submit_pos={} submit_pos={}",
        doorbell_debug_id,
        pkt_count,
        written,
        start_submit_pos,
        tls.submit_pos));
    if(written != pkt_count)
    {
        ROCP_WARNING << "Write-interceptor changed packet count. "
                     << "queue=" << state_ptr->hsa_queue << ", input_pkt_count=" << pkt_count
                     << ", written_pkt_count=" << written;
    }

    state_ptr->next_scan_pos   = scan_end;
    state_ptr->next_submit_pos = tls.submit_pos;
    queue_interposition_debug_log(fmt::format("process_doorbell_impl state-advanced doorbell_id={} "
                                              "next_scan_pos={} next_submit_pos={} {}",
                                              doorbell_debug_id,
                                              state_ptr->next_scan_pos,
                                              state_ptr->next_submit_pos,
                                              debug_state_summary(state_ptr)));

    auto real_rdid = __atomic_load_n(state_ptr->real_rdid, __ATOMIC_ACQUIRE);
    auto ring_used = (state_ptr->next_submit_pos - real_rdid);
    if(ring_used > state_ptr->ring_size)
    {
        ROCP_WARNING << "Queue-intercept observed ring usage beyond ring size. queue="
                     << state_ptr->hsa_queue << ", ring_used=" << ring_used
                     << ", ring_size=" << state_ptr->ring_size << ", scan_pos=" << scan_pos
                     << ", scan_end=" << scan_end
                     << ", next_submit_pos=" << state_ptr->next_submit_pos;
    }

    queue_interposition_debug_log(fmt::format(
        "process_doorbell_impl publish-final begin doorbell_id={} ring_used={} real_rdid={} "
        "next_submit_pos={}",
        doorbell_debug_id,
        ring_used,
        real_rdid,
        state_ptr->next_submit_pos));
    publish_submitted_packets(state_ptr, state_ptr->next_submit_pos);
    queue_interposition_debug_log(
        fmt::format("process_doorbell_impl publish-final end doorbell_id={} {}",
                    doorbell_debug_id,
                    debug_state_summary(state_ptr)));

    tls.ring_doorbell             = nullptr;
    tls.last_published_submit_pos = 0;
    tls.state                     = nullptr;
    queue_interposition_debug_log(
        fmt::format("process_doorbell_impl tls-cleared doorbell_id={} total_elapsed_us={}",
                    doorbell_debug_id,
                    debug_elapsed_us(lock_start)));
}

std::shared_ptr<QueueState>
create_queue_state(const hsa_queue_t* queue, bool overwrite)
{
    queue_interposition_debug_log(fmt::format("create_queue_state begin queue={} overwrite={}",
                                              fmt::ptr(static_cast<const void*>(queue)),
                                              overwrite));

    if(!queue)
    {
        queue_interposition_debug_log("create_queue_state null queue");
        return nullptr;
    }

    // this is needed for OpenMP target offload which, unlike HIP, does not automatically enable
    // profiler for queues it creates.
    if(get_amd_ext_table() && get_amd_ext_table()->hsa_amd_profiling_set_profiler_enabled_fn)
    {
        ROCP_HSA_TABLE_CALL(WARNING,
                            get_amd_ext_table()->hsa_amd_profiling_set_profiler_enabled_fn(
                                const_cast<hsa_queue_t*>(queue), true))
            << fmt::format("Could not enable profiler for hsa_queue_t{{.id={}}}", queue->id);
    }

    if(!overwrite)
    {
        if(auto existing = lookup_queue_state(queue, false))
        {
            queue_interposition_debug_log(fmt::format("create_queue_state existing queue={} {}",
                                                      fmt::ptr(static_cast<const void*>(queue)),
                                                      debug_state_summary(existing.get())));
            return existing;
        }
    }

    auto*              amd_queue = reinterpret_cast<amd_queue_t*>(const_cast<hsa_queue_t*>(queue));
    auto               state     = std::make_shared<QueueState>();
    volatile uint64_t* wdid_addr = &amd_queue->write_dispatch_id;
    volatile uint64_t* rdid_addr = &amd_queue->read_dispatch_id;
    uint64_t           current_wdid = __atomic_load_n(wdid_addr, __ATOMIC_ACQUIRE);
    state->ring_buf                 = queue->base_address;
    state->ring_size                = queue->size;
    state->ring_mask                = queue->size - 1;
    state->real_wdid                = wdid_addr;
    state->real_rdid                = rdid_addr;
    state->hsa_queue                = queue;
    state->doorbell_signal          = queue->doorbell_signal;
    state->virtual_wptr.store(current_wdid, std::memory_order_relaxed);
    state->next_scan_pos   = current_wdid;
    state->next_submit_pos = current_wdid;

    queue_interposition_debug_log(fmt::format(
        "create_queue_state initialized queue={} amd_queue={} current_wdid={} current_rdid={} {}",
        fmt::ptr(static_cast<const void*>(queue)),
        fmt::ptr(amd_queue),
        current_wdid,
        __atomic_load_n(rdid_addr, __ATOMIC_ACQUIRE),
        debug_state_summary(state.get())));

    return get_queue_registry().wlock([&](auto& map) {
        map[queue] = state;
        queue_interposition_debug_log(
            fmt::format("create_queue_state registered queue={} registry_size={} {}",
                        fmt::ptr(static_cast<const void*>(queue)),
                        map.size(),
                        debug_state_summary(state.get())));
        return state;
    });
}

void
destroy_queue_state(const hsa_queue_t* queue)
{
    queue_interposition_debug_log(fmt::format("destroy_queue_state begin queue={}",
                                              fmt::ptr(static_cast<const void*>(queue))));
    get_queue_registry().wlock(
        [&](auto& map, const auto* _queue_v) {
            auto itr = map.find(_queue_v);
            if(itr != map.end())
            {
                queue_interposition_debug_log(
                    fmt::format("destroy_queue_state erased queue={} registry_size_before={} {}",
                                fmt::ptr(static_cast<const void*>(_queue_v)),
                                map.size(),
                                debug_state_summary(itr->second.get())));
                map.erase(itr);
            }
            else
            {
                queue_interposition_debug_log(
                    fmt::format("destroy_queue_state missing queue={} registry_size={}",
                                fmt::ptr(static_cast<const void*>(_queue_v)),
                                map.size()));
            }
        },
        queue);
}

namespace
{
namespace impl
{
// The 16 wrappers differ only by HSA suffix + memory order; generated via macros below.

// add_write_index: uint64_t(const hsa_queue_t*, uint64_t)
#define ROCP_QUEUE_ADD_WRITE_INDEX(SUFFIX, ORDER)                                                  \
    uint64_t queue_add_write_index_##SUFFIX(const hsa_queue_t* q, uint64_t v)                      \
    {                                                                                              \
        if(should_bypass_inline_intercept())                                                       \
            return get_next_table()->hsa_queue_add_write_index_##SUFFIX##_fn(q, v);                \
        if(auto s = lookup_queue_state(q, s_intercept_dynamic.load(std::memory_order_acquire)); s) \
            return add_write_index_impl(s.get(), v, ORDER);                                        \
        return get_next_table()->hsa_queue_add_write_index_##SUFFIX##_fn(q, v);                    \
    }

ROCP_QUEUE_ADD_WRITE_INDEX(relaxed, std::memory_order_relaxed)
ROCP_QUEUE_ADD_WRITE_INDEX(scacq_screl, std::memory_order_acq_rel)
ROCP_QUEUE_ADD_WRITE_INDEX(scacquire, std::memory_order_acquire)
ROCP_QUEUE_ADD_WRITE_INDEX(screlease, std::memory_order_release)

#undef ROCP_QUEUE_ADD_WRITE_INDEX

// store_write_index: void(const hsa_queue_t*, uint64_t)
#define ROCP_QUEUE_STORE_WRITE_INDEX(SUFFIX, ORDER)                                                \
    void queue_store_write_index_##SUFFIX(const hsa_queue_t* q, uint64_t v)                        \
    {                                                                                              \
        if(should_bypass_inline_intercept())                                                       \
        {                                                                                          \
            get_next_table()->hsa_queue_store_write_index_##SUFFIX##_fn(q, v);                     \
            return;                                                                                \
        }                                                                                          \
        if(auto s = lookup_queue_state(q, s_intercept_dynamic.load(std::memory_order_acquire)); s) \
        {                                                                                          \
            store_write_index_impl(s.get(), v, ORDER);                                             \
            return;                                                                                \
        }                                                                                          \
        get_next_table()->hsa_queue_store_write_index_##SUFFIX##_fn(q, v);                         \
    }

ROCP_QUEUE_STORE_WRITE_INDEX(relaxed, std::memory_order_relaxed)
ROCP_QUEUE_STORE_WRITE_INDEX(screlease, std::memory_order_release)

#undef ROCP_QUEUE_STORE_WRITE_INDEX

// cas_write_index: uint64_t(const hsa_queue_t*, uint64_t expected, uint64_t value)
#define ROCP_QUEUE_CAS_WRITE_INDEX(SUFFIX, ORDER)                                                  \
    uint64_t queue_cas_write_index_##SUFFIX(                                                       \
        const hsa_queue_t* q, uint64_t expected, uint64_t value)                                   \
    {                                                                                              \
        if(should_bypass_inline_intercept())                                                       \
            return get_next_table()->hsa_queue_cas_write_index_##SUFFIX##_fn(q, expected, value);  \
        if(auto s = lookup_queue_state(q, s_intercept_dynamic.load(std::memory_order_acquire)); s) \
            return cas_write_index_impl(s.get(), expected, value, ORDER);                          \
        return get_next_table()->hsa_queue_cas_write_index_##SUFFIX##_fn(q, expected, value);      \
    }

ROCP_QUEUE_CAS_WRITE_INDEX(relaxed, std::memory_order_relaxed)
ROCP_QUEUE_CAS_WRITE_INDEX(scacq_screl, std::memory_order_acq_rel)
ROCP_QUEUE_CAS_WRITE_INDEX(scacquire, std::memory_order_acquire)
ROCP_QUEUE_CAS_WRITE_INDEX(screlease, std::memory_order_release)

#undef ROCP_QUEUE_CAS_WRITE_INDEX

// load_write_index: uint64_t(const hsa_queue_t*)
#define ROCP_QUEUE_LOAD_WRITE_INDEX(SUFFIX, ORDER)                                                 \
    uint64_t queue_load_write_index_##SUFFIX(const hsa_queue_t* q)                                 \
    {                                                                                              \
        if(should_bypass_inline_intercept())                                                       \
            return get_next_table()->hsa_queue_load_write_index_##SUFFIX##_fn(q);                  \
        if(auto s = lookup_queue_state(q, s_intercept_dynamic.load(std::memory_order_acquire)); s) \
            return load_write_index_impl(s.get(), ORDER);                                          \
        return get_next_table()->hsa_queue_load_write_index_##SUFFIX##_fn(q);                      \
    }

ROCP_QUEUE_LOAD_WRITE_INDEX(relaxed, std::memory_order_relaxed)
ROCP_QUEUE_LOAD_WRITE_INDEX(scacquire, std::memory_order_acquire)

#undef ROCP_QUEUE_LOAD_WRITE_INDEX

// signal stores: void(hsa_signal_t, hsa_signal_value_t); NAME selects hsa_signal_<NAME>_fn.
#define ROCP_SIGNAL_STORE(NAME)                                                                    \
    void signal_##NAME(hsa_signal_t sig, hsa_signal_value_t val)                                   \
    {                                                                                              \
        queue_interposition_debug_log(                                                             \
            fmt::format("signal_" #NAME " begin signal={} value={}", sig.handle, val));            \
        if(should_bypass_inline_intercept())                                                       \
        {                                                                                          \
            queue_interposition_debug_log(                                                         \
                fmt::format("signal_" #NAME " bypass signal={} value={}", sig.handle, val));       \
            get_next_table()->hsa_signal_##NAME##_fn(sig, val);                                    \
            return;                                                                                \
        }                                                                                          \
        /* it is too late to create queue state at this point so do not create if missing. */      \
        constexpr auto create_if_missing = false;                                                  \
        if(auto s = lookup_queue_state_by_doorbell(sig, create_if_missing); s)                     \
        {                                                                                          \
            queue_interposition_debug_log(fmt::format("signal_" #NAME                              \
                                                      " process-doorbell signal={} value={} {}",   \
                                                      sig.handle,                                  \
                                                      val,                                         \
                                                      debug_state_summary(s.get())));              \
            process_doorbell_impl(s, val, [](hsa_signal_t db, hsa_signal_value_t v) {              \
                get_next_table()->hsa_signal_##NAME##_fn(db, v);                                   \
            });                                                                                    \
            queue_interposition_debug_log(                                                         \
                fmt::format("signal_" #NAME " processed signal={} value={}", sig.handle, val));    \
            return;                                                                                \
        }                                                                                          \
        queue_interposition_debug_log(                                                             \
            fmt::format("signal_" #NAME " passthrough signal={} value={}", sig.handle, val));      \
        get_next_table()->hsa_signal_##NAME##_fn(sig, val);                                        \
    }

ROCP_SIGNAL_STORE(store_relaxed)
ROCP_SIGNAL_STORE(store_screlease)
ROCP_SIGNAL_STORE(silent_store_relaxed)
ROCP_SIGNAL_STORE(silent_store_screlease)

#undef ROCP_SIGNAL_STORE
}  // namespace impl
}  // namespace

bool
supports_queue_interposition()
{
    return s_intercept_installed.load(std::memory_order_acquire);
}

void
interposition_sync()
{
    queue_interposition_debug_log("interposition_sync begin");
    if(async_signal_handler_exists())  // query without constructing
    {
        constexpr auto async_only = true;
        if(auto* tg = get_async_signal_handler(); tg)
        {
            queue_interposition_debug_log(
                fmt::format("interposition_sync join begin task_group={}", fmt::ptr(tg)));
            tg->join(async_only);
            queue_interposition_debug_log(
                fmt::format("interposition_sync join end task_group={}", fmt::ptr(tg)));
        }
    }
    queue_interposition_debug_log("interposition_sync end");
}

void
interposition_init(CoreApiTable* core_table, bool enabled)
{
    ROCP_INFO << "[queue-intercept] inline intercept path ENGAGED (tracing-only, no expansion)";
    queue_interposition_debug_log(fmt::format(
        "interposition_init begin core_table={} enabled={}", fmt::ptr(core_table), enabled));

    // save a pointer to the original
    get_original_table() = core_table;

    // Save current table entries as our next-in-chain (tracing functors when called
    // after update_table, or raw HSA functions otherwise)
    *get_next_table() = *core_table;

    // Dynamic queue discovery: when enabled, the write-index wrappers create QueueState on
    // first encounter for queues we did not observe at hsa_queue_create. Enabled only when
    // attachment is not supported; in attachment mode this has been observed to deadlock.
    // TODO(rocprofiler-sdk): root-cause the attachment-mode deadlock so it can be enabled there.
    s_intercept_dynamic.store(!registration::supports_attachment(), std::memory_order_release);

    // mark that intercept has been installed
    s_intercept_installed.store(true, std::memory_order_release);

    core_table->hsa_queue_add_write_index_relaxed_fn     = impl::queue_add_write_index_relaxed;
    core_table->hsa_queue_add_write_index_scacq_screl_fn = impl::queue_add_write_index_scacq_screl;
    core_table->hsa_queue_add_write_index_scacquire_fn   = impl::queue_add_write_index_scacquire;
    core_table->hsa_queue_add_write_index_screlease_fn   = impl::queue_add_write_index_screlease;

    core_table->hsa_queue_store_write_index_relaxed_fn   = impl::queue_store_write_index_relaxed;
    core_table->hsa_queue_store_write_index_screlease_fn = impl::queue_store_write_index_screlease;

    core_table->hsa_queue_cas_write_index_relaxed_fn     = impl::queue_cas_write_index_relaxed;
    core_table->hsa_queue_cas_write_index_scacq_screl_fn = impl::queue_cas_write_index_scacq_screl;
    core_table->hsa_queue_cas_write_index_scacquire_fn   = impl::queue_cas_write_index_scacquire;
    core_table->hsa_queue_cas_write_index_screlease_fn   = impl::queue_cas_write_index_screlease;

    core_table->hsa_queue_load_write_index_relaxed_fn   = impl::queue_load_write_index_relaxed;
    core_table->hsa_queue_load_write_index_scacquire_fn = impl::queue_load_write_index_scacquire;

    core_table->hsa_signal_store_relaxed_fn          = impl::signal_store_relaxed;
    core_table->hsa_signal_store_screlease_fn        = impl::signal_store_screlease;
    core_table->hsa_signal_silent_store_relaxed_fn   = impl::signal_silent_store_relaxed;
    core_table->hsa_signal_silent_store_screlease_fn = impl::signal_silent_store_screlease;

    // mark that intercept has been activated
    s_intercept_active.store(enabled, std::memory_order_release);
    queue_interposition_debug_log(fmt::format(
        "interposition_init end installed={} active={} dynamic={} original_table={} next_table={}",
        s_intercept_installed.load(std::memory_order_acquire),
        s_intercept_active.load(std::memory_order_acquire),
        s_intercept_dynamic.load(std::memory_order_acquire),
        fmt::ptr(get_original_table()),
        fmt::ptr(get_next_table())));
}

void
interposition_fini()
{
    // disable dynamic discovery of queues
    s_intercept_dynamic.store(false, std::memory_order_release);

    // disable active interception
    s_intercept_active.store(false, std::memory_order_release);

    // wait for any in-flight signal handlers to complete and clean up the signal pool
    interposition_sync();

    // clean up signal pool
    signal_pool_fini();

    get_queue_registry().wlock([](auto& map) { map.clear(); });
}
}  // namespace queue_interposition
}  // namespace hsa
}  // namespace rocprofiler
