# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT
#
# HRR sub-allocation map capture shim.
#
# Auto-imported by CPython at interpreter startup when this directory is on
# PYTHONPATH. It periodically snapshots PyTorch's HIP caching-allocator
# segment->block layout (torch.cuda.memory._snapshot) and pushes a compact
# binary blob into libamdhip64's HRR capture writer via the exported symbols
#   int  hipHrrCaptureActive(void);
#   void hipHrrCaptureSubAllocSnapshot(const void* blob, uint64_t len);
#   void hipHrrCaptureSubAllocTimeline(const void* blob, uint64_t len);
# The writer stores it content-addressed (crash-safe temp+rename) and records a
# small HRR_SUBALLOC_SNAPSHOT / HRR_SUBALLOC_TIMELINE event, so HRR's existing
# checkpoint/crash-finalize durability preserves the latest map even when the
# captured run dies on a GPU fault. Binary layout MUST match hrr_suballoc.h.
#
# Canonical copy lives in-tree at hrr/shim/sitecustomize.py; the hrr-testing
# harness copy must be kept in sync with this file.
#
# Disabled unless capture is active, so importing it outside a capture run is a
# no-op beyond starting one idle polling thread.

import os
import struct
import sys
import threading
import time

_MAGIC = 0x42415348   # "HSAB"
_TL_MAGIC = 0x4C545348  # "HSTL"
_VERSION = 1
_INTERVAL_S = float(os.environ.get("HRR_SUBALLOC_INTERVAL_S", "2.0"))
_VERBOSE = os.environ.get("HRR_SUBALLOC_VERBOSE", "") not in ("", "0")
# Full alloc/free timeline (precise per-kernel layout). On by default; the
# timeline is what lets replay reconstruct the exact live block set at each
# kernel. Set HRR_SUBALLOC_TIMELINE=0 to fall back to coarse snapshots only.
_TIMELINE = os.environ.get("HRR_SUBALLOC_TIMELINE", "1") not in ("", "0")
# PyTorch trace ring depth. Must comfortably exceed the number of alloc/free
# events between two polls so none are lost before we read them.
_TL_MAX_ENTRIES = int(os.environ.get("HRR_SUBALLOC_TL_MAX_ENTRIES", "1000000"))

# PyTorch device_trace action -> timeline record action code (hrr_suballoc.h).
_TL_ACTIONS = {
    "alloc": 0,           # HRR_TL_ALLOC
    "free_requested": 1,  # HRR_TL_FREE (block leaves the active set)
    "free_completed": 1,  # idempotent erase at playback
    "segment_alloc": 2,   # HRR_TL_SEGMENT_ALLOC
    "segment_free": 3,    # HRR_TL_SEGMENT_FREE
}


def _log(msg):
    if _VERBOSE:
        sys.stderr.write("[HRR suballoc shim] %s\n" % msg)
        sys.stderr.flush()


def _load_api():
    import ctypes
    for soname in ("libamdhip64.so.7", "libamdhip64.so", "libamdhip64.so.6"):
        try:
            lib = ctypes.CDLL(soname)
        except OSError:
            continue
        if not hasattr(lib, "hipHrrCaptureSubAllocSnapshot"):
            continue
        active = lib.hipHrrCaptureActive
        active.restype = ctypes.c_int
        active.argtypes = []
        push = lib.hipHrrCaptureSubAllocSnapshot
        push.restype = None
        push.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
        push_tl = None
        if hasattr(lib, "hipHrrCaptureSubAllocTimeline"):
            push_tl = lib.hipHrrCaptureSubAllocTimeline
            push_tl.restype = None
            push_tl.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
        return ctypes, active, push, push_tl
    return None, None, None, None


def _build_blob(segments):
    out = bytearray()
    out += struct.pack("<IIQ", _MAGIC, _VERSION, len(segments))
    for seg in segments:
        addr = int(seg.get("address", 0))
        total = int(seg.get("total_size", 0))
        blocks = seg.get("blocks", []) or []
        out += struct.pack("<QQII", addr, total, len(blocks), 0)
        off = 0
        for b in blocks:
            size = int(b.get("size", 0))
            active = 1 if b.get("state", "") == "active_allocated" else 0
            out += struct.pack("<QQB7x", off, size, active)
            off += size
    return bytes(out)


def _enable_history(torch):
    # Record the alloc/free trace WITHOUT C++ context; "python" stacks are the
    # lightest accepted value (frames are ignored by us). Returns True on success.
    try:
        torch.cuda.memory._record_memory_history(
            enabled="all", context=None, stacks="python",
            max_entries=_TL_MAX_ENTRIES)
        return True
    except Exception as e:
        _log("could not enable memory history: %r" % e)
        return False


def _build_baseline_blob(snap):
    # Blocks allocated *before* history was enabled (model weights, persistent
    # buffers, RNG state, ...) are not in device_traces, so the replayer would
    # see kernel pointers into them as "inside a segment but no live block" =
    # false OOB. Seed the timeline with the current live layout, stamped at
    # mono_ns=0 so it precedes every real event and covers even pre-enable
    # kernels. Subsequent frees/allocs in the trace correctly mutate it.
    segments = snap.get("segments", []) if isinstance(snap, dict) else []
    recs = []
    for seg in segments:
        base = int(seg.get("address", 0))
        total = int(seg.get("total_size", 0))
        recs.append((2, base, total))  # HRR_TL_SEGMENT_ALLOC
        off = 0
        for b in seg.get("blocks", []) or []:
            size = int(b.get("size", 0))
            if b.get("state", "") == "active_allocated":
                recs.append((0, base + off, size))  # HRR_TL_ALLOC
            off += size
    if not recs:
        return None
    out = bytearray()
    out += struct.pack("<IIQ", _TL_MAGIC, _VERSION, len(recs))
    for code, addr, size in recs:
        out += struct.pack("<B7xQQq", code, addr, size, 0)
    return bytes(out)


def _build_timeline_blob(snap, watermark):
    # Convert PyTorch trace entries newer than `watermark` (time_us) into a
    # timeline delta blob. time_us is CLOCK_REALTIME microseconds; convert to the
    # CLOCK_MONOTONIC ns used by HRR event headers via the current clock offset.
    traces = snap.get("device_traces", []) if isinstance(snap, dict) else []
    offset_ns = (time.clock_gettime_ns(time.CLOCK_MONOTONIC)
                 - time.clock_gettime_ns(time.CLOCK_REALTIME))
    recs = []
    new_watermark = watermark
    for dev in traces:
        for e in dev:
            tu = int(e.get("time_us", 0))
            if tu <= watermark:
                continue
            code = _TL_ACTIONS.get(e.get("action"))
            if code is None:
                continue
            recs.append((tu, code, int(e.get("addr", 0)), int(e.get("size", 0))))
            if tu > new_watermark:
                new_watermark = tu
    if not recs:
        return None, watermark
    recs.sort(key=lambda r: r[0])  # PyTorch trace order == time order
    out = bytearray()
    out += struct.pack("<IIQ", _TL_MAGIC, _VERSION, len(recs))
    for tu, code, addr, size in recs:
        out += struct.pack("<B7xQQq", code, addr, size, tu * 1000 + offset_ns)
    return bytes(out), new_watermark


def _worker():
    # libamdhip64 and its deps (librocprofiler-register, etc.) are not yet
    # resolvable at interpreter startup; they become loadable once torch
    # dlopens the HIP runtime. So wait for torch first, then resolve the
    # exported HRR symbols (retrying), then poll for capture to go active.
    torch = None
    for _ in range(600):  # up to ~10 min for a large model import
        torch = sys.modules.get("torch")
        if torch is not None and getattr(torch, "cuda", None) is not None:
            break
        time.sleep(1.0)
    if torch is None:
        _log("torch never imported; shim inert")
        return

    ctypes = active = push = push_tl = None
    for _ in range(120):
        ctypes, active, push, push_tl = _load_api()
        if push is not None:
            break
        time.sleep(1.0)
    if push is None:
        _log("libamdhip64 HRR sub-alloc symbols not found; shim inert")
        return

    # Wait for capture to be active (writer opens at HIP init).
    while True:
        try:
            if active() == 1:
                break
        except Exception:
            return
        time.sleep(1.0)

    use_tl = _TIMELINE and push_tl is not None and _enable_history(torch)
    if use_tl:
        # Seed the timeline with the live layout at enable time so pre-existing
        # blocks (weights/persistent buffers) are not seen as OOB.
        try:
            base_blob = _build_baseline_blob(torch.cuda.memory._snapshot())
            if base_blob is not None:
                buf = (ctypes.c_char * len(base_blob)).from_buffer_copy(base_blob)
                push_tl(buf, len(base_blob))
                _log("pushed baseline timeline: %d bytes (%d records)"
                     % (len(base_blob), (len(base_blob) - 16) // 32))
        except Exception as e:
            _log("baseline timeline error: %r" % e)
    _log("capture active; torch present — starting loop (interval=%ss, timeline=%s)"
         % (_INTERVAL_S, use_tl))
    last_blob = None
    n = 0
    n_tl = 0
    tl_recs = 0
    watermark = 0
    while True:
        try:
            if active() != 1:
                time.sleep(_INTERVAL_S)
                continue
            snap = torch.cuda.memory._snapshot()
            segments = snap.get("segments", []) if isinstance(snap, dict) else []
            if segments:
                blob = _build_blob(segments)
                # Skip pushing an unchanged layout back-to-back (compare full
                # content so a changed-but-same-length layout is never missed).
                if blob != last_blob:
                    buf = (ctypes.c_char * len(blob)).from_buffer_copy(blob)
                    push(buf, len(blob))
                    last_blob = blob
                    n += 1
                    if _VERBOSE:
                        nblk = sum(len(s.get("blocks", []) or []) for s in segments)
                        _log("pushed snapshot #%d: %d segments, %d blocks, %d bytes"
                             % (n, len(segments), nblk, len(blob)))
            if use_tl:
                tl_blob, watermark = _build_timeline_blob(snap, watermark)
                if tl_blob is not None:
                    buf = (ctypes.c_char * len(tl_blob)).from_buffer_copy(tl_blob)
                    push_tl(buf, len(tl_blob))
                    n_tl += 1
                    # 16-byte header, 32-byte records.
                    tl_recs += (len(tl_blob) - 16) // 32
                    if _VERBOSE:
                        _log("pushed timeline #%d: %d new events (%d total), %d bytes"
                             % (n_tl, (len(tl_blob) - 16) // 32, tl_recs, len(tl_blob)))
        except Exception as e:  # never let the shim kill the workload
            _log("snapshot error: %r" % e)
        time.sleep(_INTERVAL_S)


def _start():
    try:
        t = threading.Thread(target=_worker, name="hrr-suballoc", daemon=True)
        t.start()
    except Exception:
        pass


_start()
