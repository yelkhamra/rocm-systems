/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Shmem-free LL128 pack / poll / unpack helpers for the DDA fabric path.
 *
 * These are standalone (warp-cooperative) ports of the register mechanics in
 * device/prims_ll128.h (loadRegsBegin + loadRegsFinish, the recvReduceSendCopy
 * SEND store, its RECV poll loop, and storeRegs). The wire format is identical:
 * a 128-byte line = 16 x uint64_t = 15 data words + 1 flag word (last word),
 * so a receiver that observes the flag word sees all 120 payload bytes (this
 * relies on the gfx1250 128-byte non-tearing line + system-scope stores).
 *
 * Simplifications vs prims_ll128 (valid because the DDA fabric all-gather
 * guarantees 16-byte aligned buffers and perRankBytes % 16 == 0, so no line
 * ever straddles the payload end):
 *   - no shmem staging path (loads/stores go straight to registers),
 *   - out-of-range chunks of a partial slice are skipped rather than staged.
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

#pragma once

#include <cstdint>

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_HCC__)
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif

#include "nccl_device/rccl_ptr.h"

namespace meta::comms {
namespace ll128 {

// LL128 line geometry (gfx1250: 128-byte non-tearing line, wave32). Mirrors the
// NCCL_LL128_* constants in device.h but kept local so this header has no
// dependency on the device-internal build.
constexpr int kWarp = 32;                                    // WARP_SIZE
constexpr int kLineElems = 16;                               // 128B / 8B
constexpr int kLineSkip = 2 * kWarp / kLineElems;            // 4
constexpr int kWordsPerThread = 8;                           // NCCL_LL128_SHMEM_ELEMS_PER_THREAD
constexpr int kPairs = kWordsPerThread / 2;                  // 4 register pairs / thread
constexpr int kWireWordPerSlice = kWarp * kWordsPerThread;   // 256 u64 = 16 lines
constexpr int kDataWordPerSlice =
    kWireWordPerSlice - kWireWordPerSlice / kLineElems;      // 240 data u64
constexpr int kDataBytesPerSlice = kDataWordPerSlice * 8;    // 1920 payload bytes
constexpr int kWireBytesPerSlice = kWireWordPerSlice * 8;    // 2048 wire bytes

// Lanes 7,15,23,31 own the flag word of their 8-lane (128-byte) line.
__device__ __forceinline__ bool isFlagLane(int wid) {
  return (wid % (kLineElems / 2)) == (kLineElems / 2 - 1);
}

// Dense 16-byte-chunk index for register-pair g of lane wid (== the prims_ll128
// `ix` formula). Compensates for the flag holes so the packed payload in the
// user buffer stays gap-free and coalesced.
__device__ __forceinline__ int chunkIx(int g, int wid) {
  return g * kWarp - kLineSkip * (g / 2) + wid - (g % 2) * (wid / (kLineElems / 2));
}

// 16-byte (b128) system-scope, cache-bypassing store/load of a uint64_t pair,
// same builtin the DDA LL path already uses (see CollCommon.h ddaLLStoreLineB128).
__device__ __forceinline__ void store128(uint64_t* dst, uint64_t lo, uint64_t hi) {
#if RCCL_HAVE_GLOBAL_DWORDX4_BUILTINS
  union {
    v4u v;
    uint64_t w[2];
  } u;
  u.w[0] = lo;
  u.w[1] = hi;
  __builtin_amdgcn_global_store_b128((v4u_gptr)dst, u.v, RCCL_SYSTEM_SYNCSCOPE);
#else
  __builtin_nontemporal_store(lo, (u64_gptr)dst + 0);
  __builtin_nontemporal_store(hi, (u64_gptr)dst + 1);
#endif
  asm volatile("" ::: "memory");
}

__device__ __forceinline__ void load128(const uint64_t* src, uint64_t& lo, uint64_t& hi) {
  asm volatile("" ::: "memory");
#if RCCL_HAVE_GLOBAL_DWORDX4_BUILTINS
  union {
    v4u v;
    uint64_t w[2];
  } u;
  u.v = __builtin_amdgcn_global_load_b128((v4u_gptr)src, RCCL_SYSTEM_SYNCSCOPE);
  lo = u.w[0];
  hi = u.w[1];
#else
  lo = __builtin_nontemporal_load((u64_gptr)src + 0);
  hi = __builtin_nontemporal_load((u64_gptr)src + 1);
#endif
}

// (1) Dense load of a slice's payload from `src` into registers, then the
// flag-lane shuffle (== loadRegsBegin aligned-path + loadRegsFinish). `eltN` is
// the number of T elements in this slice (<= kDataBytesPerSlice/sizeof(T)).
template <typename T>
__device__ __forceinline__ void loadDense(
    uint64_t (&regs)[kWordsPerThread], const T* src, int eltN, int wid, bool flag) {
  constexpr int EltPer16B = 16 / sizeof(T);
#pragma unroll
  for (int g = 0; g < kPairs; g++) {
    if (!flag || g % 2 == 0) {
      int ix = chunkIx(g, wid);
      if (ix * EltPer16B < eltN)
        load128(reinterpret_cast<const uint64_t*>(src + ix * EltPer16B),
                regs[2 * g], regs[2 * g + 1]);
    }
  }
#pragma unroll
  for (int g = 1; g < kPairs; g += 2)  // move flag-lane data out of odd regs
    if (flag) regs[2 * g] = regs[2 * g - 1];
}

// (2) Store one slice to the wire with the flag word embedded on the flag lane
// (== recvReduceSendCopy SEND block). `wire` already includes the 2*wid lane
// offset; each pair lands one line's worth 32 u64 apart.
__device__ __forceinline__ void storeWire(
    uint64_t* wire, const uint64_t (&regs)[kWordsPerThread], uint64_t flag, bool flagLane) {
#pragma unroll
  for (int u = 0; u < kWordsPerThread; u += 2)
    store128(wire + u * kWarp, regs[u], flagLane ? flag : regs[u + 1]);
}

// (3) Poll the wire until every line's flag lands, then one clean reload
// (== recvReduceSendCopy RECV block). Returns raw wire words in `vr`.
__device__ __forceinline__ void pollWire(
    const uint64_t* wire, uint64_t (&vr)[kWordsPerThread], uint64_t flag, bool flagLane) {
  bool needReload;
  do {
    needReload = false;
#pragma unroll
    for (int u = 0; u < kWordsPerThread; u += 2) {
      load128(wire + u * kWarp, vr[u], vr[u + 1]);
      needReload |= flagLane && (vr[u + 1] != flag);
    }
  } while (__any(needReload));
#pragma unroll
  for (int u = 0; u < kWordsPerThread; u += 2)
    load128(wire + u * kWarp, vr[u], vr[u + 1]);
}

// (4) Flag-lane un-shuffle then dense store of registers into `dst`
// (== storeRegs, shmem-free: out-of-range chunks are simply skipped).
template <typename T>
__device__ __forceinline__ void storeDense(
    T* dst, uint64_t (&regs)[kWordsPerThread], int eltN, int wid, bool flag) {
  constexpr int EltPer16B = 16 / sizeof(T);
#pragma unroll
  for (int g = 1; g < kPairs; g += 2)  // reverse the load shuffle
    if (flag) regs[2 * g - 1] = regs[2 * g];
#pragma unroll
  for (int g = 0; g < kPairs; g++) {
    if (!flag || g % 2 == 0) {
      int ix = chunkIx(g, wid);
      if (ix * EltPer16B < eltN)
        store128(reinterpret_cast<uint64_t*>(dst + ix * EltPer16B),
                 regs[2 * g], regs[2 * g + 1]);
    }
  }
}

}  // namespace ll128
}  // namespace meta::comms
