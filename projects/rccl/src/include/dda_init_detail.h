/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

#pragma once

#include "algorithms/CollCommon.h"
#include "fabric_gpu_barrier.h"
#include "ipc_gpu_barrier.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <new>

#define DDA_IPC_MAXBLOCKS 24
#define DDA_IPC_BUFFER_SIZE 268435456

#define DDA_FABRIC_MAXBLOCKS 256

namespace nccl_dda_detail {

constexpr int kDdaNranks = meta::comms::NRANKS;

// Per-comm IPC barrier state stored in ncclComm::ddaIpcBarrierState.
struct DdaIpcBarrierState {
  std::unique_ptr<meta::comms::IpcGpuBarrierResources> resources;
  meta::comms::IpcGpuBarrier barrierHost;
};

// Per-comm fabric barrier state stored in ncclComm::ddaFabricBarrierState.
struct DdaFabricBarrierState {
  std::unique_ptr<meta::comms::FabricGpuBarrierResources> resources;
  meta::comms::FabricGpuBarrier barrierHost;
};

inline int ddaMaxNBlocksForScratch() {
  unsigned maxBlocks = DDA_IPC_MAXBLOCKS;
  return static_cast<int>(maxBlocks);
}

inline int ddaFabricMaxNBlocksForScratch() {
  static int maxBlocks = -1;
  if (maxBlocks < 0) {
    int n = DDA_FABRIC_MAXBLOCKS;
    const char* s = getenv("RCCL_DDA_FABRIC_MAXBLOCKS");
    if (s != nullptr) {
      n = atoi(s);
    }
    if (n < 1) {
      n = 1;
    }
    if (n > 256) {
      n = 256;
    }
    maxBlocks = n;
  }
  return maxBlocks;
}

// Size (in bytes) of the per-comm DDA fabric VMM scratch buffer.
//
// The scratch is shared by every fabric collective and protocol tier. Its size
// only needs to cover the largest footprint the fabric path can actually issue
// for this comm; anything bigger is unusable because the collective is gated by
// the same thresholds (the kernels check `need > ddaScratchBytes` and fall back
// to the generic path when the scratch is too small, so under-sizing is a perf
// fallback, never a correctness bug).
//
// The dominant consumer is LL128 AllReduce, whose footprint is
//   2 banks * nRanks * ceil(ceil(msgBytes/8)/15) * 128B      (see LLLine128)
// for a message up to DDA_LL128_THRESHOLD (hard-capped at 1 GiB in the
// launcher). The Simple tier needs up to DDA_THRESHOLD bytes. We size to the
// worst of those for this comm's nRanks, plus a small margin for the fixed
// LL / AllGather / AllToAll / ReduceScatter slots.
//
// This replaces the former hard-coded 10 GiB (DDA_FABRIC_BUFFER_SIZE) which was
// ~80x larger than anything the path could use and was fully committed via
// cudaMemset at init.
//
// Env overrides:
//   RCCL_DDA_FABRIC_BUFFER_SIZE=<bytes>  force an exact size; 0 disables the
//                                        fabric DDA path (matches old bytes==0).
//   RCCL_DDA_THRESHOLD / RCCL_DDA_LL128_THRESHOLD are read so the scratch tracks
//   any user-tuned tier caps.
inline size_t ddaFabricScratchBytes(int nRanks) {
  // Explicit override wins (may be 0 to disable, preserving the old behaviour
  // where DDA_FABRIC_BUFFER_SIZE==0 short-circuited init).
  if (const char* ov = getenv("RCCL_DDA_FABRIC_BUFFER_SIZE")) {
    if (*ov) {
      long long v = atoll(ov);
      return v > 0 ? (size_t)v : 0;
    }
  }

  auto envBytes = [](const char* name, long long dflt) -> long long {
    const char* s = getenv(name);
    if (s && *s) {
      long long v = atoll(s);
      if (v > 0) return v;
    }
    return dflt;
  };

  if (nRanks < 1) {
    nRanks = 1;
  }

  const size_t simpleCap = (size_t)envBytes("RCCL_DDA_THRESHOLD", 134217728LL); // 128 MiB
  size_t ll128Cap = (size_t)envBytes("RCCL_DDA_LL128_THRESHOLD", 33554432LL);   //  32 MiB
  // LL128 AllReduce is hard-capped in the launcher regardless of the configured
  // threshold (kDdaLL128ArMaxBytes = 1 GiB).
  const size_t kLL128ArHardMax = 1073741824ULL; // 1 GiB
  if (ll128Cap > kLL128ArHardMax) {
    ll128Cap = kLL128ArHardMax;
  }

  // LL128 line geometry (see CollCommon_ll128.h): 128B lines, 15 payload words.
  const size_t words = (ll128Cap + 7) / 8;
  const size_t lines = (words + 14) / 15;
  const size_t ll128Ar = (size_t)2 * (size_t)nRanks * lines * 128;

  size_t bytes = simpleCap > ll128Ar ? simpleCap : ll128Ar;
  bytes += bytes / 8; // ~12% margin for the fixed LL/AG/A2A/RS slot arrays
  return bytes;
}

constexpr int kDdaLLAgMaxBlocksPerPeer = 8;

// The LL AllReduce tier is intentionally narrow (tiny messages, latency-bound),
// so it uses its own small epoch array instead of the shared 256-wide one. This
// keeps its per-launch epoch reset cheap (see ddaLLEpochEnd). Because it shares
// scratch bytes with the LL128 tier, its flags live in a disjoint high namespace
// (seeded below) so a leftover LL128 flag can never false-match an LL flag.
constexpr int      kDdaFabricLLArMaxBlocks = 24;
constexpr uint32_t kDdaLLArEpochSeed       = 0x40000000u; // first LL flag = seed+1

// Number of device epoch cells for the LL collectives. it is sized for the larger of the two
// max(AG total blocks, AR total blocks).
inline size_t ddaLLEpochCount(int nRanks, int arMaxBlocks) {
  const size_t ag = (size_t)nRanks * (size_t)kDdaLLAgMaxBlocksPerPeer;
  const size_t ar = (size_t)arMaxBlocks;
  return ag > ar ? ag : ar;
}

} // namespace nccl_dda_detail
