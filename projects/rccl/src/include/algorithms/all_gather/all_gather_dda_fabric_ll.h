/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * LL-protocol all-gather device kernel for the DDA path. Each 16-byte line holds
 * 8 bytes of payload and two 4-byte flags; the flags carry cross-rank sync, so
 * no GPU barrier is needed. Staging uses the DDA scratch (comm->ddaScratch,
 * reachable via comm->ddaPeerPtrsDev).
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

#pragma once

#include <cstddef>
#include <cstdint>

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_HCC__)
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif

#include "algorithms/CollCommon.h"
#include "algorithms/ll128_pack.h"

namespace meta::comms {

// Per-rank scratch slot capacity and hard per-rank cap (enforced in the
// eligibility check); fixes the slot stride at compile time so the double-buffered
// layout is identical on every rank and call. The effective size gate is the total
// gathered size (DDA_ALLGATHER_LL_THRESHOLD, see collectives.cc), so the actual
// per-rank payload is <= total / nRanks and usually well under this cap.
// Footprint = 2 banks * nRanks * (kDdaLLAgMaxPerRankBytes * 2) for the 8B->16B
// expansion; 36 MiB at 128 KiB / 72 ranks, within the 64 MiB DDA scratch.
constexpr size_t kDdaLLAgMaxPerRankBytes = 131072;                     // 128 KiB
constexpr size_t kDdaLLAgSlotStridePkts  = kDdaLLAgMaxPerRankBytes / 8; // 16384

// LL all-gather kernel. 2D grid: grid.x == nRanks selects the peer (column b
// owns peer b); grid.y == blocksPerPeer splits that peer's packets into gridDim.y
// contiguous chunks (chunk == blockIdx.y). grid.y == 1 is one block per peer.
//
// The self column copies sendbuff -> recvbuff[self] locally; other columns
// scatter this rank's chunk into peer b's slot, then poll their own slot b for
// peer b's chunk. Threads split the chunk's packet range.
template <typename T, int NRANKS_CT>
#if defined(USE_ROCM)
__launch_bounds__(512)
#endif
__global__ void ddaAllGatherFabricLL(
    T* const* __restrict__ peerScratch,    // ddaPeerPtrsDev: nRanks scratch bases
    T* __restrict__ recvbuff,              // local user output
    const T* __restrict__ sendbuff,        // local user input
    size_t perRankBytes,                   // per-rank payload; multiple of 16
    int selfRank,
    int nRanksRt,
    uint32_t* __restrict__ epochDev,       // per-block LL epoch cells
    int epochLen) {                        // number of cells in epochDev

  const int nRanks = NRANKS_CT ? NRANKS_CT : nRanksRt;
  const int peer = blockIdx.x;             // grid.x == nRanks: one column/peer
  if (peer >= nRanks) return;              // safety if grid.x > nRanks
  const int chunk = blockIdx.y;            // grid.y == blocksPerPeer
  const int nChunks = gridDim.y;           // >= 1
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;
  const size_t nPk = perRankBytes >> 3;    // 8 payload bytes per packet
  const size_t slot = kDdaLLAgSlotStridePkts;

  // Flat block id + total launched blocks. tid 0 reads our own epoch cell (all
  // cells hold the same value) and derives this launch's flag on the device, so
  // nothing is baked into a HIP graph capture. bank = flag & 1.
  const int flatBlockId = blockIdx.x * gridDim.y + blockIdx.y;
  const int total = gridDim.x * gridDim.y;
  __shared__ uint32_t s_flag;
  if (tid == 0) {
    uint32_t f = epochDev[flatBlockId] + 1u;
    if (f == 0u) f = 2u;                   // skip 0 sentinel; keep bank parity
    s_flag = f;
  }
  __syncthreads();
  const uint32_t flag = s_flag;
  const size_t bankOffsetPkts = (size_t)(flag & 1u) * (size_t)nRanks * slot;

  // This block's packet range [pkBegin, pkEnd); [0, nPk) when nChunks == 1.
  const size_t pkPerChunk = (nPk + (size_t)nChunks - 1) / (size_t)nChunks;
  const size_t pkBegin = (size_t)chunk * pkPerChunk;
  size_t pkEnd = pkBegin + pkPerChunk;
  if (pkEnd > nPk) pkEnd = nPk;

  if (peer == selfRank) {
    // self column: local copy sendbuff -> recvbuff[self].
    const uint4* s4 = reinterpret_cast<const uint4*>(sendbuff);
    uint4* d4 = reinterpret_cast<uint4*>(
        reinterpret_cast<char*>(recvbuff) + (size_t)selfRank * perRankBytes);
    const size_t nVec = perRankBytes >> 4; // number of 16B chunks
    // split the copy across this peer's blocks too.
    const size_t vecPerChunk = (nVec + (size_t)nChunks - 1) / (size_t)nChunks;
    const size_t vBegin = (size_t)chunk * vecPerChunk;
    size_t vEnd = vBegin + vecPerChunk;
    if (vEnd > nVec) vEnd = nVec;
    for (size_t i = vBegin + tid; i < vEnd; i += nthreads) {
      const uint4* p = &s4[i];
      uint4 v;
      v.x = __builtin_nontemporal_load(&p->x);
      v.y = __builtin_nontemporal_load(&p->y);
      v.z = __builtin_nontemporal_load(&p->z);
      v.w = __builtin_nontemporal_load(&p->w);
      uint4* q = &d4[i];
      __builtin_nontemporal_store(v.x, &q->x);
      __builtin_nontemporal_store(v.y, &q->y);
      __builtin_nontemporal_store(v.z, &q->z);
      __builtin_nontemporal_store(v.w, &q->w);
    }
  } else {
    // scatter: write my payload into peer's slot (== selfRank).
    const uint32_t* sw = reinterpret_cast<const uint32_t*>(sendbuff);
    LLPacket16* dst = reinterpret_cast<LLPacket16*>(peerScratch[peer]) +
        (size_t)selfRank * slot + bankOffsetPkts;
    for (size_t pk = pkBegin + tid; pk < pkEnd; pk += nthreads) {
      ddaLLStoreLineB128(
          reinterpret_cast<uint32_t*>(&dst[pk]),
          sw[2 * pk], flag, sw[2 * pk + 1], flag);
    }

    // gather: poll my slot for peer, unpack into recvbuff[peer].
    volatile LLPacket16* src =
        reinterpret_cast<LLPacket16*>(peerScratch[selfRank]) + bankOffsetPkts +
        (size_t)peer * slot;
    uint32_t* out = reinterpret_cast<uint32_t*>(
        reinterpret_cast<char*>(recvbuff) + (size_t)peer * perRankBytes);
    for (size_t pk = pkBegin + tid; pk < pkEnd; pk += nthreads) {
      uint32_t d0, f0, d1, f1;
      do {
        ddaLLLoadLineB128(
            reinterpret_cast<const uint32_t*>(const_cast<LLPacket16*>(&src[pk])),
            d0, f0, d1, f1);
      } while (f0 != flag || f1 != flag);
      out[2 * pk] = d0;
      out[2 * pk + 1] = d1;
    }
  }

  if (tid == 0) {
    for (int e = flatBlockId; e < epochLen; e += total) {
      epochDev[e] = flag;
    }
  }
}

// LL128 all-gather kernel. Same 2D grid and epoch/bank scheme as the LL kernel
// above, but the wire uses the 128-byte LL128 line (15 payload words + 1 flag
// word) instead of the 16-byte LL line, so the flag overhead drops from 2x to
// 16/15. Each 128-byte line is written cooperatively by 8 lanes of a warp and
// relies on the gfx1250 128-byte non-tearing line for atomic visibility.
//
// The self column copies sendbuff -> recvbuff[self] locally; other columns pack
// this rank's slice range into peer b's slot (LL128 scatter == prims_ll128
// send), then poll their own slot b and unpack peer b's payload (LL128 gather
// == prims_ll128 recv). Work is warp-cooperative: warps stripe the block's
// 1920-byte slices. Instantiated as int8_t (raw-byte copy), so EltPer16B == 16
// and every slice's eltN stays a multiple of 16.
template <typename T, int NRANKS_CT>
#if defined(USE_ROCM)
__launch_bounds__(256)
#endif
__global__ void ddaAllGatherFabricLL128(
    T* const* __restrict__ peerScratch,    // ddaPeerPtrsDev: nRanks scratch bases
    T* __restrict__ recvbuff,              // local user output
    const T* __restrict__ sendbuff,        // local user input
    size_t perRankBytes,                   // per-rank payload; multiple of 16
    int selfRank,
    int nRanksRt,
    uint32_t* __restrict__ epochDev,       // per-block LL epoch cells
    int epochLen) {                        // number of cells in epochDev

  const int nRanks = NRANKS_CT ? NRANKS_CT : nRanksRt;
  const int peer = blockIdx.x;             // grid.x == nRanks: one column/peer
  if (peer >= nRanks) return;              // safety if grid.x > nRanks
  const int chunk = blockIdx.y;            // grid.y == blocksPerPeer
  const int nChunks = gridDim.y;           // >= 1
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;
  const int lane = tid % ll128::kWarp;
  const int warp = tid / ll128::kWarp;
  const int nwarps = nthreads / ll128::kWarp;
  const bool flagLane = ll128::isFlagLane(lane);

  // tid 0 derives this launch's flag from our epoch cell (device-derived so
  // nothing is baked into a graph capture). bank = flag & 1. The 64-bit wire
  // flag word carries the 32-bit epoch in both halves.
  const int flatBlockId = blockIdx.x * gridDim.y + blockIdx.y;
  const int total = gridDim.x * gridDim.y;
  __shared__ uint32_t s_flag;
  if (tid == 0) {
    uint32_t f = epochDev[flatBlockId] + 1u;
    if (f == 0u) f = 2u;                   // skip 0 sentinel; keep bank parity
    s_flag = f;
  }
  __syncthreads();
  const uint32_t flag32 = s_flag;
  const uint64_t flag = ((uint64_t)flag32 << 32) | (uint64_t)flag32;
  const uint32_t bank = flag32 & 1u;

  // Per-rank slot stride (u64 words), derived from perRankBytes so host and
  // device agree; must match ddaLL128AgSlotWords() on the host.
  const size_t slicesTotal =
      (perRankBytes + ll128::kDataBytesPerSlice - 1) / ll128::kDataBytesPerSlice;
  const uint64_t slotWords = slicesTotal * (uint64_t)ll128::kWireWordPerSlice;
  const uint64_t bankWords = (uint64_t)bank * (uint64_t)nRanks * slotWords;

  // This block's slice range [sliceBeg, sliceEnd); [0, slicesTotal) when nChunks == 1.
  const size_t slicesPerChunk = (slicesTotal + (size_t)nChunks - 1) / (size_t)nChunks;
  const size_t sliceBeg = (size_t)chunk * slicesPerChunk;
  size_t sliceEnd = sliceBeg + slicesPerChunk;
  if (sliceEnd > slicesTotal) sliceEnd = slicesTotal;

  if (peer == selfRank) {
    // self column: local copy sendbuff -> recvbuff[self] over this block's byte
    // range, 16B-vectorized and split across all threads.
    size_t byteBeg = sliceBeg * (size_t)ll128::kDataBytesPerSlice;
    size_t byteEnd = sliceEnd * (size_t)ll128::kDataBytesPerSlice;
    if (byteEnd > perRankBytes) byteEnd = perRankBytes;
    if (byteBeg > perRankBytes) byteBeg = perRankBytes;
    const uint4* s4 = reinterpret_cast<const uint4*>(
        reinterpret_cast<const char*>(sendbuff) + byteBeg);
    uint4* d4 = reinterpret_cast<uint4*>(
        reinterpret_cast<char*>(recvbuff) + (size_t)selfRank * perRankBytes + byteBeg);
    const size_t nVec = (byteEnd - byteBeg) >> 4;  // 16B chunks
    for (size_t i = tid; i < nVec; i += nthreads) {
      const uint4* p = &s4[i];
      uint4 v;
      v.x = __builtin_nontemporal_load(&p->x);
      v.y = __builtin_nontemporal_load(&p->y);
      v.z = __builtin_nontemporal_load(&p->z);
      v.w = __builtin_nontemporal_load(&p->w);
      uint4* q = &d4[i];
      __builtin_nontemporal_store(v.x, &q->x);
      __builtin_nontemporal_store(v.y, &q->y);
      __builtin_nontemporal_store(v.z, &q->z);
      __builtin_nontemporal_store(v.w, &q->w);
    }
  } else {
    // scatter base: my payload -> peer's slot (== my selfRank slot in peer's scratch).
    uint64_t* scatterSlot = reinterpret_cast<uint64_t*>(peerScratch[peer]) +
        bankWords + (uint64_t)selfRank * slotWords;
    // gather base: poll my slot for peer (== peer's slot in my scratch).
    const uint64_t* gatherSlot = reinterpret_cast<const uint64_t*>(peerScratch[selfRank]) +
        bankWords + (uint64_t)peer * slotWords;
    const int8_t* srcBytes = reinterpret_cast<const int8_t*>(sendbuff);
    int8_t* dstBytes = reinterpret_cast<int8_t*>(recvbuff) + (size_t)peer * perRankBytes;

    // scatter: pack + store each slice in this warp's stride.
    for (size_t s = sliceBeg + warp; s < sliceEnd; s += nwarps) {
      const size_t dataByte = s * (size_t)ll128::kDataBytesPerSlice;
      const size_t rem = perRankBytes - dataByte;
      const int eltInSlice =
          rem < (size_t)ll128::kDataBytesPerSlice ? (int)rem : ll128::kDataBytesPerSlice;
      uint64_t regs[ll128::kWordsPerThread];
      ll128::loadDense<int8_t>(regs, srcBytes + dataByte, eltInSlice, lane, flagLane);
      ll128::storeWire(scatterSlot + s * ll128::kWireWordPerSlice + 2 * lane,
                       regs, flag, flagLane);
    }

    // gather: poll + unpack each slice in this warp's stride.
    for (size_t s = sliceBeg + warp; s < sliceEnd; s += nwarps) {
      const size_t dataByte = s * (size_t)ll128::kDataBytesPerSlice;
      const size_t rem = perRankBytes - dataByte;
      const int eltInSlice =
          rem < (size_t)ll128::kDataBytesPerSlice ? (int)rem : ll128::kDataBytesPerSlice;
      uint64_t vr[ll128::kWordsPerThread];
      ll128::pollWire(gatherSlot + s * ll128::kWireWordPerSlice + 2 * lane,
                      vr, flag, flagLane);
      ll128::storeDense<int8_t>(dstBytes + dataByte, vr, eltInSlice, lane, flagLane);
    }
  }

  if (tid == 0) {
    for (int e = flatBlockId; e < epochLen; e += total) {
      epochDev[e] = flag32;
    }
  }
}

} // namespace meta::comms
