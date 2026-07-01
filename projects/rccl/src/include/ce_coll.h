/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_CE_COLL_H_
#define NCCL_CE_COLL_H_

#include "nccl.h"
#include "nccl_common.h"
#include "bitops.h"
#include <thread>
#include <mutex>
#include <condition_variable>
#include <cuda_runtime.h>   // cudaStream_t, cudaEvent_t 

// Memory operations per rank for different synchronization protocols
#define NCCL_CE_SYNC_OPS_PER_RANK_MC 2
#define NCCL_CE_SYNC_OPS_PER_RANK_UC 3
#define RCCL_CE_NUM_COPY_STREAMS 8

// Pipeline depth (number of in-flight scratch buffers) is runtime-tunable via
// RCCL_CE_PIPELINE_DEPTH. The fixed-size ring/event arrays are sized to the
// compile-time maximum; the resolved runtime depth is stored in
// ncclCePipeline::depth and must satisfy 1 <= depth <= RCCL_CE_PIPELINE_MAX_DEPTH.
#define RCCL_CE_PIPELINE_DEFAULT_DEPTH 2  // double buffer
#define RCCL_CE_PIPELINE_MAX_DEPTH     16

// One copy-back job handed producer (thread-1) -> copy worker (thread-2).
struct ncclCeCopyJob {
  uint8_t* userRecv;     // user recvbuff base
  uint8_t* scratchHalf;  // scratch double-buffer half base
  size_t   perRankSub;   // src stride in scratch (= subChunkBytes); 0 for scatter
  size_t   userStride;   // dst stride in userRecv (= chunkBytes);   0 for scatter
  size_t   userOff;      // byte offset within each user slot (= off)
  size_t   n;            // bytes this round
  int      nCopies;      // per-rank copies (nRanks; or 1 for scatter)
  int      buf;          // double-buffer index 0/1
  uint8_t* localSrc;     // local send source (= sendBuff+off); slot localCopyIdx is
                         // copied from here directly, bypassing the scratch bounce
  int      localCopyIdx; // rank slot sourced from localSrc instead of scratch (-1 = none)
  bool     stop;         // sentinel to terminate worker
};

struct ncclCePipeline {
  struct ncclComm* comm;
  std::thread worker;
  std::mutex  mtx;
  std::condition_variable cvFull;   // a job was posted
  std::condition_variable cvEmpty;  // a slot was freed
  int depth;                                            // runtime pipeline depth (1..MAX)
  ncclCeCopyJob ring[RCCL_CE_PIPELINE_MAX_DEPTH];
  int head, tail, count;
  cudaStream_t copyStream;                              // thread-2's stream
  cudaEvent_t  readyEvent[RCCL_CE_PIPELINE_MAX_DEPTH];  // producer -> consumer (scratch filled)
  cudaEvent_t  doneEvent[RCCL_CE_PIPELINE_MAX_DEPTH];   // consumer -> producer (scratch free)
  ncclResult_t asyncErr;                                // sticky error from worker
};

struct ncclCeColl {
  uint8_t* baseUCSymReadyPtr;
  uint8_t* baseUCSymComplPtr;
  size_t baseUCSymReadyOffset;
  size_t baseUCSymComplOffset;
  uint32_t ceSeqNum;
  bool useCompletePtr;
  uint32_t intraBatchSyncFreq;
  uint64_t intraBatchSyncMsgThreshold;
  struct ncclDevrWindow* ceSyncWin;
  int nCopyStreams;
  cudaStream_t copyStreams[RCCL_CE_NUM_COPY_STREAMS];
  cudaEvent_t copyEvents[RCCL_CE_NUM_COPY_STREAMS];
#ifdef ENABLE_FAULT_INJECTION
  uint32_t ceFaults;  // bitmask of CE_FAULT_* bits; see ce_fault_inject.h
#endif
  struct ncclCePipeline* pipeline;  // non-null when CE_PIPELINE enabled
};

struct ncclCeInitTask {
  struct ncclCeInitTask *next;
  struct ncclComm* comm;
};

struct alignas(16) ncclCeCollArgs {
  ncclFunc_t func;
  int rootRank;
  ncclDataType_t datatype;
  size_t nElts;
  size_t eltSize;
  uint8_t* sendBuff;
  uint8_t* recvBuff;
  struct ncclDevrWindow* sendWin;
  struct ncclDevrWindow* recvWin;
  void* collApiEventHandle;  // Parent API event handle for profiler hierarchy
  void* ceCollProfHandle;     // CE collective profiler event handle
  bool useDda;
  void** ddaPeerBases;      // host-side table of every rank's DDA scratch base pointer
  void*  ddaUserRecvBuff;   // user recvbuff (using DDA staging) or NULL otherwise (if recvbuffer is using symmetric windows)
  size_t ddaCopyBackBytes;  // bytes to copy scratch -> user recvbuff 
  bool   ceDdaPipeline;
  size_t ceDdaSubChunkBytes;
};

struct ncclCeBatchOpsParams {
  void** dsts;
  void** srcs;
  size_t* sizes;
  size_t numOps;
  bool intraBatchSync;
#ifdef CE_BATCH_ASYNC_SUPPORTED
  hipMemcpyAttributes* attrs;
  size_t* attrIdxs;
  size_t numAttrs;
#endif
};

bool ncclCeAvailable(struct ncclComm* comm, ncclFunc_t coll, int/*ncclDevRedOp_t*/ red, ncclDataType_t ty, ncclSymRegType_t winRegType);
bool ncclCeScartchAvailable(struct ncclComm* comm, ncclFunc_t coll, int/*ncclDevRedOp_t*/ red, ncclDataType_t ty, ncclSymRegType_t winRegType);
bool ncclCeImplemented(ncclFunc_t coll, int/*ncclDevRedOp_t*/ red, ncclDataType_t ty);

ncclResult_t ncclCeInit(struct ncclComm* comm);

ncclResult_t ncclCeFinalize(struct ncclComm* comm);

ncclResult_t ncclMemOpSync(struct ncclComm* comm, cudaStream_t stream, void* ceCollHandle);

ncclResult_t ncclLaunchCeColl(struct ncclComm* comm, struct ncclKernelPlan* plan);

ncclResult_t ncclCeAllGather(struct ncclComm* comm, struct ncclCeCollArgs* args, cudaStream_t stream);

ncclResult_t ncclCeScatter(struct ncclComm* comm, struct ncclCeCollArgs* args, cudaStream_t stream);

ncclResult_t ncclCeGather(struct ncclComm* comm, struct ncclCeCollArgs* args, cudaStream_t stream);

ncclResult_t ncclCeAlltoAll(struct ncclComm* comm, struct ncclCeCollArgs* args, cudaStream_t stream);
#endif /* NCCL_CE_COLL_H_ */
