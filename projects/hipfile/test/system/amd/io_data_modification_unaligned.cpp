/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

// Unaligned byte-granular sweep (Fallback only): the aligned combined guard from
// io_data_modification_bytes.cpp, swept across the file/buffer/size alignment
// cross-product, plus an extending-write sweep. See io_data_modification.cpp for the
// int32 variant.

#include "context.h"
#include "configuration.h"
#include "hipfile-warnings.h"
#include "hipfile.h"

#include "io-verify.hpp"
#include "test-common.h"
#include "test-options.h"

#include <array>
#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>
#include <string>
#include <tuple>
#include <unistd.h>
#include <vector>

extern SystemTestOptions test_env;

using namespace hipFile;
using namespace hipFileTest;

HIPFILE_WARN_NO_GLOBAL_CTOR_OFF

namespace {

// Aligned base offsets (borrowed from HipFileVerifyBytes.RoundTripGuardsAllRegions):
//  - file data sits one chunk + one file sentinel region block in, so it is PAST the
//    chunk boundary with room for a head file sentinel region bracket immediately
//    before it.
//  - device data sits after one head device sentinel region block.
// Unaligned variants add +1 byte to these bases.
constexpr hoff_t kFileOffBase = static_cast<hoff_t>(kChunkBytes + kFourKiB);
constexpr hoff_t kBufOffBase  = static_cast<hoff_t>(kFourKiB);

// One alignment axis: delta==0 is the aligned base, delta==1 is the base+1 byte.
struct OffParam {
    hoff_t      delta;
    std::string name;
};

HIPFILE_WARN_NO_EXIT_DTOR_OFF
const std::array<OffParam, 2> file_offsets{{
    {0, "file_aligned"},
    {1, "file_unaligned"},
}};
const std::array<OffParam, 2> buffer_offsets{{
    {0, "buffer_aligned"},
    {1, "buffer_unaligned"},
}};
// Four size categories: {small, large} x {aligned, +1 unaligned}. large crosses
// the 16 MiB fallback chunk boundary so the chunked loop iterates more than once.
const std::array<SizeParam, 4> sizes{{
    {kFourKiB, "small_aligned"},
    {kFourKiB + 1, "small_unaligned"},
    {kChunkBytes + kFourKiB, "large_aligned"},
    {kChunkBytes + kFourKiB + 1, "large_unaligned"},
}};
HIPFILE_WARN_NO_EXIT_DTOR_ON

enum class ExtKind { AppendFromEmpty, Append, Hole };

struct ExtRow {
    hoff_t      base_len;
    hoff_t      file_off;
    size_t      io_bytes;
    hoff_t      buf_off;
    ExtKind     kind;
    std::string name;
};

// Unaligned delta is +1 byte (maximally adversarial). Bases:
//   aligned nonzero base  = kChunkBytes
//   unaligned nonzero base= kChunkBytes + 1
//   aligned size          = kFourKiB ; unaligned size = kFourKiB + 1
//   large size            = kChunkBytes + kFourKiB (+1 for unaligned large)
//   hole file_off         = base_len + a gap; aligned gap = kFourKiB, unaligned gap = kFourKiB + 1
HIPFILE_WARN_NO_EXIT_DTOR_OFF
const std::array<ExtRow, 14> ext_rows{{
    {0, 0, kFourKiB, 0, ExtKind::AppendFromEmpty, "empty_contiguous_aligned_size"},
    {0, 0, kFourKiB + 1, 0, ExtKind::AppendFromEmpty, "empty_contiguous_unaligned_size"},
    {static_cast<hoff_t>(kChunkBytes + 1), static_cast<hoff_t>(kChunkBytes + 1), kFourKiB, 0, ExtKind::Append,
     "append_unaligned_base"},
    {static_cast<hoff_t>(kChunkBytes), static_cast<hoff_t>(kChunkBytes), kFourKiB + 1, 0, ExtKind::Append,
     "append_aligned_base_unaligned_size"},
    {0, static_cast<hoff_t>(kFourKiB), kFourKiB, 0, ExtKind::Hole, "hole_from_empty_aligned_off"},
    {0, static_cast<hoff_t>(kFourKiB + 1), kFourKiB, 0, ExtKind::Hole, "hole_from_empty_unaligned_off"},
    {static_cast<hoff_t>(kChunkBytes + 1), static_cast<hoff_t>(kChunkBytes + 1 + kFourKiB + 1), kFourKiB, 0,
     ExtKind::Hole, "hole_from_unaligned_base"},
    {static_cast<hoff_t>(kChunkBytes), static_cast<hoff_t>(kChunkBytes + kFourKiB + 1), kFourKiB + 1, 0,
     ExtKind::Hole, "hole_unaligned_off_unaligned_size"},
    {static_cast<hoff_t>(kChunkBytes), static_cast<hoff_t>(kChunkBytes), kFourKiB, 1, ExtKind::Append,
     "append_unaligned_buffer"},
    {0, static_cast<hoff_t>(kFourKiB), kFourKiB, 1, ExtKind::Hole, "hole_unaligned_buffer"},
    {static_cast<hoff_t>(kChunkBytes + 1), static_cast<hoff_t>(kChunkBytes + 1), kFourKiB + 1, 1,
     ExtKind::Append, "append_unaligned_base_unaligned_buffer"},
    {static_cast<hoff_t>(kChunkBytes + 1), static_cast<hoff_t>(kChunkBytes + 1 + kFourKiB + 1), kFourKiB + 1,
     1, ExtKind::Hole, "hole_all_unaligned"},
    {0, static_cast<hoff_t>(kFourKiB), kChunkBytes + kFourKiB, 0, ExtKind::Hole, "large_hole_cross_chunk"},
    {static_cast<hoff_t>(kChunkBytes + 1), static_cast<hoff_t>(kChunkBytes + 1), kChunkBytes + kFourKiB + 1,
     0, ExtKind::Append, "large_append_unaligned_base"},
}};
HIPFILE_WARN_NO_EXIT_DTOR_ON

} // namespace

// ---------------------------------------------------------------------------
// Unaligned all-regions round-trip (Fallback only). The same combined guard as
// HipFileVerifyBytes.RoundTripGuardsAllRegions -- device sentinel region on both
// sides, file sentinel region on both sides, a file_offset PAST the 16 MiB chunk
// boundary, every-other modify policy -- but swept across the cross-product of:
//   file offset   : aligned base           vs base + 1 byte
//   buffer offset : aligned base           vs base + 1 byte
//   size          : small/large x aligned/(+1 byte) unaligned
// 2 x 2 x 4 = 16 tests, all Fallback (Fastpath requires 4K alignment and cannot
// service the unaligned combinations; with Fastpath disabled, Fallback::score is
// the only backend that accepts the device buffer).
// ---------------------------------------------------------------------------
struct HipFileVerifyUnaligned : public testing::TestWithParam<std::tuple<OffParam, OffParam, SizeParam>> {

    Tmpfile         tmpfile;
    hipFileHandle_t tmpfile_handle{nullptr};
    void           *device_buffer{nullptr};
    size_t          io_bytes{0};     // bytes transferred using the hipFile API
    size_t          buffer_bytes{0}; // total device buffer size
    hoff_t          file_off{0};     // (possibly unaligned) file offset of the data
    hoff_t          buf_off{0};      // (possibly unaligned) device buffer offset of the data

    HipFileVerifyUnaligned() : tmpfile{test_env.ais_capable_dir}
    {
    }

    hoff_t fileDelta() const
    {
        return std::get<0>(GetParam()).delta;
    }
    hoff_t bufferDelta() const
    {
        return std::get<1>(GetParam()).delta;
    }

    void SetUp() override
    {
        // Fallback only: disable both, enable fallback. score() then has exactly one
        // candidate, so every combination -- aligned or not -- runs the fallback path.
        Context<Configuration>::get()->fastpath(false);
        Context<Configuration>::get()->fallback(true);

        io_bytes = std::get<2>(GetParam()).bytes;
        file_off = kFileOffBase + fileDelta();
        buf_off  = kBufOffBase + bufferDelta();

        // Over-allocate one device sentinel region block on each side of the data.
        // With a +1 buffer delta the data shifts one byte into the tail, leaving
        // 4 KiB - 1 bytes of tail device sentinel region -- still a non-empty guard on
        // both sides. Device layout (each sentinel region ~kFourKiB, data io_bytes; data
        // begins at buffer offset buf_off = kBufOffBase (+1 when buffer_unaligned)):
        // [head device sentinel region][data][tail device sentinel region]
        buffer_bytes = io_bytes + 2 * kFourKiB;

        // File layout (each sentinel region kFourKiB, data io_bytes; data begins at file
        // offset file_off past the chunk boundary). Size through the tail bracket:
        // [head file sentinel region][data][tail file sentinel region]
        const hoff_t tail_off = file_off + static_cast<hoff_t>(io_bytes);
        ASSERT_EQ(0, ftruncate(tmpfile.fd, static_cast<off_t>(tail_off + static_cast<hoff_t>(kFourKiB))));

        hipFileDescr_t descr{};
        descr.type      = hipFileHandleTypeOpaqueFD;
        descr.handle.fd = tmpfile.fd;
        ASSERT_EQ(HIPFILE_SUCCESS, hipFileHandleRegister(&tmpfile_handle, &descr));

        ASSERT_EQ(hipSuccess, hipMalloc(&device_buffer, buffer_bytes));
        ASSERT_EQ(hipSuccess, hipMemset(device_buffer, kByteDevSlack, buffer_bytes));
        // Sync so the async sentinel memset completes before the out-of-band
        // fastpath hipFileRead DMA, which is not ordered against prior null-stream
        // work (GDS/cuFile semantics: sync reads race pending stream ops).
        ASSERT_EQ(hipSuccess, hipDeviceSynchronize());
    }

    void TearDown() override
    {
        if (device_buffer) {
            ASSERT_EQ(hipSuccess, hipFree(device_buffer));
        }
        hipFileHandleDeregister(tmpfile_handle);
    }

    unsigned char *bufferStart() const
    {
        return static_cast<unsigned char *>(device_buffer);
    }
};

// Read the data from its (possibly unaligned) past-chunk file_offset into the
// (possibly unaligned) middle of the device allocation, modify every-other byte on
// the GPU, then write it back to the SAME file_offset. One round-trip proves the
// fallback path landed exactly io_bytes at exactly file_off, touched only the data
// region of the device allocation, and left both file sentinel region brackets and
// both device sentinel region blocks intact -- for every offset/size alignment
// combination.
TEST_P(HipFileVerifyUnaligned, RoundTripGuardsAllRegions)
{
    const size_t n        = io_bytes; // one byte per element
    const size_t slack_n  = kFourKiB; // file sentinel region bracket size in bytes
    const hoff_t head_off = file_off - static_cast<hoff_t>(kFourKiB); // head file sentinel region byte offset
    const hoff_t tail_off = file_off + static_cast<hoff_t>(io_bytes);
    constexpr size_t kStride = 2; // every-other byte

    // File layout (each sentinel region slack_n bytes, data n bytes; data begins at
    // file offset file_off past the chunk boundary):
    // [unwritten sparse hole = 0][head file sentinel region = 0x55][data = 0xFF][tail file sentinel region =
    // 0x55]. Everything below head_off (i.e. [0, file_off - kFourKiB)) is never seeded: ftruncate left it a
    // zero-filled sparse hole. It exists to push file_off past the chunk boundary; it is outside the transfer
    // window, so the round-trip must leave it untouched (checked below -- catches a far under-run past the
    // head bracket). Brackets are computed from the actual (possibly unaligned) file_off, so they stay
    // contiguous with the data and a one-byte over/under-run corrupts a bracket.
    seedFileBytesConstant(tmpfile.fd, head_off, slack_n, kByteFileSlack);
    seedFileBytesConstant(tmpfile.fd, file_off, n, kByteEntry);
    seedFileBytesConstant(tmpfile.fd, tail_off, slack_n, kByteFileSlack);

    // Read ONLY the data from its past-chunk file_offset into the buffer middle.
    ASSERT_EQ(static_cast<ssize_t>(io_bytes),
              hipFileRead(tmpfile_handle, device_buffer, io_bytes, file_off, buf_off));

    // Device layout (each sentinel region ~kFourKiB, data n bytes; data begins at buffer
    // offset buf_off, which may be +1 unaligned within the head block):
    // [head device sentinel region][data][tail device sentinel region]
    unsigned char *data = bufferStart() + static_cast<size_t>(buf_off);
    launchAndVerifyBytes(bufferStart(), buffer_bytes, data, n, defaultGrid(n), dim3(kDefaultBlockSize),
                         kStride);

    // Write the (every-other-modified) data back to the SAME past-chunk offset.
    ASSERT_EQ(static_cast<ssize_t>(io_bytes),
              hipFileWrite(tmpfile_handle, device_buffer, io_bytes, file_off, buf_off));

    // Sparse hole below the head bracket stayed zero (no far under-run past the bracket).
    assertHoleZero(tmpfile.fd, 0, head_off);
    // File sentinel region untouched on both sides; data matches the every-other policy.
    std::vector<unsigned char> head = readFileBytes(tmpfile.fd, head_off, slack_n);
    assertBytesConstant(head.data(), 0, slack_n, kByteFileSlack);
    std::vector<unsigned char> body = readFileBytes(tmpfile.fd, file_off, n);
    assertBytesModified(body.data(), n, kStride);
    std::vector<unsigned char> tail = readFileBytes(tmpfile.fd, tail_off, slack_n);
    assertBytesConstant(tail.data(), 0, slack_n, kByteFileSlack);
}

static std::string
unalignedName(const testing::TestParamInfo<HipFileVerifyUnaligned::ParamType> &info)
{
    return std::get<0>(info.param).name + "_" + std::get<1>(info.param).name + "_" +
           std::get<2>(info.param).name;
}

INSTANTIATE_TEST_SUITE_P(, HipFileVerifyUnaligned,
                         testing::Combine(testing::ValuesIn(file_offsets), testing::ValuesIn(buffer_offsets),
                                          testing::ValuesIn(sizes)),
                         unalignedName); // 2 file x 2 buffer x 4 size = 16

struct HipFileExtendUnaligned : public testing::TestWithParam<ExtRow> {

    Tmpfile         tmpfile;
    hipFileHandle_t tmpfile_handle{nullptr};
    void           *device_buffer{nullptr};
    size_t          buffer_bytes{0};

    HipFileExtendUnaligned() : tmpfile{test_env.ais_capable_dir}
    {
    }

    void SetUp() override
    {
        // Fallback only.
        Context<Configuration>::get()->fastpath(false);
        Context<Configuration>::get()->fallback(true);

        const ExtRow &r = GetParam();
        ASSERT_EQ(0, ftruncate(tmpfile.fd, static_cast<off_t>(r.base_len)));

        // Device layout (each sentinel region ~kFourKiB, data r.io_bytes; data begins at
        // buffer offset r.buf_off, which may be +1 unaligned within the head block):
        // [head device sentinel region][data][tail device sentinel region]
        buffer_bytes = r.io_bytes + 2 * kFourKiB;

        hipFileDescr_t descr{};
        descr.type      = hipFileHandleTypeOpaqueFD;
        descr.handle.fd = tmpfile.fd;
        ASSERT_EQ(HIPFILE_SUCCESS, hipFileHandleRegister(&tmpfile_handle, &descr));

        ASSERT_EQ(hipSuccess, hipMalloc(&device_buffer, buffer_bytes));
        ASSERT_EQ(hipSuccess, hipMemset(device_buffer, kByteDevSlack, buffer_bytes));
    }

    void TearDown() override
    {
        if (device_buffer) {
            ASSERT_EQ(hipSuccess, hipFree(device_buffer));
        }
        hipFileHandleDeregister(tmpfile_handle);
    }

    unsigned char *bufferStart() const
    {
        return static_cast<unsigned char *>(device_buffer);
    }
};

TEST_P(HipFileExtendUnaligned, Extends)
{
    const ExtRow    &r        = GetParam();
    const size_t     n        = r.io_bytes;
    constexpr size_t kStride  = 2;
    const hoff_t     head_off = r.file_off - 1; // one byte before data (append only)

    // Seed the data region of the device buffer to kByteEntry (memset already set
    // the whole allocation to kByteDevSlack; overwrite just the data window).
    unsigned char *data = bufferStart() + static_cast<size_t>(r.buf_off);
    ASSERT_EQ(hipSuccess, hipMemset(data, kByteEntry, n));

    launchAndVerifyBytes(bufferStart(), buffer_bytes, data, n, defaultGrid(n), dim3(kDefaultBlockSize),
                         kStride);

    const bool has_head_slack = (r.kind == ExtKind::Append);
    if (has_head_slack) {
        seedFileBytesConstant(tmpfile.fd, head_off, 1, kByteFileSlack);
    }

    ASSERT_EQ(static_cast<ssize_t>(r.io_bytes),
              hipFileWrite(tmpfile_handle, device_buffer, r.io_bytes, r.file_off, r.buf_off));

    // #2 data correct.
    std::vector<unsigned char> body = readFileBytes(tmpfile.fd, r.file_off, n);
    assertBytesModified(body.data(), n, kStride);

    // #3 Hole zero-fill.
    if (r.file_off > r.base_len) {
        assertHoleZero(tmpfile.fd, r.base_len, r.file_off);
    }

    // #4 Final size.
    ASSERT_EQ(r.file_off + static_cast<hoff_t>(r.io_bytes), fileSize(tmpfile.fd));

    // #6 Head file sentinel region intact (append only).
    if (has_head_slack) {
        std::vector<unsigned char> head = readFileBytes(tmpfile.fd, head_off, 1);
        assertBytesConstant(head.data(), 0, 1, kByteFileSlack);
    }
}

static std::string
extRowName(const testing::TestParamInfo<HipFileExtendUnaligned::ParamType> &info)
{
    return info.param.name;
}

INSTANTIATE_TEST_SUITE_P(, HipFileExtendUnaligned, testing::ValuesIn(ext_rows), extRowName); // 14

HIPFILE_WARN_NO_GLOBAL_CTOR_ON
