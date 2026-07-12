/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

// int32 (pattern) data-modification round-trips: isolated read/write, offset guards,
// a combined all-regions guard, and an extending-write sweep. See
// io_data_modification_bytes.cpp / io_data_modification_unaligned.cpp for the byte
// variants.

#include "context.h"
#include "configuration.h"
#include "hipfile-warnings.h"
#include "hipfile.h"

#include "io.hpp"
#include "io-verify.hpp"
#include "test-common.h"
#include "test-options.h"

#include <array>
#include <cstdint>
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

HIPFILE_WARN_NO_EXIT_DTOR_OFF
const std::array<SizeParam, 5> verify_sizes{{
    {kFourKiB, "sub_chunk"},
    {kChunkBytes - kFourKiB, "near_chunk"},
    {kChunkBytes, "exact_chunk"},
    {kChunkBytes + kFourKiB, "cross_chunk"},
    {2 * kChunkBytes, "multi_chunk"},
}};
HIPFILE_WARN_NO_EXIT_DTOR_ON

enum class GridMode { Default, OneBlock, ManyBlocks };
struct BlockParam {
    GridMode    mode;
    std::string name;
};
// Use lots of blocks to exercise many CUs/XCDs.
constexpr unsigned kManyBlocks = 300;

dim3
gridFor(GridMode mode, size_t n)
{
    switch (mode) {
        case GridMode::OneBlock:
            return dim3(1);
        case GridMode::ManyBlocks:
            return dim3(kManyBlocks);
        case GridMode::Default:
            return defaultGrid(n);
        default:
            return defaultGrid(n);
    }
}

// ---------------------------------------------------------------------------
// Helpers for extending the file length.
// ---------------------------------------------------------------------------
enum class ExtendKind { AppendFromEmpty, Append, Hole, OverwriteAppend };

struct ScenarioParam {
    hoff_t      base_len; // file length in SetUp BEFORE the extending write
    hoff_t      file_off; // where the data write starts
    ExtendKind  kind;
    std::string name;
};

HIPFILE_WARN_NO_EXIT_DTOR_OFF
const std::array<ScenarioParam, 5> extend_scenarios{{
    {0, 0, ExtendKind::AppendFromEmpty, "extend_empty_contiguous"},
    {static_cast<hoff_t>(kChunkBytes), static_cast<hoff_t>(kChunkBytes), ExtendKind::Append,
     "append_aligned"},
    {0, static_cast<hoff_t>(kChunkBytes), ExtendKind::Hole, "hole_from_empty"},
    {static_cast<hoff_t>(kChunkBytes), static_cast<hoff_t>(2 * kChunkBytes), ExtendKind::Hole,
     "hole_from_aligned"},
    // Modify the data before the EOF and extend the length of the file.
    {static_cast<hoff_t>(kChunkBytes + kFourKiB / 2), static_cast<hoff_t>(kChunkBytes),
     ExtendKind::OverwriteAppend, "overwrite_and_append"},
}};
const std::array<SizeParam, 2>     extend_sizes{{
    {kFourKiB, "small"},
    {kChunkBytes + kFourKiB, "large"},
}};
HIPFILE_WARN_NO_EXIT_DTOR_ON

} // namespace

struct HipFileVerify : public testing::TestWithParam<std::tuple<IoTestParam, SizeParam>> {

    Tmpfile         tmpfile;
    hipFileHandle_t tmpfile_handle{nullptr};
    void           *device_buffer{nullptr};
    size_t          io_bytes{0};     // bytes transferred using the hipFile API
    size_t          buffer_bytes{0}; // total device buffer size

    HipFileVerify() : tmpfile{test_env.ais_capable_dir}
    {
    }

    IoTestBackend backend() const
    {
        return std::get<0>(GetParam()).backend;
    }

    void SetUp() override
    {
        Context<Configuration>::get()->fastpath(false);
        Context<Configuration>::get()->fallback(false);

        switch (backend()) {
            case IoTestBackend::Fastpath:
                Context<Configuration>::get()->fastpath(true);
                break;
            case IoTestBackend::Fallback:
                Context<Configuration>::get()->fallback(true);
                break;
            default:
                FAIL() << "Unsupported IoTestBackend";
        }

        io_bytes = std::get<1>(GetParam()).bytes;
        // The doubling kernel will also verify that the sentinel regions of the device buffer are unmodified.
        // Device layout (each sentinel region kFourKiB, data io_bytes):
        // [head device sentinel region][data][tail device sentinel region]
        buffer_bytes = io_bytes + 2 * kFourKiB;

        // Size the file to hold io_bytes of data plus a leading empty chunk.
        ASSERT_EQ(0, ftruncate(tmpfile.fd, static_cast<off_t>(io_bytes + kChunkBytes)));

        hipFileDescr_t descr{};
        descr.type      = hipFileHandleTypeOpaqueFD;
        descr.handle.fd = tmpfile.fd;
        ASSERT_EQ(HIPFILE_SUCCESS, hipFileHandleRegister(&tmpfile_handle, &descr));

        ASSERT_EQ(hipSuccess, hipMalloc(&device_buffer, buffer_bytes));
        ASSERT_EQ(hipSuccess, hipMemset(device_buffer, kSentinelByte, buffer_bytes));
        // hipMemset is not synchronous w.r.t. the host, and hipFileRead is not ordered w.r.t. the stream.
        ASSERT_EQ(hipSuccess, hipDeviceSynchronize());
    }

    void TearDown() override
    {
        if (device_buffer) {
            ASSERT_EQ(hipSuccess, hipFree(device_buffer));
        }
        hipFileHandleDeregister(tmpfile_handle);
    }

    size_t elems() const
    {
        return io_bytes / sizeof(int32_t);
    }

    size_t bufferElems() const
    {
        return buffer_bytes / sizeof(int32_t);
    }

    int32_t *bufferStart() const
    {
        return static_cast<int32_t *>(device_buffer);
    }
};

// Isolated hipFileWrite test.
TEST_P(HipFileVerify, WritePersistsDoubledData)
{
    const size_t n = elems();
    seedDevicePattern(device_buffer, 0, n);

    launchAndVerify(bufferStart(), bufferElems(), bufferStart(), n, defaultGrid(n), dim3(kDefaultBlockSize));

    ASSERT_EQ(static_cast<ssize_t>(io_bytes), hipFileWrite(tmpfile_handle, device_buffer, io_bytes, 0, 0));

    std::vector<int32_t> file = readFileInts(tmpfile.fd, 0, n);
    assertDoubledPattern(file.data(), n);
}

// Isolated hipFileRead test.
TEST_P(HipFileVerify, ReadDeliversDoubledData)
{
    const size_t n = elems();
    seedFilePattern(tmpfile.fd, 0, n);

    ASSERT_EQ(static_cast<ssize_t>(io_bytes), hipFileRead(tmpfile_handle, device_buffer, io_bytes, 0, 0));

    launchAndVerify(bufferStart(), bufferElems(), bufferStart(), n, defaultGrid(n), dim3(kDefaultBlockSize));

    std::vector<int32_t> buf = readbackInts(device_buffer, 0, n);
    assertDoubledPattern(buf.data(), n);
}

// hipFileRead + hipFileWrite at offset into device buffer (places sentinels before and after data).
// i.e., verify that data from hipFileRead does not clobber outside specified device buffer.
TEST_P(HipFileVerify, RoundTripGuardsDeviceSlack)
{
    const size_t n       = elems();
    const hoff_t buf_off = static_cast<hoff_t>(kFourKiB); // one head device sentinel region block

    seedFilePattern(tmpfile.fd, 0, n);
    ASSERT_EQ(static_cast<ssize_t>(io_bytes),
              hipFileRead(tmpfile_handle, device_buffer, io_bytes, 0, buf_off));

    // Device layout (each sentinel region slackElems() ints, data n ints):
    // [head device sentinel region][data][tail device sentinel region]
    int32_t *data = bufferStart() + slackElems();
    launchAndVerify(bufferStart(), bufferElems(), data, n, defaultGrid(n), dim3(kDefaultBlockSize));

    ASSERT_EQ(static_cast<ssize_t>(io_bytes),
              hipFileWrite(tmpfile_handle, device_buffer, io_bytes, 0, buf_off));

    std::vector<int32_t> file = readFileInts(tmpfile.fd, 0, n);
    assertDoubledPattern(file.data(), n);
}

// hipFileRead + hipFileWrite at offset into file buffer (places sentinels before and after data).
// i.e., verify that data from hipFileWrite does not clobber outside specified file buffer.
TEST_P(HipFileVerify, RoundTripGuardsFileSlack)
{
    const size_t n         = elems();
    const size_t bracket_n = kFourKiB / sizeof(int32_t);    // file sentinel region head/tail each
    const hoff_t file_off  = static_cast<hoff_t>(kFourKiB); // data starts after head
    const size_t total_n   = bracket_n + n + bracket_n;     // head + data + tail
    const hoff_t tail_off  = static_cast<hoff_t>((bracket_n + n) * sizeof(int32_t));

    // File layout (each sentinel region bracket_n ints, data n ints):
    // [head file sentinel region = -1][data = i+1][tail file sentinel region = -1].
    seedFileConstant(tmpfile.fd, 0, bracket_n, kSentinel);
    seedFilePattern(tmpfile.fd, file_off, n);
    seedFileConstant(tmpfile.fd, tail_off, bracket_n, kSentinel);

    ASSERT_EQ(static_cast<ssize_t>(io_bytes),
              hipFileRead(tmpfile_handle, device_buffer, io_bytes, file_off, 0));

    launchAndVerify(bufferStart(), bufferElems(), bufferStart(), n, defaultGrid(n), dim3(kDefaultBlockSize));

    ASSERT_EQ(static_cast<ssize_t>(io_bytes),
              hipFileWrite(tmpfile_handle, device_buffer, io_bytes, file_off, 0));

    std::vector<int32_t> file = readFileInts(tmpfile.fd, 0, total_n);
    assertConstant(file.data(), 0, bracket_n, kSentinel);
    assertDoubledPattern(file.data() + bracket_n, n);
    assertConstant(file.data(), bracket_n + n, total_n, kSentinel);
}

static std::string
verifyName(const testing::TestParamInfo<HipFileVerify::ParamType> &info)
{
    return std::get<0>(info.param).name + "_" + std::get<1>(info.param).name;
}

INSTANTIATE_TEST_SUITE_P(, HipFileVerify,
                         testing::Combine(testing::ValuesIn(io_test_params), testing::ValuesIn(verify_sizes)),
                         verifyName);

// ---------------------------------------------------------------------------
// Combined all-regions round-trip (AIHIPFILE-233). One transfer exercises the device
// sentinel region (device over-allocation on both sides), the file sentinel region
// (in-file sentinel brackets on both sides), and a file_offset PAST the 16 MiB chunk
// boundary, using
// a parameterized modify_stride so different fractions of each cache line are
// touched. A block-count knob lets the sweep spread the grid across CUs/XCDs on
// multi-XCD parts, exercising both backends' device-memory coherence.
//
// Cache-line coverage from modify_stride (data is int32 -> 4 B/element; the data
// region starts 4 KiB into the allocation, so index 0 is 64 B- and 128 B-aligned):
//   - a 64 B line holds 16 elements, a 128 B line holds 32 elements;
//   - the kernel doubles element d iff d % stride == 0, so a line is fully SKIPPED
//     when it contains no multiple of stride (i.e. stride > elements-per-line).
//   - stride 2  (8 B): every line partially modified (dense baseline).
//   - stride 32 (128 B): every other 64 B line fully skipped; 128 B lines partial.
//   - stride 64 (256 B): whole lines skipped at BOTH 64 B and 128 B granularity.
//   - stride 512 (2 KiB): block isolation -- with 300 blocks x 256 threads the
//     grid-stride window means only even blocks contain a multiple of 512, so half
//     the blocks verify-in (load) the data but never write it.
// ---------------------------------------------------------------------------
struct StrideParam {
    size_t      stride;
    std::string name;
};

struct HipFileVerifyCombined
    : public testing::TestWithParam<std::tuple<IoTestParam, SizeParam, BlockParam, StrideParam>> {

    Tmpfile         tmpfile;
    hipFileHandle_t tmpfile_handle{nullptr};
    void           *device_buffer{nullptr};
    size_t          io_bytes{0};     // bytes transferred using the hipFile API
    size_t          buffer_bytes{0}; // total device buffer size

    HipFileVerifyCombined() : tmpfile{test_env.ais_capable_dir}
    {
    }

    IoTestBackend backend() const
    {
        return std::get<0>(GetParam()).backend;
    }

    GridMode gridMode() const
    {
        return std::get<2>(GetParam()).mode;
    }

    size_t stride() const
    {
        return std::get<3>(GetParam()).stride;
    }

    void SetUp() override
    {
        Context<Configuration>::get()->fastpath(false);
        Context<Configuration>::get()->fallback(false);

        switch (backend()) {
            case IoTestBackend::Fastpath:
                Context<Configuration>::get()->fastpath(true);
                break;
            case IoTestBackend::Fallback:
                Context<Configuration>::get()->fallback(true);
                break;
            default:
                FAIL() << "Unsupported IoTestBackend";
        }

        io_bytes = std::get<1>(GetParam()).bytes;
        // Over-allocate a device sentinel region block on each side of the data so the
        // kernel can confirm hipFile touched only the data region of the device
        // allocation. Device layout (each sentinel region kFourKiB, data io_bytes):
        // [head device sentinel region][data][tail device sentinel region].
        buffer_bytes = io_bytes + 2 * kFourKiB;

        // File layout (each sentinel region kFourKiB, data io_bytes; data begins at file
        // offset kCombinedFileOff past the chunk boundary):
        // [head file sentinel region][data][tail file sentinel region].
        const hoff_t tail_off = kCombinedFileOff + static_cast<hoff_t>(io_bytes);
        ASSERT_EQ(0, ftruncate(tmpfile.fd, static_cast<off_t>(tail_off + static_cast<hoff_t>(kFourKiB))));

        hipFileDescr_t descr{};
        descr.type      = hipFileHandleTypeOpaqueFD;
        descr.handle.fd = tmpfile.fd;
        ASSERT_EQ(HIPFILE_SUCCESS, hipFileHandleRegister(&tmpfile_handle, &descr));

        ASSERT_EQ(hipSuccess, hipMalloc(&device_buffer, buffer_bytes));
        ASSERT_EQ(hipSuccess, hipMemset(device_buffer, kSentinelByte, buffer_bytes));
        // hipMemset is not synchronous w.r.t. the host, and hipFileRead is not ordered w.r.t. the stream.
        ASSERT_EQ(hipSuccess, hipDeviceSynchronize());
    }

    void TearDown() override
    {
        if (device_buffer) {
            ASSERT_EQ(hipSuccess, hipFree(device_buffer));
        }
        hipFileHandleDeregister(tmpfile_handle);
    }

    size_t elems() const
    {
        return io_bytes / sizeof(int32_t);
    }

    size_t bufferElems() const
    {
        return buffer_bytes / sizeof(int32_t);
    }

    int32_t *bufferStart() const
    {
        return static_cast<int32_t *>(device_buffer);
    }
};

// Verify that reading data from an (kChunkBytes + sentinel region) offset into an
// offset of the device buffer reads the specified data, does not clobber the rest
// of the device buffer, and correctly writes back to the file without clobbering.
TEST_P(HipFileVerifyCombined, RoundTripGuardsAllRegions)
{
    const size_t n        = elems();
    const size_t slack_n  = kFourKiB / sizeof(int32_t);
    const hoff_t buf_off  = static_cast<hoff_t>(kFourKiB); // device head device sentinel region
    const hoff_t file_off = kCombinedFileOff;
    const hoff_t head_off = file_off - static_cast<hoff_t>(kFourKiB); // head file sentinel region
    const hoff_t tail_off = file_off + static_cast<hoff_t>(io_bytes);
    const size_t kStride  = stride();

    // File layout (hole is kChunkBytes, each sentinel region slack_n ints, data n ints):
    // [unwritten hole = 0][head file sentinel region = -1][data = i+1][tail file sentinel region = -1].
    seedFileConstant(tmpfile.fd, head_off, slack_n, kSentinel);
    seedFilePattern(tmpfile.fd, file_off, n);
    seedFileConstant(tmpfile.fd, tail_off, slack_n, kSentinel);

    ASSERT_EQ(static_cast<ssize_t>(io_bytes),
              hipFileRead(tmpfile_handle, device_buffer, io_bytes, file_off, buf_off));

    // Device layout (each sentinel region slackElems() ints, data n ints):
    // [head device sentinel region][data][tail device sentinel region].
    int32_t *data = bufferStart() + slackElems();
    launchAndVerify(bufferStart(), bufferElems(), data, n, gridFor(gridMode(), n), dim3(kDefaultBlockSize),
                    kStride);

    ASSERT_EQ(static_cast<ssize_t>(io_bytes),
              hipFileWrite(tmpfile_handle, device_buffer, io_bytes, file_off, buf_off));

    assertHoleZero(tmpfile.fd, 0, head_off);
    std::vector<int32_t> head = readFileInts(tmpfile.fd, head_off, slack_n);
    assertConstant(head.data(), 0, slack_n, kSentinel);
    std::vector<int32_t> body = readFileInts(tmpfile.fd, file_off, n);
    assertModifiedPattern(body.data(), n, kStride);
    std::vector<int32_t> tail = readFileInts(tmpfile.fd, tail_off, slack_n);
    assertConstant(tail.data(), 0, slack_n, kSentinel);
}

HIPFILE_WARN_NO_EXIT_DTOR_OFF
const std::array<BlockParam, 1> default_blocks{{{GridMode::Default, "auto"}}};
const std::array<BlockParam, 2> sweep_blocks{{
    {GridMode::OneBlock, "1blk"},
    {GridMode::ManyBlocks, "300blk"},
}};
const std::array<SizeParam, 1>  multi_only{{{2 * kChunkBytes, "multi_chunk"}}};
// stride 2  (8 B): every cache line partially modified (dense baseline).
// stride 32 (128 B): every other 64 B line fully skipped; 128 B lines partial.
// stride 64 (256 B): whole lines skipped at BOTH 64 B and 128 B granularity.
// stride 512 (2 KiB): with 300 blocks x 256 threads, only even blocks contain a multiple of 512, so half
//                     the blocks verify-in (load) the data but never write anything.
const std::array<StrideParam, 3> line_strides{{
    {2, "stride2"},
    {32, "stride32"},
    {64, "stride64"},
}};
const std::array<StrideParam, 2> block_strides{{
    {2, "stride2"},
    {512, "stride512"},
}};
HIPFILE_WARN_NO_EXIT_DTOR_ON

static std::string
combinedName(const testing::TestParamInfo<HipFileVerifyCombined::ParamType> &info)
{
    return std::get<0>(info.param).name + "_" + std::get<1>(info.param).name + "_" +
           std::get<2>(info.param).name + "_" + std::get<3>(info.param).name;
}

INSTANTIATE_TEST_SUITE_P(Sizes, HipFileVerifyCombined,
                         testing::Combine(testing::ValuesIn(io_test_params),
                                          testing::ValuesIn(combined_sizes),
                                          testing::ValuesIn(default_blocks), testing::ValuesIn(line_strides)),
                         combinedName);

INSTANTIATE_TEST_SUITE_P(Blocks, HipFileVerifyCombined,
                         testing::Combine(testing::ValuesIn(io_test_params), testing::ValuesIn(multi_only),
                                          testing::ValuesIn(sweep_blocks), testing::ValuesIn(block_strides)),
                         combinedName);

// ---------------------------------------------------------------------------
// Tests for extending the file length.
// ---------------------------------------------------------------------------

struct HipFileExtend : public testing::TestWithParam<std::tuple<IoTestParam, ScenarioParam, SizeParam>> {

    Tmpfile         tmpfile;
    hipFileHandle_t tmpfile_handle{nullptr};
    void           *device_buffer{nullptr};
    size_t          io_bytes{0};
    size_t          buffer_bytes{0};
    hoff_t          base_len{0};
    hoff_t          file_off{0};
    ExtendKind      kind{ExtendKind::AppendFromEmpty};

    HipFileExtend() : tmpfile{test_env.ais_capable_dir}
    {
    }

    IoTestBackend backend() const
    {
        return std::get<0>(GetParam()).backend;
    }

    void SetUp() override
    {
        Context<Configuration>::get()->fastpath(false);
        Context<Configuration>::get()->fallback(false);
        switch (backend()) {
            case IoTestBackend::Fastpath:
                Context<Configuration>::get()->fastpath(true);
                break;
            case IoTestBackend::Fallback:
                Context<Configuration>::get()->fallback(true);
                break;
            default:
                FAIL() << "Unsupported IoTestBackend";
        }

        const ScenarioParam &sc = std::get<1>(GetParam());
        base_len                = sc.base_len;
        file_off                = sc.file_off;
        kind                    = sc.kind;
        io_bytes                = std::get<2>(GetParam()).bytes;

        // Size file to base_len to be extended.
        ASSERT_EQ(0, ftruncate(tmpfile.fd, static_cast<off_t>(base_len)));

        // One head sentinel block + data + one tail sentinel block.
        // [head device sentinel region][data][tail device sentinel region].
        buffer_bytes = io_bytes + 2 * kFourKiB;

        hipFileDescr_t descr{};
        descr.type      = hipFileHandleTypeOpaqueFD;
        descr.handle.fd = tmpfile.fd;
        ASSERT_EQ(HIPFILE_SUCCESS, hipFileHandleRegister(&tmpfile_handle, &descr));

        ASSERT_EQ(hipSuccess, hipMalloc(&device_buffer, buffer_bytes));
        ASSERT_EQ(hipSuccess, hipMemset(device_buffer, kSentinelByte, buffer_bytes));
    }

    void TearDown() override
    {
        if (device_buffer) {
            ASSERT_EQ(hipSuccess, hipFree(device_buffer));
        }
        hipFileHandleDeregister(tmpfile_handle);
    }

    size_t elems() const
    {
        return io_bytes / sizeof(int32_t);
    }
    size_t bufferElems() const
    {
        return buffer_bytes / sizeof(int32_t);
    }
    int32_t *bufferStart() const
    {
        return static_cast<int32_t *>(device_buffer);
    }
};

// Test that hipFileWrite can extend the lenght of the file and that contents of
// the file outside of the written region are not clobbered.
TEST_P(HipFileExtend, Extends)
{
    const size_t n       = elems();
    const size_t slack_n = slackElems();
    const hoff_t buf_off =
        static_cast<hoff_t>(kFourKiB); // aligned head device sentinel region (fastpath-legal)
    constexpr size_t kStride = 2;

    // Device layout (each sentinel region slack_n ints, data n ints):
    // [head device sentinel region][data][tail device sentinel region].
    // data starts after the head device sentinel region block; seed the index pattern there.
    int32_t *data = bufferStart() + slack_n;
    seedDevicePattern(device_buffer, buf_off, n);

    launchAndVerify(bufferStart(), bufferElems(), data, n, defaultGrid(n), dim3(kDefaultBlockSize), kStride);

    // Fill the entire existing-data region below the write, [0, min(file_off,
    // base_len)), with sentinels so an under-run into existing data is caught
    // ANYWHERE below file_off, not just in the int immediately before it. The
    // hole portion [base_len, file_off) (if any) is left untouched and checked as
    // zero below; the overwritten portion [file_off, base_len) (if any) is
    // clobbered by the write and checked as payload.
    const hoff_t preserved_end = (file_off < base_len) ? file_off : base_len;
    const size_t preserved_n   = static_cast<size_t>(preserved_end) / sizeof(int32_t);
    if (preserved_n > 0) {
        seedFileConstant(tmpfile.fd, 0, preserved_n, kSentinel);
    }

    ASSERT_EQ(static_cast<ssize_t>(io_bytes),
              hipFileWrite(tmpfile_handle, device_buffer, io_bytes, file_off, buf_off));

    std::vector<int32_t> body = readFileInts(tmpfile.fd, file_off, n);
    assertModifiedPattern(body.data(), n, kStride);

    // Hole zero-fill.
    if (file_off > base_len) {
        assertHoleZero(tmpfile.fd, base_len, file_off);
    }

    // Final size correct.
    ASSERT_EQ(file_off + static_cast<hoff_t>(io_bytes), fileSize(tmpfile.fd));

    // Existing data below the write is fully intact.
    if (preserved_n > 0) {
        std::vector<int32_t> head = readFileInts(tmpfile.fd, 0, preserved_n);
        assertConstant(head.data(), 0, preserved_n, kSentinel);
    }
}

static std::string
extendName(const testing::TestParamInfo<HipFileExtend::ParamType> &info)
{
    return std::get<0>(info.param).name + "_" + std::get<1>(info.param).name + "_" +
           std::get<2>(info.param).name;
}

INSTANTIATE_TEST_SUITE_P(, HipFileExtend,
                         testing::Combine(testing::ValuesIn(io_test_params),
                                          testing::ValuesIn(extend_scenarios),
                                          testing::ValuesIn(extend_sizes)),
                         extendName);

HIPFILE_WARN_NO_GLOBAL_CTOR_ON
