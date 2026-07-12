/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

// Byte-granular, aligned combined round-trip: device + file sentinel guards, a
// file_offset past the chunk boundary, every-other-byte modify. See
// io_data_modification.cpp for the int32 variant and io_data_modification_unaligned.cpp
// for the unaligned byte sweep.

#include "context.h"
#include "configuration.h"
#include "hipfile-warnings.h"
#include "hipfile.h"

#include "io.hpp"
#include "io-verify.hpp"
#include "test-common.h"
#include "test-options.h"

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

// Combined test suite to exercise:
//    - untouched file and device buffer sentinel region
//    - file_offset past the chunking boundary
//    - modifying every other data element
struct HipFileVerifyBytes : public testing::TestWithParam<std::tuple<IoTestParam, SizeParam>> {

    Tmpfile         tmpfile;
    hipFileHandle_t tmpfile_handle{nullptr};
    void           *device_buffer{nullptr};
    size_t          io_bytes{0};     // bytes transferred using the hipFile API
    size_t          buffer_bytes{0}; // total device buffer size

    HipFileVerifyBytes() : tmpfile{test_env.ais_capable_dir}
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
        // The kernel will also verify that the sentinel regions of the device buffer are unmodified.
        // Device layout (each sentinel region kFourKiB, data io_bytes):
        // [head device sentinel region][data][tail device sentinel region]
        buffer_bytes = io_bytes + 2 * kFourKiB;

        // File layout (each sentinel region kFourKiB, data io_bytes; data begins at file
        // offset kCombinedFileOff, which is one chunk + kFourKiB in so it clears the boundary):
        // [head file sentinel region][data][tail file sentinel region]
        const hoff_t tail_off = kCombinedFileOff + static_cast<hoff_t>(io_bytes);
        ASSERT_EQ(0, ftruncate(tmpfile.fd, static_cast<off_t>(tail_off + static_cast<hoff_t>(kFourKiB))));

        hipFileDescr_t descr{};
        descr.type      = hipFileHandleTypeOpaqueFD;
        descr.handle.fd = tmpfile.fd;
        ASSERT_EQ(HIPFILE_SUCCESS, hipFileHandleRegister(&tmpfile_handle, &descr));

        ASSERT_EQ(hipSuccess, hipMalloc(&device_buffer, buffer_bytes));
        ASSERT_EQ(hipSuccess, hipMemset(device_buffer, kByteDevSlack, buffer_bytes));
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

    unsigned char *bufferStart() const
    {
        return static_cast<unsigned char *>(device_buffer);
    }
};

// All-regions guard (byte granularity): the data is read from a past-chunk
// file_offset (bracketed by the file sentinel region on disk) into the MIDDLE of the
// device allocation (bracketed by the device sentinel region), modified every-other
// byte on the GPU, then written back to the SAME past-chunk file_offset. One
// round-trip proves hipFile landed exactly io_bytes at exactly file_offset and
// touched only the data region of the device allocation -- leaving both device
// sentinel region blocks and both file sentinel region brackets intact.
TEST_P(HipFileVerifyBytes, RoundTripGuardsAllRegions)
{
    const size_t n        = io_bytes; // one byte per element
    const size_t slack_n  = kFourKiB; // file sentinel region bracket size in bytes
    const hoff_t buf_off  = static_cast<hoff_t>(kFourKiB);
    const hoff_t file_off = kCombinedFileOff;
    const hoff_t head_off = file_off - static_cast<hoff_t>(kFourKiB); // head file sentinel region byte offset
    const hoff_t tail_off = file_off + static_cast<hoff_t>(io_bytes);
    constexpr size_t kStride = 2; // every-other byte

    // File layout (each sentinel region slack_n bytes, data n bytes; data begins at
    // file offset file_off past the chunk boundary):
    // [unwritten sparse hole = 0][head file sentinel region = 0x55][data = 0xFF][tail file sentinel region =
    // 0x55] Everything below head_off (i.e. [0, file_off - kFourKiB)) is never seeded: ftruncate left it a
    // zero-filled sparse hole. It exists to push file_off past the chunk boundary and off chunk alignment; it
    // is outside the transfer window, so the round-trip must leave it untouched (checked below -- catches a
    // far under-run past the head bracket).
    seedFileBytesConstant(tmpfile.fd, head_off, slack_n, kByteFileSlack);
    seedFileBytesConstant(tmpfile.fd, file_off, n, kByteEntry);
    seedFileBytesConstant(tmpfile.fd, tail_off, slack_n, kByteFileSlack);

    // Read ONLY the data from its past-chunk file_offset into the buffer MIDDLE.
    ASSERT_EQ(static_cast<ssize_t>(io_bytes),
              hipFileRead(tmpfile_handle, device_buffer, io_bytes, file_off, buf_off));

    // Device layout (each sentinel region kFourKiB, data n bytes; data begins at buffer
    // offset buf_off = kFourKiB):
    // [head device sentinel region][data][tail device sentinel region]
    unsigned char *data = bufferStart() + kFourKiB;
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
byteName(const testing::TestParamInfo<HipFileVerifyBytes::ParamType> &info)
{
    return std::get<0>(info.param).name + "_" + std::get<1>(info.param).name;
}

INSTANTIATE_TEST_SUITE_P(, HipFileVerifyBytes,
                         testing::Combine(testing::ValuesIn(io_test_params),
                                          testing::ValuesIn(combined_sizes)),
                         byteName); // 2 backends x 3 sizes = 6

HIPFILE_WARN_NO_GLOBAL_CTOR_ON
