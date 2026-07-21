/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "hipfile-literals.h"
#include "hipfile-warnings.h"
#include "hipfile.h"
#include "test-common.h"
#include "test-options.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>
#include <sys/resource.h>
#include <sys/types.h>
#include <unistd.h>
#include <vector>

extern SystemTestOptions test_env;

namespace {

struct BatchOpCookie {
    size_t index;
};

std::vector<uint8_t>
pattern(size_t size, uint8_t seed)
{
    std::vector<uint8_t> data(size);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<uint8_t>(seed + i);
    }
    return data;
}

void
write_all(int fd, const std::vector<uint8_t> &data, off_t offset)
{
    size_t written{};
    while (written < data.size()) {
        const ssize_t result =
            pwrite(fd, data.data() + written, data.size() - written, offset + static_cast<off_t>(written));
        ASSERT_GT(result, 0);
        written += static_cast<size_t>(result);
    }
    ASSERT_EQ(fdatasync(fd), 0);
}

std::vector<uint8_t>
read_all(int fd, size_t size, off_t offset)
{
    std::vector<uint8_t> data(size);
    size_t               bytes_read{};
    while (bytes_read < data.size()) {
        const ssize_t result = pread(fd, data.data() + bytes_read, data.size() - bytes_read,
                                     offset + static_cast<off_t>(bytes_read));
        if (result <= 0) {
            ADD_FAILURE() << "pread returned " << result;
            return data;
        }
        bytes_read += static_cast<size_t>(result);
    }
    return data;
}

struct BatchTest : public testing::Test {
    Tmpfile         tmpfile{test_env.ais_capable_dir};
    hipFileHandle_t file_handle{};
    void           *device_buffer{};
    size_t          op_size{4096};
    unsigned        op_count{4};
    size_t          file_size{op_size * op_count};

    size_t                     device_buffer_size{file_size};
    hipFileBatchHandle_t       batch_handle{};
    std::vector<uint8_t>       host_buffer;
    std::vector<BatchOpCookie> cookies;

    void SetUp() override
    {
        ASSERT_EQ(ftruncate(tmpfile.fd, static_cast<off_t>(file_size)), 0);

        hipFileDescr_t descr{};
        descr.type      = hipFileHandleTypeOpaqueFD;
        descr.handle.fd = tmpfile.fd;
        ASSERT_EQ(hipFileHandleRegister(&file_handle, &descr), HIPFILE_SUCCESS);

        ASSERT_EQ(hipMalloc(&device_buffer, device_buffer_size), hipSuccess);
        ASSERT_EQ(hipMemset(device_buffer, 0, device_buffer_size), hipSuccess);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        ASSERT_EQ(hipFileBufRegister(device_buffer, device_buffer_size, 0), HIPFILE_SUCCESS);

        host_buffer = pattern(device_buffer_size, 0x31);
        cookies.resize(op_count);
        for (size_t i = 0; i < cookies.size(); ++i) {
            cookies[i].index = i;
        }
    }

    void TearDown() override
    {
        if (batch_handle != nullptr) {
            hipFileBatchIODestroy(batch_handle);
            batch_handle = nullptr;
        }
        if (device_buffer != nullptr) {
            EXPECT_EQ(hipFileBufDeregister(device_buffer), HIPFILE_SUCCESS);
            EXPECT_EQ(hipFree(device_buffer), hipSuccess);
            device_buffer = nullptr;
        }
        if (file_handle != nullptr) {
            hipFileHandleDeregister(file_handle);
            file_handle = nullptr;
        }
        while (hipFileUseCount() > 0) {
            EXPECT_EQ(hipFileDriverClose(), HIPFILE_SUCCESS);
        }
    }

    void setupBatch(unsigned capacity)
    {
        ASSERT_EQ(hipFileBatchIOSetUp(&batch_handle, capacity), HIPFILE_SUCCESS);
        ASSERT_NE(batch_handle, nullptr);
    }

    hipFileIOParams_t makeOp(size_t index, hipFileOpcode_t opcode)
    {
        hipFileIOParams_t op{};
        op.mode                  = hipFileBatch;
        op.u.batch.devPtr_base   = device_buffer;
        op.u.batch.file_offset   = static_cast<int64_t>(index * op_size);
        op.u.batch.devPtr_offset = static_cast<int64_t>(index * op_size);
        op.u.batch.size          = op_size;
        op.fh                    = file_handle;
        op.opcode                = opcode;
        op.cookie                = &cookies[index];
        return op;
    }

    std::vector<hipFileIOEvents_t> waitForEvents(unsigned expected)
    {
        std::vector<hipFileIOEvents_t> events(expected);
        unsigned                       nr = expected;
        EXPECT_EQ(hipFileBatchIOGetStatus(batch_handle, expected, &nr, events.data(), nullptr),
                  HIPFILE_SUCCESS);
        events.resize(nr);
        return events;
    }

    void expectCompleteEvents(const std::vector<hipFileIOEvents_t> &events, unsigned expected)
    {
        ASSERT_EQ(events.size(), expected);
        std::vector<size_t> seen;
        seen.reserve(events.size());
        for (const auto &event : events) {
            ASSERT_NE(event.cookie, nullptr);
            const auto *cookie = static_cast<const BatchOpCookie *>(event.cookie);
            EXPECT_LT(cookie->index, op_count);
            seen.push_back(cookie->index);
            EXPECT_EQ(event.status, hipFileComplete);
            EXPECT_EQ(event.ret, op_size);
        }
        std::sort(seen.begin(), seen.end());
        ASSERT_EQ(std::adjacent_find(seen.begin(), seen.end()), seen.end());
    }
};

struct BatchWriteFailureTest : public BatchTest {
    void SetUp() override
    {
        file_size          = 1_MiB;
        device_buffer_size = 1_MiB;
#if defined(__HIP_PLATFORM_NVIDIA__)
        // This fixture has a test that constrains process file writes. Keep cuFile logging out
        // of that limit so only the target I/O sees the failure.
        ASSERT_EQ(setenv("CUFILE_LOGFILE_PATH", "/dev/null", 1), 0);
#endif
        BatchTest::SetUp();
    }
};

struct BatchCancelTest : public BatchTest {
    void SetUp() override
    {
        op_count           = 128;
        file_size          = op_size * op_count;
        device_buffer_size = file_size;
        BatchTest::SetUp();
    }
};

struct ScopedSignalAction {
    int ignore(int signal_number_)
    {
        signal_number = signal_number_;
        old_handler   = std::signal(signal_number, SIG_IGN);
        active        = old_handler != SIG_ERR;
        return active ? 0 : -1;
    }

    ~ScopedSignalAction()
    {
        if (active) {
            (void)std::signal(signal_number, old_handler);
        }
    }

    using SignalHandler = void (*)(int);

    int           signal_number{};
    SignalHandler old_handler{SIG_DFL};
    bool          active{};
};

struct ScopedFileSizeLimit {
    int limit(rlim_t soft_limit)
    {
        if (getrlimit(RLIMIT_FSIZE, &old_limit) != 0) {
            return -1;
        }

        struct rlimit new_limit = old_limit;
        new_limit.rlim_cur      = soft_limit;
        const int status        = setrlimit(RLIMIT_FSIZE, &new_limit);
        active                  = status == 0;
        return status;
    }

    ~ScopedFileSizeLimit()
    {
        if (active) {
            (void)setrlimit(RLIMIT_FSIZE, &old_limit);
        }
    }

    struct rlimit old_limit {};
    bool          active{};
};

} // namespace

HIPFILE_WARN_NO_GLOBAL_CTOR_OFF

TEST_F(BatchTest, BatchReadSingleOperation)
{
    const auto input = pattern(op_size, 0x41);
    write_all(tmpfile.fd, input, 0);
    ASSERT_EQ(hipMemset(device_buffer, 0, device_buffer_size), hipSuccess);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    setupBatch(1);
    auto op = makeOp(0, hipFileBatchRead);
    ASSERT_EQ(hipFileBatchIOSubmit(batch_handle, 1, &op, 0), HIPFILE_SUCCESS);

    const auto events = waitForEvents(1);
    expectCompleteEvents(events, 1);

    std::vector<uint8_t> output(op_size);
    ASSERT_EQ(hipMemcpy(output.data(), device_buffer, output.size(), hipMemcpyDeviceToHost), hipSuccess);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    ASSERT_EQ(output, input);
}

TEST_F(BatchTest, BatchReadMultipleOperations)
{
    const auto input = pattern(file_size, 0x51);
    write_all(tmpfile.fd, input, 0);
    ASSERT_EQ(hipMemset(device_buffer, 0, device_buffer_size), hipSuccess);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    setupBatch(op_count);
    std::vector<hipFileIOParams_t> ops(op_count);
    for (size_t i = 0; i < ops.size(); ++i) {
        ops[i] = makeOp(i, hipFileBatchRead);
    }
    ASSERT_EQ(hipFileBatchIOSubmit(batch_handle, static_cast<unsigned>(ops.size()), ops.data(), 0),
              HIPFILE_SUCCESS);

    const auto events = waitForEvents(static_cast<unsigned>(ops.size()));
    expectCompleteEvents(events, static_cast<unsigned>(ops.size()));

    std::vector<uint8_t> output(file_size);
    ASSERT_EQ(hipMemcpy(output.data(), device_buffer, output.size(), hipMemcpyDeviceToHost), hipSuccess);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    ASSERT_EQ(output, input);
}

TEST_F(BatchTest, BatchWriteSingleOperation)
{
    const auto input = pattern(op_size, 0x61);
    ASSERT_EQ(hipMemcpy(device_buffer, input.data(), input.size(), hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    setupBatch(1);
    auto op = makeOp(0, hipFileBatchWrite);
    ASSERT_EQ(hipFileBatchIOSubmit(batch_handle, 1, &op, 0), HIPFILE_SUCCESS);

    const auto events = waitForEvents(1);
    expectCompleteEvents(events, 1);
    ASSERT_EQ(read_all(tmpfile.fd, op_size, 0), input);
}

TEST_F(BatchTest, BatchWriteMultipleOperations)
{
    const auto input = pattern(file_size, 0x71);
    ASSERT_EQ(hipMemcpy(device_buffer, input.data(), input.size(), hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    setupBatch(op_count);
    std::vector<hipFileIOParams_t> ops(op_count);
    for (size_t i = 0; i < ops.size(); ++i) {
        ops[i] = makeOp(i, hipFileBatchWrite);
    }
    ASSERT_EQ(hipFileBatchIOSubmit(batch_handle, static_cast<unsigned>(ops.size()), ops.data(), 0),
              HIPFILE_SUCCESS);

    const auto events = waitForEvents(static_cast<unsigned>(ops.size()));
    expectCompleteEvents(events, static_cast<unsigned>(ops.size()));
    ASSERT_EQ(read_all(tmpfile.fd, file_size, 0), input);
}

TEST_F(BatchTest, ReusedBatchHandleAcceptsSequentialFullBatches)
{
    const auto input = pattern(file_size, 0x35);
    write_all(tmpfile.fd, input, 0);

    setupBatch(op_count);

    for (int round = 0; round < 2; ++round) {
        ASSERT_EQ(hipMemset(device_buffer, 0, device_buffer_size), hipSuccess);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        std::vector<hipFileIOParams_t> ops(op_count);
        for (size_t i = 0; i < ops.size(); ++i) {
            ops[i] = makeOp(i, hipFileBatchRead);
        }
        ASSERT_EQ(hipFileBatchIOSubmit(batch_handle, static_cast<unsigned>(ops.size()), ops.data(), 0),
                  HIPFILE_SUCCESS)
            << "submit failed on round " << round;

        const auto events = waitForEvents(static_cast<unsigned>(ops.size()));
        expectCompleteEvents(events, static_cast<unsigned>(ops.size()));

        std::vector<uint8_t> output(file_size);
        ASSERT_EQ(hipMemcpy(output.data(), device_buffer, output.size(), hipMemcpyDeviceToHost), hipSuccess);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        ASSERT_EQ(output, input) << "data mismatch on round " << round;
    }
}

TEST_F(BatchWriteFailureTest, FailedBatchWriteReportsErrorInEventRet)
{
    ScopedSignalAction xfsz;
    ASSERT_EQ(xfsz.ignore(SIGXFSZ), 0);

    ScopedFileSizeLimit file_size_limit;
    ASSERT_EQ(file_size_limit.limit(static_cast<rlim_t>(file_size)), 0);

    setupBatch(1);
    auto op                = makeOp(0, hipFileBatchWrite);
    op.u.batch.size        = file_size;
    op.u.batch.file_offset = static_cast<hoff_t>(file_size);

    ASSERT_EQ(hipFileBatchIOSubmit(batch_handle, 1, &op, 0), HIPFILE_SUCCESS);

    const auto events = waitForEvents(1);
    ASSERT_EQ(events.size(), 1);

    const auto &event = events[0];
    ASSERT_EQ(event.cookie, &cookies[0]);
    EXPECT_EQ(event.status, hipFileFailed);

#if defined(__HIP_PLATFORM_NVIDIA__)
    const auto expected_ret = -static_cast<ssize_t>(EIO);
#else
    const auto expected_ret = -static_cast<ssize_t>(EFBIG);
#endif
    EXPECT_EQ(static_cast<ssize_t>(event.ret), expected_ret);
    EXPECT_EQ(event.ret, static_cast<size_t>(expected_ret));
}

TEST_F(BatchTest, GetStatusWithSmallEventBufferReturnsRemainingEventsLater)
{
    const auto input = pattern(file_size, 0x21);
    write_all(tmpfile.fd, input, 0);

    setupBatch(op_count);
    std::vector<hipFileIOParams_t> ops(op_count);
    for (size_t i = 0; i < ops.size(); ++i) {
        ops[i] = makeOp(i, hipFileBatchRead);
    }
    ASSERT_EQ(hipFileBatchIOSubmit(batch_handle, static_cast<unsigned>(ops.size()), ops.data(), 0),
              HIPFILE_SUCCESS);

    std::vector<hipFileIOEvents_t> all_events;
    while (all_events.size() < op_count) {
        std::array<hipFileIOEvents_t, 2> events{};
        unsigned                         nr = events.size();
        ASSERT_EQ(hipFileBatchIOGetStatus(batch_handle, 1, &nr, events.data(), nullptr), HIPFILE_SUCCESS);
        ASSERT_GT(nr, 0);
        ASSERT_LE(nr, events.size());
        all_events.insert(all_events.end(), events.begin(), events.begin() + nr);
    }
    expectCompleteEvents(all_events, op_count);
}

TEST_F(BatchTest, GetStatusNoOutstandingReturnsZero)
{
    setupBatch(1);
    hipFileIOEvents_t event{};
    unsigned          nr = 1;
    struct timespec   timeout {
        1, 0
    };

    ASSERT_EQ(hipFileBatchIOGetStatus(batch_handle, 1, &nr, &event, &timeout), HIPFILE_SUCCESS);
    ASSERT_EQ(nr, 0);
}

TEST_F(BatchTest, CancelEmptyBatchSucceeds)
{
    setupBatch(1);
    ASSERT_EQ(hipFileBatchIOCancel(batch_handle), HIPFILE_SUCCESS);

    hipFileIOEvents_t event{};
    unsigned          nr = 1;
    ASSERT_EQ(hipFileBatchIOGetStatus(batch_handle, 0, &nr, &event, nullptr), HIPFILE_SUCCESS);
    ASSERT_EQ(nr, 0);
}

TEST_F(BatchCancelTest, CancelFullBatchReportsTerminalStatus)
{
    const auto input = pattern(file_size, 0x19);
    write_all(tmpfile.fd, input, 0);

    setupBatch(op_count);
    std::vector<hipFileIOParams_t> ops(op_count);
    for (size_t i = 0; i < ops.size(); ++i) {
        ops[i] = makeOp(i, hipFileBatchRead);
    }
    ASSERT_EQ(hipFileBatchIOSubmit(batch_handle, static_cast<unsigned>(ops.size()), ops.data(), 0),
              HIPFILE_SUCCESS);

    ASSERT_EQ(hipFileBatchIOCancel(batch_handle), HIPFILE_SUCCESS);

    const auto events = waitForEvents(op_count);
    ASSERT_EQ(events.size(), op_count);
    std::vector<size_t> seen;
    seen.reserve(events.size());
    for (const auto &event : events) {
        ASSERT_NE(event.cookie, nullptr);
        const auto *cookie = static_cast<const BatchOpCookie *>(event.cookie);
        EXPECT_LT(cookie->index, op_count);
        seen.push_back(cookie->index);
        EXPECT_TRUE(event.status == hipFileCanceled || event.status == hipFileComplete)
            << "unexpected status " << event.status;
    }
    std::sort(seen.begin(), seen.end());
    ASSERT_EQ(std::adjacent_find(seen.begin(), seen.end()), seen.end());
}

TEST_F(BatchTest, DestroyEmptyBatchDoesNotCrash)
{
    setupBatch(1);
    hipFileBatchIODestroy(batch_handle);
    batch_handle = nullptr;

    hipFileBatchIODestroy(batch_handle);
}

TEST_F(BatchTest, SubmitRejectsInvalidArguments)
{
    setupBatch(1);
    auto                             op = makeOp(0, hipFileBatchRead);
    std::array<hipFileIOParams_t, 2> ops{op, makeOp(1, hipFileBatchRead)};

#if defined(__HIP_PLATFORM_NVIDIA__)
    constexpr auto invalid_submit_arg_error      = HipFileOpError(hipFileInternalError);
    constexpr auto invalid_submit_capacity_error = HipFileOpError(hipFileInternalError);
#else
    constexpr auto invalid_submit_arg_error      = HipFileOpError(hipFileInvalidValue);
    constexpr auto invalid_submit_capacity_error = HipFileOpError(hipFileBatchFull);
#endif
    ASSERT_EQ(hipFileBatchIOSubmit(batch_handle, 0, &op, 0), invalid_submit_arg_error);
    ASSERT_EQ(hipFileBatchIOSubmit(batch_handle, 1, nullptr, 0), invalid_submit_arg_error);
    ASSERT_EQ(hipFileBatchIOSubmit(batch_handle, ops.size(), ops.data(), 0), invalid_submit_capacity_error);
}

TEST_F(BatchTest, GetStatusRejectsInvalidArguments)
{
    setupBatch(1);
    hipFileIOEvents_t event{};
    unsigned          nr = 1;

#if defined(__HIP_PLATFORM_NVIDIA__)
    constexpr auto invalid_status_arg_error = HipFileOpError(hipFileInternalError);
#else
    constexpr auto invalid_status_arg_error = HipFileOpError(hipFileInvalidValue);
#endif
    ASSERT_EQ(hipFileBatchIOGetStatus(batch_handle, 0, nullptr, &event, nullptr), invalid_status_arg_error);
    ASSERT_EQ(hipFileBatchIOGetStatus(batch_handle, 0, &nr, nullptr, nullptr), invalid_status_arg_error);
    ASSERT_EQ(hipFileBatchIOGetStatus(batch_handle, 2, &nr, &event, nullptr), invalid_status_arg_error);
    nr = 0;
    ASSERT_EQ(hipFileBatchIOGetStatus(batch_handle, 0, &nr, &event, nullptr), invalid_status_arg_error);
}

HIPFILE_WARN_NO_GLOBAL_CTOR_ON
