/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "fabric_write.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <iostream>
#include <limits>

#include "amd_smi/amdsmi.h"

TestFabricWrite::TestFabricWrite() : TestBase() {
  set_title("AMDSMI Fabric Write Test");
  set_description(
      "Tests amdsmi_set_gpu_fabric_ppod_config(), amdsmi_set_gpu_fabric_vpod_config(), "
      "and amdsmi_set_gpu_fabric_station_config() for input validation and hw paths, "
      "along with read->set+commit->read round-trips with matching getters");
}

TestFabricWrite::~TestFabricWrite() {}

void TestFabricWrite::SetUp() { TestBase::SetUp(); }

void TestFabricWrite::DisplayTestInfo() { TestBase::DisplayTestInfo(); }

void TestFabricWrite::DisplayResults() const { TestBase::DisplayResults(); }

void TestFabricWrite::Close() { TestBase::Close(); }

/**
 *  Helpers to make minimal fabric config structs
 *      - Used to test input validation and hardware paths
 *      - All fields are set to 0 or false, except for the mask and commit bit
 *      - The mask is set to the minimal required bits for the test
 *      - The commit bit is set to false
 *      - The struct is returned
 */
static auto make_minimal_ppod_config() -> amdsmi_fabric_ppod_config_t {
  amdsmi_fabric_ppod_config_t ppod_config = {};
  ppod_config.version = AMDSMI_FABRIC_PPOD_CONFIG_V1;
  ppod_config.mask = AMDSMI_FABRIC_PPOD_FIELD_ACCEL_ID;
  ppod_config.commit = false;
  ppod_config.data.accelerator_id = 0;
  return ppod_config;
}

static auto make_minimal_vpod_config() -> amdsmi_fabric_vpod_config_t {
  amdsmi_fabric_vpod_config_t vpod_config = {};
  vpod_config.version = AMDSMI_FABRIC_VPOD_CONFIG_V1;
  vpod_config.mask = AMDSMI_FABRIC_VPOD_FIELD_VPOD_ID;
  vpod_config.commit = false;
  vpod_config.data.vpod_id = 0;
  return vpod_config;
}

static auto make_minimal_station_config() -> amdsmi_fabric_station_config_t {
  amdsmi_fabric_station_config_t station_config = {};
  station_config.version = AMDSMI_FABRIC_STATION_CONFIG_V1;
  station_config.mask = AMDSMI_FABRIC_DF_FIELD_STATION_FLAGS;
  station_config.commit = false;
  station_config.data.station_flags = 0;
  return station_config;
}

/**
 *  Run the test
 */
void TestFabricWrite::Run() {
  TestBase::Run();
  if (setup_failed_) {
    IF_VERB(STANDARD) { std::cout << "** SetUp Failed for this test. Skipping.**" << "\n"; }
    return;
  }

  /**
   *    Null handle rejection (no device required)
   */
  {
    auto ppod = make_minimal_ppod_config();
    ASSERT_EQ(amdsmi_set_gpu_fabric_ppod_config(nullptr, &ppod), AMDSMI_STATUS_INVAL);

    auto vpod = make_minimal_vpod_config();
    ASSERT_EQ(amdsmi_set_gpu_fabric_vpod_config(nullptr, &vpod), AMDSMI_STATUS_INVAL);

    auto station = make_minimal_station_config();
    ASSERT_EQ(amdsmi_set_gpu_fabric_station_config(nullptr, &station), AMDSMI_STATUS_INVAL);

    amdsmi_fabric_info_t info_rd = {};
    ASSERT_EQ(amdsmi_get_gpu_fabric_info(nullptr, &info_rd), AMDSMI_STATUS_INVAL);
  }

  /**
   *    Null struct pointer rejection
   */
  if (num_monitor_devs() == 0) {
    IF_VERB(STANDARD) {
      std::cout << "\tNo GPU devices found. Skipping device-level validation tests" << "\n";
    }
    return;
  }
  auto device = processor_handles_[0];

  {
    ASSERT_EQ(amdsmi_set_gpu_fabric_ppod_config(device, nullptr), AMDSMI_STATUS_INVAL);
    ASSERT_EQ(amdsmi_set_gpu_fabric_vpod_config(device, nullptr), AMDSMI_STATUS_INVAL);
    ASSERT_EQ(amdsmi_set_gpu_fabric_station_config(device, nullptr), AMDSMI_STATUS_INVAL);

    ASSERT_EQ(amdsmi_get_gpu_fabric_info(device, nullptr), AMDSMI_STATUS_INVAL);
  }

  /**
   *    Version mismatch rejection
   */
  {
    auto ppod = make_minimal_ppod_config();
    ppod.version = 0;
    ASSERT_EQ(amdsmi_set_gpu_fabric_ppod_config(device, &ppod), AMDSMI_STATUS_INVAL);

    auto vpod = make_minimal_vpod_config();
    vpod.version = 0;
    ASSERT_EQ(amdsmi_set_gpu_fabric_vpod_config(device, &vpod), AMDSMI_STATUS_INVAL);

    auto station = make_minimal_station_config();
    station.version = 0;
    ASSERT_EQ(amdsmi_set_gpu_fabric_station_config(device, &station), AMDSMI_STATUS_INVAL);
  }

  /**
   *    No-op request rejection
   */
  {
    auto ppod = make_minimal_ppod_config();
    ppod.mask = 0;
    ppod.commit = false;
    ASSERT_EQ(amdsmi_set_gpu_fabric_ppod_config(device, &ppod), AMDSMI_STATUS_INVAL);

    auto vpod = make_minimal_vpod_config();
    vpod.mask = 0;
    vpod.commit = false;
    ASSERT_EQ(amdsmi_set_gpu_fabric_vpod_config(device, &vpod), AMDSMI_STATUS_INVAL);

    auto station = make_minimal_station_config();
    station.mask = 0;
    station.commit = false;
    ASSERT_EQ(amdsmi_set_gpu_fabric_station_config(device, &station), AMDSMI_STATUS_INVAL);
  }

  /**
   *    Invalid mask bits rejection
   */
  {
    auto ppod = make_minimal_ppod_config();
    ppod.mask = 0xFFFFFFFF;
    ASSERT_EQ(amdsmi_set_gpu_fabric_ppod_config(device, &ppod), AMDSMI_STATUS_INVAL);

    auto vpod = make_minimal_vpod_config();
    vpod.mask = 0xFFFFFFFF;
    ASSERT_EQ(amdsmi_set_gpu_fabric_vpod_config(device, &vpod), AMDSMI_STATUS_INVAL);

    auto station = make_minimal_station_config();
    station.mask = 0xFFFFFFFF;
    ASSERT_EQ(amdsmi_set_gpu_fabric_station_config(device, &station), AMDSMI_STATUS_INVAL);
  }

  /**
   *    Ppod: LOCAL_ACCELS with count=0 rejected
   */
  {
    auto ppod = make_minimal_ppod_config();
    ppod.mask = AMDSMI_FABRIC_PPOD_FIELD_LOCAL_ACCELS;
    ppod.data.local_accelerator_count = 0;
    ASSERT_EQ(amdsmi_set_gpu_fabric_ppod_config(device, &ppod), AMDSMI_STATUS_INVAL);
  }

  /**
   *    Ppod: LOCAL_ACCELS with count > max rejected
   */
  {
    auto ppod = make_minimal_ppod_config();
    ppod.mask = AMDSMI_FABRIC_PPOD_FIELD_LOCAL_ACCELS;
    ppod.data.local_accelerator_count = (AMDSMI_FABRIC_MAX_LOCAL_GPUS + 1);
    ASSERT_EQ(amdsmi_set_gpu_fabric_ppod_config(device, &ppod), AMDSMI_STATUS_INVAL);
  }

  /**
   *    Vpod: invalid addr_mode rejected before any sysfs write
   */
  {
    auto vpod = make_minimal_vpod_config();
    vpod.mask = (AMDSMI_FABRIC_VPOD_FIELD_VPOD_ID | AMDSMI_FABRIC_VPOD_FIELD_ADDR_MODE);
    vpod.data.addr_mode = AMDSMI_FABRIC_NPA_ADDRESS_MODE_UNKNOWN;
    ASSERT_EQ(amdsmi_set_gpu_fabric_vpod_config(device, &vpod), AMDSMI_STATUS_INVAL);
  }

  /**
   *    Hw path: accepts valid requests or reports NOT_SUPPORTED
   */
  for (auto dv_ind = uint32_t(0); dv_ind < num_monitor_devs(); ++dv_ind) {
    auto dev = processor_handles_[dv_ind];
    PrintDeviceHeader(dev);

    {
      auto ppod = make_minimal_ppod_config();
      auto status_code = amdsmi_set_gpu_fabric_ppod_config(dev, &ppod);
      if (status_code == AMDSMI_STATUS_NOT_SUPPORTED) {
        IF_VERB(STANDARD) {
          std::cout << "\t**amdsmi_set_gpu_fabric_ppod_config() not supported on this system"
                    << "\n";
        }
      } else {
        ASSERT_EQ(status_code, AMDSMI_STATUS_SUCCESS);
      }
    }

    {
      auto vpod = make_minimal_vpod_config();
      auto status_code = amdsmi_set_gpu_fabric_vpod_config(dev, &vpod);
      if (status_code == AMDSMI_STATUS_NOT_SUPPORTED) {
        IF_VERB(STANDARD) {
          std::cout << "\t**amdsmi_set_gpu_fabric_vpod_config() not supported on this system"
                    << "\n";
        }
      } else {
        ASSERT_EQ(status_code, AMDSMI_STATUS_SUCCESS);
      }
    }

    {
      auto station = make_minimal_station_config();
      auto status_code = amdsmi_set_gpu_fabric_station_config(dev, &station);
      if (status_code == AMDSMI_STATUS_NOT_SUPPORTED) {
        IF_VERB(STANDARD) {
          std::cout << "\t**amdsmi_set_gpu_fabric_station_config() not supported on this system"
                    << "\n";
        }
      } else {
        ASSERT_EQ(status_code, AMDSMI_STATUS_SUCCESS);
      }
    }
  }

  /**
   *    Round-trip: (set+commit) -> read back via amdsmi_get_gpu_fabric_info()
   *      - The per-plane getters were removed; amdsmi_get_gpu_fabric_info() is the sole reader
   *      - A baseline read establishes whether fabric is present (NOT_SUPPORTED skips the plane)
   *      - We then write a known value with commit=true and read it back from the info struct
   *      - Absent/unconfigured fields read back at their sentinel (uints at UINT*_MAX), so the
   *        baseline value is only restored when it was not a sentinel
   */
  for (auto dv_ind = uint32_t(0); dv_ind < num_monitor_devs(); ++dv_ind) {
    auto dev = processor_handles_[dv_ind];
    PrintDeviceHeader(dev);

    auto baseline_info = amdsmi_fabric_info_t{};
    auto baseline_status = amdsmi_get_gpu_fabric_info(dev, &baseline_info);
    if (baseline_status == AMDSMI_STATUS_NOT_SUPPORTED) {
      IF_VERB(STANDARD) {
        std::cout << "\t**Fabric round-trip not supported on this system" << "\n";
      }
      continue;
    }
    ASSERT_TRUE((baseline_status == AMDSMI_STATUS_SUCCESS) ||
                (baseline_status == AMDSMI_STATUS_NO_DATA));

    const auto& baseline_v1 = baseline_info.fabric_info.fabric_version.v1;

    /**
     *  Ppod: accelerator_id round-trip
     */
    {
      const auto baseline_accel_id = baseline_v1.ppod.accelerator_id;

      auto wr = make_minimal_ppod_config();
      wr.mask = AMDSMI_FABRIC_PPOD_FIELD_ACCEL_ID;
      wr.commit = true;
      wr.data.accelerator_id = 7;
      auto set_status = amdsmi_set_gpu_fabric_ppod_config(dev, &wr);
      if (set_status != AMDSMI_STATUS_NOT_SUPPORTED) {
        ASSERT_EQ(set_status, AMDSMI_STATUS_SUCCESS);

        auto post = amdsmi_fabric_info_t{};
        ASSERT_EQ(amdsmi_get_gpu_fabric_info(dev, &post), AMDSMI_STATUS_SUCCESS);
        ASSERT_EQ(post.fabric_info.fabric_version.v1.ppod.accelerator_id, 7u);

        if (baseline_accel_id != std::numeric_limits<uint32_t>::max()) {
          auto restore = make_minimal_ppod_config();
          restore.mask = AMDSMI_FABRIC_PPOD_FIELD_ACCEL_ID;
          restore.commit = true;
          restore.data.accelerator_id = baseline_accel_id;
          ASSERT_EQ(amdsmi_set_gpu_fabric_ppod_config(dev, &restore), AMDSMI_STATUS_SUCCESS);
        }
      }
    }

    /**
     *  Vpod: vpod_id round-trip
     */
    {
      const auto baseline_vpod_id = baseline_v1.vpod.vpod_id;

      auto wr = make_minimal_vpod_config();
      wr.mask = AMDSMI_FABRIC_VPOD_FIELD_VPOD_ID;
      wr.commit = true;
      wr.data.vpod_id = 3;
      auto set_status = amdsmi_set_gpu_fabric_vpod_config(dev, &wr);
      if (set_status != AMDSMI_STATUS_NOT_SUPPORTED) {
        ASSERT_EQ(set_status, AMDSMI_STATUS_SUCCESS);

        auto post = amdsmi_fabric_info_t{};
        ASSERT_EQ(amdsmi_get_gpu_fabric_info(dev, &post), AMDSMI_STATUS_SUCCESS);
        ASSERT_EQ(post.fabric_info.fabric_version.v1.vpod.vpod_id, 3u);

        if (baseline_vpod_id != std::numeric_limits<uint32_t>::max()) {
          auto restore = make_minimal_vpod_config();
          restore.mask = AMDSMI_FABRIC_VPOD_FIELD_VPOD_ID;
          restore.commit = true;
          restore.data.vpod_id = baseline_vpod_id;
          ASSERT_EQ(amdsmi_set_gpu_fabric_vpod_config(dev, &restore), AMDSMI_STATUS_SUCCESS);
        }
      }
    }

    /**
     *  Station: station_flags round-trip
     */
    {
      const auto baseline_station_flags = baseline_v1.station.station_flags;

      auto wr = make_minimal_station_config();
      wr.mask = AMDSMI_FABRIC_DF_FIELD_STATION_FLAGS;
      wr.commit = true;
      wr.data.station_flags = 1;
      auto set_status = amdsmi_set_gpu_fabric_station_config(dev, &wr);
      if (set_status != AMDSMI_STATUS_NOT_SUPPORTED) {
        ASSERT_EQ(set_status, AMDSMI_STATUS_SUCCESS);

        auto post = amdsmi_fabric_info_t{};
        ASSERT_EQ(amdsmi_get_gpu_fabric_info(dev, &post), AMDSMI_STATUS_SUCCESS);
        ASSERT_EQ(post.fabric_info.fabric_version.v1.station.station_flags, 1u);

        if (baseline_station_flags != std::numeric_limits<uint32_t>::max()) {
          auto restore = make_minimal_station_config();
          restore.mask = AMDSMI_FABRIC_DF_FIELD_STATION_FLAGS;
          restore.commit = true;
          restore.data.station_flags = baseline_station_flags;
          ASSERT_EQ(amdsmi_set_gpu_fabric_station_config(dev, &restore), AMDSMI_STATUS_SUCCESS);
        }
      }
    }
  }
}
