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

#include "fabric_read.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>

#include "amd_smi/amdsmi.h"
#include "amd_smi/impl/amd_smi_gpu_device.h"
#include "amd_smi/impl/amd_smi_utils.h"

// Category mask covering all telemetry categories
static constexpr uint32_t kAllCategories =
    AMDSMI_FABRIC_TELEMETRY_CATEGORY_MASK_UALOE | AMDSMI_FABRIC_TELEMETRY_CATEGORY_MASK_SWITCH |
    AMDSMI_FABRIC_TELEMETRY_CATEGORY_MASK_CRYPTO | AMDSMI_FABRIC_TELEMETRY_CATEGORY_MASK_PFC |
    AMDSMI_FABRIC_TELEMETRY_CATEGORY_MASK_NETPORT |
    AMDSMI_FABRIC_TELEMETRY_CATEGORY_MASK_DERIVED_UALOE |
    AMDSMI_FABRIC_TELEMETRY_CATEGORY_MASK_DERIVED_NETPORT;

TestFabricRead::TestFabricRead() : TestBase() {
  set_title("AMDSMI Fabric (UALoE) Read Test");
  set_description(
      "This test verifies that fabric device info and telemetry data "
      "can be read properly via amdsmi_get_gpu_fabric_info(), "
      "amdsmi_alloc_fabric_telemetry(), amdsmi_get_fabric_telemetry_data(), "
      "amdsmi_free_fabric_telemetry(), and amdsmi_fabric_telem_id_to_string().");
}

TestFabricRead::~TestFabricRead(void) {}

void TestFabricRead::SetUp(void) {
  TestBase::SetUp();
  return;
}

void TestFabricRead::DisplayTestInfo(void) { TestBase::DisplayTestInfo(); }

void TestFabricRead::DisplayResults(void) const {
  TestBase::DisplayResults();
  return;
}

void TestFabricRead::Close() {
  // This will close handles opened within rsmitst utility calls and call
  // amdsmi_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}

void TestFabricRead::Run(void) {
  amdsmi_status_t err;

  TestBase::Run();
  if (setup_failed_) {
    IF_VERB(STANDARD) { std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl; }
    return;
  }

  for (uint32_t dv_ind = 0; dv_ind < num_monitor_devs(); ++dv_ind) {
    auto device = processor_handles_[dv_ind];
    PrintDeviceHeader(device);

    // Without a UALoE session the telemetry APIs must report NOT_SUPPORTED,
    // not NOT_INIT.
    {
      amdsmi_fabric_telemetry_t* probe = nullptr;
      err = amdsmi_alloc_fabric_telemetry(device, kAllCategories, &probe);
      ASSERT_NE(err, AMDSMI_STATUS_NOT_INIT);
      if (err == AMDSMI_STATUS_SUCCESS) {
        ASSERT_NE(probe, nullptr);
        err = amdsmi_get_fabric_telemetry_data(device, probe);
        ASSERT_NE(err, AMDSMI_STATUS_NOT_INIT);
        err = amdsmi_free_fabric_telemetry(device, probe);
        ASSERT_NE(err, AMDSMI_STATUS_NOT_INIT);
      } else {
        amdsmi_fabric_telemetry_t dummy = {};
        err = amdsmi_get_fabric_telemetry_data(device, &dummy);
        ASSERT_NE(err, AMDSMI_STATUS_NOT_INIT);
        err = amdsmi_free_fabric_telemetry(device, &dummy);
        ASSERT_NE(err, AMDSMI_STATUS_NOT_INIT);
      }
    }

    // Test amdsmi_get_gpu_fabric_info
    IF_VERB(STANDARD) { std::cout << "\t** Testing amdsmi_get_gpu_fabric_info()" << std::endl; }

    amdsmi_fabric_info_t fabric_info = {};
    err = amdsmi_get_gpu_fabric_info(device, &fabric_info);

    if (err == AMDSMI_STATUS_NOT_SUPPORTED) {
      IF_VERB(STANDARD) {
        std::cout << "\t**amdsmi_get_gpu_fabric_info() is not supported"
                     " on this system"
                  << std::endl;
      }
      continue;
    } else if (err != AMDSMI_STATUS_SUCCESS && err != AMDSMI_STATUS_NO_DATA) {
      CHK_ERR_ASRT(err)
    } else {
      IF_VERB(STANDARD) {
        if (err == AMDSMI_STATUS_NO_DATA) {
          std::cout << "\t**amdsmi_get_gpu_fabric_info() returned NO_DATA "
                       "(no UALoE sysfs content); BDF may still be valid"
                    << std::endl;
        }
        const auto& v1 = fabric_info.fabric_info.fabric_version.v1;

        // Render a byte array (ppod_id, lane_en_bitmap) as contiguous "0x.." hex.
        auto to_hex = [](const uint8_t* data, std::size_t count) {
          auto os = std::ostringstream();
          os << "0x";
          for (auto i = std::size_t{0}; i < count; ++i) {
            os << std::hex << std::setw(2) << std::setfill('0') << static_cast<unsigned>(data[i]);
          }
          return os.str();
        };

        // Render the valid entries of a u32 id array; UINT32_MAX marks an unset slot.
        auto to_id_list = [](const uint32_t* data, std::size_t count) {
          auto os = std::ostringstream();
          auto first = true;
          for (auto i = std::size_t{0}; i < count; ++i) {
            if ((data[i] == std::numeric_limits<uint32_t>::max())) {
              continue;
            }
            os << (first ? "" : ", ") << data[i];
            first = false;
          }
          return (first ? std::string("(none)") : os.str());
        };

        std::cout << "\t\tversion:                  " << fabric_info.fabric_info.version << "\n"
                  << "\t\tfabric_type:              " << v1.fabric_type << "\n"
                  << "\t\taccel_state:              " << v1.accel_state << "\n"
                  << "\t\taccelerator_id:           " << v1.ppod.accelerator_id << "\n"
                  << "\t\tbandwidth:                " << v1.ppod.bandwidth << " Mb/s\n"
                  << "\t\tlatency:                  " << v1.ppod.latency << " ns\n"
                  << "\t\tppod_id:                  "
                  << to_hex(v1.ppod.ppod_id, AMDSMI_MAX_UUID_ELEMENTS) << "\n"
                  << "\t\tppod_size:                " << v1.ppod.ppod_size << "\n"
                  << "\t\tlocal_accelerators:       "
                  << to_id_list(v1.ppod.local_accelerators, AMDSMI_FABRIC_MAX_LOCAL_GPUS) << "\n"
                  << "\t\tlocal_accelerator_count:  " << v1.ppod.local_accelerator_count << "\n"
                  << "\t\tvpod_id:                  " << v1.vpod.vpod_id << "\n"
                  << "\t\tvpod_size:                " << v1.vpod.vpod_size << "\n"
                  << "\t\tvpod_active_accelerators: "
                  << to_id_list(v1.vpod.vpod_active_accelerators,
                                AMDSMI_FABRIC_ACTIVE_ACCELERATORS_BITMAP_SIZE)
                  << "\n"
                  << "\t\taddr_mode:                " << v1.vpod.addr_mode << "\n"
                  << "\t\tstation_flags:            " << v1.station.station_flags << "\n"
                  << "\t\tnum_stations:             "
                  << static_cast<unsigned>(v1.station.num_stations) << "\n"
                  << "\t\tlane_en_bitmap:           "
                  << to_hex(v1.station.lane_en_bitmap, AMDSMI_FABRIC_MAX_BITMAP_SIZE) << "\n";
      }
    }

    /**
     * Cross-check commit propagation: read each config domain from the write
     * subtree (setup/config/df) and from the flat surface, then report per-field
     * whether the two agree. Diagnostic only -- never asserts, since a mismatch
     * may simply mean the driver does not mirror committed values into the flat
     * files. Both readers sentinel-init identically, so a plain value compare is
     * meaningful even for fields absent on a surface.
     */
    IF_VERB(STANDARD) {
      amd::smi::AMDSmiGPUDevice* dev = nullptr;
      if (((get_gpu_device_from_handle(device, &dev) == AMDSMI_STATUS_SUCCESS) &&
           (dev != nullptr))) {
        std::cout << "\t** Comparing flat vs subtree fabric config (commit propagation)"
                  << std::endl;

        auto show_scalar = [](const char* name, uint64_t s, uint64_t f) {
          std::cout << "\t\t" << name << ": " << ((s == f) ? "MATCH" : "MISMATCH")
                    << " (subtree=" << s << " flat=" << f << ")\n";
        };
        auto show_array = [](const char* name, const auto* s, const auto* f, std::size_t n) {
          std::cout << "\t\t" << name << ": " << (std::equal(s, (s + n), f) ? "MATCH" : "MISMATCH")
                    << "\n";
        };

        {
          auto sub = amdsmi_fabric_ppod_config_t{};
          sub.version = AMDSMI_FABRIC_PPOD_CONFIG_V1;
          sub.mask = (AMDSMI_FABRIC_PPOD_FIELD_ACCEL_ID | AMDSMI_FABRIC_PPOD_FIELD_PPOD_ID |
                      AMDSMI_FABRIC_PPOD_FIELD_PPOD_SIZE | AMDSMI_FABRIC_PPOD_FIELD_LOCAL_ACCELS |
                      AMDSMI_FABRIC_PPOD_FIELD_BANDWIDTH | AMDSMI_FABRIC_PPOD_FIELD_LATENCY);
          auto flat = sub;
          const auto s_st = dev->query_fabric_ppod_config(sub);
          const auto f_st = dev->query_fabric_ppod_config_flat(flat);
          std::cout << "\t\t[ppod] subtree_status=" << s_st << " flat_status=" << f_st << "\n";
          if (((s_st == AMDSMI_STATUS_SUCCESS) || (f_st == AMDSMI_STATUS_SUCCESS))) {
            show_scalar("ppod.accelerator_id", sub.data.accelerator_id, flat.data.accelerator_id);
            show_array("ppod.ppod_id", sub.data.ppod_id, flat.data.ppod_id,
                       AMDSMI_MAX_UUID_ELEMENTS);
            show_scalar("ppod.ppod_size", sub.data.ppod_size, flat.data.ppod_size);
            show_array("ppod.local_accelerators", sub.data.local_accelerators,
                       flat.data.local_accelerators, AMDSMI_FABRIC_MAX_LOCAL_GPUS);
            show_scalar("ppod.local_accelerator_count", sub.data.local_accelerator_count,
                        flat.data.local_accelerator_count);
            show_scalar("ppod.bandwidth", sub.data.bandwidth, flat.data.bandwidth);
            show_scalar("ppod.latency", sub.data.latency, flat.data.latency);
          }
        }

        {
          auto sub = amdsmi_fabric_vpod_config_t{};
          sub.version = AMDSMI_FABRIC_VPOD_CONFIG_V1;
          sub.mask =
              (AMDSMI_FABRIC_VPOD_FIELD_VPOD_ID | AMDSMI_FABRIC_VPOD_FIELD_VPOD_SIZE |
               AMDSMI_FABRIC_VPOD_FIELD_VPOD_ACTIVE_ACCELS | AMDSMI_FABRIC_VPOD_FIELD_ADDR_MODE);
          auto flat = sub;
          const auto s_st = dev->query_fabric_vpod_config(sub);
          const auto f_st = dev->query_fabric_vpod_config_flat(flat);
          std::cout << "\t\t[vpod] subtree_status=" << s_st << " flat_status=" << f_st << "\n";
          if (((s_st == AMDSMI_STATUS_SUCCESS) || (f_st == AMDSMI_STATUS_SUCCESS))) {
            show_scalar("vpod.vpod_id", sub.data.vpod_id, flat.data.vpod_id);
            show_scalar("vpod.vpod_size", sub.data.vpod_size, flat.data.vpod_size);
            show_array("vpod.vpod_active_accelerators", sub.data.vpod_active_accelerators,
                       flat.data.vpod_active_accelerators,
                       AMDSMI_FABRIC_ACTIVE_ACCELERATORS_BITMAP_SIZE);
            show_scalar("vpod.addr_mode", static_cast<uint64_t>(sub.data.addr_mode),
                        static_cast<uint64_t>(flat.data.addr_mode));
          }
        }

        {
          auto sub = amdsmi_fabric_station_config_t{};
          sub.version = AMDSMI_FABRIC_STATION_CONFIG_V1;
          sub.mask = (AMDSMI_FABRIC_DF_FIELD_STATION_FLAGS | AMDSMI_FABRIC_DF_FIELD_LANE_EN_BITMAP |
                      AMDSMI_FABRIC_DF_FIELD_NUM_STATIONS);
          auto flat = sub;
          const auto s_st = dev->query_fabric_station_config(sub);
          const auto f_st = dev->query_fabric_station_config_flat(flat);
          std::cout << "\t\t[station] subtree_status=" << s_st << " flat_status=" << f_st << "\n";
          if (((s_st == AMDSMI_STATUS_SUCCESS) || (f_st == AMDSMI_STATUS_SUCCESS))) {
            show_scalar("station.station_flags", sub.data.station_flags, flat.data.station_flags);
            show_scalar("station.num_stations", sub.data.num_stations, flat.data.num_stations);
            show_array("station.lane_en_bitmap", sub.data.lane_en_bitmap, flat.data.lane_en_bitmap,
                       AMDSMI_FABRIC_MAX_BITMAP_SIZE);
          }
        }
      }
    }

    // Null-pointer validation
    err = amdsmi_get_gpu_fabric_info(device, nullptr);
    ASSERT_EQ(err, AMDSMI_STATUS_INVAL);

    /**
     * Test amdsmi_alloc_fabric_telemetry(), amdsmi_get_fabric_telemetry_data(),
     * amdsmi_free_fabric_telemetry()
     */
    IF_VERB(STANDARD) {
      std::cout << "\t** Testing amdsmi_alloc_fabric_telemetry() / "
                   "amdsmi_get_fabric_telemetry_data() / "
                   "amdsmi_free_fabric_telemetry()"
                << std::endl;
    }

    amdsmi_fabric_telemetry_t* tel = nullptr;
    err = amdsmi_alloc_fabric_telemetry(device, kAllCategories, &tel);

    if (err == AMDSMI_STATUS_NOT_SUPPORTED) {
      IF_VERB(STANDARD) {
        std::cout << "\t**amdsmi_alloc_fabric_telemetry() is not supported"
                     " on this machine"
                  << std::endl;
      }
      continue;
    }
    CHK_ERR_ASRT(err)
    ASSERT_NE(tel, nullptr);

    err = amdsmi_get_fabric_telemetry_data(device, tel);
    if (err == AMDSMI_STATUS_NOT_SUPPORTED) {
      IF_VERB(STANDARD) {
        std::cout << "\t**amdsmi_get_fabric_telemetry_data() is not supported"
                     " on this machine"
                  << std::endl;
      }
      // Still free the allocated buffer before continuing
      amdsmi_free_fabric_telemetry(device, tel);
      continue;
    }
    CHK_ERR_ASRT(err)

    IF_VERB(STANDARD) {
      // Walk datasets and print a sample of telemetry items
      for (uint32_t cat = 0; cat < AMDSMI_FABRIC_TELEMETRY_CATEGORY_MAX; ++cat) {
        if (!tel->datasets[cat]) continue;
        const auto& ds = *tel->datasets[cat];
        std::cout << "\t\tcategory[" << cat << "]"
                  << "  instances=" << ds.instance_count << "  gen_count=" << ds.generation_count
                  << "\n";

        for (uint32_t inst = 0; inst < ds.instance_count; ++inst) {
          const auto& in = ds.instances[inst];
          std::cout << "\t\t  instance[" << inst << "] " << in.name.text
                    << "  logical_idx=" << in.logical_idx << "  items=" << in.item_count << "\n";

          // ── amdsmi_fabric_telem_id_to_string ─────────────────────────────
          for (uint32_t item = 0; item < in.item_count; ++item) {
            const auto& it = in.items[item];
            const char* name = amdsmi_fabric_telem_id_to_string(it.id);
            std::cout << "\t\t    [" << item << "] id=0x" << std::hex << it.id << std::dec
                      << "  name=" << (name ? name : "NULL") << "  value=" << it.value << "\n";

            // amdsmi_fabric_telem_id_to_string must return a non-null string
            // for any valid id obtained from the driver
            ASSERT_NE(name, nullptr);
          }
        }
      }
    }

    // amdsmi_free_fabric_telemetry
    err = amdsmi_free_fabric_telemetry(device, tel);
    CHK_ERR_ASRT(err)

    // Null-pointer validation for alloc/free
    err = amdsmi_alloc_fabric_telemetry(device, kAllCategories, nullptr);
    ASSERT_EQ(err, AMDSMI_STATUS_INVAL);

    err = amdsmi_free_fabric_telemetry(device, nullptr);
    ASSERT_EQ(err, AMDSMI_STATUS_INVAL);
  }
}
