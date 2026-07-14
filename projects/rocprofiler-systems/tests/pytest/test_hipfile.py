# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Tests for hipFile GPU-direct storage I/O telemetry.

Runs the hipfile-test workload under rocprof-sys with ROCPROFSYS_USE_HIPFILE=ON
and validates that the hipFile per-GPU I/O counters and process-global
file/buffer registration counters appear in the ROCPD database and Perfetto
trace, as implemented by the hipFile PMC collector.

The test is skipped automatically when rocprof-sys was not built with hipFile
support (ROCPROFSYS_USE_HIPFILE=OFF), because the hipfile-test binary is only
built in that configuration.
"""

from __future__ import annotations

import pytest
from pathlib import Path

from conftest import RocprofsysTest

pytestmark = [pytest.mark.gpu]


@pytest.fixture
def hipfile_env() -> dict[str, str]:
    """Environment enabling hipFile telemetry via process sampling."""
    return {
        "ROCPROFSYS_USE_HIPFILE": "ON",
        "ROCPROFSYS_USE_PROCESS_SAMPLING": "ON",
        # Sample frequently so the short-lived workload is captured many times.
        "ROCPROFSYS_PROCESS_SAMPLING_FREQ": "100",
        "ROCPROFSYS_ROCM_DOMAINS": "hip_runtime_api",
    }


@pytest.fixture
def hipfile_rules(validation_rules_dir: Path) -> list[Path]:
    """Validation rules for hipFile ROCPD output."""
    rules_dir = validation_rules_dir / "hipfile"
    return [
        rules_dir / "validation-rules.json",
        rules_dir / "hipfile-rules.json",
    ]


class TestHipFile(RocprofsysTest):
    """Tests for hipFile I/O telemetry (Perfetto and ROCPD)."""

    @pytest.mark.timeout(180)
    @pytest.mark.parametrize(
        "mode", [pytest.param("sys_run", marks=pytest.mark.rocpd("hipfile_env"))]
    )
    def test_telemetry(self, mode, hipfile_env, hipfile_rules):
        """Run hipfile-test and validate hipFile counters in ROCPD + Perfetto."""
        workload_file = self.test_output_dir / "hipfile-test.bin"
        result = self.run_test(
            mode,
            "hipfile-test",
            env=hipfile_env,
            run_args=[str(workload_file), "0", "5"],
        )
        self.assert_regex(result)

        if mode == "sys_run":
            self.assert_rocpd(result, rules_files=hipfile_rules)
            self.assert_perfetto(
                result,
                counter_names=[
                    "hipFile GPU0 File Registrations",
                    "hipFile GPU0 Buffer Registrations",
                ],
                subtest_name="Perfetto hipFile counter validation",
            )
