#!/usr/bin/env python3
#
# Copyright (C) Advanced Micro Devices. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""GPU-independent unit tests for the amd-smi CLI set_value snap helper.

These tests exercise SetValueCommands._snap_clk_limit_to_dpm in-process with a
stubbed amdsmi_get_clk_freq, so they need no GPU and run anywhere the CLI module
imports.
"""

import os
import sys
import unittest
from unittest import mock

# Import the amd-smi CLI set_value module in-process so its GPU-independent
# helper (_snap_clk_limit_to_dpm) can be unit tested with a stubbed
# amdsmi_get_clk_freq. common.py owns path resolution, sys.path setup, and
# amdsmi loading; set_value.py lives in ../../amdsmi_cli/subcommands relative to
# this test file. The whole import is guarded so a missing amdsmi package or a
# CLI layout change only skips these tests instead of breaking collection.
try:
    import common  # noqa: F401  (side effect: sys.path setup + amdsmi load)

    _AMDSMI_CLI_DIR = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "amdsmi_cli")
    )
    for _cli_path in (_AMDSMI_CLI_DIR, os.path.join(_AMDSMI_CLI_DIR, "subcommands")):
        if os.path.isdir(_cli_path) and _cli_path not in sys.path:
            sys.path.append(_cli_path)

    import set_value as amdsmi_set_value

    _SET_VALUE_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - only hit on missing deps / layout changes
    amdsmi_set_value = None
    _SET_VALUE_IMPORT_ERROR = exc


@unittest.skipUnless(
    amdsmi_set_value is not None,
    f"amd-smi CLI set_value module not importable: {_SET_VALUE_IMPORT_ERROR}",
)
class TestSnapClkLimitToDpm(unittest.TestCase):
    """GPU-independent unit tests for SetValueCommands._snap_clk_limit_to_dpm.

    The helper snaps a requested ``max`` clk-limit DOWN to the largest reachable
    DPM level so the enforced cap never exceeds the request. amdsmi_get_clk_freq
    is stubbed, so these tests need no GPU and run anywhere the CLI module
    imports. They guard the snap-down mapping (the regression the existing
    hardware CLI test cannot reach, since it only feeds back an already-aligned
    DPM value) plus the N/A / empty / library-error fallbacks.
    """

    # Example fclk DPM levels (Hz). Deliberately unsorted and seeded with a 0 Hz
    # entry to prove the helper sorts the levels and filters out f <= 0.
    FCLK_DPM_HZ = [1600_000_000, 0, 1200_000_000, 2000_000_000, 1900_000_000]

    @staticmethod
    def _snap(requested_mhz, frequency):
        with mock.patch.object(
            amdsmi_set_value.amdsmi_interface,
            "amdsmi_get_clk_freq",
            return_value={"frequency": list(frequency)},
        ):
            return amdsmi_set_value.SetValueCommands._snap_clk_limit_to_dpm(
                None, None, requested_mhz
            )

    def test_exact_dpm_levels_unchanged(self):
        for level in (1200, 1600, 1900, 2000):
            with self.subTest(level=level):
                self.assertEqual(self._snap(level, self.FCLK_DPM_HZ), level)

    def test_between_levels_snaps_down(self):
        # Every off-level request must resolve to the largest DPM <= requested.
        expected = {
            1300: 1200,
            1500: 1200,
            1599: 1200,
            1700: 1600,
            1850: 1600,
            1899: 1600,
            1950: 1900,
            1999: 1900,
        }
        for requested, want in expected.items():
            with self.subTest(requested=requested):
                self.assertEqual(self._snap(requested, self.FCLK_DPM_HZ), want)

    def test_above_highest_caps_to_highest_dpm(self):
        self.assertEqual(self._snap(2500, self.FCLK_DPM_HZ), 2000)

    def test_below_lowest_returns_none(self):
        # The caller rejects val < min_clk first; the helper must not invent a
        # level below the lowest reachable DPM.
        self.assertIsNone(self._snap(1000, self.FCLK_DPM_HZ))

    def test_empty_frequency_list_returns_none(self):
        self.assertIsNone(self._snap(1600, []))

    def test_missing_frequency_key_returns_none(self):
        with mock.patch.object(
            amdsmi_set_value.amdsmi_interface, "amdsmi_get_clk_freq", return_value={}
        ):
            self.assertIsNone(
                amdsmi_set_value.SetValueCommands._snap_clk_limit_to_dpm(None, None, 1600)
            )

    def test_library_exception_returns_none(self):
        err = amdsmi_set_value.amdsmi_exception.AmdSmiLibraryException(
            amdsmi_set_value.amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_NOT_SUPPORTED
        )
        with mock.patch.object(
            amdsmi_set_value.amdsmi_interface, "amdsmi_get_clk_freq", side_effect=err
        ):
            self.assertIsNone(
                amdsmi_set_value.SetValueCommands._snap_clk_limit_to_dpm(None, None, 1600)
            )

    def test_sclk_two_level_snaps_down(self):
        # Two-level domain ({500, 2100} MHz): an unaligned request snaps down to 500.
        sclk_hz = [500_000_000, 2100_000_000]
        self.assertEqual(self._snap(1500, sclk_hz), 500)
        self.assertEqual(self._snap(2100, sclk_hz), 2100)

    def test_mclk_levels_snap_down(self):
        # Multi-level domain ({900, 1100, 1200, 1300} MHz).
        mclk_hz = [900_000_000, 1100_000_000, 1200_000_000, 1300_000_000]
        self.assertEqual(self._snap(1250, mclk_hz), 1200)
        self.assertEqual(self._snap(1099, mclk_hz), 900)


if __name__ == "__main__":
    unittest.main()
