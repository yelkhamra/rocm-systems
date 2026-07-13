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

"""Mock-based unit tests for the ``amd-smi set --clk-limit`` DPM snap helper.

These tests drive ``SetValueCommands._snap_clk_limit_to_dpm`` with the C library
fully stubbed, so they run without GPU hardware or the compiled ``amdsmi``
package. The helper snaps a requested ``max`` clk-limit DOWN to the largest
reachable DPM level so the enforced cap never exceeds the request. That
snap-down mapping (ROCM-25290) is the regression the hardware CLI ``set`` test
cannot reach -- it only feeds back an already-aligned DPM value -- and the empty
/ missing / library-error cases lock in the safe ``None`` fallback the caller
relies on.
"""

import importlib.util
import os
import sys
import types
import unittest

from common.common import amdsmi_path

# set_value.py lives in the amd-smi CLI, which exists in two layouts:
#   * source checkout: <repo>/projects/amdsmi/amdsmi_cli (sibling of tests/)
#   * installed:       <rocm>/libexec/amdsmi_cli (amdsmi_path is the sibling
#                      <rocm>/share/amd_smi)
# Prefer the in-tree source when running from a checkout so the test exercises
# the code under review; fall back to the installed CLI otherwise.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SOURCE_CLI_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", "..", "..", "..", "amdsmi_cli"))
_INSTALLED_CLI_DIR = os.path.join(
    os.path.dirname(os.path.dirname(amdsmi_path)), "libexec", "amdsmi_cli"
)


def _resolve_cli_dir():
    for cli_dir in (_SOURCE_CLI_DIR, _INSTALLED_CLI_DIR):
        if os.path.isfile(os.path.join(cli_dir, "subcommands", "set_value.py")):
            return cli_dir
    return None


_CLI_DIR = _resolve_cli_dir()
SET_VALUE_PATH = os.path.join(_CLI_DIR, "subcommands", "set_value.py") if _CLI_DIR else ""

# AMDSMI_STATUS_NOT_SUPPORTED sentinel used by the stubbed library-error path.
_STATUS_NOT_SUPPORTED = 8


class _FakeLibraryException(Exception):
    """Stand-in for ``amdsmi_exception.AmdSmiLibraryException``.

    Mirrors the two accessors the helper and its callers use, ``get_error_code``
    and ``get_error_info``.
    """

    def __init__(self, err_code=_STATUS_NOT_SUPPORTED, message="mock error"):
        super().__init__(message)
        self._err_code = err_code
        self._message = message

    def get_error_code(self):
        return self._err_code

    def get_error_info(self):
        return self._message


def _install_fake_amdsmi():
    """Register a stub ``amdsmi`` package so ``set_value.py`` imports cleanly.

    Returns the fake ``amdsmi_interface`` module so individual tests can swap in
    per-case ``amdsmi_get_clk_freq`` return values or side effects.
    """
    amdsmi_pkg = types.ModuleType("amdsmi")
    interface = types.ModuleType("amdsmi.amdsmi_interface")
    exception = types.ModuleType("amdsmi.amdsmi_exception")
    wrapper = types.ModuleType("amdsmi.amdsmi_wrapper")

    # Constants set_value.py binds at import time; the values are irrelevant here.
    interface.AMDSMI_MAX_PPT_LIMIT = 0
    interface.AMDSMI_MAX_UTIL = 100
    wrapper.AMDSMI_STATUS_NOT_SUPPORTED = _STATUS_NOT_SUPPORTED
    interface.amdsmi_wrapper = wrapper
    # Overwritten per-test; the empty default keeps the snap path inert.
    interface.amdsmi_get_clk_freq = lambda _handle, _clk_type: {"frequency": []}

    exception.AmdSmiLibraryException = _FakeLibraryException

    amdsmi_pkg.amdsmi_interface = interface
    amdsmi_pkg.amdsmi_exception = exception

    sys.modules["amdsmi"] = amdsmi_pkg
    sys.modules["amdsmi.amdsmi_interface"] = interface
    sys.modules["amdsmi.amdsmi_exception"] = exception
    sys.modules["amdsmi.amdsmi_wrapper"] = wrapper
    return interface


def _load_set_value_module():
    # set_value.py imports the sibling ``amdsmi_cli_exceptions`` module by bare
    # name, so the CLI dir must be importable before the module is executed.
    if _CLI_DIR and _CLI_DIR not in sys.path:
        sys.path.insert(0, _CLI_DIR)
    spec = importlib.util.spec_from_file_location("set_value_under_test", SET_VALUE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestSnapClkLimitToDpm(unittest.TestCase):
    """Unit tests for ``SetValueCommands._snap_clk_limit_to_dpm`` (ROCM-25290)."""

    # Example fclk DPM levels (Hz). Deliberately unsorted and seeded with a 0 Hz
    # entry to prove the helper sorts the levels and filters out f <= 0.
    FCLK_DPM_HZ = [1600_000_000, 0, 1200_000_000, 2000_000_000, 1900_000_000]

    _SAVED_MODULE_NAMES = (
        "amdsmi",
        "amdsmi.amdsmi_interface",
        "amdsmi.amdsmi_exception",
        "amdsmi.amdsmi_wrapper",
    )

    @classmethod
    def setUpClass(cls):
        if not SET_VALUE_PATH:
            raise unittest.SkipTest("amd-smi CLI set_value.py not found (source or installed)")
        # Snapshot any real amdsmi already loaded so the stub does not leak into
        # sibling suites sharing the interpreter; restored in tearDownClass.
        cls._saved_modules = {name: sys.modules.get(name) for name in cls._SAVED_MODULE_NAMES}
        cls.interface = _install_fake_amdsmi()
        cls.module = _load_set_value_module()

    @classmethod
    def tearDownClass(cls):
        for name, saved in cls._saved_modules.items():
            if saved is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = saved

    def _snap(self, requested_mhz, frequency):
        # Shared harness: point the stubbed C entry point at a fixed DPM table
        # (Hz) so every case drives the real helper with controlled input.
        freq_info = {"frequency": list(frequency)}
        self.interface.amdsmi_get_clk_freq = lambda _handle, _clk_type: freq_info
        return self.module.SetValueCommands._snap_clk_limit_to_dpm(None, None, requested_mhz)

    def test_exact_dpm_levels_unchanged(self):
        # Requests already on a DPM level pass through untouched: only snap when
        # we must, never move an aligned cap.
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
        # A request above the top DPM level clamps to the highest level -- we
        # never invent a cap the hardware cannot reach.
        self.assertEqual(self._snap(2500, self.FCLK_DPM_HZ), 2000)

    def test_below_lowest_returns_none(self):
        # The caller rejects val < min_clk first; the helper must not invent a
        # level below the lowest reachable DPM.
        self.assertIsNone(self._snap(1000, self.FCLK_DPM_HZ))

    def test_empty_frequency_list_returns_none(self):
        # No DPM levels reported -> no snap; the helper returns None so the
        # caller falls back to the raw request instead of crashing.
        self.assertIsNone(self._snap(1600, []))

    def test_missing_frequency_key_returns_none(self):
        # Malformed driver payload (no "frequency" key) also degrades to None.
        self.interface.amdsmi_get_clk_freq = lambda _handle, _clk_type: {}
        self.assertIsNone(self.module.SetValueCommands._snap_clk_limit_to_dpm(None, None, 1600))

    def test_library_exception_returns_none(self):
        # If the library call itself raises, the helper swallows it and returns
        # None so a set never fails just because the DPM query did.
        def _raise(_handle, _clk_type):
            raise _FakeLibraryException(_STATUS_NOT_SUPPORTED)

        self.interface.amdsmi_get_clk_freq = _raise
        self.assertIsNone(self.module.SetValueCommands._snap_clk_limit_to_dpm(None, None, 1600))

    def test_two_level_dpm_snaps_down(self):
        # Sparse two-level DPM table ({500, 2100} MHz): an unaligned request
        # snaps down to 500. sclk is excluded from snapping at the CLI call site
        # (it honors the exact requested max), so this only covers the generic
        # discrete-table path used by mclk/fclk.
        two_level_hz = [500_000_000, 2100_000_000]
        self.assertEqual(self._snap(1500, two_level_hz), 500)
        self.assertEqual(self._snap(2100, two_level_hz), 2100)

    def test_mclk_levels_snap_down(self):
        # Multi-level domain ({900, 1100, 1200, 1300} MHz).
        mclk_hz = [900_000_000, 1100_000_000, 1200_000_000, 1300_000_000]
        self.assertEqual(self._snap(1250, mclk_hz), 1200)
        self.assertEqual(self._snap(1099, mclk_hz), 900)


if __name__ == "__main__":
    unittest.main()
