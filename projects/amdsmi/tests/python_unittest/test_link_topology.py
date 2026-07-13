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

"""Unit tests for the unified ``amdsmi_get_link_topology`` binding.

These tests are hardware independent. They validate that the ctypes
``amdsmi_link_topology_t`` structure matches the C ABI (64 bytes, matching the
host struct), that the high-level ``amdsmi_get_link_topology`` symbol is
exported, and that argument validation rejects non-handle inputs before any
library call is made. The whole suite is skipped when the ``amdsmi`` package
cannot be imported (for example when the shared library is not built), so it is
safe to collect in any environment.
"""

import ctypes
import unittest

try:
    import amdsmi
    from amdsmi import amdsmi_interface
    from amdsmi import amdsmi_wrapper

    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - depends on build/runtime env
    amdsmi = None
    amdsmi_interface = None
    amdsmi_wrapper = None
    _IMPORT_ERROR = exc


@unittest.skipIf(amdsmi is None, f"amdsmi package unavailable: {_IMPORT_ERROR}")
class TestLinkTopology(unittest.TestCase):
    def test_struct_size_matches_host_abi(self):
        # The unified struct must be 64 bytes so the baremetal and host
        # interfaces are binary compatible.
        self.assertEqual(ctypes.sizeof(amdsmi_wrapper.amdsmi_link_topology_t), 64)

    def test_struct_fields(self):
        struct_type = amdsmi_wrapper.amdsmi_link_topology_t
        field_names = [name for name, *_ in struct_type._fields_]
        for expected in (
            "weight",
            "link_status",
            "link_type",
            "num_hops",
            "fb_sharing",
            "reserved",
        ):
            self.assertIn(expected, field_names)

        # weight leads the struct and the reserved area holds 10 uint32 words.
        self.assertEqual(struct_type.weight.offset, 0)
        instance = struct_type()
        self.assertEqual(len(instance.reserved), 10)

    def test_symbol_is_exported(self):
        self.assertTrue(hasattr(amdsmi, "amdsmi_get_link_topology"))

    def test_rejects_non_handle_arguments(self):
        # Argument validation happens before any library call, so this raises
        # without needing a GPU present.
        with self.assertRaises(amdsmi_interface.AmdSmiParameterException):
            amdsmi_interface.amdsmi_get_link_topology("not-a-handle", "also-bad")


if __name__ == "__main__":
    unittest.main()
