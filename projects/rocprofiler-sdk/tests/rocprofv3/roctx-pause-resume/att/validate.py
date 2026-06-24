#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from pathlib import Path
import sys
import pytest


def test_att_data(json_data, output_dir):
    data = json_data["rocprofiler-sdk-tool"]
    strings = data["strings"]

    assert "att_filenames" in strings.keys(), "No att_filenames in output"
    assert len(strings["att_filenames"]) > 0, "Expected ATT data with marker tracing"

    att_files = list(Path(output_dir).glob("*.att"))
    assert len(att_files) > 0, f"No .att files found in {output_dir}"


def test_marker_data(json_data):
    data = json_data["rocprofiler-sdk-tool"]
    marker_records = data["buffer_records"].get("marker_api", [])
    assert len(marker_records) > 0, "Expected marker API records"


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
