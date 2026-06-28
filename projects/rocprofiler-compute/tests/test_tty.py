# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for src/utils/tty.py."""

import argparse

import pandas as pd

from utils.tty import format_node_args, format_table_output, print_operator_node
from utils.utils_analysis import CallTreeNode, KernelStats


def make_args() -> argparse.Namespace:
    """Minimal args for the plain-table render path."""
    return argparse.Namespace(decimal=2, view=None, normal_unit="per_wave")


def test_format_table_output_suppresses_empty_column() -> None:
    """A non-PC-sampling table with an all-'N/A' column is suppressed."""
    df = pd.DataFrame({"Metric": ["a", "b"], "Value": ["N/A", "N/A"]})
    content = format_table_output(
        make_args(),
        {"id": 1101, "title": "Some Table"},
        df,
        "metric_table",
        runs={"only": object()},
    )
    assert content == ""


def test_format_table_output_keeps_pc_sampling_table_21_1() -> None:
    """PC sampling table 21.1 is shown even with an all-'N/A' source column."""
    df = pd.DataFrame({
        "source_line": ["N/A", "N/A"],
        "instruction": ["v_mov", "v_add"],
        "count": [3, 1],
    })
    content = format_table_output(
        make_args(),
        {"id": 2101, "title": "PC Sampling"},
        df,
        "pc_sampling_table",
        runs={"only": object()},
    )
    assert content != ""
    assert "v_mov" in content


def test_format_node_args_present() -> None:
    """A node with recorded args produces an ' args=(...)' segment."""
    node = CallTreeNode(name="aten::mm", args="(self=float32[2x2])")
    assert format_node_args(node) == " args=(self=float32[2x2])"


def test_format_node_args_absent() -> None:
    """A node without args (or only empty parens) renders no segment."""
    assert format_node_args(CallTreeNode(name="aten::mm")) == ""
    assert format_node_args(CallTreeNode(name="aten::mm", args="()")) == ""


def test_print_operator_node_shows_args_by_default(capsys) -> None:
    """Operator args are shown in the call-tree display without any opt-in."""
    node = CallTreeNode(name="aten::mm", args="(self=float32[2x2])")
    node.kernels["kernel_gemm"] = KernelStats(launches=1, total_duration_ns=1000.0)

    print_operator_node(node)

    out = capsys.readouterr().out
    assert "aten::mm" in out
    assert "args=(self=float32[2x2])" in out
