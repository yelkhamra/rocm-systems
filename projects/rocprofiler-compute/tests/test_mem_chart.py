# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Tests for memory chart renderers (gfx9, gfx11) and shared helpers."""

import pytest

from utils import mem_chart_common, mem_chart_gfx9, mem_chart_gfx11
from utils.mem_chart_common import strip_ansi
from utils.utils_analysis import format_bw_human_readable

# =============================================================================
# format_bw_human_readable (from utils_analysis, used by chart renderers)
# =============================================================================


class TestFormatBwHumanReadable:
    @pytest.mark.parametrize(
        "value, unit, prec, expected",
        [
            (1e12, "Bytes/s", 1, "1.0 TB/s"),
            (2.5e12, "Bytes/s", 1, "2.5 TB/s"),
            (1e9, "Bytes/s", 1, "1.0 GB/s"),
            (100e9, "Bytes/s", 1, "100.0 GB/s"),
            (1e6, "Bytes/s", 1, "1.0 MB/s"),
            (1e3, "Bytes/s", 1, "1.0 KB/s"),
            (500, "Bytes/s", 1, "500.0 B/s"),
            (0, "Bytes/s", 1, "0.0 B/s"),
            (100, "GB/s", 1, "100.0 GB/s"),
            (1500, "GB/s", 1, "1.5 TB/s"),
            (None, "Bytes/s", 1, "N/A"),
            ("invalid", "Bytes/s", 1, "N/A"),
        ],
    )
    def test_format(self, value, unit, prec, expected):
        assert format_bw_human_readable(value, unit, prec) == expected

    @pytest.mark.parametrize(
        "prec, expected",
        [(0, "123 GB/s"), (1, "123.5 GB/s"), (2, "123.46 GB/s")],
    )
    def test_precision(self, prec, expected):
        assert format_bw_human_readable(123.456789e9, "Bytes/s", prec) == expected


# =============================================================================
# mem_chart_common helpers
# =============================================================================


class TestFormatValue:
    @pytest.mark.parametrize(
        "value, unit, prec, expected",
        [
            (85.5, "%", 1, "85.5%"),
            (None, "%", 1, "N/A"),
            ("50.5", "%", 1, "50.5%"),
            ("invalid", "%", 1, "invalid"),
        ],
    )
    def test_format(self, value, unit, prec, expected):
        assert mem_chart_common.format_value(value, unit, prec) == expected

    def test_bytes_per_second_routes_to_human_readable(self):
        assert "GB/s" in mem_chart_common.format_value(100e9, "Bytes/s", 1)


class TestFormatSci:
    @pytest.mark.parametrize(
        "value, expected_contains",
        [(100, "100"), (999, "999"), (1_000_000, "e"), (None, "N/A"), (-500, "-500")],
    )
    def test_format(self, value, expected_contains):
        assert expected_contains in mem_chart_common.format_sci(value)


class TestBar:
    @pytest.mark.parametrize(
        "pct, filled, empty",
        [(100, 10, 0), (0, 0, 10), (50, 5, 5), (None, 0, 10), (150, 10, 0)],
    )
    def test_bar(self, pct, filled, empty):
        assert mem_chart_common.bar(pct, 10) == "█" * filled + "░" * empty


class TestSafeFloatSum:
    @pytest.mark.parametrize(
        "args, expected",
        [((1.5, None, 2.5), 4.0), ((None, None), None), (("10", 5), 15.0)],
    )
    def test_sum(self, args, expected):
        assert mem_chart_common.safe_float_sum(*args) == expected


class TestScaleOrNone:
    @pytest.mark.parametrize(
        "value, factor, expected",
        [(10, 128, 1280.0), (None, 128, None), ("5", 64, 320.0)],
    )
    def test_scale(self, value, factor, expected):
        assert mem_chart_common.scale_or_none(value, factor) == expected


class TestFmtEdge:
    @pytest.mark.parametrize(
        "label, value, check_in, check_not_in",
        [("Read", 1_500_000, "1.50e+06", None), ("Write", None, "Write", ":")],
    )
    def test_edge(self, label, value, check_in, check_not_in):
        result = mem_chart_common.fmt_edge(label, value)
        assert check_in in result
        if check_not_in is not None:
            assert check_not_in not in result


class TestBwColor:
    @pytest.mark.parametrize(
        "value, peak, default, expected",
        [
            (10, 100, "white", "dim green"),
            (50, 100, "white", "yellow"),
            (90, 100, "white", "red"),
            (None, 100, "white", "white"),
            (50, None, "white", "white"),
            (50, 0, "white", "white"),
        ],
    )
    def test_color(self, value, peak, default, expected):
        assert mem_chart_common.bw_color(value, peak, default) == expected


class TestFormatMemChartHeading:
    @pytest.mark.parametrize(
        "unit, panel_id, expected",
        [
            ("per_kernel", 300, "3. Memory Chart (Normalization: per_kernel)"),
            ("per_wave", 500, "5. Memory Chart (Normalization: per_wave)"),
        ],
    )
    def test_heading(self, unit, panel_id, expected):
        result = mem_chart_common.format_mem_chart_heading(unit, panel_id=panel_id)
        assert result == expected


class TestBuildLegend:
    def test_contains_read_write_atomic(self):
        legend = mem_chart_common.build_legend()
        assert "Read" in legend and "Write" in legend and "Atomic" in legend

    def test_stall_optional(self):
        assert "Stall" not in mem_chart_common.build_legend()
        assert "Stall" in mem_chart_common.build_legend(include_stall=True)


# =============================================================================
# gfx11 chart
# =============================================================================


_GFX11_TITLE = "3. Memory Chart (Normalization: per_kernel)"


class TestPlotMemChartGfx11:
    def test_full_chart(self):
        result = mem_chart_gfx11.plot_mem_chart(
            mem_chart_gfx11.get_sample_metrics(), chart_title=_GFX11_TITLE
        )
        clean = strip_ansi(result)
        assert len(result) > 100
        assert "3. Memory Chart" in clean
        assert "GPU" in clean and "System Memory" in clean

    @pytest.mark.parametrize("block", ["TCP", "GL1 Cache", "GL2 Cache", "GCEA", "DRAM"])
    def test_contains_arch_element(self, block):
        result = mem_chart_gfx11.plot_mem_chart(
            mem_chart_gfx11.get_sample_metrics(), chart_title=_GFX11_TITLE
        )
        assert block in result

    def test_sparse_metrics(self):
        result = mem_chart_gfx11.plot_mem_chart({}, chart_title=_GFX11_TITLE)
        assert isinstance(result, str) and len(result) > 0

    def test_normalize_drops_unknown_fills_missing(self):
        raw = {"GL0 Cache Hit Rate (TCP Cache)": 1.0, "noise": 99}
        norm = mem_chart_gfx11.normalize_mem_chart_metrics(raw)
        assert norm["GL0 Cache Hit Rate (TCP Cache)"] == 1.0
        assert "noise" not in norm
        assert norm["ICache Requests"] is None


class TestDefaultSampleMetricsGfx11:
    def test_keys_match_panel_keys(self):
        assert set(mem_chart_gfx11.DEFAULT_SAMPLE_METRICS) == set(
            mem_chart_gfx11.MEM_CHART_PANEL_METRIC_KEYS
        )

    @pytest.mark.parametrize("prefix", ["TCP", "LDS", "GL1", "GL2", "DRAM"])
    def test_hierarchy_level_present(self, prefix):
        assert any(prefix in k for k in mem_chart_gfx11.DEFAULT_SAMPLE_METRICS)


# =============================================================================
# gfx9 chart
# =============================================================================


class TestPlotMemChartGfx9:
    def test_full_chart(self):
        result = mem_chart_gfx9.plot_mem_chart(
            "per_kernel", mem_chart_gfx9.get_sample_metrics()
        )
        clean = strip_ansi(result)
        assert len(result) > 100
        assert "Legend" in clean

    @pytest.mark.parametrize(
        "block",
        ["Kernel", "VL1D", "LDS", "sL1D", "L1I", "L2", "Data Fabric", "MALL", "HBM"],
    )
    def test_contains_arch_element(self, block):
        result = mem_chart_gfx9.plot_mem_chart(
            "per_kernel", mem_chart_gfx9.get_sample_metrics()
        )
        assert block in strip_ansi(result)

    def test_sparse_metrics(self):
        result = mem_chart_gfx9.plot_mem_chart("per_kernel", {})
        assert isinstance(result, str) and len(result) > 0

    def test_chart_title_override(self):
        title = "Custom Title"
        result = mem_chart_gfx9.plot_mem_chart(
            "per_kernel", mem_chart_gfx9.get_sample_metrics(), chart_title=title
        )
        assert title in strip_ansi(result)


class TestGfx9GpuArch:
    @pytest.mark.parametrize(
        "arch, expect_xgmi",
        [("gfx950", True), ("gfx942", False), (None, False)],
    )
    def test_xgmi_pcie_gating(self, arch, expect_xgmi):
        result = mem_chart_gfx9.plot_mem_chart(
            "per_kernel", mem_chart_gfx9.get_sample_metrics(), gpu_arch=arch
        )
        clean = strip_ansi(result)
        assert ("XGMI" in clean) == expect_xgmi
        assert ("PCIe" in clean) == expect_xgmi


class TestComputePeakBw:
    def test_valid_specs(self):
        specs = {
            "max_sclk": "2100",
            "max_mclk": "1600",
            "cu_per_gpu": "304",
            "total_l2_chan": "128",
            "num_memory_channels": "128",
            "sqc_per_gpu": "38",
        }
        p = mem_chart_gfx9.compute_peak_bw(specs)
        assert all(
            v is not None and v > 0 for v in [p.hbm, p.l2, p.vl1d, p.lds, p.sl1d, p.l1i]
        )

    def test_empty_specs(self):
        p = mem_chart_gfx9.compute_peak_bw({})
        assert p.hbm is None and p.l2 is None


class TestNormalizeAndSampleGfx9:
    def test_normalize_preserves_known_drops_unknown(self):
        result = mem_chart_gfx9.normalize_mem_chart_metrics({
            "VL1 Hit": 92,
            "UNKNOWN": 42,
        })
        assert result["VL1 Hit"] == 92
        assert "UNKNOWN" not in result
        assert result["HBM Rd"] is None

    def test_sample_keys_match_panel_keys(self):
        assert set(mem_chart_gfx9.DEFAULT_SAMPLE_METRICS.keys()) == set(
            mem_chart_gfx9.MEM_CHART_PANEL_METRIC_KEYS
        )

    @pytest.mark.parametrize("prefix", ["VL1", "sL1D", "IL1", "L2", "Fabric", "HBM"])
    def test_hierarchy_level_present(self, prefix):
        assert any(prefix in k for k in mem_chart_gfx9.DEFAULT_SAMPLE_METRICS)

    def test_returns_copy(self):
        a = mem_chart_gfx9.get_sample_metrics()
        b = mem_chart_gfx9.get_sample_metrics()
        a["VL1 Hit"] = -999
        assert b["VL1 Hit"] != -999
