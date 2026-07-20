# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit coverage for the interactive roofline HTML layer.

Exercises the color/axis helpers, the per-kernel trace builder, the view-model
serialization, and the standalone-document assembler. None of these tests touch
the analyze CLI.
"""

import argparse
import json

import plotly.graph_objects as go

from roofline.roofline_html import (
    PLOT_DIV_ID,
    RooflineViewModel,
    build_interactive_document,
)
from roofline.roofline_main import (
    _KERNEL_PALETTE,
    Roofline,
    build_kernel_colors,
    get_color,
    roofline_axis_bounds,
)


class MockMspec:
    def __init__(self, gpu_model: str, gpu_series: str, gpu_arch: str) -> None:
        self.gpu_model = gpu_model
        self.gpu_series = gpu_series
        self.gpu_arch = gpu_arch


def make_roofline() -> Roofline:
    run_parameters: dict[str, object] = {
        "workload_dir": "",
        "device_id": 0,
        "sort_type": "kernels",
        "mem_level": "ALL",
        "roofline_data_type": ["FP32"],
    }
    mspec = MockMspec("MI200", "mi200", "gfx90a")
    return Roofline(argparse.Namespace(), mspec, run_parameters)


# A ceiling_data with enough structure for _determine_kernel_bound_status.
CEILING = {
    "hbm": [[0.01, 1.0], [1.0, 1500.0], 1500.0],
    "l2": [[0.01, 1.0], [1.0, 3000.0], 3000.0],
    "valu": [[0.01, 1000.0], [9000.0, 9000.0], 9000.0],
    "matrix_ops": [[0.01, 1000.0], [90000.0, 90000.0], 90000.0],
}


# =============================================================================
# Color helper
# =============================================================================


def test_get_color_maps_each_level_to_a_distinct_html_color() -> None:
    colors = {get_color(level) for level in ["l0", "l1", "l2", "hbm", "lds"]}
    assert len(colors) == 5, "each memory level must have its own color"


def test_get_color_accepts_ai_prefixed_keys() -> None:
    assert get_color("ai_hbm") == get_color("hbm")


def test_get_color_rejects_unknown_category() -> None:
    try:
        get_color("does-not-exist")
    except RuntimeError:
        return
    raise AssertionError("get_color must reject an unknown category")


# =============================================================================
# Kernel colors
# =============================================================================


def test_build_kernel_colors_unique_within_palette() -> None:
    colors = build_kernel_colors(10)
    assert len(colors) == 10
    assert len(set(colors)) == 10, "colors must be unique up to the palette size"


def test_build_kernel_colors_cycles_past_palette() -> None:
    size = len(_KERNEL_PALETTE)
    colors = build_kernel_colors(size + 3)
    assert len(colors) == size + 3
    assert colors[size] == colors[0], "colors cycle once the palette is exhausted"


def test_build_kernel_colors_empty() -> None:
    assert build_kernel_colors(0) == []


# =============================================================================
# Axis bounds
# =============================================================================


def test_roofline_axis_bounds_spans_roofs_and_points() -> None:
    ai_data = {"ai_hbm": [[0.5], [100.0]]}
    x_lo, x_hi, y_lo, y_hi = roofline_axis_bounds(CEILING, ai_data, ["HBM", "L2"])
    # Bounds are padded outward, so they strictly enclose the data extremes.
    assert x_lo < 0.5
    assert x_hi > 1.0
    assert y_lo < 100.0
    assert y_hi > 9000.0


def test_roofline_axis_bounds_defaults_when_empty() -> None:
    assert roofline_axis_bounds({}, {}, []) == (0.01, 1000.0, 1.0, 100000.0)


# =============================================================================
# Per-kernel trace builder (circles-only; memory level lives in the model)
# =============================================================================


def test_build_kernel_traces_one_marker_trace_per_kernel() -> None:
    roofline = make_roofline()
    # kernel "kA" has an L2 and an HBM point; "kB" only an L2 point (its HBM
    # entry is zeroed out and must be dropped).
    roofline._Roofline__ai_data = {
        "ai_l2": [[2.0, 5.0], [200.0, 400.0]],
        "ai_hbm": [[1.0, 0.0], [200.0, 0.0]],
        "kernelNames": ["kA", "kB"],
    }
    colors = build_kernel_colors(2)

    traces, model = roofline._build_kernel_traces(
        kernel_names=["kA", "kB"],
        kernel_colors=colors,
        sanitized_cache_hierarchy=["HBM", "L2"],
        ceiling_data=CEILING,
        ops_flops="FLOP",
    )

    assert len(traces) == 2
    assert [t.name for t in traces] == ["kA", "kB"]
    # One color per kernel; points are drawn as uniform circles (no per-peak
    # marker-shape table any more).
    assert traces[0].mode == "markers"
    assert traces[0].marker.color == colors[0]
    assert traces[1].marker.color == colors[1]
    assert traces[0].marker.symbol is None, "points are circles, not per-peak shapes"
    assert all(t.showlegend is False for t in traces)

    # The memory level of each point is carried in the view model, not the trace.
    assert [p["peak"] for p in model[0]["points"]] == ["L2", "HBM"]
    assert [p["peak"] for p in model[1]["points"]] == ["L2"]
    valid = {"Memory", "Compute", "Unknown"}
    assert all(p["status"] in valid for k in model for p in k["points"])

    # customdata carries the fully-rendered per-point hover; the kernel name is
    # embedded in it.
    assert list(traces[0].customdata[0]) == [model[0]["points"][0]["hover"]]
    assert "kA" in model[0]["points"][0]["hover"]


def test_build_kernel_traces_skips_kernels_without_points() -> None:
    roofline = make_roofline()
    roofline._Roofline__ai_data = {
        "ai_hbm": [[0.0, 3.0], [0.0, 500.0]],
        "kernelNames": ["empty", "present"],
    }
    traces, model = roofline._build_kernel_traces(
        kernel_names=["empty", "present"],
        kernel_colors=build_kernel_colors(2),
        sanitized_cache_hierarchy=["HBM"],
        ceiling_data=CEILING,
        ops_flops="FLOP",
    )
    assert [t.name for t in traces] == ["present"]
    assert [k["name"] for k in model] == ["present"]


# =============================================================================
# View model serialization
# =============================================================================


def make_view_model() -> RooflineViewModel:
    return RooflineViewModel(
        peaks=["L2", "HBM"],
        peak_colors={"L2": "#009E73", "HBM": "#D55E00"},
        default_peak="HBM",
        kernels=[
            {
                "name": "kA",
                "color": "#123456",
                "traceIndex": 0,
                "points": [
                    {"peak": "HBM", "ai": 1.2, "perf": 300.0, "status": "Memory"}
                ],
            }
        ],
        kernel_trace_indices=[0],
    )


def test_view_model_to_json_round_trips() -> None:
    payload = json.loads(make_view_model().to_json())
    assert payload["divId"] == PLOT_DIV_ID
    assert payload["defaultPeak"] == "HBM"
    assert payload["peaks"] == ["L2", "HBM"]
    assert payload["peakColors"] == {"L2": "#009E73", "HBM": "#D55E00"}
    assert payload["kernelTraceIndices"] == [0]
    assert payload["kernels"][0]["name"] == "kA"
    # Roof/ceiling filtering and client-side overlay sampling are driven by
    # these fields.
    for key in (
        "rooflineTraces",
        "computeTraces",
        "computeOverlayTraces",
        "ceilingDenseHi",
        "roofSamples",
    ):
        assert key in payload


def test_view_model_to_json_escapes_script_close() -> None:
    model = RooflineViewModel(kernels=[{"name": "evil</script>", "points": []}])
    serialized = model.to_json()
    assert "</script>" not in serialized, "must not allow a script element to close"
    # Still valid JSON that decodes back to the original kernel name.
    assert json.loads(serialized)["kernels"][0]["name"] == "evil</script>"


def test_empty_view_model_is_serializable() -> None:
    payload = json.loads(RooflineViewModel().to_json())
    assert payload["kernels"] == []
    assert payload["defaultPeak"] is None


# =============================================================================
# Document assembler
# =============================================================================


def test_build_interactive_document_includes_controls_and_model() -> None:
    fig = go.Figure()
    # A roof line whose legend name must survive into the document text.
    fig.add_trace(go.Scatter(x=[0.01, 1.0], y=[1.0, 1500.0], name="HBM"))
    fig.add_trace(go.Scatter(x=[0.01, 1000.0], y=[9000.0, 9000.0], name="Peak VALU"))
    fig.add_trace(go.Scatter(x=[1.2], y=[300.0], name="kA", mode="markers"))

    document = build_interactive_document(fig, make_view_model(), title="Doc")

    for marker in [
        "roofline-peak-select",
        "roofline-show-all",
        'id="roofline-model"',
        "roofline-kernel-list",
        "roofline-roof-list",
        PLOT_DIV_ID,
        "Plotly.newPlot",
    ]:
        assert marker in document, f"document missing {marker!r}"


def test_build_interactive_document_is_self_contained() -> None:
    """plotly.js is inlined (offline), not pulled from a CDN script tag."""
    document = build_interactive_document(go.Figure(), RooflineViewModel())
    assert '<script src="https://cdn.plot.ly' not in document
    assert "<!DOCTYPE html>" in document
    # The inlined library makes the document large; a CDN reference would not.
    assert len(document) > 1_000_000
