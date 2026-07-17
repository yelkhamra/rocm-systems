# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Assemble the interactive standalone roofline HTML document.

This module wraps the roofline plotly figure in a self-contained page
that adds two things to the plotly figure:
* a memory-peak dropdown, and
* click-to-isolate kernel filtering.

Both are driven by a single embedded JSON model.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import plotly.graph_objects as go

PLOT_DIV_ID = "roofline-plot"

_ASSETS_DIR = Path(__file__).parent / "assets"

_PLOT_CONFIG: dict[str, Any] = {
    "displaylogo": False,
    "responsive": True,
    "scrollZoom": True,
    "doubleClick": False,
    "modeBarButtonsToRemove": ["autoScale2d"],
}


@dataclass
class RooflineViewModel:
    """Client-facing description of the interactive roofline.

    Attributes:
        peaks: Ordered memory levels that have at least one point (e.g.
            ["L1", "L2", "HBM", "LDS"]).
        peak_colors: Map from memory level to its roof color, used to color an
            isolated kernel's dots by memory level
        default_peak: Memory region shown on load
        kernels: One entry per plotted kernel:
            {"name", "color", "traceIndex", "count", "totalTime",
            "pctRuntime", "limiter", "points": [{"peak", "ai", "perf",
            "status", "pctRoof"}]}. count/totalTime/pctRuntime
            are the dispatch count, aggregate time (in time_unit), and
            percent of total runtime; limiter is the specific binding roof;
            pctRoof is the percent of the roofline achieved at each point.
            Any of these may be None when the underlying data is missing.
        kernel_trace_indices: Indices into figure.data of the per-kernel
            scatter traces, in the same order as kernels.
        roofline_traces: Bandwidth-roof (memory-level) line traces; clicking one
            in the legend isolates it, each {"level", "traceIndex", "bandwidth"}.
        compute_traces: Horizontal compute-ceiling traces (VALU/matrix), each
            {"traceIndex", "peakPerf"}. These always stay shown, but their
            left endpoint tracks the steepest *visible* diagonal.
        roof_max_ai: Right-edge AI the roofs extrapolate to.
        div_id: Id of the Plotly graph div.
    """

    peaks: list[str] = field(default_factory=list)
    peak_colors: dict[str, str] = field(default_factory=dict)
    default_peak: Optional[str] = None
    kernels: list[dict[str, Any]] = field(default_factory=list)
    kernel_trace_indices: list[int] = field(default_factory=list)
    roofline_traces: list[dict[str, Any]] = field(default_factory=list)
    compute_traces: list[dict[str, Any]] = field(default_factory=list)
    roof_max_ai: float = 0.0
    div_id: str = PLOT_DIV_ID

    def to_json(self) -> str:
        """Serialize the model for embedding in a <script> tag.

        </ is escaped so a kernel name can never prematurely close the
        surrounding script element.
        """
        payload = {
            "divId": self.div_id,
            "peaks": self.peaks,
            "peakColors": self.peak_colors,
            "defaultPeak": self.default_peak,
            "kernels": self.kernels,
            "kernelTraceIndices": self.kernel_trace_indices,
            "rooflineTraces": self.roofline_traces,
            "computeTraces": self.compute_traces,
            "roofMaxAi": self.roof_max_ai,
        }
        return json.dumps(payload, allow_nan=True).replace("</", "<\\/")


def _read_asset(name: str) -> str:
    """Read a bundled asset (CSS/JS) that will be inlined into the document."""
    return (_ASSETS_DIR / name).read_text(encoding="utf-8")


def _escape_html(text: str) -> str:
    return (
        text
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


_PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>__TITLE__</title>
<style>
__CSS__
</style>
</head>
<body>
<div class="roofline-app">
  <div class="roofline-toolbar">
    <label class="roofline-control" for="roofline-peak-select">Memory peak
      <select id="roofline-peak-select"
              aria-label="Memory peak for kernel points"></select>
    </label>
    <label class="roofline-control roofline-toggle"
           title="Automatically fit the view (recenter and zoom) to the visible points and the roofs they are measured against whenever you change the memory peak or kernel selection. Turn off to keep your own pan/zoom.">
      <input type="checkbox" id="roofline-auto-zoom">
      Auto-fit
    </label>
  </div>
  <div class="roofline-body">
    <div class="roofline-plot-col">
__PLOT_FRAGMENT__
    </div>
    <div class="roofline-panel-wrap">
    <aside class="roofline-panel">
      <div class="roofline-panel-title">
        <span class="roofline-panel-title-label">Kernels
          <span id="roofline-kernel-count" class="roofline-kernel-count"></span>
        </span>
        <button type="button" id="roofline-show-all"
                class="roofline-btn roofline-btn-sm">Show all kernels</button>
      </div>
      <p class="roofline-panel-help">Click a row to show only that kernel; click
        again to show all. Ctrl+click (&#8984;+click on Mac) to add or remove
        kernels.</p>
      <ul id="roofline-kernel-list" class="roofline-kernel-list"></ul>
    </aside>
    </div>
  </div>
</div>
<script id="roofline-model" type="application/json">__MODEL_JSON__</script>
<script>
__JS__
</script>
</body>
</html>
"""


def build_interactive_document(
    figure: go.Figure,
    view_model: RooflineViewModel,
    title: str = "Empirical Roofline Analysis",
) -> str:
    """Build a fully self-contained interactive roofline HTML document."""
    fragment = figure.to_html(
        full_html=False,
        include_plotlyjs=True,
        div_id=view_model.div_id,
        config=_PLOT_CONFIG,
    )

    substitutions = {
        "TITLE": _escape_html(title),
        "CSS": _read_asset("roofline_plot.css"),
        "PLOT_FRAGMENT": fragment,
        "MODEL_JSON": view_model.to_json(),
        "JS": _read_asset("roofline_plot.js"),
    }
    return re.sub(
        r"__(TITLE|CSS|PLOT_FRAGMENT|MODEL_JSON|JS)__",
        lambda match: substitutions[match.group(1)],
        _PAGE_TEMPLATE,
    )
