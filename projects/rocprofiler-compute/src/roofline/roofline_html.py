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
    "modeBarButtonsToRemove": ["autoScale2d"],
}


@dataclass
class RooflineViewModel:
    """Client-facing description of the interactive roofline.

    Attributes:
        peaks: Ordered memory levels that have at least one point (e.g.
            ``["L1", "L2", "HBM", "LDS"]``).
        peak_symbols: Map from memory level to Plotly marker symbol; the shape
            identifies the peak (kernel identity is carried by color instead).
        default_peak: Level selected when the page loads. A single default peak
            means the plot opens with one dot per kernel rather than the full
            cloud of every level at once. ``None`` when there are no points.
        kernels: One entry per plotted kernel:
            ``{"name", "color", "traceIndex", "points": [{"peak", "ai",
            "perf", "status"}]}``.
        kernel_trace_indices: Indices into ``figure.data`` of the per-kernel
            scatter traces, in the same order as ``kernels``.
        div_id: Id of the Plotly graph div.
    """

    peaks: list[str] = field(default_factory=list)
    peak_symbols: dict[str, str] = field(default_factory=dict)
    default_peak: Optional[str] = None
    kernels: list[dict[str, Any]] = field(default_factory=list)
    kernel_trace_indices: list[int] = field(default_factory=list)
    div_id: str = PLOT_DIV_ID

    def to_json(self) -> str:
        """Serialize the model for embedding in a ``<script>`` tag.

        ``</`` is escaped so a kernel name can never prematurely close the
        surrounding script element.
        """
        payload = {
            "divId": self.div_id,
            "peaks": self.peaks,
            "peakSymbols": self.peak_symbols,
            "defaultPeak": self.default_peak,
            "kernels": self.kernels,
            "kernelTraceIndices": self.kernel_trace_indices,
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
    <button type="button" id="roofline-show-all" class="roofline-btn">
      Show all kernels
    </button>
  </div>
  <div class="roofline-body">
    <div class="roofline-plot-col">
__PLOT_FRAGMENT__
    </div>
    <aside class="roofline-panel">
      <div class="roofline-panel-title">
        Kernels
        <span id="roofline-kernel-count" class="roofline-kernel-count"></span>
      </div>
      <p class="roofline-panel-help">Click a row to show only that kernel; click
        again to show all. Ctrl+click (&#8984;+click on Mac) to add or remove
        kernels.</p>
      <ul id="roofline-kernel-list" class="roofline-kernel-list"></ul>
      <div id="roofline-kernel-details" class="roofline-kernel-details"></div>
    </aside>
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
    """Build a fully self-contained interactive roofline HTML document.

    The Plotly figure is rendered as an offline fragment (embedded ``plotly.js``,
    no CDN) and wrapped with the memory-peak dropdown, the kernel legend panel,
    the JSON model, and the inlined CSS/JS controller.
    """
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
