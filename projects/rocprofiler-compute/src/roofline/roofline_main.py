# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import argparse
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import plotext as plt
import plotly.colors as pcolors
import plotly.graph_objects as go
from dash import dcc, html

from roofline.roofline_html import (
    RooflineViewModel,
    build_interactive_document,
)
from utils.logger import (
    console_debug,
    console_error,
    console_log,
    console_warning,
    demarcate,
)
from utils.roofline_calc import (
    CACHE_LEVELS,
    SUPPORTED_DATATYPES,
    OpsSupport,
    construct_roof,
    sanitize_mem_level,
)
from utils.specs import MachineSpecs
from utils.utils_analysis import get_matrix_ops_type

# ROOFLINE_SUPPORTED lists the supported gfx architectures, check against this list
# before doing any roofline-related work
ROOFLINE_SUPPORTED = [
    "gfx90a",
    "gfx940",
    "gfx941",
    "gfx942",
    "gfx950",
    "gfx1150",
    "gfx1151",
    "gfx1152",
]

# A kernel's memory peaks are told apart by marker shape. There are only ever
# five memory levels, so these shapes never collide; kernel identity is carried
# by color and the legend panel instead (which is what lets us drop the old
# ten-symbol, reused-across-kernels scheme entirely).
_PEAK_SYMBOLS: dict[str, str] = {
    "L0": "circle",
    "L1": "square",
    "L2": "diamond",
    "HBM": "cross",
    "LDS": "triangle-up",
}

# One color per kernel from a high-contrast qualitative palette. Colors are
# unique up to the palette size and only cycle beyond it; the legend panel and
# hover always carry the name, so cycling stays unambiguous.
_KERNEL_PALETTE: list[str] = pcolors.qualitative.Dark24 + pcolors.qualitative.Light24

# Which memory peak the roofline opens on. "all" shows every level's dot per
# kernel; a specific level (e.g. "HBM") would instead open with one dot per
# kernel against that single roof.
_DEFAULT_PEAK = "all"

# Roofs represent mathematically infinite lines - a bandwidth diagonal is
# y = BW * AI (a line through the origin), and a compute ceiling is a horizontal
# line. We draw the segments so far past the data on both ends that no amount of
# realistic panning/zooming reaches an endpoint. A log axis can never reach 0,
# so "toward the origin" just means an arbitrarily small AI. (The "Autoscale"
# modebar button is removed so it can't fit the view to these endpoints; "Reset
# axes" still restores the sensible initial range.)
_ROOF_EXTRAP_MIN_AI = 1e-150
_ROOF_EXTRAP_MAX_AI = 1e150

# Per cache-level / compute-roof trace colors. Keyed by category, with one
# entry per rendering backend so both the HTML (Plotly hex) and CLI (plotext
# token) plots draw from a single source of truth.
_TRACE_COLORS: dict[str, dict[str, str]] = {
    "l0": {"html": "#F0E442", "cli": "brown+"},
    "l1": {"html": "#0072B2", "cli": "red+"},
    "l2": {"html": "#009E73", "cli": "green+"},
    "hbm": {"html": "#D55E00", "cli": "blue+"},
    "lds": {"html": "#E69F00", "cli": "orange+"},
    "valu": {"html": "#CC79A7", "cli": "white"},
    "matrix_ops": {"html": "#56B4E9", "cli": "magenta+"},
}


def get_color(category: str, backend: str = "html") -> str:
    key = category.removeprefix("ai_").lower()

    if key not in _TRACE_COLORS:
        raise RuntimeError(f"Invalid category passed to get_color(): {category}")
    if backend not in _TRACE_COLORS[key]:
        raise RuntimeError(f"Invalid backend passed to get_color(): {backend}")

    return _TRACE_COLORS[key][backend]


def to_int(value: Union[float, None]) -> Union[int, float]:
    if value is None:
        return np.nan
    return int(value)


def peak_symbol(level_name: str) -> str:
    """Return the Plotly marker shape used for a memory peak."""
    return _PEAK_SYMBOLS.get(level_name.upper(), "circle")


def build_kernel_colors(num_kernels: int) -> list[str]:
    """Assign one color per kernel, cycling only past the palette size."""
    if num_kernels <= 0:
        return []
    palette = _KERNEL_PALETTE
    return [palette[index % len(palette)] for index in range(num_kernels)]


def roofline_axis_bounds(
    ceiling_data: dict[str, Any],
    ai_data: dict[str, Any],
    sanitized_cache_hierarchy: list[str],
    pad: float = 1.6,
) -> tuple[float, float, float, float]:
    """Compute explicit ``(x_lo, x_hi, y_lo, y_hi)`` log-axis bounds.

    This is only the initial view: bounds span every roof ridge and every kernel
    point (shown or not) so nothing is clipped on open and toggling the
    peak/kernel filters never moves a point outside the view. ``pad`` widens the
    range slightly on each side. The roofs themselves are drawn far past these
    bounds (see ``_ROOF_EXTRAP_*``) so panning reveals them continuing.
    """
    roof_keys = [level.lower() for level in sanitized_cache_hierarchy]
    roof_keys += ["valu", "matrix_ops"]

    xs: list[float] = []
    ys: list[float] = []

    for key in roof_keys:
        line = ceiling_data.get(key)
        if line and len(line) >= 2 and line[0] and line[1]:
            xs.extend(v for v in line[0] if v is not None and v > 0)
            ys.extend(v for v in line[1] if v is not None and v > 0)

    for cache_level in CACHE_LEVELS:
        points = ai_data.get(cache_level)
        if not points:
            continue
        xs.extend(v for v in points[0] if v is not None and v > 0)
        ys.extend(v for v in points[1] if v is not None and v > 0)

    if not xs or not ys:
        return 0.01, 1000.0, 1.0, 100000.0

    return min(xs) / pad, max(xs) * pad, min(ys) / pad, max(ys) * pad


class Roofline:
    def __init__(
        self,
        args: argparse.Namespace,
        mspec: MachineSpecs,
        run_parameters: dict[str, Any],
    ) -> None:
        self.__args = args
        self.__mspec = mspec
        self.__run_parameters = run_parameters
        self.__ai_data: Optional[dict[str, Any]] = None
        self.__ceiling_data: Optional[dict[str, Any]] = None
        self.__figure = go.Figure()
        # Interactive view models keyed by "OP"/"FLOP" (the figure a kernel
        # scatter belongs to), populated while building the Plotly figures.
        self.__view_models: dict[str, RooflineViewModel] = {}

    def get_args(self) -> argparse.Namespace:
        return self.__args

    def roof_setup(self) -> None:
        workload_dir_val = self.__run_parameters.get("workload_dir")

        if not workload_dir_val:
            console_error(
                "Workload directory is not set. Cannot perform setup.", exit=False
            )
            return

        base_dir = str(workload_dir_val)

        base_path = Path(base_dir)

        if base_path.name == "workloads" and base_path.parent == Path.cwd():
            app_name = getattr(self.__args, "name", "default_app_name")
            gpu_model_name = getattr(self.__mspec, "gpu_model", "default_gpu_model")

            # Create the new path
            new_path = base_path / app_name / gpu_model_name

            # Update workload_dir with the new path, maintaining original data structure
            if isinstance(workload_dir_val, list):
                # Update the nested list structure
                if isinstance(workload_dir_val[0], (list, tuple)):
                    self.__run_parameters["workload_dir"][0][0] = str(new_path)
                else:
                    self.__run_parameters["workload_dir"][0] = str(new_path)
            else:
                # Update string value
                self.__run_parameters["workload_dir"] = str(new_path)

            final_dir = str(new_path)
        else:
            final_dir = base_dir

        # Create the directory
        Path(final_dir).mkdir(parents=True, exist_ok=True)

    def _determine_kernel_bound_status(
        self,
        ai_value: float,
        performance: float,
        cache_level: str,
        ceiling_data: dict[str, Any],
    ) -> str:
        """
        Calculate if a kernel point is memory-bound or compute-bound
        based on its own cache level's roofline
        """
        cache_key = cache_level.replace("ai_", "")

        # Get bw for this cache level
        if cache_key not in ceiling_data or not ceiling_data[cache_key]:
            return "Unknown"

        cache_data = ceiling_data[cache_key]
        if not isinstance(cache_data, (list, tuple)) or len(cache_data) < 3:
            return "Unknown"

        bandwidth = cache_data[2]

        # Get min peak performance
        min_peak = float("inf")
        if "valu" in ceiling_data and ceiling_data["valu"]:
            min_peak = min(min_peak, ceiling_data["valu"][2])
        if "matrix_ops" in ceiling_data and ceiling_data["matrix_ops"]:
            min_peak = min(min_peak, ceiling_data["matrix_ops"][2])

        if min_peak == float("inf"):
            return "Unknown"

        x_intersect = min_peak / bandwidth

        if ai_value < x_intersect:
            return "Memory Bound"
        else:
            return "Compute Bound"

    def _build_kernel_traces(
        self,
        kernel_names: list[str],
        kernel_colors: list[str],
        sanitized_cache_hierarchy: list[str],
        ceiling_data: dict[str, Any],
        ops_flops: str,
    ) -> tuple[list[go.Scatter], list[dict[str, Any]]]:
        """Build one marker trace per kernel plus the matching view-model data.

        Each kernel becomes a single scatter trace carrying all of its points
        across the memory peaks: ``marker.color`` identifies the kernel and
        ``marker.symbol`` identifies the peak. The returned model list mirrors
        the traces one-for-one and is the source of truth the client-side
        controller restyles from.
        """
        hovertemplate = (
            "<b>%{fullData.name}</b><br>"
            "%{customdata[0]} peak<br>"
            f"AI %{{x:.2f}} {ops_flops}s/Byte<br>"
            f"Perf %{{y:.2f}} G{ops_flops}/s<br>"
            "%{customdata[1]}<extra></extra>"
        )

        traces: list[go.Scatter] = []
        kernels_model: list[dict[str, Any]] = []

        for kernel_index, kernel_name in enumerate(kernel_names):
            xs: list[float] = []
            ys: list[float] = []
            symbols: list[str] = []
            customdata: list[list[str]] = []
            points: list[dict[str, Any]] = []

            for cache_level in CACHE_LEVELS:
                level_name = cache_level.removeprefix("ai_").upper()
                if level_name not in sanitized_cache_hierarchy:
                    continue
                level_points = self.__ai_data.get(cache_level)
                if not level_points or kernel_index >= len(level_points[0]):
                    continue
                ai_value = level_points[0][kernel_index]
                performance = level_points[1][kernel_index]
                if not (ai_value > 0 and performance > 0):
                    continue

                status = self._determine_kernel_bound_status(
                    ai_value=ai_value,
                    performance=performance,
                    cache_level=cache_level,
                    ceiling_data=ceiling_data,
                )
                xs.append(ai_value)
                ys.append(performance)
                symbols.append(peak_symbol(level_name))
                customdata.append([level_name, status])
                points.append({
                    "peak": level_name,
                    "ai": ai_value,
                    "perf": performance,
                    "status": status,
                })

            if not xs:
                continue

            color = (
                kernel_colors[kernel_index]
                if kernel_index < len(kernel_colors)
                else None
            )
            traces.append(
                go.Scatter(
                    x=xs,
                    y=ys,
                    name=kernel_name,
                    mode="markers",
                    legendgroup=kernel_name,
                    showlegend=False,
                    marker=dict(
                        color=color,
                        size=10,
                        symbol=symbols,
                        line=dict(width=0.5, color="black"),
                    ),
                    customdata=customdata,
                    hovertemplate=hovertemplate,
                )
            )
            kernels_model.append({
                "name": kernel_name,
                "color": color,
                "points": points,
            })

        return traces, kernels_model

    @demarcate
    def construct_plotly_figures(
        self, ai_data: dict[str, Any]
    ) -> tuple[Optional[go.Figure], Optional[go.Figure], str, str]:
        """
        Build raw Plotly figure objects from pre-computed AI data.

        Returns (ops_figure, flops_figure, ops_dt_list, flops_dt_list).
        No I/O or HTML wrapping.
        """
        self.roof_setup()
        self.__view_models = {}

        console_debug("roofline", f"Path: {self.__run_parameters.get('workload_dir')}")

        self.__ai_data = ai_data

        msg = "AI at each mem level:"
        for key, value in self.__ai_data.items():
            msg += f"\n\t{key} -> {value}"
        console_debug(msg)

        kernel_names_data = None
        if self.__ai_data and "kernelNames" in self.__ai_data:
            original_kernel_names = self.__ai_data.get("kernelNames", [])
            filtered_kernel_names = [
                name
                for name in original_kernel_names
                if name != "nan" and isinstance(name, str)
            ]
            if len(filtered_kernel_names) > 0:
                kernel_names_data = {
                    "kernel_names": filtered_kernel_names,
                    "num_kernels": len(filtered_kernel_names),
                }

        ops_figure = flops_figure = None
        ops_dt_list = flops_dt_list = ""

        for dt in self.__run_parameters.get("roofline_data_type", []):
            gpu_arch = getattr(self.__mspec, "gpu_arch", "unknown_arch")
            if (
                "SUPPORTED_DATATYPES" not in globals()
                or gpu_arch not in SUPPORTED_DATATYPES.keys()
                or str(dt) not in SUPPORTED_DATATYPES[gpu_arch].keys()
            ):
                console_error(
                    f"{dt} is not a supported datatype for roofline profiling on "
                    f"{getattr(self.__mspec, 'gpu_model', 'N/A')} (arch: {gpu_arch})- "
                    f"cannot construct HTML plot",
                    exit=False,
                )
                continue

            ops_flops = "Ops" if str(dt).startswith("I") else "Flops"

            if ops_flops == "Ops":
                if ops_figure:
                    ops_figure = self.generate_plot(
                        dtype=str(dt),
                        fig=ops_figure,
                    )
                else:
                    ops_figure = self.generate_plot(
                        dtype=str(dt),
                        kernel_names_data=kernel_names_data,
                    )
                ops_dt_list += "_" + str(dt)

            if ops_flops == "Flops":
                if flops_figure:
                    flops_figure = self.generate_plot(
                        dtype=str(dt),
                        fig=flops_figure,
                    )
                else:
                    flops_figure = self.generate_plot(
                        dtype=str(dt),
                        kernel_names_data=kernel_names_data,
                    )
                flops_dt_list += "_" + str(dt)

        return ops_figure, flops_figure, ops_dt_list, flops_dt_list

    def save_html_files(
        self,
        ops_figure: Optional[go.Figure],
        flops_figure: Optional[go.Figure],
        ops_dt_list: str,
        flops_dt_list: str,
    ) -> None:
        """Write Plotly figures to standalone HTML files on disk."""
        dev_id = str(self.__run_parameters["device_id"])
        kernel_list = ""
        if self.__run_parameters.get("kernel_filter", False):
            kernels = getattr(self.__args, "gpu_kernel", None)
            if kernels:
                flat = [
                    str(k)
                    for group in kernels
                    for k in (group if isinstance(group, list) else [group])
                ]
                for name in sorted(flat):
                    kernel_list += "_" + name

        workload_dir = self.__run_parameters["workload_dir"]
        prefix = f"{workload_dir}/empirRoof_gpu-{dev_id}"

        wrote = False
        if ops_figure:
            document = build_interactive_document(
                ops_figure,
                self.__view_models.get("OP", RooflineViewModel()),
                title="Empirical Roofline Analysis (Ops)",
            )
            path = f"{prefix}{ops_dt_list}{kernel_list}.html"
            Path(path).write_text(document, encoding="utf-8")
            wrote = True

        if flops_figure:
            document = build_interactive_document(
                flops_figure,
                self.__view_models.get("FLOP", RooflineViewModel()),
                title="Empirical Roofline Analysis (Flops)",
            )
            path = f"{prefix}{flops_dt_list}{kernel_list}.html"
            Path(path).write_text(document, encoding="utf-8")
            wrote = True

        if wrote:
            console_log("roofline", "Roofline HTML files saved.")

    @staticmethod
    def generate_html_section(
        ops_figure: Optional[go.Figure],
        flops_figure: Optional[go.Figure],
    ) -> Optional[html.Section]:
        """Wrap Plotly figures in Dash HTML components for WebUI embedding."""
        if ops_figure is None and flops_figure is None:
            return None

        ops_graph = (
            html.Div(
                className="float-child",
                children=[
                    html.H3(children="Empirical Roofline Analysis (Ops)"),
                    dcc.Graph(figure=ops_figure),
                ],
            )
            if ops_figure
            else None
        )

        flops_graph = (
            html.Div(
                className="float-child",
                children=[
                    html.H3(children="Empirical Roofline Analysis (Flops)"),
                    dcc.Graph(figure=flops_figure),
                ],
            )
            if flops_figure
            else None
        )

        return html.Section(
            id="roofline",
            children=[
                html.Div(
                    className="float-container",
                    children=[
                        ops_graph,
                        flops_graph,
                    ],
                )
            ],
        )

    @demarcate
    def generate_plot(
        self,
        dtype: str,
        fig: Optional[go.Figure] = None,
        kernel_names_data: Optional[dict] = None,
    ) -> go.Figure:
        """
        Create graph object from ai_data (coordinate points) and ceiling_data
        (peak FLOP and BW) data.
        """
        is_new_figure = fig is None
        has_kernel_names = kernel_names_data is not None and is_new_figure
        skipAI = not is_new_figure

        subplot_row = None
        total_figure_height = 600  # default height

        sanitized_cache_hierarchy = sanitize_mem_level(
            self.__run_parameters["mem_level"], self.__mspec.gpu_model
        )

        num_kernels = 0
        kernel_colors: list[str] = []

        if is_new_figure:
            if has_kernel_names:
                raw_kernel_names = kernel_names_data.get("kernel_names", [])
                num_kernels = len(raw_kernel_names)
                kernel_colors = build_kernel_colors(num_kernels)
                # A single roofline plot. Kernel identity is carried by color
                # plus the HTML legend panel, so the former plot-point and
                # kernel-name symbol tables (and their extra subplots) are gone.
                total_figure_height = 640
                fig = go.Figure()
                skipAI = False
            else:
                # generate an empty figure object in the
                # event that no kernel names are provided
                fig = go.Figure()
        else:
            # Adding to an existing figure (stacking another datatype's roofs).
            skipAI = True

        self.__ceiling_data = construct_roof(
            roofline_parameters=self.__run_parameters,
            dtype=dtype,
            mspec=self.__mspec,
            ai_data=self.__ai_data,
        )
        console_debug("roofline", f"Ceiling data:\n{self.__ceiling_data}")

        if all(
            v is None or all(x is None for x in v) for v in self.__ceiling_data.values()
        ):
            console_warning(
                "Unable to generate roofline plot due to missing or corrupted "
                "benchmark data. Returning empty figure."
            )
            return fig if fig is not None else go.Figure()

        ops_flops = "OP" if dtype.startswith("I") else "FLOP"
        subplot_kwargs = {"row": subplot_row, "col": 1} if subplot_row else {}

        # Explicit log-axis bounds spanning every roof and point, so filtering
        # by peak/kernel never rescales the view and horizontal peak roofs are
        # drawn all the way to the right edge.
        x_lo, x_hi, y_lo, y_hi = roofline_axis_bounds(
            self.__ceiling_data, self.__ai_data or {}, sanitized_cache_hierarchy
        )

        #######################
        # Plot Application AI
        #######################
        # One scatter trace per kernel (color = kernel, marker shape = peak).
        # The matching view model is the source of truth the client controller
        # restyles from when the peak dropdown or a kernel toggle changes.
        if ops_flops == "FLOP" and not skipAI and has_kernel_names:
            kernel_names = self.__ai_data.get("kernelNames", [])
            kernel_traces, kernels_model = self._build_kernel_traces(
                kernel_names=kernel_names,
                kernel_colors=kernel_colors,
                sanitized_cache_hierarchy=sanitized_cache_hierarchy,
                ceiling_data=self.__ceiling_data,
                ops_flops=ops_flops,
            )

            first_index = len(fig.data)
            for kernel_trace in kernel_traces:
                fig.add_trace(kernel_trace, **subplot_kwargs)
            trace_indices = list(range(first_index, first_index + len(kernel_traces)))
            for offset, kernel in enumerate(kernels_model):
                kernel["traceIndex"] = trace_indices[offset]

            present_peaks: list[str] = []
            for cache_level in CACHE_LEVELS:
                level_name = cache_level.removeprefix("ai_").upper()
                if level_name not in sanitized_cache_hierarchy:
                    continue
                if any(
                    point["peak"] == level_name
                    for kernel in kernels_model
                    for point in kernel["points"]
                ):
                    present_peaks.append(level_name)

            default_peak = (
                _DEFAULT_PEAK
                if (_DEFAULT_PEAK == "all" or _DEFAULT_PEAK in present_peaks)
                else (present_peaks[0] if present_peaks else "all")
            )
            self.__view_models[ops_flops] = RooflineViewModel(
                peaks=present_peaks,
                peak_symbols={peak: peak_symbol(peak) for peak in present_peaks},
                default_peak=default_peak,
                kernels=kernels_model,
                kernel_trace_indices=trace_indices,
            )

        #######################
        # Bandwidth Ceilings
        #######################
        bandwidth_lines = []
        for level in sanitized_cache_hierarchy:
            key = level.lower()
            line_data = self.__ceiling_data.get(key)
            if (
                line_data
                and isinstance(line_data, (list, tuple))
                and len(line_data) >= 3
            ):
                bandwidth_lines.append({
                    "key": key,
                    "level": level,
                    "x": line_data[0],
                    "y": line_data[1],
                    "value": line_data[2],
                    "dtype": dtype,
                })

        # Track the trace index of each memory-level roof so the peak dropdown
        # can show/hide the matching bandwidth line
        roofline_trace_indices: dict[str, int] = {}

        for bw_line in bandwidth_lines:
            value = to_int(bw_line["value"])
            level = bw_line["level"]

            trace_to_update = None
            for trace in fig.data:
                is_correct_level = trace.name and trace.name.startswith(
                    f"{level.upper()}-"
                )
                has_correct_value = False
                if trace.name and "<br>" in trace.name:
                    try:
                        # Extract value from legend name
                        value_part = trace.name.split("<br>")[1]
                        existing_val = int(value_part.split()[0])
                        if existing_val == value:
                            has_correct_value = True
                    except (ValueError, IndexError):
                        pass

                if is_correct_level and has_correct_value:
                    trace_to_update = trace
                    break

            if trace_to_update:
                try:
                    # Extract existing datatypes from name
                    name_part = trace_to_update.name.split("<br>")[0]
                    existing_dts_str = name_part.split("-", 1)[1]
                    existing_dts = [dt.strip() for dt in existing_dts_str.split(",")]
                except Exception:
                    continue

                all_dts = sorted(list(set(existing_dts + [dtype])))
                all_dts_str = ", ".join(all_dts)
                legend_name = f"{level.upper()}-{all_dts_str}<br>{value} GB/s"
                fig.update_traces(
                    patch={
                        "name": legend_name,
                        "hovertemplate": f"<b>{legend_name}</b><extra></extra>",
                    },
                    selector={"name": trace_to_update.name},
                )
            else:
                # New bandwidth line with value in legend. The diagonal is the
                # infinite line y = BW * AI up to its ridge; draw it from an
                # arbitrarily small AI (toward the origin) to the ridge so
                # panning down-left never reaches an endpoint.
                legend_name = f"{level.upper()}-{dtype}<br>{value} GB/s"
                peak_bw_val = bw_line["value"]
                ridge_x = bw_line["x"][1]
                ridge_y = bw_line["y"][1]
                low_x = _ROOF_EXTRAP_MIN_AI
                low_y = (
                    peak_bw_val * _ROOF_EXTRAP_MIN_AI if peak_bw_val > 0 else ridge_y
                )
                fig.add_trace(
                    go.Scatter(
                        x=[low_x, ridge_x],
                        y=[low_y, ridge_y],
                        name=legend_name,
                        mode="lines",
                        line=dict(color=get_color(level.lower())),
                        hovertemplate=f"<b>{legend_name}</b><extra></extra>",
                    ),
                    **subplot_kwargs,
                )
                roofline_trace_indices[level.upper()] = (
                    len(fig.data) - 1,
                    bw_line["value"],
                )

        # Attach the memory-roof trace indices (with bandwidth) to the view model
        # built this call so the client controller can filter roofs by the
        # selected peak and snap the compute ceilings to the steepest visible one.
        if ops_flops == "FLOP" and not skipAI and has_kernel_names:
            view_model = self.__view_models.get(ops_flops)
            if view_model is not None:
                view_model.roofline_traces = [
                    {"level": roof_level, "traceIndex": idx, "bandwidth": bw}
                    for roof_level, (idx, bw) in roofline_trace_indices.items()
                ]
                view_model.roof_max_ai = _ROOF_EXTRAP_MAX_AI

        max_bw = max((bw_line["value"] for bw_line in bandwidth_lines), default=0.0)

        #######################
        # Peak Performance
        #######################
        valu_data = (
            self.__ceiling_data.get("valu")
            if OpsSupport.VALU in SUPPORTED_DATATYPES[self.__mspec.gpu_arch][dtype]
            else None
        )
        matrix_data = (
            self.__ceiling_data.get("matrix_ops")
            if OpsSupport.MATRIX in SUPPORTED_DATATYPES[self.__mspec.gpu_arch][dtype]
            else None
        )

        if valu_data:
            legend_name = f"Peak VALU-{dtype}<br>{to_int(valu_data[2])} G{ops_flops}/s"
            peak_perf = valu_data[1][0]
            left_x = peak_perf / max_bw if max_bw > 0 else valu_data[0][0]
            fig.add_trace(
                go.Scatter(
                    x=[left_x, _ROOF_EXTRAP_MAX_AI],
                    y=[peak_perf, peak_perf],
                    name=legend_name,
                    mode="lines",
                    line=dict(color=get_color("valu")),
                    hovertemplate=f"<b>{legend_name}</b><extra></extra>",
                ),
                **subplot_kwargs,
            )
            compute_vm = self.__view_models.get(ops_flops)
            if compute_vm is not None:
                compute_vm.compute_traces.append(
                    {"traceIndex": len(fig.data) - 1, "peakPerf": peak_perf}
                )

        if matrix_data:
            matrix_ops_type = get_matrix_ops_type(
                getattr(self.__mspec, "gpu_series", "unknown_series")
            )
            legend_name = (
                f"Peak {matrix_ops_type}-{dtype}<br>"
                f"{to_int(matrix_data[2])} G{ops_flops}/s"
            )
            # Extend left to the steepest (first) diagonal and right past the
            # data so panning never reaches an end of the compute ceiling.
            peak_perf = matrix_data[1][0]
            left_x = peak_perf / max_bw if max_bw > 0 else matrix_data[0][0]
            fig.add_trace(
                go.Scatter(
                    x=[left_x, _ROOF_EXTRAP_MAX_AI],
                    y=[peak_perf, peak_perf],
                    name=legend_name,
                    mode="lines",
                    line=dict(color=get_color("matrix_ops")),
                    hovertemplate=f"<b>{legend_name}</b><extra></extra>",
                ),
                **subplot_kwargs,
            )
            compute_vm = self.__view_models.get(ops_flops)
            if compute_vm is not None:
                compute_vm.compute_traces.append(
                    {"traceIndex": len(fig.data) - 1, "peakPerf": peak_perf}
                )

        #######################
        # Layout Configuration
        #######################
        if is_new_figure:
            fig.update_xaxes(
                type="log",
                range=[float(np.log10(x_lo)), float(np.log10(x_hi))],
                title_text=f"Arithmetic Intensity ({ops_flops}s/Byte)",
                gridcolor="rgba(0, 0, 0, 0.08)",
            )
            fig.update_yaxes(
                type="log",
                range=[float(np.log10(y_lo)), float(np.log10(y_hi))],
                title_text=f"Performance (G{ops_flops}/sec)",
                gridcolor="rgba(0, 0, 0, 0.08)",
            )
            # Make the plot pan on drag / zoom on wheel.
            # No fixed width
            fig.update_layout(
                template="plotly_white",
                title=dict(
                    text=f"Empirical Roofline Analysis ({dtype})",
                    x=0.5,
                    xanchor="center",
                    font=dict(size=15),
                ),
                height=int(total_figure_height),
                dragmode="pan",
                hovermode="closest",
                margin=dict(l=70, r=40, b=55, t=62, pad=4),
                legend=dict(
                    title=dict(text="Roofs (click to toggle)", font=dict(size=11)),
                    orientation="v",
                    yanchor="bottom",
                    y=0.02,
                    xanchor="right",
                    x=0.99,
                    font=dict(size=11),
                    bgcolor="rgba(255, 255, 255, 0.72)",
                    bordercolor="rgba(0, 0, 0, 0.12)",
                    borderwidth=1,
                    itemsizing="constant",
                ),
                hoverlabel=dict(bgcolor="white", font_size=11),
            )

        # For additional datatypes stacked onto an existing figure, extend the
        # title so it lists every datatype shown.
        if not is_new_figure and fig.layout.title.text:
            title_text = fig.layout.title.text
            if "(" in title_text and ")" in title_text:
                prefix = title_text.split("(")[0]
                existing_types = title_text.split("(")[1].split(")")[0]
                if dtype not in existing_types.split(", "):
                    fig.layout.title.text = f"{prefix}({existing_types}, {dtype})"

        return fig

    def cli_generate_plot(
        self,
        dtype: str,
        ai_data: dict[str, Any],
    ) -> Optional[str]:
        """
        Plot CLI mode roofline analysis in terminal using plotext

        :param dtype: The datatype to be profiled
        :param ai_data: Pre-computed arithmetic intensity data from calc_ai_analyze
        :return: Build the current figure using plot.build(),
        or None if datatype is not valid for the architecture
        :rtype: str or None
        """
        console_debug("roofline", "Generating roofline plot for CLI")

        if not (str(dtype) in SUPPORTED_DATATYPES[str(self.__mspec.gpu_arch)].keys()):
            console_error(
                f"{dtype} is not a supported datatype for roofline profiling on "
                f"{getattr(self.__mspec, 'gpu_model', 'N/A')} (arch: "
                f"{self.__mspec.gpu_arch})- cannot construct CLI plot",
                exit=False,
            )
            return

        if not ai_data:
            console_warning(
                "roofline",
                "Skipping roofline charting due to invalid arithmetic intensity data",
            )
            return

        self.__ai_data = ai_data

        workload_dir = self.__run_parameters.get("workload_dir", "")
        if not (Path(workload_dir) / "roofline.csv").is_file():
            console_log(
                "roofline",
                f"{workload_dir}/roofline.csv does not exist",
            )
            return None

        self.__ceiling_data = construct_roof(
            roofline_parameters=self.__run_parameters,
            dtype=dtype,
            mspec=self.__mspec,
        )

        self.roof_setup()

        # Check proper datatype input - takes single str
        if not isinstance(dtype, str):
            console_error("Unsupported datatype input - must be str")

        sanitized_cache_hierarchy = sanitize_mem_level(
            self.__run_parameters["mem_level"], self.__mspec.gpu_model
        )

        kernel_markers = {
            0: "star",
            1: "cross",
            2: "sd",
            3: "shamrock",
            4: "at",
            5: "atom",
        }

        plt.clf()
        plt.plotsize(plt.tw(), plt.th())

        ops_flops = "OP" if dtype.startswith("I") else "FLOP"

        for cache_level in sanitized_cache_hierarchy:
            cache_key = cache_level.lower()

            # cache_data layout:
            #   [0] list[float] — x-axis coords for AI: [start_AI, ridge_point_AI]
            #   [1] list[float] — y-axis coords for performance: [start_perf, peak_perf]
            #   [2] float       — scalar peak bandwidth (GB/s)
            cache_data = self.__ceiling_data.get(cache_key)

            if not cache_data or cache_data[0] is None:
                continue
            plt.plot(
                cache_data[0],
                cache_data[1],
                label=f"{cache_level}-{dtype}",
                marker="braille",
                color=get_color(cache_level, backend="cli"),
            )
            plt.text(
                f"{round(cache_data[2])} GB/s",
                x=cache_data[0][0],
                y=cache_data[1][0],
                background="black",
                color="white",
                alignment="left",
            )
            console_debug(
                "roofline",
                f"{cache_level}: [{cache_data[0][0]},"
                f"{cache_data[0][1]}], "
                f"[{cache_data[1][0]},"
                f"{cache_data[1][1]}], "
                f"{cache_data[2]}",
            )

        # Plot VALU and Matrix Ops Peak
        if (
            OpsSupport.VALU in SUPPORTED_DATATYPES[self.__mspec.gpu_arch][dtype]
            and self.__ceiling_data["valu"]
            and self.__ceiling_data["valu"][0] is not None
        ):
            valu_y = [
                max(self.__ceiling_data["valu"][1][0] - 0.1, 1e-9),
                max(self.__ceiling_data["valu"][1][1] - 0.1, 1e-9),
            ]
            plt.plot(
                self.__ceiling_data["valu"][0],
                valu_y,
                label=f"Peak VALU-{dtype}",
                marker="braille",
                color=get_color("valu", backend="cli"),
            )
            plt.text(
                f"{round(self.__ceiling_data['valu'][2])} G{ops_flops}/s",
                x=self.__ceiling_data["valu"][0][1] - 800,
                y=self.__ceiling_data["valu"][1][1],
                background="black",
                color="white",
                alignment="right",
            )
            console_debug(
                "roofline",
                f"VALU: [{self.__ceiling_data['valu'][0][0]},"
                f"{self.__ceiling_data['valu'][0][1]}], "
                f"[{self.__ceiling_data['valu'][1][0]},"
                f"{self.__ceiling_data['valu'][1][1]}], "
                f"{self.__ceiling_data['valu'][2]}",
            )
        else:
            console_warning(f"No PEAK measurement available for {dtype}")

        if (
            OpsSupport.MATRIX in SUPPORTED_DATATYPES[self.__mspec.gpu_arch][dtype]
            and self.__ceiling_data["matrix_ops"]
            and self.__ceiling_data["matrix_ops"][0] is not None
        ):
            matrix_ops_type = get_matrix_ops_type(
                getattr(self.__mspec, "gpu_series", "unknown_series")
            )
            matrix_y = [
                max(self.__ceiling_data["matrix_ops"][1][0] - 0.1, 1e-9),
                max(self.__ceiling_data["matrix_ops"][1][1] - 0.1, 1e-9),
            ]
            plt.plot(
                self.__ceiling_data["matrix_ops"][0],
                matrix_y,
                label=f"Peak {matrix_ops_type}-{dtype}",
                marker="braille",
                color=get_color("matrix_ops", backend="cli"),
            )
            plt.text(
                f"{round(self.__ceiling_data['matrix_ops'][2])} G{ops_flops}/s",
                x=self.__ceiling_data["matrix_ops"][0][1] - 800,
                y=self.__ceiling_data["matrix_ops"][1][1],
                background="black",
                color="white",
                alignment="right",
            )
            console_debug(
                "roofline",
                f"Matrix Ops: [{self.__ceiling_data['matrix_ops'][0][0]},"
                f"{self.__ceiling_data['matrix_ops'][0][1]}], "
                f"[{self.__ceiling_data['matrix_ops'][1][0]},"
                f"{self.__ceiling_data['matrix_ops'][1][1]}], "
                f"{self.__ceiling_data['matrix_ops'][2]}",
            )
        else:
            console_warning(f"No Matrix Ops measurement available for {dtype}")

        # Plot Application AI
        for cache_level in sanitized_cache_hierarchy:
            key = f"ai_{cache_level.lower()}"
            if key not in self.__ai_data:
                continue

            kernel_names = self.__ai_data.get("kernelNames", [])
            for i in range(len(self.__ai_data.get("kernelNames", []))):
                # Zero intensity level means no data reported for this cache level
                if i >= len(self.__ai_data[key][0]) or i >= len(self.__ai_data[key][1]):
                    console_debug(
                        "roofline",
                        f"AI_{kernel_names[i]}: array too short, skipped",
                    )
                    continue

                if self.__ai_data[key][0][i] > 0 and self.__ai_data[key][1][i] > 0:
                    plt.plot(
                        [self.__ai_data[key][0][i]],
                        [self.__ai_data[key][1][i]],
                        label=f"AI_{cache_level}_{kernel_names[i][:40]}",
                        color=get_color(cache_level, backend="cli"),
                        marker=kernel_markers[i % len(kernel_markers)],
                    )

                console_debug(
                    "roofline",
                    f"AI_{kernel_names[i]}: {self.__ai_data[key][0][i]}, "
                    f"{self.__ai_data[key][1][i]}",
                )
        plt.xlabel(f"Arithmetic Intensity ({ops_flops}s/Byte)")
        plt.ylabel("Performance (GFLOP/sec)")
        wdir = self.__run_parameters.get("workload_dir", "")
        plt.title(f"Roofline ({dtype}) - {wdir}")

        # Canvas config
        plt.theme("pro")
        plt.xscale("log")
        plt.yscale("log")

        # Build figure
        # Print plot using `plt._utility.write(self.cli_generate_plot(dtype))`
        return plt.build()

    def get_dtype(self) -> list[str]:
        """
        Return the data types requested by the user (else the default data type)
        for the roofline plot.
        """
        return self.__run_parameters["roofline_data_type"]
