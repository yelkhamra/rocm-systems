# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""CDNA memory chart renderer (MI200/MI300/MI350)."""

import argparse
import json
import pathlib
from io import StringIO
from typing import Any, Optional, Union

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from utils.mem_chart_common import (
    COLORS,
    PeakBandwidths,
    build_legend,
    bw_color,
    colored,
    format_edge,
    format_mem_chart_heading,
    format_value,
    make_arrows,
    metric_line,
    progress_bar,
    scale_or_none,
    strip_ansi,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MEM_CHART_DEFAULT_ROWS: tuple[tuple[str, Union[int, float, None]], ...] = (
    ("Wavefront Occupancy", 8),
    ("Wave Life", 4200),
    ("SALU", 1200),
    ("SMEM", 45),
    ("VALU", 3500),
    ("Matrix Ops", 800),
    ("VMEM", 220),
    ("LDS", 150),
    ("GWS", 0),
    ("BR", 90),
    ("Active CUs (deprecated)", 110),
    ("Num CUs", 110),
    ("VGPR", 64),
    ("SGPR", 32),
    ("LDS Allocation", 32768),
    ("Scratch Allocation", 0),
    ("Wavefronts", 16384),
    ("Workgroups", 256),
    ("Flat Read", 80),
    ("Flat Write", 20),
    ("Flat Atomic", 4),
    ("Buffer Read", 3000),
    ("Buffer Write", 400),
    ("Buffer Atomic", 8),
    ("LDS Req", 150),
    ("LDS Util", 45),
    ("LDS Latency", 28),
    ("LDS Read", None),
    ("LDS Write", None),
    ("LDS Atomic", None),
    ("VL1 Rd", 3200),
    ("VL1 Wr", 480),
    ("VL1 Atomic", 12),
    ("VL1 Hit", 92),
    ("VL1 Lat", 180),
    ("VL1 Coalesce", 87),
    ("VL1 Stall", 5),
    ("VL1_L2 Rd", 256),
    ("VL1_L2 Wr", 48),
    ("VL1_L2 Atomic", 12),
    ("sL1D Rd", 45),
    ("sL1D Hit", 98),
    ("sL1D Lat", 85),
    ("sL1D_L2 Rd", 1),
    ("sL1D_L2 Wr", 0),
    ("sL1D_L2 Atomic", 0),
    ("IL1 Fetch", 32),
    ("IL1 Hit", 99),
    ("IL1 Lat", 42),
    ("IL1_L2 Rd", 1),
    ("L2 Rd", 300),
    ("L2 Wr", 52),
    ("L2 Atomic", 12),
    ("L2 Hit", 85),
    ("L2 Rd Lat", 220),
    ("L2 Wr Lat", 180),
    ("Fabric_L2 Rd", 45),
    ("Fabric_L2 Wr", 8),
    ("Fabric_L2 Atomic", 1),
    ("L2-Fabric Read BW", 45e9),
    ("L2-Fabric Write and Atomic BW", 8e9),
    ("Fabric Rd Lat", 350),
    ("Fabric Wr Lat", 280),
    ("Fabric Atomic Lat", 310),
    ("HBM Rd", 42),
    ("HBM Wr", 7),
    ("xGMI Read BW", None),
    ("xGMI Write BW", None),
    ("xGMI Atomic BW", None),
    ("PCIe Read BW", None),
    ("PCIe Write BW", None),
    ("PCIe Atomic BW", None),
)

MEM_CHART_PANEL_METRIC_KEYS: tuple[str, ...] = tuple(
    k for k, _ in _MEM_CHART_DEFAULT_ROWS
)

DEFAULT_SAMPLE_METRICS: dict[str, Union[int, float, None]] = dict(
    _MEM_CHART_DEFAULT_ROWS
)


# ---------------------------------------------------------------------------
# Public API helpers
# ---------------------------------------------------------------------------


def normalize_mem_chart_metrics(metric_dict: dict[str, Any]) -> dict[str, Any]:
    """Filter/reorder input to panel key order; missing keys become None."""
    return {k: metric_dict.get(k) for k in MEM_CHART_PANEL_METRIC_KEYS}


def compute_peak_bw(sys_info_row: dict[str, Any]) -> PeakBandwidths:
    """Compute theoretical peak BW (GB/s) from MachineSpecs fields."""
    sclk = _float_or_zero(sys_info_row, "max_sclk")
    mclk = _float_or_zero(sys_info_row, "max_mclk")
    cus = _float_or_zero(sys_info_row, "cu_per_gpu")
    l2_chan = _float_or_zero(sys_info_row, "total_l2_chan")
    mem_ch = _float_or_zero(sys_info_row, "num_memory_channels")
    sqcs = _float_or_zero(sys_info_row, "sqc_per_gpu")

    return PeakBandwidths(
        hbm=mclk / 1000 * 32 * mem_ch if mem_ch else None,
        l2=sclk / 1000 * 128 * l2_chan if l2_chan else None,
        vl1d=sclk / 1000 * 128 * cus if cus else None,
        lds=sclk * cus * 0.128 if cus else None,
        sl1d=sclk / 1000 * 64 * sqcs if sqcs else None,
        l1i=sclk / 1000 * 64 * sqcs if sqcs else None,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _float_or_zero(row: dict[str, Any], key: str) -> float:
    """Extract a float from *row[key]*, defaulting to 0.0."""
    val = row.get(key)
    if val is None:
        return 0.0
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------


def _extract_metrics(metric_dict: dict[str, Any]) -> dict[str, Any]:
    """Extract rendered metrics from the flat dict. Missing keys → None."""
    get = metric_dict.get
    metrics: dict[str, Any] = {}

    # Kernel→L1 request edges
    metrics["flat_read"] = get("Flat Read")
    metrics["flat_write"] = get("Flat Write")
    metrics["flat_atomic"] = get("Flat Atomic")
    metrics["buffer_read"] = get("Buffer Read")
    metrics["buffer_write"] = get("Buffer Write")
    metrics["buffer_atomic"] = get("Buffer Atomic")
    metrics["lds_req"] = get("LDS Req")
    metrics["lds_util"] = get("LDS Util")
    metrics["lds_read"] = get("LDS Read")
    metrics["lds_write"] = get("LDS Write")
    metrics["lds_atomic"] = get("LDS Atomic")
    metrics["smem_rd"] = get("sL1D Rd")
    metrics["icache_rd"] = get("IL1 Fetch")

    # L1 cache panels
    metrics["vl1_hit"] = get("VL1 Hit")
    metrics["sl1d_hit"] = get("sL1D Hit")
    metrics["il1_hit"] = get("IL1 Hit")

    # L1→L2 bytes moved (128B/read, 64B/write, 64B/atomic)
    metrics["vl1_l2_rd_bytes"] = scale_or_none(get("VL1_L2 Rd"), 128)
    metrics["vl1_l2_wr_bytes"] = scale_or_none(get("VL1_L2 Wr"), 64)
    metrics["vl1_l2_atomic_bytes"] = scale_or_none(get("VL1_L2 Atomic"), 64)
    metrics["sl1d_l2_rd_bytes"] = scale_or_none(get("sL1D_L2 Rd"), 64)
    metrics["il1_l2_rd_bytes"] = scale_or_none(get("IL1_L2 Rd"), 64)

    # L2 panel
    metrics["l2_hit"] = get("L2 Hit")

    # L2→Fabric BW (Bytes/s)
    metrics["l2_fabric_read_bw"] = get("L2-Fabric Read BW")
    metrics["l2_fabric_wr_at_bw"] = get("L2-Fabric Write and Atomic BW")

    # xGMI / PCIe BW (gfx950 only)
    metrics["xgmi_read_bw"] = get("xGMI Read BW")
    metrics["xgmi_write_bw"] = get("xGMI Write BW")
    metrics["xgmi_atomic_bw"] = get("xGMI Atomic BW")
    metrics["pcie_read_bw"] = get("PCIe Read BW")
    metrics["pcie_write_bw"] = get("PCIe Write BW")
    metrics["pcie_atomic_bw"] = get("PCIe Atomic BW")

    return metrics


# ---------------------------------------------------------------------------
# Diagram building
# ---------------------------------------------------------------------------


_KERNEL_ARROW_LEN = 16
_STD_ARROW_LEN = 12


_VL1D_H = 14
_LDS_H = 10
_SL1D_H = 4
_L1I_H = 4
_TOTAL_H = _VL1D_H + _LDS_H + _SL1D_H + _L1I_H  # 30


def _pad_to(lines: list[str], target: int) -> list[str]:
    """Pad or truncate *lines* to exactly *target* rows (returns new list)."""
    padded = lines + [""] * max(0, target - len(lines))
    return padded[:target]


def _build_kernel_panel() -> Panel:
    """Build the Kernel (shader core) panel at full diagram height."""
    return Panel(
        "\n" * 6 + "[dim]Shader Core[/dim]\n[dim]Wave Execution[/dim]",
        title=f"[bold {COLORS['kernel']}]Kernel[/bold {COLORS['kernel']}]",
        border_style=COLORS["kernel"],
        width=14,
        height=_TOTAL_H,
    )


def _build_request_edges(
    metrics: dict[str, Any],
    arrows: dict[str, str],
) -> Text:
    """Edges from Kernel to L1 caches, aligned to panel heights."""
    color_read = COLORS["read"]
    color_write = COLORS["write"]
    color_atomic = COLORS["atomic"]
    arrow_left = arrows["left"]
    arrow_right = arrows["right"]
    arrow_both = arrows["both"]

    # VL1D scope — Non-buffer + Buffer requests
    flat_rd = format_edge("Read", metrics["flat_read"])
    flat_wr = format_edge("Write", metrics["flat_write"])
    flat_at = format_edge("Atomic", metrics["flat_atomic"])
    buf_rd = format_edge("Read", metrics["buffer_read"])
    buf_wr = format_edge("Write", metrics["buffer_write"])
    vl1d_lines = [
        "[white]Non-buffer Request[/white]",
        colored(flat_rd, color_read),
        colored(arrow_left, color_read),
        colored(flat_wr, color_write),
        colored(arrow_right, color_write),
        colored(flat_at, color_atomic),
        colored(arrow_both, color_atomic),
        "[white]Buffer Request[/white]",
        colored(buf_rd, color_read),
        colored(arrow_left, color_read),
        colored(buf_wr, color_write),
        colored(arrow_right, color_write),
    ]

    # LDS scope
    if metrics["lds_read"] is not None:
        lds_rd = format_edge("Read", metrics["lds_read"])
        lds_wr = format_edge("Write", metrics["lds_write"])
        lds_at = format_edge("Atomic", metrics["lds_atomic"])
        lds_instr = format_edge("Instr", metrics["lds_req"])
        lds_lines = [
            "[white]LDS[/white]",
            colored(lds_rd, color_read),
            colored(arrow_left, color_read),
            colored(lds_wr, color_write),
            colored(arrow_right, color_write),
            colored(lds_at, color_atomic),
            colored(arrow_both, color_atomic),
            colored(lds_instr, color_read),
            colored(arrow_both, color_read),
        ]
    else:
        lds_lines = [
            "[white]LDS[/white]",
            f"[{color_read}]{format_edge('Instr', metrics['lds_req'])}[/{color_read}]",
            f"[{color_read}]{arrow_both}[/{color_read}]",
        ]

    # sL1D scope — SMEM
    sl1d_lines = [
        "[white]SMEM[/white]",
        f"[{color_read}]{format_edge('Read', metrics['smem_rd'])}[/{color_read}]",
        f"[{color_read}]{arrow_left}[/{color_read}]",
    ]

    # L1I scope — ICACHE
    l1i_lines = [
        "[white]ICACHE[/white]",
        f"[{color_read}]{format_edge('Read', metrics['icache_rd'])}[/{color_read}]",
        f"[{color_read}]{arrow_left}[/{color_read}]",
    ]

    lines = (
        _pad_to(vl1d_lines, _VL1D_H)
        + _pad_to(lds_lines, _LDS_H)
        + _pad_to(sl1d_lines, _SL1D_H)
        + _pad_to(l1i_lines, _L1I_H)
    )
    return Text.from_markup("\n".join(lines))


def _build_l1_stack(metrics: dict[str, Any]) -> Table:
    """Build vertically stacked L1 cache panels: VL1D, LDS, sL1D, L1I."""
    color_block = COLORS["block"]

    vl1_panel = Panel(
        f"{metric_line('Hit', metrics['vl1_hit'], '%', COLORS['hit'])}\n"
        f"[dim]{progress_bar(metrics['vl1_hit'])}[/dim]",
        title=f"[bold {color_block}]VL1D[/bold {color_block}]",
        border_style=color_block,
        width=20,
        height=_VL1D_H,
    )

    lds_panel = Panel(
        f"{metric_line('Util', metrics['lds_util'], '%', COLORS['util'])}\n"
        f"[dim]{progress_bar(metrics['lds_util'])}[/dim]",
        title=f"[bold {COLORS['lds']}]LDS[/bold {COLORS['lds']}]",
        border_style=COLORS["lds"],
        width=20,
        height=_LDS_H,
    )

    sl1d_panel = Panel(
        f"{metric_line('Hit', metrics['sl1d_hit'], '%', COLORS['hit'])}\n"
        f"[dim]{progress_bar(metrics['sl1d_hit'])}[/dim]",
        title=f"[bold {color_block}]sL1D[/bold {color_block}]",
        border_style=color_block,
        width=20,
        height=_SL1D_H,
    )

    l1i_panel = Panel(
        f"{metric_line('Hit', metrics['il1_hit'], '%', COLORS['hit'])}\n"
        f"[dim]{progress_bar(metrics['il1_hit'])}[/dim]",
        title=f"[bold {color_block}]L1I[/bold {color_block}]",
        border_style=color_block,
        width=20,
        height=_L1I_H,
    )

    stack = Table.grid(padding=0)
    stack.add_column()
    stack.add_row(vl1_panel)
    stack.add_row(lds_panel)
    stack.add_row(sl1d_panel)
    stack.add_row(l1i_panel)
    return stack


def _build_l1_l2_edges(
    metrics: dict[str, Any],
    arrows: dict[str, str],
    peak_bw: Optional[PeakBandwidths] = None,
) -> Text:
    """L1→L2 edge column: bytes moved (VL1D Rd/Wr/Atomic, sL1D Rd, L1I Rd)."""
    vl1_peak = peak_bw.vl1d if peak_bw else None
    sl1d_peak = peak_bw.sl1d if peak_bw else None
    l1i_peak = peak_bw.l1i if peak_bw else None
    color_read = bw_color(metrics.get("vl1_l2_rd_bytes"), vl1_peak, COLORS["read"])
    color_write = bw_color(metrics.get("vl1_l2_wr_bytes"), vl1_peak, COLORS["write"])
    color_atomic = bw_color(
        metrics.get("vl1_l2_atomic_bytes"), vl1_peak, COLORS["atomic"]
    )
    arrow_left = arrows["left"]
    arrow_right = arrows["right"]

    vl1_rd_bw = format_value(metrics["vl1_l2_rd_bytes"], "Bytes/s", 1)
    vl1_wr_bw = format_value(metrics["vl1_l2_wr_bytes"], "Bytes/s", 1)
    vl1_at_bw = format_value(metrics["vl1_l2_atomic_bytes"], "Bytes/s", 1)
    sl1d_rd_bw = format_value(metrics["sl1d_l2_rd_bytes"], "Bytes/s", 1)
    il1_rd_bw = format_value(metrics["il1_l2_rd_bytes"], "Bytes/s", 1)
    color_sl1d = bw_color(metrics.get("sl1d_l2_rd_bytes"), sl1d_peak, COLORS["read"])
    color_l1i = bw_color(metrics.get("il1_l2_rd_bytes"), l1i_peak, COLORS["read"])

    vl1d_lines = [
        f"[{color_read}]Read BW[/{color_read}]",
        f"[{color_read}]{vl1_rd_bw}[/{color_read}]",
        f"[{color_read}]{arrow_left}[/{color_read}]",
        "",
        f"[{color_write}]Write BW[/{color_write}]",
        f"[{color_write}]{vl1_wr_bw}[/{color_write}]",
        f"[{color_write}]{arrow_right}[/{color_write}]",
        "",
        f"[{color_atomic}]Atomic BW[/{color_atomic}]",
        f"[{color_atomic}]{vl1_at_bw}[/{color_atomic}]",
        f"[{color_atomic}]{arrows['both']}[/{color_atomic}]",
    ]
    sl1d_lines = [
        f"[{color_sl1d}]Read BW[/{color_sl1d}]",
        f"[{color_sl1d}]{sl1d_rd_bw}[/{color_sl1d}]",
        f"[{COLORS['read']}]{arrow_left}[/{COLORS['read']}]",
    ]
    l1i_lines = [
        f"[{color_l1i}]Read BW[/{color_l1i}]",
        f"[{color_l1i}]{il1_rd_bw}[/{color_l1i}]",
        f"[{COLORS['read']}]{arrow_left}[/{COLORS['read']}]",
    ]

    lines = (
        _pad_to(vl1d_lines, _VL1D_H)
        + _pad_to([], _LDS_H)
        + _pad_to(sl1d_lines, _SL1D_H)
        + _pad_to(l1i_lines, _L1I_H)
    )
    return Text.from_markup("\n".join(lines))


def _build_l2_panel(metrics: dict[str, Any]) -> Panel:
    color_block = COLORS["block"]
    return Panel(
        f"{metric_line('Hit', metrics['l2_hit'], '%', COLORS['hit'])}\n"
        f"[dim]{progress_bar(metrics['l2_hit'])}[/dim]",
        title=f"[bold {color_block}]L2[/bold {color_block}]",
        border_style=color_block,
        width=18,
        height=_TOTAL_H,
    )


def _build_l2_fabric_edges(
    metrics: dict[str, Any],
    arrows: dict[str, str],
    peak_bw: Optional[PeakBandwidths] = None,
) -> Text:
    """L2→Fabric edges: Read BW and Write/Atomic BW."""
    l2_peak = peak_bw.l2 if peak_bw else None
    color_read = bw_color(metrics.get("l2_fabric_read_bw"), l2_peak, COLORS["read"])
    color_write = bw_color(metrics.get("l2_fabric_wr_at_bw"), l2_peak, COLORS["write"])
    arrow_left = arrows["left"]
    arrow_right = arrows["right"]

    rd_bw = format_value(metrics["l2_fabric_read_bw"], "Bytes/s", 1)
    wr_at_bw = format_value(metrics["l2_fabric_wr_at_bw"], "Bytes/s", 1)

    content = [
        f"[{color_read}]Read BW[/{color_read}]",
        f"[{color_read}]{rd_bw}[/{color_read}]",
        f"[{color_read}]{arrow_left}[/{color_read}]",
        "",
        f"[{color_write}]Write/Atomic BW[/{color_write}]",
        f"[{color_write}]{wr_at_bw}[/{color_write}]",
        f"[{color_write}]{arrow_right}[/{color_write}]",
    ]
    offset = (_TOTAL_H - len(content)) // 2
    lines = [""] * offset + content
    lines = _pad_to(lines, _TOTAL_H)
    return Text.from_markup("\n".join(lines))


def _ip_block(
    title: str,
    width: int,
    border_style: str = COLORS["block"],
    content: str = "",
) -> Panel:
    """Create an IP block panel with standard height."""
    return Panel(
        content,
        title=f"[bold {border_style}]{title}[/bold {border_style}]",
        border_style=border_style,
        width=width,
        height=_TOTAL_H,
    )


def _build_xgmi_row(console: Console, metrics: dict[str, Any]) -> None:
    """Render the xGMI block above the main diagram with BW metrics."""
    color_read = COLORS["read"]
    color_write = COLORS["write"]
    color_atomic = COLORS["atomic"]
    read_bw = format_value(metrics.get("xgmi_read_bw"), "Bytes/s", 1)
    write_bw = format_value(metrics.get("xgmi_write_bw"), "Bytes/s", 1)
    atomic_bw = format_value(metrics.get("xgmi_atomic_bw"), "Bytes/s", 1)

    xgmi_panel = Panel(
        "[dim]XGMI (to Peer GPU)[/dim]",
        border_style="bright_yellow",
        width=30,
        height=3,
    )
    xgmi_layout = Table.grid(padding=0)
    xgmi_layout.add_column(width=97)
    xgmi_layout.add_column()
    xgmi_layout.add_row("", xgmi_panel)
    console.print(xgmi_layout)

    pad = " " * 100
    arrow_lines = Text.from_markup(
        f"{pad}[{color_read}]|^  Read BW    {read_bw}[/{color_read}]\n"
        f"{pad}[{color_write}]||  Write BW   {write_bw}[/{color_write}]\n"
        f"{pad}[{color_atomic}]||  Atomic BW  {atomic_bw}[/{color_atomic}]"
    )
    console.print(arrow_lines)


def _build_pcie_row(console: Console, metrics: dict[str, Any]) -> None:
    """Render the PCIe block below the main diagram with BW metrics."""
    color_read = COLORS["read"]
    color_write = COLORS["write"]
    color_atomic = COLORS["atomic"]
    read_bw = format_value(metrics.get("pcie_read_bw"), "Bytes/s", 1)
    write_bw = format_value(metrics.get("pcie_write_bw"), "Bytes/s", 1)
    atomic_bw = format_value(metrics.get("pcie_atomic_bw"), "Bytes/s", 1)

    pad = " " * 100
    arrow_lines = Text.from_markup(
        f"{pad}[{color_read}]||  Read BW    {read_bw}[/{color_read}]\n"
        f"{pad}[{color_write}]||  Write BW   {write_bw}[/{color_write}]\n"
        f"{pad}[{color_atomic}]V|  Atomic BW  {atomic_bw}[/{color_atomic}]"
    )
    console.print(arrow_lines)

    pcie_panel = Panel(
        "[dim]PCIe (to CPU or Non-XGMI connected GPU)[/dim]",
        border_style="dark_olive_green3",
        width=50,
        height=3,
    )
    pcie_layout = Table.grid(padding=0)
    pcie_layout.add_column(width=87)
    pcie_layout.add_column()
    pcie_layout.add_row("", pcie_panel)
    console.print(pcie_layout)


def _measure_grid(grid: Table, console_width: int = 240) -> tuple[int, int]:
    """Return (total_width, fabric_col) by rendering the grid to a buffer."""
    buf = StringIO()
    tmp = Console(file=buf, force_terminal=False, width=console_width, height=5)
    tmp.print(grid)
    clean = strip_ansi(buf.getvalue())
    total = 0
    fabric_col = 0
    for line in clean.split("\n"):
        stripped = line.rstrip()
        total = max(total, len(stripped))
        if "Data Fabric" in stripped and not fabric_col:
            fabric_col = stripped.index("Data Fabric") - 5
    return total, fabric_col


def _print_scope_bar(console: Console, total_width: int, fabric_col: int) -> None:
    """Print scope bar spanning *total_width*, split at *fabric_col*."""
    gpu_label = " [dim]GPU (XCD)[/dim] "
    fabric_label = " [dim]Fabric / Memory[/dim] "
    gpu_label_plain = " GPU (XCD) "
    fabric_label_plain = " Fabric / Memory "

    gpu_section = fabric_col - 1  # -1 for leading '|'
    fab_section = total_width - fabric_col - 2  # -2 for middle '|' and trailing '|'

    gpu_pad_l = (gpu_section - len(gpu_label_plain)) // 2
    gpu_pad_r = gpu_section - len(gpu_label_plain) - gpu_pad_l
    fab_pad_l = (fab_section - len(fabric_label_plain)) // 2
    fab_pad_r = fab_section - len(fabric_label_plain) - fab_pad_l

    console.print(
        f"|{'-' * gpu_pad_l}{gpu_label}{'-' * gpu_pad_r}"
        f"|{'-' * fab_pad_l}{fabric_label}{'-' * fab_pad_r}|"
    )


# ---------------------------------------------------------------------------
# Main diagram assembly
# ---------------------------------------------------------------------------


def create_mem_chart_diagram(
    metric_dict: dict[str, Any],
    console: Console,
    show_debug: bool = False,
    chart_title: str = "",
    gpu_arch: Optional[str] = None,
    peak_bw: Optional[PeakBandwidths] = None,
) -> None:
    """Create the CDNA memory diagram matching the reference PNG layout."""
    metrics = _extract_metrics(metric_dict)
    kernel_arrows = make_arrows(_KERNEL_ARROW_LEN)
    std_arrows = make_arrows(_STD_ARROW_LEN)
    is_gfx950 = gpu_arch is not None and gpu_arch.startswith("gfx950")

    # Build main diagram grid first (needed to measure width for scope bar)
    kernel = _build_kernel_panel()
    req_edges = _build_request_edges(metrics, kernel_arrows)
    l1_stack = _build_l1_stack(metrics)
    l1_l2_edges = _build_l1_l2_edges(metrics, std_arrows, peak_bw)
    l2 = _build_l2_panel(metrics)
    l2_fab_edges = _build_l2_fabric_edges(metrics, std_arrows, peak_bw)
    fabric = _ip_block("Data Fabric", 22, "bright_magenta")
    mall = _ip_block("MALL", 18, "indian_red")
    umc = _ip_block("UMC", 8)
    hbm_content = "\n\n[bold bright_green]HBM[/bold bright_green]"
    hbm = _ip_block("HBM", 10, "bright_yellow", hbm_content)

    main_layout = Table.grid(padding=0)
    for _ in range(10):
        main_layout.add_column()
    main_layout.add_row(
        kernel,
        req_edges,
        l1_stack,
        l1_l2_edges,
        l2,
        l2_fab_edges,
        fabric,
        mall,
        umc,
        hbm,
    )

    chart_width, fabric_col = _measure_grid(main_layout, console.width)

    console.print()
    if chart_title:
        console.print(f"[bold]{chart_title}[/bold]")

    if is_gfx950:
        _build_xgmi_row(console, metrics)
        console.print()
    _print_scope_bar(console, chart_width, fabric_col)
    console.print()

    console.print(main_layout)
    console.print()

    if is_gfx950:
        _build_pcie_row(console, metrics)
        console.print()

    console.print(build_legend())
    console.print()

    if show_debug:
        console.print("[dim]Architecture Notes (CDNA):[/dim]")
        console.print("  VL1D: Per-CU vector data cache (Buffer/Non-buffer requests)")
        console.print("  LDS: Local Data Share, on-CU scratchpad")
        console.print("  sL1D: Per-CU scalar data cache (SMEM requests)")
        console.print("  L1I: Per-CU instruction cache (ICACHE requests)")
        console.print("  L2 (TCC): Shared last-level cache")
        console.print("  Data Fabric: Infinity Fabric interconnect")
        console.print("  MALL: Mid-level Address Lookup Layer (MI300+)")
        console.print("  UMC: Unified Memory Controller")
        console.print("  HBM: High Bandwidth Memory")
        console.print(
            "  xGMI: Inter-GPU link (MI350 has individual counters;"
            " earlier cards use 'traffic to remote')"
        )
        console.print(
            "  PCIe: Host/non-xGMI link (MI350 has individual counters;"
            " earlier cards use 'traffic to remote')"
        )
        console.print()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def plot_mem_chart(
    normal_unit: str,
    metric_dict: dict[str, Any],
    *,
    chart_title: Optional[str] = None,
    gpu_arch: Optional[str] = None,
    peak_bw: Optional[PeakBandwidths] = None,
) -> str:
    """Render the CDNA memory chart and return as a string."""
    flat = normalize_mem_chart_metrics(metric_dict)
    resolved_heading = (
        format_mem_chart_heading(normal_unit, panel_id=300)
        if chart_title is None
        else chart_title
    )
    buf = StringIO()
    console = Console(file=buf, force_terminal=True, width=240, height=80)
    create_mem_chart_diagram(
        flat,
        console,
        show_debug=False,
        chart_title=resolved_heading,
        gpu_arch=gpu_arch,
        peak_bw=peak_bw,
    )
    return buf.getvalue()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _render_to_plain_text(
    metrics: dict[str, Any],
    heading: str,
    show_debug: bool,
    gpu_arch: Optional[str] = None,
    peak_bw: Optional[PeakBandwidths] = None,
) -> str:
    """Render chart to plain text (no ANSI codes)."""
    buf = StringIO()
    console = Console(file=buf, force_terminal=False, width=240, height=80)
    create_mem_chart_diagram(
        metrics,
        console,
        show_debug=show_debug,
        chart_title=heading,
        gpu_arch=gpu_arch,
        peak_bw=peak_bw,
    )
    return strip_ansi(buf.getvalue())


def main() -> None:
    arg_parser = argparse.ArgumentParser(
        description="CDNA Memory Chart - CLI",
    )
    arg_parser.add_argument(
        "--data",
        "-d",
        help="JSON file with metrics data",
    )
    arg_parser.add_argument(
        "--debug",
        action="store_true",
        help="Show debug info",
    )
    arg_parser.add_argument(
        "--norm",
        default="per_kernel",
        help="Normalization unit",
    )
    arg_parser.add_argument(
        "--arch", default=None, help="GPU architecture (e.g. gfx950)"
    )
    arg_parser.add_argument("--txt", help="Write plain text to file")
    arg_parser.add_argument("--svg", help="Write SVG to file")
    args = arg_parser.parse_args()

    if args.data:
        with pathlib.Path(args.data).open(encoding="utf-8") as f:
            metrics = json.load(f)
    else:
        metrics = dict(DEFAULT_SAMPLE_METRICS)

    heading = format_mem_chart_heading(args.norm)

    if args.txt:
        clean = _render_to_plain_text(
            metrics,
            heading,
            args.debug,
            gpu_arch=args.arch,
        )
        with pathlib.Path(args.txt).open("w", encoding="utf-8") as f:
            f.write(clean)
        return

    if args.svg:
        console = Console(record=True, width=240, height=80)
        create_mem_chart_diagram(
            metrics,
            console,
            show_debug=args.debug,
            chart_title=heading,
            gpu_arch=args.arch,
        )
        console.save_svg(args.svg, title="CDNA Memory Chart")
        return

    console = Console(width=240)
    create_mem_chart_diagram(
        metrics,
        console,
        show_debug=args.debug,
        chart_title=heading,
        gpu_arch=args.arch,
    )


if __name__ == "__main__":
    main()
