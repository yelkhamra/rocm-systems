# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""
RDNA3.5 Memory Architecture Diagram - CLI Visualization
=============================================================================

USAGE:
    python mem_chart_gfx11.py [--data metrics.json] [--debug]
        [--txt file.txt] [--svg file.svg]

API:
    normalize_mem_chart_metrics(metric_dict) -> flat ordered dict for UIs
    plot_mem_chart(metric_dict, *, chart_title=...) -> str

Metric dict keys must match the Memory Chart panel YAML for RDNA3.5:

    src/rocprof_compute_soc/analysis_configs/gfx115x/0300_memory_chart.yaml

Use ``MEM_CHART_PANEL_METRIC_KEYS`` for the authoritative ordered list.
(If a future gfx target adds ``0300_memory_chart.yaml``, keep keys aligned there.)

Bandwidth values are **Bytes/s**, matching the YAML ``unit: Bytes/s`` rows.

RDNA3.5 MEMORY HIERARCHY (GCEA = Graphics Core Efficiency Arbiter):
   Kernel -> GL0 (TCP Cache)      -> GL1 Cache -> GL2 Cache -> GCEA -> System Memory
         -> SQC (ICache/DCache)   -> GL1 Cache -> GL2 Cache -> GCEA -> System Memory
         -> LDS (Local Data Share) [stays on CU, no GL1 Cache connection]
"""

import argparse
import json
import pathlib
from io import StringIO
from typing import Any, Union

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from utils.mem_chart_common import (
    COLORS,
    build_legend,
    colored,
    format_edge,
    format_mem_chart_heading,
    format_value,
    make_arrows,
    metric_line,
    progress_bar,
    safe_float_sum,
    strip_ansi,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Keys = ``metric:`` names under each ``metric_table`` in
# ``analysis_configs/gfx115x/0300_memory_chart.yaml`` (tables 301–309), in panel order.
# Commented-out YAML metrics (e.g. TCP Atomic, LDS direct read/write) are omitted.
_MEM_CHART_DEFAULT_ROWS: tuple[tuple[str, Union[int, float]], ...] = (
    # Table 301: Instruction Cache
    ("ICache Requests", 450),
    ("ICache Utilization", 45.2),
    ("ICache Hit Rate", 98.5),
    ("ICache Miss Rate", 1.5),
    ("ICache Request Stall Rate", 2.1),
    ("ICache-GL1 Read Bandwidth", 57.6e9),
    # Table 302: Scalar Data Cache
    ("Dcache Requests", 225),
    ("Dcache Utilization", 38.7),
    ("Dcache Hit Rate", 95.3),
    ("Dcache Request Stall Rate", 1.8),
    ("Dcache-GL1 Read Bandwidth", 28.8e9),
    # Table 303: TCP Cache (GL0 Vector Cache)
    ("TCP Total Requests", 1_250_000),
    ("TCP Read Requests", 875_000),
    ("TCP Write Requests", 375_000),
    ("TCP Miss Requests", 150_000),
    ("GL0 Cache Hit Rate (TCP Cache)", 88.0),
    ("GL0 Cache BW (TCP Cache)", 80e9),
    # Table 304: LDS
    ("LDS Instructions", 125_000),
    ("LDS Atomic Instructions", 10_000),
    ("LDS Instruction Cycles", 250_000),
    ("LDS Wait Cycles", 12_500),
    ("LDS Bank Conflict Rate", 4.0),
    ("LDS Estimated Bandwidth", 256e9),
    # Table 305: TCP-GL1 Interface
    ("TCP-GL1 Read Requests", 150_000),
    ("TCP-GL1 Write Requests", 50_000),
    ("TCP-GL1 Read Bandwidth", 96e9),
    ("TCP-GL1 Write Bandwidth", 32e9),
    # Table 306: GL1 Cache (L1)
    ("GL1 Cache Utilization", 65.2),
    ("GL1 Cache Total Requests", 200_000),
    ("GL1 Cache Read Requests", 150_000),
    ("GL1 Cache Write Requests", 50_000),
    ("GL1 Cache Miss Requests", 30_000),
    ("GL1 Cache Hit Rate", 85.0),
    ("GL1 Cache Starve Rate", 5.2),
    ("GL1 Cache Stall GL2 Backpressure", 8.5),
    # Table 307: GL1-GL2 Interface
    ("GL1-GL2 Read Requests", 30_000),
    ("GL1-GL2 Write Requests", 10_000),
    ("GL1-GL2 Read Bandwidth", 48e9),
    ("GL1-GL2 Write Bandwidth", 16e9),
    ("GL1-GL2 Read Latency", 85.2),
    ("GL1-GL2 Write Latency", 62.4),
    # Table 308: GL2 Cache (L2)
    ("GL2 Cache Utilization", 74.2),
    ("GL2 Cache Total Requests", 40_000),
    ("GL2 Cache Read Requests", 30_000),
    ("GL2 Cache Write Requests", 10_000),
    ("GL2 Cache Atomic Requests", 1_000),
    ("GL2 Cache Hit Rate", 82.5),
    ("GL2 Cache Read BW", 64e9),
    ("GL2 Cache Write BW", 24e9),
    # Table 309: Graphics Core Efficiency Arbiter (GCEA) to System Memory
    ("SARB Utilization", 52.3),
    ("SARB Stall Rate", 12.4),
    ("DRAM Read Requests", 25_000),
    ("DRAM Write Requests", 8_000),
    ("DRAM Read Bandwidth", 100e9),
    ("DRAM Write Bandwidth", 60e9),
    ("Read Returns", 25_000),
    ("Write Returns", 8_000),
)

MEM_CHART_PANEL_METRIC_KEYS: tuple[str, ...] = tuple(
    k for k, _ in _MEM_CHART_DEFAULT_ROWS
)

DEFAULT_SAMPLE_METRICS: dict[str, Union[int, float]] = dict(_MEM_CHART_DEFAULT_ROWS)


# ---------------------------------------------------------------------------
# Public API: heading, normalization, sample data
# ---------------------------------------------------------------------------


def normalize_mem_chart_metrics(metric_dict: dict[str, Any]) -> dict[str, Any]:
    """Return a single flat map: YAML metric name -> value, panel order.

    All keys in ``MEM_CHART_PANEL_METRIC_KEYS`` are present; unknown input keys
    are dropped. Use before rendering or serializing for front-ends.
    """
    return {k: metric_dict.get(k) for k in MEM_CHART_PANEL_METRIC_KEYS}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _print_mem_chart_scope_bar(console: Console) -> None:
    """Horizontal rule: GPU span vs System Memory (above the diagram body)."""
    console.print(
        "|"
        + "-" * 62
        + " [dim]GPU[/dim] "
        + "-" * 62
        + "|"
        + "-" * 4
        + " [dim]System Memory[/dim] "
        + "-" * 4
        + "|"
    )


# ---------------------------------------------------------------------------
# Diagram construction: _extract_metrics, _build_kernel_and_l0,
#   _build_cache_columns, _build_memory_columns, create_mem_chart_diagram
# ---------------------------------------------------------------------------


def _extract_metrics(metric_dict: dict[str, Any]) -> dict[str, Any]:
    """Pull all needed values from the flat metric dict."""
    metrics: dict[str, Any] = {}

    metrics["icache_req"] = metric_dict.get("ICache Requests")
    metrics["icache_hit"] = metric_dict.get("ICache Hit Rate")
    metrics["icache_gl1_bw"] = metric_dict.get("ICache-GL1 Read Bandwidth")

    metrics["dcache_req"] = metric_dict.get("Dcache Requests")
    metrics["dcache_hit"] = metric_dict.get("Dcache Hit Rate")
    metrics["dcache_gl1_bw"] = metric_dict.get("Dcache-GL1 Read Bandwidth")

    metrics["tcp_read_req"] = metric_dict.get("TCP Read Requests")
    metrics["tcp_write_req"] = metric_dict.get("TCP Write Requests")
    metrics["tcp_hit"] = metric_dict.get("GL0 Cache Hit Rate (TCP Cache)")
    metrics["tcp_bw"] = metric_dict.get("GL0 Cache BW (TCP Cache)")

    metrics["lds_insts"] = metric_dict.get("LDS Instructions")
    metrics["lds_inst_cycles"] = metric_dict.get("LDS Instruction Cycles")
    metrics["lds_atomic_insts"] = metric_dict.get("LDS Atomic Instructions")
    metrics["lds_bw"] = metric_dict.get("LDS Estimated Bandwidth")
    metrics["lds_bank_conflict"] = metric_dict.get("LDS Bank Conflict Rate")

    metrics["tcp_gl1_read_bw"] = metric_dict.get("TCP-GL1 Read Bandwidth")
    metrics["tcp_gl1_write_bw"] = metric_dict.get("TCP-GL1 Write Bandwidth")

    metrics["sqc_gl1_read_bw"] = safe_float_sum(
        metrics["icache_gl1_bw"], metrics["dcache_gl1_bw"]
    )

    metrics["gl1c_util"] = metric_dict.get("GL1 Cache Utilization")
    metrics["gl1c_hit"] = metric_dict.get("GL1 Cache Hit Rate")
    metrics["gl1c_stall_gl2"] = metric_dict.get("GL1 Cache Stall GL2 Backpressure")

    metrics["gl1_gl2_read_bw"] = metric_dict.get("GL1-GL2 Read Bandwidth")
    metrics["gl1_gl2_write_bw"] = metric_dict.get("GL1-GL2 Write Bandwidth")

    metrics["gl2c_util"] = metric_dict.get("GL2 Cache Utilization")
    metrics["gl2c_hit"] = metric_dict.get("GL2 Cache Hit Rate")
    metrics["gl2c_read_bw"] = metric_dict.get("GL2 Cache Read BW")
    metrics["gl2c_write_bw"] = metric_dict.get("GL2 Cache Write BW")

    metrics["sarb_util"] = metric_dict.get("SARB Utilization")
    metrics["sarb_stall"] = metric_dict.get("SARB Stall Rate")
    metrics["dram_read_bw"] = metric_dict.get("DRAM Read Bandwidth", 0)
    metrics["dram_write_bw"] = metric_dict.get("DRAM Write Bandwidth", 0)

    metrics["total_bw"] = (metrics["dram_read_bw"] or 0) + (
        metrics["dram_write_bw"] or 0
    )

    return metrics


_DIAGRAM_HEIGHT = 30


def _build_bw_edge_column(
    read_bw_str: str,
    write_bw_str: str,
    arrows: dict[str, str],
    height: int = _DIAGRAM_HEIGHT,
    offset: int = 11,
) -> Text:
    """Build a Read/Write BW edge column at *offset* rows from the top."""
    color_read = COLORS["read"]
    color_write = COLORS["write"]
    content = [
        colored("Read BW", color_read),
        colored(read_bw_str, color_read),
        colored(arrows["left"], color_read),
        "",
        colored("Write BW", color_write),
        colored(write_bw_str, color_write),
        colored(arrows["right"], color_write),
    ]
    lines = [""] * offset + content
    lines += [""] * max(0, height - len(lines))
    return Text.from_markup("\n".join(lines[:height]))


def _build_gfx11_kernel_panel() -> Panel:
    """Build the Kernel (shader core) panel at full diagram height."""
    return Panel(
        "\n" * 11 + "[dim]Shader Core[/dim]\n[dim]Wave Execution[/dim]",
        title=colored("Kernel", COLORS["kernel"]),
        border_style=COLORS["kernel"],
        width=14,
        height=_DIAGRAM_HEIGHT,
    )


def _build_kernel_edges(
    metrics: dict[str, Any],
    kernel_arrows: dict[str, str],
) -> Text:
    """Build edge labels from Kernel to LDS/TCP/SQC."""
    color_read = COLORS["read"]
    color_write = COLORS["write"]
    color_atomic = COLORS["atomic"]
    arrow_left = kernel_arrows["left"]
    arrow_right = kernel_arrows["right"]
    arrow_both = kernel_arrows["both"]

    lds_rd = format_edge("Read", metrics["lds_insts"])
    lds_wr = format_edge("Write", metrics["lds_inst_cycles"])
    lds_at = format_edge("Atomic", metrics["lds_atomic_insts"])
    tcp_rd = format_edge("Read", metrics["tcp_read_req"])
    tcp_wr = format_edge("Write", metrics["tcp_write_req"])
    icache_edge = format_edge("ICache", metrics["icache_req"])
    dcache_edge = format_edge("DCache", metrics["dcache_req"])

    lines = [
        "",
        "     [white]Request[/white]",
        "",
        colored(lds_rd, color_read),
        colored(arrow_left, color_read),
        colored(lds_wr, color_write),
        colored(arrow_right, color_write),
        colored(lds_at, color_atomic),
        colored(arrow_both, color_atomic),
        "",
        "",
        "",
        "",
        colored(tcp_rd, color_read),
        colored(arrow_left, color_read),
        colored(tcp_wr, color_write),
        colored(arrow_right, color_write),
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        colored(icache_edge, color_read),
        colored(arrow_left, color_read),
        colored(dcache_edge, color_read),
        colored(arrow_left, color_read),
    ]
    return Text.from_markup("\n".join(lines))


def _build_l0_stack(metrics: dict[str, Any]) -> Table:
    """Build vertically stacked L0 panels: LDS, GL0 (TCP), SQC."""
    color_block = COLORS["block"]

    lds_bw_line = (
        metric_line("BW", metrics["lds_bw"], "Bytes/s", COLORS["bw"])
        if metrics["lds_bw"]
        else ""
    )
    lds_conflict_line = (
        metric_line(
            "Bank Conflict",
            metrics["lds_bank_conflict"],
            "%",
            COLORS["stall"],
        )
        if metrics["lds_bank_conflict"]
        else ""
    )
    lds_panel = Panel(
        f"{lds_bw_line}\n{lds_conflict_line}",
        title=f"[bold {color_block}]LDS[/bold {color_block}]",
        border_style=color_block,
        width=20,
        height=10,
    )

    tcp_bw_line = (
        metric_line("BW", metrics["tcp_bw"], "Bytes/s", COLORS["bw"])
        if metrics["tcp_bw"]
        else ""
    )
    tcp_hit_line = metric_line("Hit Rate", metrics["tcp_hit"], "%", COLORS["hit"])
    tcp_panel = Panel(
        f"{tcp_hit_line}\n{tcp_bw_line}",
        title=f"[bold {color_block}]GL0 (TCP Cache)[/bold {color_block}]",
        border_style=color_block,
        width=20,
        height=10,
    )

    icache_line = metric_line("ICache", metrics["icache_hit"], "%", COLORS["hit"])
    dcache_line = metric_line("DCache", metrics["dcache_hit"], "%", COLORS["hit"])
    sqc_panel = Panel(
        f"{icache_line}\n{dcache_line}",
        title=f"[bold {color_block}]SQC[/bold {color_block}]",
        border_style=color_block,
        width=20,
        height=10,
    )

    stack = Table.grid(padding=0)
    stack.add_column()
    stack.add_row(lds_panel)
    stack.add_row(tcp_panel)
    stack.add_row(sqc_panel)
    return stack


def _build_gl1_edges(
    metrics: dict[str, Any],
    std_arrows: dict[str, str],
) -> Text:
    """Build TCP->GL1 and SQC->GL1 BW edge labels."""
    color_read = COLORS["read"]
    color_write = COLORS["write"]
    arrow_left = std_arrows["left"]
    arrow_right = std_arrows["right"]

    tcp_gl1_rd = format_value(metrics["tcp_gl1_read_bw"], "Bytes/s", 1)
    tcp_gl1_wr = format_value(metrics["tcp_gl1_write_bw"], "Bytes/s", 1)
    sqc_gl1_rd = format_value(metrics["sqc_gl1_read_bw"], "Bytes/s", 1)

    lines = [""] * 11 + [
        colored("Read BW", color_read),
        colored(tcp_gl1_rd, color_read),
        colored(arrow_left, color_read),
        "",
        colored("Write BW", color_write),
        colored(tcp_gl1_wr, color_write),
        colored(arrow_right, color_write),
        "",
        "",
        "",
        colored("Read BW", color_read),
        colored(sqc_gl1_rd, color_read),
        colored(arrow_left, color_read),
    ]
    return Text.from_markup("\n".join(lines))


def _build_gl1_panel(metrics: dict[str, Any]) -> Panel:
    """Build the GL1 Cache panel."""
    color_block = COLORS["block"]
    return Panel(
        f"{metric_line('Util', metrics['gl1c_util'], '%', COLORS['util'])}\n"
        f"[dim]{progress_bar(metrics['gl1c_util'])}[/dim]\n"
        "\n"
        f"{metric_line('Hit Rate', metrics['gl1c_hit'], '%', COLORS['hit'])}\n"
        f"[dim]{progress_bar(metrics['gl1c_hit'])}[/dim]\n"
        "\n"
        f"{metric_line('GL2 Stall', metrics['gl1c_stall_gl2'], '%', COLORS['stall'])}",
        title=f"[bold {color_block}]GL1 Cache[/bold {color_block}]",
        border_style=color_block,
        width=16,
        height=_DIAGRAM_HEIGHT,
    )


def _build_gl2_panel(metrics: dict[str, Any]) -> Panel:
    """Build the GL2 Cache panel."""
    color_block = COLORS["block"]
    return Panel(
        f"{metric_line('Util', metrics['gl2c_util'], '%', COLORS['util'])}\n"
        f"[dim]{progress_bar(metrics['gl2c_util'])}[/dim]\n"
        "\n"
        f"{metric_line('Hit Rate', metrics['gl2c_hit'], '%', COLORS['hit'])}\n"
        f"[dim]{progress_bar(metrics['gl2c_hit'])}[/dim]",
        title=f"[bold {color_block}]GL2 Cache[/bold {color_block}]",
        border_style=color_block,
        width=16,
        height=_DIAGRAM_HEIGHT,
    )


def _build_gcea_panel(metrics: dict[str, Any]) -> Panel:
    """Build the GCEA (Graphics Core Efficiency Arbiter) panel."""
    color_block = COLORS["block"]
    return Panel(
        f"{metric_line('SysArb Util', metrics['sarb_util'], '%', COLORS['util'])}\n"
        f"[dim]{progress_bar(metrics['sarb_util'])}[/dim]\n"
        "\n"
        f"{metric_line('Stall', metrics['sarb_stall'], '%', COLORS['stall'])}",
        title=f"[bold {color_block}]GCEA[/bold {color_block}]",
        border_style=color_block,
        width=16,
        height=_DIAGRAM_HEIGHT,
    )


def _build_dram_panel(metrics: dict[str, Any]) -> Panel:
    """Build the DRAM panel with total bandwidth."""
    color_block = COLORS["block"]
    total = format_value(metrics["total_bw"], "Bytes/s", 1)
    return Panel(
        "[dim]DDR5/LPDDR5[/dim]\n"
        "\n"
        f"Total: [bold bright_green]{total}[/bold bright_green]",
        title=f"[bold {color_block}]DRAM[/bold {color_block}]",
        border_style=color_block,
        width=16,
        height=_DIAGRAM_HEIGHT,
    )


# ---------------------------------------------------------------------------
# Main diagram assembly
# ---------------------------------------------------------------------------


def create_mem_chart_diagram(
    metric_dict: dict[str, Any],
    console: Console,
    show_debug: bool = False,
    chart_title: str = "",
) -> None:
    """Create the RDNA3.5 memory diagram."""
    metrics = _extract_metrics(metric_dict)

    console.print()
    if chart_title:
        console.print(f"[bold]{chart_title}[/bold]")
    _print_mem_chart_scope_bar(console)
    console.print()

    std_arrows = make_arrows(9)
    kernel_arrows = make_arrows(16)

    gl1_gl2_edges = _build_bw_edge_column(
        format_value(metrics["gl1_gl2_read_bw"], "Bytes/s", 1),
        format_value(metrics["gl1_gl2_write_bw"], "Bytes/s", 1),
        std_arrows,
    )
    gl2_gcea_edges = _build_bw_edge_column(
        format_value(metrics["gl2c_read_bw"], "Bytes/s", 1),
        format_value(metrics["gl2c_write_bw"], "Bytes/s", 1),
        std_arrows,
    )
    dram_edges = _build_bw_edge_column(
        format_value(metrics["dram_read_bw"], "Bytes/s", 1),
        format_value(metrics["dram_write_bw"], "Bytes/s", 1),
        std_arrows,
    )

    main_layout = Table.grid(padding=0)
    for _ in range(11):
        main_layout.add_column()

    main_layout.add_row(
        _build_gfx11_kernel_panel(),
        _build_kernel_edges(metrics, kernel_arrows),
        _build_l0_stack(metrics),
        _build_gl1_edges(metrics, std_arrows),
        _build_gl1_panel(metrics),
        gl1_gl2_edges,
        _build_gl2_panel(metrics),
        gl2_gcea_edges,
        _build_gcea_panel(metrics),
        dram_edges,
        _build_dram_panel(metrics),
    )

    console.print(main_layout)
    console.print()
    console.print(build_legend(include_stall=True))
    console.print()

    if show_debug:
        console.print("[dim]Architecture Notes:[/dim]")
        console.print("  TCP (Texture Cache Pipe): L0 vector cache for VMEM operations")
        console.print(
            "  LDS (Local Data Share): On-CU scratchpad, NO GL1 Cache connection"
        )
        console.print("  SQC (Sequencer Cache): ICache + DCache for scalar operations")
        console.print()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def plot_mem_chart(
    metric_dict: dict[str, Any],
    *,
    chart_title: str,
) -> str:
    """Plot the memory chart and return as string.

    ``metric_dict`` keys should match ``0300_memory_chart.yaml`` (gfx115x).
    ``chart_title``: full heading line printed above the diagram.

    Note: unlike gfx9's ``plot_mem_chart``, this function has no
    ``normal_unit`` parameter because all callers supply ``chart_title``
    directly — there is no need for a fallback heading.
    """
    flat = normalize_mem_chart_metrics(metric_dict)
    buf = StringIO()
    console = Console(file=buf, force_terminal=True, width=200, height=80)
    create_mem_chart_diagram(
        flat,
        console,
        show_debug=False,
        chart_title=chart_title,
    )
    return buf.getvalue()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    arg_parser = argparse.ArgumentParser(description="Memory Chart - CLI Visualization")
    arg_parser.add_argument("--data", "-d", help="JSON file with metrics data")
    arg_parser.add_argument("--debug", action="store_true", help="Show debug info")
    arg_parser.add_argument("--arch", default="gfx1151", help="Architecture name")
    arg_parser.add_argument("--norm", default="per_kernel", help="Normalization unit")
    arg_parser.add_argument("--txt", "-t", help="Output to plain text file")
    arg_parser.add_argument("--svg", help="Output to SVG file")
    args = arg_parser.parse_args()

    if args.data:
        with pathlib.Path(args.data).open(encoding="utf-8") as f:
            metric_dict = normalize_mem_chart_metrics(json.load(f))
    else:
        metric_dict = dict(DEFAULT_SAMPLE_METRICS)

    heading = format_mem_chart_heading(args.norm)

    if args.txt:
        buf = StringIO()
        console = Console(
            file=buf,
            force_terminal=True,
            width=200,
            height=80,
            no_color=True,
        )
        create_mem_chart_diagram(
            metric_dict,
            console,
            args.debug,
            chart_title=heading,
        )
        plain = strip_ansi(buf.getvalue())
        with pathlib.Path(args.txt).open("w", encoding="utf-8") as f:
            f.write(plain)
        print(f"Output written to {args.txt}")
    elif args.svg:
        svg_console = Console(
            file=StringIO(),
            force_terminal=True,
            width=200,
            height=80,
            record=True,
        )
        create_mem_chart_diagram(
            metric_dict,
            svg_console,
            args.debug,
            chart_title=heading,
        )
        svg_output = svg_console.export_svg(title=heading)
        with pathlib.Path(args.svg).open("w", encoding="utf-8") as f:
            f.write(svg_output)
        print(f"SVG saved to {args.svg}")
    else:
        buf = StringIO()
        console = Console(file=buf, force_terminal=True, width=200, height=80)
        create_mem_chart_diagram(
            metric_dict,
            console,
            args.debug,
            chart_title=heading,
        )
        print(buf.getvalue())


if __name__ == "__main__":
    main()
