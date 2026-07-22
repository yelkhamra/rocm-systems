# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Shared helpers for memory chart renderers (gfx9, gfx11)."""

import math
import re
from dataclasses import dataclass
from typing import Any, Optional, Union

from utils.utils_analysis import format_bw_human_readable

COLORS = {
    "kernel": "green",
    "block": "blue",
    "tcp": "cyan",
    "lds": "magenta",
    "sqc": "yellow",
    "read": "bright_cyan",
    "write": "bright_yellow",
    "atomic": "bright_magenta",
    "util": "bright_green",
    "hit": "yellow",
    "stall": "indian_red",
    "bw": "bright_cyan",
}


def format_value(
    value: Union[int, float, str, None], unit: str = "", precision: int = 1
) -> str:
    """Format a metric value with unit. Returns 'N/A' for None/NaN/invalid."""
    if value is None:
        return "N/A"
    if unit in ("GB/s", "Bytes/s"):
        return format_bw_human_readable(value, unit, precision)
    try:
        numeric = float(value)
    except (ValueError, TypeError):
        return "N/A"
    if math.isnan(numeric):
        return "N/A"
    if unit == "%":
        return f"{numeric:.{precision}f}%"
    return f"{numeric:.{precision}f}{unit}"


def format_scientific(value: Union[int, float, str, None], precision: int = 2) -> str:
    """Format as rounded integer (<1000) or scientific notation (>=1000)."""
    if value is None:
        return "N/A"
    try:
        numeric = float(value)
    except (ValueError, TypeError):
        return "N/A"
    if math.isnan(numeric):
        return "N/A"
    if abs(numeric) < 1000:
        return str(round(numeric))
    return f"{numeric:.{precision}e}"


def colored(text: str, color: str) -> str:
    """Wrap *text* in Rich color markup tags."""
    return f"[{color}]{text}[/{color}]"


def metric_line(
    label: str,
    value: Any,  # noqa: ANN401
    unit: str = "%",
    color: str = "bright_green",
) -> str:
    """Rich markup line: 'label value_with_unit' in *color*."""
    return f"{label} {colored(format_value(value, unit), color)}"


def progress_bar(percent: Optional[float], width: int = 10) -> str:
    """Unicode progress bar. None/NaN/invalid -> empty bar."""
    if percent is None:
        return "░" * width
    try:
        numeric = float(percent)
    except (ValueError, TypeError):
        return "░" * width
    if math.isnan(numeric):
        return "░" * width
    filled = int(width * min(100, max(0, numeric)) / 100)
    return "█" * filled + "░" * (width - filled)


def safe_float_sum(
    *values: Union[int, float, str, None],
) -> Optional[float]:
    """Sum numeric values, skipping None/NaN/unparseable. None if all invalid."""
    terms: list[float] = []
    for value in values:
        try:
            numeric = float(value)  # type: ignore[arg-type]
        except (ValueError, TypeError):
            continue
        if not math.isnan(numeric):
            terms.append(numeric)
    return sum(terms) if terms else None


def scale_or_none(value: Any, factor: float) -> Optional[float]:  # noqa: ANN401
    """Return ``factor * value`` if *value* is valid, else None."""
    if value is None:
        return None
    try:
        result = factor * float(value)
    except (ValueError, TypeError):
        return None
    if math.isnan(result):
        return None
    return result


def format_edge(
    label: str,
    value: Any,  # noqa: ANN401
    width: int = 7,
) -> str:
    """Format edge label with optional scientific-notation value."""
    label_str = f"{label:<{width}}"
    if value is not None:
        value_str = f": {format_scientific(value):>7}"
    else:
        value_str = ""
    return f"{label_str}{value_str}"


def make_arrows(length: int) -> dict[str, str]:
    """Build arrow dict with left/right/both keys of given length."""
    return {
        "left": "<" + "-" * (length - 1),
        "right": "-" * (length - 1) + ">",
        "both": "<" + "-" * (length - 2) + ">",
    }


def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from *text*."""
    return re.sub(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])", "", text)


def format_mem_chart_heading(
    normal_unit: str,
    *,
    panel_id: int = 300,
    section_label: str = "Memory Chart",
) -> str:
    """Build heading: '{panel_id//100}. {label} (Normalization: {unit})'."""
    section = max(0, int(panel_id)) // 100
    return f"{section}. {section_label} (Normalization: {normal_unit})"


_LEGEND_ENTRIES: tuple[tuple[str, str, str], ...] = (
    ("<----", "Read", "read"),
    ("---->", "Write", "write"),
    ("<--->", "Atomic", "atomic"),
    ("█", "Util", "util"),
    ("█", "Hit%", "hit"),
)

_STALL_ENTRY: tuple[str, str, str] = ("█", "Stall", "stall")


def build_legend(include_stall: bool = False) -> str:
    """Build the color legend string from ``_LEGEND_ENTRIES``."""
    entries = list(_LEGEND_ENTRIES)
    if include_stall:
        entries.append(_STALL_ENTRY)
    items = [
        f"{colored(symbol, COLORS[color_key])} {label}"
        for symbol, label, color_key in entries
    ]
    return f"[dim]Legend:[/dim] {'  '.join(items)}"


# ---------------------------------------------------------------------------
# BW color-coding (NCU-style % of peak)
# ---------------------------------------------------------------------------


@dataclass
class PeakBandwidths:
    """Theoretical peak bandwidths (GB/s) per memory level."""

    hbm: Optional[float] = None
    l2: Optional[float] = None
    vl1d: Optional[float] = None
    lds: Optional[float] = None
    sl1d: Optional[float] = None
    l1i: Optional[float] = None


def bw_color(
    value: Optional[float],
    peak: Optional[float],
    default: str = "white",
) -> str:
    """Rich color by utilization: green(low) -> yellow(mid) -> red(high)."""
    if value is None or peak is None or peak <= 0:
        return default
    try:
        pct = 100.0 * float(value) / float(peak)
    except (ValueError, TypeError):
        return default
    if math.isnan(pct):
        return default
    if pct < 20:
        return "dim green"
    if pct < 40:
        return "green"
    if pct < 60:
        return "yellow"
    if pct < 80:
        return "bright_yellow"
    return "red"
