# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Shared helpers for memory chart renderers (gfx9, gfx11)."""

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
    """Format a metric value with unit. Returns 'N/A' for None."""
    if value is None:
        return "N/A"
    if isinstance(value, str):
        try:
            value = float(value)
        except (ValueError, TypeError):
            return value
    if unit == "%":
        return f"{value:.{precision}f}%"
    if unit in ("GB/s", "Bytes/s"):
        return format_bw_human_readable(value, unit, precision)
    return f"{value:.{precision}f}{unit}"


def format_sci(value: Union[int, float, str, None], precision: int = 2) -> str:
    """Format as integer (<1000) or scientific notation (>=1000)."""
    if value is None:
        return "N/A"
    try:
        value = float(value)
    except (ValueError, TypeError):
        return "N/A"
    if abs(value) < 1000:
        return f"{int(value)}"
    return f"{value:.{precision}e}"


def metric_line(
    label: str,
    value: Any,  # noqa: ANN401
    unit: str = "%",
    color: str = "bright_green",
) -> str:
    """Rich markup line: 'label value_with_unit' in *color*."""
    return f"{label} [{color}]{format_value(value, unit)}[/{color}]"


def bar(pct: Optional[float], w: int = 10) -> str:
    """Unicode progress bar. None/invalid → empty bar."""
    if pct is None:
        return "░" * w
    try:
        pct = float(pct)
    except (ValueError, TypeError):
        return "░" * w
    filled = int(w * min(100, max(0, pct)) / 100)
    return "█" * filled + "░" * (w - filled)


def safe_float_sum(
    *values: Union[int, float, str, None],
) -> Optional[float]:
    """Sum non-None numeric values. Returns None if all invalid."""
    total = 0.0
    any_valid = False
    for v in values:
        if v is not None:
            try:
                total += float(v)
                any_valid = True
            except (ValueError, TypeError):
                pass
    return total if any_valid else None


def scale_or_none(value: Any, factor: float) -> Optional[float]:  # noqa: ANN401
    """Return ``factor * value`` if *value* is not None, else None."""
    if value is None:
        return None
    try:
        return factor * float(value)
    except (ValueError, TypeError):
        return None


def fmt_edge(
    label: str,
    value: Any,  # noqa: ANN401
    width: int = 7,
) -> str:
    """Format edge label with optional scientific-notation value."""
    label_str = f"{label:<{width}}"
    if value is not None:
        value_str = f": {format_sci(value):>7}"
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


def build_legend(include_stall: bool = False) -> str:
    """Build the color legend string."""
    parts = [
        f"[dim]Legend:[/dim] "
        f"[{COLORS['read']}]<----[/{COLORS['read']}] Read  "
        f"[{COLORS['write']}]---->[/{COLORS['write']}] Write  "
        f"[{COLORS['atomic']}]<--->[/{COLORS['atomic']}] Atomic  "
        f"[{COLORS['util']}]█[/{COLORS['util']}] Util  "
        f"[{COLORS['hit']}]█[/{COLORS['hit']}] Hit%"
    ]
    if include_stall:
        parts.append(f"  [{COLORS['stall']}]█[/{COLORS['stall']}] Stall")
    return "".join(parts)


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
    """Rich color by utilization: green(low) → yellow(mid) → red(high)."""
    if value is None or peak is None or peak <= 0:
        return default
    try:
        pct = 100.0 * float(value) / float(peak)
    except (ValueError, TypeError, ZeroDivisionError):
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
