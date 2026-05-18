"""Small reusable matplotlib helpers shared across analyses."""
from __future__ import annotations

from typing import Optional


def apply_axis_scales(ax, *, log_x: bool = False, log_y: bool = False,
                      min_x: Optional[float] = None, min_y: Optional[float] = None) -> None:
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    if min_x is not None:
        ax.set_xlim(left=min_x)
    if min_y is not None:
        ax.set_ylim(bottom=min_y)


def plot_suffix(*, log_x: bool = False, log_y: bool = False) -> str:
    parts: list[str] = []
    if log_x:
        parts.append("logx")
    if log_y:
        parts.append("logy")
    return ("_" + "_".join(parts)) if parts else ""


def clamp(val, lo, hi):
    return max(lo, min(hi, val))
