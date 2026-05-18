"""Paper-ready matplotlib styling: Times serif, large fonts, PDF, no titles.

Usage at the top of any plotting script's main():

    from evo_replay.paper_style import apply
    apply()

This:
  - sets serif/Times rcParams + bigger axis labels/ticks/legend
  - embeds TrueType in PDFs (pdf.fonttype = 42)
  - replaces ``Axes.set_title`` / ``Figure.suptitle`` with no-ops
    (existing code can keep calling them; they just don't render)
  - patches ``Figure.savefig`` so any ``*.png`` argument is rewritten to
    ``*.pdf`` — so existing aggregators emit PDFs without code edits
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.figure
import matplotlib.axes


_PAPER_RCPARAMS = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "font.size": 18,
    "axes.labelsize": 18,
    "axes.titlesize": 18,            # ignored once titles are disabled
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 14,
    "figure.titlesize": 18,
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "lines.linewidth": 1.8,
    "lines.markersize": 6,
    "axes.grid": True,
    "axes.grid.which": "major",
    "grid.alpha": 0.30,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
}

_applied = False


def apply(disable_titles: bool = True, force_pdf: bool = True) -> None:
    global _applied
    if _applied:
        return
    _applied = True

    matplotlib.rcParams.update(_PAPER_RCPARAMS)

    if disable_titles:
        matplotlib.axes.Axes.set_title = lambda self, *a, **k: None  # type: ignore[method-assign]
        matplotlib.figure.Figure.suptitle = lambda self, *a, **k: None  # type: ignore[method-assign]

    if force_pdf:
        _orig_savefig = matplotlib.figure.Figure.savefig

        def _savefig_pdf(self, fname, *args, **kwargs):
            if isinstance(fname, (str, Path)):
                p = Path(fname)
                if p.suffix.lower() == ".png":
                    fname = p.with_suffix(".pdf")
            return _orig_savefig(self, fname, *args, **kwargs)

        matplotlib.figure.Figure.savefig = _savefig_pdf  # type: ignore[method-assign]
