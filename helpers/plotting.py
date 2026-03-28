from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib as mpl
# Always render plots headlessly so verification works on machines without a GUI.
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PALETTE = {
    "ink": "#183642",
    "primary": "#1f5b75",
    "secondary": "#bc5b2c",
    "accent": "#8f3b76",
    "success": "#136f63",
    "warning": "#9a6700",
    "danger": "#a11d33",
    "neutral": "#5f5b56",
    "soft": "#9ca3af",
    "background": "#f7f7f5",
    "axes": "#fcfcfb",
    "grid": "#d7d2ca",
}


@dataclass(frozen=True)
class PlotConfig:
    dpi: int = 220
    animation_dpi: int = 100
    figure_facecolor: str = PALETTE["background"]
    axes_facecolor: str = PALETTE["axes"]
    title_size: float = 12.0
    label_size: float = 10.5
    font_size: float = 10.0


DEFAULT_PLOT_CONFIG = PlotConfig()


def configure_matplotlib(plot_config: PlotConfig | None = None) -> None:
    """Apply a consistent, reviewer-friendly plotting style."""

    config = plot_config or DEFAULT_PLOT_CONFIG
    mpl.rcParams.update(
        {
            "figure.facecolor": config.figure_facecolor,
            "axes.facecolor": config.axes_facecolor,
            "savefig.facecolor": config.figure_facecolor,
            "axes.edgecolor": "#3d3a37",
            "axes.grid": True,
            "grid.alpha": 0.45,
            "grid.color": PALETTE["grid"],
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": config.title_size,
            "axes.labelsize": config.label_size,
            "font.size": config.font_size,
            "font.family": "DejaVu Sans",
            "legend.frameon": False,
        }
    )


def save_figure(figure: plt.Figure, output_path: Path, *, dpi: int | None = None) -> None:
    """Persist a matplotlib figure and close over the required directory setup."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi or DEFAULT_PLOT_CONFIG.dpi, bbox_inches="tight")


def close_and_save(figure: plt.Figure, output_path: Path, *, dpi: int | None = None) -> None:
    """Save and immediately close a figure."""

    save_figure(figure, output_path, dpi=dpi)
    plt.close(figure)


def set_equal_3d_axes(axis: Any, points: np.ndarray) -> None:
    """Set equal scaling on a 3D axis using the provided point cloud."""

    points = np.asarray(points, dtype=float)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * max(np.max(maxs - mins), 1.0)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)


def add_note(axis: Any, text: str, *, x: float = 0.01, y: float = 0.99) -> None:
    """Add a compact explanatory note box inside an axis."""

    axis.text(
        x,
        y,
        text,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=9.2,
        family="monospace",
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#f3eee7", "edgecolor": "#d5c8b8"},
    )
