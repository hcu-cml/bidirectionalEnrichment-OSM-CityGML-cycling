"""
kde_utils.py
------------
Utility helpers for loading, fitting, caching and visualising the residual–distance
kernel–density estimate used by the HMM emission model. The plotting helper can
optionally overlay Gaussian baselines (half-normal) for comparison.
"""
from __future__ import annotations

import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional
from scipy.stats import gaussian_kde


def fit_kde(
    distances: np.ndarray,
    bandwidth: str | float = "silverman",
) -> gaussian_kde:
    """
    Fit a 1-D KDE on the given distance.

    Parameters
    ----------
    distances : np.ndarray
        distances (shape: (N,))
    bandwidth : {"scott", "silverman"} or float, optional
        Bandwidth rule or multiplier passed to `gaussian_kde`.

    Returns
    -------
    scipy.stats.gaussian_kde
    """
    distances = np.asarray(distances, dtype=float)
    return gaussian_kde(distances, bw_method=bandwidth)


def load_kde(
    dist_file: str | Path,
    bandwidth: str | float = "silverman",
) -> Optional[gaussian_kde]:
    """
    Load residual distances from dist_file and return a fitted KDE.

    Returns None if the distance file does not exist.
    """
    dist_file = Path(dist_file)
    if not dist_file.exists():
        return None

    distances = np.load(dist_file)
    kde = fit_kde(distances, bandwidth=bandwidth)

    return kde

def plot_kde(
    kde: gaussian_kde,
    distances: np.ndarray,
    max_x: float | None = None,
    bins: int = 30,
    show_halfnormal: bool = True,
    fixed_sigma: float | None = None,
    save_path: str | Path | None = None,
    dpi: int = 300,
) -> None:
    """
    Quick histogram + KDE overlay for sanity-checking the fitted model.

    Parameters
    ----------
    kde : gaussian_kde
        A KDE fitted on distances
    distances : np.ndarray
        Raw (non-negative) residual distances in metres.
    max_x : float | None
        Upper x-limit for plotting; if None, uses max(distances) or 10 m.
    bins : int
        Histogram bins.
    show_halfnormal : bool
        If True, overlay a half-normal baseline on [0, inf).
    fixed_sigma : float | None
        If provided and show_halfnormal is True, uses this sigma for the
        half-normal instead of the MLE estimate.
    save_path : str | Path | None
        If provided, saves the figure to this path (e.g., ".../kde_hist_gaussian.pdf").
    dpi : int
        Dots-per-inch when saving raster formats; ignored for vector PDFs.
    """
    import numpy as _np
    import matplotlib.pyplot as _plt
    from scipy.stats import halfnorm as _halfnorm, truncnorm as _truncnorm

    distances = _np.asarray(distances, dtype=float)
    distances = distances[_np.isfinite(distances)]
    distances = distances[distances > 0]  # strictly positive
    if distances.size == 0:
        raise ValueError("No positive distances to plot.")

    # x-axis support
    x_max = max_x if (max_x is not None) else max(10.0, distances.max())
    xs = _np.linspace(0.0, x_max, 400)

    # Figure & histogram
    _plt.figure(figsize=(9, 5))
    _plt.hist(distances, bins=bins, range=(0, x_max), density=True,
              alpha=0.45, edgecolor="black", label="Histogram")

    # KDE curve
    kde_y = kde(xs)
    _plt.plot(xs, kde_y, linewidth=2, label="KDE")

    # Half-normal baseline (recommended)
    if show_halfnormal:
        if fixed_sigma is not None and fixed_sigma > 0:
            sigma_hn = float(fixed_sigma)
            label_hn = f"Gaussian"
        else:
            # MLE for half-normal: sigma = sqrt(mean(d^2))
            sigma_hn = float(_np.sqrt(_np.mean(distances ** 2)))
            label_hn = f"Gaussian"
        hn_y = _halfnorm.pdf(xs, loc=0.0, scale=sigma_hn)
        _plt.plot(xs, hn_y, linestyle="--", linewidth=2, label=label_hn)

    _plt.title("Residual distance distribution: KDE vs Gaussian Distributions")
    _plt.xlabel("Distance (m)")
    _plt.ylabel("Density")
    _plt.legend()
    _plt.grid(True, alpha=0.25)
    _plt.tight_layout()

    if save_path is not None:
        from pathlib import Path as _Path
        out = _Path(save_path)
        _plt.savefig(out, dpi=dpi, bbox_inches="tight")

    _plt.show()
    
    