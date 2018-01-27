"""Implements quantile-quantile plots"""

import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

from .abline import abline


def qqplot(y, qfunc=None, ax=None, **kwargs):
    """Draw a QQ plot comparing a data sample to a theoretical distribution.

    Parameters
    ----------
    y: array-like
        The data sample.
    qfunc: callable, optional
        The quantile function ("inverse" cumulative distribution function) of
        the theoretical distribution the data is being compared to.
        If this is not specified, the normal distribution quantile function will
        be used.
    ax: matplotlib axis, optional
        The axis on which to draw the QQ plot.
        If this is not specified, the current axis will be used.
    kwargs: dict
        Additional keyword arguments to pass to the "scatter" function when
        drawing the scatter plot of the order statistics against the true
        quantiles.

    Returns
    -------
    The axis on which the line was drawn.
    """
    # Order statistics of the data sample
    y = np.sort(y)
    n = y.size

    # Percentiles for the theoretical distribution
    p = (np.arange(n) + 0.5) / n

    if qfunc is None:
        # Normal QQ plot by default
        x = st.norm.ppf(p, loc=y.mean(), scale=y.std())
    else:
        x = qfunc(p)

    if "c" not in kwargs:
        # Default color represents the distance to the diagonal line
        kwargs["c"] = np.abs(x - y)
    if "cmap" not in kwargs:
        # Default color map
        kwargs["cmap"] = plt.get_cmap("coolwarm")

    if ax is None:
        ax = plt.gca()

    ax.scatter(x, y, zorder=2, **kwargs)
    ax.set_xlabel("Theoretical quantiles")
    ax.set_ylabel("Observed quantiles")

    # Setup equal x and y axes
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    bounds = (min(x_min, y_min), max(x_max, y_max))
    ax.set_xlim(bounds)
    ax.set_ylim(bounds)
    ax.set_aspect("equal")

    # Diagonal line for quantile comparison
    abline(0, 1, ax=ax, ls="--", c="k", zorder=1)

    return ax
