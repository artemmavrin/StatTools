"""Implements quantile-quantile plots"""

import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt


def qqplot(y, qfunc=None, qfunc_kwargs=None, ax=None, ax_kwargs=None, *args,
           **kwargs):
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
    qfunc_kwargs: dict, optional
        Additional keyword arguments for the quantile function.
    ax: matplotlib axis, optional
        The axis on which to draw the QQ plot.
        If this is not specified, the current axis will be used.
    ax_kwargs: dict, optional
        Additional keyword arguments to pass to matplotlib.pyplot.gca() if
        axis is specified.
    args: sequence
        Additional positional arguments to pass to matplotlib.pyplot.scatter().
    kwargs: dict
        Additional keyword arguments to pass to matplotlib.pyplot.scatter().
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
        if qfunc_kwargs is None:
            qfunc_kwargs = {}
        x = qfunc(p, **qfunc_kwargs)

    if "c" not in kwargs:
        # Default color represents the distance to the diagonal line
        kwargs["c"] = np.abs(x - y)
    if "cmap" not in kwargs:
        # Default color map
        kwargs["cmap"] = plt.get_cmap("coolwarm")

    if ax is None:
        if ax_kwargs is None:
            ax_kwargs = {}
        ax = plt.gca(**ax_kwargs)

    ax.scatter(x, y, zorder=2, *args, **kwargs)
    ax.set_xlabel("Theoretical quantiles")
    ax.set_ylabel("Observed quantiles")

    # Diagonal line for comparison
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    bounds = (min(x_min, y_min), max(x_max, y_max))
    ax.plot(bounds, bounds, c="black", zorder=1)
    ax.set_xlim(bounds)
    ax.set_ylim(bounds)
    ax.set_aspect("equal")
