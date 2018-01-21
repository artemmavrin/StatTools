"""Implements quantile-quantile plots"""

import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt


def qqplot(x, inv_cdf=None, inv_cdf_kwargs=None, ax=None, ax_kwargs=None, *args,
           **kwargs):
    """Draw a QQ plot for a data sample.

    Parameters
    ----------
    x: array-like
        The data
    inv_cdf: callable, optional
        The inverse cumulative distribution function of the theoretical
        distribution the data is being compared to.
        If this is not specified, the normal distribution will be used.
    inv_cdf_kwargs: dict, optional
        Additional keyword arguments for the inverse cumulative distribution
        function.
    ax: matplotlib axis
        The axis on which to draw the QQ plot.
        If this is not specified, the current axis will be used.
    ax_kwargs: dict, optional
        Additional keyword arguments to pass to matplotlib.pyplot.gca()
    args: sequence
        Additional positional arguments to pass to matplotlib.pyplot.scatter().
    kwargs: dict
        Additional keyword arguments to pass to matplotlib.pyplot.scatter().
    """
    x = np.sort(x)
    n = x.size

    if inv_cdf is None:
        # Normal QQ plot by default
        y = st.norm.ppf((np.arange(n) + 0.5) / n, loc=x.mean(), scale=x.std())
    else:
        if inv_cdf_kwargs is None:
            inv_cdf_kwargs = {}
        y = inv_cdf((np.arange(n) + 0.5) / n, **inv_cdf_kwargs)

    if "c" not in kwargs:
        # Default color represents the distance to the diagonal line
        kwargs["c"] = np.sqrt((x - (x + y) / 2) ** 2 + (y - (x + y) / 2) ** 2)
    if "cmap" not in kwargs:
        # Default color map
        kwargs["cmap"] = plt.cm.coolwarm

    if ax is None:
        if ax_kwargs is None:
            ax_kwargs = {}
        ax = plt.gca(**ax_kwargs)

    ax.scatter(x, y, zorder=2, *args, **kwargs)

    # diagonal line for comparison
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    a = (min(x_min, y_min), max(x_max, y_max))
    ax.plot(a, a, c="black", zorder=1)
    ax.set_xlim(a)
    ax.set_ylim(a)
