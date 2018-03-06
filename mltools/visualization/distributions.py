"""Implements plots for visualizing probability distributions."""

import numbers

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st

from .abline import abline


def _rug_plot(data, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    rug_params = {"ymin": 0, "ymax": 0.1, "c": "gray", "alpha": 0.5,
                  "zorder": 3}
    rug_params.update(kwargs)
    for x in data:
        ax.axvline(x, **rug_params)


def ecdf_plot(data, cdf=None, rug=False, cb=False, alpha=0.05, ax=None,
              cdf_kwargs=None, rug_kwargs=None, cb_kwargs=None, **kwargs):
    """Draw an empirical cumulative distribution function.

    Parameters
    ----------
    data: array-like
        The data sample.
    cdf: callable or str or scipy.stats.rv_continuous object, optional
        If this is not None, plot a true cumulative distribution function behind
        the empirical distribution function. If this is a function, it is used
        as the CDF. If this is a scipy.stats.rv_continuous object, a continuous
        distribution will be fitted and its CDF will be used. If this is a
        string, it should be the name of a scipy.stats.rv_continuous
        distribution.
    rug: bool, optional
        Indicate whether to draw a rug plot of the data.
    cb: bool, optional
        Indicate whether to plot a non-parametric confidence band based on the
        DKW (Dvoretzky–Kiefer–Wolfowitz) inequality.
    alpha: float, optional
        Specify the significance level for the confidence band.
    cdf_kwargs: dict, optional
        Keyword arguments to pass to the function plotting the CDF. Ignored if
        `cdf` is None.
    rug_kwargs: dict, optional
        Keyword arguments to pass to the function plotting the rug plot. Ignored
        if `rug` is False.
    cb_kwargs: dict, optional
        Keyword arguments to pass to the function plotting the confidence bands.
        Ignored if `cdf` is None.
    ax: matplotlib axis, optional
        The axis on which to draw the QQ plot.
        If this is not specified, the current axis will be used.
    kwargs: dict, optional
        Additional keyword arguments to pass to the plot function when drawing
        the ECDF.

    Returns
    -------
    The axis on which the line was drawn.
    """
    # Validate data
    if np.ndim(data) != 1:
        raise ValueError("Data must be 1-dimensional")

    # Compute the ECDF
    data = np.asarray(data)
    points, counts = np.unique(data, return_counts=True)
    size = 2 * len(points) - 1
    x_ecdf = np.zeros(size)
    y_ecdf = np.zeros(size)

    x_ecdf[0] = points[0]
    y_ecdf[0] = counts[0] / data.size
    for i in range(len(points) - 1):
        x_ecdf[2 * i + 1] = points[i + 1]
        x_ecdf[2 * i + 2] = points[i + 1]
        y_ecdf[2 * i + 1] = y_ecdf[2 * i]
        y_ecdf[2 * i + 2] = y_ecdf[2 * i] + counts[i + 1] / data.size

    if ax is None:
        ax = plt.gca()

    # Plot the ECDF
    ecdf_params = {"zorder": 2, "label": "Empirical CDF"}
    ecdf_params.update(kwargs)
    ax.plot(x_ecdf, y_ecdf, **ecdf_params)

    if cdf is not None:
        # Determine distribution
        if isinstance(cdf, str):
            if (hasattr(st, cdf) and
                    isinstance(getattr(st, cdf), st.rv_continuous)):
                dist = getattr(st, cdf)
                cdf = dist(*dist.fit(data)).cdf
                label = "Estimated CDF"
            else:
                raise ValueError(f"Cannot resolve distribution name {cdf}")
        elif isinstance(cdf, st.rv_continuous):
            cdf = cdf(*cdf.fit(data)).cdf
            label = "Estimated CDF"
        elif callable(cdf):
            # Theoretical CDF was specified
            label = "Theoretical CDF"
        else:
            raise TypeError("Parameter 'cdf' must be callable.")

        x_min, x_max = ax.get_xlim()

        # Plot the CDF
        x = np.linspace(x_min, x_max, num=200)
        cdf_params = {"ls": "--", "c": "r", "zorder": 1, "label": label}
        if cdf_kwargs is not None:
            cdf_params.update(cdf_kwargs)
        ax.plot(x, cdf(x), **cdf_params)

        # Reset x axis bounds
        ax.set_xlim((x_min, x_max))

    if cb:
        # Validate significance level `alpha`
        if not isinstance(alpha, numbers.Real):
            raise TypeError("Parameter 'alpha' must be a float")
        alpha = float(alpha)
        if alpha < 0 or alpha > 1:
            raise ValueError("Parameter 'alpha' must be in the range [0, 1]")

        # Compute upper and lower confidence bounds
        e = np.sqrt(np.log(2 / alpha) / (2 * len(data)))
        lower = [max(y_ - e, 0) for y_ in y_ecdf]
        upper = [min(y_ + e, 1) for y_ in y_ecdf]

        # Plot confidence band
        cb_params = {"c": "gray", "ls": "--", "zorder": 1.5}
        if cb_kwargs is not None:
            cb_params.update(cb_kwargs)
        label = f"{100 * (1 - alpha):.0f}% Confidence Band"
        ax.plot(x_ecdf, upper, label=label, **cb_params)
        ax.plot(x_ecdf, lower, **cb_params)

    if rug:
        if rug_kwargs is None:
            rug_kwargs = {}
        _rug_plot(data, ax=ax, **rug_kwargs)

    ax.set_xlabel("Value")
    ax.set_ylabel("Cumulative Probability")

    return ax


def qq_plot(data, quantile=None, ax=None, diag=True, rug=False, square=False,
            diag_kwargs=None, rug_kwargs=None, **kwargs):
    """Draw a QQ plot comparing a data sample to a theoretical distribution.

    Parameters
    ----------
    data: array-like
        The data sample.
    quantile: callable or scipy.stats.rv_continuous object, or str optional
        The quantile function ("inverse" cumulative distribution function) of
        the theoretical distribution the data is being compared to.
        If this is not specified, the normal distribution quantile function will
        be used. If this is a scipy.stats.rv_continuous object, then a
        continuous distribution will be fit to the data, and the corresponding
        distribution's quantile function will be used. If this is a string, then
        it should be the name of a scipy.stats.rv_continuous distribution.
    ax: matplotlib axis, optional
        The axis on which to draw the QQ plot.
        If this is not specified, the current axis will be used.
    diag: bool, optional
        Indicate whether to draw a diagonal line indicating an ideal fit.
    rug: bool, optional
        Indicate whether to draw a rug plot of the data.
    square: bool, optional
        Indicate whether to set a square aspect ratio with equal x and y limits.
    diag_kwargs: dict, optional
        Keyword arguments to pass to the function plotting the diagonal line.
        Ignored if `diag` is False.
    rug_kwargs: dict, optional
        Keyword arguments to pass to the function plotting the rug plot. Ignored
        if `rug` is False.
    kwargs: dict
        Additional keyword arguments to pass to the "scatter" function when
        drawing the scatter plot of the order statistics against the true
        quantiles.

    Returns
    -------
    The axis on which the line was drawn.
    """
    # Validate data
    if np.ndim(data) != 1:
        raise ValueError("Data must be 1-dimensional")

    # Order statistics of the data sample
    data = np.sort(data)
    n = data.size

    # Percentiles for the theoretical distribution
    p = (np.arange(n) + 0.5) / n

    if quantile is None:
        # Normal QQ plot by default
        quantiles = st.norm(loc=data.mean(), scale=data.std()).ppf(p)
    elif isinstance(quantile, str):
        # Fit a continuous distribution to the data
        if (hasattr(st, quantile) and
                isinstance(getattr(st, quantile), st.rv_continuous)):
            dist = getattr(st, quantile)
            quantile = dist(*dist.fit(data)).ppf
            quantiles = quantile(p)
        else:
            raise ValueError(f"Cannot resolve distribution name {quantile}")
    elif isinstance(quantile, st.rv_continuous):
        # Fit a continuous distribution to the data
        quantiles = quantile(*quantile.fit(data)).ppf(p)
    elif callable(quantile):
        quantiles = quantile(p)
    else:
        t = type(quantile)
        raise TypeError(f"Invalid type for parameter 'quantile': {t}")

    if ax is None:
        ax = plt.gca()

    # Default QQ plot scatter plot parameters. The default color represents the
    # distance to the diagonal line
    scatter_params = {"zorder": 2, "c": np.abs(quantiles - data),
                      "cmap": plt.get_cmap("coolwarm")}
    scatter_params.update(kwargs)
    ax.scatter(data, quantiles, **scatter_params)

    if square:
        # Setup equal x and y axes
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        bounds = (min(x_min, y_min), max(x_max, y_max))
        ax.set_xlim(bounds)
        ax.set_ylim(bounds)
        ax.set_aspect("equal")

    if diag:
        # Diagonal line for quantile comparison
        diag_params = {"ls": "--", "c": "k", "zorder": 1}
        if diag_kwargs is not None:
            diag_params.update(diag_kwargs)
        abline(0, 1, ax=ax, **diag_params)

    if rug:
        if rug_kwargs is None:
            rug_kwargs = {}
        _rug_plot(data, ax=ax, **rug_kwargs)

    ax.set_xlabel("Observed Values")
    ax.set_ylabel("Theoretical Quantiles")

    return ax
