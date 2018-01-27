"""Minimal Python port of R's "abline" function."""

import matplotlib.pyplot as plt


def abline(a, b, ax=None, **kwargs):
    """Draw a line with a prescribed slope and intercept.

    Parameters
    ----------
    a: float
        Intercept.
    b: float
        Slope.
    ax: matplotlib axis, optional
        The axis on which to draw the line.
        If this is not specified, the current axis will be used.
    kwargs: dict, optional
        Additional keyword arguments to pass to the "plot" function when drawing
        the line.

    Returns
    -------
    The axis on which the line was drawn.
    """
    if ax is None:
        ax = plt.gca()

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    x = (x_min, x_max)
    y = (a + b * x_min, a + b * x_max)
    ax.plot(x, y, **kwargs)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    return ax
