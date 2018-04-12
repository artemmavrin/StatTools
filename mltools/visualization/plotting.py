"""Functions for plotting curves."""

import matplotlib.pyplot as plt
import numpy as np

from ..utils.validation import validate_func


def abline(a, b, x_min=None, x_max=None, ax=None, **kwargs):
    """Draw a line with a prescribed slope and intercept.

    This is a minimal Python port of R's abline().

    Parameters
    ----------
    a : float
        Intercept.
    b : float
        Slope.
    x_min : float, optional
        Smallest x value. If not provided, grabs the smallest x value from the
        given axes.
    x_max : float, optional
        Biggest x value. If not provided, grabs the biggest x value from the
        given axes.
    ax : matplotlib.axes.Axes, optional
        The axis on which to draw the line.
        If this is not specified, the current axis will be used.
    kwargs : dict, optional
        Additional keyword arguments to pass to the "plot" function when drawing
        the line.

    Returns
    -------
    The axis on which the line was drawn.
    """
    if ax is None:
        ax = plt.gca()

    # Get bounds if not provided
    y_min, y_max = ax.get_ylim()
    reset_y = False
    if x_min is None or x_max is None:
        x_min, x_max = ax.get_xlim()
        reset_y = True

    # Plot the line
    x = (x_min, x_max)
    y = (a + b * x_min, a + b * x_max)
    ax.plot(x, y, **kwargs)

    # Set the axes bounds
    ax.set(xlim=(x_min, x_max))
    if reset_y:
        ax.set(ylim=(y_min, y_max))

    return ax


def func_plot(func, x_min=None, x_max=None, num=50, ax=None, **kwargs):
    """Plot a function.

    Parameters
    ----------
    func : callable or str
        The function to plot. If this is a string, it must be the name of a
        NumPy array method.
    x_min : float, optional
        Smallest x value. If not provided, grabs the smallest x value from the
        given axes.
    x_max : float, optional
        Biggest x value. If not provided, grabs the biggest x value from the
        given axes.
    num : int, optional
        Number of points to plot.
    ax : matplotlib.axes.Axes, optional
        The axis on which to draw the line.
        If this is not specified, the current axis will be used.
    kwargs : dict, optional
        Additional keyword arguments to pass to the "plot" function when drawing
        the function.

    Returns
    -------
    The axis on which the line was drawn.
    """
    func = validate_func(func)

    if ax is None:
        ax = plt.gca()

    # Get bounds if not provided
    y_min, y_max = ax.get_ylim()
    reset_y = False
    if x_min is None or x_max is None:
        x_min, x_max = ax.get_xlim()
        reset_y = True

    # Plot the function
    x = np.linspace(x_min, x_max, num)
    try:
        y = func(x)
    except TypeError:
        y = np.asarray(list(map(func, x)))
    ax.plot(x, y, **kwargs)

    # Set the axes bounds
    ax.set(xlim=(x_min, x_max))
    if reset_y:
        ax.set(ylim=(y_min, y_max))

    return ax
