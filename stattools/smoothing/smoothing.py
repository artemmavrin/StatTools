"""Scatterplot smoothing (AKA fitting a curve through a scatterplot)."""

import abc

from ..ensemble.bagging import BaggingRegressor
from ..generic.predictors import Regressor
from ..visualization import func_plot


class ScatterplotSmoother(Regressor, metaclass=abc.ABCMeta):
    """Abstract base class for scatterplot smoothers."""

    def fit_plot(self, x_min=None, x_max=None, num=500, ax=None, **kwargs):
        """Plot the scatterplot smoother curve.

        Parameters
        ----------
        x_min : float, optional
            Smallest explanatory variable observation. If not provided, grabs
            the smallest x value from the given axes.
        x_max : float, optional
            Biggest explanatory variable observation. If not provided, grabs the
            biggest x value from the given axes.
        num : int, optional
            Number of points to plot.
        ax : matplotlib.axes.Axes, optional
            The axes on which to draw the plot.
        kwargs : dict, optional
            Additional keyword arguments to pass to plot()

        Returns
        -------
        The matplotlib.axes.Axes object on which the plot was drawn.
        """
        return func_plot(func=self.predict, x_min=x_min, x_max=x_max, num=num,
                         ax=ax, **kwargs)


class BaggingSmoother(ScatterplotSmoother, BaggingRegressor):
    """Use bagging (bootstrap aggregating) to potentially improve a smoother."""

    def __init__(self, base: type, *args, **kwargs):
        """Initialize a BaggingSmoother.

        Parameters
        ----------
        base : ScatterplotSmoother subclass
            The underlying smoother type.
        args : sequence, optional
            Positional arguments for the estimator constructor.
        kwargs : dict, optional
            Keyword arguments for the estimator constructor.
        """
        if not issubclass(base, ScatterplotSmoother):
            raise TypeError(
                "Parameter 'base' must be a ScatterplotSmoother subclass.")

        super(BaggingSmoother, self).__init__(base, *args, **kwargs)
