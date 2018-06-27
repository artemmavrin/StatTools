"""Implements the Kaplan-Meier estimator for non-parametric survival function
estimation.

References
----------
E. L. Kaplan and P. Meier. "Nonparametric estimation from incomplete
    observations". Journal of the American Statistical Association, Volume 53,
    Issue 282 (1958), 457--481. doi:10.2307/2281868
"""

import matplotlib.pyplot as plt
import numpy as np

from ..generic import Fittable
from ..utils import validate_samples


class KaplanMeier(Fittable):
    """Non-parametric survival function estimator for right-censored data.

    Properties
    ----------
    time : numpy.ndarray
        Vector of observed event times.
    event : numpy.ndarray
        Failure indicator (0=right-censored, 1=failure).
    survival : numpy.ndarray
        Estimate of the survival function at each observed failure time.
    """

    time: np.ndarray = None
    event: np.ndarray = None
    survival: np.ndarray = None

    # List of sorted distinct uncensored failure times
    _failure: np.ndarray = None

    # List of sorted distinct censored times
    _censor: np.ndarray = None

    def fit(self, time, event=None):
        """Fit the Kaplan-Meier estimator using Efron's "redistribution to the
        right" algorithm.

        Parameters
        ----------
        time : array-like, of shape (n,)
            Vector of observed event times.
        event : array-like, of shape (n,)
            Vector of 0's and 1's, 0 indicating a right-censored event, 1
            indicating a failure.

        Returns
        -------
        This KaplanMeier instance.

        References
        ----------
        Bradley Efron. "The two sample problem with censored data". Proceedings
            of the Fifth Berkeley Symposium on Mathematical Statistics and
            Probability, Volume 4 (1967), 831--853
            https://projecteuclid.org/euclid.bsmsp/1200513831
        """
        # Validate parameters
        if event is None:
            event = np.ones(np.shape(time))
        time, event = validate_samples(time, event, n_dim=1, equal_lengths=True)
        if any(t <= 0 for t in time):
            raise ValueError("Entries of parameter 'time' must be positive.")
        if any(x not in (0, 1) for x in event):
            raise ValueError("Entries of parameter 'event' must be 0 or 1.")

        # Sort the times in increasing order, putting failures before censored
        # times in the case of ties
        ind = np.lexsort((1 - event, time))
        self.time = time[ind]
        self.event = event[ind]
        n = len(self.time)

        # Assign uniform mass to each observation initially
        mass = np.repeat(1 / n, repeats=n)

        # Going from left to right, at each censored time, redistribute its mass
        # to all times to the right
        for i in range(n - 1):
            if self.event[i] == 0:
                mass[(i + 1):] += mass[i] / len(mass[(i + 1):])
                mass[i] = 0

        # List of distinct uncensored event times
        self._failure = np.unique(self.time[self.event == 1])
        k = len(self._failure)

        # Compute the Kaplan-Meier estimate at each uncensored event time
        self.survival = np.ndarray(shape=(k,), dtype=np.float_)
        prob = 1.0
        for i in range(k):
            prob -= np.sum(mass[self.time == self._failure[i]])
            self.survival[i] = prob

        # Also grab the distinct censored event times
        self._censor = np.unique(self.time[self.event == 0])

        self.fitted = True
        return self

    def plot(self, ax=None, marker="+", **kwargs):
        """Plot the Kaplan-Meier survival curve.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            The axes on which to draw the line.
            If this is not specified, the current axis will be used.
        marker : str, optional
            Type of marker to use to indicate censored events. Can be None.
        kwargs : dict, optional
            Additional keyword arguments to pass to the "plot" function.

        Returns
        -------
        The matplotlib.axes.Axes object on which the curve was drawn.
        """
        if not self.fitted:
            raise self.unfitted_exception

        if ax is None:
            ax = plt.gca()

        # Plot the survival curve
        x = np.concatenate(([0.0], self._failure, [self.time[-1]]))
        y = np.concatenate(([1.0], self.survival, [self.survival[-1]]))
        params = dict(where="post", label="Kaplan-Meier estimate")
        params.update(kwargs)
        p = ax.step(x, y, **params)

        # Mark the censored times
        if marker is not None:
            color = p[0].get_color()
            i = 0
            for t in self._censor:
                if i >= len(self.survival):
                    ax.plot(t, self.survival[-1], marker=marker, color=color,
                            markeredgewidth=1)
                else:
                    while i < len(self.survival) and self._failure[i] <= t:
                        i += 1
                    ax.plot(t, self.survival[i - 1], marker=marker, color=color,
                            markeredgewidth=1)

        # Configure axes
        ax.set(xlabel="Time", ylabel="Survival Probability")
        ax.autoscale(enable=True, axis="x")
        ax.set(xlim=(0, None))
        ax.set(ylim=(0, None))

        return ax
