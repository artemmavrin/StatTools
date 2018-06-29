"""Simulate sample paths of a Poisson process."""

import matplotlib.pyplot as plt
import numpy as np

from ..utils.validation import validate_float, validate_int


class PoissonProcess(object):
    """Simulate a sample path of a Poisson process.

    A Poisson process (with rate parameter λ) is continuous time stochastic
    process (N(t) : t ≥ 0) such that
    1) N(0) = 0,
    2) N(s + t) - N(s) has the Poisson distribution with mean λt,
    3) For all times 0 ≤ t_0 < t1 < ... < tn, the random variables
        N(t_0), N(t_1) - N(t_0), ..., N(t_n) - N(t_{n-1})
       are independent.

    Intuitively, N(s + t) - N(s) counts the number of "events" (or "arrivals",
    or "hits") occurring in the interval from s to s + t, as long as these
    events occur independently in disjoint intervals and the times (or
    distances) between events have the memoryless property.
    Poisson process arise naturally in many contexts. For example, recombination
    counts along segments of a genome can be modelled as a Poisson process.

    Properties
    ----------
    rate : float
        The average number of arrivals/hits/counts in an interval of length 1.
    random_state : numpy.random.RandomState
        The random number generator.
    """
    rate: float = None
    random_state: np.random.RandomState = None

    # The number of the last arrival computed
    _count: int = 0

    # The last arrival time computed
    _time: float = 0.0

    # The expanding array of arrival times
    _times: np.ndarray = None

    def __init__(self, rate=1.0, random_state=None):
        """Initialize a Poisson process by specifying the rate.

        Parameters
        ----------
        rate : float, optional
            A positive number, representing the average number of
            arrivals/hits/counts in an interval of length 1.
        random_state : int or numpy.random.RandomState object, optional
            A valid initializer for a numpy.random.RandomState object.
        """
        # Validate the rate
        self.rate = validate_float(rate, "rate", positive=True)

        # Seed the RNG
        if isinstance(random_state, np.random.RandomState):
            self.random_state = random_state
        else:
            self.random_state = np.random.RandomState(random_state)

        # Initialize the expanding times array
        self._times = np.zeros(shape=(1,), dtype=np.float_)

    def __next__(self):
        """Generate the next arrival of the Poisson process."""
        # Wait time until the next arrival
        wait = self.random_state.exponential(scale=(1 / self.rate), size=1)

        # Get the next arrival time and increment the arival count
        self._time = self._time + np.asscalar(wait)
        self._count += 1

        # Append the current time to the _times array, enlarging it if necessary
        if self._count >= len(self._times):
            self._times = np.resize(self._times, (2 * len(self._times),))
        self._times[self._count - 1] = self._time
        return self._time

    def __iter__(self):
        """Return self to adhere to the iterator protocol."""
        return self

    def times(self, n=None) -> np.ndarray:
        """Get the first n arrival times of the Poisson process.

        Parameters
        ----------
        n : int, optional
            Specify how many times to return. If not specified, all the
            currently generated times are returned.

        Returns
        -------
        One-dimensional NumPy array of Poisson process arrival times.
        """
        # Validate parameters
        if n is None:
            n = self._count
        else:
            n = validate_int(n, "n", minimum=1)

        # Generated more arrival times if needed
        while self._count < n:
            next(self)

        return self._times[:n]

    def plot(self, end, ax=None, **kwargs):
        """Plot one sample path of the Poisson process on the interval [0, end].

        Parameters
        ----------
        end : positive float
            The final time.
        ax : matplotlib.axes.Axes, optional
            The axes on which to draw the plot
        kwargs : dict
            Keyword arguments to pass to ax.step().

        Returns
        -------
        The matplotlib.axes.Axes object on which the plot was drawn.
        """
        # Validate parameters
        end = validate_float(end, "end", positive=True)

        # Get the axes to draw on if necessary
        if ax is None:
            ax = plt.gca()

        # Generated more arrival times if needed
        while self._time <= end:
            next(self)

        # Get the count of the first time exceeding the end time
        n = np.asscalar(np.argmax(self._times > end)) + 1

        times = np.concatenate(([0.0], self.times(n)))
        counts = np.arange(n + 1)

        ax.step(times, counts, where="post", **kwargs)
        ax.set(xlim=(0, end))
        ax.set(ylim=(0, None))

        return ax
