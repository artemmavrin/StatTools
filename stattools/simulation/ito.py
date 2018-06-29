"""Simulate sample paths of Itô diffusions."""

import matplotlib.pyplot as plt
import numpy as np

from ..utils.validation import validate_float


class ItoDiffusion(object):
    """Simulate an approximate sample path of an Itô diffusion.

    Properties
    ----------
    drift : callable
    diffusion : callable
        The drift and diffusion coefficients of the process. They are both
        functions of time.
    x0 : float
        The starting point of the diffusion at time 0.
    step : float
        The mesh size for time interval partitioning
    random_state : numpy.random.RandomState
        The random number generator.
    """
    drift = None
    diffusion = None
    x0: float = None
    step: float = None
    random_state: np.random.RandomState = None

    # The last value of the process computed
    _value: int = None

    # The number of discrete times at which the process was computed
    _steps: int = 0

    # The expanding array of sample path values
    _path: np.ndarray = None

    def __init__(self, drift, diffusion, x0=0.0, step=1e-3, random_state=None):
        """Initialize the coefficients of the Itô diffusion.

        Parameters
        ----------
        drift : callable
            The drift coefficient (a numeric function of time).
        diffusion : callable
            The diffusion coefficient (a numeric function of time).
        x0 : float
            The starting point of the diffusion at time 0.
        step : float, optional
            The mesh size for time interval partitioning
        random_state : int or numpy.random.RandomState object, optional
            The random number generator.
        """
        # Validate parameters
        if callable(drift):
            self.drift = np.vectorize(drift)
        else:
            raise ValueError("Parameter 'drift' must be callable.")
        if callable(diffusion):
            self.diffusion = np.vectorize(diffusion)
        else:
            raise ValueError("Parameter 'diffusion' must be callable.")
        self.x0 = validate_float(x0, "x0")
        self.step = validate_float(step, "step", positive=True)

        # Seed the RNG
        if isinstance(random_state, np.random.RandomState):
            self.random_state = random_state
        else:
            self.random_state = np.random.RandomState(random_state)

        # Initialize the expanding sample path array and the last value computed
        self._path = np.asarray([self.x0])
        self._value = self.x0

    def __next__(self):
        """Generate the value of the Itô diffusion."""
        # Brownian motion increment
        bm_increment = self.random_state.normal(loc=0.0,
                                                scale=np.sqrt(self.step),
                                                size=1)

        # Generate the next value of the process by the Euler-Maruyama method
        self._value = self._value + self.drift(
            self._value) * self.step + self.diffusion(
            self._value) * bm_increment

        # Increment the time
        self._steps += 1

        # Append the current value to the _path array, enlarging it if necessary
        if self._steps >= len(self._path):
            self._path = np.resize(self._path, (2 * len(self._path),))
        self._path[self._steps] = self._value
        return self._value

    def __iter__(self):
        """Return self to adhere to the iterator protocol."""
        return self

    def path(self, end=None):
        """Get a sample path up to a specified end time.

        Parameters
        ----------
        end : float, optional
            The positive final time. If not given, the entire sample path
            computed so far will be returned.

        Returns
        -------
        t : numpy.ndarray
            The discretized time interval starting at 0 and ending at the
            smallest integer multiple of the step size exceeding `end`.
        x : numpy.ndarray
            The values of the sample path at the times in `t`.
        """
        # Validate parameters
        end = validate_float(end, "end", positive=True)

        # Generate more of the path if necessary
        while self._steps * self.step <= end:
            next(self)

        n = int(end / self.step) + 1
        t = np.linspace(start=0.0, stop=(n * self.step), num=(n + 1))
        x = self._path[:(n + 1)]

        return t, x

    def plot(self, end, ax=None, **kwargs):
        """Plot the Itô diffusion sample path on the interval [0, end].

        Parameters
        ----------
        end : positive float
            The final time.
        ax : matplotlib.axes.Axes, optional
            The axes on which to draw the plot
        kwargs : dict
            Keyword arguments to pass to ax.plot().

        Returns
        -------
        The matplotlib.axes.Axes object on which the plot was drawn.
        """
        # Get the axes to draw on if necessary
        if ax is None:
            ax = plt.gca()

        t, x = self.path(end)
        ax.plot(t, x, **kwargs)
        ax.set(xlim=(0, end))

        return ax


class BrownianMotion(ItoDiffusion):
    """Simulate Brownian motion (with drift).
    Brownian motion with drift is a stochastic process X satisfying the SDE
    dX(t) = μ dt + σ dB(t), where μ and σ are constants and B is standard
    Brownian motion.
    Explicitly, X(t) = X(0) + μt + σB(t). In particular, if μ = 0 and σ = 1,
    then X is Brownian motion started at X(0).
    """

    def __init__(self, mu=0.0, sigma=1.0, x0=0.0, step=1e-3, random_state=None):
        """Initialize a Brownian motion with drift.
        If no parameters are specified, initialize a standard Brownian motion.

        Parameters
        ----------
        mu : float
            The constant drift rate.
        sigma : float
            The constant standard deviation scaling.
        x0 : float
            The starting point of the Brownian motion with drift at time 0.
        step : float, optional
            The mesh size for time interval partitioning
        random_state : int or numpy.random.RandomState object, optional
            The random number generator.
        """
        super(BrownianMotion, self).__init__(drift=(lambda _: mu),
                                             diffusion=(lambda _: sigma),
                                             x0=x0, step=step,
                                             random_state=random_state)


class GeometricBM(ItoDiffusion):
    """Simulate geometric Brownian motion.
    Geometric Brownian motion is a stochastic process X satisfying the SDE
    dX(t) = μX(t) dt + σX(t) dB(t), where μ and σ are constants and B is
    standard Brownian motion. This process comes up in, e.g., the Black-Scholes
    model of financial markets as the stock price process.
    Explicitly, X(t) = X(0)exp((μ - (1/2)σ^2)t + σB(t)).
    """

    def __init__(self, mu, sigma, x0=1.0, step=1e-3, random_state=None):
        """Initialize a geometric Brownian motion.

        Parameters
        ----------
        mu : float
            The constant drift rate.
        sigma : float
            The constant volatility.
        x0 : float
            The starting point of the geometric Brownian motion at time 0.
        step : float, optional
            The mesh size for time interval partitioning
        random_state : int or numpy.random.RandomState object, optional
            The random number generator.
        """
        super(GeometricBM, self).__init__(drift=(lambda x: mu * x),
                                          diffusion=(lambda x: sigma * x),
                                          x0=x0, step=step,
                                          random_state=random_state)


class OrnsteinUhlenbeck(ItoDiffusion):
    """Simulate an Ornstein-Uhlenbeck process.
    An Ornstein-Uhlenbeck process is a stochastic process X satisfying the SDE
    dX(t) = θ(μ - X(t)) dt + σ dB(t), where θ, μ, and σ are constants and B is
    standard Brownian motion.
    """

    def __init__(self, theta, mu, sigma, x0=1.0, step=1e-3, random_state=None):
        """Initialize an Ornstein-Uhlenbeck process.

        Parameters
        ----------
        theta : float
            The constant decay rate.
        mu : float
            The asymptotic mean (equilibrium).
        sigma : float
            The constant standard deviation scaling.
        x0 : float
            The starting point of the Ornstein-Uhlenbeck process at time 0.
        step : float, optional
            The mesh size for time interval partitioning
        random_state : int or numpy.random.RandomState object, optional
            The random number generator.
        """
        super(OrnsteinUhlenbeck, self).__init__(drift=(lambda x:
                                                       theta * (mu - x)),
                                                diffusion=(lambda _: sigma),
                                                x0=x0, step=step,
                                                random_state=random_state)
