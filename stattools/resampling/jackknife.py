"""The Quenouille-Tukey jackknife for bias and standard error estimation."""

import numpy as np
import scipy.stats as st

from ..utils import validate_samples, validate_func, validate_float


class Jackknife(object):
    """Class for computing jackknife estimates for bias and standard error."""

    # Jackknife sampling distribution of leave-one-out statistics
    dist: np.ndarray

    # Sample size
    n_sample: int

    # Observed value of the statistic
    observed = None

    def __init__(self, *data, stat, **kwargs):
        """Initialize a Jackknife object by computing each leave-one-out
        statistic for a sample.

        Parameters
        ----------
        data: sequence of arrays
            Data on which to perform the leave-one-out resampling procedure.
            Each array in this sequence should have the same length (i.e.,
            sample size), but there is no restriction on the shape otherwise.
        stat: callable or str
            The statistic to compute from the data. If this parameter is a
            string, then it should be the name of a NumPy array method (e.g.,
            "mean" or "std"). If this parameter is a function, then it should
            accept as many arrays (and of the same shape) as are in `data`, and
            it should accept arrays of length one less than the arrays in
            `data`. The statistic is not assumed to be scalar-valued. This
            parameter is necessarily a keyword argument.
        kwargs: dict, optional
            Additional keyword arguments to pass to the function represented by
            the parameter `stat`.
        """
        data = validate_samples(*data, equal_lengths=True, ret_list=True)
        n_sample = len(data[0])
        stat = validate_func(stat, **kwargs)

        # We do not pre-allocate an array for the jackknife distribution of the
        # statistic because we do not know the dimension of `stat`'s output
        # beforehand.
        dist_jack = []

        # Perform the jackknife by leaving out each observation in the sample
        # and re-computing the statistic on the remaining subsample
        for i in range(n_sample):
            mask = np.arange(n_sample) != i
            dist_jack.append(stat(*(x[mask] for x in data)))

        # Store jackknife leave-one-out statistics and the observed statistic
        self.dist = np.asarray(dist_jack)
        self.observed = stat(*data)

        # Store the sample size
        self.n_sample = n_sample

    def bias(self):
        """Jackknife bias estimate."""
        return (self.n_sample - 1) * (self.dist.mean(axis=0) - self.observed)

    def estimate(self):
        """Jackknife bias-corrected estimate."""
        return self.observed - self.bias()

    def var(self):
        """Jackknife estimate for the variance of the statistic."""
        return (self.n_sample - 1) * self.dist.var(axis=0, ddof=0)

    def se(self):
        """Jackknife estimate for the standard error of the statistic."""
        return np.sqrt(self.var())

    def ci(self, alpha=0.05, tail="two-sided"):
        """Approximate coefficient-(1-alpha) jackknife confidence interval.

        This uses either the Student's t distribution or (if the sample size is
        greater than 30) the standard normal distribution to return the interval
        (observed + lower_quantile * SE, observed + upper_quantile * SE).

        Parameters
        ----------
        alpha: float in range [0, 1], optional
            Significance level.
        tail: "two-sided" (default), "left", or "right"
            Specify the tails of the confidence interval.

        Returns
        -------
        lower: float
            Lower confidence interval endpoint.
        upper: float
            Upper confidence interval endpoint.
        """
        # Validate significance level `alpha`
        alpha = validate_float(alpha, "alpha", minimum=0.0, maximum=1.0)

        # Determine which distribution's quantile function to use
        if self.n_sample <= 30:
            quantile = st.t(df=(self.n_sample - 1)).ppf
        else:
            quantile = st.norm(loc=0, scale=1).ppf

        # Compute the lower and upper confidence interval endpoints
        estimate = self.estimate()
        se = self.se()
        if tail == "two-sided":
            lower = estimate + quantile(alpha / 2) * se
            upper = estimate + quantile(1 - alpha / 2) * se
        elif tail == "left":
            lower = -np.inf
            upper = estimate + quantile(1 - alpha) * se
        elif tail == "right":
            lower = estimate + quantile(alpha) * se
            upper = np.inf
        else:
            raise ValueError(f"Unsupported value for parameter 'tail': {tail}")

        return lower, upper
