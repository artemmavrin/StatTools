"""Implements the HypothesisTest abstract base class"""

import abc
from collections import namedtuple

fields = ("statistic", "p_value", "lower", "upper")
HypothesisTestResult = namedtuple("HypothesisTestResult", field_names=fields)


class HypothesisTest(metaclass=abc.ABCMeta):
    """Abstract base class for various hypothesis tests."""

    @abc.abstractmethod
    def test(self, *args, **kwargs) -> HypothesisTestResult:
        """The test() method should perform the hypothesis test and return the
        result, which should be a HypothesisTestResult object containing the
        observed value of the test statistic, a p-value, and lower and upper
        bounds for a confidence interval at a specified significance level.
        """
        pass
