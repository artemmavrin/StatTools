"""Implements the HypothesisTest abstract base class"""

import abc
import collections

HypothesisTestResult = collections.namedtuple("HypothesisTestResult",
                                              ("statistic", "p_value"))


class HypothesisTest(metaclass=abc.ABCMeta):
    """Abstract base class for various hypothesis tests.
    """

    @abc.abstractmethod
    def test(self, *args, **kwargs) -> HypothesisTestResult:
        """The test() method should perform the hypothesis test and return the
        result, which should be a HypothesisTestResult object. The "statistic"
        field should contain the test statistic of the data, and the "p_value"
        should be the p-value of the test.
        """
        pass
