"""Unit tests for the GLM class."""

import string
import unittest

import numpy as np
import pandas as pd

from stattools.glm.glm import GLM


class ExampleGLM(GLM):
    def __init__(self, standardize=True, fit_intercept=True):
        self.standardize = standardize
        self.fit_intercept = fit_intercept

    def fit(self, x, y, names=None):
        # Validate input
        x = self._preprocess_features(x=x, names=names)
        y = self._preprocess_response(y=y, x=x)

        self._coef = np.zeros((self._p,))

        self.fitted = True
        return self

    _inv_link = staticmethod(lambda x: x)


class TestGLM(unittest.TestCase):
    def test_names(self):
        """Check that feature name extraction works."""
        n, p = 100, 20

        # Case 1: no names given
        x = np.arange(n * p).reshape((n, p))
        y = np.arange(len(x))
        model = ExampleGLM().fit(x, y)
        expected_names = ["x" + str(n) for n in range(p)]
        np.testing.assert_equal(model.names, expected_names)

        # Case 2: list of names given
        names = [c for c in string.ascii_lowercase[:p]]
        model = ExampleGLM().fit(x, y, names=names)
        np.testing.assert_equal(model.names, names)

        # Case 3: get names from a pandas DataFrame
        x_df = pd.DataFrame(x, columns=names)
        model = ExampleGLM().fit(x_df, y)
        np.testing.assert_equal(model.names, names)

        # Case 4: overwrite names of pandas DataFrame
        alt_names = list(map(str, range(p)))
        model = ExampleGLM().fit(x_df, y, names=alt_names)
        np.testing.assert_equal(model.names, alt_names)

        # Case 5: invalid names
        bad_names_1 = list(map(str, range(p + 1)))
        bad_names_2 = list(map(str, range(p - 1)))
        bad_names_3 = list(range(p))

        for bad_names in (bad_names_1, bad_names_2, bad_names_3):
            with self.assertRaises(ValueError):
                _ = ExampleGLM().fit(x, y, names=bad_names)


if __name__ == "__main__":
    unittest.main()
