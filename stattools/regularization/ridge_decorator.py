"""Defines the ridge regression (L2 regularization) decorator."""

import numpy as np

from ..utils import validate_float


def ridge(penalty, loss=None):
    """Create a ridge regression (L2 regularization/penalty) decorator.

    Parameters
    ----------
    penalty : float
        Regularization constant. Must be positive.

    loss : callable, optional
        Loss function to penalize.

    Returns
    -------
    If `loss` is not specified, a ridge regression decorator with parameter
    `penalty` is returned. Otherwise, the penalized version of `loss` is
    returned.
    """
    penalty = validate_float(penalty, "penalty", positive=True)

    def _ridge(func):
        if not callable(func):
            raise TypeError(f"Loss function {func} is not callable")

        class RidgeDecorator(object):
            def __init__(self):
                self.func = func
                self.penalty = penalty

            def __call__(self, x, *args, **kwargs):
                penalty = self.penalty * np.dot(x, x)
                return self.func(x, *args, **kwargs) + penalty

            if hasattr(func, "grad") and callable(func.grad):
                def grad(self, x, *args, **kwargs):
                    penalty = 2.0 * np.multiply(self.penalty, x)
                    return self.func.grad(x, *args, **kwargs) + penalty

            if hasattr(func, "hess") and callable(func.hess):
                def hess(self, x, *args, **kwargs):
                    penalty = 2.0 * self.penalty * np.identity(np.shape(x)[0])
                    return self.func.hess(x, *args, **kwargs) + penalty

        return RidgeDecorator()

    if loss is None:
        return _ridge
    else:
        return _ridge(loss)
