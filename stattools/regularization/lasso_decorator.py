"""Defines the LASSO regression (L1 regularization) decorator."""

import numpy as np

from ..utils import validate_float


def lasso(penalty, loss=None):
    """Create a LASSO regression (L1 regularization/penalty) decorator.

    Parameters
    ----------
    penalty : float
        Regularization constant. Must be positive.

    loss : callable, optional
        Loss function to penalize.

    Returns
    -------
    If `loss` is not specified, a LASSO regression decorator with parameter
    `penalty` is returned. Otherwise, the penalized version of `loss` is
    returned.
    """
    penalty = validate_float(penalty, "penalty", positive=True)

    def _lasso(func):
        if not callable(func):
            raise TypeError(f"Loss function {func} is not callable")

        class LASSODecorator(object):
            def __init__(self):
                self.func = func
                self.penalty = penalty

            def __call__(self, x, *args, **kwargs):
                penalty = self.penalty * np.linalg.norm(x, ord=1)
                return self.func(x, *args, **kwargs) + penalty

            if hasattr(func, "grad") and callable(func.grad):
                def grad(self, x, *args, **kwargs):
                    penalty = self.penalty * np.sign(x)
                    return self.func.grad(x, *args, **kwargs) + penalty

        return LASSODecorator()

    if loss is None:
        return _lasso
    else:
        return _lasso(loss)
