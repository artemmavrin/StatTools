"""Defines the NewtonRaphson class."""

import numpy as np

from .base import Optimizer
from ..utils import validate_int


class NewtonRaphson(Optimizer):
    """Find stationary points of functions using the Newton-Raphson method."""

    def __init__(self, iterations=1000):
        """Initialize the parameters of a Newton-Raphson method object.

        Parameters
        ----------
        iterations: int, optional
            Number of iterations of the algorithm to perform.
        """
        self.iterations = validate_int(iterations, "iterations", minimum=1)

    def optimize(self, x0, func, grad=None, hess=None, args=None, kwargs=None,
                 callback=None):
        """Approximate a stationary point of the objective function.

        Parameters
        ----------
        x0: array-like
            Initial guess for the minimizer.
        func: callable
            The objective function to minimize.
        grad: callable, optional
            Gradient/Jacobian (vector of first derivatives) of the objective
            function. This must be a function returning a 1D array. If it is not
            specified, then `func` needs to have a 'grad' attribute.
        hess: callable, optional
            Hessian (matrix of mixed second derivatives) of the objective
            function. This must be a function returning a 2D array. If it is not
            specified, then `func` needs to have a 'hess' attribute.
        args: sequence, optional
            Extra positional arguments to pass to the objective function and
            gradient.
        kwargs: dict, optional
            Extra keyword arguments to pass to the objective function and
            gradient.
        callback: callable, optional
            Function to call at every iteration of the algorithm. The function
            is called on the current value of the parameter being minimized
            along with the extra arguments specified by `args` and `kwargs`.
            For example, `callback` could be a function that prints the value of
            the objective function at each iteration.

        Returns
        -------
        x : array-like
            An approximate stationary point of the objective function.
        """
        if not callable(func):
            raise ValueError(f"Objective function {func} is not callable")

        if grad is None:
            if hasattr(func, "grad"):
                grad = func.grad
            else:
                raise ValueError("Could not detect objective function gradient")
        if not callable(grad):
            raise ValueError(f"Gradient {grad} is not callable")

        if hess is None:
            if hasattr(func, "hess"):
                hess = func.hess
            else:
                raise ValueError("Could not detect objective function Hessian")
        if not callable(hess):
            raise ValueError(f"Hessian {hess} is not callable")

        if args is None:
            args = ()
        if kwargs is None:
            kwargs = {}

        x = np.asarray(x0)
        if callback is not None:
            callback(x, *args, **kwargs)

        for _ in range(int(self.iterations)):
            a = np.atleast_2d(hess(x, *args, **kwargs))
            b = np.atleast_1d(grad(x, *args, **kwargs))
            u, *_ = np.linalg.lstsq(a, b, rcond=None)
            x = x - u
            if callback is not None:
                callback(x, *args, **kwargs)
        return x
