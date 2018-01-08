"""Defines the GradientDescent class."""

import numbers

import numpy as np

from . import Minimizer


class GradientDescent(Minimizer):
    """Unconstrained batch gradient descent with momentum."""

    def __init__(self, rate=0.1, momentum=0.0, anneal=np.inf, iterations=10000):
        """Initialize the parameters of a gradient descent object.

        Parameters
        ----------
        rate: float, optional
            Step size/learning rate. Must be positive.
        momentum: float, optional
            Momentum parameter. Must be positive.
        anneal: float, optional
            Factor determining the annealing schedule of the learning rate. Must
            be positive. Smaller values lead to faster shrinking of the learning
            rate over time.
        iterations: int, optional
            Number of iterations of the algorithm to perform. Must be positive.
        """
        if not isinstance(rate, numbers.Real) or rate <= 0:
            raise TypeError("Parameter 'rate' must be a positive float")
        if not isinstance(momentum, numbers.Real) or momentum < 0:
            raise TypeError("Parameter 'momentum' must be a non-negative float")
        if not isinstance(anneal, numbers.Real) or anneal <= 0:
            raise TypeError("Parameter 'anneal' must be a positive float")
        if not isinstance(iterations, numbers.Integral) or iterations <= 0:
            raise TypeError("Parameter 'iterations' must be a positive int")

        self.rate = rate
        self.momentum = momentum
        self.anneal = anneal
        self.iterations = iterations

    def minimize(self, x0, func, grad=None, args=None, kwargs=None,
                 callback=None):
        """Approximate a minimizer of the objective function.

        Parameters
        ----------
        func: callable
            The objective function to minimize.
        x0: array-like
            Initial guess for the minimizer.
        grad: callable, optional
            Gradient/Jacobian (vector of first derivatives) of the objective
            function. This must be a function returning a 1D array. If it is not
            specified, then `func` needs to have a 'grad' attribute.
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
            The approximate minimizer of the objective function.
        """
        if not callable(func):
            raise ValueError(f"Objective function {func} is not callable")

        if grad is None:
            if hasattr(func, "grad"):
                grad = func.grad
            else:
                raise ValueError("Could not detect function gradient")
        if not callable(grad):
            raise ValueError(f"Gradient {grad} is not callable")

        if args is None:
            args = ()
        if kwargs is None:
            kwargs = {}

        x = np.asarray(x0)
        update = np.zeros(x.shape)
        if callback is not None:
            callback(x, *args, **kwargs)

        for t in range(int(self.iterations)):
            rate = self.rate / (1 + t / self.anneal)
            update = self.momentum * update - rate * grad(x, *args, **kwargs)
            x = x + update
            if callback is not None:
                callback(x, *args, **kwargs)
        return x
