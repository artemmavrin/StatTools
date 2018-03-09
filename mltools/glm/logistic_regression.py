"""Defines the LogisticRegression class."""

import numbers

import numpy as np

from .generalized_linear_model import GeneralizedLinearModel
from ..generic import BinaryClassifier
from ..optimization import Optimizer
from ..regularization import lasso, ridge


def sigmoid(x):
    """Compute the sigmoid/logistic activation function 1 / (1 + exp(-x))."""
    return 1.0 / (1.0 + np.exp(np.negative(x)))


class CrossEntropyLoss(object):
    """Average cross entropy loss function for logistic regression.

    Minimizing this loss function is equivalent to maximizing the likelihood
    function in the logistic regression model.
    """

    def __init__(self, x, y):
        """Initialize with the training data.

        Parameters
        ----------
        x : array-like, shape (n, p)
            Explanatory variable.
        y : array-like, shape (n, )
            Response variable.
        """
        self.x = np.asarray(x)
        self.y = np.asarray(y)

        if len(x) != len(y):
            raise ValueError(
                f"Unequal number of observations: {len(x)} != {len(y)}")

        self.n = len(self.x)

    def __call__(self, coef):
        """Compute the average cross entropy loss for the training data."""
        logits = self.x.dot(coef)
        return np.mean(np.log1p(np.exp(-logits)) + (1 - self.y) * logits)

    def grad(self, coef):
        """Compute the gradient of the average cross entropy loss."""
        return -self.x.T.dot(self.y - sigmoid(self.x.dot(coef))) / self.n

    def hess(self, coef):
        """Compute the Hessian of the average cross entropy loss."""
        logits = self.x.dot(coef)
        weights = np.diag(sigmoid(logits) * (1.0 - sigmoid(logits)))
        return self.x.T.dot(weights).dot(self.x) / self.n


class LogisticRegression(GeneralizedLinearModel, BinaryClassifier):
    """Logistic regression via maximum likelihood estimation."""

    # Average cross entropy loss function
    loss: CrossEntropyLoss = None

    # The link function for logistic regression is the logit function, whose
    # inverse is the sigmoid function
    _inv_link = staticmethod(sigmoid)

    def __init__(self, penalty=None, lam=0.1, fit_intercept=True):
        """Initialize a LogisticRegression object.

        Parameters
        ----------
        penalty : None, "l1", or "l2", optional
            Type of regularization to impose on the loss function (if any).
            If None:
                No regularization.
            If "l1":
                L^1 regularization (LASSO - least absolute shrinkage and
                selection operator)
            If "l2":
                L^2 regularization (ridge regression)
        lam : positive float, optional
            Regularization parameter. Ignored if `penalty` is None.
        fit_intercept : bool, optional
            Indicates whether the module should fit an intercept term.
        """
        self.penalty = penalty
        self.fit_intercept = fit_intercept

        # Validate `lam`
        if penalty is not None:
            if not isinstance(lam, numbers.Real) or lam <= 0:
                raise ValueError("Parameter 'lam' must be a positive float.")
            else:
                self.lam = float(lam)

    def fit(self, x, y, optimizer, *args, **kwargs):
        """Fit the logistic regression model.

        Parameters
        ----------
        x : array-like, shape (n, p)
            Explanatory variable.
        y : array-like, shape (n, )
            Response variable.
        optimizer : Optimizer, optional
            Specifies the optimization algorithm used. If the model's `penalty`
            is None, this is ignored. Otherwise, this is required because it
            specifies how to minimize the penalized loss function.
        args : sequence, optional
            Additional positional arguments to pass to `optimizer`'s optimize().
        kwargs : dict, optional
            Additional keyword arguments to pass to `optimizer`'s optimize().

        Returns
        -------
        This LogisticRegression instance is returned.
        """
        # Validate input
        x = self._preprocess_x(x, fitting=True)
        y = self._preprocess_classes(y)
        y = self._preprocess_y(y)

        # Maximum likelihood estimation by minimizing the average cross entropy
        self.loss = CrossEntropyLoss(x, y)

        if self.penalty == "l1":
            self.loss = lasso(self.lam, self.loss)
        elif self.penalty == "l2":
            self.loss = ridge(self.lam, self.loss)
        elif self.penalty is not None:
            raise ValueError(f"Unknown penalty type: {self.penalty}")

        if not isinstance(optimizer, Optimizer):
            raise ValueError(f"Unknown minimization method: {optimizer}")

        self.coef = optimizer.optimize(x0=np.zeros(x.shape[1]), func=self.loss,
                                       *args, **kwargs)

        self._fitted = True
        return self

    def predict_prob(self, x):
        """Predict probability that the explanatory variable corresponds to the
        first class label.

        Parameters
        ----------
        x : array-like, shape (n, p)
            Explanatory variable.

        Returns
        -------
        P(y=C0|x) = sigmoid(x * coef)
        """
        return self.estimate(x)
