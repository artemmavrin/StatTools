"""Defines the LogisticRegression class."""

import numpy as np

from .generalized_linear_model import GeneralizedLinearModel
from ..classification import BinaryClassifier
from ..optimization import Minimizer
from ..regularization import lasso, ridge


def sigmoid(x):
    """Compute the sigmoid/logistic activation function 1 / (1 + exp(-x))."""
    return 1.0 / (1.0 + np.exp(np.negative(x)))


class LogisticLoss(object):
    """Average cross entropy loss function for logistic regression."""

    def __init__(self, x, y):
        """Initialize with the training data.

        Parameters
        ----------
        x : array-like
            Training data feature matrix.
        y : array-like
            Training data target vector.
        """
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.n_samples = self.x.shape[0]

        if self.n_samples != self.y.shape[0]:
            raise ValueError(
                f"Unequal number of samples: "
                f"{self.n_samples} feature samples, "
                f"{self.y.shape[0]} target samples")

    def __call__(self, w):
        """Compute the average cross entropy loss for the training data."""
        logits = self.x.dot(w)
        loss = np.sum(np.log1p(np.exp(-logits))) + np.sum(logits[self.y == 0])
        return loss / self.n_samples

    def grad(self, w):
        """Compute the gradient of the average cross entropy loss."""
        return -self.x.T.dot(self.y - sigmoid(self.x.dot(w))) / self.n_samples

    def hess(self, w):
        """Compute the Hessian of the average cross entropy loss."""
        logits = self.x.dot(w)
        weights = np.diag(sigmoid(logits) * (1.0 - sigmoid(logits)))
        return self.x.T.dot(weights).dot(self.x) / self.n_samples


class LogisticRegression(GeneralizedLinearModel, BinaryClassifier):
    # Average cross entropy loss function
    loss = None

    def __init__(self, penalty="l2", lam=0.1, intercept=True):
        """Initialize a logistic regression model.

        Parameters
        ----------
        penalty: str, optional
            Type of regularization to impose (none if None).
            Currently supported:
            None
                No regularization
            "l1"
                L1 regularization (AKA LASSO regression)
            "l2"
                L2 regularization (AKA ridge regression)
        lam: float, optional
            Regularization parameter.
        intercept: bool, optional
            Indicates whether the module should fit an intercept term.
        """
        self.penalty = penalty
        self.lam = lam
        self.intercept = intercept

    # The link function for logistic regression is the logit function, whose
    # inverse is the sigmoid function
    _inv_link = staticmethod(sigmoid)

    def fit(self, x, y, minimizer, *args, **kwargs):
        """Fit the logistic regression model.

        Parameters
        ----------
        x: array-like
            Feature matrix. Should be 2-dimensional of shape
            (n_samples, n_features). If 1-dimensional, will be treated as if of
            shape (n_samples, 1) (i.e., 1 feature column).
        y: array-like
            Target vector. Should consist of binary class labels of some kind.
        minimizer: Minimizer
            Specifies the optimization algorithm used.
        args: sequence
            Additional positional arguments to pass to the minimizer's
            `minimize` method.
        kwargs: dict
            Additional keyword arguments to pass to the minimizer's `minimize`
            method.

        Returns
        -------
        This LogisticRegression instance.
        """
        y = self._preprocess_classes(y)
        y = self._preprocess_target(y)

        x = self._preprocess_features(x, training=True)

        self.loss = LogisticLoss(x, y)

        if self.penalty == "l1":
            self.loss = lasso(self.lam, self.loss)
        elif self.penalty == "l2":
            self.loss = ridge(self.lam, self.loss)
        elif self.penalty is not None:
            raise ValueError(f"Unknown penalty type: {self.penalty}")

        if not isinstance(minimizer, Minimizer):
            raise ValueError(f"Unknown minimization method: {minimizer}")

        w0 = np.zeros(x.shape[1])

        self._weights = minimizer.minimize(x0=w0, func=self.loss,
                                           *args, **kwargs)
        self._fitted = True
        return self

    def predict_prob(self, x):
        """Predict probability that data corresponds to the first class label.

        Parameters
        ----------
        x: array-like
            Feature matrix.

        Returns
        -------
        P(y=C0|x)
        """
        return self.estimate(x)
