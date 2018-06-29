"""Defines the LogisticRegression class."""

import numpy as np

from .glm import GLM
from ..generic import Classifier
from ..optimization import Optimizer
from ..regularization import lasso, ridge
from ..utils import validate_bool
from ..utils import validate_float


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


class LogisticRegression(GLM, Classifier):
    """Logistic regression via maximum likelihood estimation."""

    # Regularization type
    reg = None

    # Regularization type
    penalty: float = None

    # Loss function
    loss = None

    # The link function for logistic regression is the logit function, whose
    # inverse is the sigmoid function
    _inv_link = staticmethod(sigmoid)

    def __init__(self, reg=None, penalty=0.1, standardize=True,
                 fit_intercept=True):
        """Initialize a LogisticRegression object.

        Parameters
        ----------
        reg : None, "l1", or "l2", optional
            Type of regularization to impose on the loss function (if any).
            If None: No regularization.
            If "l1": L^1 regularization.
            If "l2": L^2 regularization.
        penalty : positive float, optional
            Regularization parameter. Ignored if `reg` is None.
        standardize : bool, optional
            Indicates whether the explanatory and response variables should be
            centered to have mean 0 and scaled to have variance 1.
        fit_intercept : bool, optional
            Indicates whether the model should fit an intercept term.
        """
        self.reg = reg
        self.standardize = validate_bool(standardize, "standardize")
        self.fit_intercept = validate_bool(fit_intercept, "fit_intercept")
        self.penalty = validate_float(penalty, "penalty", positive=True)

    def fit(self, x, y, optimizer, names=None, *args, **kwargs):
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
        names : list, optional
            List of feature names corresponding to the columns of `x`.
        args : sequence, optional
            Additional positional arguments to pass to `optimizer`'s optimize().
        kwargs : dict, optional
            Additional keyword arguments to pass to `optimizer`'s optimize().

        Returns
        -------
        This LogisticRegression instance.
        """
        # Validate input
        x = self._preprocess_features(x=x, names=names)
        y = self._preprocess_classes(y=y, max_classes=2)
        y = self._preprocess_response(y=y, x=x)

        # Maximum likelihood estimation by minimizing the average cross entropy
        self.loss = CrossEntropyLoss(x, y)

        if self.reg == "l1":
            self.loss = lasso(penalty=self.penalty, loss=self.loss)
        elif self.reg == "l2":
            self.loss = ridge(penalty=self.penalty, loss=self.loss)
        elif self.reg is not None:
            raise ValueError(f"Unknown penalty type: {self.reg}")

        if not isinstance(optimizer, Optimizer):
            raise ValueError(f"Unknown minimization method: {optimizer}")

        self._coef = optimizer.optimize(x0=np.zeros(x.shape[1]), func=self.loss,
                                        *args, **kwargs)

        self.fitted = True
        return self

    def predict_prob(self, x):
        """Return estimated probability that the response corresponding to a
        set of features belongs to each possible class.

        Parameters
        ----------
        x : array-like, shape (n, p)
            Explanatory variable.

        Returns
        -------
        Matrix of shape (len(x), 2). The (i, j)-th entry is the probability that
        the i-th observation corresponds to the j-th class.
        """
        estimate = self.estimate(x)
        return np.column_stack((1 - estimate, estimate))
