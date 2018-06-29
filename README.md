# StatTools

[![PyPI version](https://badge.fury.io/py/stattools.svg)](https://badge.fury.io/py/stattools)

Statistical learning and inference algorithms implemented in pure Python (version 3.6 or later).

## Installation

The latest version of StatTools can be installed directly after cloning from GitHub:

    git clone https://github.com/artemmavrin/StatTools.git
    cd StatTools
    make install

Moreover, StatTools is on the [Python Package Index (PyPI)](https://pypi.org/project/stattools/), so a recent version of it can be installed with the `pip` utility:

    pip install stattools

## Dependencies

* [NumPy](http://www.numpy.org)
* [SciPy](https://www.scipy.org)
* [pandas](https://pandas.pydata.org)
* [Matplotlib](https://matplotlib.org)

## Examples

### Regression

* [Simple linear regression for fitting a line through a scatter plot](https://github.com/artemmavrin/StatTools/blob/master/examples/Simple%20Linear%20Regression.ipynb)
* [Ridge regression](https://github.com/artemmavrin/StatTools/blob/master/examples/Ridge%20Regression.ipynb)
* [Elastic net regularization (including LASSO and ridge regression as special cases)](https://github.com/artemmavrin/StatTools/blob/master/examples/Elastic%20Net.ipynb)
* [Fitting a polynomial curve to a scatter plot](https://github.com/artemmavrin/StatTools/blob/master/examples/Polynomial%20Smoothing.ipynb)
* [Various scatterplot smoothers applied to a sine curve with Gaussian noise](https://github.com/artemmavrin/StatTools/blob/master/examples/Scatterplot%20Smoothers.ipynb)

### Classification

* [Logistic regression for breast cancer diagnosis](https://github.com/artemmavrin/StatTools/blob/master/examples/Logistic%20Regression.ipynb)
* [Multiclass logistic regression for handwritten digit recognition](https://github.com/artemmavrin/StatTools/blob/master/examples/Multiclass%20Logistic%20Regression.ipynb)

### Unsupervised Learning

* [K-means clustering for grouping unlabelled data together](https://github.com/artemmavrin/StatTools/blob/master/examples/K-Means%20Clustering.ipynb)
* [Estimation of Gaussian mixture models](https://github.com/artemmavrin/StatTools/blob/master/examples/Gaussian%20Mixture%20Models.ipynb)
* [Principal component analysis applied to handwritten digits](https://github.com/artemmavrin/StatTools/blob/master/examples/Principal%20Component%20Analysis.ipynb)
* [Kernel density estimation for histogram smoothing](https://github.com/artemmavrin/StatTools/blob/master/examples/Kernel%20Density%20Estimation.ipynb)

### Non-Parametric Statistics

* [The bootstrap (ordinary and Bayesian) and the jackknife for standard error estimation](https://github.com/artemmavrin/StatTools/blob/master/examples/Bootstrap%20and%20Jackknife.ipynb)
* [Bootstrap confidence intervals](https://github.com/artemmavrin/StatTools/blob/master/examples/Bootstrap%20Confidence%20Intervals.ipynb)
* [Exact and Monte Carlo permutation tests](https://github.com/artemmavrin/StatTools/blob/master/examples/Permutation%20Test.ipynb)
* [The Kaplan-Meier survivor function estimator](https://github.com/artemmavrin/StatTools/blob/master/examples/Kaplan-Meier%20Estimator.ipynb)

### Ensemble Methods

* [Using bagging to improve logistic regression accuracy](https://github.com/artemmavrin/StatTools/blob/master/examples/Bagging%20Logistic%20Regression.ipynb)

### Data Visualization

* [Plotting lines and function curves](https://github.com/artemmavrin/StatTools/blob/master/examples/Plotting%20Lines%20and%20Functions.ipynb)
* [Drawing empirical distribution functions](https://github.com/artemmavrin/StatTools/blob/master/examples/Empirical%20Distribution%20Functions.ipynb)
* [Drawing quantile-quantile (QQ) plots](https://github.com/artemmavrin/StatTools/blob/master/examples/Quantile-Quantile%20Plots.ipynb)

### Simulation

* [Simulating sample paths of Poisson processes](https://github.com/artemmavrin/StatTools/blob/master/examples/Poisson%20Process.ipynb)
* [Simulating sample paths of Itô diffusions (for example, Brownian motion)](https://github.com/artemmavrin/StatTools/blob/master/examples/Ito%20Diffusions.ipynb)
