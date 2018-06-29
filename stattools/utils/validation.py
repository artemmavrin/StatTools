"""Functions for input validation."""

import functools
import numbers

import numpy as np


def validate_bool(x, name) -> bool:
    """Validate boolean function parameters.

    Parameters
    ----------
    x : object
        Object to validate as a bool.
    name : dtr
        Name of the function parameter.

    Returns
    -------
    bool(x) if x is a bool.

    Raises
    ------
    TypeError if x is not a bool.
    """
    if isinstance(x, bool):
        return bool(x)
    else:
        raise TypeError(f"Parameter '{name}' must be boolean.")


def validate_int(x, name, minimum=None, maximum=None) -> int:
    """Validate integer function parameters.

    Parameters
    ----------
    x : object
        Object to validate as an int.
    name : str
        Name of the function parameter.
    minimum : int, optional
        Minimum value x can take (inclusive).
    maximum : int, optional
        Maximum value x can take (inclusive).

    Returns
    -------
    int(x) if x is an int satisfying the additional constraints.

    Raises
    ------
    TypeError if x cannot be coerced into an int.
    ValueError if x < minimum or x > maximum.
    """
    if isinstance(x, numbers.Integral):
        x = int(x)
        if minimum is not None:
            minimum = validate_int(minimum, "minimum")
            if x < minimum:
                raise ValueError(
                    f"Parameter '{name}' must be at least {minimum}")
        if maximum is not None:
            maximum = validate_int(maximum, "maximum")
            if x > maximum:
                raise ValueError(
                    f"Parameter '{name}' must be at most {maximum}")

    else:
        raise TypeError(f"Parameter '{name}' must be an int.")
    return x


def validate_float(x, name, positive=False, minimum=None,
                   maximum=None) -> float:
    """Validate float function parameters.

    Parameters
    ----------
    x : object
        Object to validate as a float.
    name : str
        Name of the function parameter.
    positive : bool
        If True, x must be positive. If False, x can be any float.
    minimum : int, optional
        Minimum value x can take (inclusive).
    maximum : int, optional
        Maximum value x can take (inclusive).

    Returns
    -------
    float(x) if x is a float satisfying the additional constraints.

    Raises
    ------
    TypeError if x cannot be coerced into a float.
    ValueError if x < minimum or x > maximum or positive is True but x <= 0.
    """
    if isinstance(x, numbers.Real):
        x = float(x)
        positive = validate_bool(positive, "positive")
        if positive and x <= 0.0:
            raise ValueError(f"Parameter '{name}' must be positive.")
        if minimum is not None:
            minimum = validate_float(minimum, "minimum")
            if x < minimum:
                raise ValueError(
                    f"Parameter '{name}' must be at least {minimum}")
        if maximum is not None:
            maximum = validate_float(maximum, "maximum")
            if x > maximum:
                raise ValueError(
                    f"Parameter '{name}' must be at most {maximum}")
    else:
        raise TypeError(f"Parameter '{name}' must be a float.")
    return x


def validate_samples(*samples, n_dim=None, equal_lengths=False,
                     equal_shapes=False, ret_list=False):
    """Preprocess and validate multiple data samples.

    Parameters
    ----------
    samples : sequence of arrays
        List of data samples to preprocess and validate.
    n_dim : int or tuple of 1's, 2's, and None's, optional
        Maximum number of dimensions for each sample (either 1, 2, or None).
        If `ndim` is an int, each samples must have at most `ndim` dimensions.
        If `ndim` is a tuple, each non-None item in the tuple represents the
        maximum number of dimensions of the sample at the same index. In either
        case, the samples are coerced to have the maximum number of dimensions.
    equal_lengths : bool, optional
        Indicates whether each sample should have the same length (i.e., number
        of observations).
    equal_shapes : bool, optional
        Indicates whether each sample should have the same shape (apart from the
        length).
    ret_list : bool, optional
        Indicates whether the samples should be returned as a list even if there
        is only one sample.

    Returns
    -------
    samples : list or numpy.ndarray
        If multiple samples are provided or `ret_list` is True, a list
        containing the preprocessed and validated samples is returned.
        Otherwise, the single modified sample is returned.
    """
    # Number of samples
    n_samples = len(samples)

    # Ensure at least one sample is provided
    if n_samples == 0:
        raise ValueError("At least one sample is needed for validation.")

    # Coerce each sample into a NumPy ndarray of dimension at least 1.
    samples = list(map(np.atleast_1d, samples))

    # Check for number of dimensions if necessary
    if n_dim is not None:
        # If `n_dim` is an int, convert it to a tuple
        if isinstance(n_dim, numbers.Integral):
            n_dim = tuple(n_dim for _ in range(n_samples))

        if isinstance(n_dim, tuple):
            # Validate `n_dim` as a tuple containing only 1's, 2's, and None's
            if any(not isinstance(d, numbers.Integral) for d in n_dim
                   if d is not None):
                raise TypeError("Parameter 'n_dim' must be 1, 2, or None.")
            n_dim = tuple(int(d) if isinstance(d, numbers.Integral) else None
                          for d in n_dim)
            if any(d not in (1, 2) for d in n_dim if d is not None):
                raise TypeError("Parameter 'n_dim' must be 1, 2, or None.")
            if len(n_dim) != n_samples:
                raise ValueError(
                    "Parameter `n_dim` must have one entry for each sample.")

            # Check the dimension of each array
            for i, (sample, d) in enumerate(zip(samples, n_dim)):
                if d is None:
                    continue
                if sample.ndim == 1 and d == 2:
                    samples[i] = sample.reshape(-1, 1)
                elif sample.ndim > d:
                    raise ValueError(f"Sample {i} has too many dimensions.")
        else:
            raise TypeError("Parameter 'n_dim' must be an int or a tuple.")

    # Check for equal array lengths if necessary
    if equal_lengths and n_samples > 1:
        if any(len(x) != len(samples[0]) for x in samples[1:]):
            raise ValueError("Each sample must have the same length.")

    # Check for equal array shapes (except for length) if necessary
    if equal_shapes and n_samples > 1:
        if any(x.shape[1:] != samples[0].shape[1:] for x in samples[1:]):
            raise ValueError("Sample shapes differ.")

    if ret_list or n_samples > 1:
        return samples
    else:
        return samples[0]


def validate_func(func, *args, **kwargs):
    """Ensure that `func` is either callable or the name of a NumPy array
    method.

    Parameters
    ----------
    func : callable or str
        A function or name of a NumPy array method.
    args : sequence, optional
        Positional arguments to pass to `func`.
    kwargs : dict, optional
        Keyword arguments to pass to `func`.

    Returns
    -------
    func : callable
        Function with the specified positional and keyword arguments already
        supplied.
    """
    if callable(func):
        func = functools.partial(func, *args, **kwargs)
    elif isinstance(func, str):
        if hasattr(np.ndarray, func) and callable(getattr(np.ndarray, func)):
            name = func

            def func(*pos, **kws):
                return getattr(np.asarray(pos[0]), name)(**kws)

            func = functools.partial(func, *args, **kwargs)
        else:
            raise AttributeError(f"NumPy arrays have no method '{func}'")
    else:
        raise ValueError(f"Invalid parameter 'func': {func}")

    return func
