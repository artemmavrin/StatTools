"""Functions for data validation."""

import numbers

import numpy as np


def validate_data(*data, max_ndim=None, equal_lengths=False,
                  equal_shapes=False, return_list=False):
    """Apply necessary preprocessing and validation to input data.

    Parameters
    ----------
    data : sequence of arrays
        List of data to preprocess and validate.
    max_ndim : int or tuple of 1's, 2's, or Nones, optional
        Maximum number of dimensions for each array in `data` (either 1, 2, or
        None). If `max_ndim` is an int, each array in `data` must have the same
        maximum number of dimensions. If `max_ndim` is a tuple, each non-None
        item in the tuple represents the maximum number of dimensions of the
        array in `data` at the same index. In either case, the arrays in `data`
        are coerced to have the maximum number of dimensions.
    equal_lengths : bool, optional
        Indicates whether each array in `data` should have the same length.
    return_list : bool, optional
        Indicates whether the data should be returned as a list no matter what,
        even if it contains only one array.

    Returns
    -------
    data : sequence of numpy.ndarrays, or single numpy.ndarray
        If multiple arrays are provided, a list containing the modifications
        of each one is returned. Otherwise, the single modified array is
        returned.
    """
    # Ensure some data is provided
    if len(data) == 0:
        raise ValueError("No data provided.")

    # Coerce each array in `data` into a NumPy array
    data = list(map(np.atleast_1d, data))

    # Check for number of dimensions if necessary
    if max_ndim is not None:
        # If `max_ndim` is an int, convert it to a tuple
        if isinstance(max_ndim, numbers.Integral):
            max_ndim = tuple(max_ndim for _ in data)

        if isinstance(max_ndim, tuple):
            # Validate `max_ndim` as a tuple of 1's, 2's, and None's
            if any(not isinstance(d, numbers.Integral) for d in max_ndim
                   if d is not None):
                raise TypeError("Non-integer maximum number of dimensions.")
            if any(d not in (1, 2) for d in max_ndim if d is not None):
                raise ValueError("Maximum number of dimensions can be 1 or 2.")
            if len(max_ndim) != len(data):
                raise ValueError("Incompatible `data` and `max_ndim`.")

            for i, (x, d) in enumerate(zip(data, max_ndim)):
                if d is None:
                    continue
                if x.ndim < d == 2:
                    data[i] = np.atleast_2d(x).T
                elif x.ndim > d:
                    raise ValueError(
                        f"Array at index {i} has too many dimensions")
        else:
            raise TypeError("Parameter `max_ndim` must be an int or a tuple.")

    # Check for equal array lengths if necessary
    if equal_lengths and any(len(x) != len(data[0]) for x in data[1:]):
        raise ValueError(
            "Each array in the data must have the same length.")

    # Check for equal array shapes (except for length) if necessary
    if equal_shapes and any(x.shape[1:] != data[0].shape[1:] for x in data[1:]):
        raise ValueError("Incompatible data array shapes.")

    if return_list or len(data) > 1:
        return data
    else:
        return data[0]
