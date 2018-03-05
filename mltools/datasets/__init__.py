"""Some small datasets."""

import os
from collections import namedtuple

import numpy as np
import pandas as pd

# Name of current directory
_dir = os.path.dirname(os.path.realpath(__file__))


# Get full filename
def _full_fname(fname):
    return _dir + "/data/" + fname


def load_lsat(df=True):
    """Load the (LSAT, GPA) data from
        Efron, Bradley. Computers and the theory of statistics: thinking the
        unthinkable. SIAM Rev. 21 (1979), no. 4, 460–480.
        DOI: https://doi.org/10.1137/1021092

    This data consist of n=15 samples of average LSAT scores and GPAs for 1973
    entering classes of American law schools.

    Parameters
    ----------
    df: bool, optional
        If True, return the data as a pandas DataFrame with columns "LSAT" and
        "GPA".
        If False, return the LSAT and GPA vectors as two NumPy arrays

    Returns
    -------
    data: pandas.DataFrame
        pandas DataFrame with LSAT and GPA columns (returned if `df` is True).
    lsat: numpy.ndarray
        NumPy array of average LSAT scores (returned if `df` is False).
    gpa: numpy.ndarray
        NumPy array of average GPAs (returned if `df` is False).
    """
    fname = "lsat.txt"
    data = pd.read_table(_full_fname(fname), header=0, delim_whitespace=True)
    if df:
        return data
    else:
        LSATData = namedtuple("LSATData", ("lsat", "gpa"))
        return LSATData(lsat=np.asarray(data.LSAT), gpa=np.asarray(data.GPA))
