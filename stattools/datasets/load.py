"""Functions for loading the data."""

import os
import pathlib

import pandas as pd

# Name of current directory
_cwd = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))


# Get full filename of a dataset
def _full_filename(filename):
    return _cwd.joinpath("data", filename)


def load_lsat_gpa():
    """Load the (LSAT, GPA) dataset. The data consist of 15 pairs of average
    LSAT scores and GPAs for 1973 entering classes of American law schools.

    Returns
    -------
    data: pandas.DataFrame
        pandas DataFrame with LSAT and GPA columns.

    References
    ----------
    Bradley Efron. "Computers and the Theory of Statistics: Thinking the
        Unthinkable". SIAM Review, Volume 21, Number 4 (1979), pp 460--480.
        doi:10.1137/1021092
    """
    return pd.read_csv(_full_filename("lsat_gpa.csv"), header=0)


def load_old_faithful():
    """Load the Old Faithful eruption dataset. The data consist of 272 pairs of
    eruption durations and waiting times until the next eruption of the Old
    Faithful geyser in Yellowstone National Park, Wyoming, USA. Both features
    are measured in minutes.

    Returns
    -------
    data: pandas.DataFrame
        pandas DataFrame with Duration and Wait columns.

    References
    ----------
    A. Azzalini and A. W. Bowman. "A Look at Some Data on the Old Faithful
        Geyser". Applied Statistics, Volume 39, Number 3 (1990), pp 357--365.
        doi:10.2307/2347385
    """
    return pd.read_csv(_full_filename("old_faithful.csv"), header=0)
