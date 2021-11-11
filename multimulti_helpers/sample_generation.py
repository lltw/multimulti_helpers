import numpy as np
import numpy.typing as npt
import pandas as pd


def _validate_y_labels_type(y_labels: npt.ArrayLike) -> npt.NDArray[np.int_]:
    """
    Checks if y_label is valid pd.DataFrame or np.ndarray.

    Valid pd.DataFrame is non-empty, it's columns are of dtypes "category" or
    "int64" and has no missing values. Valid np.ndarray is 2-D, of dtype "int64"
    and has no missing values.

    Parameters
    ----------
    y_labels: array_like

    Returnes
    --------
    np.ndarray
        2-D np.ndarray dtype "int64"

    Raises
    ------
    TypeError
        If the y_label is not non-empty pd.DataFrame, with no missing values, which
        columns are of dtypes "category" or "int64", or 2-D np.ndarray of dtype "int64"
        with no missing values.
    """
    is_valid_df = (
        isinstance(y_labels, pd.DataFrame)
        and (not y_labels.empty)
        and (y_labels.isna().any().sum() == 0)
        and set(x.name for x in y_labels.dtypes).issubset({"category", "int64"})
    )
    is_valid_ndarray = (
        isinstance(y_labels, np.ndarray)
        and (len(y_labels.shape) == 2)
        and y_labels.dtype.name == "int64"
        and (sum(np.concatenate(np.isnan(y_labels))) == 0)
    )

    if not (is_valid_df or is_valid_ndarray):
        raise TypeError(
            "y_labels must have no missing values and be a non-empty pd.DataFrame"
            + " with columns of dtypes 'category' or 'int', or a 2-D np.ndarray of"
            + " dtype 'int64'."
        )

    if isinstance(y_labels, pd.DataFrame):
        y_labels = y_labels.to_numpy(dtype="int64")

    return y_labels


def _check_if_y_labels_is_binary(y_labels: npt.ArrayLike) -> None:
    """
    Checks if y_labels is a 2-D np.ndarray consisting only of 0's and 1's.

    Parameters
    ----------
    y_labels: array_like

    Raises
    ------
    ValueError
        If the y_label is not or 2-D np.ndarray of dtype "int64" consisting only
        of of 0's and 1's.
    """
    if not set(np.unique(y_labels).astype(int)).issubset({0, 1}):
        raise ValueError(
            "y_labels must a 2-D np.ndarray consisting only of 0's and 1's."
        )


def _validate_sample_size(
    y_labels: npt.ArrayLike, size: float or int, min_count: int
) -> int:
    """
    If size =< 1, converts it to floor of size times number of rows in y_labels.
    Checks if specified size is equal to or greater than min_count times number of rows
    in y_labels.

    Parameters
    ----------
    y_labels: array_like
        2-D np.ndarray consisting only of 0's and 1's
    size: float
        Size of a sample. Sizes <= 1 are treated as fractions of original sample size.
        Should be at least min_count times number of columns in the original data.
    min_count: int, default: 5
        Minimum count of instances of each binary class to be insured in a sample.

    Returns:
    int
        size of sample

    Raises
    ------
    ValueError
        If size < min_count * y_label.shape[1] or if size > y_labels.shape[0]
    """

    if size <= 1:
        size = int(np.floor(len(y_labels) * size))

    if size < min_count * y_labels.shape[1]:
        raise ValueError(
            f"Sample size ({size}) is to small to always ensure min_count instances of"
            + " each binary class. Sample size should be at least"
            + f" {min_count * y_labels.shape[1]} (min_count ({min_count}) * number"
            + f" of different classes ({y_labels.shape[1]}))."
        )
    elif size > y_labels.shape[0]:
        raise ValueError(
            f"Sample size ({size}) exceeds that number of rows in"
            + f" original data ({y_labels.shape[0]}). "
        )

    return size


def _check_y_labels_min_counts(y_labels: npt.ArrayLike, min_count: int = 5) -> None:
    """
    Checks if each binary class in y_lables has at least min_count instances,
    if not raises a ValueError.

    Parameters
    ----------
    y_labels: array_like
        2-D np.ndarray of 0's and 1's representing multiple binary classes (each column
        is a binary class)
    min_count: int, default: 5
        Minimum count of instances of each binary class to be insured in a sample.

    Raises:
    -------
    ValueError
        If there is at least one class in y_labels with less than min_count.
    """

    # check if each binary class in y_lables np.ndarray has at least min_count
    # instances if not, store the index of a column corresponding to the
    # underrepresented class
    underrepresented_classes = []
    for i, column in zip(range(y_labels.shape[1]), y_labels.T):
        if column.sum() < min_count:
            underrepresented_classes.append(i)

    # if there are underrepresented classes, throw an error
    if len(underrepresented_classes) == 1:
        raise ValueError(
            "Column "
            + str(underrepresented_classes[0])
            + " of y_labels has not enough (less than min_count)"
            + " instances of it's class."
        )
    elif len(underrepresented_classes) > 1:
        raise ValueError(
            "Columns "
            + ", ".join([str(x) for x in underrepresented_classes])
            + " of y_labels have not enough (less than min_count) instances"
            + " of their respective classes."
        )


def _generate_minimal_sample_indices(
    y_labels: npt.ArrayLike, min_count: int = 5, rng=None
) -> npt.NDArray[np.int_]:
    """
    Generates minimal random sample with the count of each label >= `min_count`.

    Parameters
    ----------
    y_labels: array_like
        2-D np.ndarray of 0's and 1's representing multiple binary classes (column
        is a binary class)
    min_count: int, default: 5
        Minimum count of instances of each binary class to be insured in a sample.
    rng: Generator instance or int, default: None
        Generator instance or seed (int) to create a Generator.

    Returns
    -------
    ndarray
        1-D ndarray of unique sample indices.
    """
    rng = np.random.default_rng(rng)

    ensured_max_size = min_count * y_labels.shape[1]
    ensured_sample_idx = np.empty(ensured_max_size, dtype="int")

    # for each label sample `min_count` indices from indices of instances of class 1's
    for i, column in zip(range(y_labels.shape[1]), y_labels.T):
        ensured_sample_idx[i * min_count : (i + 1) * min_count] = rng.choice(
            np.asarray(column == 1).nonzero()[0], size=min_count, replace=False
        )

    return np.unique(ensured_sample_idx)


def multilabel_sample(
    y_labels: npt.ArrayLike,
    size: float or int = 1000,
    min_count: int = 5,
    seed: int = None,
) -> npt.NDArray[np.int_]:
    """
    Takes a pd.DataFrame or np.ndarray matrix of binary labels `y_lables`
    and returns sorted np.ndarray of indices for the sample of size `size`
    if `size` > 1 or floor(`size` * len(y)) if size =< 1.
    The sample is guaranteed to have > `min_count` of each label.

    Parameters
    ----------
    y_labels: array_like
        2-D array (pd.DataFrame or np.ndarray) of 0's and 1's representing multiple
        binary classes (each column is a binary class)
    size: float, default: 1000
        Size of a sample. Sizes <= 1 are treated as fractions of original sample size.
        Should be at least min_size_min_count_times greater than min_count times number
        of columns in the original data.
    min_count: int, default: 5
        Minimum count of instances of each binary class to be insured in a sample.
    seed: int, default: None
        Seed for random sampling.

    Returns
    -------
    ndarray
        Sorted 1d ndarray of sample indices.
    """

    # check if y_labels is a valid pd.DataFrame or np.ndarray
    # if y_labels is pd.DataFrame, convert it to ndarray
    y_labels = _validate_y_labels_type(y_labels)

    # check if y_labels is binary
    _check_if_y_labels_is_binary(y_labels)

    # check if size isn't too small and convert it to the integer
    size = _validate_sample_size(y_labels, size, min_count)

    # check if each class in y_lables has at least min_count instances
    _check_y_labels_min_counts(y_labels, min_count)

    # set seed and  rng
    seed = np.random.randint(100) if seed is None else seed
    rng = np.random.default_rng(seed)

    # generate sample indices:
    # step I: ensure count of each label >= `min_count`
    ensured_sample_idx = _generate_minimal_sample_indices(
        y_labels=y_labels, min_count=min_count, rng=rng
    )

    # step II: generate remaining indices
    remaining_size = size - ensured_sample_idx.shape[0]
    remaining_idx = np.setdiff1d(
        np.arange(0, y_labels.shape[0]), ensured_sample_idx, assume_unique=True
    )
    remaining_sample_idx = rng.choice(remaining_idx, size=remaining_size, replace=False)

    # concatenate and then sort ensured and remaining indices
    sample_idx = np.concatenate((ensured_sample_idx, remaining_sample_idx))
    sample_idx.sort()

    return sample_idx
