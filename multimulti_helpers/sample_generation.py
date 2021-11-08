import numpy as np
import numpy.typing as npt
import pandas as pd


def _check_y_labels_min_counts(y_labels: npt.ArrayLike, min_count: int = 5) -> bool:
    """
    Checks if each binary class in y_lables has at least min_count instances,
    if not raises a ValueError.
    """

    # check if each binary class in y_lables np.ndarray has at least min_count
    # instances if not, store the index of a column corresponding to the
    # underrepresented class
    underrepresented_classes = []
    for i, column in zip(range(y_labels.shape[1]), y_labels.T):
        if column.sum() < min_count:
            underrepresented_classes.append(i)

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

    return True


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
    size: float = 1000.0,
    min_count: int = 5,
    min_size_min_count_times: int = 100,
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
        binary classes (column is a binary class)
    size: float, default: 1000
        Size of a sample. Sizes <= 1 are treated as fractions of original sample size.
        Should be at least min_size_min_count_times greater than min_count times number
        of columns in the original data.
    min_count: int, default: 5
        Minimum count of instances of each binary class to be insured in a sample.
    min_size_min_count_times: int, default: 5
        Coefficient to control for minimum sample size. Sample size should be >=
        min_size_min_count_times * (min_count * number of columns in the original data).
    seed: int, default: None
        Seed for random sampling.

    Returns
    -------
    ndarray
        Sorted 1d ndarray of sample indices.

    Raises
    ------
    TypeError
        If the y_label is not of type pd.DataFrame or np.ndarray

    ValueError
        If y_labels is not a of 0's and 1's

    ValueError
        If size < min_size_min_count_times * min_count * y_label.shape[1]
    """

    # check if y_labels is pd.DataFrame or np.ndarray
    if not (isinstance(y_labels, pd.DataFrame) or isinstance(y_labels, np.ndarray)):
        raise TypeError("y_label must be of type pd.DataFrame or np.ndarray")

    # check if y_labels is binary
    if set(np.unique(y_labels).astype(int)) not in {0, 1}:
        raise ValueError("y_labels must be a matrix of 0's and 1's")

    # if y_labels is pd.DataFrame, convert it to ndarray
    if isinstance(y_labels, pd.DataFrame):
        y_labels = y_labels.to_numpy()

    # convert size to integer
    if size <= 1:
        size = np.floor(len(y_labels) * size)

    # check if size isn't too small
    msg = (
        f"Specified sample size {size} is less than min_size_min_count_times"
        + f" ({min_size_min_count_times}) * min_count ({min_count})"
        + f" * number of different classes ({y_labels.shape[1]})"
        + f" = {min_size_min_count_times * min_count * y_labels.shape[1]}."
    )

    if size < min_size_min_count_times * min_count * y_labels.shape[1]:
        raise ValueError(msg)

    # check if each class in y_lables has at least min_count instances
    _check_y_labels_min_counts(y_labels, min_count)

    # set seed and  rng
    seed = np.random.randint(100) if seed is None else seed
    rng = np.random.default_rng(seed)

    # generate sample indices:
    # step I: ensure count of each label >= `min_count`
    ensured_sample_idx = _generate_minimal_sample_indices(
        y_labels=y_labels, min_count=min_count, seed=seed
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
