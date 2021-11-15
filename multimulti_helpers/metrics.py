import numpy as np
import numpy.typing as npt

from multimulti_helpers.sample_generation import _check_if_y_labels_is_binary


def _check_input_array_type(input_array: npt.ArrayLike) -> None:
    """
    Checks if input is a 2-D np.ndarray of dtype 'float' with no missing values.

    Parameters
    ----------
    predicted: array_like
        2-D np.ndarray

    Raises
    ------
    TypeError
        If the input is not or 2-D np.ndarray of dtype 'float' with no missing values.
    """

    is_valid_input_array = (
        isinstance(input_array, np.ndarray)
        and (len(input_array.shape) == 2)
        and input_array.dtype.name.startswith("float")
        and (sum(np.concatenate(np.isnan(input_array))) == 0)
    )

    if not is_valid_input_array:
        raise TypeError(
            "Input must be a 2-D np.ndarray of dtype 'float' with no missing values."
        )


def _check_if_valid_probabilities(
    input_array: npt.ArrayLike, float_resolution: float = 1e-15
) -> None:
    """
    Checks if values of input np.ndarray are within [0.0, 1.0] interval.

    Parameters
    ----------
    predicted: array_like
        2-D np.ndarray of the predictions (probabilities: [0.0, 1.0])
    float_resolution: float, default = 1e-15
        approximate decimal resolution of floats, default for np.float64

    Raises
    ------
    ValueError
        If input np.ndarray must contains a value from outside [0.0, 1.0] interval.
    """

    for value in np.nditer(input_array):
        if value > 1.0 + float_resolution or value < 0.0 - float_resolution:
            raise ValueError(
                "Input np.ndarray must contain only values from [0.0, 1.0] interval."
            )


def multi_multi_log_loss(
    predicted,
    actual,
    class_column_indices,
    float_resolution: float = 1e-15,
    check_input=False,
):
    """
    Multi-class, multi-label version of Logarithmic Loss metric (the average of
    Logarithmic Losses calculated for each label).

    Parameters
    ----------
    predicted: array_like
        2-D np.ndarray of the predictions (probabilities [0, 1])
    actual:  array_like
        2-D np.ndarray of 0's and 1's representing multiple binary classes
        (each column is a binary class) of the same shape as the predictions.
    class_column_indices: array_like
        1-D list-like object populated with range objects created as follows:
            (1) the first range object starts as zero,
            (2) other range object starts where previous range object ends
            (3) range object have lengths equal to number of classes within
                corresponding labels
    float_resolution: float, default = 1e-15
        approximate decimal resolution of floats, default for np.float64

    Returns
    -------
    float
        The multi-multi log loss score for this set of predictions.

    Raises
    ------
    TypeError
            If shapes of predicted array and actual array differ.

    """

    if check_input:
        # check if 'predicted' and 'actual' are 2-D np.ndarray of dtype 'float'
        # with no missing values
        _check_input_array_type(predicted)
        _check_input_array_type(actual)

        # check if np.ndarray 'predicted' contains only probabilities
        _check_if_valid_probabilities(predicted, float_resolution)

        # check if np.ndarray 'actual' is binary
        _check_if_y_labels_is_binary(actual)

    # check if 'predicted' and 'actual' np.ndarrays have the same shape
    if not (predicted.shape == actual.shape):
        raise TypeError(
            f"Shapes of predicted array {predicted.shape} and actual array"
            + f" {actual.shape} differ."
        )

    label_scores = np.ones(len(class_column_indices), dtype=np.float64)

    # calculate log loss for each set of columns that belong to a class:
    for k, this_label_classes_indices in enumerate(class_column_indices):

        # get just the class columns for this label
        preds_k = predicted[:, this_label_classes_indices]

        # normalize so probabilities sum to one (unless sum is zero, then we clip)
        preds_k /= np.clip(preds_k.sum(axis=1).reshape(-1, 1), float_resolution, np.inf)

        actual_k = actual[:, this_label_classes_indices]

        # shrink predictions
        y_hats = np.clip(preds_k, float_resolution, 1 - float_resolution)

        # calculate the log loss funtioon for the label
        sum_logs = np.sum(actual_k * np.log(y_hats))
        label_scores[k] = (-1.0 / actual.shape[0]) * sum_logs

    return np.average(label_scores)
