import pytest
from contextlib import nullcontext as does_not_raise

from re import escape
import numpy as np
import pandas as pd

from multimulti_helpers.sample_generation import (
    _validate_y_labels_type,
    _check_if_y_labels_is_binary,
    _validate_sample_size,
    _check_y_labels_min_counts,
    _generate_minimal_sample_indices,
    multilabel_sample,
)

# Prepare normal test pd.DataFrames and np.ndarrays
y_labels_1_pd = pd.DataFrame(
    {
        "x_1": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
        "x_2": [1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
        "x_3": [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1],
    }
)
y_labels_2_pd = pd.DataFrame(
    {
        "x_1": [0, 0, 0, 0, 0, 1, 1, 0],
    }
)

y_labels_df_with_na = pd.DataFrame({"x_1": [0, np.NaN, 0], "x_2": [0, 1, 0]})
y_labels_with_na = y_labels_df_with_na.to_numpy()

y_labels_1_df_category = y_labels_1_pd.astype("category")
y_labels_1_df_category_and_int = y_labels_1_pd.astype(
    {"x_1": "int64", "x_2": "category", "x_3": "category"}
)

y_labels_1 = y_labels_1_pd.to_numpy(dtype="int64")
y_labels_2 = y_labels_2_pd.to_numpy(dtype="int64")


class Test_Checks(object):
    """
    Test input checks in multimulti_helpers.sample_generation.multilabel_sample
    function.
    """

    @pytest.mark.parametrize(
        "y_labels",
        [
            (y_labels_1_pd),  # Normal pd.DataFrame with dtypes 'int64'
            (y_labels_1_df_category),  # Normal pd.DataFrame with dtype 'category'
            (
                y_labels_1_df_category_and_int
            ),  # Normal pd.DataFrame with dtypes 'int64' and 'category'
            (y_labels_2_pd),  # Normal pd.DataFrame
        ],
    )
    def test_df_dataframe_to_np_ndarray(self, y_labels):
        func_name = _validate_y_labels_type.__name__
        try:
            actual = _validate_y_labels_type(y_labels)
            assert isinstance(actual, np.ndarray)
            assert len(actual.shape) == 2
            assert actual.dtype.name == "int64"
        except Exception as exc:
            exc_name = exc.__class__.__name__
            assert False, f"'{func_name}' raised an exception {exc_name} '{exc}'"

    type_error = pytest.raises(
        TypeError,
        match=escape(
            "y_labels must have no missing values and be a non-empty pd.DataFrame"
            + " with columns of dtypes 'category' or 'int', or a 2-D np.ndarray of"
            + " dtype 'int64'."
        ),
    )

    @pytest.mark.parametrize(
        "y_labels, expected",
        [
            (y_labels_1, does_not_raise()),
            (y_labels_2, does_not_raise()),
            (None, type_error),
            (list(), type_error),
            ([[0, 1, 1], [0, 0, 1]], type_error),
            (pd.DataFrame(), type_error),  # Empty pd.DataFrame
            (np.empty((5, 2)), type_error),  # Empty np.ndarray
            (y_labels_df_with_na, type_error),  # pd.DataFrame with missing values
            (y_labels_with_na, type_error),  # np.ndarray with missing values,
            (np.array([0, 1, 1]), type_error),  # Wrong dimension np.ndarray
            (
                np.array([[[0, 1, 0], [1, 0, 0]], [[0, 1, 0], [0, 1, 0]]]),
                type_error,
            ),  # Wrong dimension np.ndarray
            (np.array([[0, 1.0, 1], [0, 1, 0]]), type_error),  # Wrong type np.ndarray
        ],
    )
    def test_wrong_types(self, y_labels, expected):
        func_name = _validate_y_labels_type.__name__
        try:
            with expected:
                _validate_y_labels_type(y_labels)
        except Exception as exc:
            exc_name = exc.__class__.__name__
            assert False, f"'{func_name}' raised an exception {exc_name} '{exc}'"

    value_error_not_binary = pytest.raises(
        ValueError,
        match=escape("y_labels must a 2-D np.ndarray consisting only of 0's and 1's."),
    )

    @pytest.mark.parametrize(
        "y_labels, expected",
        [
            (y_labels_1, does_not_raise()),
            (y_labels_2, does_not_raise()),
            (np.array([[1, 2], [0, 1]]), value_error_not_binary),
        ],
    )
    def test_not_binary(self, y_labels, expected):
        func_name = _check_if_y_labels_is_binary.__name__
        try:
            with expected:
                _check_if_y_labels_is_binary(y_labels)
        except Exception as exc:
            exc_name = exc.__class__.__name__
            assert False, f"'{func_name}' raised an exception {exc_name} '{exc}'"

    wrong_size_exp_msg_1 = (
        "Sample size ({size}) is to small to always ensure min_count instances of"
        + " each binary class. Sample size should be at least"
        + " {min_count_times_y_labels_shape_1} (min_count ({min_count}) * number"
        + " of different classes ({y_labels_shape_1}))."
    )
    wrong_size_exp_msg_2 = (
        "Sample size ({size}) exceeds that number of rows in original data"
        + " ({y_labels_shape_0})."
    )

    @pytest.mark.parametrize(
        "y_labels, size, min_count, expected",
        [
            (y_labels_1, 3, 1, does_not_raise()),  # Marginal
            (y_labels_1, 6, 2, does_not_raise()),
            (y_labels_2, 1, 1, does_not_raise()),
            (y_labels_2, 2, 2, does_not_raise()),
            (y_labels_1, 14, 2, does_not_raise()),
            (  # Not enough
                y_labels_1,
                5,
                2,
                pytest.raises(
                    ValueError,
                    match=escape(
                        wrong_size_exp_msg_1.format(
                            size=5,
                            min_count_times_y_labels_shape_1=6,
                            min_count=2,
                            y_labels_shape_1=3,
                        )
                    ),
                ),
            ),
            (  # Too much
                y_labels_1,
                15,
                2,
                pytest.raises(
                    ValueError,
                    match=escape(
                        wrong_size_exp_msg_2.format(
                            size=15,
                            y_labels_shape_0=14,
                        )
                    ),
                ),
            ),
        ],
    )
    def test_wrong_size(self, y_labels, size, min_count, expected):
        func_name = _validate_sample_size.__name__
        try:
            with expected:
                _validate_sample_size(y_labels, size, min_count)
        except Exception as exc:
            exc_name = exc.__class__.__name__
            assert False, f"'{func_name}' raised an exception {exc_name} '{exc}'"

    @pytest.mark.parametrize(
        "y_labels,min_count,expected",
        [
            (y_labels_1, 1, does_not_raise()),
            (y_labels_2, 1, does_not_raise()),
            (
                y_labels_1,
                3,
                pytest.raises(
                    ValueError,
                    match=escape(
                        "Column 0 of y_labels has not enough (less than min_count)"
                        + " instances of it's class."
                    ),
                ),
            ),
            (
                y_labels_1,
                4,
                pytest.raises(
                    ValueError,
                    match=escape(
                        "Columns 0, 2 of y_labels have not enough (less than min_count)"
                        + " instances of their respective classes."
                    ),
                ),
            ),
            (
                y_labels_2,
                3,
                pytest.raises(
                    ValueError,
                    match=escape(
                        "Column 0 of y_labels has not enough (less than min_count)"
                        + " instances of it's class."
                    ),
                ),
            ),
        ],
    )
    def test_underrepresented_classes_in_y(self, y_labels, min_count, expected):
        func_name = _check_y_labels_min_counts.__name__
        try:
            with expected:
                _check_y_labels_min_counts(y_labels, min_count)
        except Exception as exc:
            exc_name = exc.__class__.__name__
            assert False, f"'{func_name}' raised an exception {exc_name} '{exc}'"


class Test_GenerateMinimalSampleIndices(object):
    @pytest.mark.parametrize(
        "y_labels, min_count, expected",
        [
            (y_labels_2, 1, {5, 6}),  # Marginal without seed
            (y_labels_1, 1, {0, 1, 6, 7, 8, 9, 10, 11, 13}),  # Marginal without seed
        ],
    )
    def test_minimal_sample_indices_without_seed(self, y_labels, min_count, expected):
        func_name = _generate_minimal_sample_indices.__name__
        try:
            actual = set(_generate_minimal_sample_indices(y_labels, min_count))
            msg = f"'{func_name}' returned {actual}, expected value shoud be"
            f" in {expected}."
            assert actual.issubset(expected), msg
        except Exception as exc:
            exc_name = exc.__class__.__name__
            assert False, f"'{func_name}' raised an exception {exc_name} '{exc}'"

    @pytest.mark.parametrize(
        "y_labels, min_count, rng, expected",
        [
            (y_labels_1, 1, 42, np.array([8, 9, 11])),  # Marginal with seed
            (y_labels_1, 2, 42, np.array([1, 7, 8, 9, 10, 11])),  # Normal with seed
        ],
    )
    def test_minimal_sample_indices_with_seed(self, y_labels, min_count, rng, expected):
        func_name = _generate_minimal_sample_indices.__name__
        try:
            actual = _generate_minimal_sample_indices(y_labels, min_count, rng)
            msg = f"'{func_name}' returned {actual}, expected {expected}."
            assert np.array_equal(actual, expected), msg
        except Exception as exc:
            exc_name = exc.__class__.__name__
            assert False, f"'{func_name}' raised an exception {exc_name} '{exc}'"


class Test_MultilabelSample(object):
    @pytest.mark.parametrize(
        "y_labels, size, min_count, seed, expected",
        [
            (y_labels_1, 0.5, 2, 42, np.array([1, 2, 7, 8, 9, 10, 11])),
            (y_labels_1, 8, 2, 42, np.array([0, 1, 2, 7, 8, 9, 10, 11])),
        ],
    )
    def test_multilabel_sample_with_seed(
        self, y_labels, size, min_count, seed, expected
    ):
        func_name = multilabel_sample.__name__
        try:
            actual = multilabel_sample(y_labels, size, min_count, seed)
            msg = f"'{func_name}' returned {actual}, expected {expected}."
            assert np.array_equal(actual, expected), msg
        except Exception as exc:
            exc_name = exc.__class__.__name__
            assert False, f"'{func_name}' raised an exception {exc_name} '{exc}'"

    @pytest.mark.parametrize(
        "y_labels, size, min_count, expected_size",
        [
            (y_labels_1, 0.6, 2, 8),
            (y_labels_1, 10, 2, 10),
        ],
    )
    def test_multilabel_sample_without_seed(
        self, y_labels, size, min_count, expected_size
    ):
        func_name = multilabel_sample.__name__
        try:
            actual = multilabel_sample(y_labels, size, min_count, seed=None)
            assert actual.shape[0] == expected_size
            assert _check_y_labels_min_counts(y_labels_1[actual], min_count) is None
        except Exception as exc:
            exc_name = exc.__class__.__name__
            assert False, f"'{func_name}' raised an exception {exc_name} '{exc}'"
