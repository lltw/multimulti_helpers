import pytest

from re import escape
import numpy as np
import pandas as pd

from multimulti_helpers.sample_generation import (
    _check_y_labels_min_counts,
    _generate_minimal_sample_indices,
)

# Prepare test pd.DataFrames and np.ndarrays
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

y_labels_2 = y_labels_2_pd.to_numpy()
y_labels_1 = y_labels_1_pd.to_numpy()


class Test_CheckYLabelsMinCounts(object):
    @pytest.mark.parametrize(
        "y_labels, min_count, expected",
        [
            (y_labels_1, 1, True),
            (y_labels_2, 1, True),
        ],
    )
    def test_each_column_have_enough_counts(self, y_labels, min_count, expected):
        func_name = _check_y_labels_min_counts.__name__
        try:
            actual = _check_y_labels_min_counts(y_labels, min_count)
            msg = f"'{func_name}' returned {actual}, expected {expected}."
            assert actual is expected, msg
        except Exception as exc:
            assert False, f"'{func_name}' raised an exception '{exc}'"

    @pytest.mark.parametrize(
        "y_labels,min_count,expectation",
        [
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
    def test_at_least_one_column_has_not_enough_counts(
        self, y_labels, min_count, expectation
    ):
        func_name = _check_y_labels_min_counts.__name__
        try:
            with expectation:
                _check_y_labels_min_counts(y_labels, min_count)
        except Exception as exc:
            assert False, f"'{func_name}' raised an exception '{exc}'"


class Test_GenerateMinimalSampleIndices(object):
    @pytest.mark.parametrize(
        "y_labels, min_count, expected",
        [
            (y_labels_2, 1, {5, 6}),  # Marginal without seed
        ],
    )
    def test_without_seed(self, y_labels, min_count, expected):
        func_name = _generate_minimal_sample_indices.__name__
        try:
            actual = set(_generate_minimal_sample_indices(y_labels, min_count))
            msg = f"'{func_name}' returned {actual}, expected value shoud be"
            f" in {expected}."
            assert actual.issubset(expected), msg
        except Exception as exc:
            assert False, f"'{func_name}' raised an exception '{exc}'"

    @pytest.mark.parametrize(
        "y_labels, min_count, rng, expected",
        [
            (y_labels_1, 1, 42, np.array([8, 9, 11])),  # Marginal with seed
            (y_labels_1, 2, 42, np.array([1, 7, 8, 9, 10, 11])),  # Normal with seed
        ],
    )
    def test_with_seed(self, y_labels, min_count, rng, expected):
        func_name = _generate_minimal_sample_indices.__name__
        try:
            actual = _generate_minimal_sample_indices(y_labels, min_count, rng)
            msg = f"'{func_name}' returned {actual}, expected {expected}."
            assert np.array_equal(actual, expected), msg
        except Exception as exc:
            assert False, f"'{func_name}' raised an exception '{exc}'"
