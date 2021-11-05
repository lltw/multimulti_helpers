import pytest

from re import escape
import numpy as np
import pandas as pd

from multimulti_helpers.sample_generation import (
    _check_y_labels_min_counts,
    _generate_minimal_sample_indices,
)


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

seed = 42


class Test_CheckYLabelsMinCounts(object):
    def test__check_y_labels_min_counts_1(self):
        func_name = _check_y_labels_min_counts.__name__
        try:
            actual = _check_y_labels_min_counts(y_labels_1, min_count=1)
            expected = None
            msg = f"'{func_name}' returned {actual}, expected {expected}."
            assert actual is expected, msg
        except Exception as exc:
            assert False, f"'{func_name}' raised an exception '{exc}'"

    def test__check_y_labels_min_counts_2(self):
        func_name = _check_y_labels_min_counts.__name__
        try:
            actual = _check_y_labels_min_counts(y_labels_2, min_count=1)
            expected = None
            msg = f"'{func_name}' returned {actual}, expected {expected}."
            assert actual is expected, msg
        except Exception as exc:
            assert False, f"'{func_name}' raised an exception '{exc}'"

    def test__check_y_labels_min_counts_3(self):
        exc_msg = escape(
            "Column 0 of y_labels has not enough (less than min_count)"
            + " instances of it's class."
        )
        with pytest.raises(ValueError, match=exc_msg):
            _check_y_labels_min_counts(y_labels_1, min_count=3)

    def test__check_y_labels_min_counts_4(self):
        exc_msg = escape(
            "Columns 0, 2 of y_labels have not enough (less than min_count) instances"
            + " of their respective classes."
        )
        with pytest.raises(ValueError, match=exc_msg):
            _check_y_labels_min_counts(y_labels_1, min_count=4)


class Test_GenerateMinimalSampleIndices(object):

    # Marginal without seed
    def test__generate_minimal_sample_indices_1(self):
        func_name = _generate_minimal_sample_indices.__name__
        try:
            actual = set(
                _generate_minimal_sample_indices(y_labels=y_labels_2, min_count=1)
            )
            expected = {5, 6}
            msg = f"'{func_name}' returned {actual}, expected value shoud be in"
            +" {expected}."
            assert actual.issubset(expected), msg
        except Exception as exc:
            assert False, f"'{func_name}' raised an exception '{exc}'"

    # Marginal with seed
    def test__generate_minimal_sample_indices_2(self, rng=42):
        func_name = _generate_minimal_sample_indices.__name__
        try:
            actual = _generate_minimal_sample_indices(
                y_labels=y_labels_1, min_count=1, rng=rng
            )
            expected = np.array([8, 9, 11])
            msg = f"'{func_name}' returned {actual}, expected {expected}."
            assert np.array_equal(actual, expected), msg
        except Exception as exc:
            assert False, f"'{func_name}' raised an exception '{exc}'"

    # Normal with seed
    def test__generate_minimal_sample_indices_3(self, rng=42):
        func_name = _generate_minimal_sample_indices.__name__
        try:
            actual = _generate_minimal_sample_indices(
                y_labels=y_labels_1, min_count=2, rng=rng
            )
            expected = np.array([1, 7, 8, 9, 10, 11])
            msg = f"'{func_name}' returned {actual}, expected {expected}."
            assert np.array_equal(actual, expected), msg
        except Exception as exc:
            assert False, f"'{func_name}' raised an exception '{exc}'"
