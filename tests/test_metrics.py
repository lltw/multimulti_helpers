import pytest
from contextlib import nullcontext as does_not_raise

from re import escape
import numpy as np

from multimulti_helpers.metrics import (
    _check_input_array_type,
    _check_if_valid_probabilities,
)


test_predicted = np.array([[0.1, 0.2, 0.0], [0.9, 0.85, 0.0], [0.15, 0.45, 1.0]])
test_predicted_with_na = np.array(
    [[0.1, np.NaN, 0.0], [0.9, 0.85, 0.0], [0.15, 0.45, 0.0]]
)
test_predicted_with_inf = np.array(
    [[0.1, np.inf, 0.0], [0.9, 0.85, 0.0], [0.15, 0.45, 0.0]]
)


class Test_Checks(object):

    type_error = pytest.raises(
        TypeError,
        match=escape(
            "Input must be a 2-D np.ndarray of dtype 'float' with no missing values."
        ),
    )

    @pytest.mark.parametrize(
        "input_array, expected",
        [
            (test_predicted, does_not_raise()),
            (np.empty((5, 2)), does_not_raise()),  # Empty np.ndarray
            (None, type_error),
            (list(), type_error),
            ([[0.0, 0.1, 0.15], [0.0, 0.95, 0.8]], type_error),
            (test_predicted_with_na, type_error),  # With missing values,
            (np.array([0.0, 0.1, 0.9]), type_error),  # Wrong dimension np.ndarray
            (
                np.array(
                    [
                        [[0.0, 0.15, 0.0], [0.8, 0.6, 0.0]],
                        [[0.0, 0.1, 0.50], [0.35, 0.2, 0.0]],
                    ]
                ),
                type_error,
            ),  # Wrong dimension np.ndarray
        ],
    )
    def test_wrong_types(self, input_array, expected):
        func_name = _check_input_array_type.__name__
        try:
            with expected:
                _check_input_array_type(input_array)
        except Exception as exc:
            exc_name = exc.__class__.__name__
            assert False, f"'{func_name}' raised an exception {exc_name} '{exc}'"

    value_error = pytest.raises(
        ValueError,
        match=escape(
            "Input np.ndarray must contain only values from [0.0, 1.0] interval."
        ),
    )

    float64_resolution = np.finfo(np.float64).resolution
    float32_resolution = np.finfo(np.float32).resolution

    @pytest.mark.parametrize(
        "input_array, float_resolution, expected",
        [
            (test_predicted, 1e-15, does_not_raise()),
            (test_predicted_with_inf, 1e-06, value_error),
            (
                np.array([[-float64_resolution, 0.5], [0.5, 0.5]]),
                1e-15,
                does_not_raise(),
            ),  # Below the [0.0, 1.0] interval by specified float resolution
            (
                np.array([[1.0 + float64_resolution, 0.5], [0.5, 0.5]]),
                1e-15,
                does_not_raise(),
            ),  # Above the [0.0, 1.0] interval by specified float resolution
            (test_predicted_with_inf, 1e-15, value_error),  # Contains np.inf
            (np.array([[-5.0, 0.5], [0.5, 0.5]]), 1e-15, value_error),
            (
                np.array([[5.0, 0.5], [0.5, 0.5]]),
                1e-15,
                value_error,
            ),  # Below the [0.0, 1.0] interval by more than specified float resolution
            (np.array([[5.0, 0.5], [0.5, 0.5]]), 1e-15, value_error),
            (
                np.array([[1.0 + float32_resolution, 0.5], [0.5, 0.5]]),
                1e-15,
                value_error,
            ),  # Above the [0.0, 1.0] interval by more than specified float resolution
        ],
    )
    def test_if_values_are_probabilities(self, input_array, float_resolution, expected):
        func_name = _check_if_valid_probabilities.__name__
        try:
            with expected:
                _check_if_valid_probabilities(input_array, float_resolution)
        except Exception as exc:
            exc_name = exc.__class__.__name__
            assert False, f"'{func_name}' raised an exception {exc_name} '{exc}'"
