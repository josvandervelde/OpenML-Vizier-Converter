import numpy as np
import pytest as pytest
import vizier.pyvizier as vz

from search_space import _heuristically_determine_scale_type


@pytest.mark.parametrize("expected_scale_type,possible_values", [
    (None, [1]),  # if None, then the parameter will be added as discrete parameter
    (None, [1, 2]),
    (vz.ScaleType.LINEAR, [1, 2, 3]),
    (vz.ScaleType.LINEAR, [1, 2, 3, 4, 5]),
    (vz.ScaleType.LINEAR, [1, 2, 3, 5]),
    (None, [1, 3, 4]),
    (None, [1, 2, 5]),
    (None, [1, 2, 5, 6]),
    (vz.ScaleType.LINEAR, [1, 2, 4, 6, 7, 9, 11]),
    (vz.ScaleType.LINEAR, [1, 2, 4, 5, 7, 8, 9]),
    (None, [1, 2, 5, 6, 7, 8, 9]),
    (vz.ScaleType.LINEAR, np.arange(0.1, 1, 0.1)),
    (vz.ScaleType.LINEAR, [0.1, 0.2, 0.4, 0.5, 0.7, 0.8, 0.9]),
    (vz.ScaleType.LINEAR, [0.1, 0.2, 0.25, 0.3, 0.4, 0.5]),
    (vz.ScaleType.LOG, [1, 2, 4]),
    (None, [11, 12, 14]),
    (vz.ScaleType.LOG, [11, 12, 14, 18]),
    (None, [1, 2, 4, 8, 12]),
    (vz.ScaleType.LOG, [2 ** x for x in range(10)]),
    (vz.ScaleType.LOG, [10 ** x for x in range(-5, 5)]),
    (vz.ScaleType.LOG, [10 + 2 ** x for x in range(10)]),
    (vz.ScaleType.LOG, [10 ** 3 + 10 ** x for x in range(-5, 5)]),
    (vz.ScaleType.REVERSE_LOG, [-(2 ** x) for x in range(10)]),
    (vz.ScaleType.REVERSE_LOG, [10 - (2 ** x) for x in range(10)]),
    (vz.ScaleType.REVERSE_LOG, [10 - (10 ** x) for x in range(-5, 5)])
])
def test_scale_type(expected_scale_type: vz.ScaleType | None, possible_values):
    assert _heuristically_determine_scale_type(possible_values) == expected_scale_type
