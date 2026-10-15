import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "code"))
from conformal_utils import finite_sample_quantile


def test_finite_sample_quantile_uses_order_statistic():
    assert finite_sample_quantile([1.0, 2.0, 3.0, 4.0], 0.8) == 4.0


@pytest.mark.parametrize("n", [1, 2, 3])
def test_finite_sample_quantile_is_unbounded_when_rank_exceeds_sample(n):
    assert np.isinf(finite_sample_quantile(np.arange(1, n + 1), 0.8))


def test_finite_sample_quantile_rejects_empty_calibration():
    with pytest.raises(ValueError):
        finite_sample_quantile([], 0.8)
