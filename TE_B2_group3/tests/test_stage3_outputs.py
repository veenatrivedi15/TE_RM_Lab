from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).parents[1]


def _check_intervals(path, actual, lower, upper, width, covered):
    artifact = ROOT / path
    if not artifact.exists():
        pytest.skip(f"generated reference artifact not present: {artifact}")
    frame = pd.read_csv(artifact)
    required = {actual, lower, upper, width, covered}
    assert required <= set(frame.columns)
    assert frame[lower].ge(0).all()
    assert frame[upper].ge(frame[lower]).all()
    assert np.allclose(frame[width], frame[upper] - frame[lower])
    expected = frame[actual].between(frame[lower], frame[upper])
    assert frame[covered].astype(bool).equals(expected)
    return frame


def test_adaptive_row_artifact_is_reproducible():
    frame = _check_intervals(
        "results/preds/stage3_adaptive_intervals.csv",
        "actual", "lo", "hi", "width", "covered",
    )
    summary = pd.read_csv(ROOT / "results/tables/T30_conformal_summary.csv")
    for _, row in summary.iterrows():
        subset = frame[(frame.model == row.model) & (frame.method == row.method)]
        assert len(subset) == row.n
        assert np.isclose(subset.covered.mean(), row.coverage)
        assert np.isclose(subset.width.replace([np.inf, -np.inf], np.nan).mean(), row.mean_width_t)
        assert np.isclose(subset.width.median(), row.median_width_t)


def test_mondrian_row_artifact_is_reproducible():
    frame = _check_intervals(
        "results/preds/stage3_mondrian_intervals.csv",
        "actual", "lo", "hi", "width", "covered",
    )
    summary = pd.read_csv(ROOT / "results/tables/T33_mondrian_by_size.csv")
    if "size_q" not in frame.columns:
        pytest.skip("reference artifact predates persisted Mondrian size labels")
    for _, row in summary.iterrows():
        subset = frame[frame.size_q == row.size_q]
        assert len(subset) == row.n
        assert np.isclose(subset.covered.mean(), row.coverage)
        assert np.isclose(subset.width.mean(), row.mean_width)
