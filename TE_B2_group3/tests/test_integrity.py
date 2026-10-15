"""Integrity tests. Run from package root:  python -m pytest -q tests
Needs data/modeling_dataset_step1.csv and code/stage3_benchmark.py."""
import os, subprocess, sys, tempfile, shutil
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "modeling_dataset_step1.csv")
S3 = os.path.join(ROOT, "code", "stage3_benchmark.py")


def _load():
    return pd.read_csv(DATA, parse_dates=["start_time"])


def test_panel_complete():
    d = _load()
    assert d.source_id.nunique() == 166 and d.start_time.nunique() == 67
    assert len(d) == 166 * 67 and not d.duplicated(["source_id", "start_time"]).any()
    assert (d.start_time.dt.day == 1).all()
    gaps = d.sort_values(["source_id", "start_time"]).groupby("source_id").start_time.apply(lambda s: (s.diff().dropna().dt.days > 31).sum())
    assert gaps.sum() == 0


def test_calendar_factor_leap_year():
    f = lambda a, b: pd.Timestamp(b).days_in_month / pd.Timestamp(a).days_in_month
    assert abs(f("2024-01-01", "2024-02-01") - 29 / 31) < 1e-12      # leap February
    assert abs(f("2025-02-01", "2025-03-01") - 31 / 28) < 1e-12
    assert f("2025-12-01", "2026-01-01") == 1.0


def test_no_future_information_in_forecast(t=30):
    """Forecast for origin t must not change if (a) all months after t+1 are deleted and
    (b) the target-month actuals (t+1) are scrambled."""
    d = _load().sort_values(["source_id", "start_time"])
    months = np.sort(d.start_time.unique())
    tgt = months[t + 1]
    tmp = tempfile.mkdtemp()
    try:
        full = os.path.join(tmp, "full"); cut = os.path.join(tmp, "cut")
        for p in (full, cut):
            os.makedirs(p)
        d.to_csv(os.path.join(full, "d.csv"), index=False)
        c = d[d.start_time <= tgt].copy()
        rng = np.random.default_rng(0)
        m = c.start_time == tgt
        c.loc[m, "emissions_quantity"] = rng.uniform(0, 3e5, m.sum())
        c.to_csv(os.path.join(cut, "d.csv"), index=False)
        for p in (full, cut):
            r = subprocess.run([sys.executable, S3, "d.csv", "--start", str(t), "--end", str(t)], cwd=p, capture_output=True, text=True)
            assert r.returncode == 0, r.stderr
        a = pd.read_csv(os.path.join(full, "out/preds/s3", f"origin_{t:03d}.csv")).drop(columns=["actual"])
        b = pd.read_csv(os.path.join(cut, "out/preds/s3", f"origin_{t:03d}.csv")).drop(columns=["actual"])
        pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-9)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
