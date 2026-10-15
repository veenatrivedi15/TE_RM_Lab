"""Stage 3: extended rolling-origin benchmark (one-month-ahead, per plant).

All forecasts for origin t use only data observed up to and including month t.
ML models are refit at every origin on rows whose target month is <= t.
Hyper-parameters are fixed a priori (no tuning on any evaluation period).

Usage:  python stage3_benchmark.py DATA.csv [--start IDX] [--end IDX]
Resumable: one file per origin in out/preds/s3/.
"""
import os, sys, time, warnings, argparse
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.forecasting.theta import ThetaModel

warnings.filterwarnings("ignore")
SEED = 42
MIN_TRAIN_ORIGINS = 12      # first evaluated origin has >= 12 labelled training origins
LAG_MAX = 11                # features need y_{t-11}

ap = argparse.ArgumentParser()
ap.add_argument("data"); ap.add_argument("--start", type=int, default=None); ap.add_argument("--end", type=int, default=None)
ap.add_argument("--skip-stat", action="store_true")
a = ap.parse_args()

OUT = "out"; S3 = os.path.join(OUT, "preds", "s3"); os.makedirs(S3, exist_ok=True)

raw = pd.read_csv(a.data, parse_dates=["start_time"]).sort_values(["source_id", "start_time"])
months = np.sort(raw.start_time.unique()); NM = len(months)
plants = np.sort(raw.source_id.unique()); NP = len(plants)
piv = lambda c: raw.pivot(index="source_id", columns="start_time", values=c).loc[plants, months].values.astype(float)
Y, CAP = piv("emissions_quantity"), piv("capacity")
assert not np.isnan(Y).any() and Y.shape == (NP, NM), "panel must be complete"
mt = pd.DatetimeIndex(months)
D = mt.days_in_month.values.astype(float)             # days in each month (known in advance)
R = Y / D                                             # daily emission rate
CD = CAP / D                                          # daily capacity
MON = mt.month.values

first_origin = LAG_MAX + MIN_TRAIN_ORIGINS            # origin index of first evaluation
last_origin = NM - 2                                  # target = last month
lo = a.start if a.start is not None else first_origin
hi = a.end if a.end is not None else last_origin


def feats(t, level):
    """feature matrix for origin t (all plants). level=True -> tonnes, else daily rate."""
    base = (Y if level else R)
    cap = CAP if level else CD
    X = np.column_stack([base[:, t], base[:, t - 1], base[:, t - 2], base[:, t - 11], cap[:, t],
                         np.full(NP, D[t + 1]), np.full(NP, MON[t + 1])])
    return X


def train_set(t, level):
    Xs, ys = [], []
    for s in range(LAG_MAX, t):                       # origin s has target s+1 <= t  -> known at t
        Xs.append(feats(s, level)); ys.append((Y if level else R)[:, s + 1])
    return np.vstack(Xs), np.concatenate(ys)


def stat_forecasts(t):
    ses = np.zeros(NP); th = np.zeros(NP)
    for p in range(NP):
        h = R[p, :t + 1]
        fb = h[-1]
        if not np.any(h > 0):
            continue
        try:
            ses[p] = float(SimpleExpSmoothing(h, initialization_method="estimated").fit().forecast(1)[0])
        except Exception:
            ses[p] = fb
        try:
            fc = ThetaModel(h, period=12, deseasonalize=len(h) >= 24, method="additive").fit().forecast(1)
            th[p] = float(np.asarray(fc)[0])
        except Exception:
            th[p] = fb
    return np.maximum(ses, 0) * D[t + 1], np.maximum(th, 0) * D[t + 1]


for t in range(lo, hi + 1):
    f = os.path.join(S3, f"origin_{t:03d}.csv")
    if os.path.exists(f):
        continue
    t0 = time.time()
    P = {}
    P["Persistence"] = Y[:, t]
    P["CalAdj persistence"] = Y[:, t] * D[t + 1] / D[t]
    P["Seasonal naive"] = Y[:, t - 11]
    P["CalAdj seasonal naive"] = Y[:, t - 11] * D[t + 1] / D[t - 11]
    P["CalAdj 3-mo mean rate"] = R[:, t - 2:t + 1].mean(1) * D[t + 1]
    if not a.skip_stat:
        P["SES (daily rate)"], P["Theta (daily rate)"] = stat_forecasts(t)
    # ML (fixed hyper-parameters)
    Xtr, ytr = train_set(t, level=False); Xte = feats(t, level=False)
    rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=2, random_state=SEED, n_jobs=-1).fit(Xtr, ytr)
    P["RF (daily rate)"] = np.maximum(rf.predict(Xte), 0) * D[t + 1]
    hg = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.06, random_state=SEED).fit(Xtr, ytr)
    P["HGB (daily rate)"] = np.maximum(hg.predict(Xte), 0) * D[t + 1]
    Xtr, ytr = train_set(t, level=True); Xte = feats(t, level=True)
    rfl = RandomForestRegressor(n_estimators=100, min_samples_leaf=2, random_state=SEED, n_jobs=-1).fit(Xtr, ytr)
    P["RF (tonnes)"] = np.maximum(rfl.predict(Xte), 0)
    df = pd.DataFrame({"origin_idx": t, "origin": months[t], "target_month": months[t + 1],
                       "source_id": plants, "actual": Y[:, t + 1], **P})
    df.to_csv(f, index=False)
    print(f"origin {t} ({str(months[t])[:7]}) done in {time.time() - t0:.1f}s", flush=True)
