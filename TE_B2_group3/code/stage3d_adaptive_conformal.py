"""Stage 3d: static split-conformal vs Adaptive Conformal Inference (Gibbs & Candes 2021).
Causal: at origin t only targets of origins < t are used. Panel feedback: err_t = share of plants
missed at origin t (average of per-series miscoverage indicators). alpha=0.10, gamma fixed a priori.
Scores normalised by plant scale (mean actual of that plant over past observed targets).
Outputs results/tables/T30-T32."""
import glob, os
import numpy as np, pandas as pd
from conformal_utils import finite_sample_quantile
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# pipeline mode: stages run inside work/ and write work/out; shipped-package mode: curated results/
BASE = "out" if os.path.isdir("out/preds/s3") else f"{ROOT}/results"
ALPHA, GAMMA, WARM = 0.10, 0.01, 8
P = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{BASE}/preds/s3/origin_*.csv"))])
P = P.sort_values(["origin_idx", "source_id"]).reset_index(drop=True)
origins = sorted(P.origin_idx.unique())

def run(model, gamma=GAMMA, adaptive=True):
    rows, a_t = [], ALPHA
    hist = []                                    # list of dataframes (past origins)
    for t in origins:
        cur = P[P.origin_idx == t]
        if len(hist) >= WARM:
            past = pd.concat(hist)
            scale_map = past.groupby("source_id")["actual"].mean()
            sc = (past.actual - past[model]).abs() / past.source_id.map(scale_map).clip(lower=1.0)
            sc = sc.values
            a_use = a_t if adaptive else ALPHA
            q = finite_sample_quantile(sc, 1 - a_use) if 0 < a_use < 1 else (np.inf if a_use <= 0 else 0.0)
            c = cur.source_id.map(scale_map).fillna(cur.actual.mean()).clip(lower=1.0)
            lo = np.maximum(cur[model] - q * c, 0)
            hi = np.maximum(cur[model] + q * c, lo)
            miss = ((cur.actual < lo) | (cur.actual > hi)).astype(float)
            rows.append(pd.DataFrame(dict(origin_idx=t, target_month=cur.target_month.values, source_id=cur.source_id.values,
                model=model, method=("ACI" if adaptive else "static"), prediction=cur[model].values,
                actual=cur.actual.values, lo=lo.values, hi=hi.values, width=(hi - lo).values,
                miss=miss.values, covered=(1 - miss).values, alpha_t=a_use,
                calibration_n=len(sc), qhat=q, scale=c.values)))
            if adaptive: a_t = a_t + gamma * (ALPHA - miss.mean())
        hist.append(cur)
    return pd.concat(rows)

res = {}
for model in ["RF (daily rate)", "CalAdj persistence"]:
    res[(model, "static")] = run(model, adaptive=False)
    for g in [0.005, 0.01, 0.02, 0.05]:
        res[(model, f"ACI g={g}")] = run(model, gamma=g)
all_rows = []
for (model, method), r in res.items():
    r = r.copy()
    r["method"] = method
    all_rows.append(r)
R = pd.concat(all_rows, ignore_index=True)
R.to_csv(f"{BASE}/preds/stage3_adaptive_intervals.csv", index=False)
summ = []
for (m, meth), r in res.items():
    last = r[r.origin_idx >= r.origin_idx.max() - 11]
    byo = r.groupby("origin_idx").miss.mean()
    summ.append(dict(model=m, method=meth, n=len(r), coverage=r.covered.mean(),
        mean_width_t=r.width.replace(np.inf, np.nan).mean(), median_width_t=r.width.median(),
        coverage_last12=1 - last.miss.mean(), worst_origin_cov=1 - byo.max(), best_origin_cov=1 - byo.min(),
        sd_origin_cov=byo.std(), share_origins_below_80=(1 - byo < 0.80).mean()))
S = pd.DataFrame(summ); S.to_csv(f"{BASE}/tables/T30_conformal_summary.csv", index=False)
print(S.round(3).to_string(index=False))
# coverage by plant-size quartile and by calendar year for the primary configuration
prim = res[("RF (daily rate)", f"ACI g={GAMMA}")].copy(); stat = res[("RF (daily rate)", "static")].copy()
size = P.groupby("source_id")["actual"].mean(); q = pd.qcut(size, 4, labels=["Q1 small", "Q2", "Q3", "Q4 large"])
out = []
for name, r in [("static", stat), ("ACI", prim)]:
    r["size_q"] = r.source_id.map(q); r["year"] = pd.to_datetime(r.target_month).dt.year
    for key in ["size_q", "year"]:
        g = r.groupby(key, observed=True).agg(coverage=("covered", "mean"), n=("covered", "size"))
        g["method"] = name; g["by"] = key; out.append(g.reset_index().rename(columns={key: "level"}))
T31 = pd.concat(out); T31.to_csv(f"{BASE}/tables/T31_conformal_coverage_by_group.csv", index=False)
print(T31.round(3).to_string(index=False))
# zero-target rows
for name, r in [("static", stat), ("ACI", prim)]:
    z = r[r.actual <= 1.0]; print(name, "coverage on zero-target rows", round(1 - z.miss.mean(), 3), "n", len(z))
prim.groupby("origin_idx").agg(alpha_t=("alpha_t", "first"), miss=("miss", "mean")).to_csv(f"{BASE}/tables/T32_aci_path.csv")
print(prim.groupby("origin_idx").alpha_t.first().describe().round(3))
