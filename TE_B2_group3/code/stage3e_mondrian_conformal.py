"""Stage 3e: size-conditional (Mondrian) split conformal, causal. Size class = quartile of plant's past mean actual
(computed only from targets observed before origin t); separate calibration quantile per class; plus a 'zero-regime'
class for plants whose last observed target was <=1 t. alpha=0.10, WARM=8 origins."""
import glob, os, numpy as np, pandas as pd
from conformal_utils import finite_sample_quantile
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# pipeline mode: stages run inside work/ and write work/out; shipped-package mode: curated results/
BASE = "out" if os.path.isdir("out/preds/s3") else f"{ROOT}/results"
ALPHA, WARM, M = 0.10, 8, "RF (daily rate)"
P = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{BASE}/preds/s3/origin_*.csv"))]).sort_values(["origin_idx","source_id"])
origins = sorted(P.origin_idx.unique()); hist, rows = [], []
def cls(df, past):
    mean = past.groupby("source_id").actual.mean(); last = past.sort_values("origin_idx").groupby("source_id").actual.last()
    q = pd.qcut(mean, 4, labels=False, duplicates="drop"); c = df.source_id.map(q).fillna(1).astype(int)
    z = df.source_id.map(last).fillna(1.0) <= 1.0
    return c.where(~z, 99), mean
for t in origins:
    cur = P[P.origin_idx == t]
    if len(hist) >= WARM:
        past = pd.concat(hist, ignore_index=True); pc, mean = cls(past, past)
        # class of each past row = class of the plant as known now (uses only past data)
        past = past.assign(cl=past.source_id.map(pc.groupby(past.source_id).first()).fillna(1).astype(int))
        ps = past.sort_values(["source_id", "origin_idx"])
        past["zprev"] = (ps.groupby("source_id").actual.shift(1).fillna(1.0) <= 1.0).reindex(past.index)
        past["cl"] = np.where(past.zprev, 99, past.cl)
        past["s"] = (past.actual - past[M]).abs() / past.source_id.map(mean).clip(lower=1.0)
        cc, _ = cls(cur, past); sc = cur.source_id.map(mean).fillna(cur.actual.mean()).clip(lower=1.0)
        qs = {}
        ns = {}
        for k in cc.unique():
            s = np.sort(past.loc[past.cl == k, "s"].values)
            ns[k] = len(s)
            if len(s) < 30:
                s = np.sort(past.s.values)
            qs[k] = finite_sample_quantile(s, 1 - ALPHA)
        w = cc.map(qs) * sc
        lo = np.maximum(cur[M] - w, 0)
        hi = np.maximum(cur[M] + w, lo)
        rows.append(pd.DataFrame(dict(origin_idx=t, target_month=cur.target_month.values,
            source_id=cur.source_id.values, model=M, prediction=cur[M].values,
            cl=cc.values, actual=cur.actual.values, lo=lo.values, hi=hi.values,
            width=(hi - lo).values, covered=((cur.actual.values >= lo.values) &
            (cur.actual.values <= hi.values)), miss=((cur.actual.values < lo.values) |
            (cur.actual.values > hi.values)), calibration_n=cc.map(ns).values,
            qhat=cc.map(qs).values, scale=sc.values)))
    hist.append(cur)
R = pd.concat(rows, ignore_index=True)
R["covered"] = ((R.actual >= R.lo) & (R.actual <= R.hi))
R["miss"] = (~R.covered).astype(float)
size = P.groupby("source_id").actual.mean(); R["size_q"] = R.source_id.map(pd.qcut(size, 4, labels=["Q1 small","Q2","Q3","Q4 large"]))
R.to_csv(f"{BASE}/preds/stage3_mondrian_intervals.csv", index=False)
g = R.groupby("size_q", observed=True).agg(coverage=("covered", "mean"), mean_width=("width", "mean"), n=("covered","size"))
print("overall coverage", round(1-R.miss.mean(),3), "mean width", round((R.hi-R.lo).mean()), "n", len(R)); print(g.round(3))
z = R[R.actual <= 1.0]; print("zero-target coverage", round(1-z.miss.mean(),3), len(z))
byo = R.groupby("origin_idx").miss.mean(); print("worst origin cov", round(1-byo.max(),3), "sd", round(byo.std(),3))
g.to_csv(f"{BASE}/tables/T33_mondrian_by_size.csv")
