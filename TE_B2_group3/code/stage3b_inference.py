"""Stage 3b: metrics + dependence-aware inference on stage-3 predictions.

Convention: delta = loss(model) - loss(reference). Negative => model better.
Inference unit = forecast ORIGIN (month): per-origin loss is the mean absolute error over all plants,
so cross-sectional dependence among plants within a month is handled by aggregation.
"""
import os, glob, json
import numpy as np, pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

OUT = "out"; TB = os.path.join(OUT, "tables"); os.makedirs(TB, exist_ok=True)
RNG = np.random.default_rng(42)
B = 10000
REF = "CalAdj persistence"

df = pd.concat([pd.read_csv(f, parse_dates=["origin", "target_month"]) for f in sorted(glob.glob(os.path.join(OUT, "preds", "s3", "origin_*.csv")))])
MODELS = [c for c in df.columns if c not in ("origin_idx", "origin", "target_month", "source_id", "actual")]
origins = np.sort(df.origin_idx.unique()); T = len(origins)
assert len(df) == T * df.source_id.nunique()
PAPER_WINDOW = origins[-13:]           # same 13 origins as the original hold-out (Jun 2025 .. Jun 2026)
EARLY_WINDOW = origins[:-13]

# training-only MASE scale: in-sample one-step naive MAE per plant up to first evaluated origin
raw = pd.read_csv("modeling_dataset_step1.csv", parse_dates=["start_time"]).sort_values(["source_id", "start_time"])
Yw = raw.pivot(index="source_id", columns="start_time", values="emissions_quantity")
t0 = int(origins[0])
scale = pd.Series(np.abs(np.diff(Yw.values[:, :t0 + 1], axis=1)).mean(1), index=Yw.index)


def metrics(sub):
    rows = []
    for m in MODELS:
        e = sub[m] - sub.actual
        ae = e.abs()
        pm = ae.groupby(sub.source_id).mean() / scale.reindex(ae.groupby(sub.source_id).mean().index)
        pm = pm.replace([np.inf, -np.inf], np.nan).dropna()
        rows.append(dict(Model=m, MAE=ae.mean(), RMSE=np.sqrt((e ** 2).mean()), WAPE_pct=100 * ae.sum() / sub.actual.sum(),
                         Bias=e.mean(), MASE_mean=pm.mean(), MASE_median=pm.median()))
    r = pd.DataFrame(rows).set_index("Model")
    r["RelMAE_vs_CalAdj"] = r.MAE / r.loc[REF, "MAE"]
    r["RelMAE_vs_Persistence"] = r.MAE / r.loc["Persistence", "MAE"]
    return r


def loss_by_origin(sub, m):
    return (sub[m] - sub.actual).abs().groupby(sub.origin_idx).mean()


def mbb_ci(d, block, B=B):
    n = len(d); k = int(np.ceil(n / block))
    starts = RNG.integers(0, n, size=(B, k))
    idx = ((starts[:, :, None] + np.arange(block)[None, None, :]) % n).reshape(B, -1)[:, :n]
    bm = d[idx].mean(1)
    return np.percentile(bm, [2.5, 97.5])


def dm_hac(d):
    n = len(d); L = max(1, int(n ** (1 / 3))); dc = d - d.mean()
    g = lambda k: (dc[k:] * dc[:n - k]).sum() / n
    var = g(0) + 2 * sum((1 - k / (L + 1)) * g(k) for k in range(1, L + 1))
    var = max(var, 1e-12)
    stat = d.mean() / np.sqrt(var / n) * np.sqrt((n - 1) / n)       # Harvey-Leybourne-Newbold, h = 1
    return stat, 2 * stats.t.sf(abs(stat), n - 1)


def inference(sub, ref, label):
    rows = []
    for m in MODELS:
        if m == ref:
            continue
        d = (loss_by_origin(sub, m) - loss_by_origin(sub, ref)).values
        stat, p = dm_hac(d)
        lo3, hi3 = mbb_ci(d, 3); lo1, hi1 = mbb_ci(d, 1); lo6, hi6 = mbb_ci(d, 6)
        # plant-level (secondary; ignores shared-month dependence)
        fm = (sub[m] - sub.actual).abs().groupby(sub.source_id).mean() - (sub[ref] - sub.actual).abs().groupby(sub.source_id).mean()
        try:
            pw = stats.wilcoxon(fm.values).pvalue
        except ValueError:
            pw = np.nan
        rows.append(dict(Window=label, Reference=ref, Model=m, n_origins=len(d), Delta_MAE=d.mean(), CI95_block3_lo=lo3, CI95_block3_hi=hi3,
                         CI95_block1_lo=lo1, CI95_block1_hi=hi1, CI95_block6_lo=lo6, CI95_block6_hi=hi6, DM_stat=stat, DM_p=p,
                         Origins_better=int((d < 0).sum()), Plants_better=int((fm < 0).sum()), PlantLevel_Wilcoxon_p=pw))
    r = pd.DataFrame(rows)
    r["DM_p_Holm"] = multipletests(r.DM_p, method="holm")[1]
    r["CI_excludes_0_block3"] = (r.CI95_block3_lo > 0) | (r.CI95_block3_hi < 0)
    return r


wins = {"all_origins": df, "paper_test_window_13": df[df.origin_idx.isin(PAPER_WINDOW)], "earlier_origins": df[df.origin_idx.isin(EARLY_WINDOW)]}
acc = pd.concat([metrics(s).assign(Window=k, n_origins=s.origin_idx.nunique(), n_rows=len(s)).reset_index() for k, s in wins.items()])
acc.to_csv(os.path.join(TB, "T20_benchmark_accuracy.csv"), index=False)

inf = pd.concat([inference(df, REF, "all_origins"), inference(df[df.origin_idx.isin(PAPER_WINDOW)], REF, "paper_test_window_13"),
                 inference(df, "Persistence", "all_origins"), inference(df[df.origin_idx.isin(PAPER_WINDOW)], "Persistence", "paper_test_window_13")])
inf.to_csv(os.path.join(TB, "T21_inference_month_block_DM.csv"), index=False)

# by target year and by calendar month (with n)
df["year"] = df.target_month.dt.year
by_year = pd.concat([metrics(g).assign(Year=y, n_origins=g.origin_idx.nunique()).reset_index() for y, g in df.groupby("year")])
by_year.to_csv(os.path.join(TB, "T22_accuracy_by_target_year.csv"), index=False)
df["cal_month"] = df.target_month.dt.month
bm_rows = []
for cm, g in df.groupby("cal_month"):
    for m in ["Persistence", REF, "RF (daily rate)", "RF (tonnes)"]:
        bm_rows.append(dict(cal_month=cm, n_origins=g.origin_idx.nunique(), Model=m, MAE=(g[m] - g.actual).abs().mean()))
pd.DataFrame(bm_rows).pivot(index=["cal_month", "n_origins"], columns="Model", values="MAE").reset_index().to_csv(os.path.join(TB, "T23_mae_by_calendar_month.csv"), index=False)

# per-origin loss table for plotting
lo = pd.DataFrame({m: loss_by_origin(df, m) for m in MODELS}); lo.insert(0, "target_month", df.groupby("origin_idx").target_month.first())
lo.to_csv(os.path.join(TB, "T24_mae_by_origin.csv"))

# console summary
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
for k in wins:
    print(f"\n=== {k} (origins={wins[k].origin_idx.nunique()}, rows={len(wins[k])}) ===")
    print(acc[acc.Window == k].set_index("Model")[["MAE", "RMSE", "WAPE_pct", "MASE_median", "RelMAE_vs_CalAdj"]].round(3).sort_values("MAE").to_string())
print("\n=== INFERENCE vs", REF, "(delta<0 => model better) ===")
cols = ["Window", "Model", "n_origins", "Delta_MAE", "CI95_block3_lo", "CI95_block3_hi", "DM_p", "DM_p_Holm", "Origins_better", "Plants_better", "PlantLevel_Wilcoxon_p"]
print(inf[inf.Reference == REF][cols].round(3).to_string(index=False))
print("\n=== INFERENCE vs Persistence (paper's framing) ===")
print(inf[inf.Reference == "Persistence"][cols].round(3).to_string(index=False))


# --- squared-error loss (RMSE-type) inference: does ML reduce LARGE errors? -------------------------
def mse_by_origin(sub, m):
    return ((sub[m] - sub.actual) ** 2).groupby(sub.origin_idx).mean()

rows = []
for m in ["RF (daily rate)", "HGB (daily rate)", "RF (tonnes)", "Theta (daily rate)"]:
    d = (mse_by_origin(df, m) - mse_by_origin(df, REF)).values / 1e6      # units: 1e6 t^2
    stat, p = dm_hac(d); lo, hi = mbb_ci(d, 3)
    rows.append(dict(Model=m, Reference=REF, Loss="squared error", n_origins=len(d), Delta_MSE_1e6=d.mean(), CI95_block3_lo=lo, CI95_block3_hi=hi,
                     DM_p=p, Origins_better=int((d < 0).sum())))
sq = pd.DataFrame(rows); sq["DM_p_Holm"] = multipletests(sq.DM_p, method="holm")[1]
sq.to_csv(os.path.join(TB, "T25_inference_squared_loss.csv"), index=False)
print("\n=== SQUARED-ERROR inference vs", REF, "(all origins) ===")
print(sq.round(3).to_string(index=False))
