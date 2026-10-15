"""
Stage 1b: (i) external validity check against the official Index of Eight Core Industries (cement),
         (ii) size-stratified (Mondrian) conformalised quantile intervals.
ICI values are transcribed from PIB Release ID 2276691 (22 Jun 2026), Annexure-II; May-26 is provisional.
"""
import numpy as np, pandas as pd, json, os, warnings
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from conformal_utils import finite_sample_quantile
warnings.filterwarnings("ignore")
OUT = "out"
for _sub in ("preds", "tables", "figures"):
    os.makedirs(os.path.join(OUT, _sub), exist_ok=True)
T = lambda n: os.path.join(OUT, "tables", n)
ICI_PATH = "T13_external_ICI_cement_PIB.csv"
ICI_COLUMNS = ["month", "ici_cement_index", "ici_cement_yoy_pct", "provisional"]
ici = pd.read_csv(ICI_PATH, parse_dates=["month"])
if list(ici.columns) != ICI_COLUMNS:
    raise ValueError(f"{ICI_PATH} must have columns {ICI_COLUMNS}")
if ici.empty or ici[ICI_COLUMNS].isna().any().any():
    raise ValueError(f"{ICI_PATH} contains missing values or no observations")
if ici.month.duplicated().any() or not ici.month.is_monotonic_increasing:
    raise ValueError(f"{ICI_PATH} months must be unique and sorted")
if (ici.ici_cement_index <= 0).any() or ~ici.provisional.isin([True, False]).all():
    raise ValueError(f"{ICI_PATH} contains invalid index or provisional values")
ici.to_csv(T("T13_external_ICI_cement_PIB.csv"), index=False)

raw = pd.read_csv("modeling_dataset_step1.csv", parse_dates=["start_time"])
ct = raw.groupby("start_time").emissions_quantity.sum()
days = ct.index.days_in_month
ctd = pd.DataFrame({"ct_total": ct, "ct_per_day": ct / days})
ctd["ct_yoy_pct"] = (ctd.ct_total / ctd.ct_total.shift(12) - 1) * 100
ctd["ct_perday_yoy_pct"] = (ctd.ct_per_day / ctd.ct_per_day.shift(12) - 1) * 100
ctd["ct_activity"] = raw.groupby("start_time").activity.sum(); ctd["ct_activity_yoy_pct"] = (ctd.ct_activity / ctd.ct_activity.shift(12) - 1) * 100
m = ici.merge(ctd.reset_index().rename(columns={"start_time": "month"}), on="month")
m["gap_pp"] = m.ct_yoy_pct - m.ici_cement_yoy_pct
m.to_csv(T("T13b_external_comparison.csv"), index=False)
r_lvl, p_lvl = stats.pearsonr(m.ici_cement_index, m.ct_total); r_pd, p_pd = stats.pearsonr(m.ici_cement_index, m.ct_per_day)
rs_lvl = stats.spearmanr(m.ici_cement_index, m.ct_total)[0]
d_ici = m.ici_cement_index.diff().dropna(); d_ct = m.ct_total.diff().dropna(); r_d, p_d = stats.pearsonr(d_ici, d_ct)
ext = dict(n_months=len(m), r_level_total=r_lvl, p_level_total=p_lvl, spearman_level_total=rs_lvl, r_level_perday=r_pd, p_level_perday=p_pd, r_diff=r_d, p_diff=p_d,
           mean_ici_yoy=m.ici_cement_yoy_pct.mean(), mean_ct_yoy=m.ct_yoy_pct.mean(), mean_ct_perday_yoy=m.ct_perday_yoy_pct.mean(), mean_ct_activity_yoy=m.ct_activity_yoy_pct.mean(),
           n_months_ct_yoy_below_ici=int((m.ct_yoy_pct < m.ici_cement_yoy_pct).sum()),
           ici_peak_to_trough=float(m.ici_cement_index.min() / m.ici_cement_index.max() - 1), ct_peak_to_trough=float(m.ct_total.min() / m.ct_total.max() - 1))
# full-year growth: Climate TRACE total for 12 months to Apr-2026 vs 12 months to Apr-2025
def tot(a, b): return ct[a:b].sum()
ext["ct_12m_to_Apr26_vs_prev_pct"] = (tot("2025-05-01", "2026-04-01") / tot("2024-05-01", "2025-04-01") - 1) * 100
with open(T("external_key_numbers.json"), "w", encoding="utf-8") as f:
    json.dump(ext, f, indent=2)
print(json.dumps(ext, indent=1)); print(m.round(2).to_string())

# ---------------- Mondrian CQR by size quartile ----------------
SEED = 42
d = raw.sort_values(["source_id", "start_time"]).reset_index(drop=True); g = d.groupby("source_id").emissions_quantity
d["target"] = g.shift(-1); d["y_t"] = d.emissions_quantity; d["y_tm1"] = g.shift(1); d["y_tm2"] = g.shift(2); d["y_tm11"] = g.shift(11)
d["activity_t"] = d.activity; d["capacity_t"] = d.capacity
tm = d.start_time + pd.offsets.MonthBegin(1); d["days_tgt"] = tm.dt.days_in_month; d["m_sin"] = np.sin(2*np.pi*tm.dt.month/12); d["m_cos"] = np.cos(2*np.pi*tm.dt.month/12)
FE = ["y_t", "y_tm1", "y_tm2", "m_sin", "m_cos", "days_tgt", "activity_t", "capacity_t", "y_tm11"]
M = d.dropna(subset=["target", "y_t", "y_tm1", "y_tm2", "y_tm11", "activity_t", "capacity_t"]).reset_index(drop=True)
mo = np.sort(M.start_time.unique()); te = M[M.start_time.isin(mo[-13:])]; tv = mo[:-13]; fit = M[M.start_time.isin(tv[:-12])]; val = M[M.start_time.isin(tv[-12:])]
cap = M[M.start_time.isin(tv)].groupby("source_id").capacity_t.median(); q = pd.qcut(cap, 4, labels=["Q1 (smallest)", "Q2", "Q3", "Q4 (largest)"])
def qh(a): return HistGradientBoostingRegressor(loss="quantile", quantile=a, learning_rate=0.05, max_iter=300, l2_regularization=1.0, random_state=SEED)
lo, hi = qh(.1).fit(fit[FE], fit.target), qh(.9).fit(fit[FE], fit.target)
def band(x): a, b = lo.predict(x[FE]), hi.predict(x[FE]); return np.minimum(a, b), np.maximum(a, b)
lv, hv = band(val); lt, ht = band(te); sv = np.maximum(lv - val.target.values, val.target.values - hv)
qv = val.source_id.map(q).values; qt = te.source_id.map(q).values; qhat = {}
for k in q.cat.categories:
    qhat[k] = finite_sample_quantile(sv[qv == k], 0.8)
adj = np.array([qhat[k] for k in qt]); yt = te.target.values
L, H = np.maximum(lt - adj, 0), np.maximum(ht + adj, 0)
cov = (yt >= L) & (yt <= H)
res = pd.DataFrame({"size_q": qt, "cov": cov, "width": H - L}).groupby("size_q").agg(n=("cov", "size"), coverage=("cov", "mean"), mean_width=("width", "mean")).reset_index()
tot = pd.DataFrame([{"size_q": "All", "n": len(cov), "coverage": cov.mean(), "mean_width": (H - L).mean()}])
a = 0.2; iscore = ((H - L) + (2/a)*np.maximum(L - yt, 0) + (2/a)*np.maximum(yt - H, 0)).mean()
pd.concat([res, tot]).to_csv(T("T10c_mondrian_cqr.csv"), index=False)
json.dump({"mondrian_interval_score": float(iscore), "mondrian_rel_width": float((H - L).mean() / yt.mean()), "qhat": {k: float(v) for k, v in qhat.items()}}, open(T("mondrian_key.json"), "w"), indent=2)
print(pd.concat([res, tot]).round(3).to_string()); print("interval score", round(iscore, 0), "rel width", round((H - L).mean() / yt.mean(), 3))
