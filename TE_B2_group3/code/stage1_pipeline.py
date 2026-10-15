"""
Stage 1: reproducible analysis pipeline.
Forecasting facility-level monthly CO2 emissions of Indian cement plants (Climate TRACE v5.11.0).
All tables are written to out/tables, predictions to out/preds.  Seeds fixed (SEED=42).
Run:  python stage1_pipeline.py modeling_dataset_step1.csv
"""
import sys, os, time, json, warnings
import numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error as MAE, mean_squared_error as MSE, r2_score
from statsmodels.stats.multitest import multipletests
from conformal_utils import finite_sample_quantile
warnings.filterwarnings("ignore")

SEED = 42; RNG = np.random.default_rng(SEED)
DATA = sys.argv[1] if len(sys.argv) > 1 else "modeling_dataset_step1.csv"
OUT = "out"
T = lambda n: os.path.join(OUT, "tables", n); P = lambda n: os.path.join(OUT, "preds", n)
for _sub in ("preds", "tables", "figures"):
    os.makedirs(os.path.join(OUT, _sub), exist_ok=True)
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)
KEY = {}   # key numbers for the manuscript

# ------------------------------------------------------------------ data
raw = pd.read_csv(DATA, parse_dates=["start_time"]).sort_values(["source_id", "start_time"]).reset_index(drop=True)
g = raw.groupby("source_id")
KEY.update(n_rows=len(raw), n_fac=int(raw.source_id.nunique()), start=str(raw.start_time.min().date()),
           end=str(raw.start_time.max().date()), n_zero=int((raw.emissions_quantity == 0).sum()),
           n_zero_fac=int(raw.loc[raw.emissions_quantity == 0, "source_id"].nunique()))

# data-construction audit: repeat of the same calendar month one year earlier
raw["y12"] = g.emissions_quantity.shift(12); raw["a12"] = g.activity.shift(12); raw["ef12"] = g.emissions_factor.shift(12)
ok = raw.y12.notna()
close = lambda a, b: np.isclose(a, b, rtol=1e-9, atol=0)
raw["rep_y"] = close(raw.emissions_quantity, raw.y12) & (raw.emissions_quantity > 0)
raw["rep_a"] = close(raw.activity, raw.a12) & (raw.activity > 0)
raw["rep_ef"] = close(raw.emissions_factor, raw.ef12) & (raw.emissions_factor > 0)
aud = raw[ok].assign(year=lambda x: x.start_time.dt.year)
audit_year = aud.groupby("year").agg(n=("rep_y", "size"), repeated_emissions=("rep_y", "mean"),
                                     repeated_activity=("rep_a", "mean"), repeated_EF=("rep_ef", "mean")).reset_index()
audit_flag = aud.groupby("other10").agg(n=("rep_y", "size"), repeated_emissions=("rep_y", "mean"),
                                        repeated_activity=("rep_a", "mean"), repeated_EF=("rep_ef", "mean")).reset_index()
audit_yf = aud.groupby(["year", "other10"]).rep_y.mean().unstack().reset_index()
audit_year.to_csv(T("T2_audit_by_year.csv"), index=False); audit_flag.to_csv(T("T2_audit_by_flag.csv"), index=False)
audit_yf.to_csv(T("T2_audit_year_by_flag.csv"), index=False)
KEY["repeat_share_overall"] = float(aud.rep_y.mean()); KEY["flag_share"] = raw.other10.value_counts(normalize=True).round(4).to_dict()
rel = ((raw.activity * raw.emissions_factor - raw.emissions_quantity).abs() / (raw.emissions_quantity.abs() + 1e-9)).max()
KEY["identity_max_rel_err"] = float(rel)

# features
df = raw.copy(); g = df.groupby("source_id").emissions_quantity
df["y_t"] = df.emissions_quantity
for k in (1, 2, 11, 12, 13, 14): df[f"y_tm{k}"] = g.shift(k)
df["activity_t"] = df.activity; df["capacity_t"] = df.capacity
df["flag_tgt"] = df.groupby("source_id").other10.shift(-1)
def add_target(d, h):
    d = d.copy(); gg = d.groupby("source_id").emissions_quantity
    d["target"] = gg.shift(-h); d["y_seas"] = gg.shift(12 - h) if h < 12 else d["y_t"]
    tm = d.start_time + pd.offsets.MonthBegin(h)
    d["days_tgt"] = tm.dt.days_in_month; d["m_sin"] = np.sin(2*np.pi*tm.dt.month/12); d["m_cos"] = np.cos(2*np.pi*tm.dt.month/12)
    d["target_month"] = tm
    return d
df = add_target(df, 1); df["y_tm11"] = df["y_seas"]            # y(t-11): same calendar month as target (h=1)
df["copy_tgt"] = close(df.target, df.y_tm11) & (df.target > 0)
den = (df.y_tm12 + df.y_tm13 + df.y_tm14)
df["base_ratio"] = np.where(den > 0, df.y_tm11 * (df.y_t + df.y_tm1 + df.y_tm2) / den, df.y_tm11)
df["base_ma3"] = (df.y_t + df.y_tm1 + df.y_tm2) / 3
NEED = ["target", "y_t", "y_tm1", "y_tm2", "y_tm11", "activity_t", "capacity_t"]
M = df.dropna(subset=NEED).reset_index(drop=True)
months = np.sort(M.start_time.unique()); test_m = months[-13:]; trval_m = months[:-13]; val_m = trval_m[-12:]; fit_m = trval_m[:-12]
tr, te, fit, val = [M[M.start_time.isin(x)] for x in (trval_m, test_m, fit_m, val_m)]
KEY.update(n_model=len(M), n_train=len(tr), n_fit=len(fit), n_val=len(val), n_test=len(te),
           train_range=f"{pd.Timestamp(trval_m[0]).date()}..{pd.Timestamp(trval_m[-1]).date()}",
           val_range=f"{pd.Timestamp(val_m[0]).date()}..{pd.Timestamp(val_m[-1]).date()}",
           test_range=f"{pd.Timestamp(test_m[0]).date()}..{pd.Timestamp(test_m[-1]).date()}",
           copy_share_all=float(M.copy_tgt.mean()), copy_share_test=float(te.copy_tgt.mean()), copy_share_train=float(tr.copy_tgt.mean()))
log("data ready", KEY["n_model"], "rows;", "copy share test", round(KEY["copy_share_test"], 3))

# descriptive table
d = raw[["emissions_quantity", "activity", "capacity", "emissions_factor"]]
desc = pd.DataFrame({"Mean": d.mean(), "Median": d.median(), "Mode": d.apply(lambda s: s.mode().iloc[0]), "Min": d.min(), "Max": d.max(),
                     "SD": d.std(), "Q1": d.quantile(.25), "Q3": d.quantile(.75), "Skewness": d.skew()}).T
desc.to_csv(T("T1_descriptive.csv"))

# ------------------------------------------------------------------ helpers
SETS = {"A": ["y_t", "y_tm1", "y_tm2"],
        "B": ["y_t", "y_tm1", "y_tm2", "m_sin", "m_cos", "days_tgt"],
        "C": ["y_t", "y_tm1", "y_tm2", "m_sin", "m_cos", "days_tgt", "activity_t", "capacity_t"],
        "D": ["y_t", "y_tm1", "y_tm2", "m_sin", "m_cos", "days_tgt", "activity_t", "capacity_t", "y_tm11"]}
def wape(y, p): s = np.abs(y).sum(); return np.nan if s == 0 else np.abs(y - p).sum() / s * 100
def metr(y, p):
    y = np.asarray(y); p = np.asarray(p); m = MSE(y, p)
    return dict(MAE=MAE(y, p), RMSE=np.sqrt(m), R2=r2_score(y, p), WAPE=wape(y, p))
mk = {"RF": lambda n=200: RandomForestRegressor(n, min_samples_leaf=3, n_jobs=-1, random_state=SEED),
      "HGB": lambda: HistGradientBoostingRegressor(learning_rate=0.05, max_iter=300, l2_regularization=1.0, random_state=SEED)}
BASE = {"Persistence": "y_t", "Moving average (3)": "base_ma3", "Seasonal naive": "y_tm11", "Seasonal-ratio naive": "base_ratio"}
def fit_predict(trn, tst, feats, ntree=200):
    out = {k: tst[c].values for k, c in BASE.items()}
    lr = LinearRegression().fit(trn[feats], trn.target); out["Linear Regression"] = lr.predict(tst[feats])
    m = mk["RF"](ntree).fit(trn[feats], trn.target); out["Random Forest"] = m.predict(tst[feats])
    m = mk["HGB"]().fit(trn[feats], trn.target); out["HistGradientBoosting"] = m.predict(tst[feats])
    m = mk["RF"](ntree).fit(trn[feats], trn.target - trn.y_t); out["RF on change"] = tst.y_t.values + m.predict(tst[feats])
    m = mk["HGB"]().fit(trn[feats], trn.target - trn.y_t); out["HGB on change"] = tst.y_t.values + m.predict(tst[feats])
    return out
MODELS_ML = ["Linear Regression", "Random Forest", "HistGradientBoosting", "RF on change", "HGB on change"]

# ------------------------------------------------------------------ feature-set + model selection on VALIDATION only
rows = []
for s, f in SETS.items():
    for name, mdl in {"Linear": LinearRegression(), "RF": mk["RF"](150)}.items():
        mdl.fit(fit[f], fit.target); rows.append(dict(FeatureSet=s, Model=name, n_features=len(f), **metr(val.target, mdl.predict(val[f]))))
fs_tab = pd.DataFrame(rows); fs_tab.to_csv(T("T3_feature_set_validation.csv"), index=False)
best_fs = fs_tab[fs_tab.Model == "RF"].sort_values("MAE").iloc[0].FeatureSet; FE = SETS[best_fs]; KEY["best_fs"] = best_fs
log("feature set chosen on validation:", best_fs)
vp = fit_predict(fit, val, FE, 150); vtab = pd.DataFrame({k: metr(val.target, v) for k, v in vp.items()}).T
vtab.to_csv(T("T3b_model_validation.csv")); sel_model = vtab.loc[MODELS_ML].MAE.idxmin(); KEY["validation_selected_model"] = sel_model
log("validation-selected model:", sel_model)

# ------------------------------------------------------------------ hold-out (train once)
hp = fit_predict(tr, te, FE, 300)
pd.DataFrame({"source_id": te.source_id.values, "origin": te.start_time.values, "actual": te.target.values, **hp}).to_csv(P("holdout_predictions.csv"), index=False)
# MASE scale: per-plant in-sample one-step naive error (train period)
scale = (tr.target - tr.y_t).abs().groupby(tr.source_id).mean(); scale = scale[scale > 0]
def mase(pred):
    e = (te.target.values - pred); s = pd.Series(np.abs(e), index=te.source_id.values).groupby(level=0).mean()
    idx = s.index.intersection(scale.index); r = s[idx] / scale[idx]; return r.mean(), r.median()
hold = pd.DataFrame({k: metr(te.target, v) for k, v in hp.items()}).T
hold["RelMAE_vs_persistence"] = hold.MAE / hold.loc["Persistence", "MAE"]
hold[["MASE_mean", "MASE_median"]] = pd.DataFrame({k: mase(v) for k, v in hp.items()}).T.values
hold.to_csv(T("T4_holdout_accuracy.csv")); log("hold-out done")

# ------------------------------------------------------------------ rolling origin (refit monthly)
rp = []
for mth in test_m:
    a = M[M.start_time < mth]; b = M[M.start_time == mth]
    pr = fit_predict(a, b, FE, 150)
    rp.append(pd.DataFrame({"source_id": b.source_id.values, "origin": pd.Timestamp(mth), "actual": b.target.values,
                            "copy_tgt": b.copy_tgt.values, "flag_tgt": b.flag_tgt.values, **pr}))
    log("rolling fold", pd.Timestamp(mth).date())
rp = pd.concat(rp, ignore_index=True); rp.to_csv(P("rolling_predictions.csv"), index=False)
ALLM = list(BASE) + MODELS_ML
fold = pd.DataFrame([{"origin": o, "Model": m, **metr(gdf.actual, gdf[m])} for o, gdf in rp.groupby("origin") for m in ALLM])
fold.to_csv(T("T5a_rolling_fold_metrics.csv"), index=False)
sm = fold.groupby("Model")[["MAE", "RMSE", "R2", "WAPE"]].agg(["mean", "std"]); sm.columns = [f"{a}_{b}" for a, b in sm.columns]
sm.to_csv(T("T5_rolling_summary.csv")); log("rolling done")

# ------------------------------------------------------------------ statistical tests (vs persistence)
def tests(pred_df, label):
    pe = pred_df.assign(**{m: (pred_df.actual - pred_df[m]).abs() for m in ALLM})
    fac = pe.groupby("source_id")[ALLM].mean(); mon = pe.groupby("origin")[ALLM].mean(); res = []
    idx = RNG.integers(0, len(fac), size=(2000, len(fac)))
    for m in [x for x in ALLM if x != "Persistence"]:
        dlt = (fac[m] - fac["Persistence"]).values; boots = dlt[idx].mean(axis=1)
        try: p = stats.wilcoxon(dlt).pvalue
        except ValueError: p = np.nan
        dm = (mon[m] - mon["Persistence"]).values
        try: pm = stats.wilcoxon(dm).pvalue
        except ValueError: pm = np.nan
        res.append(dict(Evaluation=label, Model=m, MeanDiff_facilityMAE=dlt.mean(), CI_low=np.percentile(boots, 2.5), CI_high=np.percentile(boots, 97.5),
                        Wilcoxon_p=p, Facilities_better=int((dlt < 0).sum()), N_facilities=len(dlt),
                        Months_better=int((dm < 0).sum()), N_months=len(dm), Month_Wilcoxon_p=pm))
    r = pd.DataFrame(res); r["Holm_p"] = multipletests(r.Wilcoxon_p.fillna(1), method="holm")[1]; return r
tt = pd.concat([tests(rp, "Rolling-origin"),
                tests(pd.read_csv(P("holdout_predictions.csv"), parse_dates=["origin"]), "Hold-out")]); tt.to_csv(T("T6_significance_vs_persistence.csv"), index=False)
log("tests done")

# ------------------------------------------------------------------ robustness strata (rolling predictions)
def strat(df_, name):
    out = []
    for m in ALLM:
        out.append(dict(Stratum=name, Model=m, n=len(df_), MAE=MAE(df_.actual, df_[m]), WAPE=wape(df_.actual.values, df_[m].values)))
    o = pd.DataFrame(out); o["RelMAE"] = o.MAE / o.loc[o.Model == "Persistence", "MAE"].iloc[0]; return o
rp["copy_tgt"] = rp.copy_tgt.astype(bool)
S = [strat(rp, "All facility-months"), strat(rp[rp.flag_tgt == "satellite"], "Satellite-flagged target only"),
     strat(rp[~rp.copy_tgt], "Excluding copied-from-prior-year targets"), strat(rp[rp.actual > 0], "Positive targets only"),
     strat(rp[(rp.flag_tgt == "satellite") & ~rp.copy_tgt & (rp.actual > 0)], "Satellite, not copied, positive")]
pd.concat(S).to_csv(T("T7_robustness_strata.csv"), index=False)
# retrain variants on hold-out: (i) without seasonal lag (set C); (ii) trained and tested only on non-copied rows
rv = []
for label, f, trn, tst in [("Feature set C (no seasonal lag)", SETS["C"], tr, te),
                           ("Feature set D, train & test non-copied rows only", FE, tr[~tr.copy_tgt], te[~te.copy_tgt]),
                           ("Feature set C, train & test non-copied rows only", SETS["C"], tr[~tr.copy_tgt], te[~te.copy_tgt])]:
    pr = fit_predict(trn, tst, f, 200); base = MAE(tst.target, pr["Persistence"])
    for m, v in pr.items(): rv.append(dict(Variant=label, Model=m, n_test=len(tst), MAE=MAE(tst.target, v), RelMAE=MAE(tst.target, v) / base))
pd.DataFrame(rv).to_csv(T("T7b_robustness_retrained.csv"), index=False); log("robustness done")

# ------------------------------------------------------------------ error analysis (validation-selected model) on rolling predictions
best = sel_model; rp["abs_best"] = (rp.actual - rp[best]).abs(); rp["abs_pers"] = (rp.actual - rp["Persistence"]).abs()
rp["tgt_month"] = (rp.origin + pd.offsets.MonthBegin(1)).dt.month
cap = tr.groupby("source_id").capacity_t.median(); q = pd.qcut(cap, 4, labels=["Q1 (smallest)", "Q2", "Q3", "Q4 (largest)"]); rp["size_q"] = rp.source_id.map(q)
def grp(by):
    return rp.groupby(by).apply(lambda x: pd.Series(dict(n=len(x), MAE_model=x.abs_best.mean(), MAE_persistence=x.abs_pers.mean(),
            WAPE_model=wape(x.actual.values, x[best].values), WAPE_persistence=wape(x.actual.values, x.Persistence.values)))).reset_index()
grp("size_q").to_csv(T("T8a_error_by_size.csv"), index=False); grp("tgt_month").to_csv(T("T8b_error_by_month.csv"), index=False)
rp["zero"] = rp.actual == 0; grp("zero").to_csv(T("T8c_error_by_zero.csv"), index=False)
# national aggregate
agg = rp.groupby("origin")[["actual", best, "Persistence", "Seasonal naive", "Seasonal-ratio naive"]].sum()
agg.to_csv(T("T9_national_total.csv"))
ape = lambda c: ((agg[c] - agg.actual).abs() / agg.actual * 100)
KEY["national_MAPE"] = {c: float(ape(c).mean()) for c in [best, "Persistence", "Seasonal naive", "Seasonal-ratio naive"]}
KEY["national_bias_pct"] = {c: float(((agg[c] - agg.actual) / agg.actual * 100).mean()) for c in [best, "Persistence"]}

# ------------------------------------------------------------------ prediction intervals (quantile HGB + conformal calibration)
def qhgb(a, f): return HistGradientBoostingRegressor(loss="quantile", quantile=a, learning_rate=0.05, max_iter=300, l2_regularization=1.0, random_state=SEED)
lo = qhgb(.1, FE).fit(fit[FE], fit.target); hi = qhgb(.9, FE).fit(fit[FE], fit.target)
lv, hv = np.minimum(lo.predict(val[FE]), hi.predict(val[FE])), np.maximum(lo.predict(val[FE]), hi.predict(val[FE]))
score = np.maximum(lv - val.target.values, val.target.values - hv)
qhat = finite_sample_quantile(score, 0.8)
lt, ht = lo.predict(te[FE]), hi.predict(te[FE]); lt, ht = np.minimum(lt, ht), np.maximum(ht, lt)
yt = te.target.values; lt = np.maximum(lt, 0); ht = np.maximum(ht, lt)
def cover(l, h): return ((yt >= l) & (yt <= h))
def iscore(l, h, a=0.2): return (h - l) + (2 / a) * np.maximum(l - yt, 0) + (2 / a) * np.maximum(yt - h, 0)
cqr_l, cqr_h = np.maximum(lt - qhat, 0), ht + qhat
pi = pd.DataFrame([dict(Method="Quantile HGB (10-90%)", Coverage=cover(lt, ht).mean(), MeanWidth=(ht - lt).mean(), RelWidth=(ht - lt).mean() / yt.mean(), IntervalScore=iscore(lt, ht).mean()),
                   dict(Method="Conformalised quantile (CQR)", Coverage=cover(cqr_l, cqr_h).mean(), MeanWidth=(cqr_h - cqr_l).mean(), RelWidth=(cqr_h - cqr_l).mean() / yt.mean(), IntervalScore=iscore(cqr_l, cqr_h).mean())])
pi["Nominal"] = 0.8; pi.to_csv(T("T10_prediction_intervals.csv"), index=False); KEY["cqr_qhat"] = float(qhat)
pid = te[["source_id", "start_time", "target"]].assign(q10=lt, q90=ht, cqr_lo=cqr_l, cqr_hi=cqr_h, size_q=te.source_id.map(q).values, month=(te.start_time + pd.offsets.MonthBegin(1)).dt.month.values)
pid["cov_raw"] = cover(lt, ht); pid["cov_cqr"] = cover(cqr_l, cqr_h); pid.to_csv(P("interval_predictions.csv"), index=False)
pid.groupby("size_q").agg(n=("target", "size"), coverage_raw=("cov_raw", "mean"), coverage_cqr=("cov_cqr", "mean")).reset_index().to_csv(T("T10b_interval_by_size.csv"), index=False)
log("intervals done")

# ------------------------------------------------------------------ multi-horizon hold-out
H = []
test_tgt_first = pd.Timestamp(test_m[0]) + pd.offsets.MonthBegin(1); test_tgt_last = pd.Timestamp(test_m[-1]) + pd.offsets.MonthBegin(1)
for h in (1, 2, 3, 6, 12):
    D = add_target(df, h); D["y_tm11"] = D["y_seas"]
    D = D.dropna(subset=["target", "y_t", "y_tm1", "y_tm2", "y_seas", "activity_t", "capacity_t"])
    t_first = test_tgt_first - pd.DateOffset(months=h)       # first test origin
    test_h = D[(D.target_month >= test_tgt_first) & (D.target_month <= test_tgt_last)]
    train_h = D[D.target_month <= t_first]                   # targets already known at first test origin
    fh = ["y_t", "y_tm1", "y_tm2", "m_sin", "m_cos", "days_tgt", "activity_t", "capacity_t", "y_seas"]
    pr = {"Persistence": test_h.y_t.values, "Seasonal naive": test_h.y_seas.values}
    pr["Linear Regression"] = LinearRegression().fit(train_h[fh], train_h.target).predict(test_h[fh])
    pr["HistGradientBoosting"] = mk["HGB"]().fit(train_h[fh], train_h.target).predict(test_h[fh])
    pr["Random Forest"] = mk["RF"](120).fit(train_h[fh], train_h.target).predict(test_h[fh])
    pr["HGB on change"] = test_h.y_t.values + mk["HGB"]().fit(train_h[fh], train_h.target - train_h.y_t).predict(test_h[fh])
    b = MAE(test_h.target, pr["Persistence"])
    for m, v in pr.items(): H.append(dict(Horizon=h, Model=m, n_train=len(train_h), n_test=len(test_h), MAE=MAE(test_h.target, v), RMSE=np.sqrt(MSE(test_h.target, v)),
                                         WAPE=wape(test_h.target.values, v), RelMAE=MAE(test_h.target, v) / b))
    log("horizon", h, "done")
pd.DataFrame(H).to_csv(T("T11_multi_horizon.csv"), index=False)

# ------------------------------------------------------------------ permutation importance (RF level model, hold-out)
rf = mk["RF"](300).fit(tr[FE], tr.target)
pim = permutation_importance(rf, te[FE], te.target, n_repeats=5, random_state=SEED, scoring="neg_mean_absolute_error", n_jobs=-1)
pd.DataFrame({"feature": FE, "MAE_increase": pim.importances_mean, "sd": pim.importances_std}).sort_values("MAE_increase", ascending=False).to_csv(T("T12_permutation_importance.csv"), index=False)

# aggregate series for figures (all months)
raw.groupby("start_time").emissions_quantity.sum().rename("total").to_csv(T("F_total_series.csv"))
raw.groupby(["source_id"]).capacity.median().to_csv(T("F_capacity.csv"))
json.dump(KEY, open(T("key_numbers.json"), "w"), indent=2, default=str)
log("ALL DONE")
