import numpy as np, pandas as pd, json, os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.dates as mdates
T = lambda n: os.path.join("out", "tables", n); F = lambda n: os.path.join("out", "figures", n)
os.makedirs("out/figures", exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 120, "savefig.dpi": 300, "axes.grid": True, "grid.alpha": .25, "grid.linewidth": .5})
C = {"base": "#8c8c8c", "lin": "#4C78A8", "rf": "#2E7D32", "hgb": "#E69F00", "chg": "#B2182B", "obs": "#111111", "ici": "#7B3294", "ct": "#1B7837"}
lab = lambda ax, s: ax.text(-0.02, 1.07, s, transform=ax.transAxes, fontweight="bold", fontsize=10, va="bottom", ha="right")
raw = pd.read_csv("modeling_dataset_step1.csv", parse_dates=["start_time"])
tot = raw.groupby("start_time").emissions_quantity.sum()
# ---------------- Fig 1
fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.5), gridspec_kw={"width_ratios": [1.5, 1, 1]})
ax[0].plot(tot.index, tot / 1e6, color=C["ct"], lw=1.2); ax[0].axvspan(pd.Timestamp("2025-07-01"), tot.index.max(), color="#f0c040", alpha=.25, lw=0)
ax[0].set(ylabel="Total CO$_2$ (Mt per month)", title="Total emissions, 166 plants"); ax[0].xaxis.set_major_locator(mdates.YearLocator()); ax[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax[0].text(.97, .06, "test targets", transform=ax[0].transAxes, ha="right", fontsize=6.5, color="#8a6d00"); lab(ax[0], "a")
pdy = (tot / tot.index.days_in_month / 1e3); pm = pdy.groupby(pdy.index.month).mean()
ax[1].bar(pm.index, pm.values, color=C["ct"], width=.75); ax[1].set(xlabel="Calendar month", ylabel="kt CO$_2$ per day", title="Mean emissions per day", xticks=range(1, 13)); lab(ax[1], "b")
ax[2].hist(np.log10(raw.loc[raw.emissions_quantity > 0, "emissions_quantity"]), bins=45, color=C["lin"], edgecolor="white", lw=.3)
ax[2].set(xlabel="log$_{10}$ monthly emissions (t)", ylabel="Facility-months", title="Distribution (zeros excluded)"); lab(ax[2], "c")
plt.tight_layout(); plt.savefig(F("Fig1_data_overview.png")); plt.close()
# ---------------- Fig 2: audit + external validity
ay = pd.read_csv(T("T2_audit_year_by_flag.csv")); ayr = pd.read_csv(T("T2_audit_by_year.csv")); m = pd.read_csv(T("T13b_external_comparison.csv"), parse_dates=["month"])
fig = plt.figure(figsize=(7.4, 5.2)); gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05])
a = fig.add_subplot(gs[0, 0]); w = .2; yrs = ay.year.values; x = np.arange(len(yrs))
a.bar(x - 1.5*w, ayr.repeated_emissions * 100, w, color="#222", label="All")
for i, (c, col) in enumerate([("satellite", C["ct"]), ("extrapolation (national)", C["lin"]), ("extrapolation (plant)", C["hgb"])]):
    a.bar(x + (i - .5) * w, ay[c] * 100, w, color=col, label=c.replace("extrapolation", "extrap."))
a.set(xticks=x, xticklabels=yrs, ylabel="% of facility-months", title="Values identical to same month, prior year"); a.legend(frameon=False, fontsize=6); lab(a, "a")
b = fig.add_subplot(gs[0, 1]); nz = lambda s: s / s.mean() * 100
b.plot(m.month, nz(m.ici_cement_index), "-o", ms=3, color=C["ici"], label="Official cement production index (ICI)")
b.plot(m.month, nz(m.ct_total), "-s", ms=3, color=C["ct"], label="Climate TRACE total (166 plants)")
b.set(ylabel="Index (mean of 13 months = 100)", title=f"Seasonal shape agrees (r = {np.corrcoef(m.ici_cement_index, m.ct_total)[0,1]:.2f})")
b.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%y")); b.xaxis.set_major_locator(mdates.MonthLocator(interval=2)); b.legend(frameon=False, fontsize=6, loc="upper left"); lab(b, "b")
c = fig.add_subplot(gs[1, :]); x = np.arange(len(m)); w = .38
c.bar(x - w/2, m.ici_cement_yoy_pct, w, color=C["ici"], label="Official cement production (ICI), y/y")
c.bar(x + w/2, m.ct_yoy_pct, w, color=C["ct"], label="Climate TRACE total emissions, y/y")
c.axhline(0, color="k", lw=.6); c.set(xticks=x, xticklabels=[d.strftime("%b %y") + ("*" if p else "") for d, p in zip(m.month, m.provisional)], ylabel="Year-on-year change (%)",
  title="Inter-annual growth is not reproduced by the inventory (*provisional)"); c.set_ylim(-5, 19); c.legend(frameon=False, loc="upper right", ncol=2); lab(c, "c")
plt.tight_layout(); plt.savefig(F("Fig2_audit_external.png")); plt.close()
# ---------------- Fig 3: accuracy + significance
h = pd.read_csv(T("T4_holdout_accuracy.csv"), index_col=0); t6 = pd.read_csv(T("T6_significance_vs_persistence.csv")); t6 = t6[(t6.Evaluation == "Rolling-origin")]
order = ["Persistence", "Moving average (3)", "Seasonal naive", "Linear Regression", "Random Forest", "HistGradientBoosting", "RF on change", "HGB on change"]
colr = {"Persistence": C["base"], "Moving average (3)": C["base"], "Seasonal naive": C["base"], "Linear Regression": C["lin"], "Random Forest": C["rf"], "HistGradientBoosting": C["hgb"], "RF on change": C["chg"], "HGB on change": "#9E6B8B"}
fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.1), gridspec_kw={"width_ratios": [1.15, 1]})
v = h.loc[order, "MAE"] / 1e3; ax[0].barh(range(len(order)), v, color=[colr[o] for o in order]); ax[0].set_yticks(range(len(order))); ax[0].set_yticklabels(order); ax[0].invert_yaxis()
ax[0].axvline(v["Persistence"], color="k", ls="--", lw=.7)
for i, o in enumerate(order): ax[0].text(v[o] + .15, i, f"{v[o]:.2f}  ({(h.loc[o,'RelMAE_vs_persistence']-1)*100:+.1f}%)", va="center", fontsize=6.5)
ax[0].set(xlabel="Hold-out MAE (kt CO$_2$ per facility-month)", xlim=(0, 17), title="Hold-out accuracy (2,158 facility-months)"); lab(ax[0], "a")
ml = ["Linear Regression", "Random Forest", "HistGradientBoosting", "RF on change", "HGB on change", "Seasonal naive", "Moving average (3)"]
t = t6.set_index("Model").loc[ml]; y = np.arange(len(ml))
ax[1].errorbar(t.MeanDiff_facilityMAE / 1e3, y, xerr=[(t.MeanDiff_facilityMAE - t.CI_low) / 1e3, (t.CI_high - t.MeanDiff_facilityMAE) / 1e3], fmt="o", color="#222", capsize=2, ms=4)
ax[1].axvline(0, color="k", lw=.7); ax[1].set_yticks(y); ax[1].set_yticklabels(ml); ax[1].invert_yaxis()
ax[1].set(xlabel="Change in facility MAE vs persistence (kt)\n(negative = better; 95% bootstrap CI)", title="Rolling-origin, 166 facilities"); lab(ax[1], "b"); ax[1].set_xlim(-3, 2)
plt.tight_layout(); plt.savefig(F("Fig3_accuracy.png")); plt.close()
# ---------------- Fig 4: rolling, month, size
f5 = pd.read_csv(T("T5a_rolling_fold_metrics.csv"), parse_dates=["origin"]); mm = pd.read_csv(T("T8b_error_by_month.csv")); ss = pd.read_csv(T("T8a_error_by_size.csv"))
fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.6), gridspec_kw={"width_ratios": [1.35, 1.2, .85]})
for mname, col, ls in [("Persistence", C["base"], "-"), ("Seasonal naive", "#555", ":"), ("Linear Regression", C["lin"], "-"), ("Random Forest", C["rf"], "-"), ("HistGradientBoosting", C["hgb"], "-")]:
    d = f5[f5.Model == mname]; ax[0].plot(d.origin + pd.offsets.MonthBegin(1), d.MAE / 1e3, color=col, ls=ls, lw=1.2, marker="o", ms=2.2, label=mname)
ax[0].set(ylabel="MAE (kt)", title="Rolling-origin MAE by target month"); ax[0].xaxis.set_major_formatter(mdates.DateFormatter("%b\n%y")); ax[0].xaxis.set_major_locator(mdates.MonthLocator(interval=3)); ax[0].set_ylim(4, 30); ax[0].legend(frameon=False, fontsize=5.6, ncol=3, loc="upper center", columnspacing=.8, handlelength=1.6); lab(ax[0], "a")
x = np.arange(12); ax[1].bar(x - .2, mm.MAE_persistence / 1e3, .4, color=C["base"], label="Persistence"); ax[1].bar(x + .2, mm.MAE_model / 1e3, .4, color=C["rf"], label="Random Forest")
ax[1].set(xticks=x, xticklabels=mm.tgt_month, xlabel="Target calendar month", ylabel="MAE (kt)", title="Error by target month"); ax[1].legend(frameon=False, fontsize=6); lab(ax[1], "b")
x = np.arange(4); ax[2].bar(x - .2, ss.WAPE_persistence, .4, color=C["base"]); ax[2].bar(x + .2, ss.WAPE_model, .4, color=C["rf"])
ax[2].set(xticks=x, xticklabels=["Q1\nsmall", "Q2", "Q3", "Q4\nlarge"], ylabel="WAPE (%)", title="Error by plant size"); lab(ax[2], "c")
plt.tight_layout(); plt.savefig(F("Fig4_error_structure.png")); plt.close()
# ---------------- Fig 5: national + horizon
nt = pd.read_csv(T("T9_national_total.csv"), parse_dates=["origin"]); hz = pd.read_csv(T("T11_multi_horizon.csv"))
fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.8), gridspec_kw={"width_ratios": [1.25, 1]})
d = nt.copy(); d["tm"] = d.origin + pd.offsets.MonthBegin(1)
ax[0].plot(d.tm, d.actual / 1e6, color=C["obs"], lw=1.8, marker="o", ms=3, label="Actual")
ax[0].plot(d.tm, d["Random Forest"] / 1e6, color=C["rf"], marker="s", ms=2.5, lw=1, label="Random Forest")
ax[0].plot(d.tm, d["Seasonal naive"] / 1e6, color=C["chg"], marker="^", ms=2.5, lw=1, ls="--", label="Seasonal naive")
ax[0].plot(d.tm, d["Persistence"] / 1e6, color=C["base"], marker="d", ms=2.5, lw=1, label="Persistence")
ax[0].set(ylabel="Total CO$_2$ (Mt per month)", title="National total, one month ahead"); ax[0].xaxis.set_major_formatter(mdates.DateFormatter("%b\n%y")); ax[0].xaxis.set_major_locator(mdates.MonthLocator(interval=2)); ax[0].legend(frameon=False, fontsize=6); lab(ax[0], "a")
for mname, col, ls in [("Seasonal naive", C["chg"], "--"), ("Linear Regression", C["lin"], "-"), ("Random Forest", C["rf"], "-"), ("HistGradientBoosting", C["hgb"], "-")]:
    s = hz[hz.Model == mname]; ax[1].plot(s.Horizon, s.RelMAE, marker="o", ms=3, color=col, ls=ls, label=mname)
ax[1].axhline(1, color=C["base"], lw=1, label="Persistence"); ax[1].set(xlabel="Forecast horizon (months)", ylabel="MAE relative to persistence", title="Skill versus horizon", xticks=[1, 2, 3, 6, 12]); ax[1].legend(frameon=False, fontsize=6); lab(ax[1], "b")
plt.tight_layout(); plt.savefig(F("Fig5_national_horizon.png")); plt.close()
# ---------------- Fig 6: intervals
raw_ = pd.read_csv(T("T10b_interval_by_size.csv")); mo = pd.read_csv(T("T10c_mondrian_cqr.csv")); ip = pd.read_csv("out/preds/interval_predictions.csv")
fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.6)); x = np.arange(4); w = .27
ax[0].bar(x - w, raw_.coverage_raw * 100, w, color="#bbb", label="Quantile HGB (raw)"); ax[0].bar(x, raw_.coverage_cqr * 100, w, color=C["lin"], label="CQR (global)")
ax[0].bar(x + w, mo.coverage.iloc[:4] * 100, w, color=C["rf"], label="CQR (size-stratified)"); ax[0].axhline(80, color="k", ls="--", lw=.8)
ax[0].set(xticks=x, xticklabels=["Q1 small", "Q2", "Q3", "Q4 large"], ylabel="Empirical coverage (%)", ylim=(40, 100), title="80% interval coverage by plant size"); ax[0].legend(frameon=False, fontsize=6, loc="upper center", bbox_to_anchor=(.45, -.17), ncol=4, columnspacing=.9, handletextpad=.2); lab(ax[0], "a")
mt = ip.groupby("size_q").target.mean().reindex(["Q1 (smallest)", "Q2", "Q3", "Q4 (largest)"]).values
ax[1].bar(x, mo.mean_width.iloc[:4].values / mt * 100, .55, color=C["rf"]); ax[1].set(xticks=x, xticklabels=["Q1 small", "Q2", "Q3", "Q4 large"], ylabel="Mean width / mean actual (%)", title="Relative interval width (size-stratified CQR)"); lab(ax[1], "b")
plt.tight_layout(); plt.savefig(F("Fig6_intervals.png")); plt.close()
# ---------------- Fig 7: robustness + importance
s7 = pd.read_csv(T("T7_robustness_strata.csv")); s7b = pd.read_csv(T("T7b_robustness_retrained.csv")); pi = pd.read_csv(T("T12_permutation_importance.csv"))
rows = [("All facility-months", s7[s7.Stratum == "All facility-months"].set_index("Model").RelMAE), ("Satellite-flagged targets", s7[s7.Stratum == "Satellite-flagged target only"].set_index("Model").RelMAE),
        ("Excl. copied targets", s7[s7.Stratum == "Excluding copied-from-prior-year targets"].set_index("Model").RelMAE), ("Positive targets only", s7[s7.Stratum == "Positive targets only"].set_index("Model").RelMAE),
        ("Satellite, not copied, >0", s7[s7.Stratum == "Satellite, not copied, positive"].set_index("Model").RelMAE),
        ("Retrained: no seasonal lag", s7b[s7b.Variant == "Feature set C (no seasonal lag)"].set_index("Model").RelMAE),
        ("Retrained: excl. copied rows", s7b[s7b.Variant == "Feature set D, train & test non-copied rows only"].set_index("Model").RelMAE)]
fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.3), gridspec_kw={"width_ratios": [1.3, 1]}); y = np.arange(len(rows))
for mname, col, mk in [("Linear Regression", C["lin"], "o"), ("Random Forest", C["rf"], "s"), ("HistGradientBoosting", C["hgb"], "^"), ("Seasonal naive", C["chg"], "D")]:
    ax[0].scatter([r[1][mname] for r in rows], y, color=col, marker=mk, s=22, label=mname, zorder=3)
ax[0].axvline(1, color="k", lw=.8, ls="--"); ax[0].set_yticks(y); ax[0].set_yticklabels([r[0] for r in rows]); ax[0].invert_yaxis()
ax[0].set(xlabel="MAE relative to persistence (< 1 is better)", title="Robustness of the gain over persistence", xlim=(.75, 1.12)); ax[0].legend(frameon=False, fontsize=6, loc="upper center", bbox_to_anchor=(.45, -.17), ncol=4, columnspacing=.9, handletextpad=.2); lab(ax[0], "a")
p = pi.sort_values("MAE_increase"); nice = {"y_t": "Emissions, month t", "y_tm11": "Emissions, t-11 (seasonal lag)", "y_tm1": "Emissions, t-1", "y_tm2": "Emissions, t-2", "activity_t": "Activity, t", "capacity_t": "Capacity, t", "m_cos": "Month (cos)", "m_sin": "Month (sin)", "days_tgt": "Days in target month"}
ax[1].barh([nice[f] for f in p.feature], p.MAE_increase / 1e3, xerr=p.sd / 1e3, color=C["rf"], capsize=1.5); ax[1].set(xlabel="MAE increase when shuffled (kt)", title="Permutation importance (RF)"); lab(ax[1], "b")
plt.tight_layout(); plt.savefig(F("Fig7_robustness_importance.png"), bbox_inches="tight"); plt.close()
print("figures:", sorted(os.listdir("out/figures")))
