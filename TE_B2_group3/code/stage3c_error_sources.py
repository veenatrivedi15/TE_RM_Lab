"""Stage 3c: where does RF(daily rate) beat CalAdj persistence? (squared-error decomposition)
Reads results/preds/s3/origin_*.csv and the modeling dataset. Writes results/tables/T26-T28."""
import glob, os
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# pipeline mode: stages run inside work/ and write work/out; shipped-package mode: curated results/
BASE = "out" if os.path.isdir("out/preds/s3") else f"{ROOT}/results"
P = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{BASE}/preds/s3/origin_*.csv"))])
P["target_month"] = pd.to_datetime(P["target_month"])
REF, MOD = "CalAdj persistence", "RF (daily rate)"
P["e_ref"] = P["actual"] - P[REF]; P["e_mod"] = P["actual"] - P[MOD]
P["se_gain"] = P["e_ref"]**2 - P["e_mod"]**2          # >0: RF better
P["ae_gain"] = P["e_ref"].abs() - P["e_mod"].abs()
tot = P["se_gain"].sum()
# plant size = mean actual tonnes per plant over the evaluation window
size = P.groupby("source_id")["actual"].mean()
P["size_q"] = pd.qcut(P["source_id"].map(size), 4, labels=["Q1 small","Q2","Q3","Q4 large"])
P["zero"] = P["actual"] <= 1.0
P["ref_zero"] = P[REF] <= 1.0
P["restart"] = P["zero"] ^ P["ref_zero"]               # regime change (stop or restart)
P["big_ref_err"] = P["e_ref"].abs() > P["e_ref"].abs().quantile(0.99)

def summ(g):
    return pd.Series(dict(n=len(g), mae_ref=g.e_ref.abs().mean(), mae_rf=g.e_mod.abs().mean(),
        mse_gain_share=g.se_gain.sum()/tot, rf_better_frac=(g.se_gain>0).mean()))
out = {}
out["T26_by_size"] = P.groupby("size_q", observed=True).apply(summ)
out["T27_by_regime"] = P.groupby(["zero","restart"]).apply(summ)
P["cal_month"] = P["target_month"].dt.month
out["T28_by_calendar_month"] = P.groupby("cal_month").apply(summ)
# concentration: share of total squared-error gain from top 1% / 5% of rows
s = P["se_gain"].sort_values(ascending=False)
conc = pd.DataFrame({"top_share":[0.001,0.01,0.05,0.10]})
conc["share_of_total_gain"] = [s.iloc[:int(len(s)*q)].sum()/tot for q in conc.top_share]
out["T29_gain_concentration"] = conc.set_index("top_share")
# gain excluding regime-change rows
ex = P[~P.restart & ~P.zero]
out["T29b_excl_zero_and_restart"] = pd.DataFrame({"n":[len(ex)], "rmse_ref":[np.sqrt((ex.e_ref**2).mean())],
    "rmse_rf":[np.sqrt((ex.e_mod**2).mean())], "mae_ref":[ex.e_ref.abs().mean()], "mae_rf":[ex.e_mod.abs().mean()]})
# top plants by gain
pl = P.groupby("source_id").se_gain.sum().sort_values(ascending=False)
out["T29c_top_plants"] = (pl.head(10)/tot).rename("share_of_total_gain").to_frame()
os.makedirs(f"{BASE}/tables", exist_ok=True)
for k, v in out.items():
    v.to_csv(f"{BASE}/tables/{k}.csv"); print("\n==", k); print(v.round(3).to_string())
print("\nrows", len(P), "total SE gain", tot, "zero rows", int(P.zero.sum()), "restart rows", int(P.restart.sum()))
