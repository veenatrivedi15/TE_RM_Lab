"""Stage 3f: panel total emissions vs official ICI cement YoY growth (data/external_ici.csv; within-series YoY only).
Writes results/tables/T34_ici_comparison_by_year.csv and T35_ici_comparison_summary.csv."""
import os, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# pipeline mode: stages run inside work/ and write work/out; shipped-package mode: curated results/
BASE = "out" if os.path.isdir("out/tables") else f"{ROOT}/results"
d = pd.read_csv(f"{ROOT}/data/modeling_dataset_step1.csv", usecols=["start_time", "emissions_quantity"])
d["m"] = pd.to_datetime(d.start_time).dt.to_period("M"); tot = d.groupby("m").emissions_quantity.sum()
py = (tot / tot.shift(12) - 1) * 100
e = pd.read_csv(f"{ROOT}/data/external_ici.csv"); by_year, summ = [], []
for s, g in e.groupby("series"):
    iy = g.set_index(pd.PeriodIndex(g.month, freq="M")).cement_yoy_pct
    j = pd.DataFrame({"ici_yoy": iy, "panel_yoy": py}).dropna(); j["gap_pp"] = j.panel_yoy - j.ici_yoy; j["year"] = [p.year for p in j.index]
    summ.append(dict(series=s, first=str(j.index.min()), last=str(j.index.max()), n_months=len(j), corr=j.ici_yoy.corr(j.panel_yoy),
                     mean_gap_pp=j.gap_pp.mean(), sd_gap_pp=j.gap_pp.std()))
    t = j.groupby("year")[["ici_yoy", "panel_yoy", "gap_pp"]].mean(); t["series"] = s; by_year.append(t.reset_index())
os.makedirs(f"{BASE}/tables", exist_ok=True)
A, B = pd.concat(by_year), pd.DataFrame(summ)
A.to_csv(f"{BASE}/tables/T34_ici_comparison_by_year.csv", index=False); B.to_csv(f"{BASE}/tables/T35_ici_comparison_summary.csv", index=False)
print(B.round(2).to_string(index=False)); print(A.round(2).to_string(index=False))
