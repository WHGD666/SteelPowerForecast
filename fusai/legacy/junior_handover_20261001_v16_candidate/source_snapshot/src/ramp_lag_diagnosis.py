"""Diagnose trend-lag in 24h predictions against the fuel-implied level path."""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "src")

train = pd.read_csv("outputs/prepared_v2/train_prepared.csv", parse_dates=["datetime"])
te = pd.read_csv("outputs/prepared_v2/test_prepared.csv", parse_dates=["datetime"])
fuels = ["generator_use_blast_furnace_gas", "generator_use_converter_gas"]

Xg = np.column_stack([train[f].to_numpy(float) for f in fuels] + [np.ones(len(train))])
yg = train["generator_all"].to_numpy(float)
coefg, *_ = np.linalg.lstsq(Xg, yg, rcond=None)

allg = pd.concat([train, te], ignore_index=True)
implied = np.clip(allg[fuels].to_numpy(float) @ coefg[:2] + coefg[2], 0, None)
allg["implied_gall"] = implied
allg["date"] = allg["datetime"].dt.date
daily_implied = allg.groupby("date")["implied_gall"].mean()

l = pd.read_csv(
    "outputs/submissions/20260929T_g1_70fuel_gall_farclim_v11/l_result.csv",
    parse_dates=["datetime"],
)
l["target_day"] = (l["datetime"] + pd.Timedelta(hours=24)).dt.date
pred24 = l.groupby("target_day")["generator_all_t+1440_pred"].mean()

cmp = pd.DataFrame({"our_pred_24h": pred24})
cmp["implied_actual_next_day"] = daily_implied.reindex(cmp.index)
cmp["error"] = cmp["our_pred_24h"] - cmp["implied_actual_next_day"]
print(cmp.round(1).to_string())
print()
print("ramp days (10/2-10/6) mean error:", round(cmp.loc["2025-10-02":"2025-10-06", "error"].mean(), 1), "MW")
print("all-period mean error:", round(cmp["error"].mean(), 1), "MW")
