"""
generate_synthetic_ehr.py — Synthetic EHR generation
Simulates patient timelines for sequential prediction (30-day readmission).

Features mimic real medical data: imbalance, temporal dependencies, missingness.
Run: python src/generate_synthetic_ehr.py  -> data/synthetic_ehr.csv + data/sequences.npz
"""
import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)
N_PATIENTS = 3000
MAX_VISITS = 8

records = []
seq_data = []  # for transformer: list of visit sequences

for pid in range(N_PATIENTS):
    n_visits = np.random.randint(2, MAX_VISITS+1)
    age = np.random.randint(30, 85)
    sex = np.random.choice([0,1])
    # baseline risk
    base_risk = -4.5 + 0.04*age + 0.3*sex
    visits = []
    for v in range(n_visits):
        # labs drift over time
        hr = np.random.normal(75 + 2*v, 12)
        sbp = np.random.normal(125 - 1.5*v, 15)
        wbc = np.random.normal(7 + 0.6*v + (2 if base_risk>-1 else 0), 2.5)
        # diagnosis codes (3 most recent)
        diag = np.random.choice([0,1,2,3,4], p=[0.5,0.15,0.15,0.1,0.1])
        visits.append([hr/100, sbp/200, wbc/15, diag/4, age/100, sex])
        # missingness 8%
        if np.random.rand() < 0.08:
            visits[-1][np.random.randint(0,3)] = np.nan
    
    # label: readmission within 30d — depends on *trend* not just last value (so transformer can win)
    visits_arr = np.array(visits)
    # impute missing with mean for label logic
    visits_filled = np.where(np.isnan(visits_arr), 0.7, visits_arr)
    trend = visits_filled[-1,2] - visits_filled[0,2]  # wbc rising
    logit = base_risk + 1.8*trend + 0.9*visits_filled[-1,3] + np.random.normal(0,0.8)
    prob = 1/(1+np.exp(-logit))
    label = int(prob > 0.5)

    # flatten for classical ML (last visit + trend + counts)
    last = visits_filled[-1]
    trend_hr = visits_filled[-1,0]-visits_filled[0,0]
    records.append({
        "patient_id": pid,
        "age": age, "sex": sex, "n_visits": n_visits,
        "hr_last": last[0], "sbp_last": last[1], "wbc_last": last[2], "diag_last": last[3],
        "wbc_trend": trend, "hr_trend": trend_hr,
        "label": label
    })
    # pad sequences to MAX_VISITS for transformer
    seq = np.zeros((MAX_VISITS, 6))
    seq[:n_visits] = visits_filled
    mask = np.zeros(MAX_VISITS)
    mask[:n_visits] = 1
    seq_data.append((seq, mask, label))

df = pd.DataFrame(records)
Path("data").mkdir(exist_ok=True)
df.to_csv("data/synthetic_ehr.csv", index=False)

# save sequences
seqs = np.stack([s for s,_,_ in seq_data])
masks = np.stack([m for _,m,_ in seq_data])
labels = np.array([l for _,_,l in seq_data])
np.savez("data/sequences.npz", seqs=seqs, masks=masks, labels=labels)

print(f"Generated {len(df)} patients -> data/synthetic_ehr.csv")
print(df["label"].value_counts(normalize=True))
print("Prevalence:", df["label"].mean())
print("Saved sequences:", seqs.shape)
