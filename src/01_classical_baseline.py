"""
01_classical_baseline.py — Classical ML for medical prediction (proper medical eval)
Run: python src/01_classical_baseline.py  -> plots/classical_*.png + results/classical_metrics.json
"""
import pandas as pd, numpy as np, json
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import matplotlib.pyplot as plt

Path("results").mkdir(exist_ok=True)
Path("plots").mkdir(exist_ok=True)

df = pd.read_csv("data/synthetic_ehr.csv")
X = df.drop(columns=["patient_id","label"]).values
y = df["label"].values
print(f"Data: {X.shape}, prevalence {y.mean():.3f} — imbalanced, AUPRC matters!")

models = {
    "LogisticRegression": LogisticRegression(class_weight="balanced", max_iter=500),
    "RandomForest": RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample", random_state=42)
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}
for name, model in models.items():
    aucs, auprcs = [], []
    for tr, te in cv.split(X,y):
        model.fit(X[tr], y[tr])
        prob = model.predict_proba(X[te])[:,1]
        aucs.append(roc_auc_score(y[te], prob))
        auprcs.append(average_precision_score(y[te], prob))
    results[name] = {"AUROC": float(np.mean(aucs)), "AUPRC": float(np.mean(auprcs)),
                     "AUROC_std": float(np.std(aucs)), "AUPRC_std": float(np.std(auprcs))}
    print(f"{name}: AUROC {np.mean(aucs):.3f}±{np.std(aucs):.3f}, AUPRC {np.mean(auprcs):.3f}±{np.std(auprcs):.3f}")

# save
with open("results/classical_metrics.json","w") as f: json.dump(results,f,indent=2)

# plot
fig, ax = plt.subplots(figsize=(7,4))
names = list(results.keys())
aucs = [results[n]["AUROC"] for n in names]
auprcs = [results[n]["AUPRC"] for n in names]
x = np.arange(len(names))
ax.bar(x-0.2, aucs, 0.4, label="AUROC", color="#4a90e2")
ax.bar(x+0.2, auprcs, 0.4, label="AUPRC (more informative for imbalance)", color="#e67e22")
ax.set_xticks(x, names); ax.set_ylim(0,1); ax.legend(); ax.set_title("Classical Baselines — 5-fold CV (medical metrics)")
ax.set_ylabel("score")
for i, v in enumerate(aucs): ax.text(i-0.2, v+0.02, f"{v:.3f}", ha="center", fontsize=9)
for i, v in enumerate(auprcs): ax.text(i+0.2, v+0.02, f"{v:.3f}", ha="center", fontsize=9)
fig.tight_layout(); fig.savefig("plots/classical_metrics.png", dpi=180); plt.close()
print("Saved plots/classical_metrics.png + results/classical_metrics.json")
