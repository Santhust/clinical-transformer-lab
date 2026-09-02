import json, matplotlib.pyplot as plt, numpy as np
from pathlib import Path
with open("results/classical_metrics.json") as f: c=json.load(f)
with open("results/transformer_metrics.json") as f: t=json.load(f)

names = ["LogReg", "RandomForest", "Transformer\n(EHR)"]
aurocs = [c["LogisticRegression"]["AUROC"], c["RandomForest"]["AUROC"], t["AUROC"]]
auprcs = [c["LogisticRegression"]["AUPRC"], c["RandomForest"]["AUPRC"], t["AUPRC"]]

fig, ax = plt.subplots(figsize=(8,4.5))
x = np.arange(len(names))
ax.bar(x-0.2, aurocs, 0.4, label="AUROC", color="#4a90e2")
ax.bar(x+0.2, auprcs, 0.4, label="AUPRC", color="#e67e22")
ax.set_xticks(x, names); ax.set_ylim(0,1); ax.legend()
ax.set_title("Medical Prediction: Classical vs Transformer (synthetic EHR, n=3000, 9.7% prevalence)\n5-fold CV classical, hold-out transformer")
ax.set_ylabel("score")
for i, v in enumerate(aurocs): ax.text(i-0.2, v+0.02, f"{v:.3f}", ha="center", fontsize=9)
for i, v in enumerate(auprcs): ax.text(i+0.2, v+0.02, f"{v:.3f}", ha="center", fontsize=9)
fig.tight_layout(); Path("plots").mkdir(exist_ok=True); fig.savefig("plots/comparison.png", dpi=180); plt.close()
print("Saved plots/comparison.png")

# Generate summary for site
summary = {
    "classical": c,
    "transformer": t,
    "note": "Synthetic data: trend matters. Classical with handcrafted trend features is strong baseline. Transformer learns temporal pattern without feature engineering — gap closes with more data / longer sequences."
}
with open("results/summary.json","w") as f: json.dump(summary,f,indent=2)
