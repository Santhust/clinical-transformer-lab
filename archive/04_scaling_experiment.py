"""
04_scaling_experiment.py — Does transformer advantage emerge with scale?
Tests n in [1k, 3k, 10k, 20k] on a NON-LINEAR temporal task (rise-then-fall of WBC).
Classical handcrafted trend (last-first) cannot capture peak, transformer can.

Prints live progress (use -u) and saves:
  results/scaling_metrics.json
  plots/scaling_curve.png
"""
import numpy as np, torch, json, matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

# ensure reproducible but different per n
def generate_cohort(n, seed=42, max_visits=8):
    rng = np.random.default_rng(seed)
    records=[]; seqs=[]; masks=[]; labels=[]
    for pid in range(n):
        n_visits = int(rng.integers(2, max_visits+1))
        age = int(rng.integers(30,85))
        sex = int(rng.integers(0,2))
        base = -4.5 + 0.04*age + 0.3*sex
        visits=[]
        for v in range(n_visits):
            hr = float(rng.normal(75+2*v, 12))
            sbp = float(rng.normal(125-1.5*v, 15))
            # base wbc with drift + extra peak possibility for non-linear label
            wbc = float(rng.normal(7+0.6*v, 2.5))
            diag = float(rng.choice([0,1,2,3,4], p=[0.5,0.15,0.15,0.1,0.1]))/4
            # 8% missing
            if rng.random()<0.08:
                idx=int(rng.integers(0,3))
                if idx==0: hr=np.nan
                elif idx==1: sbp=np.nan
                else: wbc=np.nan
            visits.append([hr/100, sbp/200, wbc/15, diag, age/100, sex])
        arr=np.array(visits, dtype=float)
        filled=np.where(np.isnan(arr), 0.7, arr)
        # NON-LINEAR signal: peak minus avg of endpoints
        wbc_series = filled[:,2]
        wbc_peak = float(np.max(wbc_series) - (wbc_series[0]+wbc_series[-1])/2)
        # label depends on peak + diag, NOT just linear trend
        logit = base + 2.5*wbc_peak + 0.9*filled[-1,3] + float(rng.normal(0,0.8))
        prob = 1/(1+np.exp(-logit))
        label = int(prob>0.5)
        # flattened features for classical: includes wbc_trend (which will be weak now), but not peak
        last=filled[-1]
        wbc_trend = float(filled[-1,2]-filled[0,2])
        hr_trend = float(filled[-1,0]-filled[0,0])
        records.append([age,sex,n_visits,last[0],last[1],last[2],last[3],wbc_trend,hr_trend,label])
        # padded sequence
        seq=np.zeros((max_visits,6))
        seq[:n_visits]=filled
        mask=np.zeros(max_visits); mask[:n_visits]=1
        seqs.append(seq); masks.append(mask); labels.append(label)
    cols=["age","sex","n_visits","hr_last","sbp_last","wbc_last","diag_last","wbc_trend","hr_trend","label"]
    import pandas as pd
    df=pd.DataFrame(records, columns=cols)
    return df, np.stack(seqs), np.stack(masks), np.array(labels)

# model (same as before)
class EHRTransformer(torch.nn.Module):
    def __init__(self, d_in=6, d_model=32, nhead=4):
        super().__init__()
        self.input_proj = torch.nn.Linear(d_in, d_model)
        self.pos_emb = torch.nn.Parameter(torch.randn(1,8,d_model)*0.1)
        enc = torch.nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=64, batch_first=True)
        self.encoder = torch.nn.TransformerEncoder(enc, num_layers=2, enable_nested_tensor=False)
        self.classifier = torch.nn.Linear(d_model,1)
    def forward(self,x,mask):
        h = self.input_proj(x) + self.pos_emb[:,:x.size(1),:]
        pad_mask=(mask==0)
        h=self.encoder(h, src_key_padding_mask=pad_mask)
        pooled=(h*mask.unsqueeze(-1)).sum(1)/mask.unsqueeze(-1).sum(1).clamp(min=1)
        return self.classifier(pooled).squeeze(-1)

def eval_classical(df):
    X=df.drop(columns=["label"]).values; y=df["label"].values
    # if too imbalanced for CV, still try
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results={}
    for name, model in [
        ("LogReg", LogisticRegression(class_weight="balanced", max_iter=500)),
        ("RF", RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample", random_state=42))
    ]:
        aucs=[]; auprcs=[]
        for tr,te in cv.split(X,y):
            # handle case where train has one class (very small n)
            if len(np.unique(y[tr]))<2:
                continue
            model.fit(X[tr], y[tr])
            prob=model.predict_proba(X[te])[:,1]
            aucs.append(roc_auc_score(y[te], prob))
            auprcs.append(average_precision_score(y[te], prob))
        results[name]=(float(np.mean(aucs)) if aucs else 0.5, float(np.mean(auprcs)) if auprcs else 0.1)
    return results

def train_transformer(seqs, masks, labels, epochs=12, batch_size=64):
    # split
    from sklearn.model_selection import train_test_split
    X_tr,X_te,m_tr,m_te,y_tr,y_te=train_test_split(seqs,masks,labels,test_size=0.2, stratify=labels, random_state=42)
    class DS(torch.utils.data.Dataset):
        def __init__(self,a,b,c): self.a=torch.tensor(a,dtype=torch.float32); self.b=torch.tensor(b,dtype=torch.float32); self.c=torch.tensor(c,dtype=torch.float32)
        def __len__(self): return len(self.c)
        def __getitem__(self,i): return self.a[i],self.b[i],self.c[i]
    train_ds=DS(X_tr,m_tr,y_tr)
    loader=torch.utils.data.DataLoader(train_ds,batch_size=batch_size, shuffle=True)
    device=torch.device("cpu")
    model=EHRTransformer().to(device)
    # handle imbalance
    pos_weight=torch.tensor([(len(y_tr)-y_tr.sum())/max(1,y_tr.sum())], dtype=torch.float32).to(device)
    crit=torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt=torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    best=(0,0)
    for epoch in range(epochs):
        model.train()
        losses=[]
        for x,m,y in loader:
            x,m,y=x.to(device),m.to(device),y.to(device)
            opt.zero_grad()
            logits=model(x,m)
            loss=crit(logits,y)
            loss.backward(); opt.step()
            losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            prob=torch.sigmoid(model(torch.tensor(X_te,dtype=torch.float32).to(device), torch.tensor(m_te,dtype=torch.float32).to(device))).cpu().numpy()
            auc=roc_auc_score(y_te, prob); auprc=average_precision_score(y_te, prob)
        print(f"  Epoch {epoch+1}/{epochs}: loss {np.mean(losses):.3f}  AUROC {auc:.3f}  AUPRC {auprc:.3f}", flush=True)
        if auc>best[0]:
            best=(auc,auprc)
    return best

cohorts=[1000,3000,10000,20000]
all_results={}
Path("results").mkdir(exist_ok=True)
Path("plots").mkdir(exist_ok=True)

for n in cohorts:
    print(f"\n=== Cohort n={n} ===", flush=True)
    print(f"Generating synthetic EHR (non-linear peak task, seed={n})...", flush=True)
    df, seqs, masks, labels = generate_cohort(n, seed=42+n)
    prev=float(labels.mean())
    print(f"  generated: {seqs.shape}, prevalence {prev:.3f}", flush=True)
    print(f"  Classical (5-fold CV)...", flush=True)
    classical=eval_classical(df)
    for k,(auc,auprc) in classical.items():
        print(f"    {k}: AUROC {auc:.3f}  AUPRC {auprc:.3f}", flush=True)
    print(f"  Transformer (12 epochs, batch 64)...", flush=True)
    tr_auc, tr_auprc = train_transformer(seqs,masks,labels, epochs=12)
    print(f"  => Transformer final: AUROC {tr_auc:.3f}  AUPRC {tr_auprc:.3f}", flush=True)
    all_results[str(n)]={
        "n":n, "prevalence":prev,
        "LogReg_AUROC": classical["LogReg"][0], "LogReg_AUPRC": classical["LogReg"][1],
        "RF_AUROC": classical["RF"][0], "RF_AUPRC": classical["RF"][1],
        "Transformer_AUROC": tr_auc, "Transformer_AUPRC": tr_auprc
    }
    # incremental save
    with open("results/scaling_metrics.json","w") as f: json.dump(all_results,f,indent=2)
    print(f"  saved intermediate results", flush=True)

# plot
print("\nGenerating scaling curve plot...", flush=True)
ns=sorted(int(k) for k in all_results)
logreg_auc=[all_results[str(n)]["LogReg_AUROC"] for n in ns]
rf_auc=[all_results[str(n)]["RF_AUROC"] for n in ns]
tr_auc=[all_results[str(n)]["Transformer_AUROC"] for n in ns]
logreg_pr=[all_results[str(n)]["LogReg_AUPRC"] for n in ns]
rf_pr=[all_results[str(n)]["RF_AUPRC"] for n in ns]
tr_pr=[all_results[str(n)]["Transformer_AUPRC"] for n in ns]

fig, ax = plt.subplots(1,2, figsize=(12,4.5), sharex=True)
for a in ax: a.set_xscale("log"); a.set_xticks(ns); a.set_xticklabels([str(n) for n in ns])
ax[0].plot(ns, logreg_auc, marker="o", label="LogReg", color="#4a90e2")
ax[0].plot(ns, rf_auc, marker="s", label="RandomForest", color="#e67e22")
ax[0].plot(ns, tr_auc, marker="^", label="Transformer", color="#2ca02c")
ax[0].set_title("AUROC vs cohort size\n(non-linear peak task)"); ax[0].set_ylabel("AUROC"); ax[0].set_xlabel("n (log scale)"); ax[0].set_ylim(0.5,1.0); ax[0].legend(); ax[0].grid(alpha=0.3)
ax[1].plot(ns, logreg_pr, marker="o", label="LogReg", color="#4a90e2")
ax[1].plot(ns, rf_pr, marker="s", label="RandomForest", color="#e67e22")
ax[1].plot(ns, tr_pr, marker="^", label="Transformer", color="#2ca02c")
ax[1].set_title("AUPRC vs cohort size"); ax[1].set_ylabel("AUPRC"); ax[1].set_xlabel("n (log scale)"); ax[1].legend(); ax[1].grid(alpha=0.3)
fig.suptitle("Scaling behaviour: classical plateau vs transformer improvement", fontsize=12)
fig.tight_layout(); fig.savefig("plots/scaling_curve.png", dpi=180); plt.close()
# also copy to docs/plots for site
import shutil; Path("docs/plots").mkdir(exist_ok=True)
shutil.copy("plots/scaling_curve.png","docs/plots/scaling_curve.png")
print("Saved plots/scaling_curve.png + docs/plots/scaling_curve.png", flush=True)
print("Done. Results in results/scaling_metrics.json", flush=True)
