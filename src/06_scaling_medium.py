"""
Scaling with medium model — tests if larger capacity improves performance
Medium: d_model=64, 3 layers, 4 heads, ff=128, single seed per n
Saves: results/scaling_medium.json, plots/scaling_medium.png
"""
import numpy as np, torch, json, matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

def generate_cohort(n, seed, max_visits=8):
    rng=np.random.default_rng(seed)
    records=[]; seqs=[]; masks=[]; labels=[]
    for pid in range(n):
        n_visits=int(rng.integers(2,max_visits+1))
        age=int(rng.integers(30,85)); sex=int(rng.integers(0,2))
        base=-4.5+0.04*age+0.3*sex
        visits=[]
        for v in range(n_visits):
            hr=float(rng.normal(75+2*v,12)); sbp=float(rng.normal(125-1.5*v,15))
            wbc=float(rng.normal(7+0.6*v,2.5))
            diag=float(rng.choice([0,1,2,3,4],p=[0.5,0.15,0.15,0.1,0.1]))/4
            if rng.random()<0.08:
                idx=int(rng.integers(0,3))
                if idx==0: hr=np.nan
                elif idx==1: sbp=np.nan
                else: wbc=np.nan
            visits.append([hr/100,sbp/200,wbc/15,diag,age/100,sex])
        arr=np.array(visits,dtype=float); filled=np.where(np.isnan(arr),0.7,arr)
        wbc_series=filled[:,2]
        wbc_peak=float(np.max(wbc_series)-(wbc_series[0]+wbc_series[-1])/2)
        logit=base+2.5*wbc_peak+0.9*filled[-1,3]+float(rng.normal(0,0.8))
        prob=1/(1+np.exp(-logit)); label=int(prob>0.5)
        last=filled[-1]; wbc_trend=float(filled[-1,2]-filled[0,2]); hr_trend=float(filled[-1,0]-filled[0,0])
        records.append([age,sex,n_visits,last[0],last[1],last[2],last[3],wbc_trend,hr_trend,label])
        seq=np.zeros((max_visits,6)); seq[:n_visits]=filled
        mask=np.zeros(max_visits); mask[:n_visits]=1
        seqs.append(seq); masks.append(mask); labels.append(label)
    cols=["age","sex","n_visits","hr_last","sbp_last","wbc_last","diag_last","wbc_trend","hr_trend","label"]
    import pandas as pd
    df=pd.DataFrame(records,columns=cols)
    return df, np.stack(seqs), np.stack(masks), np.array(labels)

class EHRTransformerMedium(torch.nn.Module):
    def __init__(self,d_in=6,d_model=64,nhead=4):
        super().__init__()
        self.input_proj=torch.nn.Linear(d_in,d_model)
        self.pos_emb=torch.nn.Parameter(torch.randn(1,8,d_model)*0.1)
        enc=torch.nn.TransformerEncoderLayer(d_model=d_model,nhead=nhead,dim_feedforward=128,batch_first=True)
        self.encoder=torch.nn.TransformerEncoder(enc,num_layers=3,enable_nested_tensor=False)
        self.classifier=torch.nn.Linear(d_model,1)
    def forward(self,x,mask):
        h=self.input_proj(x)+self.pos_emb[:,:x.size(1),:]
        pad_mask=(mask==0)
        h=self.encoder(h,src_key_padding_mask=pad_mask)
        pooled=(h*mask.unsqueeze(-1)).sum(1)/mask.unsqueeze(-1).sum(1).clamp(min=1)
        return self.classifier(pooled).squeeze(-1)

def eval_classical(df):
    X=df.drop(columns=["label"]).values; y=df["label"].values
    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    out={}
    for name,model in [("LogReg",LogisticRegression(class_weight="balanced",max_iter=500)),("RF",RandomForestClassifier(n_estimators=200,class_weight="balanced_subsample",random_state=42))]:
        aucs=[]; auprcs=[]
        for tr,te in cv.split(X,y):
            if len(np.unique(y[tr]))<2: continue
            model.fit(X[tr],y[tr]); prob=model.predict_proba(X[te])[:,1]
            aucs.append(roc_auc_score(y[te],prob)); auprcs.append(average_precision_score(y[te],prob))
        out[name]=(float(np.mean(aucs)),float(np.mean(auprcs)))
    return out

def train_transformer(seqs,masks,labels,epochs=12,batch_size=64):
    from sklearn.model_selection import train_test_split
    X_tr,X_te,m_tr,m_te,y_tr,y_te=train_test_split(seqs,masks,labels,test_size=0.2,stratify=labels,random_state=42)
    class DS(torch.utils.data.Dataset):
        def __init__(self,a,b,c): self.a=torch.tensor(a,dtype=torch.float32); self.b=torch.tensor(b,dtype=torch.float32); self.c=torch.tensor(c,dtype=torch.float32)
        def __len__(self): return len(self.c)
        def __getitem__(self,i): return self.a[i],self.b[i],self.c[i]
    train_ds=DS(X_tr,m_tr,y_tr); loader=torch.utils.data.DataLoader(train_ds,batch_size=batch_size,shuffle=True)
    device=torch.device("cpu")
    model=EHRTransformerMedium().to(device)
    pos_weight=torch.tensor([(len(y_tr)-y_tr.sum())/max(1,y_tr.sum())],dtype=torch.float32).to(device)
    crit=torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=1e-4)
    best=(0,0)
    for epoch in range(epochs):
        model.train()
        for x,m,y in loader:
            x,m,y=x.to(device),m.to(device),y.to(device)
            opt.zero_grad(); loss=crit(model(x,m),y); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            prob=torch.sigmoid(model(torch.tensor(X_te,dtype=torch.float32).to(device),torch.tensor(m_te,dtype=torch.float32).to(device))).cpu().numpy()
            auc=roc_auc_score(y_te,prob); auprc=average_precision_score(y_te,prob)
        print(f"  Epoch {epoch+1}/{epochs}: AUROC {auc:.3f}  AUPRC {auprc:.3f}",flush=True)
        if auc>best[0]: best=(auc,auprc)
    return best

cohorts=[1000,3000,10000,20000]
results={}
Path("results").mkdir(exist_ok=True); Path("plots").mkdir(exist_ok=True)

for n in cohorts:
    print(f"\n=== Medium model n={n} ===",flush=True)
    df,seqs,masks,labels=generate_cohort(n,seed=42+n)
    print(f" prevalence {labels.mean():.3f}",flush=True)
    classical=eval_classical(df)
    print(f" Classical LogReg {classical['LogReg'][0]:.3f}/{classical['LogReg'][1]:.3f}  RF {classical['RF'][0]:.3f}/{classical['RF'][1]:.3f}",flush=True)
    tr_auc,tr_auprc=train_transformer(seqs,masks,labels)
    print(f" => Transformer medium {tr_auc:.3f}/{tr_auprc:.3f}",flush=True)
    results[str(n)]={"n":n, "LogReg":classical["LogReg"], "RF":classical["RF"], "Transformer":(tr_auc,tr_auprc)}
    with open("results/scaling_medium.json","w") as f: json.dump(results,f,indent=2)

# plot vs small
import json as js
with open("results/scaling_small_3seeds.json") as f: small=js.load(f)
ns=sorted(int(k) for k in results)
small_auc=np.array([small[str(n)]["Transformer_AUROC_mean"] for n in ns])
medium_auc=np.array([results[str(n)]["Transformer"][0] for n in ns])
small_pr=np.array([small[str(n)]["Transformer_AUPRC_mean"] for n in ns])
medium_pr=np.array([results[str(n)]["Transformer"][1] for n in ns])
logreg_auc=np.array([results[str(n)]["LogReg"][0] for n in ns])
logreg_pr=np.array([results[str(n)]["LogReg"][1] for n in ns])

fig,ax=plt.subplots(1,2,figsize=(12,4.5),sharex=True)
for a in ax: a.set_xscale("log"); a.set_xticks(ns); a.set_xticklabels([str(n) for n in ns]); a.grid(alpha=0.25)
ax[0].plot(ns, logreg_auc, marker="o", label="LogReg", color="#4a90e2")
ax[0].plot(ns, small_auc, marker="^", label="Transformer small (32/2)", color="#2ca02c")
ax[0].plot(ns, medium_auc, marker="D", label="Transformer medium (64/3)", color="#9467bd")
ax[0].set_title("AUROC: small vs medium transformer"); ax[0].set_ylabel("AUROC"); ax[0].set_xlabel("n (log)"); ax[0].legend(); ax[0].set_ylim(0.5,1.0)
ax[1].plot(ns, logreg_pr, marker="o", label="LogReg", color="#4a90e2")
ax[1].plot(ns, small_pr, marker="^", label="Transformer small", color="#2ca02c")
ax[1].plot(ns, medium_pr, marker="D", label="Transformer medium", color="#9467bd")
ax[1].set_title("AUPRC: small vs medium"); ax[1].set_ylabel("AUPRC"); ax[1].set_xlabel("n (log)"); ax[1].legend()
fig.suptitle("Capacity experiment: does larger transformer break plateau? (Pi 8GB)",fontsize=12)
fig.tight_layout(); fig.savefig("plots/scaling_medium.png",dpi=180); plt.close()
import shutil; shutil.copy("plots/scaling_medium.png","docs/plots/scaling_medium.png")
print("Saved plots/scaling_medium.png",flush=True)
