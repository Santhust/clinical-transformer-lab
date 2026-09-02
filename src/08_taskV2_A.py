"""
Task V2 Option A — Ordered sustained rise + code interaction (3-12 visits)
Label = 1 if exists i where WBC rises twice consecutively (>0.10 normalized each step)
        AND diag[i+1] in {0.25,0.75}
Tests if attention to ordered sub-sequence + cross-feature binding helps.
Small (32/2) vs Medium (64/3), n=10k,20k, 12 epochs
"""
import numpy as np, torch, json, matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

def generate_cohort_V2(n, seed, max_visits=12):
    rng=np.random.default_rng(seed)
    records=[]; seqs=[]; masks=[]; labels=[]
    for pid in range(n):
        n_visits=int(rng.integers(3,max_visits+1))  # 3-12
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
        # V2 pattern: sustained rise
        wbc_series=filled[:,2]
        has_pattern=False
        for i in range(n_visits-2):
            if (wbc_series[i+1]-wbc_series[i] > 0.10) and (wbc_series[i+2]-wbc_series[i+1] > 0.10):
                if filled[i+1,3] in [0.25,0.75]:  # diag code 1 or 3
                    has_pattern=True
                    break
        logit=base + (3.0 if has_pattern else 0) + 0.9*filled[-1,3] + float(rng.normal(0,0.8))
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

class Small(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_proj=torch.nn.Linear(6,32)
        self.pos_emb=torch.nn.Parameter(torch.randn(1,12,32)*0.1)
        enc=torch.nn.TransformerEncoderLayer(32,4,64,batch_first=True)
        self.encoder=torch.nn.TransformerEncoder(enc,num_layers=2,enable_nested_tensor=False)
        self.classifier=torch.nn.Linear(32,1)
    def forward(self,x,mask):
        h=self.input_proj(x)+self.pos_emb[:,:x.size(1),:]
        h=self.encoder(h,src_key_padding_mask=(mask==0))
        pooled=(h*mask.unsqueeze(-1)).sum(1)/mask.unsqueeze(-1).sum(1).clamp(min=1)
        return self.classifier(pooled).squeeze(-1)

class Medium(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_proj=torch.nn.Linear(6,64)
        self.pos_emb=torch.nn.Parameter(torch.randn(1,12,64)*0.1)
        enc=torch.nn.TransformerEncoderLayer(64,4,128,batch_first=True)
        self.encoder=torch.nn.TransformerEncoder(enc,num_layers=3,enable_nested_tensor=False)
        self.classifier=torch.nn.Linear(64,1)
    def forward(self,x,mask):
        h=self.input_proj(x)+self.pos_emb[:,:x.size(1),:]
        h=self.encoder(h,src_key_padding_mask=(mask==0))
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

def train(seqs,masks,labels,ModelCls,epochs=12):
    from sklearn.model_selection import train_test_split
    X_tr,X_te,m_tr,m_te,y_tr,y_te=train_test_split(seqs,masks,labels,test_size=0.2,stratify=labels,random_state=42)
    class DS(torch.utils.data.Dataset):
        def __init__(self,a,b,c): self.a=torch.tensor(a,dtype=torch.float32); self.b=torch.tensor(b,dtype=torch.float32); self.c=torch.tensor(c,dtype=torch.float32)
        def __len__(self): return len(self.c)
        def __getitem__(self,i): return self.a[i],self.b[i],self.c[i]
    ds=DS(X_tr,m_tr,y_tr); loader=torch.utils.data.DataLoader(ds,batch_size=64,shuffle=True)
    device=torch.device("cpu"); model=ModelCls().to(device)
    pos_weight=torch.tensor([(len(y_tr)-y_tr.sum())/max(1,y_tr.sum())],dtype=torch.float32).to(device)
    crit=torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=1e-4)
    best=(0,0)
    for e in range(epochs):
        model.train()
        for x,m,y in loader:
            x,m,y=x.to(device),m.to(device),y.to(device)
            opt.zero_grad(); loss=crit(model(x,m),y); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            prob=torch.sigmoid(model(torch.tensor(X_te,dtype=torch.float32).to(device),torch.tensor(m_te,dtype=torch.float32).to(device))).cpu().numpy()
            auc=roc_auc_score(y_te,prob); auprc=average_precision_score(y_te,prob)
        if auc>best[0]: best=(auc,auprc)
        print(f"   Epoch {e+1}/{epochs}: AUROC {auc:.3f} AUPRC {auprc:.3f}",flush=True)
    return best

cohorts=[10000,20000]
results={}
Path("results").mkdir(exist_ok=True)

for n in cohorts:
    print(f"\n=== Task V2-A n={n} (3-12 visits, ordered rise+diag) ===",flush=True)
    df,seqs,masks,labels=generate_cohort_V2(n,seed=100+n)
    print(f" prevalence {labels.mean():.3f}",flush=True)
    classical=eval_classical(df)
    print(f" Classical LogReg {classical['LogReg'][0]:.3f}/{classical['LogReg'][1]:.3f}  RF {classical['RF'][0]:.3f}/{classical['RF'][1]:.3f}",flush=True)
    print(f" Small transformer (32/2)...",flush=True)
    s_auc,s_pr=train(seqs,masks,labels,Small)
    print(f"  Small {s_auc:.3f}/{s_pr:.3f}",flush=True)
    print(f" Medium transformer (64/3)...",flush=True)
    m_auc,m_pr=train(seqs,masks,labels,Medium)
    print(f"  Medium {m_auc:.3f}/{m_pr:.3f}",flush=True)
    results[str(n)]={"n":n, "prevalence":float(labels.mean()), "LogReg":classical["LogReg"], "RF":classical["RF"], "Small":(s_auc,s_pr), "Medium":(m_auc,m_pr)}
    with open("results/taskV2_A.json","w") as f: json.dump(results,f,indent=2)

# plot
import matplotlib.pyplot as plt
ns=sorted(int(k) for k in results)
logreg_pr=[results[str(n)]["LogReg"][1] for n in ns]
small_pr=[results[str(n)]["Small"][1] for n in ns]
medium_pr=[results[str(n)]["Medium"][1] for n in ns]
logreg_auc=[results[str(n)]["LogReg"][0] for n in ns]
small_auc=[results[str(n)]["Small"][0] for n in ns]
medium_auc=[results[str(n)]["Medium"][0] for n in ns]

fig,ax=plt.subplots(1,2,figsize=(12,4.5),sharex=True)
ax[0].plot(ns, logreg_auc, marker="o", label="LogReg", color="#4a90e2")
ax[0].plot(ns, small_auc, marker="^", label="Small", color="#2ca02c")
ax[0].plot(ns, medium_auc, marker="D", label="Medium", color="#9467bd")
ax[0].set_title("AUROC Task V2-A"); ax[0].set_xlabel("n"); ax[0].set_ylabel("AUROC"); ax[0].legend(); ax[0].grid(alpha=0.3)
ax[1].plot(ns, logreg_pr, marker="o", label="LogReg", color="#4a90e2")
ax[1].plot(ns, small_pr, marker="^", label="Small", color="#2ca02c")
ax[1].plot(ns, medium_pr, marker="D", label="Medium", color="#9467bd")
ax[1].set_title("AUPRC Task V2-A (ordered rise+diag)"); ax[1].set_xlabel("n"); ax[1].set_ylabel("AUPRC"); ax[1].legend(); ax[1].grid(alpha=0.3)
fig.suptitle("Task V2-A: does ordered pattern need more capacity?",fontsize=12)
fig.tight_layout(); fig.savefig("plots/taskV2_A.png",dpi=180); plt.close()
import shutil; Path("docs/plots").mkdir(exist_ok=True); shutil.copy("plots/taskV2_A.png","docs/plots/taskV2_A.png")
print("Saved plots/taskV2_A.png",flush=True)
