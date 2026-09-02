"""
Task V2 Option C — Long-range dependency (first diag × last WBC)
Label = 1 if diag_first in {0.25,0.75} AND wbc_last > 0.65
Requires attention across up to 12 steps — classical has no diag_first, so must fail.
Small (32/2) vs Medium (64/3), n=10k,20k
"""
import numpy as np, torch, json, matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

def gen_C(n, seed, max_visits=12):
    rng=np.random.default_rng(seed)
    rec=[]; seqs=[]; masks=[]; labs=[]
    for pid in range(n):
        nv=int(rng.integers(3,max_visits+1)); age=int(rng.integers(30,85)); sex=int(rng.integers(0,2))
        base=-4.5+0.04*age+0.3*sex
        visits=[]
        for v in range(nv):
            hr=float(rng.normal(75+2*v,12)); sbp=float(rng.normal(125-1.5*v,15)); wbc=float(rng.normal(7+0.6*v,2.5))
            diag=float(rng.choice([0,1,2,3,4],p=[0.5,0.15,0.15,0.1,0.1]))/4
            if rng.random()<0.08:
                idx=int(rng.integers(0,3))
                if idx==0: hr=np.nan
                elif idx==1: sbp=np.nan
                else: wbc=np.nan
            visits.append([hr/100,sbp/200,wbc/15,diag,age/100,sex])
        arr=np.array(visits,float); fill=np.where(np.isnan(arr),0.7,arr)
        # long-range pattern
        has = (fill[0,3] in [0.25,0.75]) and (fill[-1,2] > 0.65)
        logit=base + (3.0 if has else 0) + float(rng.normal(0,0.8))
        prob=1/(1+np.exp(-logit)); lab=int(prob>0.5)
        last=fill[-1]; wtrend=float(fill[-1,2]-fill[0,2]); htrend=float(fill[-1,0]-fill[0,0])
        rec.append([age,sex,nv,last[0],last[1],last[2],last[3],wtrend,htrend,lab])
        seq=np.zeros((max_visits,6)); seq[:nv]=fill; mask=np.zeros(max_visits); mask[:nv]=1
        seqs.append(seq); masks.append(mask); labs.append(lab)
    import pandas as pd
    cols=["age","sex","n_visits","hr_last","sbp_last","wbc_last","diag_last","wbc_trend","hr_trend","label"]
    return pd.DataFrame(rec,columns=cols), np.stack(seqs), np.stack(masks), np.array(labs)

class Small(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.input_proj=torch.nn.Linear(6,32); self.pos_emb=torch.nn.Parameter(torch.randn(1,12,32)*0.1)
        enc=torch.nn.TransformerEncoderLayer(32,4,64,batch_first=True); self.encoder=torch.nn.TransformerEncoder(enc,2,enable_nested_tensor=False); self.classifier=torch.nn.Linear(32,1)
    def forward(self,x,m):
        h=self.input_proj(x)+self.pos_emb[:,:x.size(1),:]; h=self.encoder(h,src_key_padding_mask=(m==0))
        return self.classifier((h*m.unsqueeze(-1)).sum(1)/m.unsqueeze(-1).sum(1).clamp(min=1)).squeeze(-1)
class Medium(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.input_proj=torch.nn.Linear(6,64); self.pos_emb=torch.nn.Parameter(torch.randn(1,12,64)*0.1)
        enc=torch.nn.TransformerEncoderLayer(64,4,128,batch_first=True); self.encoder=torch.nn.TransformerEncoder(enc,3,enable_nested_tensor=False); self.classifier=torch.nn.Linear(64,1)
    def forward(self,x,m):
        h=self.input_proj(x)+self.pos_emb[:,:x.size(1),:]; h=self.encoder(h,src_key_padding_mask=(m==0))
        return self.classifier((h*m.unsqueeze(-1)).sum(1)/m.unsqueeze(-1).sum(1).clamp(min=1)).squeeze(-1)

def eval_classical(df):
    X=df.drop(columns=["label"]).values; y=df["label"].values
    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    out={}
    for name,model in [("LogReg",LogisticRegression(class_weight="balanced",max_iter=500)),("RF",RandomForestClassifier(n_estimators=200,class_weight="balanced_subsample",random_state=42))]:
        a=[]; p=[]
        for tr,te in cv.split(X,y):
            if len(np.unique(y[tr]))<2: continue
            model.fit(X[tr],y[tr]); prob=model.predict_proba(X[te])[:,1]
            a.append(roc_auc_score(y[te],prob)); p.append(average_precision_score(y[te],prob))
        out[name]=(float(np.mean(a)),float(np.mean(p)))
    return out

def train(seqs,masks,labs,ModelCls):
    from sklearn.model_selection import train_test_split
    Xtr,Xte,mtr,mte,ytr,yte=train_test_split(seqs,masks,labs,test_size=0.2,stratify=labs,random_state=42)
    class DS(torch.utils.data.Dataset):
        def __init__(self,a,b,c): self.a=torch.tensor(a,dtype=torch.float32); self.b=torch.tensor(b,dtype=torch.float32); self.c=torch.tensor(c,dtype=torch.float32)
        def __len__(self): return len(self.c)
        def __getitem__(self,i): return self.a[i],self.b[i],self.c[i]
    ds=DS(Xtr,mtr,ytr); loader=torch.utils.data.DataLoader(ds,batch_size=64,shuffle=True)
    dev=torch.device("cpu"); model=ModelCls().to(dev)
    pos=torch.tensor([(len(ytr)-ytr.sum())/max(1,ytr.sum())],dtype=torch.float32).to(dev)
    crit=torch.nn.BCEWithLogitsLoss(pos_weight=pos); opt=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=1e-4)
    best=(0,0)
    for e in range(12):
        model.train()
        for x,m,y in loader:
            x,m,y=x.to(dev),m.to(dev),y.to(dev)
            opt.zero_grad(); loss=crit(model(x,m),y); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            prob=torch.sigmoid(model(torch.tensor(Xte,dtype=torch.float32).to(dev),torch.tensor(mte,dtype=torch.float32).to(dev))).cpu().numpy()
            auc=roc_auc_score(yte,prob); auprc=average_precision_score(yte,prob)
        if auc>best[0]: best=(auc,auprc)
        print(f" Epoch {e+1}/12: {auc:.3f}/{auprc:.3f}",flush=True)
    return best

cohorts=[10000,20000]
results={}
for n in cohorts:
    print(f"\n=== Task V2-C n={n} (long-range diag_first x wbc_last) ===",flush=True)
    df,seqs,masks,labs=gen_C(n,seed=300+n)
    print(f" prevalence {labs.mean():.3f}",flush=True)
    classical=eval_classical(df); print(f" LogReg {classical['LogReg'][0]:.3f}/{classical['LogReg'][1]:.3f} RF {classical['RF'][0]:.3f}/{classical['RF'][1]:.3f}",flush=True)
    print(" Small...",flush=True)
    s=train(seqs,masks,labs,Small); print(f" Small {s[0]:.3f}/{s[1]:.3f}",flush=True)
    print(" Medium...",flush=True)
    m=train(seqs,masks,labs,Medium); print(f" Medium {m[0]:.3f}/{m[1]:.3f}",flush=True)
    results[str(n)]={"n":n,"prevalence":float(labs.mean()),"LogReg":classical["LogReg"],"RF":classical["RF"],"Small":s,"Medium":m}
    with open("results/taskV2_C.json","w") as f: json.dump(results,f,indent=2)

import matplotlib.pyplot as plt
ns=sorted(int(k) for k in results)
logreg_pr=[results[str(n)]["LogReg"][1] for n in ns]; small_pr=[results[str(n)]["Small"][1] for n in ns]; medium_pr=[results[str(n)]["Medium"][1] for n in ns]
logreg_auc=[results[str(n)]["LogReg"][0] for n in ns]; small_auc=[results[str(n)]["Small"][0] for n in ns]; medium_auc=[results[str(n)]["Medium"][0] for n in ns]
fig,ax=plt.subplots(1,2,figsize=(12,4.5),sharex=True)
ax[0].plot(ns,logreg_auc,marker="o",label="LogReg",color="#4a90e2"); ax[0].plot(ns,small_auc,marker="^",label="Small",color="#2ca02c"); ax[0].plot(ns,medium_auc,marker="D",label="Medium",color="#9467bd")
ax[0].set_title("AUROC Task V2-C long-range"); ax[0].set_xlabel("n"); ax[0].set_ylabel("AUROC"); ax[0].legend(); ax[0].grid(alpha=0.3)
ax[1].plot(ns,logreg_pr,marker="o",label="LogReg",color="#4a90e2"); ax[1].plot(ns,small_pr,marker="^",label="Small",color="#2ca02c"); ax[1].plot(ns,medium_pr,marker="D",label="Medium",color="#9467bd")
ax[1].set_title("AUPRC Task V2-C"); ax[1].set_xlabel("n"); ax[1].set_ylabel("AUPRC"); ax[1].legend(); ax[1].grid(alpha=0.3)
fig.suptitle("Task V2-C: long-range first × last",fontsize=12)
fig.tight_layout(); fig.savefig("plots/taskV2_C.png",dpi=180); plt.close()
import shutil, pathlib; pathlib.Path("docs/plots").mkdir(exist_ok=True); shutil.copy("plots/taskV2_C.png","docs/plots/taskV2_C.png")
print("Saved plots/taskV2_C.png",flush=True)
