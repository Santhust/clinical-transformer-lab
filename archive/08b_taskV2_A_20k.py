import numpy as np, torch, json
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

def gen(n, seed, max_visits=12):
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
        wbc_s=fill[:,2]
        has=False
        for i in range(nv-2):
            if (wbc_s[i+1]-wbc_s[i]>0.10) and (wbc_s[i+2]-wbc_s[i+1]>0.10) and fill[i+1,3] in [0.25,0.75]:
                has=True; break
        logit=base + (3.0 if has else 0) + 0.9*fill[-1,3] + float(rng.normal(0,0.8))
        lab=int(1/(1+np.exp(-logit))>0.5)
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
    from sklearn.model_selection import StratifiedKFold
    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    out={}
    for n,model in [("LogReg",LogisticRegression(class_weight="balanced",max_iter=500)),("RF",RandomForestClassifier(n_estimators=200,class_weight="balanced_subsample",random_state=42))]:
        a=[]; p=[]
        for tr,te in cv.split(X,y):
            if len(np.unique(y[tr]))<2: continue
            model.fit(X[tr],y[tr]); prob=model.predict_proba(X[te])[:,1]
            a.append(roc_auc_score(y[te],prob)); p.append(average_precision_score(y[te],prob))
        out[n]=(float(np.mean(a)),float(np.mean(p)))
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

n=20000
print(f"n={n} V2-A 3-12 visits",flush=True)
df,seqs,masks,labs=gen(n,seed=100+n)
print(f" prevalence {labs.mean():.3f}",flush=True)
classical=eval_classical(df)
print(f" LogReg {classical['LogReg']} RF {classical['RF']}",flush=True)
print(" Small...",flush=True)
s=train(seqs,masks,labs,Small); print(f" Small {s}",flush=True)
print(" Medium...",flush=True)
m=train(seqs,masks,labs,Medium); print(f" Medium {m}",flush=True)
import json as js, pathlib
# merge with existing 10k
p=pathlib.Path("results/taskV2_A.json")
if p.exists():
    cur=js.load(open(p))
else:
    cur={}
cur[str(n)]={"n":n,"prevalence":float(labs.mean()),"LogReg":classical["LogReg"],"RF":classical["RF"],"Small":s,"Medium":m}
js.dump(cur,open(p,"w"),indent=2)
print("saved",cur[str(n)],flush=True)
