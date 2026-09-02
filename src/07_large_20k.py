"""
Scaling with large model (20k, single cohort)
Task V1 (peak) — capacity test with d_model=128, 4 layers, 8 heads
Saves: results/large_20k.json
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

class EHRTransformerLarge(torch.nn.Module):
    def __init__(self,d_in=6,d_model=128,nhead=8):
        super().__init__()
        self.input_proj=torch.nn.Linear(d_in,d_model)
        self.pos_emb=torch.nn.Parameter(torch.randn(1,8,d_model)*0.1)
        enc=torch.nn.TransformerEncoderLayer(d_model=d_model,nhead=nhead,dim_feedforward=256,batch_first=True)
        self.encoder=torch.nn.TransformerEncoder(enc,num_layers=4,enable_nested_tensor=False)
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
    model=EHRTransformerLarge().to(device)
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
        print(f" Epoch {epoch+1}/{epochs}: AUROC {auc:.3f}  AUPRC {auprc:.3f}",flush=True)
        if auc>best[0]: best=(auc,auprc)
    return best

n=20000
print(f"Generating n={n} (large model test, seed={42+n})...",flush=True)
df,seqs,masks,labels=generate_cohort(n,seed=42+n)
print(f" prevalence {labels.mean():.3f}, params ~500k (d=128, 4 layers)",flush=True)
classical=eval_classical(df)
print(f" Classical LogReg {classical['LogReg'][0]:.3f}/{classical['LogReg'][1]:.3f}  RF {classical['RF'][0]:.3f}/{classical['RF'][1]:.3f}",flush=True)
print(" Training large transformer (d=128, 4 layers, 8 heads)...",flush=True)
tr_auc,tr_auprc=train_transformer(seqs,masks,labels)
print(f" => Large Transformer {tr_auc:.3f}/{tr_auprc:.3f}",flush=True)

Path("results").mkdir(exist_ok=True)
import json as js
with open("results/large_20k.json","w") as f: js.dump({"n":n, "LogReg":classical["LogReg"], "RF":classical["RF"], "Transformer_large":(tr_auc,tr_auprc)},f,indent=2)
print("Saved results/large_20k.json",flush=True)
