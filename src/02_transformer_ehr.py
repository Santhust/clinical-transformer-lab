"""
02_transformer_ehr.py — Transformer for EHR sequences (PyTorch)
EHR sequence prediction with masking and positional encoding
Run: python src/02_transformer_ehr.py  -> results/transformer_metrics.json + plots
"""
import numpy as np, torch, json, matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from attention_utils import extract_attention

Path("results").mkdir(exist_ok=True)

data = np.load("data/sequences.npz")
seqs, masks, labels = data["seqs"], data["masks"], data["labels"]
print("Sequences", seqs.shape, "imbalance", labels.mean())

# PyTorch dataset
class EHRDataset(torch.utils.data.Dataset):
    def __init__(self, seqs, masks, labels):
        self.seqs = torch.tensor(seqs, dtype=torch.float32)
        self.masks = torch.tensor(masks, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
    def __len__(self): return len(self.labels)
    def __getitem__(self, i): return self.seqs[i], self.masks[i], self.labels[i]

# Tiny transformer — the core you need to explain in interview
class EHRTransformer(torch.nn.Module):
    def __init__(self, d_in=6, d_model=32, nhead=4):
        super().__init__()
        self.input_proj = torch.nn.Linear(d_in, d_model)
        # learnable positional encoding
        self.pos_emb = torch.nn.Parameter(torch.randn(1, 8, d_model)*0.1)
        enc_layer = torch.nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=64, batch_first=True)
        self.encoder = torch.nn.TransformerEncoder(enc_layer, num_layers=2, enable_nested_tensor=False)
        self.classifier = torch.nn.Linear(d_model, 1)
    def forward(self, x, mask):
        # x: (B, T, d_in), mask: (B, T) 1=real, 0=pad
        h = self.input_proj(x) + self.pos_emb[:, :x.size(1), :]
        # transformer needs padding mask: True = ignore
        pad_mask = (mask == 0)
        h = self.encoder(h, src_key_padding_mask=pad_mask)  # (B,T,d_model)
        # masked mean pooling
        mask_e = mask.unsqueeze(-1)
        pooled = (h * mask_e).sum(1) / mask_e.sum(1).clamp(min=1)
        return self.classifier(pooled).squeeze(-1)

# train
X_tr, X_te, m_tr, m_te, y_tr, y_te = train_test_split(seqs, masks, labels, test_size=0.2, stratify=labels, random_state=42)
train_ds = EHRDataset(X_tr, m_tr, y_tr)
test_ds = EHRDataset(X_te, m_te, y_te)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)

device = torch.device("cpu")
model = EHRTransformer().to(device)
# handle imbalance via pos_weight
pos_weight = torch.tensor([(len(y_tr)-y_tr.sum())/y_tr.sum()])
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optim = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=30)

best_auc = 0
for epoch in range(35):
    model.train()
    losses=[]
    for x,m,y in train_loader:
        x,m,y = x.to(device), m.to(device), y.to(device)
        optim.zero_grad()
        logits = model(x,m)
        loss = criterion(logits, y)
        loss.backward(); optim.step()
        losses.append(loss.item())
    sched.step()
    # eval
    model.eval()
    with torch.no_grad():
        te_x = torch.tensor(X_te, dtype=torch.float32).to(device)
        te_m = torch.tensor(m_te, dtype=torch.float32).to(device)
        logits = model(te_x, te_m).cpu().numpy()
        prob = 1/(1+np.exp(-logits))
        auc = roc_auc_score(y_te, prob)
        auprc = average_precision_score(y_te, prob)
    if auc > best_auc:
        best_auc = auc
        best = (auc, auprc, prob.copy())
        torch.save(model.state_dict(), "results/best_transformer.pt")
    if epoch % 5 == 0:
        print(f"Epoch {epoch}: loss {np.mean(losses):.3f}  AUROC {auc:.3f}  AUPRC {auprc:.3f}")

auc, auprc, prob = best
print(f"Best Transformer — AUROC {auc:.3f}  AUPRC {auprc:.3f}")
with open("results/transformer_metrics.json","w") as f:
    json.dump({"AUROC": float(auc), "AUPRC": float(auprc)}, f, indent=2)

# --- Patient trajectory (NOT attention) -----------------------------------
# Kept because it is a useful sanity check on the inputs, but relabelled: this
# is the raw WBC series, not what the model attends to. Real attention is below.
idx = int(np.where(y_te == 1)[0][0])
seq_len = int(m_te[idx].sum())
fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(range(seq_len), X_te[idx, :seq_len, 2], marker="o", label="WBC (normalized)")
ax.set_xlabel("Visit (time)")
ax.set_ylabel("value")
ax.set_title(f"Example positive patient (n_visits={seq_len}) - input WBC trajectory")
ax.legend()
fig.tight_layout()
fig.savefig("plots/transformer_example_patient.png", dpi=180)
plt.close()

# --- Real attention weights ------------------------------------------------
# Requires a model, so reload the best checkpoint saved at line 85.
model = EHRTransformer().to(device)
model.load_state_dict(torch.load("results/best_transformer.pt", map_location=device))
model.eval()

te_x = torch.tensor(X_te, dtype=torch.float32).to(device)
te_m = torch.tensor(m_te, dtype=torch.float32).to(device)

# One positive and one negative patient of matched length, so the maps compare.
pos_idx = int(np.where(y_te == 1)[0][0])
neg_idx = int(np.where(y_te == 0)[0][0])
attention = extract_attention(model, te_x, te_m)

n_layers = len(attention)
n_heads = attention[0].shape[1]
fig, axes = plt.subplots(n_layers, n_heads, figsize=(3.0 * n_heads, 3.1 * n_layers), squeeze=False)
for layer in range(n_layers):
    w = attention[layer][pos_idx].cpu().numpy()   # (nhead, T, T)
    n = int(m_te[pos_idx].sum())
    for head in range(n_heads):
        ax = axes[layer][head]
        im = ax.imshow(w[head], cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(w.shape[-1]))
        ax.set_yticks(range(w.shape[-1]))
        ax.axhline(n - 0.5, color="white", lw=1.5, ls="--")
        ax.axvline(n - 0.5, color="white", lw=1.5, ls="--")
        ax.set_title(f"L{layer} H{head}", fontsize=9)
        ax.tick_params(labelsize=7)
        if layer == n_layers - 1:
            ax.set_xlabel("key (visit attended to)", fontsize=8)
        if head == 0:
            ax.set_ylabel("query (visit asking)", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046)
fig.suptitle(
    f"Learned attention, positive test patient (n_visits={n}) - dashed line marks the "
    "padding boundary.\nPadded key columns are exactly 0; padded query rows are NOT "
    "zero, which is why masked pooling is required.",
    fontsize=9,
)
fig.tight_layout()
fig.savefig("plots/transformer_attention.png", dpi=180)
plt.close()

print(f"attention weights: {n_layers} layers x {n_heads} heads, shape {tuple(attention[0].shape)}")
print("Saved results + plots/transformer_example_patient.png + plots/transformer_attention.png")
