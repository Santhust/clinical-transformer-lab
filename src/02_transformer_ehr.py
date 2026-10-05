"""
02_transformer_ehr.py — Transformer for EHR sequences (PyTorch)
EHR sequence prediction with masking and positional encoding
Run: python src/02_transformer_ehr.py  -> results/transformer_metrics.json + plots

Protocol
--------
Everything is seeded, so a rerun reproduces these numbers exactly.
A stratified 70/15/15 train/validation/test split is used. The best epoch is
selected on *validation* AUROC and test metrics are computed once, at the end,
from the selected checkpoint — the test set is never used for model selection.
Classical baselines are fitted on the identical split so the comparison is like
for like rather than 5-fold CV against a single holdout.
"""
import sys
from pathlib import Path

import numpy as np, torch, json, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from attention_utils import extract_attention

SEED = 42
EPOCHS = 35
BATCH_SIZE = 64
LR = 2e-3
WEIGHT_DECAY = 1e-4

np.random.seed(SEED)
torch.manual_seed(SEED)

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

# --- splits: stratified train / validation / test ---------------------------
trval_idx, te_idx = train_test_split(np.arange(len(labels)), test_size=0.15,
                                     stratify=labels, random_state=SEED)
tr_idx, va_idx = train_test_split(trval_idx, test_size=0.15 / 0.85,
                                  stratify=labels[trval_idx], random_state=SEED)
print(f"split -> train {len(tr_idx)}  val {len(va_idx)}  test {len(te_idx)}")

def take(idx, src):
    return np.asarray([src[i] for i in idx])

tr_ds = EHRDataset(take(tr_idx, seqs), take(tr_idx, masks), take(tr_idx, labels))
tr_loader = torch.utils.data.DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True)

device = torch.device("cpu")
model = EHRTransformer().to(device)

# handle imbalance via pos_weight
y_tr = torch.tensor(take(tr_idx, labels), dtype=torch.float32)
pos_weight = torch.tensor([(len(y_tr) - y_tr.sum()) / y_tr.sum()], dtype=torch.float32)
print(f"pos_weight {float(pos_weight):.3f}")
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
# T_max matches EPOCHS so the cosine completes instead of restarting
sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=EPOCHS)

def as_tensors(idx):
    return (torch.tensor(take(idx, seqs), dtype=torch.float32).to(device),
            torch.tensor(take(idx, masks), dtype=torch.float32).to(device),
            take(idx, labels))

va_x, va_m, va_y = as_tensors(va_idx)

best_val_auc, best_epoch, best_state = -1.0, -1, None
for epoch in range(EPOCHS):
    model.train()
    losses = []
    for x, m, y in tr_loader:
        x, m, y = x.to(device), m.to(device), y.to(device)
        optim.zero_grad()
        logits = model(x, m)
        loss = criterion(logits, y)
        loss.backward(); optim.step()
        losses.append(loss.item())
    sched.step()
    # model selection on VALIDATION, never on test
    model.eval()
    with torch.no_grad():
        val_logits = model(va_x, va_m).cpu().numpy()
    val_auc = roc_auc_score(va_y, val_logits)
    if val_auc > best_val_auc:
        best_val_auc, best_epoch = val_auc, epoch
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if epoch % 5 == 0:
        print(f"Epoch {epoch}: loss {np.mean(losses):.3f}  val AUROC {val_auc:.3f}")

print(f"selected epoch {best_epoch} on validation AUROC {best_val_auc:.3f}")
model.load_state_dict(best_state)
torch.save(best_state, "results/best_transformer.pt")

# --- test metrics, computed once from the selected checkpoint --------------
te_x, te_m, te_y = as_tensors(te_idx)
model.eval()
with torch.no_grad():
    logits = model(te_x, te_m).cpu().numpy()
prob = 1 / (1 + np.exp(-logits))
auc = roc_auc_score(te_y, prob)
auprc = average_precision_score(te_y, prob)
print(f"Held-out test — AUROC {auc:.3f}  AUPRC {auprc:.3f}  (AUPRC floor = {np.mean(te_y):.3f})")

# --- classical baselines on the SAME split, like for like ------------------
def classical_features(s, m):
    """Rebuild the flattened feature row the classical models expect."""
    n_visits = m.sum(1).astype(int)
    rows = []
    for seq, nv in zip(s, n_visits):
        last = seq[nv - 1]
        rows.append([
            seq[0, 4], seq[0, 5], nv,
            last[0], last[1], last[2], last[3],
            seq[nv - 1, 2] - seq[0, 2], last[0] - seq[0, 0],
        ])
    return np.array(rows)

feats = classical_features(seqs, masks)
tr_f, te_f = feats[tr_idx], feats[te_idx]
classical = {}
for name, est in [("LogisticRegression", LogisticRegression(class_weight="balanced", max_iter=500)),
                  ("RandomForest", RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample", random_state=SEED))]:
    est.fit(tr_f, take(tr_idx, labels))
    p = est.predict_proba(te_f)[:, 1]
    classical[name] = {"AUROC": float(roc_auc_score(te_y, p)),
                       "AUPRC": float(average_precision_score(te_y, p))}
    print(f"  {name:20s} AUROC {classical[name]['AUROC']:.3f}  AUPRC {classical[name]['AUPRC']:.3f}")

with open("results/transformer_metrics.json", "w") as f:
    json.dump({
        "seed": SEED,
        "split": {"train": len(tr_idx), "val": len(va_idx), "test": len(te_idx),
                  "strategy": "stratified 70/15/15", "model_selection": "best epoch by validation AUROC"},
        "epochs": EPOCHS,
        "selected_epoch": best_epoch,
        "validation_AUROC": float(best_val_auc),
        "test_AUROC": float(auc),
        "test_AUPRC": float(auprc),
        "test_prevalence": float(np.mean(te_y)),
        "classical_same_split": classical,
    }, f, indent=2)

# --- attention from the selected checkpoint --------------------------------
# Drawn from the VALIDATION split so the figure cannot be accused of test peeking.
pos_val = int(np.where(va_y == 1)[0][0])
attention = extract_attention(model, va_x, va_m)
n_visits_val = int(va_m[pos_val].sum())

# Patient trajectory (NOT attention) — a sanity check on the inputs.
fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(range(n_visits_val), va_x[pos_val, :n_visits_val, 2].cpu().numpy(), marker="o",
        label="WBC (normalized)")
ax.set_xlabel("Visit (time)")
ax.set_ylabel("value")
ax.set_title(f"Example positive validation patient (n_visits={n_visits_val}) - input WBC trajectory")
ax.legend()
fig.tight_layout()
fig.savefig("plots/transformer_example_patient.png", dpi=180)
plt.close()

n_layers = len(attention)
n_heads = attention[0].shape[1]
fig, axes = plt.subplots(n_layers, n_heads, figsize=(3.0 * n_heads, 3.1 * n_layers), squeeze=False)
for layer in range(n_layers):
    w = attention[layer][pos_val].cpu().numpy()   # (nhead, T, T)
    n = n_visits_val
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
    f"Learned attention, positive validation patient (n_visits={n}) - dashed line marks the "
    "padding boundary.\nPadded key columns are exactly 0; padded query rows are NOT "
    "zero, which is why masked pooling is required.",
    fontsize=9,
)
fig.tight_layout()
fig.savefig("plots/transformer_attention.png", dpi=180)
plt.close()

print(f"attention weights: {n_layers} layers x {n_heads} heads, shape {tuple(attention[0].shape)}")
print("Saved results/transformer_metrics.json + results/best_transformer.pt + both plots")
