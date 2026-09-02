"""
01_visual_test.py — Prove the visualization + torch loop works in pi

Run:  python 01_visual_test.py  (with .venv activated)
Outputs: plots/mlp_diagram.png , plots/attention_heatmap.png
"""
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

plots = Path("plots")
plots.mkdir(exist_ok=True)

print(f"torch {torch.__version__} — {torch.randn(2,2)}")

# --- 1. MLP Architecture Diagram (pure matplotlib, no external assets) ---
def draw_mlp(layers=[4,5,3,1], title="MLP Architecture: 4 → 5 → 3 → 1"):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title(title, fontsize=13, pad=15)

    layer_x = np.linspace(1.5, 8.5, len(layers))
    colors = ["#6baed6", "#74c476", "#fd8d3c", "#9e9ac8"]
    positions = []  # list per layer of (x,y)

    for i, (n, x) in enumerate(zip(layers, layer_x)):
        ys = np.linspace(1, 5, n) if n>1 else [3]
        # center small layers vertically
        if n < max(layers):
            ys = ys - np.mean(ys) + 3
        pos = []
        for y in ys:
            c = plt.Circle((x, y), 0.35, color=colors[i], ec="black", lw=1.2, zorder=3)
            ax.add_patch(c)
            pos.append((x,y))
        positions.append(pos)
        ax.text(x, 0.4, f"Layer {i}\n({n})", ha="center", fontsize=8, color="#333")

    # connections
    for a, b in zip(positions[:-1], positions[1:]):
        for (x1,y1) in a:
            for (x2,y2) in b:
                ax.plot([x1,y1],[x2,y2],  color="gray", lw=0.6, alpha=0.35, zorder=1)  # bug: x/y swap to test?
                # correct:
                ax.plot([x1, x2], [y1, y2], color="gray", lw=0.6, alpha=0.35, zorder=1)

    # Fix the bug line above by redrawing properly (keep code simple)
    # Actually redo cleanly — clear and redraw
    # Remove wrong lines by overplotting white then redraw — easier: just recreate
    plt.savefig(plots / "mlp_diagram.png", dpi=180, bbox_inches="tight")
    plt.close()
    print(f"saved {plots/'mlp_diagram.png'}")

# Redo cleanly without bug:
def draw_mlp_clean(layers=[4,5,3,1]):
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
    ax.set_title("MLP Architecture  4 → 5 → 3 → 1  (with biases & activations)", fontsize=12, pad=12)
    xs = np.linspace(1.5, 8.5, len(layers))
    cols = ["#4a90e2", "#50c878", "#ffa552", "#a78bfa"]
    pos = []
    for i, (n, x) in enumerate(zip(layers, xs)):
        ys = np.linspace(1.2, 4.8, n) if n>1 else np.array([3.0])
        if n < max(layers):
            ys = ys - ys.mean() + 3
        layer_pos=[]
        for y in ys:
            ax.add_patch(plt.Circle((x,y),0.36, color=cols[i], ec="black", lw=1.3, zorder=3))
            layer_pos.append((x,y))
        pos.append(layer_pos)
        ax.text(x, 0.45, f"{'Input' if i==0 else 'Hidden '+str(i) if i<len(layers)-1 else 'Output'}\n{n} neurons", ha="center", fontsize=7.5, color="#222",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#ccc", alpha=0.9))
    for l in range(len(pos)-1):
        for (x1,y1) in pos[l]:
            for (x2,y2) in pos[l+1]:
                ax.plot([x1,x2],[y1,y2], color="#888", lw=0.7, alpha=0.3, zorder=1)
    # annotations
    ax.text(5, 5.6, "Each line = weight  ×  + bias  →  activation (ReLU/Sigmoid)", ha="center", fontsize=8, style="italic", color="#555")
    fig.tight_layout()
    fig.savefig(plots/"mlp_diagram.png", dpi=180, bbox_inches="tight")
    plt.close()
    print("✓ mlp_diagram.png")

draw_mlp_clean()

# --- 2. Attention Heatmap (foreshadows transformers) ---
def draw_attention():
    np.random.seed(0)
    # Simulate attention weights for sentence: "the cat sat on the mat"
    tokens = ["the","cat","sat","on","the","mat"]
    n = len(tokens)
    # fake attention - each query attends sharply to related keys
    attn = np.random.rand(n,n) * 0.3
    for i in range(n):
        attn[i,i] += 0.6  # self
        if i>0: attn[i,i-1] += 0.4
    attn = np.exp(attn) / np.exp(attn).sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(5.2,4.5))
    im = ax.imshow(attn, cmap="viridis", vmin=0, vmax=0.5)
    ax.set_xticks(range(n), tokens); ax.set_yticks(range(n), tokens)
    ax.set_xlabel("Keys (what to look at)"); ax.set_ylabel("Queries (what is asking)")
    ax.set_title("Self-Attention Heatmap\n(darker = more attention)", fontsize=11, pad=10)
    for i in range(n):
        for j in range(n):
            ax.text(j,i, f"{attn[i,j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if attn[i,j]>0.25 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8, label="attention weight")
    fig.tight_layout()
    fig.savefig(plots/"attention_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close()
    print("✓ attention_heatmap.png")

draw_attention()
print("Done. Check plots/ folder in VSCode.")
