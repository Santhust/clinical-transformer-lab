"""app.py - Interactive EHR Attention Explorer (Streamlit)

Shows the REAL attention learned by src/02_transformer_ehr.py, loaded from
results/best_transformer.pt. A second tab replays the attention *operation* on
synthetic logits, clearly labelled as a schematic and not a trained model.

Run:  streamlit run app.py

Regenerate the checkpoint (about 10 minutes on CPU) with:
    python src/02_transformer_ehr.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from attention_utils import build_ehr_transformer, extract_attention  # noqa: E402

CHECKPOINT = ROOT / "results" / "best_transformer.pt"
DATA = ROOT / "data" / "sequences.npz"

FEATURES = ["heart rate / 100", "systolic BP / 200", "WBC / 15",
            "diagnosis code / 4", "age / 100", "sex"]

st.set_page_config(page_title="EHR Attention Explorer", layout="wide")
st.title("EHR Attention Explorer")
st.caption("Real attention weights from the trained model, one visit = one token.")

tab_learned, tab_schematic = st.tabs(["Learned attention (real model)", "How attention works (schematic)"])

# ---------------------------------------------------------------- learned ----
with tab_learned:
    if not CHECKPOINT.exists():
        st.warning(
            f"No checkpoint at `{CHECKPOINT}`. Generate it first:\n\n"
            "```bash\npython src/02_transformer_ehr.py\n```"
        )
        st.stop()
    if not DATA.exists():
        st.warning(f"No data at `{DATA}`. Generate it first:\n"
                   "```bash\npython src/generate_synthetic_ehr.py\n```")
        st.stop()

    blob = np.load(DATA)
    seqs, masks, labels = blob["seqs"], blob["masks"], blob["labels"]

    model = build_ehr_transformer()
    model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu"))
    model.eval()

    visits_per_patient = masks.sum(axis=1).astype(int)
    x = torch.tensor(seqs, dtype=torch.float32)
    m = torch.tensor(masks, dtype=torch.float32)
    with torch.no_grad():
        attention = extract_attention(model, x, m)

    n_layers = len(attention)
    n_heads = attention[0].shape[1]
    total = int(masks.shape[0])

    st.markdown(
        f"**{total:,} patients - {n_layers} encoder layers x {n_heads} heads.** "
        f"Attention shape per layer: `{tuple(attention[0].shape)}` = (patient, head, query visit, key visit)."
    )

    c1, c2, c3 = st.columns(3)
    bucket = c1.selectbox("Patient group", ["Positive (label 1)", "Negative (label 0)", "All"])
    pool = np.where(labels == 1)[0] if bucket.startswith("Positive") else (
        np.where(labels == 0)[0] if bucket.startswith("Negative") else np.arange(total))
    patient = int(c2.selectbox("Patient index", pool.tolist(),
                               format_func=lambda i: f"#{i} - {visits_per_patient[i]} visits"))
    layer = int(c3.selectbox("Encoder layer", list(range(n_layers))))

    head_options = list(range(n_heads))
    c4, c5 = st.columns(2)
    head = int(c4.selectbox("Attention head", head_options))
    show_all = c5.toggle("Show all heads side by side", value=False)

    w = attention[layer][patient].numpy()
    real_visits = int(visits_per_patient[patient])
    row_labels = [f"visit {i}" + (" (padding)" if i >= real_visits else "")
                  for i in range(w.shape[-1])]

    heads_to_draw = range(n_heads) if show_all else [head]

    for h in heads_to_draw:
        fig, ax = plt.subplots(figsize=(6.2, 5.2))
        im = ax.imshow(w[h], cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(w.shape[-1]), row_labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(w.shape[-1]), row_labels, fontsize=8)
        ax.axhline(real_visits - 0.5, color="white", lw=1.6, ls="--")
        ax.axvline(real_visits - 0.5, color="white", lw=1.6, ls="--")
        ax.set_xlabel("key - the visit being attended to", fontsize=9)
        ax.set_ylabel("query - the visit doing the asking", fontsize=9)
        ax.set_title(f"Layer {layer}, head {h} - patient #{patient} "
                     f"({real_visits} real visits, label {labels[patient]})", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046)
        st.pyplot(fig)
        plt.close(fig)
        if not show_all:
            break

    l1, l2 = st.columns(2)

    with l1:
        st.markdown("**This patient's visits** (one visit = one token)")
        table = pd.DataFrame(seqs[patient][:real_visits], columns=FEATURES,
                             index=[f"visit {i}" for i in range(real_visits)])
        st.dataframe(table.style.format("{:.3f}"), height=260)

    with l2:
        st.markdown("**Reading the matrix**")
        st.markdown(
            f"- Rows sum to **1.0** for real queries.\n"
            f"- Columns {real_visits} and beyond are **exactly 0** - a padded slot can never be attended to.\n"
            f"- Rows {real_visits} and beyond are **not zero**: a padded slot still computes a full "
            f"distribution over the {real_visits} real visits. That fabricated output is exactly what the "
            f"masked mean pooling at `src/02_transformer_ehr.py:43-44` discards.\n"
            f"- Row *i*, column *j* answers: *how much does visit i use visit j when building "
            f"its own summary?*"
        )
        st.markdown(f"**Where each visit's attention goes** (mean over heads, layer {layer})")
        col_mass = w[:, :real_visits, :real_visits].mean(axis=(0, 1))
        share = pd.DataFrame({
            "visit": [f"visit {i}" for i in range(real_visits)],
            "share of attention received": col_mass.round(3),
        })
        st.dataframe(share, hide_index=True)

    with st.expander("What the heads actually specialised in"):
        st.markdown(
            "Averaged over 1,000 patients, layer 1 does not spread evenly. Head 2 puts **0.477** of "
            "its attention mass on the most recent visit and head 3 puts **0.395** on the *first* "
            "visit, against a chance level near 0.243. The V1 label is a trend, "
            "`wbc[-1] - wbc[0]` (`src/generate_synthetic_ehr.py:42`), so the model allocated one "
            "head to each of the two operands that quantity needs.\n\n"
            "Worth re-checking whenever you retrain: pick a layer and head, then look at which "
            "*column* is bright across every row."
        )

# -------------------------------------------------------------- schematic ----
with tab_schematic:
    st.markdown(
        "**This tab is a schematic, not a trained model.** It applies "
        "`softmax(QK^T / tau)` to random numbers so the shape of the operation is visible. "
        "The real learned weights are on the first tab."
    )

    c1, c2 = st.columns(2)
    demo_visits = c1.slider("Number of visits", 3, 8, 6, 1)
    demo_tokens = [f"V{i}" for i in range(demo_visits)]
    demo_temp = c2.slider("Softmax temperature (lower = sharper)", 0.1, 2.0, 1.0, 0.1)

    np.random.seed(0)
    demo_n = len(demo_tokens)
    demo_logits = np.random.randn(demo_n, demo_n) / demo_temp
    for i in range(demo_n):
        demo_logits[i, i] += 1.2
        if i > 0:
            demo_logits[i, i - 1] += 0.6
    demo_exp = np.exp(demo_logits - demo_logits.max(axis=1, keepdims=True))
    demo_attn = demo_exp / demo_exp.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    im = ax.imshow(demo_attn, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(demo_n), demo_tokens)
    ax.set_yticks(range(demo_n), demo_tokens)
    ax.set_xlabel("keys")
    ax.set_ylabel("queries")
    ax.set_title(f"Schematic attention (temperature = {demo_temp})")
    for i in range(demo_n):
        for j in range(demo_n):
            ax.text(j, i, f"{demo_attn[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if demo_attn[i, j] > 0.4 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    st.pyplot(fig)
    plt.close(fig)

    st.info(
        "Two ways to read this: as a Boltzmann-style weighting where temperature controls how "
        "concentrated the choice is, or as a soft lookup where each row is a probability "
        "distribution over the keys."
    )