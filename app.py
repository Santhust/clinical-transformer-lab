"""app.py — Interactive Transformer Playground (Streamlit)
Run:  streamlit run app.py  (with .venv activated)
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="EHR Attention Playground", layout="wide")
st.title("🧠 EHR Attention Playground — Interactive Self-Attention")

st.markdown("Explore how **self-attention** over EHR visits works — each visit (token) decides how much to attend to every other visit. This is the core of the transformer for sequential medical prediction.")

col1, col2 = st.columns([1,2])

with col1:
    n_visits = st.slider("Number of visits in patient history", 3, 8, 6, 1)
    tokens = [f"V{i}" for i in range(n_visits)]
    st.write(f"Visits: `{tokens}` — each visit attends to all others")
    temp = st.slider("Softmax temperature τ (lower = sharper)", 0.1, 2.0, 1.0, 0.1)
    st.caption("τ controls how focused attention is. Low τ → model focuses on one visit.")
    seed = st.slider("Random seed", 0, 10, 0)

with col2:
    np.random.seed(seed)
    n = len(tokens)
    # simulate QK^T / τ -> softmax
    logits = np.random.randn(n, n) / temp
    # bias: each word slightly attends to itself and neighbors
    for i in range(n):
        logits[i,i] += 1.2
        if i>0: logits[i,i-1] += 0.6
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    attn = exp / exp.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(6,4.5))
    im = ax.imshow(attn, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(n), tokens); ax.set_yticks(range(n), tokens)
    ax.set_xlabel("Keys"); ax.set_ylabel("Queries")
    ax.set_title(f"Attention map (τ={temp})")
    for i in range(n):
        for j in range(n):
            ax.text(j,i, f"{attn[i,j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if attn[i,j]>0.4 else "black")
    plt.colorbar(im, ax=ax, shrink=0.8)
    st.pyplot(fig)

st.divider()
st.markdown("**Physics intuition:** Think of attention as a *fully-connected interaction* where every token (particle) computes its interaction strength with every other token via $QK^T/\\sqrt{d}$, then softmax normalizes it — like a Boltzmann distribution with temperature τ!")

st.info("Next: We'll replace this fake random attention with *learned* Q, K, V matrices in PyTorch.")
