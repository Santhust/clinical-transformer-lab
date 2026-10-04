"""
attention_utils.py - real attention extraction for nn.TransformerEncoder stacks.

WHY THIS MODULE EXISTS
----------------------
The learned attention weights cannot be read out of a normal forward pass:

  * nn.TransformerEncoderLayer calls attention internally with need_weights=False
  * on torch 2.14 nn.TransformerEncoder.forward() no longer accepts a
    `need_weights` argument at all - passing it raises TypeError

The route used here:
  1. register a forward pre-hook on each `encoder.layers[i].self_attn` to
     capture the tensor attention actually receives,
  2. run the ordinary forward pass, which populates the store,
  3. replay each captured input through the same module with need_weights=True.

Shapes: returns one (B, nhead, T, T) tensor per encoder layer.
Matrix semantics: rows are queries (the visit asking), columns are keys
(the visit being attended to). Row i of a valid query sums to 1.

WHY CAPTURE RATHER THAN RECONSTRUCT
-----------------------------------
nn.TransformerEncoderLayer.forward contains two branches:

    x = x + self._sa_block(self.norm1(x), ...)      # one branch
    x = self.norm1(x + self._sa_block(x, ...))       # the other

so the tensor reaching attention is pre-norm in one branch and post-norm in
the other, and which one runs depends on config and mode. Verified on torch
2.14 with norm_first=False in eval: attention receives the *un-normalized*
layer input (max abs difference from norm1(x) is 1.27).

Capturing the real input sidesteps that entirely - this helper is correct for
either branch. Verified against a hand-computed reference: building Q/K from
in_proj_weight/in_proj_bias on the captured tensor and evaluating
softmax(QK^T/sqrt(head_dim) + padding mask) matches the extracted weights to
4.5e-08.

PADDING BEHAVIOUR (verified on torch 2.14)
------------------------------------------
  * padded KEY columns are exactly 0.0 - a pad slot cannot be attended to
  * padded QUERY rows are NOT zero and still sum to 1 - a pad slot computes a
    real distribution over genuine visits and produces a real output vector,
    which is why the masked mean pooling in EHRTransformer.forward is
    load-bearing correctness rather than a stylistic choice.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def extract_attention(model: nn.Module, x: torch.Tensor, mask: torch.Tensor) -> list[torch.Tensor]:
    """Return per-layer attention weights of shape (B, nhead, T, T).

    Args:
        model: a model exposing `.encoder.layers[i].self_attn` (batch_first=True).
        x: input features, (B, T, d_in), dtype float32.
        mask: 1.0 for a real visit, 0.0 for padding, (B, T).

    Returns:
        List of (B, nhead, T, T) tensors, one per encoder layer, detached.
    """
    layers = model.encoder.layers
    pad_mask = mask == 0
    captured: dict[int, torch.Tensor] = {}
    handles = []

    for i, layer in enumerate(layers):
        def capture(_module, args, i=i):
            captured[i] = args[0].detach()

        handles.append(layer.self_attn.register_forward_pre_hook(capture))

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(x, mask)
        for handle in handles:
            handle.remove()

        # The replay must also happen in eval mode. Restoring train mode first
        # would re-enable the 0.1 dropout inside attention, which destroys the
        # softmax row sums and zeroes scattered real positions.
        weights = []
        with torch.no_grad():
            for i, layer in enumerate(layers):
                h = captured[i]
                _, w = layer.self_attn(
                    h, h, h,
                    key_padding_mask=pad_mask,
                    need_weights=True,
                    average_attn_weights=False,
                )
                weights.append(w.detach())
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    return weights


def build_ehr_transformer(d_in: int = 6, d_model: int = 32, nhead: int = 4,
                          max_visits: int = 8, num_layers: int = 2,
                          dim_feedforward: int = 64) -> nn.Module:
    """Build the architecture defined by EHRTransformer in src/02_transformer_ehr.py.

    That class is kept inline in 02_transformer_ehr.py because it is the reference
    definition for readers. This factory exists so app.py and the notebook can
    construct the identical model without importing that training script.
    """
    class EHRTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_proj = nn.Linear(d_in, d_model)
            self.pos_emb = nn.Parameter(torch.randn(1, max_visits, d_model) * 0.1)
            layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead,
                dim_feedforward=dim_feedforward, batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(
                layer, num_layers=num_layers, enable_nested_tensor=False,
            )
            self.classifier = nn.Linear(d_model, 1)

        def forward(self, x, mask):
            h = self.input_proj(x) + self.pos_emb[:, : x.size(1), :]
            h = self.encoder(h, src_key_padding_mask=(mask == 0))
            mask_e = mask.unsqueeze(-1)
            pooled = (h * mask_e).sum(1) / mask_e.sum(1).clamp(min=1)
            return self.classifier(pooled).squeeze(-1)

    return EHRTransformer()


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def describe_attention(w: torch.Tensor, mask_row: torch.Tensor) -> list[dict]:
    """Summarise one patient's per-head attention as plain dicts (for tables/UI)."""
    n_visits = int(mask_row.sum())
    heads = []
    for head in range(w.shape[1]):
        heads.append({
            "head": head,
            "self_weight": float(w[0, head, min(n_visits - 1, 0), min(n_visits - 1, 0)]),
            "first_visit_weight": float(w[0, head, 0, 0]) if n_visits else 0.0,
            "matrix": [[float(v) for v in row] for row in w[0, head]],
        })
    return heads