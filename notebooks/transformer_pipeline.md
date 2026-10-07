# Transformer Pipeline — Shape and Parameter Reference

Companion to `notebooks/architecture_walkthrough.ipynb`, which walks through the model in
`src/02_transformer_ehr.py` cell by cell. This document is the reference table: every stage of the
forward pass, its tensor shapes, and exactly where the 17,601 parameters live.

All numbers below were produced by introspecting the constructed model with
`torch.nn.Module.named_parameters()`, not computed by hand:

```python
import torch, sys
sys.path.insert(0, 'src')
from attention_utils import build_ehr_transformer
m = build_ehr_transformer()
for n, p in m.named_parameters():
    print(f'{n:52s} {str(tuple(p.shape)):16s} {p.numel():6d}')
print('TOTAL', sum(p.numel() for p in m.parameters()))
```

---

## Configuration

Defined in `attention_utils.build_ehr_transformer()` and matched exactly by the reference class
`EHRTransformer` in `src/02_transformer_ehr.py:52-70`. The factory exists so the notebook and the
training script construct an identical model and cannot drift apart.

| Symbol | Value | Meaning |
|---|---|---|
| `d_in` | 6 | input features per visit |
| `d_model` | 32 | hidden coordinates per visit |
| `nhead` | 4 | attention heads |
| `d_ff` (`dim_feedforward`) | 64 | feedforward inner width |
| `max_visits` | 8 | padded sequence length (T) |
| `num_layers` | 2 | encoder layers |

**Axis notation used throughout:**

| Axis | Symbol | Size | Meaning |
|---|---|---|---|
| Batch | `B` | 64 | patients in a training batch |
| Time | `T` | 8 | visit slots, padded |
| Feature | `d_in` | 6 | clinical measurements per visit |
| Model | `d_model` | 32 | hidden coordinates per visit |
| Head | `d_head` | 8 | `d_model // nhead` = 32 / 4 |

Attention operates over `T` (visits attend to visits). The 6 input features are mixed only *inside*
the Q/K/V projection, never against each other by attention.

---

## Table 1 — The full pipeline

| # | Stage | Type | Shape in → out | Parameters | Notes |
|---|---|---|---|---|---|
| 1 | `input_proj` | `Linear` | `(8,6)` → `(8,32)` | **224** | 192 weight + 32 bias |
| 2 | `+ pos_emb` | broadcast add | `(8,32)` → `(8,32)` | **256** | table `(1,8,32)`, sliced to `(8,32)` |
| 3 | encoder layer 0 | `TransformerEncoderLayer` | `(8,32)` → `(8,32)` | **8,544** | attention + FFN + 2 norms |
| 4 | encoder layer 1 | `TransformerEncoderLayer` | `(8,32)` → `(8,32)` | **8,544** | identical structure, independent weights |
| 5 | masked mean pool | arithmetic | `(8,32)` → `(32,)` | **0** | not a layer, just sum/divide |
| 6 | `classifier` | `Linear` | `(32,)` → `(1,)` | **33** | 32 weight + 1 bias |
| — | `encoder.norm` | `None` | — | **0** | no final norm (post-LN) |

With a real batch the shapes carry `B` on the front: `(B,8,6) → (B,8,32) → (B,32) → (B,)`.
The `squeeze(-1)` in `forward` drops the trailing 1, giving one logit per patient.

---

## Table 2 — Inside one encoder layer (8,544 parameters)

| Component | Type | Shape | Params | % of layer |
|---|---|---|---|---|
| `self_attn` | `MultiheadAttention` | — | **4,224** | 49.5% |
| ├ `in_proj_weight` | fused Q,K,V | `(96,32)` | 3,072 | 36.0% |
| ├ `in_proj_bias` | | `(96,)` | 96 | 1.1% |
| ├ `out_proj.weight` | recombines heads | `(32,32)` | 1,024 | 12.0% |
| └ `out_proj.bias` | | `(32,)` | 32 | 0.4% |
| **feedforward** | 2× `Linear` | `32→64→32` | **4,192** | 49.1% |
| ├ `linear1.weight` | widen | `(64,32)` | 2,048 | 24.0% |
| ├ `linear1.bias` | | `(64,)` | 64 | 0.7% |
| ├ `linear2.weight` | narrow | `(32,64)` | 2,048 | 24.0% |
| └ `linear2.bias` | | `(32,)` | 32 | 0.4% |
| `norm1` | `LayerNorm` | `(32,)` | 64 | 0.7% |
| `norm2` | `LayerNorm` | `(32,)` | 64 | 0.7% |
| 3× `Dropout` | no params | — | **0** | 0% |

### How each number is derived

**Fused Q/K/V projection.** The three projections compute from the same input, so PyTorch stores
them as one matrix with `3 * d_model` rows:

```
3 * d_model * d_model + 3 * d_model = 3*32*32 + 3*32 = 3,072 + 96 = 3,168
```
Shape `(96, 32)`. The `(96,)` bias is one per output coordinate. **Verified:** `3072 + 96 = 3,168`.

**Output projection.** Recombines the 4 heads back into 32 coordinates:

```
d_model * d_model + d_model = 32*32 + 32 = 1,024 + 32 = 1,056
```

**Attention subtotal:**

```
3,168 + 1,056 = 4,224
```

**Feedforward `linear1` (widen).** `(d_ff, d_model)` — output width is 64:

```
d_ff * d_model + d_ff = 64*32 + 64 = 2,048 + 64 = 2,112
```

**Feedforward `linear2` (narrow).** `(d_model, d_ff)` — maps 64 back to 32:

```
d_model * d_ff + d_model = 32*64 + 32 = 2,048 + 32 = 2,080
```

**Feedforward subtotal:**

```
2,112 + 2,080 = 4,192
```

**LayerNorm.** A scale and a bias per hidden coordinate — nothing more:

```
2 * d_model = 2*32 = 64      (per norm, two norms total)
```

**Layer subtotal:**

```
4,224 (attention) + 4,192 (feedforward) + 64 + 64 (norms) = 8,544
```

---

## Table 3 — Full per-parameter listing

Exactly as printed by torch, 29 tensors:

| Parameter | Shape | Params |
|---|---|---|
| `pos_emb` | `(1, 8, 32)` | 256 |
| `input_proj.weight` | `(32, 6)` | 192 |
| `input_proj.bias` | `(32,)` | 32 |
| `encoder.layers.0.self_attn.in_proj_weight` | `(96, 32)` | 3,072 |
| `encoder.layers.0.self_attn.in_proj_bias` | `(96,)` | 96 |
| `encoder.layers.0.self_attn.out_proj.weight` | `(32, 32)` | 1,024 |
| `encoder.layers.0.self_attn.out_proj.bias` | `(32,)` | 32 |
| `encoder.layers.0.linear1.weight` | `(64, 32)` | 2,048 |
| `encoder.layers.0.linear1.bias` | `(64,)` | 64 |
| `encoder.layers.0.linear2.weight` | `(32, 64)` | 2,048 |
| `encoder.layers.0.linear2.bias` | `(32,)` | 32 |
| `encoder.layers.0.norm1.weight` | `(32,)` | 32 |
| `encoder.layers.0.norm1.bias` | `(32,)` | 32 |
| `encoder.layers.0.norm2.weight` | `(32,)` | 32 |
| `encoder.layers.0.norm2.bias` | `(32,)` | 32 |
| `encoder.layers.1.*` | (identical shapes) | 8,544 |
| `classifier.weight` | `(1, 32)` | 32 |
| `classifier.bias` | `(1,)` | 1 |
| **TOTAL** | | **17,601** |

---

## Table 4 — Where the parameters live

| Group | Params | Share |
|---|---|---|
| Encoder (2 layers) | 17,088 | **97.1%** |
| `pos_emb` | 256 | 1.5% |
| `input_proj` | 224 | 1.3% |
| `classifier` | 33 | 0.2% |
| **Total** | **17,601** | 100% |

Arithmetic:

```
2 layers      = 2 * 8,544 = 17,088
pos_emb       = 1 * 8 * 32      =    256
input_proj    = 32*6 + 32       =    224
classifier    = 1*32 + 1        =     33
                              --------
                              17,601   matches count_parameters(model)
```

---

## The five things this table makes obvious

**1. Attention and feedforward are near-perfectly balanced.** 4,224 vs 4,192 — a difference of 32
parameters (0.8%). The component everyone calls "the transformer part" is exactly half the model.
The original 2017 paper put roughly two-thirds of parameters in the feedforward; at this scale the
two are even.

**2. The 6 input features barely matter.** `input_proj` is 224 parameters, 1.3% of the model.
Everything happens in the 32-dimensional space after it. This is why raising `d_model` from 32 to 64
roughly quadruples the parameter count while barely moving this stage.

**3. `in_proj_weight` is `(96, 32)`, not three matrices.** Q, K and V are one fused projection with
`3 * d_model` rows because all three are computed from the same input. Notebook section 8 splits
this tensor apart by hand to recover Q, K and V separately.

**4. Zero parameters for dropout, pooling, masking, or ReLU.** Only `Linear` and `LayerNorm` carry
weights. Dropout, masking, pooling and the activation are all parameter-free computation — worth
distinguishing "the model learned something" from "the model computed something."

**5. Every bias equals its layer's output width.** `Linear(6→32)` has a 32-vector bias,
`Linear(64→32)` has a 32-vector bias, `LayerNorm` has 32 scales + 32 biases. This is a reliable
sanity check when hand-counting parameters.

---

## What the model does, in order

```python
h = self.input_proj(x) + self.pos_emb[:, :x.size(1), :]   # (B,8,6) -> (B,8,32)
pad_mask = (mask == 0)                                     # True = ignore this slot
h = self.encoder(h, src_key_padding_mask=pad_mask)         # (B,8,32)
mask_e = mask.unsqueeze(-1)
pooled = (h * mask_e).sum(1) / mask_e.sum(1).clamp(min=1)  # (B,8,32) -> (B,32)
return self.classifier(pooled).squeeze(-1)                 # (B,32) -> (B,)
```

Three submodules only. Note what does *not* appear: there is no `nn.Embedding` anywhere in this
model. A word-based transformer maps integer ids to vectors through an embedding table; here a
visit arrives **already as six numbers**, so there is no vocabulary to look up and projecting those
six numbers *is* the embedding step. The only lookup-style table that survives is `pos_emb`, bounded
to 8 slots because no patient in this dataset exceeds 8 visits.

---

## Why attention is load-bearing for this task

The feedforward block processes each visit **independently** — visit 7's vector has no access to
visit 0's values. But the label rule in `src/generate_synthetic_ehr.py:42` is a difference across
the record:

```python
trend = visits_filled[-1, 2] - visits_filled[0, 2]   # last wbc minus FIRST wbc
```

No single visit contains that information, and the feedforward *provably cannot* compute it.
Something must physically bring visit 0's information to visit 7's position, and attention is the
only component that does so. Notebook section 9 checks whether any head actually learned to attend
to the ends of the record.

Note also that the label includes additive Gaussian noise (`np.random.normal(0, 0.8)`), so it is a
*noisy* function of the features, not a deterministic one. Part of the gap between AUROC and 1.0 is
irreducible by construction.

---

## Measured: what the trained model learned

Notebook section 9 loads `results/best_transformer.pt` and asks whether attention found the label
rule. Figures below are from this machine, checkpoint selected on validation AUROC of a seeded
stratified 70/15/15 split.

### Attention received per visit — patient #3, layer 1

Patient #3 has 4 real visits (the same patient printed as a grid in section 3).

| visit | attention received (mean over heads and queries) |
|---|---|
| 0 (first) | **0.578** |
| 1 | 0.134 |
| 2 | 0.069 |
| 3 (last) | 0.219 |

Both ends of the record carry 0.797 of the attention; the two middle visits share 0.203. Padding
columns measure exactly `0.000000` **after training** — masking is enforced by the mechanism
(`src_key_padding_mask` + `-inf`), not by gradients, so training cannot erode it.

For contrast, the untrained model on the same patient is flat: `0.248 0.249 0.252 0.250`, uniform
at 0.25 (attended identically to all four real visits).

### Per-head first/last attention mass, averaged over 1,000 patients

Chance level is `mean(1/k)` over the sample, not exactly 0.25 — visit counts run 2–8 with mean
5.06, so the null sits at **0.243**.

Role thresholds as used in the notebook: `last/first > 1.5` → recency head, `< 0.7` → baseline
head, otherwise mixed.

| layer | head | first | last | ratio | role (notebook) |
|---|---|---|---|---|---|
| 0 | 0 | 0.200 | 0.266 | 1.33 | mixed |
| 0 | 1 | 0.325 | 0.198 | 0.61 | baseline |
| 0 | 2 | 0.241 | 0.255 | 1.06 | mixed |
| 0 | 3 | 0.372 | 0.186 | 0.50 | baseline |
| 1 | 0 | 0.257 | 0.229 | 0.89 | mixed |
| 1 | 1 | 0.270 | 0.213 | 0.79 | mixed |
| 1 | 2 | 0.180 | **0.477** | **2.65** | **recency** |
| 1 | 3 | 0.395 | 0.236 | 0.60 | baseline |

Layer 1 shows a head biased to each end — head 2 reads the last visit, head 3 the first — which
matches the two operands of `wbc[-1] - wbc[0]`. Layer 0 shows baseline-leaning heads but no
recency head, consistent with local feature extraction rather than positional summarisation.

### Caveat: the notebook averages over all 8 query rows

The measurement uses `wn[b][:, :, 0].mean(axis=1)`, which averages over **all** T query rows
including padded ones. Padded query rows attend near-uniformly (see masks section), so including
them dilutes every ratio toward 1.0. Restricting to real query rows only makes every asymmetry
stronger:

| head | all rows | real rows only |
|---|---|---|
| L0 h0 | 1.33 | **1.59** (crosses the 1.5 threshold → recency) |
| L0 h3 | 0.50 | 0.39 |
| L1 h2 | 2.65 | **3.14** |
| L1 h3 | 0.60 | 0.30 |

The notebook's numbers are correct as computed but conservative — the labels are slightly
pessimistic versions of what is actually there.

### Self-focus (diagonal mass, layer 1, all patients)

| h0 | h1 | h2 | h3 | chance |
|---|---|---|---|---|
| 0.192 | 0.234 | 0.270 | 0.230 | 0.243 |

No head strongly attends to itself; only h0 is notably below chance. These heads do positional
work, not identity work.

### What cannot be explained

The label carries additive noise, so individual patients contradict the pattern. Patient #3 is
positive despite a *falling* wbc (`0.877 → 0.684`) — the generator's noise term decided that.
Attention cannot explain it because the value that decided it was never in the input. The model
found the dominant signal present in most patients, not a rule that holds for all of them.

---

## Conventions and gotchas

**Bias terms.** Every `Linear` uses PyTorch's default `bias=True`. The original 2017 paper writes
`Linear(x, W)` with no bias. If comparing against a paper or another codebase, that difference is
usually the discrepancy.

**Post-LN, so no final norm.** `build_ehr_transformer()` does not pass `norm=` to
`nn.TransformerEncoder`, so `encoder.norm is None` and adds 0 parameters. With `norm_first=False`
(post-LN) each layer already ends normalized, so a stack-level norm would be redundant. Pre-LN
architectures normally *do* pass a final norm, since pre-LN layers do not end normalized.

**Encoder layer has 8 submodules, decoder has 9.** Confirmed in the installed torch source
(`torch/nn/modules/transformer.py`): `dropout`, `dropout1`, `dropout2` for the encoder, plus
`dropout3` for the decoder, which stacks attention and feedforward as separate residual branches.
All three encoder dropouts are live: `dropout` inside the feedforward, `dropout1` after attention,
`dropout2` after the feedforward.

**`norm_first` is `False`.** Read it off the constructed layer rather than trusting docs — the
notebook does this in section 1. Post-LN is the 2017 arrangement; pre-LN trains more stably and is
what most modern libraries prefer.

**`activation` is stored as a bare function.** In torch 2.14, `layer.activation` is
`torch.nn.functional.relu` (a function), not an `nn.ReLU()` module — set that way at
`transformer.py:797`. So identity and equality checks against `nn.ReLU` (`is`, `==`) both return
`False`, and printing it shows `<function relu at 0x…>`. When inspecting a layer's attributes, check
`__name__` or `__module__` rather than assuming the stored object is a module.

**`L.dropout` is a module, not a number.** It prints as `Dropout(p=0.1, inplace=False)`; the
probability lives in `.p`.

**Two layers start with identical weights.** `nn.TransformerEncoder(layer, num_layers=2)` deep-copies
the template layer, so `layers[0]` and `layers[1]` hold the same values at init. They are separate
objects (`layers[0] is not layers[1]`), *not* weight-tying. Training diverges them immediately —
different gradients from the first step.

**Attention weight axes: `(nhead, T, T)`.** After indexing one patient out of `extract_attention`'s
output, axis 0 is heads, axis 1 is queries, axis 2 is keys. Which axis *survives* a reduction
therefore decides what the number means:

| reduction | leaves | gives |
|---|---|---|
| `w.mean(axis=(0, 2))` | queries | attention received per **visit** |
| `w.mean(axis=(1, 2))` | heads | a per-**head** figure |
| `w.mean(axis=(0, 1))` | keys | attention received per **visit**, over all queries |

Always check which axis is left over before naming the result. With one patient and 4 heads the two
first rows can coincidentally return the same index, so a mislabelled figure can look correct.

**A fully-masked attention row is mode-dependent.** If every key in a query row is masked, the
"correct" distribution is undefined, and torch 2.14 resolves it differently depending on mode:

| mode | path | fully-masked row |
|---|---|---|
| `train()` | `need_weights=False` → fused `scaled_dot_product_attention` | **zeros** |
| `eval()` | `torch.backends.mha` fast path | **`nan`** |

Both verified. So a degenerate input that behaves during training can silently produce NaN at
inference. Worth knowing before relying on either behaviour.

**An all-padding mask yields `classifier.bias`.** With `mask` all zeros, `clamp(min=1)` turns the
`0/0` in the pooling into `0/1`, so `pooled` is the zero vector and the output is exactly the
classifier's bias — the model's prior for "no visits". Remove the clamp and the pooling is NaN.
So `clamp(min=1)` is a genuine guard, but it only covers the pooling division: it cannot rescue a
NaN that attention already produced.

---

## Reproducing these numbers

```python
import torch, sys
sys.path.insert(0, 'src')
from attention_utils import build_ehr_transformer, count_parameters

model = build_ehr_transformer()
print(count_parameters(model))          # 17601

# grouped
groups = {
    "input_proj": model.input_proj,
    "encoder":    model.encoder,
    "classifier": model.classifier,
}
for name, mod in groups.items():
    n = sum(p.numel() for p in mod.parameters())
    print(f"{name:12s} {n:7,} ({100*n/17601:5.1f}%)")
```

Notebook section 6 (`3a4631df`) does exactly this and is the authoritative check on the figures above.

The section 9 measurements:

```python
import torch, sys, numpy as np
sys.path.insert(0, 'src')
from attention_utils import build_ehr_transformer, extract_attention

d = np.load('data/sequences.npz')
x = torch.tensor(d['seqs'][:1000], dtype=torch.float32)
m = torch.tensor(d['masks'][:1000], dtype=torch.float32)
nv = m.sum(1).numpy().astype(int)

model = build_ehr_transformer()
model.load_state_dict(torch.load('results/best_transformer.pt', map_location='cpu'))
model.eval()

with torch.no_grad():
    L = extract_attention(model, x, m)          # list, one (B, nhead, T, T) per layer

wn = L[1].numpy()                               # layer 1
first = last = 0.0
for b in range(len(wn)):
    k = int(nv[b])
    first += wn[b][:, :, 0].mean(axis=1)
    last  += wn[b][:, :, k - 1].mean(axis=1)
print('first', first / len(wn), 'last', last / len(wn))
print('chance', (1.0 / nv).mean())              # 0.243, not 0.25
```