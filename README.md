# Clinical Transformer Lab — Classical and Transformer Models for Sequential Medical Prediction

A reproducible enquiry into when sequential representation learning helps for medical prediction on longitudinal Electronic Health Record (EHR) data.

*Synthetic EHR, controlled temporal tasks, honest comparison on modest hardware.*

**Live site (GitHub Pages):** `https://Santhust.github.io/clinical-transformer-lab/` (see `docs/index.html`)
**Interactive demo:** `streamlit run app.py` (local)

## Abstract

I measure and compare the predictive efficacy of a transformer architecture against classical baselines on sequential EHR (Electronic Health Records) data. Using EHR data with controlled temporal structure — variable-length histories, class imbalance and missingness — I compare logistic regression and random forest with handcrafted trends against a PyTorch transformer with self-attention, positional encodings and padding masks. Across four tasks of increasing temporal complexity — simple peak, ordered sustained rise, cumulative exposure and long-range first–last interaction — and cohorts from 1k to 20k patients, I find the advantage is conditional. On simple and ordered patterns the transformer requires scale to match and then overtake classical performance; cumulative mean remains approximated by the last value; long-range dependency shows decisive transformer gain (AUROC 0.96 vs 0.71) even at 10k. The enquiry is fully reproducible on modest hardware (Raspberry Pi 8 GB, GNU/Linux, Raspberry Pi OS, CPU) and demonstrates when sequential representation learning matters for chronic disease monitoring, with Rheumatoid Arthritis as motivating example.

Keywords: EHR, sequential prediction, transformer, self-attention, imbalanced evaluation, synthetic data

## Study Design — Map of the Enquiry

| Task | Visits | Label rule | What it tests |
|---|---|---|---|
| V1 Baseline | 2–8 | trend `ΔWBC` | Simple non-linear peak |
| V2-A Ordered | 3–12 | 2 consecutive rises + diag code in window | Local order + cross-feature binding |
| V2-B Cumulative | 3–12 | mean WBC across visits | Integration over history |
| V2-C Long-range | 3–12 | diag_first × wbc_last interaction | Memory across 12 steps |

All share evaluation (stratified 5-fold classical, 80/20 transformer, AUROC/AUPRC) and training (AdamW, 12 epochs, batch 64).

See `docs/index.html` for the full living paper with interactive tables, per-task data previews, and versioned logs.

## Tech Stack and Concepts Explored

**Core stack:** Python — NumPy, pandas, matplotlib; PyTorch (tensors, `nn.Module`, `TransformerEncoder`, AdamW); scikit-learn (stratified CV, class-weighted loss); DataTables.js, Streamlit/Jupyter.

**Concepts:** self-attention / QKV, positional encoding, padding masks, masked pooling, imbalanced evaluation (AUROC/AUPRC), feature engineering vs representation learning, reproducibility (seed 42).

**Environment:** Python 3.13 · PyTorch 2.14 (CPU) · Raspberry Pi 8 GB — all experiments run locally.

## How to explore

**On GitHub Pages (static):** searchable 100-row previews, static plots, scrollable code/logs in each card — no install.

**Locally after `git clone`:**
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/generate_synthetic_ehr.py          # → data/synthetic_ehr.csv + sequences.npz
python src/01_classical_baseline.py           # → results/classical_metrics.json
python src/05_scaling_3seeds.py               # 3 seeds, shaded bands
python src/08_taskV2_A.py                     # V2-A
python src/09_taskV2_B.py                     # V2-B
python src/10_taskV2_C.py                     # V2-C (key finding)
streamlit run app.py                          # attention playground
jupyter lab notes.ipynb                       # secondary scratch notebook
```

## Repository structure

```
src/                 # generation + baselines + transformer + scaling + V2 tasks
  generate_synthetic_ehr.py
  01_classical_baseline.py
  02_transformer_ehr.py
  05_scaling_3seeds.py
  08_taskV2_A.py
  09_taskV2_B.py
  10_taskV2_C.py
data/                # synthetic_ehr.csv + sequences.npz (generated)
results/             # JSON + logs per run (versioned)
plots/               # all figures (generated)
docs/                # GitHub Pages site (index.html + previews + plots)
notes.ipynb          # secondary scratch notebook for quick experiments
app.py               # Streamlit interactive attention demo
```

All figures are generated from code. No external data required.

## Limitations

Synthetic data isolates methodological questions but does not model RA-specific markers (RF, anti-CCP, DAS28). Real-data extension is MIMIC-IV (300k admissions, ICD M05–M06) with attention-based interpretability as next step.

## License

GPL-3.0-only — see [LICENSE](LICENSE). Copyright (C) 2026 Santhust.
