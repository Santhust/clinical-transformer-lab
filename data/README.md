# Data — all synthetic, deterministic

- `synthetic_ehr.csv` — V1 baseline (n=3000, 2–8 visits, seed 42)
- `taskV2_A_10000.csv` / `taskV2_A_20000.csv` — Ordered pattern (seed 10100/20100, 3–12 visits)
- `taskV2_B_10000.csv` / `taskV2_B_20000.csv` — Cumulative mean (seed 10200/20200)
- `taskV2_C_10000.csv` / `taskV2_C_20000.csv` — Long-range first×last (seed 10300/20300)

All generated with same `src/*.py` logic, same seeds as plots. Preview JSONs in `docs/data_taskV2_*_preview.json` are 100-row slices. Full CSVs are reproducible via `python src/08_taskV2_A.py` etc.
