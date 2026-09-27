# Business Entity Resolution — Amazon ML Challenge 2026

Pipeline: candidate blocking (char-3gram TF-IDF, forward/reverse/address search) → ~55 pair features →
two-stage LightGBM → per-entity expected-F0.5 set selection. The dataset is **not** in this repo.

**Run it** (from the repo root; needs ~16+ GB RAM, ~25 GB disk):
```
python code/business_entity_resolution/src/cloud_run.py --data-url "<link to the contest dataset zip>"
```
or put the zip in `data_in/`, or the TSVs in `train/` and `test/`. Output goes to `results/`.

- `WEB_TASK.md` — instructions for an unattended (Claude Code web) run
- `HANDOFF.md` — full history: what was built, what scored what, diagnosis, next steps
- `code/business_entity_resolution/` — the pipeline (`src/run_all.py`, package `src/ber/`, notebook, requirements)
- `docs/` — research notes, EDA report, implementation plan · `eda/` — EDA scripts and outputs
- `tools/validate_submission.py` — the official format validator
