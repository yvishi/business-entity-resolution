# Business Entity Resolution — pipeline

```
pip install -r requirements.txt
cd src
python run_all.py --train-dir <dataset/train> --test-dir <dataset/test> --work work --out output
```

Steps (all by default, in order; resume with `--steps block_test,predict` etc. — every step writes to `work/`):
`parquet` → `prepare` (normalise, learn India state forms + transliteration from train) → `block_train` →
`train` (5-fold OOF LightGBM stage 1 + stage 2, decision tuning, **validation report**) → `block_test` → `predict`
(writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`).

Validation results: printed in the log (`STAGE1 …`, `STAGE2 …` lines with OOF macro F0.5 overall and per country,
and `blocking pair recall`) and saved to `work/train_report.json`.

Recommended (one command, resumable, ~5-7 h on an 8-core / 16 GB laptop):
```
python -W ignore run_all.py --final --train-dir <train> --test-dir <test> --work <work> --out <out>
```
`--final`: 30% of (country, state) partitions for training (capped to RAM automatically by dropping whole
partitions), 3 folds, lr 0.05, early stopping. Before the long steps an automatic **preflight** runs every step on
a small sample of the real data (~3-5 min) and validates the output format. Every step writes `<work>/.done_<step>`;
rerunning the same command resumes (blocking and prediction also resume per partition). Needs ~30 GB free disk.
`--steps decide --decision thr,0.8,uniq` rewrites the submission from saved test scores (~2 min).
Validate: `python3 utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test`
