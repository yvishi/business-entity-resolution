# Task for a Claude Code web session: train the entity-resolution pipeline and return a submission

Read `HANDOFF.md` for full background; this file is the part you act on.

## Goal
Produce a new `matching_results.tsv` for the Amazon ML Challenge 2026 business entity-resolution task and
report its validation score. The current leaderboard best is **0.97** (the "v2" code). The metric is macro
F0.5 per Source-1 entity (precision-weighted; an entity with no true match scores 1 only for an empty prediction).

## Rules you must not break (disqualification risk)
- Never fit anything on test data. Vectorizers, dictionaries and models are fit on `train/` only (already
  enforced in the code; do not change that).
- No external data, APIs, geocoding or internet lookups for business/address information. (Downloading the
  contest's own dataset from the link the user gives you is fine.)
- Final model must be MIT/Apache-2.0 licensed and at most 8B parameters (LightGBM is fine).
- Do not hard-code the country list: test contains France, which is absent from train.
- Never commit the dataset or `work/` to git (the repo is public).

## Steps
1. The dataset is **not** in the repo. The user gives you a link to the contest zip. From the repo root run:
   ```
   python code/business_entity_resolution/src/cloud_run.py --data-url "<link from the user>"
   ```
   It installs dependencies, downloads and unpacks the data into `train/` and `test/` (Google Drive links are
   handled), chooses a profile from the machine's RAM (`fast` <28 GB, `final` <100 GB, `full` otherwise),
   runs the whole pipeline, validates the output and writes `results/<timestamp>-<profile>/`.
2. It is **resumable**: if the session is interrupted or the command times out, rerun the exact same command
   and it continues from the last finished step (markers in `work/.done_*`, per-partition checkpoints; the
   data is not downloaded again). Run it in the background and poll `work/run.log` if your shell has a time limit.
3. If it fails, read the end of `work/run.log`, fix the cause, and rerun. Common causes: download blocked
   (the Drive file must be shared as "Anyone with the link"), out of memory (rerun with `--profile fast`,
   and/or set `BER_BLOCK=v2` for a smaller candidate set), or low disk space (needs ~25+ GB).
4. When it finishes, commit and push **only** `results/<timestamp>-<profile>/` (gzipped submission,
   `summary.json`, `train_report.json`, log tail). `train/`, `test/`, `data_in/` and `work/` are gitignored;
   keep them out.

## What to report back
- The profile used, runtime, and the `summary.json` numbers: OOF F0.5 for stage 1, stage 2 and stage 2 +
  calibration, overall and per country, plus whether the validator passed.
- OOF covers US and India only (train has no French labels). For reference, earlier runs scored OOF
  ~0.974-0.9755 and the 0.97 leaderboard file came from them, so **an OOF clearly above ~0.976 is the
  signal worth submitting**. Say plainly if it isn't higher.
- Do not claim a leaderboard score; only the team can submit.

## Optional, only if time remains after a successful run
Error analysis on the OOF: `WORK=work RAW=work/raw python eda/11_error_analysis.py` splits the remaining loss
into blocking misses, model misses and false merges. Report the biggest bucket; don't change the model
without re-validating.
