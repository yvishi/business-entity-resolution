# Handoff: Business Entity Resolution (ML Challenge 2026)

Everything done so far, where it lives, what scored what, and what to do next.
Branch: `claude/stoic-gauss-0h2tmn` of `github.com/ParasKaushal123/data-EDA`.

> **Code versions on this branch**
> - **v1 = commit `bd6659e`** — the pipeline that produced the **0.956** leaderboard submission (validated end to end).
> - **v3 = the latest commit** — v2 ideas (§6) plus the fixes in §6b; run with `run_all.py --final` (§8).
>   The preflight (every step on real-data samples) passes; a full-scale run had not been done when this was written.
>   To reproduce v1 exactly: `git checkout bd6659e`.

---

## 1. The challenge in one paragraph
Three sources of business records (`entity_id`, `business_name`, `business_address`, `country`). Source 1 (S1) is
deduplicated; for every S1 entity, output all matching S2/S3 records (`matching_results.tsv`, one row per S1, comma-
separated IDs, empty allowed). Metric: **macro F0.5 per S1 entity** (precision-heavy; an S1 with no true match scores 1
only for an empty prediction). Train = US + India with labels; **test adds France (15%), unseen in train**. Rules:
test data must not be used for training; final model MIT/Apache, ≤ 8B params. 10 leaderboard submissions
(**1 used: 0.956**). Final package also needs `candidate_pairs.tsv`, code, README, methodology doc.

## 2. Repository map
| Path | What |
|---|---|
| `docs/research.md` | Literature + past-competition research (Foursquare, Shopee, Quora; Sparkly, Ditto, F-measure optimisation…) |
| `docs/eda_report.md` | EDA findings (noise operators, invariants, blocking recall, France profile) |
| `docs/implementation_plan.md` | Original strategy / algorithm menu / validation / submission plan |
| `eda/*.py`, `eda/outputs/` | All EDA scripts and their raw outputs (numbered in the order they were run) |
| `code/business_entity_resolution/` | **The pipeline** (`src/run_all.py` + package `src/ber/`), `README.md`, `requirements.txt` |
| `HANDOFF.md` | This file |

Pipeline modules (`src/ber/`): `io_raw.py` (TSV→Parquet) · `text.py` (normalisation, state/region parsing,
transliteration) · `prepare.py` · `block.py` (candidate generation + prune) · `features.py` · `train.py`
(OOF training + decision tuning) · `decide.py` (metric, unique mapping, set selection) · `predict.py`.

## 3. Key data facts (from EDA — details in `docs/eda_report.md`)
- Sizes: train S1/S2/S3 = 2.21M / 5.03M / 5.29M; test = 1.73M / 4.89M / 5.08M. Test mix: India 46.8%, US 38.3%, France 15.0%.
- Every S2/S3 record links to **at most one** S1 (0 violations in 7.6M links) → one-sided unique mapping is safe.
- Cluster stats are identical per country (singletons 5.6%, mean 3.46 matches/S1) → one generator config per country.
- **Test has ~24% more S2/S3 records per S1 than train** (2.82/2.93 vs 2.28/2.40, same in all test countries) →
  probably more distractors; train-tuned thresholds are slightly too lenient for test.
- No leaks (row order / ID numbers uncorrelated with matches).
- Noise operators (share of true pairs): legal form removed (India up to 52%), Indic-script names (India 13–23%,
  7+ scripts), address parts reordered (50–67%), state spelled out (S3-US 86%), uppercase (S2-US addresses 88%),
  house-number corruption (zero-pad 3–4%, leading digit dropped 2%, one digit changed 1.7%), typos, accents, domains,
  junk prefixes, trade-name replacement (~2–4%, name shares nothing).
- Invariants: **US state 100%**, **Indian state 98.8% after canonicalisation** (native script / codes learned from train)
  → safe partition key for blocking. France: 3 regions; S2/S3 often use the département instead of the region.
- France test copies keep the S1 house number **86%** of the time (more than US 77%, India 60%) — the
  "multi-establishment" hypothesis was checked and **rejected** (`eda/13_*`). But **25.7% of French S1 names are shared
  by another S1 in the same region** (US 9%, India 25%) → address must disambiguate.

## 4. Pipeline v1 (produced the 0.956)
```
TSV → Parquet → normalise (+ India state surface forms & transliteration dictionary learned from TRAIN;
                            France region from region / département / city)
→ block per (country, canonical state/region):
    char-3gram TF-IDF (fitted on train), forward S1→pool top-20 (name|address) + top-10 (name),
    reverse pool→S1 top-5 + top-3, name-only fallback (top-5) for stateless pool records
→ prune (r_fna<10 | r_rna<3 | r_fnm<5 | r_rnm<2 | fallback)  = candidate_pairs.tsv
→ ~45 pair features (fuzzy name/address, TF-IDF cosines, ranks, house number, legal form, state, ambiguity…)
→ LightGBM stage 1 (OOF) → stage 2 with context features (ranks/gaps/competition per S1 and per target)
→ fold-ensemble prediction → one-sided unique mapping → per-S1 expected-F0.5 set selection
```
**Results**
| | |
|---|---|
| Dev set | 10% of (country, state) partitions: 361,279 S1, 10.65M candidate pairs |
| Blocking pair recall (dev) | 98.08% |
| OOF F0.5 stage 1 | 0.9728 (US 0.9767, India 0.9683) |
| OOF F0.5 stage 2 (used) | **0.9740** (US 0.9778, India 0.9697); decision `expf, 0.3, uniq` |
| Training config | `--fast`: 3 folds, 300 rounds, lr 0.12 — **stage 1 hit the round cap (underfit)** |
| Test output | 1,732,544 rows, 97,004 empty (5.6%), validator PASS |
| **Leaderboard** | **0.956** |

Laptop timings (Ryzen 7 250, 16 GB, v1): prepare ~6 min · block_train ~10 min · train ~15 min ·
**block_test ~85 min** · predict ~25 min.

## 5. Why 0.956 and not 0.974 — diagnosis after the leaderboard
1. **France is the main suspect.** LB ≈ 0.383·US + 0.468·India + 0.150·France. With US/India at their local values,
   France ≈ **0.85**; if US/India lose 0.5–1 pt at test density, France ≈ 0.88–0.91. France can't be measured locally
   (no labels) — only via leaderboard A/B (two submissions that differ only in French rows).
2. **Test density** (+24% records per S1) → more false merges than on train; v1 chose its threshold at train density.
3. **Blocking misses** (`eda/14_blocking_misses.py`, dev partitions):
   - US: 99.08% raw → 98.98% after prune. **75% of US misses = target has an empty address** (name-only fallback
     keeps only 5 of all US S1 — generic names crowd out the right one); trade names with the same address
     (`CALOBRIXZETA` for `Owens & Cannon Inc.`) — **v1 had no address-only search**.
   - India: 97.68% raw → **97.04% after prune (prune alone loses 0.64%)**; misses: address parts dropped 59%, legal form
     removed 50%, empty address 34%, Indic-script names 20%.
   - Prune alternatives (recall / candidates per S1): current India 97.04% / 29.7 → **(15,5,8,3): 97.60% / 47.2**;
     US 98.98% / 29.3 → 99.07% / 46.3.
4. **Model underfit** — best iteration = round cap in every fold (300 in v1; a local run with 600 also hit the cap).
5. Test audit (`eda/10_test_audit.py`): US/India test ≈ train (state parse, OOV tokens, scripts, patterns). France-only
   issues were found and fixed before the v1 run (region partitioning, street abbreviations `R.`/`Av.`/`All.`, `N°`,
   French stop words/legal forms, `St-` → `Saint-` city keys).

## 6. v2 changes (latest commit — written, smoke-tested only)
| File | Change | Targets |
|---|---|---|
| `text.py` | state/region part removed from the address text (state is its own feature; "Ohio" vs "OH", "தமிழ்நாடு" vs "Tamil Nadu" was noise); US/India street abbreviations expanded (Rd→road, Ave→avenue, Blvd, Dr, Ln, Ct, St, Nr→near, Opp→opposite…); consonant skeleton `skel` of the core name | address similarity, typos/transliteration |
| `block.py` | **address-only TF-IDF search** (fwd top-8, rev top-3); France search text strips legal forms / stop words / street types (these are rare in the train-fitted vocabulary so they dominated cosine); name fallback 5→15; **looser prune** `(15,5,8,3)` + address ranks; word-level TF-IDF `cw` on core names (fitted on train) | blocking misses (trade names, empty addresses, India prune loss), France calibration |
| `features.py` | new: `c_ad`, `r_fad`, `r_rad`, `skel_ratio`, `core_wcos` (IDF-weighted word overlap), `hn_ham1` (house number one digit off); **stage-2 support features** `sup_*` (does this candidate resemble the other high-scoring candidates of the same S1? distractors — copies of a different, absent business — don't) | false merges, distractors |
| `train.py` | decision rule chosen on a **test-like density view** (19% of S1 hidden → their records become distractors); prints normal/dense scores of the top rules | test density |
| `predict.py` | support features in stage 2 | — |

v2 needs a **full rerun** (normalisation changed → prepare, block, train, block_test, predict). Expect block steps
~50% slower (third search field) and more candidates (looser prune).

## 6b. v3 changes (latest commit)
| Where | Change | Why |
|---|---|---|
| `block.py` | **dev-world fix**: when training on a subset of partitions, stateless (empty-address) records are re-sampled — keep those owned by a dev S1, keep unlinked ones at the dev-S1 ratio, drop records whose owner is outside the dev set | v2 kept *all* stateless records of the country → the fallback produced masses of "phantom" negatives (1.7M vs 4.4M pairs on the 4-state A/B) that never exist at test time |
| `features.py` | house-number relation features (`hn_absdiff`, `hn_reldiff`, first/last digit equal, length diff) | 26% of the loss = distractors with a neighbouring house number |
| `predict.py` | name ambiguity counted over the whole country (was per partition — mismatch with training since v1); 3 passes: stage-1 per partition → context/support features country-wide (exactly as in training) → stage-2 per partition; checkpoints per partition | train/test feature mismatch; memory; resume |
| `train.py` | features written to a disk-backed memmap, LightGBM binned dataset + fold subsets, stage-1 checkpoint, pair budget from RAM (drops whole partitions, so density is kept) | 16 GB RAM safety |
| `run_all.py` | `--final`, automatic preflight, `.done_<step>` markers + resume, disk/RAM checks, log to `<work>/run.log` | no lost hours |

## 7. Next steps (in order)
1. **Validate v2 locally** (same dev partitions as v1, so numbers are comparable):
   ```
   python -W ignore run_all.py --fast --train-dir <train> --test-dir <test> --work <new_work_dir> --out <out> --steps parquet,prepare,block_train,train
   ```
   Compare with v1: blocking recall 98.08% and STAGE2 OOF 0.9740. Keep only if better; if a single change hurts,
   revert it (each is independent).
2. **More boosting rounds** (underfit): `$env:BER_ROUNDS=1500; $env:BER_LR=0.1` before the run (early stopping is on).
   Consider `num_leaves` 255 in `train.PARAMS`.
3. **Error analysis on the OOF** — already scripted:
   ```
   $env:WORK="<work_dir>"; $env:RAW="<work_dir>\raw"; python eda\11_error_analysis.py
   ```
   Splits the F0.5 deficit into blocking misses / model misses / false merges (distractor vs other S1's record),
   profiles each bucket by noise operator, prints examples, and re-scores decision rules at test density.
   (Works on v1's `work` folder too — it has `oof.parquet`.)
4. **Submit v2**, then use **France A/B**: submit two files that differ only in French rows (`--steps decide`
   rewrites the submission from saved scores in ~2 min) to measure France directly.
5. Further ideas, by expected value: more training data (`--dev-frac 0.2`, needs RAM); better Indic transliteration
   (IndicXlit, MIT); synthetic French validation data from external French addresses (BAN) + the measured noise
   operators (test data may not be used for training); a cross-encoder (multilingual MiniLM/mDeBERTa, MIT) on uncertain
   pairs only; pool↔pool nearest-neighbour features for distractor clusters.

## 8. How to run (Windows / PowerShell)
```powershell
$env:GIT_LFS_SKIP_SMUDGE=1
git clone -b claude/stoic-gauss-0h2tmn https://github.com/ParasKaushal123/data-EDA.git
cd data-EDA\code\business_entity_resolution
python -m venv venv; .\venv\Scripts\Activate.ps1; pip install -r requirements.txt
cd src
$env:PYTHONUTF8=1                       # Windows: Indic text breaks the default cp1252 encoding
python -W ignore run_all.py --final --train-dir <...\train> --test-dir <...\test> --work <work_dir> --out <out_dir>
```
- The run resumes if restarted with the same command (`<work_dir>\.done_<step>` markers). The log is `<work_dir>\run.log`.
- Needs ~30 GB free disk on the work drive; close browsers (16 GB RAM).
- `--steps decide --decision thr,0.8,uniq` rewrites the submission from saved test scores (~2 min).
- Validation: `python utils\validate_submission.py --matching <out>\matching_results.tsv --candidate <out>\candidate_pairs.tsv --test-dir <test> --check-ids`
- Gotchas: `git pull` failing on `.git\index.lock` → delete it; a half-failed pull → `git reset --hard origin/claude/stoic-gauss-0h2tmn`.
