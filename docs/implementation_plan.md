# Final implementation plan — Business Entity Resolution Challenge

Single source of truth for how we solve the challenge, from raw TSVs to `matching_results.tsv`.
Built on `docs/research.md` (literature + past competitions) and `docs/eda_report.md` (measured facts).
Constraints: **RTX 5050 (8 GB VRAM), 16 GB RAM, 10 submissions, test data never used for training**,
final model MIT/Apache-2.0 and ≤ 8B parameters.

Contents
1. What kind of problem this is, and what winning teams do
2. Overall strategy — how we make sure we end up with a high-accuracy model
3. Preprocessing
4. Every algorithm family, per pipeline stage, and when we try it
5. Validation, error analysis and the iteration loop
6. France (the unseen country)
7. Roadmap with go/no-go gates
8. Submission plan (10)
9. Compute, memory and code layout
10. Research phase (done) and the continuing research backlog
11. Risks

---

## 1. What kind of problem this is, and what winning teams do

### 1.1 The problem, named
| Framing | What it gives us | Evidence from our data |
|---|---|---|
| **Multi-source, clean–dirty entity resolution** | Blocking → matching → clustering toolbox (Christen; Papadakis) | S1 deduplicated; S2/S3 contain duplicates of each other and within themselves |
| **Entity linking** (mention → knowledge-base entry or NIL) | Retrieve-then-rerank (BLINK: bi-encoder + cross-encoder); each S2/S3 record links to ≤ 1 S1 | 0 violations of "≤ 1 S1 per S2/S3 record" in 7.6M links; plain TF-IDF puts the true S1 at rank 1 for **95.3% (US) / 84.9% (India)** of targets |
| **Two-stage retrieval / learning to rank** | Recall-oriented candidate generation, precision-oriented reranker, LambdaMART | |
| **Pairwise classification / similarity learning** | GBDT on similarity features; Siamese/contrastive encoders; cross-encoders | Matched vs random name similarity: median 0.86 vs 0.25 |
| **Bipartite / constrained clustering** | Unique Mapping Clustering, assignment, correlation clustering, GNN node classification | One-sided uniqueness; S2↔S3 copies of the same entity |
| **Set prediction under macro F-measure** | Bayes-optimal plug-in rule for expected F (Dembczyński et al. 2011/2013) | Metric is macro F0.5 per S1, singletons score 1 for an empty prediction |
| **Inverting a synthetic corruption process** (Febrl/GeCo-style generator) | Learn each corruption operator → targeted normalisation, features, augmentation | ~40 operators measured with frequencies (EDA §2); cluster stats identical across countries |

### 1.2 What the winners of comparable competitions did
| Competition | Winning pattern | What we take |
|---|---|---|
| Foursquare Location Matching (2022), 1st place | text + geo candidate generation → **LightGBM filter with few features** → top-40 → 2nd LightGBM with rich features (Levenshtein, Jaro-Winkler, ratios, SVD of embeddings) → 2 transformer pair classifiers → **GNN post-processing**; 0.907 → 0.946 | cascade: cheap pre-ranker → rich GBDT → neural → graph/decision stage |
| Foursquare, 4th place | two-level boosting (pre-filter GBDT, final GBDT), CatBoost+LightGBM+XGBoost; curated features beat "all features" | feature selection matters; GBDT ensembles |
| Foursquare, medal solution | FAISS kNN on multilingual embeddings (XLM-R, LaBSE, mpnet) fine-tuned with ArcFace + spatial kNN; XGBoost on 60+ features incl. **rank features** | dense blocker as a complement; rank features |
| Shopee Price Match (2021) | embedding kNN, per-query thresholds, neighbourhood blending (query expansion) | adaptive per-query decisions; neighbour-aware scores |
| Quora Question Pairs (2017) | pairwise GBDT; **graph features** (shared neighbours, k-core) were the largest gain | S2↔S3 support, competition between S1s for one target |
| US Patent Phrase Matching (2022) | DeBERTa-v3 cross-encoder ensembles | cross-encoder as a precision add-on |

### 1.3 What the literature says per stage
- **Blocking**: top-k TF-IDF/BM25 (Sparkly, PVLDB 2023) beat 8 SOTA blockers incl. deep ones; char-3-grams ≈ 5-grams;
  top-k > thresholding. DeepBlocker: bag-of-token models are robust to attribute shuffling (our addresses are rotated in 50–67% of pairs).
- **Matching**: GBDT on similarity features is the strong baseline (Magellan); PLM cross-encoders (Ditto) win on dirty text,
  +9.8% from domain-knowledge tags and augmentation; fine-tuned LLMs generalise well but are slow.
- **Decisions**: Unique Mapping Clustering is the standard clean-clean choice (Papadakis et al., EDBT 2022);
  plug-in expected-F maximisation is consistent, structured-loss direct optimisation is not.

### 1.4 The common recipe (our guiding principles)
1. A validation set that mirrors the test and the exact metric — decisions come from it, not from the leaderboard.
2. Blocking recall is the ceiling → union of complementary blockers, measured per country.
3. GBDT on rich similarity features is the workhorse; neural models add diversity, not the base.
4. A second stage that sees *context* (ranks, competitors, neighbours) gives large late gains.
5. Post-processing tuned to the exact metric.
6. Error analysis every iteration, attacking the largest loss bucket.
7. Ensembles of diverse, individually validated models.

---

## 2. Overall strategy — how we make sure we end up with a high-accuracy model

**2.1 The pipeline shape (decided):**

```
prepare ─► normalise (invert each corruption operator) ─► parse address (state/city/number/street)
   │
block (per country, per canonical state):
   forward  S1 → pool top-k   (TF-IDF char-3g on name | name+address | address; word-level on address)
   reverse  pool → S1 top-k   (entity-linking direction; strongest single signal)
   keys     house-no+street, ZIP/PIN, exact core name, glued-name (domains/handles)
   fallback cross-state name-only search for empty/stateless addresses
   union ─► PRE-RANKER (LightGBM, ~15 cheap features) ─► top-K per S1  = candidate_pairs.tsv
   │
match:
   stage 1  LightGBM on ~80 pair features            (5-fold OOF)
   stage 2  LightGBM + context features from stage-1 OOF (ranks, competition, S2↔S3 support)
   add-ons  cross-encoder on uncertain pairs; bi-encoder cosine  (only if OOF gain)
   │
decide:
   calibrate → one-sided unique mapping (each S2/S3 → ≤ 1 S1) → per-S1 expected-F0.5 set selection
   │
matching_results.tsv + validator (--check-ids)
```

**2.2 How we guarantee we find the best model (the method, not luck):**
1. **Measure before modelling.** Scorer, folds, density view and loss attribution exist before the first model.
2. **Loss attribution.** Every OOF run splits the F0.5 deficit into four buckets, per country and source:
   (a) true link never reached the candidates (blocking), (b) false merge (wrong target predicted),
   (c) missed link (in candidates, not predicted), (d) singleton error (predicted something for an empty truth or nothing
   for a non-empty one). The next experiment always targets the largest bucket.
3. **Data-centric first.** The generator is known in detail (EDA §2). Each operator gets an inverse normalisation or
   a targeted feature before we reach for bigger models — the cheapest, largest gains.
4. **One change per experiment**, logged with git hash, config, and all metrics. Accept only if the gain holds on
   both the normal and the density-matched validation view.
5. **Cheap loop, expensive confirmation.** Iterate on a 20% S1 development subset (minutes), confirm on full 5-fold
   OOF (hours), submit only confirmed gains.
6. **Model search is breadth-then-depth.** For each stage, try the algorithm families of §4 at low cost first
   (defaults, dev subset), then tune only the winners (Optuna on LightGBM; learning rate/epochs on neural).
7. **Ensemble at the end**, from models that each earned their place on OOF.

**2.3 Metric intuition that drives decisions (F0.5, macro per S1):**
- With 4 true matches: predicting 3 correct and none wrong scores 0.94; predicting all 4 plus one wrong scores 0.83.
  **One false merge costs about as much as missing two of four true links** → decisions lean to precision.
- An S1 with no true match scores 1.0 only for an empty prediction (5.6% of train S1).
- So the decision stage is not a global threshold: it is a per-S1 choice of the set with the highest expected F0.5.

---

## 3. Preprocessing

### 3.1 Storage and hygiene
- TSV → Parquet per split/source (done: `eda/00_to_parquet.py`), int32 ids, per-country partitions.
- **Collapse exact-duplicate S2/S3 rows** (25k groups in S2, 18k in S3; every group belongs to one entity in train);
  expand at output time.
- All vocabularies/IDF/dictionaries fitted on **train only** (test rule).

### 3.2 Normalisation views (each inverts measured operators)
| Operator (EDA frequency) | Inverse / view |
|---|---|
| case (upper 3–24%, lower 5–8%) | lowercase |
| accents added (5–7%; France 24%) | accent-folded copy (keep original too — French accents are real) |
| junk prefixes `>> *** -- ... # @ [ (` (2–3%) | strip leading non-alphanumerics |
| brackets `[Ltd]`, `(Brc)` (3% each) | remove bracket characters, keep content |
| duplicated tokens (`LLC LLC`) | collapse consecutive duplicates |
| leetspeak (2.5–3.4%) | digit→letter inside alphabetic tokens (separate view; digits can be real) |
| domains/handles (4–6%) | strip TLD/`@`; **glued view** (all spaces removed) of every name so `chiropracticstarpartners.com` meets `chiropractic star partners` |
| legal form dropped/added/swapped (India drop 45–52%) | legal form extracted to its own field (canonical map US/India/France); **core name** without it and without generic words |
| generic words added (16–20%) | generic-word list learned from train (words added/removed in true pairs) |
| word shuffle (6–7%) | token-sorted view |
| **Indic scripts** (names 13–23% of India pairs, 7+ scripts; addresses 22–23%) | **word-level transliteration dictionary aligned from train pairs** (native word ↔ Latin word, count-weighted) + rule-based fallback (ISO/ITRANS via `indic-transliteration`/`aksharamukha`); IndicXlit (MIT) optional if the fallback is weak |
| address part rotation (50–67%) | component parser + sorted-parts view |
| state code ↔ name (S3-US 86%), native-script state, India codes (`KA`, `HYD`, `U P`, `Keralam`) | canonical state tables (US fixed; India learned from train: 98.8% coverage); France region ↔ department (3 regions / 4 departments) |
| house number zero-pad (3–4%), leading digit dropped (2%), one digit changed (1.7%), missing (1–8%), ranges `357-361`, prefixes `B3/`, `NO 414` | canonical house number + fuzzy-number features (never a hard key) |
| street-type abbreviations (US `Dr/Ave/Trl`, vowel-drop `NRTH`; FR `R./Av./All./Imp./Bd/Rte`; IN `Nr./Opp./Rd/Marg`) | expansion tables |
| city typos/variants (`CEHCOTAH`, `Raleigh Township`) | fuzzy city similarity, not keys |
| fillers `N/A`, `PMB 6178`, `PO Box`, `OLD NO.##3` | filler removal / separate field |
| empty address (3.6–4.9%) | explicit flag; name-only path |

### 3.3 Parsed fields per record
`name_raw, name_norm, name_core, name_glued, name_sorted, legal_form, name_translit, addr_norm, addr_sorted,
house_no, street, unit, city, state_canon, postcode, fillers, is_addr_empty, script_flags`.
Unit tests: every catalogue example in `eda/outputs/01_noise_catalogue.txt` must normalise to the same core.

---

## 4. Every algorithm family, per pipeline stage, and when we try it

Legend — **Core**: in the first pipeline. **Try**: evaluated once the core works; kept only if OOF improves.
**Stretch**: only if time remains. **Skip**: considered, with the reason.

### 4.1 Candidate generation (blocking) — target ≥ 99% (US) / ≥ 98% (India) pair recall at ≤ ~60 raw candidates per S1
| Algorithm | Status | Notes |
|---|---|---|
| Canonical-state partition | **Core** | US 100%, India 98.8% invariant; cuts cost 20–50× |
| TF-IDF char-3-gram top-k, forward (name, name+addr, addr) | **Core** | measured: US 99.3% / India 95.6% with reverse |
| TF-IDF top-k, reverse (target → S1) | **Core** | top-1 alone 95.3% / 84.9% |
| Word-level TF-IDF / BM25 top-k (Sparkly) | **Core** | cheap postings; strong on Indian addresses (address words survive in 99.5–99.97%) |
| Standard keys (house-no+street prefix, ZIP/PIN, exact core name, glued name) | **Core** | high precision, near-free |
| Cross-state name-only fallback | **Core** | empty/stateless addresses |
| Pre-ranker (LightGBM, ~15 cheap features) → top-K | **Core** | Foursquare-style filter; defines `candidate_pairs.tsv` |
| Transliterated-name blocker (dictionary view) | **Core** (India) | attacks the India recall gap |
| Sorted neighbourhood on core name / street | Try | cheap complement |
| MinHash LSH on shingles | Try | faster alternative to TF-IDF if CPU-bound |
| Phonetic keys (Double Metaphone on core name / transliteration) | Try | typos, transliteration variants |
| Dense ANN: multilingual-e5-small / LaBSE / BGE-M3, then fine-tuned bi-encoder (in-batch + hard negatives), FAISS | Try | only if loss attribution shows blocking misses TF-IDF cannot reach |
| Multi-hop expansion (S1→S2→S3, duplicate clusters) | Try | recovers members of partially found clusters |
| Meta-blocking (edge weighting/pruning, JedAI) | Try | alternative to the pre-ranker |
| Canopy clustering | Skip | superseded by top-k |
| DeepBlocker autoencoder | Skip | Sparkly showed TF-IDF top-k is at least as good |

### 4.2 Pair scoring (matching)
| Algorithm | Status | Notes |
|---|---|---|
| Rules (exact normalised name+address) | **Core** | sanity baseline and high-precision seeds |
| Logistic regression on features | **Core** (baseline) | calibration reference, feature sanity |
| **LightGBM binary** (5-fold OOF) | **Core** | workhorse |
| LightGBM **LambdaRank** grouped by S1 | Try | ranking objective matches "pick the right targets per S1" |
| XGBoost, CatBoost | Try | ensemble diversity (Foursquare 4th place) |
| Random forest / ExtraTrees | Try | quick diversity, rarely best |
| MLP on the same features | Stretch | cheap diversity |
| Fellegi–Sunter with EM (Splink) | Try | unsupervised; France fallback score / feature |
| ZeroER (GMM on similarity vectors) | Stretch | France without labels |
| Char-level Siamese CNN/BiGRU (DeepMatcher-style) | Stretch | typo-robust; GPU-cheap |
| Bi-encoder (multilingual-e5-small, MIT) fine-tuned | Try | cosine feature + dense blocker |
| **Cross-encoder** (multilingual MiniLM-L12 / mDeBERTa-v3-base / XLM-R-base, MIT) Ditto-style serialisation with tagged numbers, augmentation from the operator catalogue | Try | run only on uncertain pairs (stage-1 p ∈ [0.05, 0.95]) — cascade |
| LLM ≤ 8B (Qwen2.5-7B / Qwen3-8B, Apache-2.0), 4-bit, LoRA | Stretch | too slow for the pipeline on 8 GB; use for error-analysis or distillation on a small hard set |
| SVM / kNN classifiers | Skip | dominated by GBDT at this scale |

**Feature set for GBDT (~80, all country-agnostic; no raw `country` feature):**
- Name: equality at each normalisation level; Levenshtein, Damerau, Jaro-Winkler, token sort/set/partial ratios;
  char-3-gram TF-IDF cosine; IDF-weighted token overlap; Monge-Elkan; rarest shared token IDF; core-name similarity;
  legal-form compatibility; acronym match; glued-string similarity; transliterated-name similarity; phonetic match;
  length/word-count differences; **name ambiguity** (how many S1 share the core name — "Primary Care Group" ×253).
- Address: house number exact / zero-pad / suffix-match / one-digit-diff / range-contains; street similarity after
  expansion; city exact/fuzzy; canonical state equality; postcode/PIN equality; unit equality; number-set Jaccard;
  sorted-parts similarity; component-missing flags.
- Pair meta: target source (S2/S3); blocker provenance (which blockers found it); blocker scores and ranks, both directions.
- Stage 2 context (from stage-1 OOF): rank of the target in the S1 list, gap to the best, count above thresholds;
  reciprocal rank (rank of this S1 in the target's list); best competing S1 score for this target; S2↔S3 support
  (max similarity × score to other high-scoring candidates of the same S1); estimated cluster size.

### 4.3 Decisions / clustering
| Algorithm | Status | Notes |
|---|---|---|
| Isotonic calibration of OOF scores | **Core** | needed for expected-F |
| One-sided **Unique Mapping** (each S2/S3 → its best S1 with a margin) | **Core** | enforced by the ground-truth structure |
| **Per-S1 expected-F0.5 set selection** (sort by p; exact expectation under independence via Poisson-binomial DP, K ≤ 25; includes the empty set) | **Core** | Bayes-optimal plug-in rule |
| Global / per-source / per-country thresholds | **Core** (baseline) | comparison point |
| Exact assignment (Hungarian / min-cost flow), Row-Column, Best-Match | Try | rarely beat UMC (Papadakis et al.) |
| Star clustering seeded by S1 + S2↔S3 edges | Try | admits members supported by already-accepted records |
| Correlation clustering / Leiden on the candidate graph | Stretch | |
| GNN node classification on the candidate graph (Foursquare 1st place) | Stretch | only if the stage-2 context model leaves a large false-merge bucket |
| Cluster-size (count) prediction model | Try | alternative to expected-F for choosing k |
| Plain connected components | Skip | chaining → false merges, punished by F0.5 |

### 4.4 Ensembling
Seed/fold bagging of LightGBM (Core) → weighted average of calibrated probabilities across GBDT variants (Try) →
level-2 stacking on OOF with the cross-encoder (Try). A member stays only if full-OOF F0.5 improves ≥ 0.001.

---

## 5. Validation, error analysis and the iteration loop

- **Scorer**: exact macro F0.5 (singleton rules); unit tests incl. the worked example (0.714).
- **Folds**: 5-fold GroupKFold by S1 entity; distractors always in the pool; blocking/features computed once for all
  train (label-free), models trained per fold → OOF for every candidate of every S1 (full-scale test simulation).
- **Views reported for every run**: (1) normal; (2) density-matched — hide 19% of S1 entities so their records become
  distractors, matching test's 2.82/2.93 records per S1; (3) test-mix weighted (India 46.8%, US 38.3%; France estimated
  separately); (4) leave-one-country-out (train US → score India and reverse) as the France proxy.
- **Metrics per run**: blocking pair recall + cluster coverage, candidates/S1, pair PR-AUC, OOF F0.5 (overall / country /
  source / singleton / cluster size), loss attribution buckets, runtime and peak RAM.
- **Loop**: hypothesis → dev-subset experiment (≤ 30 min) → accept/reject → nightly full OOF → submission only for
  confirmed gains. Weekly: review the 50 worst S1 per bucket and add operators/features for recurring patterns.
- **Predicted-distribution checks on test (diagnostic only)**: per country, the predicted singleton rate and matches
  per S1 should resemble train's generator statistics (5.6%, ~3.5) — large gaps flag a broken country (e.g. France).

---

## 6. France (15% of test, no labels)

1. **Country-agnostic design**: similarity features only, no country identity feature; normalisation tables include
   French street types, legal forms (SARL, SAS, SASU, EURL, SA, SCI, EI, Cie, Ets), `(France)` tag, `N°`, `bis/ter`,
   region ↔ department map.
2. **Synthetic labelled French data from external sources only** (allowed: external data; forbidden: test):
   real addresses from the Base Adresse Nationale for départements 33, 44, 59, 62 + generated business names
   (French word lists + legal forms, mirroring the template style) → apply the corruption operators with the
   frequencies measured on train (incl. French-specific: region→department, street-type abbreviation, accents) →
   S1/S2/S3-style clusters with distractors. Uses: validate French normalisation and blocking, estimate French
   F0.5 offline, optionally add to training.
3. **Transfer evidence**: leave-one-country-out results decide how far to trust transfer; prefer features that lose
   least across countries.
4. **Leaderboard A/B**: two submissions identical except for French rows isolate France's contribution.
   Implied French score ≈ (LB − 0.383·F_US − 0.468·F_India) / 0.150 with US/India from OOF.

---

## 7. Roadmap with go/no-go gates (≈ 3 weeks; compress if the deadline is closer)

| Phase | Days | Deliverable | Gate to move on |
|---|---|---|---|
| P0 Infrastructure | 1 | package layout, Parquet, scorer, folds, density view, logger, output writer + validator | scorer unit tests pass |
| P1 Normalisation + parsing | 2–3 | views of §3, transliteration dictionary, state/region tables | catalogue examples normalise correctly |
| P2 Blocking | 3–5 | partitioned forward+reverse TF-IDF, word-level, keys, fallback, pre-ranker | recall ≥ 99% US / ≥ 98% India at ≤ 60 cand/S1; full train blocks in ≤ 1 h |
| P3 Baseline end-to-end | 5–6 | LR + threshold, then LightGBM stage 1 + UMC + expected-F | full OOF F0.5 logged → **Sub 1** |
| P4 Features | 6–9 | full §4.2 feature set, feature selection | OOF gain each step; loss attribution reviewed |
| P5 Stage 2 context | 9–11 | context features, retrain | OOF gain → **Sub 2** |
| P6 France | 9–12 (parallel) | French tables, synthetic French data, LOCO study | **Subs 3–4** (A/B) |
| P7 Neural add-ons | 11–16 | bi-encoder, cross-encoder cascade, stacking | kept only if OOF ≥ +0.002 → **Subs 5–7** |
| P8 Ensemble + freeze | 16–19 | bagging, final blend, clean-room rerun, docs, zip | validator PASS; reproducible → **Sub 8** |
| Reserve | — | fixes | **Subs 9–10** |

---

## 8. Submission plan (10)

| # | Content | Question it answers |
|---|---|---|
| 1 | P3 baseline | LB ↔ OOF offset; implied France score |
| 2 | + stage-2 context | does LB move with OOF? |
| 3 | France variant A | France A/B |
| 4 | France variant B | France A/B |
| 5–7 | confirmed big OOF gains only (features, neural, ensemble) | transfer of gains |
| 8 | best full ensemble | final candidate |
| 9–10 | reserve | only for real fixes |

Every submission: written hypothesis, OOF (per country) next to LB in `experiments.csv`, validator PASS with `--check-ids`.
Final pick = best OOF (test-mix weighted) + best France evidence, not the public-LB rank (public = subset, private decides).

---

## 9. Compute, memory and code layout

**Memory plan (16 GB RAM):**
| Object | Size | Handling |
|---|---|---|
| One country's pool + views | 2–3 GB | per country, freed after use |
| TF-IDF matrix per state | ≪ 1 GB | built per state partition |
| Raw candidates (~60/S1) | ~130M pairs train → ~1.5 GB as int32/float32 | Parquet chunks |
| Pre-ranker features (~15) | computed and scored per chunk, only top-K kept | |
| Full features (~80) on top-K (~20/S1 → ~45M pairs) | too big at once | Parquet chunks; LightGBM from `lgb.Sequence` or ≤ 15M-row subsample (all positives + hardest negatives) |
| LightGBM | binned uint8, `max_bin=127`, `force_col_wise` | |

**GPU (8 GB)**: PyTorch ≥ 2.7 with CUDA 12.8 wheels (Blackwell). fp16, seq len 64. Estimated: e5-small embedding of
all records ≈ 1–1.5 h; MiniLM cross-encoder ≈ 3–5k pairs/s → cascade to ≤ 5M uncertain pairs ≈ 20–30 min.

**Code layout (matches the required package):**
```
code/business_entity_resolution/
  README.md  requirements.txt  configs/base.yaml
  src/ber/
    io.py  normalize.py  address.py  translit.py
    blocking/{tfidf_topk.py, keys.py, dense.py, prerank.py}
    features/{name.py, address.py, context.py}
    models/{gbdt.py, cross_encoder.py, bi_encoder.py}
    decide.py  metrics.py  cv.py  pipeline.py      # python -m ber.pipeline {prepare|block|features|train|predict|write}
```
Output writer produces both `matching_results.tsv` and `candidate_pairs.tsv` (matches ⊆ candidates by construction).

---

## 10. Research phase (done) and continuing research backlog

**Done** (`docs/research.md`): problem framings; Foursquare/Shopee/Quora/USPPPM solutions; Sparkly, DeepBlocker,
Magellan/DeepMatcher, Ditto, LLM-for-EM, Fellegi–Sunter/ZeroER, bipartite matching (Papadakis), F-measure
optimisation (Dembczyński), IndicXlit, multilingual encoders, Blackwell/PyTorch setup.

**Continuing backlog** (time-boxed, triggered by loss attribution rather than done up front):
| Trigger | Question | Where to look |
|---|---|---|
| India blocking misses dominate | best transliteration for business words; Aksharantar data; phonetic matching for Indic names | IndicXlit/Aksharantar papers; indic-transliteration docs |
| False merges dominate | graph/context post-processing that fixes them | Foursquare 1st-place GNN write-up; FAMER multi-source clustering; correlation clustering for ER |
| Missed links dominate | better pair models on dirty text | Ditto augmentation ops; contrastive EM (Peeters & Bizer); Unicorn |
| France weak on LB | cross-lingual transfer for EM; synthetic-data generation | GeCo/Febrl corruption models; AnyMatch/zero-shot EM |
| Runtime/RAM limits | faster top-k search | Sparkly/Lucene BM25, MinHash LSH, FAISS IVF-PQ |

---

## 11. Risks

| Risk | Mitigation |
|---|---|
| France behaves differently | country-agnostic features, synthetic French data, LOCO, France A/B subs |
| More distractors in test (+24% records per S1) | density-matched validation; precision-leaning decisions |
| India recall ceiling (95.6% now) | transliteration dictionary, word-level address blocker, canonical state, fallback |
| False merges on generic names | name-ambiguity features, address agreement required when the name is common |
| RAM blow-up | per-country/state partitions, chunked Parquet, pre-ranker before heavy features |
| Overfitting public LB | 10 subs only, decisions from OOF |
| Audit / reproducibility | pinned env, seeds, one entry point, licences listed (all MIT/Apache/BSD), no test data in training |
