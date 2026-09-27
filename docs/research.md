# Research notes: Business Entity Resolution Challenge

Purpose: what the literature and past competitions say about this *kind* of problem, and the concrete
decisions it leads to for our pipeline. (Kaggle, arXiv and VLDB pages were blocked from the research
environment, so some items are from search summaries plus prior knowledge of the papers; the
decisions below do not hinge on any single unverifiable detail.)

## 1. What problem this is

| Name in the literature | Why it applies here |
|---|---|
| **Entity resolution / record linkage**, multi-source, **clean–dirty** | Source 1 is deduplicated ("clean"); Sources 2 and 3 contain duplicates of each other and within themselves ("dirty"). |
| **Entity linking** (mention → knowledge-base entry or NIL) | Source 1 acts as the knowledge base. In train, every Source 2/3 record links to **at most one** Source 1 entity (0 violations in 7.6M links), so each S2/S3 record is a "mention" to link to one S1 entity or to nothing. BLINK-style *bi-encoder retrieval + cross-encoder rerank* is the standard architecture. |
| **Two-stage retrieval / ranking** (search, recommender systems) | Candidate generation for recall, then a reranker for precision. |
| **Bipartite / clean-clean matching** | Papadakis et al. (EDBT 2022) evaluated 8 bipartite matching algorithms; *Unique Mapping Clustering* (sort all edges by score, accept an edge if neither side is already matched, stop at a threshold) is the most popular, cheap and strong. Our variant: one-sided (S2/S3 side unique, S1 side many). |
| **Set prediction under an F-measure** | Macro F0.5 per S1 entity. Dembczyński et al. (NeurIPS 2011; ICML 2013) give the Bayes-optimal "plug-in" rule: with calibrated per-candidate probabilities, pick the prediction set that maximises *expected* F. The plug-in approach is consistent; direct structured-loss optimisation is not. |
| **Inverting a synthetic corruption process** | The data is generated (Febrl/GeCo-style corruptors: typos, abbreviations, reordering, missing values, transliteration). Learning each corruption operator from train pairs gives targeted normalisation and features. |

## 2. Past competitions closest to this one

**Kaggle Foursquare Location Matching (2022)** – the nearest analogue (noisy multilingual place names and addresses).
- 1st place (team re:waiwai), 4 stages: candidate generation from text similarity + geographic proximity →
  **LightGBM filter with a small feature set (memory-efficient) → top-40 candidates per id** into a 2nd LightGBM
  with the full feature set (Levenshtein, Jaro-Winkler, ratios/statistics, SVD of name embeddings) →
  two transformer pair classifiers → **GNN node classification as post-processing**. Score went 0.907 → 0.946.
- 4th place (Theo Viel et al.): two-level boosting — level 1 "pre-filter" GBDT, level 2 final GBDT;
  CatBoost + LightGBM + XGBoost; a curated feature set beat "all features".
- A medal solution (hobbitlab): FAISS kNN on concatenated multilingual embeddings (XLM-R, LaBSE,
  multilingual-mpnet, RemBERT, fine-tuned with ArcFace) + spatial kNN for candidates; XGBoost on 60+ features
  (TF-IDF cosine, Levenshtein, embedding distances, rank features); threshold aggregation post-processing.
- **Takeaways:** union of blockers; a cheap pre-ranker to cut candidates; GBDT on similarity + *rank* features is
  the workhorse; transformers add a few points; graph post-processing is worth a lot.

**Kaggle Shopee Price Match (2021)** – embedding kNN + per-query thresholds; neighbourhood blending
(query expansion: average an item's embedding with its neighbours') improves embeddings; always keep ≥1 match
when the metric punishes empty predictions (here the metric *rewards* correct empty predictions, so the opposite care
applies).

**Kaggle Quora Question Pairs (2017)** – pairwise GBDT, with **graph features** (shared neighbours, neighbour counts,
k-core) as the biggest single gain. For us: S2↔S3 neighbourhood agreement and "how many S1 compete for this record".

**Kaggle US Patent Phrase Matching (2022)** – DeBERTa-v3 cross-encoder ensembles dominate short text-pair similarity.

## 3. Literature by pipeline stage

### Blocking (candidate generation)
- **Sparkly** (Paulsen, Govind, Doan; PVLDB 2023): plain **top-k TF-IDF/BM25 blocking beat 8 state-of-the-art
  blockers**, including deep-learning ones; kNN-cosine on 3-grams ≈ 5-grams; top-k (not thresholded) blocking helps
  recall. q-grams improve effectiveness at a scalability cost (posting lists grow).
- **DeepBlocker** (Thirumuruganathan et al.; PVLDB 2021): self-supervised autoencoder embeddings best on
  structured/dirty data; hybrid best on textual; attribute shuffling ("dirtiness") hurts sequence models but not
  bag-of-tokens models — relevant because our addresses are rotated.
- Papadakis et al. blocking survey (ACM CSUR 2020); Christen indexing survey (TKDE 2012): standard/sorted-neighbourhood/
  q-gram/canopy blocking, **meta-blocking** to prune candidate graphs.

### Matching
- **Magellan / DeepMatcher** (Konda 2016; Mudgal et al., SIGMOD 2018): GBDT/RF on similarity features is a strong
  baseline for structured data; deep models win on dirty/textual data.
- **Ditto** (Li et al., PVLDB 2021): pre-trained transformer cross-encoder on serialised record pairs; up to +29% F1
  over earlier SOTA; domain-knowledge injection (tagging spans such as numbers), TF-IDF summarisation and
  **data augmentation** add up to +9.8%; matches earlier SOTA with half the labels.
- **LLMs for matching** (Peeters, Steiner, Bizer, 2023–24): generative LLMs generalise well zero-shot; fine-tuning
  helps small models a lot and helps in-domain transfer but can hurt cross-domain transfer.
- Unsupervised: Fellegi–Sunter with EM (Splink); ZeroER (SIGMOD 2020) — possible France fallback.

### Clustering / decisions
- Bipartite matching comparison (Papadakis et al. EDBT 2022): UMC, Best-Match, Row-Column assignment,
  Kiraly, exact assignment — UMC is the usual choice; exact assignment rarely worth its cost.
- Multi-source clustering with "clean source" constraints (Saeedi, Peukert, Rahm — FAMER).
- F-measure optimal prediction sets (Dembczyński et al.), and bipartite F-score-optimal clustering (2023).

### Transliteration / multilingual text
- **IndicXlit** (AI4Bharat; MIT; ~11M params; 21 Indic languages, native→Roman supported). Needs fairseq —
  awkward on Windows; a **word dictionary aligned from our own train pairs** is simpler and exact for this generator.
- multilingual-e5-small (MIT, 12 layers, 384-d), BGE-M3 (MIT, ~568M), LaBSE (Apache-2.0) — all allowed licences.

## 4. Hardware notes (RTX 5050 8 GB, 16 GB RAM)
- Blackwell (sm_120) needs **PyTorch ≥ 2.7 built for CUDA 12.8** (`pip install torch --index-url
  https://download.pytorch.org/whl/cu128`). Older wheels detect the GPU but fail.
- 8 GB VRAM: fine for fine-tuning e5-small/MiniLM bi-encoders and mDeBERTa-v3-base cross-encoders (fp16, batch 64–128,
  seq len 64). A 7–8B LLM only fits 4-bit quantised and is slow (≲100 pairs/s) → only for a small hard subset, or skip.
- 16 GB RAM: the binding constraint. Everything must be **per-country, chunked, Parquet-backed, float32 / int32**.

## 5. Decisions taken from the research
1. Two-stage architecture: **union of top-k TF-IDF blockers (both directions) → cheap GBDT pre-ranker → rich GBDT
   (+ optional cross-encoder) → assignment + expected-F0.5 set selection.**
2. Blocking starts with Sparkly-style char-3-gram TF-IDF top-k (measured in `eda/02_blocking_recall.py`), plus the
   reverse (entity-linking) direction, plus cheap keys; dense embeddings only if they add recall.
3. GBDT (LightGBM) with similarity + **rank/context** features is the core model.
4. Post-processing: one-sided unique mapping (each S2/S3 record → ≤1 S1) and **plug-in expected-F0.5** set selection per S1.
5. France: country-agnostic features, multilingual models, synthetic French pairs from *external* French data
   (test data may not be used for training), leave-one-country-out as the offline proxy.

## Sources
- [Foursquare 1st place (re:waiwai) write-up](https://www.kaggle.com/competitions/foursquare-location-matching/writeups/re-waiwai-1st-place-solution) · [Foursquare blog on winners](https://foursquare.com/resources/blog/developer/finding-the-right-poi-match/)
- [TheoViel/kaggle_foursquare (4th place)](https://github.com/TheoViel/kaggle_foursquare) · [hobbitlab Foursquare solution](https://github.com/hobbitlab/Foursquare-Location-Matching/blob/main/readme.md)
- [Shopee 1st place write-up](https://www.kaggle.com/competitions/shopee-product-matching/writeups/upstage-making-ai-beneficial-1st-place-solution-fr)
- [Quora winning solution (PDF)](https://etu.univ-lyon1.fr/medias/fichier/quora-winning-solution_1499946713118-pdf?ID_FICHE=74297&INLINE=FALSE)
- [Sparkly (PVLDB 2023)](https://www.vldb.org/pvldb/vol16/p1507-paulsen.pdf) · [sparkly code](https://github.com/anhaidgroup/sparkly)
- [DeepBlocker (PVLDB 2021)](https://vldb.org/pvldb/vol14/p2459-thirumuruganathan.pdf)
- [Ditto (arXiv 2004.00584)](https://arxiv.org/abs/2004.00584)
- [Entity Matching using LLMs (Peeters et al.)](https://arxiv.org/html/2310.11244v4) · [Fine-tuning LLMs for EM](https://arxiv.org/abs/2409.08185)
- [Bipartite Graph Matching Algorithms for Clean-Clean ER (Papadakis et al.)](https://arxiv.org/abs/2112.14030) · [Optimal F-score clustering for bipartite record linkage](https://arxiv.org/pdf/2311.13923)
- [An Exact Algorithm for F-Measure Maximization (NeurIPS 2011)](https://papers.nips.cc/paper/4389-an-exact-algorithm-for-f-measure-maximization) · [Plug-in rule vs structured loss (ICML 2013)](https://proceedings.mlr.press/v28/dembczynski13.html)
- [IndicXlit](https://github.com/AI4Bharat/IndicXlit) · [multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small)
- [PyTorch sm_120 / RTX 5050 support](https://discuss.pytorch.org/t/rtx-5050-sm-120-support-timeline-when-in-stable-pytorch/224288) · [PyTorch 2.7 + CUDA 12.8 for Blackwell](https://github.com/oobabooga/text-generation-webui/issues/6897)
