# EDA report

All numbers come from the scripts in `eda/` (raw outputs in `eda/outputs/`). Train = labelled data;
test files were only *profiled* (no fitting, nothing learned from them).

## 1. Data at a glance

| File | Rows | Countries |
|---|---|---|
| train S1 / S2 / S3 | 2.21M / 5.03M / 5.29M | US 60%, India 40% |
| test S1 / S2 / S3 | 1.73M / 4.89M / 5.08M | India 47%, US 38%, **France 15% (unseen)** |

- Ground truth: 7.64M links. Every S2/S3 record links to **at most one** S1 entity (0 violations).
- **Cluster statistics are identical per country** (US vs India): singleton rate 5.58% vs 5.59%, mean matches
  3.46 vs 3.46, no-S2 13.1% vs 13.0%, no-S3 12.1% vs 12.1%, p99 size 8. → one generator configuration applied per
  country; a strong prior that France follows the same statistics.
- ~26.6% of S2 and ~25.4% of S3 records are never linked (distractors).
- **Density shift:** test has 2.82 S2 and 2.93 S3 records per S1 (the same in all three countries: US 2.82,
  India 2.86, France 2.71 for S2), against 2.28 and 2.40 in train. Either more distractors or more matches per S1 in test.
- **Train vs test single-record profiles (US, India) are practically identical** (Indic-script names 23.5% vs 23.6% in
  S2-India, upper-case addresses 93.5% vs 93.4% in S2-US, lengths within 1 character, …). No covariate shift
  apart from France and density.
- **No leaks:** S1 row order vs S2 row order correlation 0.001; ID-number correlation 0.0001; cluster members are
  never adjacent in the files.

## 2. Corruption operators (400k true pairs, `01_noise_catalogue.py`)

Share of true pairs showing each operator (a pair usually shows 2–3).

**Names**

| Operator | S2-India | S2-US | S3-India | S3-US | Example |
|---|---|---|---|---|---|
| unchanged | 2.6 | 6.2 | 2.7 | 5.9 | |
| only case/punctuation/accents differ | 14.8 | 24.4 | 17.2 | 24.5 | `VT Instruments Corp` → `VT INSTRUMENTS CORP` |
| all upper case | 16.5 | 23.5 | 2.8 | 3.1 | |
| all lower case | 5.2 | 8.0 | 6.2 | 8.0 | |
| **legal form removed** | **51.8** | 23.6 | **45.0** | 21.2 | `Signature Mortgage Innovations Inc` → `Signature-Mortgage Innovations` |
| legal form added | 12.3 | 12.9 | 11.1 | 10.2 | `Cascade League` → `Cascade League Inc` |
| generic word added ("The", "Services", "Center") | 16.4 | 20.4 | 16.9 | 18.0 | `Parbati Products Limited Services Services` |
| **Indic-script transliteration** (Devanagari, Telugu, Malayalam, Bengali, Tamil, …) | **23.5** | 0 | **13.0** | 0 | `Dynamic Builders Private Limited` → `ഡൈനാമിക് ബിൽഡേഴ്സ് പ്രൈവറ്റ് ലിമിറ്റഡ്` |
| typo (edit distance 1–2 in a word) | 6.2 | 8.0 | 6.3 | 7.5 | `Interstate` → `lnterstate` |
| accent inserted | 5.0 | 7.4 | 5.9 | 7.4 | `Southern Standard LLC` → `Southern Stándard LLC` |
| word order shuffled | 5.7 | 7.3 | 5.9 | 7.0 | `Beis Signature Glass PLLC` → `Glass Beis Signature PLLC` |
| domain form | 4.3 | 5.7 | 4.7 | 5.4 | `Chiropractic Star Partners LLC` → `chiropracticstarpartners.com` |
| leetspeak digit | 2.5 | 3.4 | 2.7 | 3.3 | `Planet Traders` → `P1anet Traders` |
| junk prefix `>> *** -- ... # @ [` | 1.8 | 2.7 | 1.9 | 2.5 | `>> The Youth Association, LLC` |
| brackets `[Ltd]` / `(Brc)` | 2.6+2.8 | 3.3+3.7 | 3.0+3.1 | 3.3+3.6 | `Innovative Investments Private [Ltd]` |
| handle | 0.3 | 0.5 | 0.3 | 0.5 | `@pagelscoastal` |
| no shared word at all (trade name / glued) | 1.7 | 2.4 | 3.8 | 2.5 | `Modern Global Private Limited` → `Wexkelo` |

Also seen: duplicated tokens (`LLC LLC`, `Yamuaksh Yamuaksh`), titles (`Dr AL L0GISTICS`), heavy multi-typo
(`James Brc` → `Jahsdes (Brc)`).

**Addresses**

| Operator | S2-India | S2-US | S3-India | S3-US | Example |
|---|---|---|---|---|---|
| empty address | 3.8 | 4.9 | 4.2 | 4.5 | |
| unchanged | 0 | 0 | 4.1 | 4.4 | |
| upper case | 23.4 | 88.5 | 0 | 0 | |
| comma-parts reordered | 49.2 | 66.8 | 51.8 | 55.3 | `MA, Everett, Unit 2, 19 Stevenson Avenue` → `19 STEVENSON AVENUE, EVERETT, MA` |
| fewer parts (components dropped) | **51.9** | 14.0 | 35.6 | 2.0 | |
| state spelled out | 0 | 0.1 | 0 | **85.9** | `…, Marion, OH` → `…, Marion, Ohio` |
| state in native script | 23.2 | 0 | 21.8 | 0 | `Tamil Nadu` → `தமிழ்நாடு` |
| house number same | 78.6 | 72.4 | 72.9 | 71.0 | |
| house number zero-padded | 2.8 | 4.2 | 2.5 | 3.9 | `1500` → `01500` |
| house number leading digit dropped | 2.0 | 2.2 | 1.9 | 2.1 | `4948 Cypress Point` → `948 CYPRESS PT` |
| house number one digit changed | 1.7 | 1.7 | 1.7 | 1.5 | `1432 66 Street` → `2432 66 STREET` |
| house number missing | 0.8 | 8.2 | 2.0 | 6.7 | `1084 Crestview Drive` → `CRESTVIEW DR` |
| `N/A` inserted | 0.7 | 0.9 | 0.7 | 0.9 | |
| street-type / vowel-dropped abbreviations | frequent | frequent | | | `Trail` → `TRL`, `North` → `NRTH` |

Also seen: city typos (`INDEPENEDNCE`, `CEHCOTAH`, `Lubbck`), city variants (`Raleigh Township`,
`Scottsdale Fountain Hills`), inserted `PMB 6178` / `PO Box 6340`, house-number ranges (`357-361`), inserted
`NO 414`, `OLD NO.##3`.

## 3. Invariants — what survives corruption (`03`, `06`)

| Field (share of true pairs where it survives) | S2 | S3 |
|---|---|---|
| **US state (code or name)** | **100.0%** | **100.0%** |
| **India state, canonicalised** (learned map of codes/native scripts/variants: `KA`, `HYD`, `U P`, `Keralam`, `தமிழ்நாடு`, …) | **98.8%** | **98.8%** |
| US city (exact substring / fuzzy ≥ 85) | 83.6 / 89.1% | 84.5 / 89.9% |
| US house number (exact or zero-padded) | 81.9% | 83.1% |
| US street core word (exact / fuzzy) | 85.6 / 87.8% | 86.6 / 88.6% |
| US ZIP (when S1 has one; 6.6% of S1) | 81.0% | 82.5% |
| US name: ≥ 1 core word shared | 97.2% | 97.0% |
| India name: ≥ 1 core word shared (Latin) | 76.9% | 84.7% |
| India address: ≥ 1 word (≥ 4 letters) shared | 99.97% | 99.55% |

Implications:
- **State (US) / canonical state (India) / region–department (France) is the only safe hard partition.**
  Everything else (city, house number, ZIP, name words) is corrupted in 10–20% of true pairs → use as features and
  soft blockers, never as hard keys.
- India names are unreliable (transliteration 13–23%, legal form dropped ~50%); **India address words almost always
  survive** → address carries India.

## 4. Blocking experiments (`02`, 5,000 S1 queries per country, full train pool)

TF-IDF char-3gram (`max_df = 0.02`), cosine top-k. `nm` = name, `na` = name | address, `ad` = address.
Forward = S1 → pool top-k; reverse = each true target → its top-k S1 (entity-linking direction).

| | US | India |
|---|---|---|
| fwd `nm` k=100 recall | 72.0% | 60.2% |
| fwd `na` k=10 / 25 / 100 | 93.6 / 95.7 / 97.3% | 82.1 / 85.8 / 89.0% |
| fwd `ad` k=100 | — | 84.2% |
| **rev `na` top-1 (target's best S1 is the true one)** | **95.3%** | **84.9%** |
| rev `na` top-10 | 97.6% | 90.9% |
| union fwd `nm`+`na` k=25 + rev top-10 (≈ 47 fwd cand/S1) | **99.0%** | 94.5% |
| union fwd `nm`+`na`+`ad` k=50 + rev top-10 | **99.3%** | **95.6%** |
| S1 clusters fully covered (best union) | 97.4% | 88.3% |

Findings:
- Name alone is a poor blocker (generic names repeat: "Primary Care Group" ×253 in S1).
- Name + address with **both directions** reaches 99% in the US with ~50 candidates/S1.
- **India is the recall bottleneck** (95.6%) — see §5 for which records are missed.
- **Cost:** brute-force sparse top-k over a whole country costs ~13–22 ms per query per field on 4 cores
  (e.g. 111 s for 5,000 US queries on `na`) → ~2.5–4 h per field for the full train S1 on a laptop.
  Partitioning by state (safe, §3) is the fix — measured in §6.

## 5–8. Not yet run (scripts ready; first tasks of WP2 in the plan)
Stopped to save time; each is a script in `eda/` that runs in 5–15 min on a 4-core machine.
| Script | Question |
|---|---|
| `09_india_misses.py` | Which India pairs the blocker misses (native-script name? native-script address? empty address?) and whether dropping native script before TF-IDF helps. |
| `07_state_partitioned_blocking.py` | Recall and ms/query of the state-partitioned blocker (confirmed so far: US state parses for 100% of non-empty addresses; only the 3.6% empty addresses need a cross-state fallback). |
| `08_linking_baseline.py` | Floor score: each target → its top-1 S1 if cosine ≥ τ, scored with the exact macro F0.5. |
| `04_distractors_singletons.py` | Are unlinked records "copies of dropped S1 entities" (tests the density hypothesis), and how separable singletons are. |

## 9. France (test profile only, `05`)
- Only 3 regions (Hauts-de-France, Nouvelle-Aquitaine, Pays de la Loire) and a handful of cities (Bordeaux, Nantes,
  Lille, Tourcoing, Dunkerque, Calais, Roubaix, Saint-Nazaire, Pessac, Mérignac, …).
- S1 ends with the region; **S2/S3 often use the department instead** (Gironde, Nord, Pas-de-Calais,
  Loire-Atlantique) → region ↔ department map needed (analogous to India's native-script states).
- Street types abbreviated in S2/S3: `R` / `R.` (≈ 22% of addresses), `Av.`/`Ave`, `All.`, `Imp.`, `Bd`, `Rte`, `Pl`;
  `bis`/`ter` suffixes; `N°` prefixes; zero-padded numbers (`0049 RUE JEAN MERMOZ`); `(84) Rue De Trevise`.
- Legal forms: SARL, SAS, EURL, SA, SASU, SCI, EI, Cie, Ets (`Établissements`), `S.A.S`; legal form sometimes moved
  to the front (`SARL Astek Energie`); `(France)` tag; domain names (`nantesinstitut.com`).
- Accents in 24% of S2/S3 French names (vs 16% in S1) → accents are both real and injected.
- Names look template-generated like the US ones (`Roubaix Amicale SARL`, `Pauses & Fils SCI`).
- Same generator families as US/India (case, junk, domains, reordering, zero-padding) → country-agnostic
  normalisation plus French tables should transfer.
