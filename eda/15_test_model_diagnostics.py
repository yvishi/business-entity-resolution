"""Label-free diagnostics on the TEST set (inspection only: nothing is fitted on test).

For one partition per country (sampled S1), compare US / India / France on:
  1. predicted matches per S1 (stage-1 model score > 0.5) - generator gives ~3.46 true matches per S1 in every country
  2. blocking recall proxy: 'obvious copies' found by exhaustive fuzzy search over the whole partition
     (core-name token-set >= 90, same house number, address token-set >= 70) - share reached by the blocker
  3. the same proxy with the proposed France blocking-text fix (legal forms / stop words / street types removed)
Usage: WORK=<work with test_s*.parquet, tfidf.pkl, models.pkl> python eda/12_test_unsupervised.py
"""
import os, sys, re, pickle, time
import numpy as np, polars as pl
from rapidfuzz import fuzz
from rapidfuzz.process import cdist
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code", "business_entity_resolution", "src"))
from ber.block import topk, K_FWD_NA, K_FWD_NM, K_REV_NA, K_REV_NM, K_FWD_AD, K_REV_AD, with_fields, FIELD_TEXT
from ber.features import build, add_ambiguity, REC_COLS
from ber.predict import _pred

W = os.environ["WORK"]; N = int(os.environ.get("N_S1", "1500"))
SKIP = os.environ.get("SKIP_MODEL") == "1"
vecs = pickle.load(open(f"{W}/tfidf.pkl", "rb")); M = None if SKIP else pickle.load(open(f"{W}/models.pkl", "rb"))
s1_all = pl.read_parquet(f"{W}/test_s1.parquet", columns=REC_COLS)
pool_all = pl.concat([pl.read_parquet(f"{W}/test_s{i}.parquet", columns=REC_COLS) for i in "23"])
FR_BLOCK_STOP = r"\b(sarl|sas|sasu|eurl|sa|sci|ei|cie|ets|s a s|s a r l|france|groupe|et|de|du|des|la|le|les|l|d|rue|avenue|allee|boulevard|impasse|place|route|chemin|square|quai|bis|ter)\b"

def run(country, part, fix=False, seed=0):
    t0 = time.time()
    S = s1_all.filter((pl.col("country") == country) & (pl.col("st") == part))
    P = pool_all.filter((pl.col("country") == country) & (pl.col("st") == part))
    q = S.sample(min(N, S.height), seed=seed)
    S, P, q = with_fields(S), with_fields(P), with_fields(q)
    qid, pid, Sid = q["entity_id"].to_numpy(), P["entity_id"].to_numpy(), S["entity_id"].to_numpy()
    frames = []
    for f, kf in (("na", K_FWD_NA), ("nm", K_FWD_NM), ("ad", K_FWD_AD)):
        r, c, k, v = topk(vecs[f].transform(q[FIELD_TEXT[f]].to_list()), vecs[f].transform(P[FIELD_TEXT[f]].to_list()), kf)
        frames.append(pl.DataFrame({"s1": qid[r], "t": pid[c], f"r_f{f}": k.astype(np.int16), f"cf_{f}": v}))
    cand = frames[0]
    for fr_ in frames[1:]: cand = cand.join(fr_, on=["s1", "t"], how="full", coalesce=True)
    tt = cand["t"].unique().to_list(); Psel = P.filter(pl.col("entity_id").is_in(tt))
    for f, kr in (("na", K_REV_NA), ("nm", K_REV_NM), ("ad", K_REV_AD)):
        r, c, k, v = topk(vecs[f].transform(Psel[FIELD_TEXT[f]].to_list()), vecs[f].transform(S[FIELD_TEXT[f]].to_list()), kr)
        rv = pl.DataFrame({"s1": Sid[c], "t": Psel["entity_id"].to_numpy()[r], f"r_r{f}": k.astype(np.int16), f"cr_{f}": v})
        cand = cand.join(rv, on=["s1", "t"], how="left")
    cand = cand.with_columns(pl.col(["r_fna", "r_rna", "r_fnm", "r_rnm", "r_fad", "r_rad"]).fill_null(99), pl.lit(False).alias("fb"),
                             pl.max_horizontal("cf_na", "cr_na").alias("c_na"), pl.max_horizontal("cf_nm", "cr_nm").alias("c_nm"),
                             pl.max_horizontal("cf_ad", "cr_ad").alias("c_ad"))
    from ber.block import prune
    cand = prune(cand).select("s1", "t", "r_fna", "r_rna", "r_fnm", "r_rnm", "r_fad", "r_rad", "fb", "c_na", "c_nm", "c_ad")
    if SKIP:
        d = cand.with_columns(pl.lit(0.0).alias("p1"))
    else:
        d = add_ambiguity(build(cand, S, P, vecs), S)
        d = d.with_columns(pl.Series("p1", _pred(M["m1"], d.select(M["feats1"]).to_numpy().astype(np.float32))))
    npos = d.group_by("s1").agg((pl.col("p1") > 0.5).sum().alias("n")).join(pl.DataFrame({"s1": qid}), on="s1", how="right").fill_null(0)
    # exhaustive 'obvious copy' search over the partition
    qc, qa, qh = q["core"].to_list(), q["ad"].to_list(), q["hn"].to_list()
    Pc, Pa, Ph = P["core"].to_list(), P["ad"].to_list(), P["hn"].to_numpy()
    obvious = []
    for i in range(0, len(qc), 300):
        M1 = cdist(qc[i:i + 300], Pc, scorer=fuzz.token_set_ratio, score_cutoff=90, workers=-1, dtype=np.uint8)
        for a, b in zip(*np.nonzero(M1)):
            if qh[i + a] and qh[i + a] == Ph[b] and fuzz.token_set_ratio(qa[i + a], Pa[b]) >= 70:
                obvious.append((qid[i + a], pid[b]))
    cs = set(zip(cand["s1"].to_list(), cand["t"].to_list()))
    got = sum(1 for o in obvious if o in cs)
    hi = set(zip(*d.filter(pl.col("p1") > 0.5).select("s1", "t").to_dict(as_series=False).values()))
    sc = sum(1 for o in obvious if o in hi)
    print(f"{country:6} {part:8} fix={fix!s:5} S1 sample={q.height:,} partition S1={S.height:,} pool={P.height:,} | "
          f"pred>0.5 per S1: mean {npos['n'].mean():.2f}, zero {(npos['n'] == 0).mean():.1%} | "
          f"obvious copies {len(obvious):,} ({len(obvious)/q.height:.2f}/S1): in candidates {got/max(1,len(obvious)):.1%}, "
          f"scored>0.5 {sc/max(1,len(obvious)):.1%}  [{time.time()-t0:.0f}s]", flush=True)
    return q, d, obvious, cs


import json
res = {}
for c, p in [tuple(x.split(":")) for x in os.environ.get("PARTS", "US:OH,India:Kerala,France:PDL").split(",")]:
    q, d, obvious, cs = run(c, p)
    res[(c, p)] = (q, d)

print("\n### Predicted matches per S1 (stage-1 score), by country: generator truth ~3.46 per S1, 5.6% singletons")
for (c, p), (q, d) in res.items():
    g = d.group_by("s1").agg((pl.col("p1") > 0.5).sum().alias("n5"), (pl.col("p1") > 0.8).sum().alias("n8"),
                             ((pl.col("p1") > 0.2) & (pl.col("p1") < 0.8)).sum().alias("unsure"))
    g = pl.DataFrame({"s1": q["entity_id"]}).join(g, on="s1", how="left").fill_null(0)
    hist = g["n5"].clip(0, 8).value_counts().sort("n5")
    print(f"  {c:6} {p:7}: mean p>0.5 {g['n5'].mean():.2f} | p>0.8 {g['n8'].mean():.2f} | zero {(g['n5']==0).mean():.1%} | "
          f">=7 {(g['n5']>=7).mean():.1%} | uncertain(0.2-0.8)/S1 {g['unsure'].mean():.2f} | hist {dict(hist.iter_rows())}")

feat_show = ["c_na", "c_nm", "nm_ratio", "core_tset", "ad_tset", "ad_tsort", "hn_eq", "st_eq", "legal_eq", "amb_core", "r_fna", "r_rna"]
for (c, p), (q, d) in res.items():
    if c != "France": continue
    raw = pl.concat([pl.read_parquet(f"{W}/raw/test_source{i}.parquet") for i in "123"]).filter(pl.col("country") == c)
    RR = dict(zip(raw["entity_id"].to_list(), zip(raw["business_name"].to_list(), raw["business_address"].to_list())))
    g = d.group_by("s1").agg((pl.col("p1") > 0.5).sum().alias("n5"))
    zero = pl.DataFrame({"s1": q["entity_id"]}).join(g, on="s1", how="left").fill_null(0).filter(pl.col("n5") == 0)["s1"].to_list()
    print(f"\n### {c} {p}: S1 with ZERO predicted matches ({len(zero)}) - top candidates and features")
    for x in zero[:10]:
        print(f"\n  S1 {RR[x][0]!r} | {RR[x][1]!r}")
        for row in d.filter(pl.col("s1") == x).sort("p1", descending=True).head(4).select(["t", "p1"] + feat_show).iter_rows(named=True):
            print(f"     p={row['p1']:.2f} {RR[row['t']][0]!r:34} | {RR[row['t']][1][:55]!r:57} " +
                  " ".join(f"{k}={row[k]:.2f}" if isinstance(row[k], float) else f"{k}={row[k]}" for k in feat_show))
    many = g.filter(pl.col("n5") >= 7)["s1"].to_list()
    print(f"\n### {c} {p}: S1 with >=7 predicted matches ({len(many)})")
    for x in many[:5]:
        print(f"\n  S1 {RR[x][0]!r} | {RR[x][1]!r}")
        for row in d.filter((pl.col("s1") == x) & (pl.col("p1") > 0.5)).sort("p1", descending=True).head(9).select("t", "p1").iter_rows():
            print(f"     p={row[1]:.2f} {RR[row[0]][0]!r:34} | {RR[row[0]][1][:60]!r}")
    d.write_parquet(f"{W}/diag_{c}_{p}.parquet")
