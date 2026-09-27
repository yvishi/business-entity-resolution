"""Stage 'features': pair features for candidate pairs (all country-agnostic)."""
import numpy as np, polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein, Hamming
from rapidfuzz.process import cpdist

REC_COLS = ["entity_id", "nm", "core", "glued", "legal", "ad", "ad_sorted", "hn", "nums", "st", "addr_empty", "indic_name", "country", "skel"]


_IDF = {}


def idf_cosine(a, b, cw):
    """Binary-TF IDF cosine between word sets. IDF from the train-fitted word vocabulary; words never seen in train
    (e.g. French names) get the MAX idf - an unseen word shared by two records is strong evidence, not missing."""
    key = id(cw)
    if key not in _IDF:
        _IDF.clear(); _IDF[key] = ({w: float(cw.idf_[i]) for w, i in cw.vocabulary_.items()}, float(cw.idf_.max()))
    idf, mx = _IDF[key]
    cache, out = {}, np.empty(len(a), dtype=np.float32)
    for n, (x, y) in enumerate(zip(a, b)):
        r = cache.get((x, y))
        if r is None:
            A, B = set(x.split()), set(y.split())
            wa = {w: idf.get(w, mx) ** 2 for w in A}; wb = {w: idf.get(w, mx) ** 2 for w in B}
            num = sum(wa[w] for w in A & B); den = (sum(wa.values()) * sum(wb.values())) ** 0.5
            r = num / den if den else 0.0
            cache[(x, y)] = r
        out[n] = r
    return out


def rowdot(X, Y):
    return np.asarray(X.multiply(Y).sum(axis=1)).ravel().astype(np.float32)


def build(cand: pl.DataFrame, s1: pl.DataFrame, pool: pl.DataFrame, vecs) -> pl.DataFrame:
    """cand: s1, t, r_* ; s1/pool: normalised records. Returns cand + features."""
    a = s1.select(REC_COLS).rename({c: c + "_1" for c in REC_COLS if c != "entity_id"}).rename({"entity_id": "s1"})
    b = pool.select(REC_COLS).rename({c: c + "_2" for c in REC_COLS if c != "entity_id"}).rename({"entity_id": "t"})
    d = cand.join(a, on="s1").join(b, on="t")
    L = lambda c: d[c].to_list()
    W = dict(workers=-1, dtype=np.float32)
    f = {}
    n1, n2, c1, c2, g1, g2 = L("nm_1"), L("nm_2"), L("core_1"), L("core_2"), L("glued_1"), L("glued_2")
    a1, a2, as1, as2 = L("ad_1"), L("ad_2"), L("ad_sorted_1"), L("ad_sorted_2")
    f["nm_ratio"] = cpdist(n1, n2, scorer=fuzz.ratio, **W)
    f["nm_tsort"] = cpdist(n1, n2, scorer=fuzz.token_sort_ratio, **W)
    f["nm_tset"] = cpdist(n1, n2, scorer=fuzz.token_set_ratio, **W)
    f["core_ratio"] = cpdist(c1, c2, scorer=fuzz.ratio, **W)
    f["core_tset"] = cpdist(c1, c2, scorer=fuzz.token_set_ratio, **W)
    f["core_partial"] = cpdist(c1, c2, scorer=fuzz.partial_ratio, **W)
    f["core_jw"] = cpdist(c1, c2, scorer=JaroWinkler.normalized_similarity, **W)
    f["glued_ratio"] = cpdist(g1, g2, scorer=fuzz.ratio, **W)
    f["glued_partial"] = cpdist(g1, g2, scorer=fuzz.partial_ratio, **W)
    f["core_lev"] = cpdist(c1, c2, scorer=Levenshtein.distance, workers=-1).astype(np.float32)
    f["ad_tset"] = cpdist(a1, a2, scorer=fuzz.token_set_ratio, **W)
    f["ad_tsort"] = cpdist(as1, as2, scorer=fuzz.ratio, **W)
    f["ad_partial"] = cpdist(a1, a2, scorer=fuzz.partial_ratio, **W)
    f["ad_ratio"] = cpdist(a1, a2, scorer=fuzz.ratio, **W)
    f["skel_ratio"] = cpdist(L("skel_1"), L("skel_2"), scorer=fuzz.ratio, **W)
    if vecs is not None and "cw" in vecs:   # IDF-weighted word overlap of core names
        f["core_wcos"] = idf_cosine(c1, c2, vecs["cw"])
    f["_ham"] = cpdist(L("hn_1"), L("hn_2"), scorer=Hamming.distance, workers=-1).astype(np.int32)
    hn1_l, hn2_l = L("hn_1"), L("hn_2")
    f["hn_digit_perm"] = np.array([1.0 if h1 and h2 and len(h1) == len(h2) and h1 != h2 and sorted(h1) == sorted(h2)
                                    else 0.0 for h1, h2 in zip(hn1_l, hn2_l)], dtype=np.float32)
    d = d.with_columns([pl.Series(k, v) for k, v in f.items()])
    hn1, hn2 = pl.col("hn_1"), pl.col("hn_2")
    d = d.with_columns(
        (hn1 == hn2).and_(hn1 != "").cast(pl.Int8).alias("hn_eq"),
        ((hn1 != "") & (hn2 != "") & (hn1.str.ends_with(hn2) | hn2.str.ends_with(hn1))).cast(pl.Int8).alias("hn_suffix"),
        ((hn1 == "") | (hn2 == "")).cast(pl.Int8).alias("hn_missing"),
        pl.col("nums_1").list.set_intersection("nums_2").list.len().alias("nums_inter"),
        pl.col("nums_1").list.set_union("nums_2").list.len().alias("nums_union"),
        pl.col("hn_2").is_in(pl.col("nums_1")).cast(pl.Int8).alias("hn2_in_nums1"),
        pl.col("hn_1").is_in(pl.col("nums_2")).cast(pl.Int8).alias("hn1_in_nums2"),
        (pl.col("st_1") == pl.col("st_2")).fill_null(False).cast(pl.Int8).alias("st_eq"),
        ((hn1.str.len_chars() == hn2.str.len_chars()) & (hn1 != "") & (pl.col("_ham") == 1)).cast(pl.Int8).alias("hn_ham1"),
        (pl.col("legal_1") == pl.col("legal_2")).cast(pl.Int8).alias("legal_eq"),
        ((pl.col("legal_1") != "") & (pl.col("legal_2") != "")).cast(pl.Int8).alias("legal_both"),
        (pl.col("core_1") == pl.col("core_2")).cast(pl.Int8).alias("core_eq"),
        (pl.col("glued_1") == pl.col("glued_2")).cast(pl.Int8).alias("glued_eq"),
        pl.col("addr_empty_2").cast(pl.Int8), pl.col("indic_name_2").cast(pl.Int8),
        pl.col("nm_1").str.len_chars().alias("len_nm1"), pl.col("nm_2").str.len_chars().alias("len_nm2"),
        pl.col("ad_1").str.len_chars().alias("len_ad1"), pl.col("ad_2").str.len_chars().alias("len_ad2"),
        (pl.col("t").str.slice(0, 2) == "S3").cast(pl.Int8).alias("is_s3"),
        pl.col("fb").cast(pl.Int8),
    ).with_columns((pl.col("nums_inter") / pl.col("nums_union").clip(1)).cast(pl.Float32).alias("nums_jacc"))
    # house-number relation: corruption (digit dropped/changed, zero pad) vs a NEIGHBOURING address (distractor)
    hv1, hv2 = hn1.str.slice(0, 9).cast(pl.Int64, strict=False), hn2.str.slice(0, 9).cast(pl.Int64, strict=False)
    both = (hn1 != "") & (hn2 != "")
    d = d.with_columns(
        pl.when(both).then((hv1 - hv2).abs().clip(0, 100000)).otherwise(-1).cast(pl.Float32).alias("hn_absdiff"),
        pl.when(both).then((hv1 - hv2).abs() / pl.max_horizontal(hv1, hv2, pl.lit(1))).otherwise(-1).cast(pl.Float32).alias("hn_reldiff"),
        pl.when(both).then((hn1.str.slice(0, 1) == hn2.str.slice(0, 1)).cast(pl.Int8)).otherwise(-1).alias("hn_first_eq"),
        pl.when(both).then((hn1.str.slice(-1, 1) == hn2.str.slice(-1, 1)).cast(pl.Int8)).otherwise(-1).alias("hn_last_eq"),
        (hn1.str.len_chars().cast(pl.Int16) - hn2.str.len_chars().cast(pl.Int16)).alias("hn_lendiff"),
    )
    # context counts (label-free)
    d = d.with_columns(pl.len().over("s1").alias("n_cand_s1"), pl.len().over("t").alias("n_cand_t"))
    return d


FEATURES = ["r_fna", "r_rna", "r_fnm", "r_rnm", "fb", "nm_ratio", "nm_tsort", "nm_tset", "core_ratio", "core_tset",
            "core_partial", "core_jw", "glued_ratio", "glued_partial", "core_lev", "ad_tset", "ad_tsort", "ad_partial",
            "ad_ratio", "c_na", "c_nm", "c_ad", "r_fad", "r_rad", "skel_ratio", "core_wcos", "hn_ham1", "hn_digit_perm", "hn_absdiff", "hn_reldiff", "hn_first_eq", "hn_last_eq", "hn_lendiff", "hn_eq", "hn_suffix", "hn_missing", "nums_inter", "nums_jacc",
            "hn2_in_nums1", "hn1_in_nums2", "st_eq", "legal_eq", "legal_both", "core_eq", "glued_eq", "addr_empty_2",
            "indic_name_2", "len_nm1", "len_nm2", "len_ad1", "len_ad2", "is_s3", "n_cand_s1", "n_cand_t", "amb_core"]


def add_ambiguity(d, s1):
    """How many S1 records in the same partition share the core name (generic names like 'Primary Care Group')."""
    amb = s1.group_by(["country", "core"]).len().rename({"len": "amb_core", "core": "core_1", "country": "country_1"})
    return d.join(amb, on=["country_1", "core_1"], how="left").with_columns(pl.col("amb_core").fill_null(1))


def add_context(d, score_col="p1"):
    """Stage-2 features from stage-1 (OOF) scores."""
    return d.with_columns(
        pl.col(score_col).rank("ordinal", descending=True).over("s1").cast(pl.Int16).alias("ctx_rank_s1"),
        pl.col(score_col).rank("ordinal", descending=True).over("t").cast(pl.Int16).alias("ctx_rank_t"),
        (pl.col(score_col).max().over("s1") - pl.col(score_col)).alias("ctx_gap_s1"),
        (pl.col(score_col).max().over("t") - pl.col(score_col)).alias("ctx_gap_t"),
        (pl.col(score_col) > 0.5).sum().over("s1").cast(pl.Int16).alias("ctx_n_hi_s1"),
        (pl.col(score_col) > 0.5).sum().over("t").cast(pl.Int16).alias("ctx_n_hi_t"),
        pl.col(score_col).sum().over("s1").alias("ctx_sum_s1"),
    )

CTX = ["p1", "ctx_rank_s1", "ctx_rank_t", "ctx_gap_s1", "ctx_gap_t", "ctx_n_hi_s1", "ctx_n_hi_t", "ctx_sum_s1"]


def add_support(d: pl.DataFrame, recs: pl.DataFrame, p="p1", pmin=0.05, phi=0.5, bucket_pairs=2_000_000) -> pl.DataFrame:
    """Stage-2 'support' features: does candidate t resemble the OTHER high-scoring candidates of the same S1?
    Copies of one business resemble each other; a distractor (copy of a different, absent business) does not.
    recs: entity_id, core, ad for the targets. Processed in S1 buckets to bound memory."""
    c = d.filter(pl.col(p) >= pmin).select("s1", "t", p)
    hi = c.filter(pl.col(p) >= phi).select("s1", pl.col("t").alias("u"), pl.col(p).alias("pu"))
    r = recs.select("entity_id", "core", "ad")
    est = c.height * max(1.0, hi.height / max(1, c.select(pl.col("s1").n_unique()).item()))
    nb = int(max(1, est // bucket_pairs + 1))
    outs = []
    for b in range(nb):
        cb = c.filter(pl.col("s1").hash(seed=13) % nb == b) if nb > 1 else c
        hb = hi.filter(pl.col("s1").hash(seed=13) % nb == b) if nb > 1 else hi
        pr = cb.join(hb, on="s1").filter(pl.col("t") != pl.col("u"))
        if pr.height == 0: continue
        pr = pr.join(r.rename({"entity_id": "t", "core": "ct", "ad": "at"}), on="t").join(
            r.rename({"entity_id": "u", "core": "cu", "ad": "au"}), on="u")
        pr = pr.with_columns(pl.Series("sn", cpdist(pr["ct"].to_list(), pr["cu"].to_list(), scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)),
                             pl.Series("sa", cpdist(pr["at"].to_list(), pr["au"].to_list(), scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)))
        outs.append(pr.group_by(["s1", "t"]).agg(pl.col("sn").max().alias("sup_name"), pl.col("sa").max().alias("sup_addr"),
                                                  ((pl.col("sn") >= 80) & (pl.col("sa") >= 70)).sum().cast(pl.Int16).alias("sup_n"),
                                                  pl.col("sn").sort_by("pu", descending=True).first().alias("sup_best_name"),
                                                  pl.col("sa").sort_by("pu", descending=True).first().alias("sup_best_addr")))
        del pr
    if outs:
        g = pl.concat(outs)
    else:
        g = pl.DataFrame(schema={"s1": pl.Utf8, "t": pl.Utf8, "sup_name": pl.Float32, "sup_addr": pl.Float32, "sup_n": pl.Int16,
                                 "sup_best_name": pl.Float32, "sup_best_addr": pl.Float32})
    d = d.join(g, on=["s1", "t"], how="left", maintain_order="left")   # keep row order (aligned with the feature matrix)
    return d.with_columns(pl.col(["sup_name", "sup_addr", "sup_best_name", "sup_best_addr"]).fill_null(-1.0).cast(pl.Float32),
                          pl.col("sup_n").fill_null(0))

SUPPORT = ["sup_name", "sup_addr", "sup_n", "sup_best_name", "sup_best_addr"]
