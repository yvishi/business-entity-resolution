"""Stage 'block': candidate generation per (country, canonical state) partition.
  forward  S1 -> pool top-K on name|address and on name   (char-3gram TF-IDF, cosine)
  reverse  pool -> S1 top-K on name|address and on name   (entity-linking direction)
  fallback pool records without a state -> S1 of the whole country, name only
TF-IDF vocabulary/IDF is fitted on TRAIN only and reused for test."""
import sys, os, time, pickle, gc
import numpy as np, polars as pl, scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

# BER_BLOCK=v2 -> the lighter settings that produced the 0.97 submission (fewer candidates: less RAM/time);
# default "wide" -> larger top-k + looser prune (more recall headroom, ~1.5-2x candidates).
if os.environ.get("BER_BLOCK", "wide") == "v2":
    K_FWD_NA, K_REV_NA, K_FWD_NM, K_REV_NM, K_FWD_AD, K_REV_AD, K_FALLBACK = 20, 5, 10, 3, 8, 3, 15
    PRUNE = dict(fna=10, rna=3, fnm=5, rnm=2, fb=8, fad=3, rad=2)
else:
    K_FWD_NA, K_REV_NA, K_FWD_NM, K_REV_NM, K_FWD_AD, K_REV_AD, K_FALLBACK = 30, 10, 15, 5, 15, 5, 25
    PRUNE = dict(fna=15, rna=5, fnm=8, rnm=3, fb=12, fad=5, rad=3)
# France blocking text: legal forms / stop words / street types are frequent in French but rare in the (train-fitted)
# TF-IDF vocabulary, so they would dominate cosine similarity -> removed from the search text (France only).
FR_BLOCK_STOP = (r"\b(sarl|sas|sasu|eurl|sa|sci|ei|cie|ets|s a s|s a r l|france|groupe|et|de|du|des|la|le|les|l|d|"
                 r"rue|avenue|allee|boulevard|impasse|place|route|chemin|square|quai|bis|ter)\b")
NT = int(os.environ.get("BER_THREADS", os.cpu_count()))


def fit_vectorizers(train_pool: pl.DataFrame, work):
    sample = train_pool.sample(min(3_000_000, train_pool.height), seed=0)
    vecs = {}
    for f in ("na", "nm", "ad"):
        v = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=3, max_df=0.02, dtype=np.float32, sublinear_tf=True)
        v.fit(sample[f].to_list()); vecs[f] = v
    cw = TfidfVectorizer(analyzer="word", token_pattern=r"\S+", min_df=2, dtype=np.float32, sublinear_tf=True)
    cw.fit(sample["core"].to_list()); vecs["cw"] = cw
    pickle.dump(vecs, open(f"{work}/tfidf.pkl", "wb"))
    return vecs


def with_fields(df):
    fr = pl.col("country") == "France"
    strip = lambda c: pl.col(c).str.replace_all(FR_BLOCK_STOP, " ").str.replace_all(r"\s+", " ").str.strip_chars()
    df = df.with_columns(pl.when(fr).then(strip("nm")).otherwise(pl.col("nm")).alias("nm_b"),
                         pl.when(fr).then(strip("ad")).otherwise(pl.col("ad")).alias("ad_b"))
    return df.with_columns((pl.col("nm_b") + " | " + pl.col("ad_b")).alias("na"), pl.col("st").alias("part"))


FIELD_TEXT = {"na": "na", "nm": "nm_b", "ad": "ad_b"}


def topk(Q, X, k):
    if Q.shape[0] == 0 or X.shape[0] == 0:
        e = np.array([], int); return e, e, e, np.array([], np.float32)
    M = sp_matmul_topn(Q, X.T.tocsr(), top_n=k, threshold=0.0, n_threads=NT, sort=True).tocsr()
    rows = np.repeat(np.arange(M.shape[0]), np.diff(M.indptr))
    ranks = np.concatenate([np.arange(n) for n in np.diff(M.indptr)]) if M.nnz else np.array([], int)
    return rows, M.indices, ranks, M.data.astype(np.float32)


def dev_parts(parts, frac, seed=0):
    rng = np.random.default_rng(seed)
    keep = {p for p in parts if rng.random() < frac}
    return keep or set(parts[:1])


def _safe(p):
    return "".join(ch if ch.isalnum() else "_" for ch in str(p))


def _block_part(a, b, vecs):
    aid, bid = a["entity_id"].to_numpy(), b["entity_id"].to_numpy()
    res = {}
    for f, kf, kr in (("na", K_FWD_NA, K_REV_NA), ("nm", K_FWD_NM, K_REV_NM), ("ad", K_FWD_AD, K_REV_AD)):
        if f not in vecs: continue
        ta, tb = a[FIELD_TEXT[f]], b[FIELD_TEXT[f]]
        A, B = vecs[f].transform(ta.to_list()), vecs[f].transform(tb.to_list())
        if f == "ad":   # empty addresses carry no evidence
            A = (sp.diags((ta != "").to_numpy().astype(np.float32)) @ A).tocsr()
            B = (sp.diags((tb != "").to_numpy().astype(np.float32)) @ B).tocsr()
        r, c, k, v = topk(A, B, kf); res["f" + f] = pl.DataFrame({"s1": aid[r], "t": bid[c], f"r_f{f}": k.astype(np.int16), f"cf_{f}": v})
        r, c, k, v = topk(B, A, kr); res["r" + f] = pl.DataFrame({"s1": aid[c], "t": bid[r], f"r_r{f}": k.astype(np.int16), f"cr_{f}": v})
    d = res["fna"]
    for key in [k for k in ("rna", "fnm", "rnm", "fad", "rad") if k in res]:
        d = d.join(res[key], on=["s1", "t"], how="full", coalesce=True)
    return d.with_columns(pl.lit(False).alias("fb"))


def block_country(s1, pool, vecs, log=print, frac=1.0, links=None, ckpt=None):
    """Returns candidate DataFrame (s1, t, ranks, fb, cosines).
    links (train only): when a subset of partitions is used for development, stateless (empty-address) pool records are
      re-sampled so the dev world has the SAME density as the full data: keep records owned by a dev S1, keep unlinked
      (distractor) records at the dev S1 fraction, drop records owned by S1 outside the dev set ('phantoms').
    ckpt: directory for per-partition checkpoints (a rerun skips finished partitions)."""
    out = []
    n_all = s1.height
    parts = sorted(set(s1["part"].drop_nulls().unique().to_list()))
    if os.environ.get("BER_PARTS"):
        frac = 0.0
    dev = frac < 1.0
    if dev:
        keep = set(os.environ["BER_PARTS"].split(",")) & set(parts) if os.environ.get("BER_PARTS") else dev_parts(parts, frac)
        parts = sorted(keep)
        s1 = s1.filter(pl.col("part").is_in(parts))
        pool = pool.filter(pl.col("part").is_in(parts) | pl.col("part").is_null())
    if dev and links is not None:
        nb0 = pool.filter(pl.col("part").is_null())
        owner = links.filter(pl.col("t").is_in(nb0["entity_id"].implode()))
        own_dev = owner.filter(pl.col("s1").is_in(s1["entity_id"].implode()))["t"]
        ratio = s1.height / max(1, n_all)
        unlinked = nb0.filter(~pl.col("entity_id").is_in(owner["t"].implode()))
        unl_keep = unlinked.filter((pl.col("entity_id").hash(seed=5) % 10_000) < int(ratio * 10_000))["entity_id"]
        keep_ids = pl.concat([own_dev, unl_keep])
        pool = pool.filter(pl.col("part").is_not_null() | pl.col("entity_id").is_in(keep_ids.implode()))
        log(f"  dev-world fallback: kept {keep_ids.len():,} of {nb0.height:,} stateless records "
            f"({own_dev.len():,} owned by dev S1, {unl_keep.len():,} distractors at ratio {ratio:.3f})")
    if os.environ.get("BER_PREFLIGHT"):
        s1 = s1.filter(pl.col("entity_id").hash(seed=1) % 100 < 3) if s1.height > 3000 else s1
        pool = pool.filter(pl.col("entity_id").hash(seed=2) % 100 < 3) if pool.height > 30000 else pool
    block_country.last_s1 = s1["entity_id"]
    if ckpt: os.makedirs(ckpt, exist_ok=True)
    for p in parts:
        f_ck = os.path.join(ckpt, _safe(p) + ".parquet") if ckpt else None
        if f_ck and os.path.exists(f_ck):
            out.append(pl.read_parquet(f_ck)); log(f"  part {p}: loaded checkpoint"); continue
        a = s1.filter(pl.col("part") == p); b = pool.filter(pl.col("part") == p)
        if b.height == 0 or a.height == 0: continue
        d = _block_part(a, b, vecs)
        if f_ck: d.write_parquet(f_ck + ".tmp"); os.replace(f_ck + ".tmp", f_ck)   # atomic: a killed run never leaves half a file
        out.append(d)
        log(f"  part {p}: S1={a.height:,} pool={b.height:,} cand={d.height:,}")
    # fallback: pool records without a partition -> all S1 of the country, name only
    nb = pool.filter(pl.col("part").is_null())
    f_ck = os.path.join(ckpt, "__fallback.parquet") if ckpt else None
    if f_ck and os.path.exists(f_ck):
        out.append(pl.read_parquet(f_ck)); log("  fallback: loaded checkpoint")
    elif nb.height and s1.height:
        A, B = vecs["nm"].transform(s1["nm_b"].to_list()), vecs["nm"].transform(nb["nm_b"].to_list())
        r, c, k, v = topk(B, A, K_FALLBACK)
        fbd = pl.DataFrame({"s1": s1["entity_id"].to_numpy()[c], "t": nb["entity_id"].to_numpy()[r],
                            "r_rnm": k.astype(np.int16), "cr_nm": v, "fb": np.ones(len(r), bool)})
        if f_ck: fbd.write_parquet(f_ck + ".tmp"); os.replace(f_ck + ".tmp", f_ck)
        out.append(fbd)
        log(f"  fallback: {nb.height:,} stateless pool records")
    if not out:
        return pl.DataFrame(schema={"s1": pl.Utf8, "t": pl.Utf8, "r_fna": pl.Int16, "r_rna": pl.Int16, "r_fnm": pl.Int16,
                                    "r_rnm": pl.Int16, "r_fad": pl.Int16, "r_rad": pl.Int16, "fb": pl.Boolean,
                                    "c_na": pl.Float32, "c_nm": pl.Float32, "c_ad": pl.Float32})
    cand = pl.concat(out, how="diagonal")
    for c_ in ("r_fad", "r_rad", "cf_ad", "cr_ad", "r_fna", "r_rna", "r_fnm", "cf_na", "cr_na", "cf_nm"):
        if c_ not in cand.columns: cand = cand.with_columns(pl.lit(None, pl.Int16 if c_.startswith("r") else pl.Float32).alias(c_))
    cand = cand.group_by(["s1", "t"]).agg(
        pl.col("r_fna").min(), pl.col("r_rna").min(), pl.col("r_fnm").min(), pl.col("r_rnm").min(),
        pl.col("r_fad").min(), pl.col("r_rad").min(), pl.col("fb").any(),
        pl.max_horizontal(pl.col("cf_na").max(), pl.col("cr_na").max()).alias("c_na"),
        pl.max_horizontal(pl.col("cf_nm").max(), pl.col("cr_nm").max()).alias("c_nm"),
        pl.max_horizontal(pl.col("cf_ad").max(), pl.col("cr_ad").max()).alias("c_ad"))
    return cand.with_columns(pl.col(["r_fna", "r_rna", "r_fnm", "r_rnm", "r_fad", "r_rad"]).fill_null(99), pl.col("fb").fill_null(False))


def prune(cand: pl.DataFrame) -> pl.DataFrame:
    """Keep pairs ranked near the top by at least one blocker (defines candidate_pairs.tsv)."""
    P = PRUNE
    m = ((pl.col("r_fna") < P["fna"]) | (pl.col("r_rna") < P["rna"]) | (pl.col("r_fnm") < P["fnm"]) | (pl.col("r_rnm") < P["rnm"])
         | (pl.col("fb") & (pl.col("r_rnm") < P["fb"])))
    if "r_fad" in cand.columns: m = m | (pl.col("r_fad") < P["fad"]) | (pl.col("r_rad") < P["rad"])
    return cand.filter(m)


def main(work, split, countries, frac=1.0):
    t0 = time.time()
    cols = ["entity_id", "nm", "ad", "country", "st", "core"]
    s1 = with_fields(pl.read_parquet(f"{work}/{split}_s1.parquet", columns=cols))
    pool = with_fields(pl.concat([pl.read_parquet(f"{work}/{split}_s{i}.parquet", columns=cols) for i in "23"]))
    if os.path.exists(f"{work}/tfidf.pkl"): vecs = pickle.load(open(f"{work}/tfidf.pkl", "rb"))
    else:
        assert split == "train", "vectorizers must be fitted on train first"
        vecs = fit_vectorizers(pool, work)
    links = None
    if split == "train" and os.path.exists(f"{work}/raw/train_links.parquet"):
        links = pl.read_parquet(f"{work}/raw/train_links.parquet")
    for c in countries:
        out_f = f"{work}/cand_{split}_{c}.parquet"
        if os.path.exists(out_f) and os.path.exists(f"{work}/s1ids_{split}_{c}.parquet"):
            print(f"{c}: candidates already built (checkpoint)", flush=True); continue
        cand = block_country(s1.filter(pl.col("country") == c), pool.filter(pl.col("country") == c), vecs, frac=frac,
                             links=links, ckpt=f"{work}/blk_{split}_{c}",
                             log=lambda m: print(m, f"[{time.time()-t0:.0f}s]", flush=True))
        cand.write_parquet(out_f + ".tmp"); os.replace(out_f + ".tmp", out_f)
        pl.DataFrame({"s1": block_country.last_s1}).write_parquet(f"{work}/s1ids_{split}_{c}.parquet")
        print(f"{c}: {cand.height:,} candidates, {cand.height/max(1, block_country.last_s1.len()):.1f}/S1 [{time.time()-t0:.0f}s]", flush=True)
        del cand; gc.collect()

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3].split(","), float(sys.argv[4]) if len(sys.argv) > 4 else 1.0)
