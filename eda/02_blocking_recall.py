"""Blocking recall experiments on train (per country).
Forward: sampled S1 queries -> top-k over the full S2+S3 pool of that country.
Reverse (entity-linking view): each true target of the sampled S1 -> top-k over all S1 of that country.
Reports pair recall, 'cluster fully covered' rate, and top-1 linking accuracy."""
import sys, time, re, numpy as np, polars as pl
from unidecode import unidecode
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn
from common import load

N_Q = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
STOP = r"\b(inc|incorporated|llc|ltd|limited|pvt|private|corp|corporation|co|company|pc|pllc|llp|lp|plc|the|and|of)\b"

def norm_expr(col, drop_legal):
    e = (pl.col(col).map_elements(unidecode, return_dtype=pl.Utf8).str.to_lowercase()
         .str.replace_all(r"[^a-z0-9 ]", " "))
    if drop_legal: e = e.str.replace_all(STOP, " ")
    return e.str.replace_all(r"\s+", " ").str.strip_chars()

def prep(df):
    return df.with_columns(norm_expr("business_name", True).alias("nm"), norm_expr("business_address", False).alias("ad")) \
             .with_columns((pl.col("nm") + " | " + pl.col("ad")).alias("na"))

s1_all = load("train_source1"); pool_all = pl.concat([load("train_source2"), load("train_source3")])
links = load("train_links")
for country in ["US", "India"]:
    t0 = time.time()
    s1 = prep(s1_all.filter(pl.col("country") == country)).with_row_index("i")
    pool = prep(pool_all.filter(pl.col("country") == country)).with_row_index("j")
    q = s1.sample(N_Q, seed=1)
    lk = links.join(q.select(pl.col("entity_id").alias("s1"), "i"), on="s1").join(
        pool.select(pl.col("entity_id").alias("t"), "j"), on="t")
    truth = set(zip(lk["i"].to_list(), lk["j"].to_list()))
    per_s1 = lk.group_by("i").agg(pl.col("j"))
    print(f"\n===== {country}: S1={s1.height:,} pool={pool.height:,} queries={q.height:,} true pairs={len(truth):,}  prep {time.time()-t0:.0f}s", flush=True)
    results = {}
    for field, ng in [("nm", (3, 3)), ("na", (3, 3)), ("ad", (3, 3))]:
        t0 = time.time()
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=ng, min_df=2, max_df=0.02, dtype=np.float32, sublinear_tf=True)
        P = vec.fit_transform(pool[field].to_list())
        Q = vec.transform(q[field].to_list())
        K = 100
        t1 = time.time()
        M = sp_matmul_topn(Q, P.T.tocsr(), top_n=K, threshold=0.0, n_threads=4, sort=True)
        print(f"  [{field}] vocab={P.shape[1]:,} nnz/row={P.nnz/P.shape[0]:.1f} fwd matmul {time.time()-t1:.0f}s for {Q.shape[0]:,} queries", flush=True)
        # forward recall at several k
        found = {}
        indptr, indices = M.indptr, M.indices
        qi = q["i"].to_list()
        for k in (10, 25, 50, 100):
            cand = set()
            for r, i in enumerate(qi):
                cand.update((i, int(j)) for j in indices[indptr[r]:indptr[r] + k])
            hit = len(truth & cand)
            found[k] = cand
            print(f"  fwd {field:2} k={k:3}: pair recall {hit/len(truth):.4f}", flush=True)
        results[field] = found
        # reverse: true targets -> top-k S1
        S = vec.transform(s1[field].to_list())
        tq = lk.select("j", "i")
        T = P[tq["j"].to_numpy()]
        R = sp_matmul_topn(T, S.T.tocsr(), top_n=10, threshold=0.0, n_threads=4, sort=True)
        ti = tq["i"].to_numpy()
        top = [R.indices[R.indptr[r]:R.indptr[r + 1]] for r in range(R.shape[0])]
        for k in (1, 3, 10):
            acc = np.mean([ti[r] in set(top[r][:k].tolist()) for r in range(len(top))])
            print(f"  rev {field:2} target->S1 top-{k:2}: {acc:.4f}", flush=True)
        results[field]["rev10"] = {(int(ti[r]), int(tq['j'][r])) for r in range(len(top)) if ti[r] in set(top[r].tolist())}
        print(f"  ({field} took {time.time()-t0:.0f}s)", flush=True)
    for combo in [("nm", "na"), ("nm", "na", "ad")]:
        for k in (25, 50):
            u = set().union(*[results[f][k] for f in combo])
            n_c = len(u) / q.height
            print(f"  UNION fwd {'+'.join(combo)} k={k}: recall {len(truth & u)/len(truth):.4f}  cand/S1 {n_c:.1f}")
            u2 = u | set().union(*[results[f]["rev10"] for f in combo])
            print(f"  UNION fwd+rev10 {'+'.join(combo)} k={k}: recall {len(truth & u2)/len(truth):.4f}")
    # cluster fully covered (best union)
    u = set().union(*[results[f][50] for f in ("nm", "na", "ad")]) | set().union(*[results[f]["rev10"] for f in ("nm", "na", "ad")])
    full = np.mean([all((i, j) in u for j in js) for i, js in per_s1.iter_rows()])
    print(f"  clusters fully covered (nm+na+ad k=50 + rev10): {full:.4f}")
    del P, Q, S, T, M, R
