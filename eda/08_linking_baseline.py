"""Floor estimate: 'entity-linking' baseline. Each S2/S3 record (within its state) links to its top-1 S1 by
TF-IDF char-3gram cosine on name|address if cos >= tau (and margin over the 2nd best >= m).
Scored with the exact competition metric (macro F0.5 over S1, singletons included) on complete states."""
import sys, numpy as np, polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn
from common import load
import importlib.util
spec = importlib.util.spec_from_file_location("sp", "07_state_partitioned_blocking.py")
src = open("07_state_partitioned_blocking.py").read().split("STATES = sys.argv")[0]
exec(src)   # reuse ST, state_of, prep

def f05(pred, truth):
    if not truth and not pred: return 1.0
    if not truth or not pred: return 0.0
    tp = len(pred & truth); 
    if tp == 0: return 0.0
    p, r = tp / len(pred), tp / len(truth)
    return 1.25 * p * r / (0.25 * p + r)

states = sys.argv[1].split(",") if len(sys.argv) > 1 else ["CA", "TX", "OH", "OR", "VT"]
s1 = prep(load("train_source1").filter(pl.col("country") == "US"))
pool = prep(pl.concat([load("train_source2"), load("train_source3")]).filter(pl.col("country") == "US"))
links = load("train_links")
vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=2, max_df=0.02, dtype=np.float32, sublinear_tf=True)
vec.fit(pool.sample(2_000_000, seed=0)["na"].to_list())
rows = []
for st in states:
    a = s1.filter(pl.col("st") == st); b = pool.filter(pl.col("st") == st)
    A, B = vec.transform(a["na"].to_list()), vec.transform(b["na"].to_list())
    R = sp_matmul_topn(B, A.T.tocsr(), top_n=2, threshold=0.0, n_threads=4, sort=True)
    aid, bid = a["entity_id"].to_numpy(), b["entity_id"].to_numpy()
    for r in range(R.shape[0]):
        s, e = R.indptr[r], R.indptr[r + 1]
        if e > s:
            c1 = R.data[s]; c2 = R.data[s + 1] if e - s > 1 else 0.0
            rows.append((aid[R.indices[s]], bid[r], float(c1), float(c1 - c2)))
    truth_st = links.filter(pl.col("s1").is_in(aid))
    print(st, a.height, "S1;", len(rows), "links so far", flush=True)
L = pl.DataFrame(rows, schema=["s1", "t", "cos", "margin"], orient="row")
S1 = set(s1.filter(pl.col("st").is_in(states))["entity_id"].to_list())
truth = {}
for x, t in links.filter(pl.col("s1").is_in(list(S1))).iter_rows(): truth.setdefault(x, set()).add(t)
best = None
for tau in (0.3, 0.4, 0.5, 0.6, 0.7):
    for m in (0.0, 0.05, 0.1):
        pr = {}
        for x, t in L.filter((pl.col("cos") >= tau) & (pl.col("margin") >= m)).select("s1", "t").iter_rows():
            pr.setdefault(x, set()).add(t)
        sc = np.mean([f05(pr.get(x, set()), truth.get(x, set())) for x in S1])
        sing = [x for x in S1 if x not in truth]
        scs = np.mean([f05(pr.get(x, set()), set()) for x in sing])
        print(f"tau={tau} margin={m}: macro F0.5 = {sc:.4f}  (singletons {scs:.3f})", flush=True)
        if best is None or sc > best[0]: best = (sc, tau, m)
print("BEST", best)
