"""Which India pairs does TF-IDF name|address blocking miss? Reverse direction (target -> top-10 S1), sample of targets.
Also tests one cheap fix: drop native-script characters before TF-IDF (address digits/Latin words then dominate)."""
import re, numpy as np, polars as pl
from unidecode import unidecode
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn
from common import load
STOP = r"\b(inc|incorporated|llc|ltd|limited|pvt|private|corp|corporation|co|company|pc|pllc|llp|lp|plc|the|and|of)\b"
IND = r"[ऀ-෿]+"
def prep(df, drop_indic):
    def f(c, d):
        e = pl.col(c)
        if drop_indic: e = e.str.replace_all(IND, " ")
        e = e.map_elements(unidecode, return_dtype=pl.Utf8).str.to_lowercase().str.replace_all(r"[^a-z0-9 ]", " ")
        if d: e = e.str.replace_all(STOP, " ")
        return e.str.replace_all(r"\s+", " ").str.strip_chars()
    return df.with_columns((f("business_name", True) + " | " + f("business_address", False)).alias("na"))
s1r = load("train_source1").filter(pl.col("country") == "India")
links = load("train_links")
tg = pl.concat([load("train_source2"), load("train_source3")]).filter(pl.col("country") == "India")
q = links.join(tg.rename({"entity_id": "t"}), on="t").sample(20000, seed=5)
q = q.with_columns(pl.col("business_name").str.contains(IND).alias("indic_name"),
                   pl.col("business_address").str.contains(IND).alias("indic_addr"),
                   (pl.col("business_address") == "").alias("empty_addr"))
for drop in (False, True):
    s1 = prep(s1r, drop).with_row_index("i"); qq = prep(q, drop)
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=2, max_df=0.02, dtype=np.float32, sublinear_tf=True)
    S = vec.fit_transform(s1["na"].to_list()); Q = vec.transform(qq["na"].to_list())
    R = sp_matmul_topn(Q, S.T.tocsr(), top_n=10, threshold=0.0, n_threads=4, sort=True)
    idx = dict(zip(s1["entity_id"].to_list(), s1["i"].to_list()))
    true_i = [idx[x] for x in qq["s1"].to_list()]
    top1 = [R.indices[R.indptr[r]:R.indptr[r] + 1].tolist() for r in range(R.shape[0])]
    top10 = [set(R.indices[R.indptr[r]:R.indptr[r + 1]].tolist()) for r in range(R.shape[0])]
    qq = qq.with_columns(pl.Series("hit1", [t in a for t, a in zip(true_i, top1)]), pl.Series("hit10", [t in a for t, a in zip(true_i, top10)]))
    print(f"\n### drop_indic={drop}: top1={qq['hit1'].mean():.4f} top10={qq['hit10'].mean():.4f}")
    print(qq.group_by("indic_name", "indic_addr", "empty_addr").agg(pl.len(), pl.col("hit1").mean().round(4), pl.col("hit10").mean().round(4)).sort("len", descending=True))
    if not drop:
        print("sample misses:", qq.filter(~pl.col("hit10")).sample(8, seed=1).select("business_name", "business_address").rows())
