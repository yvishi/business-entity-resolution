"""Tests the 'distractors = copies of dropped S1 entities' hypothesis and how separable singletons are.
US only (largest), TF-IDF char-3gram on name|address, cosine top-1."""
import numpy as np, polars as pl
from unidecode import unidecode
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn
from common import load

STOP = r"\b(inc|incorporated|llc|ltd|limited|pvt|private|corp|corporation|co|company|pc|pllc|llp|lp|plc|the|and|of)\b"
def prep(df):
    f = lambda c, d: (pl.col(c).map_elements(unidecode, return_dtype=pl.Utf8).str.to_lowercase().str.replace_all(r"[^a-z0-9 ]", " ")
                      .pipe(lambda e: e.str.replace_all(STOP, " ") if d else e).str.replace_all(r"\s+", " ").str.strip_chars())
    return df.with_columns((f("business_name", True) + " | " + f("business_address", False)).alias("na"))
C = "US"
s1 = prep(load("train_source1").filter(pl.col("country") == C))
s2 = prep(load("train_source2").filter(pl.col("country") == C))
s3 = prep(load("train_source3").filter(pl.col("country") == C))
links = load("train_links")
linked = set(links["t"].to_list()); s1_with = set(links["s1"].to_list())
vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=2, max_df=0.02, dtype=np.float32, sublinear_tf=True)
vec.fit(pl.concat([s2["na"], s3["na"]]).to_list())
X1, X3 = vec.transform(s1["na"].to_list()), vec.transform(s3["na"].to_list())
def top1(Q, X):
    M = sp_matmul_topn(Q, X.T.tocsr(), top_n=1, threshold=0.0, n_threads=4)
    return np.asarray(M.max(axis=1).todense()).ravel()
s2 = s2.with_columns(pl.col("entity_id").is_in(linked).alias("linked"))
for flag in (True, False):
    sub = s2.filter(pl.col("linked") == flag).sample(15000, seed=3)
    Q = vec.transform(sub["na"].to_list())
    b3, b1 = top1(Q, X3), top1(Q, X1)
    print(f"S2 {'linked ' if flag else 'orphan '}: best cos to S3 p25/p50/p75 = {np.percentile(b3,[25,50,75]).round(3)} | best cos to S1 = {np.percentile(b1,[25,50,75]).round(3)}")
# singletons vs non-singletons: best similarity from S1 to pool
X2 = vec.transform(s2["na"].to_list())
s1 = s1.with_columns(pl.col("entity_id").is_in(s1_with).alias("has"))
for flag in (True, False):
    sub = s1.filter(pl.col("has") == flag).sample(10000, seed=4)
    Q = vec.transform(sub["na"].to_list())
    b = np.maximum(top1(Q, X2), top1(Q, X3))
    print(f"S1 {'with matches' if flag else 'singleton   '}: best cos to S2∪S3 p10/p25/p50/p75/p90 = {np.percentile(b,[10,25,50,75,90]).round(3)}")
