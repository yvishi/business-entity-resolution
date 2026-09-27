"""Which true pairs does blocking (+prune) lose, and why? (train dev partitions, labels)"""
import os, sys, numpy as np, polars as pl
from collections import Counter
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code", "business_entity_resolution", "src"))
from ber.block import prune
W, RAW = os.environ["WORK"], os.environ["RAW"]
src = open(os.path.join(os.path.dirname(__file__), "01_noise_catalogue.py")).read().split("res = defaultdict(Counter)")[0].replace("P = train_pairs().sample(400_000, seed=0)", "")
ns = {}; exec(src, ns); name_ops, addr_ops = ns["name_ops"], ns["addr_ops"]
L = pl.read_parquet(f"{RAW}/train_links.parquet")
raw = {i: pl.read_parquet(f"{RAW}/train_source{i}.parquet") for i in "123"}
for c in ("US", "India"):
    ids = pl.read_parquet(f"{W}/s1ids_train_{c}.parquet")["s1"]
    cand = pl.read_parquet(f"{W}/cand_train_{c}.parquet")
    T = L.filter(pl.col("s1").is_in(ids.implode()))
    j = T.join(cand.with_columns(pl.lit(1).alias("raw_c")), on=["s1", "t"], how="left").join(
        prune(cand).select("s1", "t").with_columns(pl.lit(1).alias("pr_c")), on=["s1", "t"], how="left").fill_null(0)
    print(f"\n### {c}: true pairs {T.height:,}; recall raw blocking {j['raw_c'].mean():.4f}; after prune {j['pr_c'].mean():.4f}")
    miss_raw = j.filter(pl.col("raw_c") == 0); miss_pr = j.filter((pl.col("raw_c") == 1) & (pl.col("pr_c") == 0))
    r1 = raw["1"].select(pl.col("entity_id").alias("s1"), pl.col("business_name").alias("n1"), pl.col("business_address").alias("a1"))
    r2 = pl.concat([raw["2"], raw["3"]]).select(pl.col("entity_id").alias("t"), pl.col("business_name").alias("n2"), pl.col("business_address").alias("a2"))
    for name, m in (("lost by blocking", miss_raw), ("lost by prune", miss_pr), ("reference (all true)", j.sample(min(20000, j.height), seed=0))):
        x = m.sample(min(8000, m.height), seed=0).join(r1, on="s1").join(r2, on="t")
        cnt = Counter()
        for n1, a1, n2, a2 in x.select("n1", "a1", "n2", "a2").iter_rows():
            for o in name_ops(n1, n2): cnt["name:" + o] += 1
            for o in addr_ops(a1, a2, c): cnt["addr:" + o] += 1
        top = sorted(cnt.items(), key=lambda kv: -kv[1])[:14]
        print(f"  {name} ({m.height:,} pairs): " + ", ".join(f"{k} {100*v/max(1,x.height):.0f}%" for k, v in top))
        if name == "lost by blocking":
            for n1, a1, n2, a2 in x.head(10).select("n1", "a1", "n2", "a2").iter_rows():
                print(f"     {n1!r:40} | {a1[:60]!r}\n     -> {n2!r:37} | {a2[:60]!r}")
