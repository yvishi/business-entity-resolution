"""Do copies of an entity keep its address? Compare countries on TEST using exact-name groups (label-free).
For S1 records whose normalised name is UNIQUE among S1 of their partition, collect pool records in the same
partition with the SAME normalised name (these are almost surely copies of that S1) and measure how often
they share the S1 house number / street words. Reference: the same statistic on TRAIN with true labels."""
import os, re, numpy as np, polars as pl
W = os.environ["WORK"]; RAW = os.environ["RAW"]
cols = ["entity_id", "nm", "core", "ad", "hn", "st", "country"]
def stats(split, country, label=None):
    s1 = pl.read_parquet(f"{W}/{split}_s1.parquet", columns=cols).filter(pl.col("country") == country)
    pool = pl.concat([pl.read_parquet(f"{W}/{split}_s{i}.parquet", columns=cols) for i in "23"]).filter(pl.col("country") == country)
    # S1 names unique within partition
    s1 = s1.with_columns(pl.len().over(["st", "nm"]).alias("n_same"))
    print(f"\n[{split} {country}] S1={s1.height:,}; share of S1 whose exact normalised name is shared by another S1 in partition: {(s1['n_same'] > 1).mean():.1%}")
    u = s1.filter(pl.col("n_same") == 1).sample(min(40000, s1.height), seed=0)
    j = u.join(pool, on=["st", "nm"], suffix="_2")
    per = j.group_by("entity_id").len()
    street = lambda a: set(t for t in a.split() if len(t) > 3 and not t.isdigit())
    same_hn = (j["hn"] == j["hn_2"]) & (j["hn"] != "")
    sw = [len(street(a) & street(b)) / max(1, len(street(a))) for a, b in zip(j["ad"].to_list(), j["ad_2"].to_list())]
    print(f"  unique-name S1 sampled: {u.height:,}; with >=1 exact-name pool record: {per.height/u.height:.1%}; mean exact-name pool records per such S1: {per['len'].mean():.2f}")
    print(f"  exact-name pool records: same house number {same_hn.mean():.1%}; >=50% of S1 street words shared {(np.array(sw) >= 0.5).mean():.1%}; empty address {(j['ad_2'] == '').mean():.1%}")
    if label is not None:
        L = pl.read_parquet(f"{RAW}/train_links.parquet")
        jj = j.join(L.rename({"s1": "entity_id", "t": "entity_id_2"}).with_columns(pl.lit(1).alias("y")), on=["entity_id", "entity_id_2"], how="left")
        print(f"  (train check) exact-name pool records that are TRUE matches: {jj['y'].fill_null(0).mean():.1%}; "
              f"true-match rate when house number differs: {jj.filter(~same_hn)['y'].fill_null(0).mean():.1%}")
    return j
for c in ("US", "India"): stats("train", c, label=True)
for c in ("US", "India", "France"): j = stats("test", c)
print("\nFrance examples (unique-name S1 and its exact-name pool records):")
for e in j["entity_id"].unique().head(6).to_list():
    g = j.filter(pl.col("entity_id") == e)
    print(f"  S1 {g['nm'][0]!r} | {g['ad'][0]!r}")
    for a in g["ad_2"].head(6).to_list(): print(f"       -> {a!r}")
