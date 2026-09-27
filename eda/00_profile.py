import polars as pl, re, json, sys
from difflib import SequenceMatcher
import os
D = os.environ.get("RAW_DIR", "data/raw") + "/"
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(200); pl.Config.set_fmt_str_lengths(80)

def load(n):
    return pl.read_csv(D + n, separator="\t", quote_char=None, infer_schema=False,
                       missing_utf8_is_empty_string=True)

src = {n: load(f"{n}.tsv") for n in ["train_source1","train_source2","train_source3","test_source1","test_source2","test_source3"]}
gt = load("train_ground_truth.tsv")

print("### BASIC PROFILE")
for n, df in src.items():
    ids = df["entity_id"]
    print(f"\n{n}: rows={df.height:,} unique_ids={ids.n_unique():,} id_prefixes={ids.str.slice(0,3).unique().to_list()}")
    for c in ["business_name","business_address","country"]:
        s = df[c]
        empty = (s.str.strip_chars() == "").sum()
        print(f"  {c}: empty={empty:,} ({empty/df.height:.2%}) distinct={s.n_unique():,}")
    print("  country:", df["country"].value_counts(sort=True).with_columns((pl.col("count")/df.height*100).round(2).alias("pct")).rows())
    dup = df.select(["business_name","business_address","country"]).is_duplicated().sum()
    print(f"  exact duplicate (name,addr,country) rows: {dup:,}")
    nl = df["business_name"].str.len_chars(); al = df["business_address"].str.len_chars()
    print(f"  name len: mean={nl.mean():.1f} p50={nl.median()} p99={nl.quantile(0.99)} max={nl.max()}")
    print(f"  addr len: mean={al.mean():.1f} p50={al.median()} p99={al.quantile(0.99)} max={al.max()}")

print("\n### TEXT CHARACTERISTICS (train)")
feats = {
    "devanagari_name": pl.col("business_name").str.contains(r"[ऀ-ॿ]"),
    "non_ascii_name": pl.col("business_name").str.contains(r"[^\x00-\x7F]"),
    "all_upper_name": (pl.col("business_name") == pl.col("business_name").str.to_uppercase()) & pl.col("business_name").str.contains("[A-Za-z]"),
    "all_lower_name": (pl.col("business_name") == pl.col("business_name").str.to_lowercase()) & pl.col("business_name").str.contains("[A-Za-z]"),
    "junk_prefix_name": pl.col("business_name").str.contains(r"^[^\p{L}\p{N}]+"),
    "domain_like_name": pl.col("business_name").str.contains(r"(?i)\.(com|net|org|in|fr|co)$"),
    "digits_in_name": pl.col("business_name").str.contains(r"\d"),
    "all_upper_addr": (pl.col("business_address") == pl.col("business_address").str.to_uppercase()) & pl.col("business_address").str.contains("[A-Za-z]"),
    "addr_starts_state": pl.col("business_address").str.contains(r"^[A-Z]{2}, "),
    "addr_has_unit": pl.col("business_address").str.contains(r"(?i)\b(unit|apt|suite|ste|flat|#)\b"),
    "addr_has_zip5": pl.col("business_address").str.contains(r"\b\d{5}\b"),
    "addr_has_pin6": pl.col("business_address").str.contains(r"\b\d{6}\b"),
    "addr_no_street_num": ~pl.col("business_address").str.contains(r"\d"),
}
rows = []
for n in ["train_source1","train_source2","train_source3"]:
    r = src[n].select([e.mean().alias(k) for k, e in feats.items()]).row(0, named=True)
    rows.append({"src": n, **{k: round(v*100, 2) for k, v in r.items()}})
print(pl.DataFrame(rows).transpose(include_header=True, column_names="src"))

print("\nJunk prefixes (top, train S2+S3):")
for n in ["train_source1","train_source2","train_source3"]:
    print(" ", n, src[n].select(pl.col("business_name").str.extract(r"^([^\p{L}\p{N}]+)", 1).alias("p")).drop_nulls()["p"].str.strip_chars().value_counts(sort=True).head(12).rows())

print("\nLegal-form tokens (last word of name), top 15 per source:")
for n in ["train_source1","train_source2","train_source3"]:
    lw = src[n]["business_name"].str.extract(r"(\S+)\s*$", 1).str.to_lowercase().str.replace_all(r"[.,]", "")
    print(" ", n, lw.value_counts(sort=True).head(15).rows())

print("\n### GROUND TRUTH")
g = gt.with_columns(pl.col("matched_entity_ids").str.split(",").alias("m"))
e = g.explode("m").filter(pl.col("m").is_not_null() & (pl.col("m") != ""))
e = e.with_columns(pl.col("m").str.slice(0,2).alias("tsrc"))
print(f"GT rows={g.height:,} unique S1={g['source1_entity_id'].n_unique():,} total links={e.height:,}")
print("empty match lists:", (g["matched_entity_ids"] == "").sum())
s1ids = set(src["train_source1"]["entity_id"]); gtids = set(g["source1_entity_id"])
print(f"S1 in GT: {len(s1ids & gtids):,}/{len(s1ids):,}; GT ids not in S1: {len(gtids - s1ids):,}")
per = e.group_by("source1_entity_id").agg((pl.col("tsrc")=="S2").sum().alias("n2"), (pl.col("tsrc")=="S3").sum().alias("n3"))
per = g.select("source1_entity_id").join(per, on="source1_entity_id", how="left").fill_null(0)
print("links per S1 (total):", per.select((pl.col("n2")+pl.col("n3")).alias("t"))["t"].value_counts(sort=True).sort("t").rows())
print("S2 matches per S1:", per["n2"].value_counts().sort("n2").rows())
print("S3 matches per S1:", per["n3"].value_counts().sort("n3").rows())
for t, n in [("S2","train_source2"),("S3","train_source3")]:
    ids = set(src[n]["entity_id"]); linked = e.filter(pl.col("tsrc")==t)
    lk = set(linked["m"])
    multi = linked.group_by("m").len().filter(pl.col("len")>1).height
    print(f"{t}: linked={len(lk):,} of {len(ids):,} ({len(lk)/len(ids):.2%}); linked ids missing from source={len(lk-ids):,}; ids linked to >1 S1={multi:,}")

print("\n### TRAIN/TEST OVERLAP")
for i in "123":
    a = set(src[f"train_source{i}"]["entity_id"]); b = set(src[f"test_source{i}"]["entity_id"])
    print(f"S{i}: shared ids={len(a&b):,}")
    ka = src[f"train_source{i}"].select(pl.concat_str(["business_name","business_address"], separator="|")).to_series()
    kb = src[f"test_source{i}"].select(pl.concat_str(["business_name","business_address"], separator="|")).to_series()
    print(f"    shared exact (name|addr)={len(set(ka)&set(kb)):,}")

print("\n### MATCHED-PAIR QUALITY (train)")
s1 = src["train_source1"].rename({c: c+"_1" for c in ["business_name","business_address","country"]})
other = pl.concat([src["train_source2"], src["train_source3"]])
pairs = e.join(s1, left_on="source1_entity_id", right_on="entity_id").join(other, left_on="m", right_on="entity_id")
print("pairs joined:", f"{pairs.height:,}")
print("country agreement by target:", pairs.group_by("tsrc").agg((pl.col("country")==pl.col("country_1")).mean().round(4)).sort("tsrc").rows())
print("country confusion (S1 -> target, mismatches):", pairs.filter(pl.col("country")!=pl.col("country_1")).group_by(["country_1","country"]).len().sort("len", descending=True).head(10).rows())
norm = lambda c: pl.col(c).str.to_lowercase().str.replace_all(r"[^\p{L}\p{N} ]", "").str.replace_all(r"\s+", " ").str.strip_chars()
pairs = pairs.with_columns(
    (pl.col("business_name")==pl.col("business_name_1")).alias("name_exact"),
    (norm("business_name")==norm("business_name_1")).alias("name_norm_eq"),
    (norm("business_address")==norm("business_address_1")).alias("addr_norm_eq"),
)
print(pairs.group_by("tsrc").agg(pl.col("name_exact").mean().round(4), pl.col("name_norm_eq").mean().round(4), pl.col("addr_norm_eq").mean().round(4)).sort("tsrc"))
smp = pairs.sample(20000, seed=0)
def sim(a, b): return SequenceMatcher(None, (a or "").lower(), (b or "").lower()).ratio()
smp = smp.with_columns(
    pl.struct(["business_name","business_name_1"]).map_elements(lambda r: sim(r["business_name"], r["business_name_1"]), return_dtype=pl.Float64).alias("nsim"),
    pl.struct(["business_address","business_address_1"]).map_elements(lambda r: sim(r["business_address"], r["business_address_1"]), return_dtype=pl.Float64).alias("asim"),
)
print("similarity on 20k sampled matched pairs:")
print(smp.group_by("tsrc").agg([pl.col("nsim").quantile(q).round(3).alias(f"name_p{int(q*100)}") for q in (.1,.25,.5)] + [pl.col("asim").quantile(q).round(3).alias(f"addr_p{int(q*100)}") for q in (.1,.25,.5)]).sort("tsrc"))
# random non-match baseline
rnd = s1.sample(20000, seed=1).with_columns(other.sample(20000, seed=2).select(["business_name","business_address"]))
rnd = rnd.with_columns(pl.struct(["business_name","business_name_1"]).map_elements(lambda r: sim(r["business_name"], r["business_name_1"]), return_dtype=pl.Float64).alias("nsim"))
print("random-pair name sim p50/p90/p99:", [round(rnd["nsim"].quantile(q),3) for q in (.5,.9,.99)])
print("\nExample matched groups:")
for sid in g.sample(6, seed=3)["source1_entity_id"]:
    r = s1.filter(pl.col("entity_id")==sid).row(0)
    print(f"  S1 {r[1]!r} | {r[2]!r} | {r[3]}")
    for x in pairs.filter(pl.col("source1_entity_id")==sid).select(["m","business_name","business_address","country"]).rows():
        print(f"     {x[0][:2]} {x[1]!r} | {x[2]!r} | {x[3]}")

# Blocking feasibility: how often do matched pairs share the first name token (normalized)?
ft = lambda c: norm(c).str.extract(r"^(\S+)", 1)
print("\nshare first normalized name token:", pairs.group_by("tsrc").agg((ft("business_name")==ft("business_name_1")).mean().round(4)).sort("tsrc").rows())
