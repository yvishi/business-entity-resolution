"""Test audit (inspection only): problems in TEST not covered by what the pipeline learned from TRAIN."""
import os, re, sys, numpy as np, polars as pl
from collections import Counter
W = os.environ["WORK"]
pl.Config.set_tbl_width_chars(200); pl.Config.set_tbl_rows(40)
cols = ["entity_id", "business_name", "business_address", "country", "nm", "core", "ad", "st", "legal", "hn"]
tr = {i: pl.read_parquet(f"{W}/train_s{i}.parquet", columns=cols) for i in "123"}
te = {i: pl.read_parquet(f"{W}/test_s{i}.parquet", columns=cols) for i in "123"}

print("## 1. state parse failures (non-empty address), train vs test")
for i in "123":
    for n, d in (("train", tr[i]), ("test", te[i])):
        x = d.filter(pl.col("country").is_in(["US", "India"]) & (pl.col("business_address") != ""))
        print(f"  S{i} {n}: {x.filter(pl.col('st').is_null()).height:,} / {x.height:,} "
              f"({x['st'].null_count()/max(1,x.height):.3%})", x.filter(pl.col("st").is_null())["business_address"].head(3).to_list())

print("\n## 2. scripts / characters in test never seen in train")
def chars(df): return Counter("".join(df["business_name"].to_list() + df["business_address"].to_list()))
ctr = Counter(); cte = Counter()
for i in "123": ctr.update(chars(tr[i].sample(min(1_000_000, tr[i].height), seed=0))); cte.update(chars(te[i].sample(min(1_000_000, te[i].height), seed=0)))
new = [(c, n, hex(ord(c))) for c, n in cte.most_common() if c not in ctr][:40]
print("  new chars (char, count in 1M-sample, codepoint):", new)

print("\n## 3. name-token novelty: share of normalised name tokens never seen in train, by country/source")
vocab = set()
for i in "123": vocab |= set(tr[i]["nm"].str.split(" ").explode().unique().to_list())
for i in "123":
    for c in ("US", "India", "France"):
        t = te[i].filter(pl.col("country") == c)["nm"].str.split(" ").explode()
        print(f"  S{i} {c}: {1 - t.is_in(pl.Series(list(vocab)).implode()).mean():.3%} OOV tokens")

print("\n## 4. France: legal form detection, core-name, address state-less, fillers")
for i in "123":
    f = te[i].filter(pl.col("country") == "France")
    print(f"  S{i}: legal detected {(f['legal'] != '').mean():.1%} | empty addr {(f['business_address']=='').mean():.1%} "
          f"| house# found {(f['hn'] != '').mean():.1%} | core==nm {(f['core']==f['nm']).mean():.1%}")
    print("     unrecognised last name tokens:", Counter(t.split()[-1] for t in f["nm"].to_list() if t).most_common(15))
    print("     address tokens (top 25):", Counter(t for a in f["ad"].sample(100000, seed=1).to_list() for t in a.split() if not t.isdigit()).most_common(25))

print("\n## 5. France partition size (single partition -> blocking cost)")
print("  S1", te["1"].filter(pl.col("country") == "France").height, "pool", sum(te[i].filter(pl.col("country") == "France").height for i in "23"))
for i in "123":
    f = te[i].filter(pl.col("country") == "France")
    reg = f["business_address"].str.split(",").list.last().str.strip_chars()
    print(f"  S{i} last address part (region/dept/city) top:", Counter(reg.to_list()).most_common(8))

print("\n## 6. patterns frequent in test names/addresses but rare in train (per 100k records)")
pats = {"N° prefix": r"N°", "bis/ter": r"\b(bis|ter|BIS|TER)\b", "(digits)": r"\(\d+\)", "+ in name": r"\+", "& in name": r"&",
        "leading legal form": r"^(SARL|SAS|SA|EURL|SCI|SASU|EI|Ets|ETS)\b", "Cie": r"\b(Cie|CIE|& Cie)\b", "(France)": r"\(France\)",
        "apostrophe ’": r"’", "zero-padded number": r"\b0\d{2,}\b", "domain": r"\.(com|fr|net|org)$", "@handle": r"^@",
        "all caps address": r"^[^a-z]*[A-Z][^a-z]*$", "comma-less address": r"^[^,]+$"}
rows = []
for name, p in pats.items():
    r = {"pattern": name}
    for split, dd in (("train", tr), ("test", te)):
        for c in ("US", "India", "France"):
            if split == "train" and c == "France": continue
            x = pl.concat([dd[i].filter(pl.col("country") == c) for i in "23"])
            col = "business_address" if "address" in name else "business_name"
            if name in ("N° prefix", "bis/ter", "(digits)", "zero-padded number"): col = "business_address"
            r[f"{split}_{c}"] = round(x[col].str.contains(p).mean() * 100, 2)
    rows.append(r)
print(pl.DataFrame(rows))
