"""Profile the French TEST records (inspection only - nothing here is used to train a model).
Which street-type abbreviations, legal forms, region/department tokens and noise patterns appear per source."""
import re, polars as pl
from collections import Counter
from common import load
pl.Config.set_tbl_width_chars(200)
for i in "123":
    df = load(f"test_source{i}").filter(pl.col("country") == "France")
    n, a = df["business_name"].to_list(), df["business_address"].to_list()
    print(f"\n=== test_source{i} France: {df.height:,} rows")
    first = Counter(); last = Counter(); streettype = Counter(); legal = Counter(); parts_n = Counter()
    for x in a:
        ps = [p.strip() for p in x.split(",") if p.strip()]
        parts_n[len(ps)] += 1
        if ps: last[ps[-1]] += 1
        m = re.search(r"\b\d+\s*(?:bis|ter)?\s+([A-Za-zÀ-ÿ\.°]+)", x)
        if m: streettype[m.group(1).lower()] += 1
    for x in n:
        t = x.split()
        if t: legal[t[-1].lower()] += 1
    print(" addr #parts:", sorted(parts_n.items())[:8])
    print(" street-type token after number (top 25):", streettype.most_common(25))
    print(" last address part (top 15):", last.most_common(15))
    print(" name last token (top 20):", legal.most_common(20))
    print(" samples:", df.sample(6, seed=int(i)).select("business_name", "business_address").rows())
