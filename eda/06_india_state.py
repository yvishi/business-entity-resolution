"""Is the Indian state an invariant once its surface forms (codes, native script, variants) are canonicalised?
S1 state = the S1 address part that is a known state/UT name. A surface-form -> state map for target parts is
learned on half of the S1 entities and evaluated on the other half."""
import re, polars as pl
from collections import Counter, defaultdict
from common import load, train_pairs
STATES = ["Andhra Pradesh","Arunachal Pradesh","Assam","Bihar","Chhattisgarh","Goa","Gujarat","Haryana","Himachal Pradesh",
 "Jharkhand","Karnataka","Kerala","Madhya Pradesh","Maharashtra","Manipur","Meghalaya","Mizoram","Nagaland","Odisha","Orissa",
 "Punjab","Rajasthan","Sikkim","Tamil Nadu","Telangana","Tripura","Uttar Pradesh","Uttarakhand","West Bengal","Delhi",
 "Jammu and Kashmir","Jammu & Kashmir","Ladakh","Puducherry","Pondicherry","Chandigarh","Andaman and Nicobar Islands",
 "Dadra and Nagar Haveli","Daman and Diu","Lakshadweep","Uttaranchal"]
SL = {s.lower(): s for s in STATES}
parts = lambda a: [p.strip() for p in a.split(",") if p.strip()]
def s1_state(a):
    for p in parts(a):
        if p.lower() in SL: return SL[p.lower()]
P = train_pairs().filter(pl.col("country") == "India").select("s1", "a1", "a2", "tsrc").with_columns((pl.col("s1").hash() % 2).alias("half"))
rows = [(s1_state(a1), a2, ts, h) for a1, a2, ts, h in P.select("a1", "a2", "tsrc", "half").iter_rows()]
print(f"S1 India pairs with a recognisable state part: {sum(r[0] is not None for r in rows)/len(rows):.4f}")
co = defaultdict(Counter)
for st, a2, ts, h in rows:
    if st and h == 0:
        for p in parts(a2):
            if p.lower() not in SL: co[p][st] += 1
surface = {p: c.most_common(1)[0][0] for p, c in co.items()
           if sum(c.values()) >= 30 and c.most_common(1)[0][1] / sum(c.values()) >= 0.98 and (len(p) <= 3 or re.search(r"[ऀ-෿]", p) or p.lower().rstrip("m") in SL)}
print("learned non-Latin/code/variant surface forms:", len(surface), [(k, v) for k, v in surface.items()][:30])
ok = Counter(); n = Counter()
for st, a2, ts, h in rows:
    if h == 1 and st and a2:
        n[ts] += 1
        if any(p.lower() in SL and SL[p.lower()] == st or surface.get(p) == st or (st in ("Odisha","Orissa") and p.lower() in ("odisha","orissa")) for p in parts(a2)): ok[ts] += 1
for ts in sorted(n): print(f"{ts}: canonical state preserved {ok[ts]/n[ts]:.4f} of {n[ts]:,}")
