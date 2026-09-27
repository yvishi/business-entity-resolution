"""Error analysis on out-of-fold (OOF) predictions of the pipeline (train, labelled).

Loss attribution: for every S1 in the validation universe, how much F0.5 would be regained by fixing
  (a) blocking misses   – true targets that never reached the candidate set
  (b) model misses      – true targets in the candidate set that were not predicted
  (c) false positives   – predicted targets that are wrong (split: distractor vs. member of another S1's cluster)
Then describe each bucket with the corruption operators of eda/01, show examples, and re-tune the decision
rule at test-like density (19% of S1 hidden -> their records become distractors).

Usage: WORK=<work dir with oof.parquet> RAW=<raw parquet dir> python eda/11_error_analysis.py
"""
import os, sys, json, re
import numpy as np, polars as pl
from collections import Counter, defaultdict
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code", "business_entity_resolution", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from ber.decide import unique_map, select, f05

W, RAW = os.environ["WORK"], os.environ["RAW"]
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(220); pl.Config.set_fmt_str_lengths(70)

rep = json.load(open(f"{W}/train_report.json"))
oof = pl.read_parquet(f"{W}/oof.parquet")                      # s1, t, y, p1, p2
stage = 2 if rep["f2"] >= rep["f1"] else 1
mode, tau, uniq = rep["best2"] if stage == 2 else rep["best1"]
d = oof.with_columns(pl.col("p2" if stage == 2 else "p1").alias("p"))
countries = [c for c in ("US", "India") if os.path.exists(f"{W}/s1ids_train_{c}.parquet")]
ids = pl.concat([pl.read_parquet(f"{W}/s1ids_train_{c}.parquet") for c in countries])["s1"]
s1raw = pl.read_parquet(f"{RAW}/train_source1.parquet").filter(pl.col("entity_id").is_in(ids.implode()))
country_of = dict(zip(s1raw["entity_id"].to_list(), s1raw["country"].cast(pl.Utf8).to_list()))
links = pl.read_parquet(f"{RAW}/train_links.parquet")
truth = defaultdict(set)
for x, t in links.filter(pl.col("s1").is_in(ids.implode())).iter_rows(): truth[x].add(t)
owner = dict(zip(links["t"].to_list(), links["s1"].to_list()))      # target -> its true S1 (any S1 in train)
C = defaultdict(set)          # candidate set = exactly the pairs the model scored (OOF rows)
for x, t in oof.select("s1", "t").iter_rows(): C[x].add(t)
ids = ids.to_list()
print(f"stage {stage}, decision {(mode, tau, uniq)}; S1={len(ids):,}; pairs={d.height:,}")

pred = select(unique_map(d) if uniq == "uniq" else d, mode=mode, tau=tau)
base = np.array([f05(pred.get(x, set()), truth.get(x, set())) for x in ids])
print(f"\n### OOF macro F0.5 = {base.mean():.4f}   (per country: " +
      ", ".join(f"{c} {base[[country_of[x] == c for x in ids]].mean():.4f}" for c in countries) + ")")

# ---------- loss attribution (counterfactual fixes, one bucket at a time)
def fix(kind):
    out = []
    for x in ids:
        T, P = truth.get(x, set()), set(pred.get(x, set()))
        if kind == "block":      P |= (T - C[x])
        elif kind == "modelFN":  P |= (T & C[x]) - P
        elif kind == "FP_distr": P -= {t for t in P - T if t not in owner}
        elif kind == "FP_other": P -= {t for t in P - T if t in owner}
        out.append(f05(P, T))
    return np.array(out)
print("\n### Loss attribution: F0.5 regained if a bucket were fixed perfectly (overall / US / India)")
deficit = 1 - base.mean()
for k, name in (("block", "blocking misses (true target not in candidates)"), ("modelFN", "model misses (candidate, not predicted)"),
                ("FP_distr", "false positives: distractor records (belong to no S1)"),
                ("FP_other", "false positives: records of ANOTHER S1's cluster")):
    g = fix(k)
    per = ", ".join(f"{c} +{(g - base)[[country_of[x] == c for x in ids]].mean():.4f}" for c in countries)
    print(f"  {name:60s} +{(g - base).mean():.4f}  ({(g - base).mean() / deficit:5.1%} of deficit)  [{per}]")
sing = [i for i, x in enumerate(ids) if not truth.get(x)]
print(f"  singleton S1: {len(sing):,} ({len(sing)/len(ids):.2%}); wrongly non-empty: {sum(1 for i in sing if pred.get(ids[i])):,}; "
      f"non-singletons predicted empty: {sum(1 for x in ids if truth.get(x) and not pred.get(x)):,}")

# ---------- describe buckets with pair operators
tg = pl.concat([pl.read_parquet(f"{RAW}/train_source{i}.parquet") for i in "23"])
rec = {**dict(zip(s1raw["entity_id"].to_list(), zip(s1raw["business_name"].to_list(), s1raw["business_address"].to_list())))}
need = set()
rows = {"block": [], "modelFN": [], "FP_distr": [], "FP_other": []}
for x in ids:
    T, P = truth.get(x, set()), pred.get(x, set())
    for t in T - C[x]: rows["block"].append((x, t))
    for t in (T & C[x]) - P: rows["modelFN"].append((x, t))
    for t in P - T: rows["FP_other" if t in owner else "FP_distr"].append((x, t))
for v in rows.values(): need |= {t for _, t in v}
tsub = tg.filter(pl.col("entity_id").is_in(list(need)))
rec.update(dict(zip(tsub["entity_id"].to_list(), zip(tsub["business_name"].to_list(), tsub["business_address"].to_list()))))
# operators from the noise catalogue
import importlib.util
spec = importlib.util.spec_from_file_location("cat", os.path.join(os.path.dirname(__file__), "01_noise_catalogue.py"))
src = open(spec.origin).read().split("res = defaultdict(Counter)")[0].replace("P = train_pairs().sample(400_000, seed=0)", "")
ns = {}; exec(src, ns)
name_ops, addr_ops = ns["name_ops"], ns["addr_ops"]
print("\n### Operator profile per bucket (% of pairs in the bucket); 'all true pairs' = reference")
ref = [(x, t) for x in ids for t in truth.get(x, ())]
rng = np.random.default_rng(0)
def prof(pairs, n=20000):
    if not pairs: return {}
    pairs = [pairs[i] for i in rng.choice(len(pairs), min(n, len(pairs)), replace=False)]
    cnt = Counter()
    for x, t in pairs:
        if t not in rec: continue
        n1, a1 = rec[x]; n2, a2 = rec[t]
        for o in name_ops(n1, n2): cnt["name:" + o] += 1
        for o in addr_ops(a1, a2, country_of[x]): cnt["addr:" + o] += 1
        cnt["src:" + t[:2]] += 1; cnt["country:" + country_of[x]] += 1
    return {k: round(100 * v / len(pairs), 1) for k, v in cnt.items()}
refs = [p for p in ref if p[1] in rec] if len(ref) < 50000 else None
tsub2 = tg.filter(pl.col("entity_id").is_in([t for _, t in (ref if len(ref) < 60000 else [ref[i] for i in rng.choice(len(ref), 60000, replace=False)])]))
rec.update(dict(zip(tsub2["entity_id"].to_list(), zip(tsub2["business_name"].to_list(), tsub2["business_address"].to_list()))))
ref_s = [p for p in ref if p[1] in rec]
profiles = {"all true pairs": prof(ref_s)}
for k in rows: profiles[k] = prof(rows[k])
keys = sorted(set().union(*[set(v) for v in profiles.values()]))
tab = pl.DataFrame([{"operator": k, **{b: profiles[b].get(k, 0.0) for b in profiles}} for k in keys])
print(tab.with_columns((pl.col("block") - pl.col("all true pairs")).alias("block-ref"),
                       (pl.col("modelFN") - pl.col("all true pairs")).alias("mFN-ref")).sort("block-ref", descending=True))
print({k: len(v) for k, v in rows.items()})

print("\n### Examples")
for k, v in rows.items():
    print(f"\n-- {k} --")
    for i in rng.choice(len(v), min(12, len(v)), replace=False) if v else []:
        x, t = v[i]
        extra = f"   [true owner of target: {rec.get(owner[t], ('?', '?'))[0]!r}]" if k == "FP_other" else ""
        print(f"  S1 {rec[x][0]!r:45} | {rec[x][1][:70]!r}\n     {t[:2]} {rec[t][0]!r:45} | {rec[t][1][:70]!r}{extra}")

# ---------- decision re-tuning at test-like density (hide 19% of S1; their records become distractors)
print("\n### Decision rules: normal density vs test-like density (19% of S1 hidden)")
hide = set(pl.Series(ids).filter(pl.Series(ids).hash(seed=11) % 100 < 19).to_list())
keep_ids = [x for x in ids if x not in hide]
dd = d.filter(~pl.col("s1").is_in(list(hide)))
res = []
for u in ("uniq", "raw"):
    src_n, src_d = (unique_map(d), unique_map(dd)) if u == "uniq" else (d, dd)
    for m, taus in (("thr", (0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9)), ("expf", (0.0, 0.2, 0.3, 0.4, 0.5, 0.6))):
        for tv in taus:
            pn = select(src_n, mode=m, tau=tv); pd_ = select(src_d, mode=m, tau=tv)
            res.append({"rule": f"{m},{tv},{u}",
                        "F_normal": round(np.mean([f05(pn.get(x, set()), truth.get(x, set())) for x in ids]), 4),
                        "F_dense": round(np.mean([f05(pd_.get(x, set()), truth.get(x, set())) for x in keep_ids]), 4)})
print(pl.DataFrame(res).sort("F_dense", descending=True).head(15))
