"""Catalogue the corruption operators by diffing true (S1 -> S2/S3) pairs.
Each pair can carry several operators; we report the % of pairs showing each, by target source and country."""
import re, random, polars as pl
from collections import Counter, defaultdict
from rapidfuzz.distance import Levenshtein
from common import *

random.seed(0)
P = train_pairs().sample(400_000, seed=0)
LEGAL = {"inc","incorporated","llc","ltd","limited","corp","corporation","co","company","pc","pllc","llp","lp",
         "pvt","private","plc","pa","p","l","c","sa","sarl","sas","group","holdings"}
GENERIC = {"center","centre","services","service","group","holdings","co","solutions","enterprises","partners",
           "company","associates","international","india","global","the","and","of"}
US_STATES = {"alabama","alaska","arizona","arkansas","california","colorado","connecticut","delaware","florida","georgia",
 "hawaii","idaho","illinois","indiana","iowa","kansas","kentucky","louisiana","maine","maryland","massachusetts","michigan",
 "minnesota","mississippi","missouri","montana","nebraska","nevada","new hampshire","new jersey","new mexico","new york",
 "north carolina","north dakota","ohio","oklahoma","oregon","pennsylvania","rhode island","south carolina","south dakota",
 "tennessee","texas","utah","vermont","virginia","washington","west virginia","wisconsin","wyoming","district of columbia"}
VOWELS = re.compile(r"[aeiou]")

def name_ops(n1, n2):
    ops = []
    if n1 == n2: return ["exact"]
    letters2 = re.sub(r"[^A-Za-z]", "", n2)
    if letters2 and letters2.isupper() and not re.sub(r"[^A-Za-z]", "", n1).isupper(): ops.append("upper")
    if letters2 and letters2.islower() and not re.sub(r"[^A-Za-z]", "", n1).islower(): ops.append("lower")
    m = re.match(r"^([^\w\s]+)\s*", n2)
    if m and not re.match(r"^[^\w\s]", n1): ops.append("junk_prefix")
    if re.search(r"[\[\]]", n2) and not re.search(r"[\[\]]", n1): ops.append("square_brackets")
    if re.search(r"\(", n2) and not re.search(r"\(", n1): ops.append("parentheses_added")
    if INDIC.search(n2): ops.append("indic_script")
    if re.search(r"(?i)\.(com|net|org|in|co|biz|info)$", n2.strip()): ops.append("domain")
    if n2.strip().startswith("@"): ops.append("handle")
    if re.search(r"[À-ÿ]", n2) and not re.search(r"[À-ÿ]", n1): ops.append("accent_added")
    if re.search(r"[A-Za-z]\d|\d[A-Za-z]", n2) and not re.search(r"[A-Za-z]\d|\d[A-Za-z]", n1): ops.append("leet_digit")
    t1, t2 = tokens(n1), tokens(n2)
    if basic_norm(n1) == basic_norm(n2):
        ops.append("cosmetic_only"); return ops
    s1, s2 = set(t1), set(t2)
    if s1 == s2 and t1 != t2: ops.append("token_shuffle")
    if not (s1 & s2) and not INDIC.search(n2) and not re.search(r"(?i)\.(com|net|org|in)$|^@", n2.strip()):
        ops.append("no_shared_token")
    added, removed = s2 - s1, s1 - s2
    if added & LEGAL: ops.append("legal_added")
    if removed & LEGAL: ops.append("legal_removed")
    if added and added <= GENERIC | LEGAL: ops.append("only_generic_added")
    if removed and removed <= GENERIC | LEGAL: ops.append("only_generic_removed")
    # typo: some removed token has an added token within edit distance 1-2
    typo = any(Levenshtein.distance(a, b) <= 2 for a in removed for b in added if len(a) > 3 and len(b) > 3 and a[0] == b[0] or (len(a) > 3 and len(b) > 3 and Levenshtein.distance(a, b) == 1))
    if typo: ops.append("typo")
    return ops

NUM = re.compile(r"\d+")
def house_number(a):
    m = NUM.search(a); return m.group() if m else None

def addr_ops(a1, a2, country):
    ops = []
    if not a2.strip(): return ["empty"]
    if a1 == a2: return ["exact"]
    l2 = re.sub(r"[^A-Za-z]", "", a2)
    if l2 and l2.isupper(): ops.append("upper")
    p1 = [basic_norm(p) for p in a1.split(",") if p.strip()]; p2 = [basic_norm(p) for p in a2.split(",") if p.strip()]
    if set(p1) == set(p2) and p1 != p2: ops.append("parts_rotated_only")
    elif p1 and p2 and p1[0] != p2[0] and set(p1) & set(p2): ops.append("parts_reordered")
    if len(p2) < len(p1): ops.append("fewer_parts")
    if len(p2) > len(p1): ops.append("more_parts")
    if re.search(r"\bN/A\b", a2) and not re.search(r"\bN/A\b", a1): ops.append("na_token")
    if INDIC.search(a2): ops.append("indic_script")
    if re.search(r"(?i)\b(near|opp|opposite|behind|beside|nr)\b", a2) and not re.search(r"(?i)\b(near|opp|opposite|behind|beside|nr)\b", a1): ops.append("landmark_added")
    l2l = a2.lower()
    if country == "US":
        if any(st in l2l for st in US_STATES) and not any(st in a1.lower() for st in US_STATES): ops.append("state_spelled_out")
    # house number
    h1 = house_number(a1); nums2 = NUM.findall(a2)
    if h1:
        if not nums2: ops.append("hn_missing")
        elif h1 in nums2: ops.append("hn_same")
        elif any(n.lstrip("0") == h1 and n.startswith("0") for n in nums2): ops.append("hn_zero_padded")
        elif any(n == h1[1:] for n in nums2) and len(h1) > 1: ops.append("hn_leading_digit_dropped")
        elif any(len(n) == len(h1) and sum(x != y for x, y in zip(n, h1)) == 1 for n in nums2): ops.append("hn_one_digit_changed")
        elif any(h1 in n for n in nums2): ops.append("hn_extended")
        else: ops.append("hn_other")
    # vowel-dropped abbreviations: token in a2 equals a1 token without vowels (len>=3)
    t1 = set(tokens(a1)); t2 = set(tokens(a2))
    devow = {VOWELS.sub("", t[0]) + VOWELS.sub("", t[1:]) if False else t[0] + VOWELS.sub("", t[1:]) for t in t1 if len(t) > 4}
    if (t2 - t1) & devow: ops.append("vowel_dropped_abbrev")
    if not (t1 & t2): ops.append("no_shared_token")
    return ops

res = defaultdict(Counter); cnt = Counter(); nops_name = defaultdict(Counter)
for n1, a1, n2, a2, c, ts in P.select("n1","a1","n2","a2","country","tsrc").iter_rows():
    k = (ts, c); cnt[k] += 1
    no = name_ops(n1, n2)
    for o in no: res[("name", o)][k] += 1
    nops_name[k][min(len([o for o in no if o not in ("exact",)]), 5)] += 1
    for o in addr_ops(a1, a2, c): res[("addr", o)][k] += 1

keys = sorted(cnt)
print("pairs sampled:", {f"{a}-{b}": cnt[(a, b)] for a, b in keys})
rows = []
for (field, op), c in sorted(res.items()):
    rows.append({"field": field, "op": op, **{f"{a}-{b}": round(100 * c[(a, b)] / cnt[(a, b)], 2) for a, b in keys}})
pl.Config.set_tbl_rows(100); pl.Config.set_tbl_width_chars(200)
print(pl.DataFrame(rows))
print("\n# name operators per pair (0..5+):")
for k in keys: print(" ", k, {i: round(100 * nops_name[k][i] / cnt[k], 1) for i in range(6)})

# examples per op (few) for the report
ex = defaultdict(list)
for n1, a1, n2, a2, c, ts in P.head(60000).select("n1","a1","n2","a2","country","tsrc").iter_rows():
    for o in name_ops(n1, n2):
        if len(ex[("name", o)]) < 3: ex[("name", o)].append((n1, n2))
    for o in addr_ops(a1, a2, c):
        if len(ex[("addr", o)]) < 2: ex[("addr", o)].append((a1, a2))
print("\n# examples")
for k, v in sorted(ex.items()):
    for e in v: print(f"  {k[0]}:{k[1]:24} {e[0]!r:60} -> {e[1]!r}")
