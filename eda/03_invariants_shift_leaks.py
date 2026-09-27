"""(a) cluster anatomy by country, (b) leak audit (row order / ID numbers), (c) field invariants in true pairs,
(d) train-vs-test single-record profile shift (incl. France)."""
import re, numpy as np, polars as pl
from rapidfuzz import fuzz
from unidecode import unidecode
from common import load, train_pairs, INDIC
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(220)

s1 = load("train_source1"); links = load("train_links"); gt = load("train_ground_truth")
# ---------- (a) cluster anatomy
g = gt.join(s1.select(pl.col("entity_id").alias("source1_entity_id"), "country"), on="source1_entity_id")
cnt = links.with_columns(pl.col("t").str.slice(0, 2).alias("src")).group_by("s1").agg(
    (pl.col("src") == "S2").sum().alias("n2"), (pl.col("src") == "S3").sum().alias("n3"))
g = g.join(cnt, left_on="source1_entity_id", right_on="s1", how="left").fill_null(0).with_columns((pl.col("n2") + pl.col("n3")).alias("n"))
print("# (a) cluster size by country")
print(g.group_by("country").agg(pl.len(), (pl.col("n") == 0).mean().alias("singleton_rate"), pl.col("n").mean().alias("mean_n"),
      (pl.col("n2") == 0).mean().alias("no_S2"), (pl.col("n3") == 0).mean().alias("no_S3"), pl.col("n").quantile(0.99).alias("p99")))

# ---------- (b) leak audit
print("\n# (b) leak audit")
s2 = load("train_source2").with_row_index("row2"); 
lk = links.join(s1.with_row_index("row1").select(pl.col("entity_id").alias("s1"), "row1"), on="s1").join(
    s2.select(pl.col("entity_id").alias("t"), "row2"), on="t")
r = np.corrcoef(lk["row1"].to_numpy(), lk["row2"].to_numpy())[0, 1]
print(f"row-order corr S1 row vs S2 row of matches: {r:.4f}")
num = lambda c: pl.col(c).str.slice(3).cast(pl.Int64)
lk2 = lk.with_columns(num("s1").alias("a"), num("t").alias("b"))
print(f"ID-number corr S1 vs S2 match: {np.corrcoef(lk2['a'].to_numpy(), lk2['b'].to_numpy())[0,1]:.4f}")
# adjacency: do members of a cluster sit next to each other in the S2 file?
adj = lk.group_by("s1").agg(pl.col("row2").sort().diff().min().alias("d")).drop_nulls()
print(f"clusters with >=2 S2 members whose closest pair of rows is adjacent (diff==1): {(adj['d']==1).mean():.4f}")
print("S1 id number range:", s1["entity_id"].str.slice(3).cast(pl.Int64).min(), s1["entity_id"].str.slice(3).cast(pl.Int64).max())

# ---------- (c) invariants
print("\n# (c) invariants in true pairs (share of pairs where S1 field survives in target)")
P = train_pairs().sample(300_000, seed=2)
US_ST = {"AL":"alabama","AK":"alaska","AZ":"arizona","AR":"arkansas","CA":"california","CO":"colorado","CT":"connecticut","DE":"delaware",
 "FL":"florida","GA":"georgia","HI":"hawaii","ID":"idaho","IL":"illinois","IN":"indiana","IA":"iowa","KS":"kansas","KY":"kentucky",
 "LA":"louisiana","ME":"maine","MD":"maryland","MA":"massachusetts","MI":"michigan","MN":"minnesota","MS":"mississippi","MO":"missouri",
 "MT":"montana","NE":"nebraska","NV":"nevada","NH":"new hampshire","NJ":"new jersey","NM":"new mexico","NY":"new york","NC":"north carolina",
 "ND":"north dakota","OH":"ohio","OK":"oklahoma","OR":"oregon","PA":"pennsylvania","RI":"rhode island","SC":"south carolina",
 "SD":"south dakota","TN":"tennessee","TX":"texas","UT":"utah","VT":"vermont","VA":"virginia","WA":"washington","WV":"west virginia",
 "WI":"wisconsin","WY":"wyoming","DC":"district of columbia"}
def parts(a): return [p.strip() for p in a.split(",") if p.strip()]
stats = {}
def add(k, cond):
    if cond is None: return
    a, b = stats.get(k, (0, 0)); stats[k] = (a + bool(cond), b + 1)
for n1, a1, n2, a2, c, ts in P.select("n1", "a1", "n2", "a2", "country", "tsrc").iter_rows():
    if not a2: continue
    A2 = unidecode(a2).lower(); p1 = parts(a1)
    key = f"{c}-{ts}"
    if c == "US":
        st = [p for p in p1 if p in US_ST]
        if st: add(f"{key} state(code or name)", st[0].lower() in re.findall(r"[a-z]+", A2) or US_ST[st[0]] in A2)
        cities = [p for p in p1 if not re.search(r"\d", p) and p not in US_ST and not p.lower().startswith("unit")]
        if cities:
            add(f"{key} city exact substring", cities[-1].lower() in A2)
            add(f"{key} city fuzzy>=85", fuzz.partial_ratio(cities[-1].lower(), A2) >= 85)
        z = re.findall(r"\b\d{5}\b", a1)
        if z: add(f"{key} zip5", z[-1] in a2)
        streetp = [p for p in p1 if re.match(r"^\d+\s+\S", p)]
        if streetp:
            m = re.match(r"^(\d+)\s+(.*)$", streetp[0]); hn, street = m.group(1), m.group(2).lower()
            core = max(re.findall(r"[a-z]+", street) or [""], key=len)
            if core:
                add(f"{key} street core token exact", core in A2)
                add(f"{key} street core token fuzzy>=80", fuzz.partial_ratio(core, A2) >= 80)
            add(f"{key} house# (exact or zero-pad)", any(x.lstrip("0") == hn for x in re.findall(r"\d+", a2)))
            add(f"{key} house# last2 digits", any(x.lstrip("0")[-2:] == hn[-2:] for x in re.findall(r"\d+", a2)))
    else:
        pin = re.findall(r"\b\d{6}\b", a1)
        if pin: add(f"{key} PIN6", pin[-1] in a2)
        last = p1[-1].lower() if p1 else ""
        add(f"{key} state (latin, exact)", last in A2)
        tok = [t for t in re.findall(r"[a-z]{4,}", unidecode(a1).lower())]
        if tok:
            add(f"{key} >=1 address word(len>=4) shared", any(t in A2 for t in tok))
            add(f"{key} >=50% address words shared", np.mean([t in A2 for t in tok]) >= 0.5)
    # name
    nn1 = set(re.findall(r"[a-z]{3,}", unidecode(n1).lower())) - {"inc","llc","ltd","limited","private","pvt","corp","the","and","group","company"}
    nn2 = unidecode(n2).lower()
    if nn1:
        add(f"{key} name >=1 core word shared (substring, unidecoded)", any(t in nn2 for t in nn1))
        add(f"{key} name first core word (substring)", sorted(nn1, key=lambda t: n1.lower().find(t))[0] in nn2)
rows = [{"invariant": k, "rate": round(a / b, 4), "n": b} for k, (a, b) in sorted(stats.items())]
print(pl.DataFrame(rows))

# ---------- (d) shift profile
print("\n# (d) single-record profile, train vs test")
def prof(df, tag):
    return df.group_by("country").agg(
        pl.len().alias("rows"),
        (pl.col("business_name").str.contains(r"[ऀ-෿]").mean() * 100).round(2).alias("indic_name%"),
        (pl.col("business_address").str.contains(r"[ऀ-෿]").mean() * 100).round(2).alias("indic_addr%"),
        (pl.col("business_name").str.contains(r"^[^\w\s]").mean() * 100).round(2).alias("junk%"),
        (pl.col("business_name").str.contains(r"(?i)\.(com|net|org|in|fr)$").mean() * 100).round(2).alias("domain%"),
        ((pl.col("business_address") == "").mean() * 100).round(2).alias("empty_addr%"),
        ((pl.col("business_address").str.to_uppercase() == pl.col("business_address")).mean() * 100).round(2).alias("upper_addr%"),
        (pl.col("business_name").str.contains(r"[À-ÿ]").mean() * 100).round(2).alias("accent%"),
        pl.col("business_name").str.len_chars().mean().round(1).alias("name_len"),
        pl.col("business_address").str.len_chars().mean().round(1).alias("addr_len"),
    ).with_columns(pl.lit(tag).alias("file")).sort("country")
out = []
for sp in ["train", "test"]:
    for i in "123":
        out.append(prof(load(f"{sp}_source{i}").with_columns(pl.col("country").cast(pl.Utf8)), f"{sp}_S{i}"))
print(pl.concat(out).select("file", pl.all().exclude("file")))
