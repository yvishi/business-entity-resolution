"""Prototype of the real blocker (US): partition S1 and pool by canonical state, TF-IDF char-3gram top-k inside
each state, both directions. Runs every S1 of a set of states as queries -> realistic recall + cost per query.
Targets with empty address fall back to a name-only search across the country (reported separately)."""
import sys, time, re, numpy as np, polars as pl
from unidecode import unidecode
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn
from common import load

ST = {"AL":"alabama","AK":"alaska","AZ":"arizona","AR":"arkansas","CA":"california","CO":"colorado","CT":"connecticut","DE":"delaware",
 "FL":"florida","GA":"georgia","HI":"hawaii","ID":"idaho","IL":"illinois","IN":"indiana","IA":"iowa","KS":"kansas","KY":"kentucky",
 "LA":"louisiana","ME":"maine","MD":"maryland","MA":"massachusetts","MI":"michigan","MN":"minnesota","MS":"mississippi","MO":"missouri",
 "MT":"montana","NE":"nebraska","NV":"nevada","NH":"new hampshire","NJ":"new jersey","NM":"new mexico","NY":"new york","NC":"north carolina",
 "ND":"north dakota","OH":"ohio","OK":"oklahoma","OR":"oregon","PA":"pennsylvania","RI":"rhode island","SC":"south carolina",
 "SD":"south dakota","TN":"tennessee","TX":"texas","UT":"utah","VT":"vermont","VA":"virginia","WA":"washington","WV":"west virginia",
 "WI":"wisconsin","WY":"wyoming","DC":"district of columbia"}
NAME2CODE = {v: k for k, v in ST.items()}
def state_of(a):
    for p in reversed([p.strip() for p in a.split(",")]):
        if p.upper() in ST: return p.upper()
        if p.lower() in NAME2CODE: return NAME2CODE[p.lower()]
    return None
STOP = r"\b(inc|incorporated|llc|ltd|limited|pvt|private|corp|corporation|co|company|pc|pllc|llp|lp|plc|the|and|of)\b"
def prep(df):
    f = lambda c, d: (pl.col(c).map_elements(unidecode, return_dtype=pl.Utf8).str.to_lowercase().str.replace_all(r"[^a-z0-9 ]", " ")
                      .pipe(lambda e: e.str.replace_all(STOP, " ") if d else e).str.replace_all(r"\s+", " ").str.strip_chars())
    return df.with_columns((f("business_name", True) + " | " + f("business_address", False)).alias("na"),
                           pl.col("business_address").map_elements(state_of, return_dtype=pl.Utf8).alias("st"))
STATES = sys.argv[1].split(",") if len(sys.argv) > 1 else ["CA", "TX", "OH", "OR", "VT"]
K_FWD, K_REV = int(sys.argv[2]) if len(sys.argv) > 2 else 30, 5
s1 = prep(load("train_source1").filter(pl.col("country") == "US"))
pool = prep(pl.concat([load("train_source2"), load("train_source3")]).filter(pl.col("country") == "US"))
links = load("train_links")
print("S1 without parsable state:", s1["st"].null_count(), " pool without state:", pool["st"].null_count(),
      f"({pool['st'].null_count()/pool.height:.2%}; empty address {(pool['business_address']=='').mean():.2%})")
vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=2, max_df=0.02, dtype=np.float32, sublinear_tf=True)
vec.fit(pool.sample(2_000_000, seed=0)["na"].to_list())
tot_q = tot_t = 0; hit = tot = 0; hit_f = 0; ncand = 0
for st in STATES:
    a = s1.filter(pl.col("st") == st); b = pool.filter(pl.col("st") == st)
    t0 = time.time()
    A, B = vec.transform(a["na"].to_list()), vec.transform(b["na"].to_list())
    F = sp_matmul_topn(A, B.T.tocsr(), top_n=K_FWD, threshold=0.0, n_threads=4)
    R = sp_matmul_topn(B, A.T.tocsr(), top_n=K_REV, threshold=0.0, n_threads=4)
    dt = time.time() - t0
    aid, bid = a["entity_id"].to_numpy(), b["entity_id"].to_numpy()
    fwd = {(aid[i], bid[j]) for i, j in zip(*F.nonzero())}
    rev = {(aid[j], bid[i]) for i, j in zip(*R.nonzero())}
    cand = fwd | rev
    truth = set(links.filter(pl.col("s1").is_in(aid)).select("s1", "t").iter_rows())
    truth_in_state = {p for p in truth if p[1] in set(bid)}
    h = len(truth & cand); hf = len(truth & fwd)
    hit += h; hit_f += hf; tot += len(truth); ncand += len(cand); tot_q += a.height; tot_t += dt
    print(f"{st}: S1={a.height:,} pool={b.height:,}  recall fwd={hf/len(truth):.4f} fwd+rev={h/len(truth):.4f} "
          f"(ceiling in-state {len(truth_in_state)/len(truth):.4f})  cand/S1={len(cand)/a.height:.1f}  {dt:.1f}s", flush=True)
print(f"TOTAL: recall fwd={hit_f/tot:.4f} fwd+rev={hit/tot:.4f}  cand/S1={ncand/tot_q:.1f}  cost {1000*tot_t/tot_q:.2f} ms per S1 query "
      f"-> est. full US train {tot_t/tot_q*s1.height/60:.1f} min on this 4-core box")
