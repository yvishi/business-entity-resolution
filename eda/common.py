"""Shared helpers for EDA scripts. Set DATA_DIR env var to the folder with the Parquet files."""
import os, re, unicodedata
import polars as pl

DATA_DIR = os.environ.get("DATA_DIR", "data/pq")

def load(name):
    return pl.read_parquet(os.path.join(DATA_DIR, f"{name}.parquet"))

def train_pairs():
    """All true (S1, target) pairs with both records' fields."""
    s1 = load("train_source1").rename({"entity_id": "s1", "business_name": "n1", "business_address": "a1"})
    tg = pl.concat([load("train_source2"), load("train_source3")]).rename(
        {"entity_id": "t", "business_name": "n2", "business_address": "a2", "country": "country_t"})
    links = load("train_links")
    return (links.join(s1, on="s1").join(tg, on="t")
                 .with_columns(pl.col("t").str.slice(0, 2).alias("tsrc")))

INDIC = re.compile(r"[ऀ-෿]")          # Devanagari .. Sinhala blocks (all Indic scripts)
NON_LATIN_LETTER = re.compile(r"[^\x00-ɏ\s\d\W]")

def strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

def basic_norm(s):
    s = strip_accents(s.lower())
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def tokens(s):
    return basic_norm(s).split()
