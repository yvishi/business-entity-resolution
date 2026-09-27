"""Record normalisation: every view here inverts a corruption operator measured in the EDA (docs/eda_report.md §2)."""
import re, unicodedata
import polars as pl
from unidecode import unidecode

LEGAL = ["private limited", "pvt ltd", "pvt", "private", "limited", "ltd", "llc", "l l c", "inc", "incorporated", "corp",
         "corporation", "co", "company", "pc", "pllc", "llp", "lp", "plc", "pa", "sarl", "sas", "sasu", "eurl", "sa", "sci",
         "ei", "cie", "ets", "etablissements", "s a s", "s a r l"]
GENERIC = ["the", "and", "of", "services", "service", "center", "centre", "group", "holdings", "solutions", "enterprises",
           "partners", "associates", "international", "india", "france", "dr", "m s", "ms"]
LEGAL_RE = r"\b(" + "|".join(sorted(LEGAL, key=len, reverse=True)) + r")\b"
GENERIC_RE = r"\b(" + "|".join(GENERIC) + r")\b"
TLD_RE = r"\b(com|net|org|in|co|biz|info|fr)\s*$"

US_STATES = {"AL":"alabama","AK":"alaska","AZ":"arizona","AR":"arkansas","CA":"california","CO":"colorado","CT":"connecticut",
 "DE":"delaware","FL":"florida","GA":"georgia","HI":"hawaii","ID":"idaho","IL":"illinois","IN":"indiana","IA":"iowa","KS":"kansas",
 "KY":"kentucky","LA":"louisiana","ME":"maine","MD":"maryland","MA":"massachusetts","MI":"michigan","MN":"minnesota",
 "MS":"mississippi","MO":"missouri","MT":"montana","NE":"nebraska","NV":"nevada","NH":"new hampshire","NJ":"new jersey",
 "NM":"new mexico","NY":"new york","NC":"north carolina","ND":"north dakota","OH":"ohio","OK":"oklahoma","OR":"oregon",
 "PA":"pennsylvania","RI":"rhode island","SC":"south carolina","SD":"south dakota","TN":"tennessee","TX":"texas","UT":"utah",
 "VT":"vermont","VA":"virginia","WA":"washington","WV":"west virginia","WI":"wisconsin","WY":"wyoming","DC":"district of columbia",
 "PR":"puerto rico"}
US_NAME2CODE = {v: k for k, v in US_STATES.items()}
IN_STATES = ["Andhra Pradesh","Arunachal Pradesh","Assam","Bihar","Chhattisgarh","Goa","Gujarat","Haryana","Himachal Pradesh",
 "Jharkhand","Karnataka","Kerala","Madhya Pradesh","Maharashtra","Manipur","Meghalaya","Mizoram","Nagaland","Odisha","Orissa",
 "Punjab","Rajasthan","Sikkim","Tamil Nadu","Telangana","Tripura","Uttar Pradesh","Uttarakhand","Uttaranchal","West Bengal",
 "Delhi","Jammu and Kashmir","Jammu & Kashmir","Ladakh","Puducherry","Pondicherry","Chandigarh","Andaman and Nicobar Islands",
 "Dadra and Nagar Haveli","Daman and Diu","Lakshadweep"]
IN_CANON = {s.lower(): s for s in IN_STATES}
IN_CANON.update({"orissa": "Odisha", "odisha": "Odisha", "pondicherry": "Puducherry", "uttaranchal": "Uttarakhand",
                 "jammu & kashmir": "Jammu and Kashmir"})


INDIC_RE = re.compile(r"[\u0900-\u0DFF]")
TRANSLIT = {}   # native-script word -> latin word, learned from train pairs (set by set_translit)


def set_translit(d):
    TRANSLIT.clear(); TRANSLIT.update(d)


# ---- France (test-only country): region canonicalisation + address/name normalisation, no training involved
FR_REGIONS = {"hauts-de-france": "HDF", "hauts de france": "HDF", "nord": "HDF", "pas-de-calais": "HDF", "pas de calais": "HDF",
              "somme": "HDF", "aisne": "HDF", "oise": "HDF",
              "nouvelle-aquitaine": "NAQ", "nouvelle aquitaine": "NAQ", "gironde": "NAQ", "landes": "NAQ", "charente-maritime": "NAQ",
              "dordogne": "NAQ", "lot-et-garonne": "NAQ", "pyrenees-atlantiques": "NAQ",
              "pays de la loire": "PDL", "pays-de-la-loire": "PDL", "loire-atlantique": "PDL", "loire atlantique": "PDL",
              "vendee": "PDL", "maine-et-loire": "PDL", "sarthe": "PDL", "mayenne": "PDL"}
FR_REGION_WORDS = r"\b(hauts de france|nouvelle aquitaine|pays de la loire|loire atlantique|pas de calais|gironde|nord|france)\b"
FR_STREET = [(r"\br\b", "rue"), (r"\bav\b|\bave\b", "avenue"), (r"\ball\b", "allee"), (r"\bbd\b|\bblvd\b", "boulevard"),
             (r"\bimp\b", "impasse"), (r"\bpl\b", "place"), (r"\brte\b", "route"), (r"\bch\b|\bchem\b", "chemin"),
             (r"\bsq\b", "square"), (r"\bste?\b", "saint"), (r"\bqu\b", "quai"), (r"\bn\b|\bno\b", " ")]
FR_NAME_STOP = r"\b(groupe|group|fils|developpement|compagnie|cie|ets|etablissements|et|de|du|des|la|le|les|l|d|france|sarl|sas|sasu|eurl|sa|sci|ei|s a s|s a r l)\b"
FR_CITY = {}   # city (lower, accent-folded) -> region, built from label-free S1 addresses at prepare time


def fr_key(p):
    k = re.sub(r"[^a-z0-9 ]", " ", unidecode(p).lower())
    k = re.sub(r"\bst[e]?\b", "saint", k)
    return re.sub(r"\s+", " ", k).strip()


def fr_region(parts):
    for p in reversed(parts):
        k = unidecode(p.strip()).lower()
        if k in FR_REGIONS: return FR_REGIONS[k]
    for p in reversed(parts):
        k = fr_key(p)
        if k in FR_CITY: return FR_CITY[k]
    return None


def learn_fr_city(s1_france: pl.DataFrame, min_n=5, purity=0.95):
    """city -> region from S1 addresses 'street, city, region' (label-free lookup table)."""
    from collections import Counter, defaultdict
    co = defaultdict(Counter)
    for a in s1_france["business_address"].to_list():
        ps = [p.strip() for p in a.split(",") if p.strip()]
        r = None
        for p in ps:
            r = FR_REGIONS.get(unidecode(p).lower()) or r
        if not r: continue
        for p in ps:
            k = fr_key(p)
            if unidecode(p).lower() not in FR_REGIONS and not re.search(r"\d", k): co[k][r] += 1
    return {k: c.most_common(1)[0][0] for k, c in co.items()
            if sum(c.values()) >= min_n and c.most_common(1)[0][1] / sum(c.values()) >= purity}


def fold(s: str) -> str:
    """Native-script words -> learned Latin word (fallback unidecode); strip trailing '| www...' URLs."""
    if s is None: return ""
    s = re.sub(r"\s*\|.*$", "", s)
    if INDIC_RE.search(s):
        s = " ".join(TRANSLIT.get(w, w) if INDIC_RE.search(w) else w for w in s.split())
    return unidecode(s)


def learn_translit(pairs: pl.DataFrame, min_n=3, purity=0.6):
    """Align S1 (Latin) and target (native script) names word-by-word when word counts agree."""
    from collections import Counter, defaultdict
    co = defaultdict(Counter)
    for n1, n2 in pairs.select("n1", "n2").iter_rows():
        if not INDIC_RE.search(n2): continue
        a = re.sub(r"[^\w\s]", " ", n1).lower().split(); b = n2.split()
        if len(a) != len(b): continue
        for x, y in zip(a, b):
            if INDIC_RE.search(y): co[y][x] += 1
    out = {}
    for y, c in co.items():
        x, n = c.most_common(1)[0]; tot = sum(c.values())
        if tot >= min_n and n / tot >= purity: out[y] = x
    return out


def _norm_expr(e):
    return e.str.to_lowercase().str.replace_all(r"[^a-z0-9 ]", " ").str.replace_all(r"\s+", " ").str.strip_chars()


def us_state(parts):
    for p in reversed(parts):
        u = p.strip()
        if u.upper() in US_STATES: return u.upper()
        c = US_NAME2CODE.get(u.lower())
        if c: return c
    return None


def in_state(parts, surface):
    for p in reversed(parts):
        p = p.strip()
        c = IN_CANON.get(p.lower())
        if c: return c
        c = surface.get(p)
        if c: return c
    return None


def learn_india_surface(pairs: pl.DataFrame, min_n=30, purity=0.98):
    """Learn target-side surface forms of Indian states (native script, codes, variants) from train pairs.
    pairs: columns a1 (S1 address), a2 (target address)."""
    from collections import Counter, defaultdict
    co = defaultdict(Counter)
    for a1, a2 in pairs.select("a1", "a2").iter_rows():
        st = in_state(a1.split(","), {})
        if not st: continue
        for p in a2.split(","):
            p = p.strip()
            if p and p.lower() not in IN_CANON: co[p][st] += 1
    out = {}
    for p, c in co.items():
        s, n = c.most_common(1)[0]; tot = sum(c.values())
        if tot >= min_n and n / tot >= purity: out[p] = s
    return out


# US/India street-type abbreviations -> full words (applied to both sides, so "1452 ELLSWORTH RD" == "1452 Ellsworth Road")
ADDR_ABBR = [(r"\brd\b", "road"), (r"\baven?\b|\bave\b|\bav\b", "avenue"), (r"\bblvd\b", "boulevard"), (r"\bdr\b", "drive"),
             (r"\bln\b", "lane"), (r"\bct\b", "court"), (r"\bpl\b", "place"), (r"\bpkwy\b", "parkway"), (r"\bhwy\b", "highway"),
             (r"\btrl\b", "trail"), (r"\bcir\b", "circle"), (r"\bst\b", "street"), (r"\bter\b", "terrace"), (r"\bsq\b", "square"),
             (r"\bapt\b", "apartment"), (r"\bste\b", "suite"), (r"\bnr\b", "near"), (r"\bopp\b", "opposite"),
             (r"\bbldg\b", "building"), (r"\bmkt\b", "market"), (r"\bn a\b", " ")]


def state_and_rest(country, address, surface):
    """(canonical state/region, address with the pure state part removed). A part is removed only when it is a
    state name/code/native-script form, never when it is a city that merely implies the state."""
    parts = [p.strip() for p in (address or "").split(",")]
    if country == "US":
        for i in range(len(parts) - 1, -1, -1):
            u = parts[i]
            c = u.upper() if u.upper() in US_STATES else US_NAME2CODE.get(u.lower())
            if c: return c, ", ".join(parts[:i] + parts[i + 1:])
        return None, address or ""
    if country == "India":
        for i in range(len(parts) - 1, -1, -1):
            p = parts[i]
            c = IN_CANON.get(p.lower())
            if c: return c, ", ".join(parts[:i] + parts[i + 1:])
            c = surface.get(p)
            if c:
                pure = len(p) <= 4 or bool(INDIC_RE.search(p)) or p.lower().rstrip("m") in IN_CANON
                return c, (", ".join(parts[:i] + parts[i + 1:]) if pure else address)
        return None, address or ""
    if country == "France":
        return fr_region(parts), address or ""
    return None, address or ""


def normalise(df: pl.DataFrame, india_surface: dict) -> pl.DataFrame:
    """df: entity_id, business_name, business_address, country -> adds normalised views + parsed fields."""
    sr = pl.struct("country", "business_address").map_elements(
        lambda r: dict(zip(("st", "rest"), state_and_rest(r["country"], r["business_address"], india_surface))),
        return_dtype=pl.Struct({"st": pl.Utf8, "rest": pl.Utf8}))
    df = df.with_columns(sr.alias("_sr")).with_columns(pl.col("_sr").struct.field("st").alias("st"),
                                                         pl.col("_sr").struct.field("rest").alias("_rest")).drop("_sr")
    name_f = pl.col("business_name").map_elements(fold, return_dtype=pl.Utf8, skip_nulls=False)
    addr_f = pl.col("_rest").map_elements(fold, return_dtype=pl.Utf8, skip_nulls=False)
    df = df.with_columns(name_f.alias("_nf"), addr_f.alias("_af"))
    nm = _norm_expr(pl.col("_nf").str.replace(TLD_RE, "").str.replace_all(r"^@", ""))
    df = df.with_columns(nm.alias("nm"), _norm_expr(pl.col("_af")).alias("ad"))
    fr = pl.col("country") == "France"
    fr_ad = pl.col("ad")
    for pat, rep in FR_STREET: fr_ad = fr_ad.str.replace_all(pat, rep)
    fr_ad = fr_ad.str.replace_all(FR_REGION_WORDS, " ").str.replace_all(r"\s+", " ").str.strip_chars()
    us_in_ad = pl.col("ad")
    for pat, rep in ADDR_ABBR: us_in_ad = us_in_ad.str.replace_all(pat, rep)
    us_in_ad = us_in_ad.str.replace_all(r"\s+", " ").str.strip_chars()
    df = df.with_columns(pl.when(fr).then(fr_ad).otherwise(us_in_ad).alias("ad"))
    core = (pl.col("nm").str.replace_all(LEGAL_RE, " ").str.replace_all(GENERIC_RE, " ")
            .str.replace_all(r"\s+", " ").str.strip_chars())
    fr_core = pl.col("nm").str.replace_all(FR_NAME_STOP, " ").str.replace_all(r"\s+", " ").str.strip_chars()
    df = df.with_columns(pl.when(pl.col("country") == "France").then(fr_core).otherwise(core).alias("core"))
    df = df.with_columns(pl.when(pl.col("core") == "").then(pl.col("nm")).otherwise(pl.col("core")).alias("core"))
    df = df.with_columns(pl.col("core").str.replace_all(" ", "").alias("glued"),
                         pl.col("nm").str.extract(LEGAL_RE, 1).fill_null("").alias("legal"),
                         (pl.col("business_address").str.strip_chars() == "").alias("addr_empty"),
                         pl.col("business_name").str.contains(r"[ऀ-෿]").alias("indic_name"))
    # numbers
    df = df.with_columns(pl.col("ad").str.extract_all(r"\d+").list.eval(pl.element().str.strip_chars_start("0")).alias("nums"))
    df = df.with_columns(pl.col("nums").list.first().fill_null("").alias("hn"))
    # sorted-token address view (rotation invariant)
    df = df.with_columns(pl.col("ad").str.split(" ").list.sort().list.join(" ").alias("ad_sorted"))
    # consonant skeleton of the core name (robust to typos, vowel drops and transliteration vowels)
    sk = pl.col("core").str.replace_all(r"[aeiouy ]", "")
    for ch in "bcdfghjklmnpqrstvwxz0123456789":
        sk = sk.str.replace_all(ch + "{2,}", ch)
    df = df.with_columns(sk.alias("skel"))
    return df.drop("_nf", "_af", "_rest")
