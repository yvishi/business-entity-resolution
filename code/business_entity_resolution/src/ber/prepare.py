"""Stage 'prepare': raw parquet -> normalised parquet (+ learned India state surface forms, from TRAIN only).
Usage: python -m ber.prepare <raw_parquet_dir> <work_dir>"""
import sys, os, json, time
import polars as pl
from .text import normalise, learn_india_surface, learn_translit, set_translit, learn_fr_city, FR_CITY

def main(raw, work, splits=("train", "test")):
    os.makedirs(work, exist_ok=True)
    t = time.time()
    s1 = pl.read_parquet(f"{raw}/train_source1.parquet")
    tg = pl.concat([pl.read_parquet(f"{raw}/train_source2.parquet"), pl.read_parquet(f"{raw}/train_source3.parquet")])
    links = pl.read_parquet(f"{raw}/train_links.parquet")
    pairs = (links.join(s1.select(pl.col("entity_id").alias("s1"), pl.col("business_address").alias("a1"), pl.col("business_name").alias("n1"),
                                  pl.col("country").cast(pl.Utf8).alias("c")), on="s1")
                  .filter(pl.col("c") == "India")
                  .join(tg.select(pl.col("entity_id").alias("t"), pl.col("business_address").alias("a2"), pl.col("business_name").alias("n2")), on="t"))
    surface = learn_india_surface(pairs.sample(min(2_000_000, pairs.height), seed=0))
    json.dump(surface, open(f"{work}/india_state_surface.json", "w", encoding="utf-8"), ensure_ascii=False)
    print("india surface forms", len(surface), f"{time.time()-t:.0f}s", flush=True)
    t1 = pl.read_parquet(f"{raw}/test_source1.parquet")
    FR_CITY.clear(); FR_CITY.update(learn_fr_city(t1.filter(pl.col("country") == "France")))
    print("france city->region entries", len(FR_CITY), flush=True)
    tr = learn_translit(pairs)
    set_translit(tr)
    json.dump(tr, open(f"{work}/translit.json", "w", encoding="utf-8"), ensure_ascii=False)
    print("translit words", len(tr), f"{time.time()-t:.0f}s", flush=True)
    del tg, pairs
    for split in splits:
        for i in "123":
            df = pl.read_parquet(f"{raw}/{split}_source{i}.parquet").with_columns(pl.col("country").cast(pl.Utf8))
            out = normalise(df, surface)
            out.write_parquet(f"{work}/{split}_s{i}.parquet")
            print(split, i, out.height, "state null:", out["st"].null_count(), f"{time.time()-t:.0f}s", flush=True)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
