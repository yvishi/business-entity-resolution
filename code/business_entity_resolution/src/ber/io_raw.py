"""Raw TSV -> Parquet (+ long-format train links)."""
import os, polars as pl

def tsv_to_parquet(train_dir, test_dir, out):
    os.makedirs(out, exist_ok=True)
    for split, d in (("train", train_dir), ("test", test_dir)):
        for i in "123":
            f = f"{d}/{split}_source{i}.tsv"
            df = pl.read_csv(f, separator="\t", quote_char=None, infer_schema=False)
            df.with_columns(pl.col("business_name").fill_null(""), pl.col("business_address").fill_null("")
                            ).write_parquet(f"{out}/{split}_source{i}.parquet")
            print("parquet", f, df.height, flush=True)
    gt = pl.read_csv(f"{train_dir}/train_ground_truth.tsv", separator="\t", quote_char=None, infer_schema=False)
    (gt.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",").alias("t")).explode("t")
       .filter(pl.col("t").is_not_null() & (pl.col("t") != "")).select(pl.col("source1_entity_id").alias("s1"), "t")
       .write_parquet(f"{out}/train_links.parquet"))
