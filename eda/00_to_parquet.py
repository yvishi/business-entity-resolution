"""Convert the raw TSVs to Parquet (fast reloads, lower memory). Usage: python eda/00_to_parquet.py <raw_dir> <out_dir>"""
import sys, os, polars as pl
raw, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
for split in ["train", "test"]:
    for i in "123":
        f = f"{split}_source{i}.tsv"
        df = pl.read_csv(os.path.join(raw, f), separator="\t", quote_char=None, infer_schema=False)
        df = df.with_columns(pl.col("business_address").fill_null(""), pl.col("business_name").fill_null(""),
                             pl.col("country").cast(pl.Categorical))
        df.write_parquet(os.path.join(out, f.replace(".tsv", ".parquet")))
        print(f, df.height, flush=True)
gt = pl.read_csv(os.path.join(raw, "train_ground_truth.tsv"), separator="\t", quote_char=None, infer_schema=False)
gt = gt.with_columns(pl.col("matched_entity_ids").fill_null(""))
gt.write_parquet(os.path.join(out, "train_ground_truth.parquet"))
# long-format links: s1, target
links = (gt.with_columns(pl.col("matched_entity_ids").str.split(",").alias("t")).explode("t")
           .filter(pl.col("t") != "").select(pl.col("source1_entity_id").alias("s1"), "t"))
links.write_parquet(os.path.join(out, "train_links.parquet"))
print("links", links.height)
