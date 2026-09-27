"""Stage 'predict': score test candidates with the trained models, apply decisions, write
output/matching_results.tsv and output/candidate_pairs.tsv. Usage: python -m ber.predict <work> <out_dir>"""
import sys, os, time, pickle
import numpy as np, polars as pl
from .features import build, add_ambiguity, add_context, add_support
from .decide import unique_map, select, apply_calibration


def _pred(models, X):
    models = models if isinstance(models, list) else [models]
    return np.mean([m.predict(X, num_iteration=m.best_iteration or None) for m in models], axis=0)


def main(work, out_dir, countries=("US", "India", "France"), keep_min=0.005):
    """Three checkpointed passes per country, matching training exactly:
      1. per partition: pair features + stage-1 score p1   -> work/pred_<c>/<part>.p1.parquet
      2. per country:   context + support features computed over ALL candidates (like training) -> _ctx.parquet
      3. per partition: stage-2 score p (keep p >= keep_min) -> <part>.p2.parquet
    Name ambiguity is counted over the whole country, exactly as in training. A rerun skips finished files.
    Memory: only one partition's features are in RAM at a time; big tables are freed before the country-wide pass."""
    import gc
    from .features import REC_COLS, CTX, SUPPORT
    from .block import prune, _safe
    t0 = time.time()
    M = pickle.load(open(f"{work}/models.pkl", "rb")); vecs = pickle.load(open(f"{work}/tfidf.pkl", "rb"))
    feats1, feats2, stage2 = M["feats1"], M["feats2"], M.get("stage", 2) == 2
    s1_all = pl.read_parquet(f"{work}/test_s1.parquet", columns=REC_COLS)
    ids = s1_all["entity_id"]
    p2_files, cl_files = [], []
    for c in countries:
        ck = f"{work}/pred_{c}"; os.makedirs(ck, exist_ok=True)
        cand = prune(pl.read_parquet(f"{work}/cand_test_{c}.parquet"))
        if cand.height == 0: print(f"{c}: no candidates", flush=True); continue
        fcl = os.path.join(ck, "_candidate_lists.parquet")
        if not os.path.exists(fcl):
            cand.group_by("s1").agg(pl.col("t").unique().sort().str.join(",").alias("candidate_entity_ids")) \
                .write_parquet(fcl + ".tmp"); os.replace(fcl + ".tmp", fcl)
        cl_files.append(fcl)
        s1 = s1_all.filter(pl.col("country") == c)
        amb = s1.group_by(["country", "core"]).len().rename({"len": "amb_core", "core": "core_1", "country": "country_1"})
        cand = cand.join(s1.select(pl.col("entity_id").alias("s1"), pl.col("st").fill_null("_none").alias("_part")), on="s1", how="left")
        groups = cand.partition_by("_part", as_dict=True); del cand
        parts = [k[0] for k in groups]
        p1f = {pt: os.path.join(ck, _safe(pt) + ".p1.parquet") for pt in parts}
        p2f = {pt: os.path.join(ck, _safe(pt) + ".p2.parquet") for pt in parts}
        fctx = os.path.join(ck, "_ctx.parquet")
        need_pool = any(not os.path.exists(p1f[pt]) for pt in parts) or (stage2 and not os.path.exists(fctx))
        pool = pl.concat([pl.scan_parquet(f"{work}/test_s{i}.parquet").select(REC_COLS).filter(pl.col("country") == c).collect()
                          for i in "23"]) if need_pool else None
        # ---- pass 1
        for pt in parts:
            cp = groups.pop((pt,)).drop("_part")
            if os.path.exists(p1f[pt]): continue
            s1p = s1.filter(pl.col("entity_id").is_in(cp["s1"].unique().implode()))
            poolp = pool.filter(pl.col("entity_id").is_in(cp["t"].unique().implode()))
            chunks = []
            for i in range(0, cp.height, 1_000_000):
                x = build(cp.slice(i, 1_000_000), s1p, poolp, vecs)
                x = x.join(amb, on=["country_1", "core_1"], how="left").with_columns(pl.col("amb_core").fill_null(1))
                x = x.select(["s1", "t"] + feats1).with_columns(pl.col(feats1).cast(pl.Float32))
                x = x.with_columns(pl.Series("p1", _pred(M["m1"], x.select(feats1).to_numpy())).cast(pl.Float32))
                chunks.append(x); del x
            pl.concat(chunks, rechunk=False).write_parquet(p1f[pt] + ".tmp"); os.replace(p1f[pt] + ".tmp", p1f[pt])
            del chunks, s1p, poolp, cp; gc.collect()
        del groups; gc.collect()
        print(f"{c}: pass 1 done ({len(parts)} partitions) [{time.time()-t0:.0f}s]", flush=True)
        # ---- pass 2 (country-wide context + support, as in training)
        if stage2 and not os.path.exists(fctx):
            small = pl.concat([pl.read_parquet(p1f[pt], columns=["s1", "t", "p1"]) for pt in parts])
            recs = None
            if "sup_name" in feats2:
                recs = pool.filter(pl.col("entity_id").is_in(small.filter(pl.col("p1") >= 0.05)["t"].unique().implode())) \
                           .select("entity_id", "core", "ad")
            del pool; pool = None; gc.collect()
            small = add_context(small)
            if recs is not None:
                small = add_support(small, recs); del recs
            small.select(["s1", "t"] + [f for f in CTX + SUPPORT if f in small.columns and f != "p1"]) \
                 .write_parquet(fctx + ".tmp"); os.replace(fctx + ".tmp", fctx)
            del small; gc.collect()
        del pool, amb, s1; gc.collect()
        print(f"{c}: pass 2 done [{time.time()-t0:.0f}s]", flush=True)
        # ---- pass 3
        ctx = pl.read_parquet(fctx) if stage2 else None
        for pt in parts:
            if not os.path.exists(p2f[pt]):
                d = pl.read_parquet(p1f[pt])
                if stage2:
                    d = d.join(ctx, on=["s1", "t"], how="left", maintain_order="left")
                    ps = [_pred(M["m2"], d.slice(i, 500_000).select(feats2).to_numpy().astype(np.float32))
                          for i in range(0, d.height, 500_000)]
                    d = d.with_columns(pl.Series("p", np.concatenate(ps) if ps else np.zeros(0)).cast(pl.Float32))
                else:
                    d = d.with_columns(pl.col("p1").alias("p"))
                d = d.select("s1", "t", "p").filter(pl.col("p") >= keep_min)
                d.write_parquet(p2f[pt] + ".tmp"); os.replace(p2f[pt] + ".tmp", p2f[pt]); del d; gc.collect()
            p2_files.append(p2f[pt])
        del ctx; gc.collect()
        print(f"{c}: pass 3 done [{time.time()-t0:.0f}s]", flush=True)
    del s1_all; gc.collect()
    d = pl.concat([pl.read_parquet(f) for f in p2_files])
    d.write_parquet(f"{work}/test_scores.parquet")   # raw (uncalibrated) scores -- calibration applied in write_outputs
    pl.concat([pl.read_parquet(f) for f in cl_files]).write_parquet(f"{work}/test_candidates.parquet")
    write_outputs(work, out_dir, d, M["decision"], ids.to_list(), M.get("calibrator"))
    print(f"[{time.time()-t0:.0f}s]")


def decide_only(work, out_dir, decision=None):
    """Re-apply a (possibly different) decision rule to saved test scores: ~2 min, no feature building."""
    M = pickle.load(open(f"{work}/models.pkl", "rb"))
    d = pl.read_parquet(f"{work}/test_scores.parquet")
    ids = pl.read_parquet(f"{work}/test_s1.parquet", columns=["entity_id"])["entity_id"].to_list()
    write_outputs(work, out_dir, d, decision or M["decision"], ids, M.get("calibrator"))


def write_outputs(work, out_dir, d, decision, ids, calibrator=None):
    if calibrator is not None:
        d = d.with_columns(pl.Series("p", apply_calibration(calibrator, d["p"].to_numpy())))
    mode, tau, uniq = decision
    pred = select(unique_map(d) if uniq == "uniq" else d, mode=mode, tau=tau)
    os.makedirs(out_dir, exist_ok=True)
    base = pl.DataFrame({"source1_entity_id": ids})
    m = pl.DataFrame({"source1_entity_id": list(pred.keys()),
                      "matched_entity_ids": [",".join(sorted(v)) for v in pred.values()]}, schema={"source1_entity_id": pl.Utf8, "matched_entity_ids": pl.Utf8})
    base.join(m, on="source1_entity_id", how="left").with_columns(pl.col("matched_entity_ids").fill_null("")) \
        .write_csv(f"{out_dir}/matching_results.tsv", separator="\t", quote_style="never")
    cands = pl.read_parquet(f"{work}/test_candidates.parquet").rename({"s1": "source1_entity_id"})
    base.join(cands, on="source1_entity_id", how="left").with_columns(pl.col("candidate_entity_ids").fill_null("")) \
        .write_csv(f"{out_dir}/candidate_pairs.tsv", separator="\t", quote_style="never")
    n = len(pred)
    print(f"decision {decision}: wrote {len(ids):,} rows; non-empty {n:,} ({n/len(ids):.1%}); mean matches "
          f"{sum(len(v) for v in pred.values())/len(ids):.2f}")
