"""Re-tune the decision layer of an EXISTING v2 run -- no retraining, no re-blocking, ~5-10 min.

    python rescore_v2.py --work <v2 work dir> --out <new output dir>

Uses what the v2 run already saved in its work dir:
  oof.parquet (s1, t, y, p1, p2), s1ids_train_*.parquet, raw/train_links.parquet, train_s1.parquet,
  models.pkl (v2's chosen stage + decision), test_scores.parquet, test_candidates.parquet, test_s1.parquet
What it tries (all scored on train OOF with the exact competition metric, US + India):
  * isotonic recalibration of the scores, CROSS-FITTED (fit on half the S1s, scored on the other half) so the
    comparison is not flattered by fitting and scoring on the same rows;
  * a finer grid of decision rules (expected-F0.5 set selection and plain thresholds, with/without unique mapping).
The winner is picked on the DENSE view (19% of S1 hidden, as in v2 -- mimics test's higher record density).
A new submission is written ONLY if the winner beats v2's own decision on that view; otherwise it says so and
writes nothing, so you never replace the 0.97 file with something that validated worse.
"""
import argparse, os, pickle, time
import numpy as np, polars as pl
from sklearn.isotonic import IsotonicRegression

B2 = 0.25
PMIN = 0.005   # pairs below this can never be selected (v2 already dropped test pairs < 0.005)


def unique_map(d):
    return d.filter(pl.col("p") == pl.col("p").max().over("t"))


def select(d, mode, tau, s1_agg):
    """Vectorised equivalent of ber.decide.select. s1_agg: per-S1 total mass and log P(no match) over ALL pairs."""
    if mode == "thr":
        return d.filter(pl.col("p") >= tau).select("s1", "t", "y")
    g = (d.sort(["s1", "p"], descending=[False, True])
           .join(s1_agg, on="s1", how="left")
           .with_columns(pl.col("p").cum_sum().over("s1").alias("cs"),
                         pl.int_range(1, pl.len() + 1).over("s1").alias("k"),
                         pl.col("p").max().over("s1").alias("pmax")))
    g = g.with_columns(((1 + B2) * pl.col("cs") / (B2 * pl.col("tot") + pl.col("k"))).alias("ef"))
    g = g.with_columns(pl.col("ef").max().over("s1").alias("efmax"))
    kbest = g.filter(pl.col("ef") == pl.col("efmax")).group_by("s1").agg(pl.col("k").min().alias("kb"))
    g = g.join(kbest, on="s1")
    keep = (pl.col("k") <= pl.col("kb")) & (pl.col("efmax") > pl.col("logp0").exp()) & (pl.col("pmax") >= tau)
    return g.filter(keep).select("s1", "t", "y")


def macro_f05(pred, universe):
    """universe: s1, ntrue (true set size from the links, so blocking misses count against recall)."""
    a = pred.group_by("s1").agg(pl.len().alias("npred"), pl.col("y").sum().alias("tp"))
    u = universe.join(a, on="s1", how="left").with_columns(pl.col("npred").fill_null(0), pl.col("tp").fill_null(0))
    P = pl.col("tp") / pl.col("npred"); R = pl.col("tp") / pl.col("ntrue")
    f = (pl.when((pl.col("ntrue") == 0) & (pl.col("npred") == 0)).then(1.0)
           .when(pl.col("tp") == 0).then(0.0)
           .otherwise((1 + B2) * P * R / (B2 * P + R)))
    return float(u.select(f.mean()).item())


def s1_aggregates(d_full):
    return d_full.group_by("s1").agg(pl.col("p").sum().alias("tot"),
                                     (1 - pl.col("p")).clip(1e-12, 1).log().sum().alias("logp0"))


def prepared(d, um):
    """(pairs >= PMIN, per-S1 aggregates) exactly as ber.decide.select sees them: unique mapping first, then the
    per-S1 mass / P(no match) over everything that survived it."""
    d = unique_map(d) if um == "uniq" else d
    return d.filter(pl.col("p") >= PMIN), s1_aggregates(d)


def fit_iso(p, y, n_max=5_000_000, seed=0):
    if len(p) > n_max:
        idx = np.random.default_rng(seed).choice(len(p), n_max, replace=False); p, y = p[idx], y[idx]
    return IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p, y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--min-gain", type=float, default=0.0003, help="dense-view gain required to write a new file")
    a = ap.parse_args(); W = a.work; t0 = time.time()
    log = lambda m: print(f"{m} [{time.time()-t0:.0f}s]", flush=True)

    M = pickle.load(open(f"{W}/models.pkl", "rb"))
    stage, v2_dec = M.get("stage", 2), tuple(M["decision"])
    pcol = "p2" if stage == 2 else "p1"
    log(f"v2 run: stage {stage}, decision {v2_dec}")

    oof = pl.read_parquet(f"{W}/oof.parquet").select("s1", "t", pl.col("y").cast(pl.Int32), pl.col(pcol).cast(pl.Float64).alias("p"))
    ids = pl.concat([pl.read_parquet(f) for f in [f"{W}/s1ids_train_{c}.parquet" for c in ("US", "India")] if os.path.exists(f)])["s1"].unique()
    links = pl.read_parquet(f"{W}/raw/train_links.parquet").filter(pl.col("s1").is_in(ids.implode()))
    uni = pl.DataFrame({"s1": ids}).join(links.group_by("s1").len().rename({"len": "ntrue"}), on="s1", how="left") \
            .with_columns(pl.col("ntrue").fill_null(0))
    hidden = pl.col("s1").hash(seed=11) % 100 < 19            # same hidden set as v2's evaluate()
    views = {"normal": (oof, uni), "dense": (oof.filter(~hidden), uni.filter(~hidden))}
    log(f"OOF pairs {oof.height:,}, S1 universe {uni.height:,} (dense: {views['dense'][1].height:,})")

    # cross-fitted calibration: each half of the S1s is calibrated by a curve fit on the other half
    half = (pl.col("s1").hash(seed=23) % 2).alias("h")
    o = oof.with_columns(half)
    pc = np.empty(o.height)
    for h in (0, 1):
        tr, te = o["h"].to_numpy() != h, o["h"].to_numpy() == h
        pc[te] = fit_iso(o["p"].to_numpy()[tr], o["y"].to_numpy()[tr]).predict(o["p"].to_numpy()[te])
    variants = {"raw": oof, "cal": o.drop("h").with_columns(pl.Series("p", pc))}
    log("cross-fitted isotonic calibration done")

    rules = [("expf", round(float(t), 2)) for t in np.arange(0.0, 0.91, 0.05)] + \
            [("thr", round(float(t), 2)) for t in np.arange(0.3, 0.96, 0.05)]
    res = {}
    for vname, dv in variants.items():
        for view, (_, u) in views.items():
            d = dv if view == "normal" else dv.filter(~hidden)
            for um in ("raw", "uniq"):
                dd, agg = prepared(d, um)
                for mode, tau in rules:
                    res[(vname, mode, tau, um, view)] = macro_f05(select(dd, mode, tau, agg), u)
        log(f"scored {vname}")

    base = ("raw", v2_dec[0], round(float(v2_dec[1]), 2), v2_dec[2])
    if base + ("dense",) not in res:   # v2 rule not on the grid -> score it explicitly
        for view, (_, u) in views.items():
            dd, agg = prepared(oof if view == "normal" else oof.filter(~hidden), v2_dec[2])
            res[base + (view,)] = macro_f05(select(dd, v2_dec[0], float(v2_dec[1]), agg), u)
    cands = {k[:4] for k in res}
    ranked = sorted(cands, key=lambda k: -res[k + ("dense",)])
    print("\nTop rules (normal / dense OOF macro F0.5):")
    for k in ranked[:8]:
        print(f"  {k}: {res[k + ('normal',)]:.5f} / {res[k + ('dense',)]:.5f}")
    best = ranked[0]
    b_d, v_d = res[best + ("dense",)], res[base + ("dense",)]
    print(f"\nv2 decision {base}: {res[base + ('normal',)]:.5f} / {v_d:.5f}")
    print(f"best          {best}: {res[best + ('normal',)]:.5f} / {b_d:.5f}   dense gain {b_d - v_d:+.5f}")
    if b_d - v_d < a.min_gain:
        print("\nNo validated improvement over v2's decision -> NOT writing a new submission. Keep the 0.97 file.")
        return

    # apply to test: calibration curve refit on ALL OOF rows (the cross-fit was only for honest scoring)
    te = pl.read_parquet(f"{W}/test_scores.parquet").select("s1", "t", pl.col("p").cast(pl.Float64))
    if best[0] == "cal":
        iso = fit_iso(oof["p"].to_numpy(), oof["y"].to_numpy())
        te = te.with_columns(pl.Series("p", iso.predict(te["p"].to_numpy())))
    d, agg = prepared(te.with_columns(pl.lit(0).alias("y")), best[3])
    pred = select(d, best[1], best[2], agg)
    os.makedirs(a.out, exist_ok=True)
    base_ids = pl.read_parquet(f"{W}/test_s1.parquet", columns=["entity_id"]).rename({"entity_id": "source1_entity_id"})
    m = pred.group_by("s1").agg(pl.col("t").sort().str.join(",").alias("matched_entity_ids")).rename({"s1": "source1_entity_id"})
    out = base_ids.join(m, on="source1_entity_id", how="left").with_columns(pl.col("matched_entity_ids").fill_null(""))
    out.write_csv(f"{a.out}/matching_results.tsv", separator="\t", quote_style="never")
    cands = pl.read_parquet(f"{W}/test_candidates.parquet").rename({"s1": "source1_entity_id"})
    base_ids.join(cands, on="source1_entity_id", how="left").with_columns(pl.col("candidate_entity_ids").fill_null("")) \
        .write_csv(f"{a.out}/candidate_pairs.tsv", separator="\t", quote_style="never")
    ne = out.filter(pl.col("matched_entity_ids") != "").height
    log(f"wrote {a.out}/matching_results.tsv: {out.height:,} rows, empty {1 - ne/out.height:.2%}, "
        f"mean matches {pred.height/out.height:.2f}  (v2 file: empty 5.94%, mean 3.35)")


if __name__ == "__main__":
    main()
