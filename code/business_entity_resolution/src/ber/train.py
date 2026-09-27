"""Stage 'train' (memory-safe, resumable).

1. Features for all (pruned) train candidates are written chunk by chunk to a disk-backed float32 array (memmap);
   keys/labels go to a small side table. RAM never holds the full feature matrix twice.
2. Stage 1: LightGBM, k-fold out-of-fold (OOF). The binned Dataset is built once; folds use subsets of it.
3. Stage 2: context + support features from stage-1 OOF scores, streamed next to the stage-1 matrix (no copy on disk).
4. Decision rule chosen on a test-like (dense) view. Fold models are saved as the final ensemble.
Checkpoints: features, stage-1 OOF and stage-1 models; a rerun resumes from the last finished part.
Usage: python -m ber.train <work> <raw> <countries>
"""
import sys, os, time, pickle, json, gc
import numpy as np, polars as pl, lightgbm as lgb
from .features import build, add_ambiguity, add_context, add_support, FEATURES, CTX, SUPPORT, REC_COLS
from .decide import unique_map, select, macro_f05, fit_calibration, apply_calibration

PARAMS = dict(objective="binary", learning_rate=float(os.environ.get("BER_LR", "0.08")), num_leaves=255,
              min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              max_bin=127, verbose=-1, num_threads=os.cpu_count(), seed=17)
ROUNDS = int(os.environ.get("BER_ROUNDS", "600"))
PATIENCE = int(os.environ.get("BER_PATIENCE", "60"))
MAX_PAIRS = int(float(os.environ.get("BER_MAX_PAIRS", "0")))   # 0 = automatic from available RAM
CHUNK = 1_000_000


def _log(t0, msg):
    print(f"{msg} [{time.time()-t0:.0f}s]", flush=True)


def _auto_max_pairs():
    if MAX_PAIRS > 0: return MAX_PAIRS
    try:
        import psutil
        gb = psutil.virtual_memory().total / 2**30
    except Exception:
        gb = 16
    # ~0.45 GB of RAM per million pairs at peak (binned data + fold subsets + evaluation frames) -> keep 45% headroom
    # (the 40M ceiling was a 16 GB-laptop safety limit; on a bigger box this now scales with detected RAM)
    return int(max(4, min(2000, gb * 0.55 / 0.45)) * 1e6)


def _load_country(work, split, c, keep_s1=None):
    from .block import prune
    cand = prune(pl.read_parquet(f"{work}/cand_{split}_{c}.parquet"))
    if keep_s1 is not None: cand = cand.filter(pl.col("s1").is_in(keep_s1.implode()))
    s1 = pl.read_parquet(f"{work}/{split}_s1.parquet", columns=REC_COLS).filter(pl.col("country") == c)
    tids = cand.select(pl.col("t").unique())
    pool = pl.concat([pl.scan_parquet(f"{work}/{split}_s{i}.parquet").select(REC_COLS)
                      .join(tids.lazy(), left_on="entity_id", right_on="t", how="semi").collect() for i in "23"])
    return s1, pool, cand


class HStack(lgb.Sequence):
    """Row access to [memmap X | in-RAM extra columns] without materialising the concatenation."""
    def __init__(self, X, E, batch_size=200_000):
        self.X, self.E, self.batch_size = X, E, batch_size
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, idx):
        if isinstance(idx, (int, np.integer)):
            return np.concatenate([self.X[idx], self.E[idx]]).astype(np.float64)
        return np.hstack([np.asarray(self.X[idx]), self.E[idx]]).astype(np.float64)   # LightGBM Sequence wants float64


def _rows(M, idx):
    if isinstance(M, HStack): return np.hstack([np.asarray(M.X[idx]), M.E[idx]]).astype(np.float32)
    return np.asarray(M[idx], dtype=np.float32)


def oof(M, y, folds, tag, t0):
    """k-fold OOF on matrix M (memmap or HStack). Returns OOF predictions and fold models."""
    n = len(y); p = np.zeros(n, dtype=np.float32); models = []
    full = lgb.Dataset(M, label=y, params={"max_bin": PARAMS["max_bin"], "verbose": -1}, free_raw_data=True).construct()
    for k in range(int(folds.max()) + 1):
        tr = np.flatnonzero(folds != k).astype(np.int32); va = np.flatnonzero(folds == k).astype(np.int32)
        dt, dv = full.subset(tr), full.subset(va)
        m = lgb.train(PARAMS, dt, ROUNDS, valid_sets=[dv], callbacks=[lgb.early_stopping(PATIENCE, verbose=False)])
        for i in range(0, len(va), 500_000):
            sl = va[i:i + 500_000]
            p[sl] = m.predict(_rows(M, sl), num_iteration=m.best_iteration or None)
        models.append(m); del dt, dv; gc.collect()
        _log(t0, f"  {tag} fold {k}: best_iter={m.best_iteration}")
    del full; gc.collect()
    return p, models


def evaluate(d, p, truth, s1_ids, countries_of):
    """Score decision rules on two views; pick the rule by the TEST-LIKE (dense) view:
    normal = all dev S1; dense = 19% of S1 hidden (their records become distractors, as in test)."""
    d = d.select("s1", "t").with_columns(pl.Series("p", p))
    ids_s = pl.Series(s1_ids)
    hide = set(ids_s.filter(ids_s.hash(seed=11) % 100 < 19).to_list())
    keep = [x for x in s1_ids if x not in hide]
    dd = d.filter(~pl.col("s1").is_in(list(hide)))
    views = {"normal": (d, unique_map(d), s1_ids), "dense": (dd, unique_map(dd), keep)}
    res = {}
    for mode, taus in (("thr", (0.5, 0.6, 0.7, 0.8, 0.85)), ("expf", (0.0, 0.3, 0.4, 0.5, 0.6, 0.7))):
        for tau in taus:
            for name in ("raw", "uniq"):
                for v, (raw_, um_, ids_) in views.items():
                    pred = select(um_ if name == "uniq" else raw_, mode=mode, tau=tau)
                    res[(mode, tau, name, v)] = macro_f05(pred, truth, ids_)
    rules = {k[:3] for k in res}
    best = max(rules, key=lambda r: res[r + ("dense",)])
    pred = select(views["normal"][1] if best[2] == "uniq" else d, mode=best[0], tau=best[1])
    by_c = {c: macro_f05(pred, truth, [x for x in s1_ids if countries_of[x] == c]) for c in set(countries_of.values())}
    flat = {r: res[r + ("normal",)] for r in rules}
    print("   top rules (normal / dense):", "; ".join(f"{r}: {res[r + ('normal',)]:.4f}/{res[r + ('dense',)]:.4f}"
                                                   for r in sorted(rules, key=lambda r: -res[r + ("dense",)])[:5]), flush=True)
    return flat, best, by_c


def featurize(work, countries, vecs, links, t0):
    """Writes work/trX1.npy (memmap, float32) + work/tr_side.parquet (s1, t, country, y). Returns (n, feats1)."""
    from .block import prune
    ids = {c: pl.read_parquet(f"{work}/s1ids_train_{c}.parquet")["s1"] for c in countries}
    s1st = pl.read_parquet(f"{work}/train_s1.parquet", columns=["entity_id", "st"]).rename({"entity_id": "s1"})
    per_part = pl.concat([prune(pl.read_parquet(f"{work}/cand_train_{c}.parquet", columns=["s1", "r_fna", "r_rna", "r_fnm", "r_rnm",
                                                                                          "r_fad", "r_rad", "fb"]))
                          .join(s1st, on="s1", how="left").group_by("st").len().with_columns(pl.lit(c).alias("c"))
                          for c in countries])
    total, cap = int(per_part["len"].sum()), _auto_max_pairs()
    if total > cap:   # RAM budget: drop whole (country, state) partitions at random -> the kept ones keep their full density
        rng = np.random.default_rng(3)
        order = rng.permutation(per_part.height)
        keep, acc = [], 0
        for i in order:
            n_i = int(per_part["len"][int(i)])
            if acc + n_i <= cap or not keep: keep.append(int(i)); acc += n_i
        kept = per_part[keep]
        for c in countries:
            sts = kept.filter(pl.col("c") == c)["st"]
            ok = s1st.filter(pl.col("st").is_in(sts.implode()))["s1"]
            ids[c] = ids[c].filter(ids[c].is_in(ok.implode()))
        _log(t0, f"pair budget {cap/1e6:.1f}M < {total/1e6:.1f}M candidates -> training on {len(keep)} of {per_part.height} "
                 f"partitions ({acc/1e6:.1f}M pairs)")
    counts = {c: prune(pl.read_parquet(f"{work}/cand_train_{c}.parquet")).filter(pl.col("s1").is_in(ids[c].implode())).height
              for c in countries}
    n_max = sum(counts.values())
    X, feats1, side, n = None, None, [], 0
    for c in countries:
        s1, pool, cand = _load_country(work, "train", c, keep_s1=ids[c])
        amb = s1.group_by(["country", "core"]).len().rename({"len": "amb_core", "core": "core_1", "country": "country_1"})
        for i in range(0, cand.height, CHUNK):
            x = build(cand.slice(i, CHUNK), s1, pool, vecs)
            x = x.join(amb, on=["country_1", "core_1"], how="left").with_columns(pl.col("amb_core").fill_null(1))
            if X is None:
                feats1 = [f for f in FEATURES if f in x.columns]
                X = np.lib.format.open_memmap(f"{work}/trX1.npy", mode="w+", dtype=np.float32, shape=(max(1, n_max), len(feats1)))
            m = x.height
            if n + m > n_max: raise RuntimeError(f"feature matrix overflow {n + m} > {n_max}")
            X[n:n + m] = x.select(feats1).cast(pl.Float32).to_numpy()
            side.append(x.select("s1", "t", pl.col("country_1").alias("country")))
            n += m
            _log(t0, f"  features {c}: {min(i + CHUNK, cand.height):,}/{cand.height:,}")
            del x
        del s1, pool, cand; gc.collect()
    X.flush(); del X
    # maintain_order="left": row i of the side table must stay row i of the feature matrix
    side = pl.concat(side).join(links.with_columns(pl.lit(1, pl.Int8).alias("y")), on=["s1", "t"], how="left",
                                maintain_order="left").with_columns(pl.col("y").fill_null(0))
    side.write_parquet(f"{work}/tr_side.parquet")
    pl.concat([v for v in ids.values()]).to_frame("s1").write_parquet(f"{work}/tr_universe.parquet")
    json.dump({"n": n, "feats1": feats1}, open(f"{work}/tr_meta.json", "w", encoding="utf-8"))
    return n, feats1


def main(work, raw, countries):
    t0 = time.time()
    vecs = pickle.load(open(f"{work}/tfidf.pkl", "rb"))
    links = pl.read_parquet(f"{raw}/train_links.parquet")
    if not os.path.exists(f"{work}/tr_meta.json"):
        featurize(work, countries, vecs, links, t0)
    meta = json.load(open(f"{work}/tr_meta.json", encoding="utf-8"))
    n, feats1 = meta["n"], meta["feats1"]
    X1 = np.load(f"{work}/trX1.npy", mmap_mode="r")[:n]
    side = pl.read_parquet(f"{work}/tr_side.parquet")
    s1_ids = pl.read_parquet(f"{work}/tr_universe.parquet")["s1"].to_list()
    truth = {}
    for x, t in links.filter(pl.col("s1").is_in(s1_ids)).iter_rows(): truth.setdefault(x, set()).add(t)
    s1c = pl.read_parquet(f"{work}/train_s1.parquet", columns=["entity_id", "country"]).filter(pl.col("entity_id").is_in(s1_ids))
    countries_of = dict(zip(s1c["entity_id"].to_list(), s1c["country"].to_list()))
    n_true = sum(len(v) for v in truth.values())
    y = side["y"].to_numpy().astype(np.float32)
    _log(t0, f"pairs={n:,} S1={len(s1_ids):,} features={len(feats1)} blocking pair recall={y.sum()/max(1,n_true):.4f}")
    folds = (side["s1"].hash(seed=7) % int(os.environ.get("BER_FOLDS", "5"))).cast(pl.Int32).to_numpy()

    # ---------------- stage 1 (checkpointed)
    if os.path.exists(f"{work}/tr_p1.npy") and os.path.exists(f"{work}/models_stage1.pkl"):
        p1 = np.load(f"{work}/tr_p1.npy"); m1 = pickle.load(open(f"{work}/models_stage1.pkl", "rb"))["m1"]
        _log(t0, "stage 1 loaded from checkpoint")
    else:
        p1, m1 = oof(X1, y, folds, "stage1", t0)
        np.save(f"{work}/tr_p1.npy", p1)
        pickle.dump({"m1": m1, "feats1": feats1}, open(f"{work}/models_stage1.pkl", "wb"))
    res, best, by_c = evaluate(side, p1, truth, s1_ids, countries_of)
    _log(t0, f"STAGE1 best {best}: F0.5={res[best]:.4f} by country {by_c}")

    # ---------------- stage 2: context + support features streamed next to X1
    d = add_context(side.select("s1", "t").with_columns(pl.Series("p1", p1)))
    tids = d.filter(pl.col("p1") >= 0.05).select(pl.col("t").unique())
    recs = pl.concat([pl.scan_parquet(f"{work}/train_s{i}.parquet").select("entity_id", "core", "ad")
                      .join(tids.lazy(), left_on="entity_id", right_on="t", how="semi").collect() for i in "23"])
    d = add_support(d, recs); del recs
    extra = CTX + SUPPORT
    E = d.select(extra).cast(pl.Float32).to_numpy(); del d; gc.collect()
    feats2 = feats1 + extra
    p2, m2 = oof(HStack(X1, E), y, folds, "stage2", t0)
    res2, best2, by_c2 = evaluate(side, p2, truth, s1_ids, countries_of)
    _log(t0, f"STAGE2 best {best2}: F0.5={res2[best2]:.4f} by country {by_c2}")

    # ---------------- isotonic calibration of stage-2 OOF scores (affects the 'expf' set-selection rule, which
    # sums raw probability mass -- miscalibrated scores bias the chosen set size, esp. for singletons). Fit on the
    # OOF predictions themselves (mildly optimistic: same data picks the calibration curve and scores it), kept
    # only if it wins on the dense (test-like density) view, same as any other decision-rule comparison here.
    iso2 = fit_calibration(p2, y)
    p2c = apply_calibration(iso2, p2)
    res2c, best2c, by_c2c = evaluate(side, p2c, truth, s1_ids, countries_of)
    _log(t0, f"STAGE2-CAL best {best2c}: F0.5={res2c[best2c]:.4f} by country {by_c2c}")

    json.dump({"best1": list(best), "f1": res[best], "best2": list(best2), "f2": res2[best2],
               "best2cal": list(best2c), "f2cal": res2c[best2c], "by_country2": by_c2, "by_country2cal": by_c2c,
               "feats1": feats1, "feats2": feats2, "pairs": n, "S1": len(s1_ids)},
              open(f"{work}/train_report.json", "w", encoding="utf-8"), indent=1)
    side.select("s1", "t", "y").with_columns(pl.Series("p1", p1), pl.Series("p2", p2)).write_parquet(f"{work}/oof.parquet")

    options = [(1, best, res[best], None), (2, best2, res2[best2], None), (2, best2c, res2c[best2c], iso2)]
    stage, decision, _, cal = max(options, key=lambda o: o[2])
    pickle.dump({"m1": m1, "m2": m2, "feats1": feats1, "feats2": feats2, "stage": stage,
                 "decision": decision, "calibrator": cal}, open(f"{work}/models.pkl", "wb"))
    _log(t0, f"models saved (fold ensembles; stage {stage}{' + isotonic calibration' if cal is not None else ''})")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3].split(","))
