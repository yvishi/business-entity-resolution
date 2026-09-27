"""Decisions: one-sided unique mapping + per-S1 set selection; exact competition metric."""
import numpy as np, polars as pl
from sklearn.isotonic import IsotonicRegression

B2 = 0.25  # beta^2 for F0.5


def fit_calibration(p, y):
    """Isotonic recalibration of predicted probabilities against OOF labels. The 'expf' decision rule sums raw
    probability mass (not just rank) to pick the expected-F0.5-optimal set size, so miscalibrated scores bias that
    choice directly -- this matters most for singletons, where the whole per-S1 score is binary (1.0 or 0.0)."""
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(p, y)
    return iso


def apply_calibration(iso, p):
    return iso.predict(np.asarray(p, dtype=np.float64)).astype(np.float32) if iso is not None else p


def f05(pred, truth):
    if not truth: return 1.0 if not pred else 0.0
    if not pred: return 0.0
    tp = len(pred & truth)
    if tp == 0: return 0.0
    p, r = tp / len(pred), tp / len(truth)
    return (1 + B2) * p * r / (B2 * p + r)


def macro_f05(pred: dict, truth: dict, s1_ids) -> float:
    return float(np.mean([f05(pred.get(x, set()), truth.get(x, set())) for x in s1_ids]))


def unique_map(d: pl.DataFrame, p="p", margin=0.0) -> pl.DataFrame:
    """Each target keeps only its best S1 (ground truth: every S2/S3 record belongs to <= 1 S1)."""
    d = d.with_columns(pl.col(p).max().over("t").alias("_best"))
    second = d.group_by("t").agg(pl.col(p).sort(descending=True).get(1, null_on_oob=True).alias("_second"))
    d = d.join(second, on="t", how="left").with_columns(pl.col("_second").fill_null(0.0))
    return d.filter((pl.col(p) == pl.col("_best")) & (pl.col(p) - pl.col("_second") >= margin)).drop("_best", "_second")


def select(d: pl.DataFrame, p="p", mode="expf", tau=0.5, extra_mass=0.0) -> dict:
    """mode 'thr': keep p >= tau. mode 'expf': per S1 choose the top-k maximising expected F0.5 (k=0 allowed)."""
    out = {}
    if mode == "thr":
        for s, t in d.filter(pl.col(p) >= tau).select("s1", "t").iter_rows():
            out.setdefault(s, set()).add(t)
        return out
    g = d.sort(["s1", p], descending=[False, True]).group_by("s1", maintain_order=True).agg(pl.col("t"), pl.col(p))
    for s, ts, ps in g.iter_rows():
        ps = np.asarray(ps, dtype=np.float64)
        tot = ps.sum() + extra_mass
        cs = np.cumsum(ps)
        k = np.arange(1, len(ps) + 1)
        ef = (1 + B2) * cs / (B2 * tot + k)
        e0 = float(np.prod(1 - ps))
        best = int(np.argmax(ef))
        if ef[best] > e0 and ps[0] >= tau:
            out[s] = set(ts[: best + 1])
    return out
