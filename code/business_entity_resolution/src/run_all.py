"""End-to-end pipeline: raw TSVs -> validation report (k-fold OOF on train) -> test submission files.

    python run_all.py --final --train-dir <train> --test-dir <test> --work <work> --out <out>

--final  : the long run (~5-7 h on an 8-core / 16 GB laptop): 30% of (country, state) partitions for training
           (automatically capped to fit RAM by dropping whole partitions), 3 folds, lr 0.05, early stopping.
--fast   : quicker/smaller settings (10% partitions, 800 rounds, lr 0.12).
Safety:
  * every step writes a marker (<work>/.done_<step>); rerunning the SAME command resumes after the last finished step
    (blocking and prediction also resume inside a step, per partition);
  * before the long steps, a PREFLIGHT runs every step on a small sample of the real data (~3-5 min) and checks the
    output format; the long run starts only if it passes;
  * disk / RAM are checked up front; the training size is capped automatically to the available RAM.
Steps: parquet, prepare, block_train, train, block_test, predict (default all). Extra: prepare_test, decide.
"""
import argparse, os, sys, time, json, shutil, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ber import io_raw, prepare, block, train, predict

STEPS = ["parquet", "prepare", "block_train", "train", "block_test", "predict"]


class Tee:
    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8"); self.o = sys.stdout
    def write(self, s): self.o.write(s); self.f.write(s); self.f.flush()
    def flush(self): self.o.flush(); self.f.flush()


def checks(work):
    free = shutil.disk_usage(work).free / 2**30
    try:
        import psutil
        ram = psutil.virtual_memory().total / 2**30; avail = psutil.virtual_memory().available / 2**30
    except Exception:
        ram = avail = float("nan")
    print(f"disk free {free:.0f} GB | RAM total {ram:.0f} GB, available {avail:.0f} GB | CPUs {os.cpu_count()}", flush=True)
    if free < 15: raise SystemExit(f"Only {free:.0f} GB free on the work drive - need at least ~35 GB. Free space or use another --work drive.")
    if free < 35: print("WARNING: less than 35 GB free on the work drive; the full run needs ~25-35 GB.", flush=True)
    if avail == avail and avail < 6: print("WARNING: less than 6 GB RAM available - close other programs (browsers) before the run.", flush=True)


def preflight(work):
    """Run every step on a small sample of the real (prepared) data and validate the outputs."""
    import polars as pl
    pf = os.path.join(work, "_preflight"); shutil.rmtree(pf, ignore_errors=True); os.makedirs(os.path.join(pf, "raw"))
    for f in ("india_state_surface.json", "translit.json"):
        if os.path.exists(os.path.join(work, f)): shutil.copy(os.path.join(work, f), pf)
    parts_tr, parts_te = ["VT", "WY", "Punjab"], ["VT", "Punjab", "PDL", "HDF"]
    sel = lambda df, parts: df.filter(pl.col("st").is_in(parts) | (pl.col("st").is_null() & (pl.col("entity_id").hash(seed=9) % 100 < 1)))
    s1 = pl.read_parquet(f"{work}/train_s1.parquet").filter(pl.col("st").is_in(parts_tr))
    s1.write_parquet(f"{pf}/train_s1.parquet")
    for i in "23": sel(pl.read_parquet(f"{work}/train_s{i}.parquet"), parts_tr).write_parquet(f"{pf}/train_s{i}.parquet")
    L = pl.read_parquet(f"{work}/raw/train_links.parquet"); L.filter(pl.col("s1").is_in(s1["entity_id"].implode())).write_parquet(f"{pf}/raw/train_links.parquet")
    t1 = pl.read_parquet(f"{work}/test_s1.parquet").filter(pl.col("st").is_in(parts_te))
    t1 = t1.filter((pl.col("country") != "France") | (pl.col("entity_id").hash(seed=4) % 100 < 3))
    t1.write_parquet(f"{pf}/test_s1.parquet")
    for i in "23":
        tp = sel(pl.read_parquet(f"{work}/test_s{i}.parquet"), parts_te)
        tp.filter((pl.col("country") != "France") | (pl.col("entity_id").hash(seed=4) % 100 < 3)).write_parquet(f"{pf}/test_s{i}.parquet")
    saved = (train.ROUNDS, train.PATIENCE, train.MAX_PAIRS, os.environ.get("BER_FOLDS"))
    # small pair budget on purpose: also exercises the RAM-cap path (drops whole partitions) used by big runs
    train.ROUNDS, train.PATIENCE, train.MAX_PAIRS = 40, 10, 350_000; os.environ["BER_FOLDS"] = "2"
    try:
        block.main(pf, "train", ["US", "India"], 1.0)
        train.main(pf, os.path.join(pf, "raw"), ["US", "India"])
        block.main(pf, "test", ["US", "India", "France"], 1.0)
        predict.main(pf, os.path.join(pf, "out"))
    finally:
        train.ROUNDS, train.PATIENCE, train.MAX_PAIRS = saved[:3]
        if saved[3] is None: os.environ.pop("BER_FOLDS", None)
        else: os.environ["BER_FOLDS"] = saved[3]
    # validate outputs like the official validator
    need = set(t1["entity_id"].to_list())
    for name, col in (("matching_results.tsv", "matched_entity_ids"), ("candidate_pairs.tsv", "candidate_entity_ids")):
        rows = [l.rstrip("\n").split("\t") for l in open(os.path.join(pf, "out", name), encoding="utf-8")]
        assert rows[0] == ["source1_entity_id", col], f"bad header in {name}"
        ids = [r[0] for r in rows[1:]]
        assert len(ids) == len(set(ids)) and set(ids) == need, f"{name}: rows do not match test S1"
        for r in rows[1:]:
            got = [x for x in r[1].split(",") if x] if len(r) > 1 else []
            assert len(got) == len(set(got)) and all(x[:3] in ("S2-", "S3-") for x in got), f"{name}: bad id list"
    rep = json.load(open(os.path.join(pf, "train_report.json"), encoding="utf-8"))
    print(f"PREFLIGHT PASSED: every step ran on real data samples (US, India, France); outputs valid. "
          f"(tiny-model F0.5 {rep['f2']:.3f} - not meaningful)", flush=True)
    shutil.rmtree(pf, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", required=True); ap.add_argument("--test-dir", required=True)
    ap.add_argument("--work", default="work"); ap.add_argument("--out", default="output")
    ap.add_argument("--dev-frac", type=float, default=None)
    ap.add_argument("--test-frac", type=float, default=1.0, help="only for smoke tests")
    ap.add_argument("--steps", default="all")
    ap.add_argument("--decision", default=None, help="for --steps decide, e.g. 'thr,0.6,uniq' or 'expf,0.2,uniq'")
    ap.add_argument("--fast", action="store_true", help="smaller/faster settings")
    ap.add_argument("--final", action="store_true", help="long-run settings (recommended)")
    ap.add_argument("--no-preflight", action="store_true")
    a = ap.parse_args()
    if a.final:
        if a.dev_frac is None: a.dev_frac = 0.3
        os.environ.setdefault("BER_FOLDS", "3"); os.environ.setdefault("BER_ROUNDS", "3000")
        os.environ.setdefault("BER_LR", "0.05"); os.environ.setdefault("BER_PATIENCE", "100")
    elif a.fast:
        if a.dev_frac is None: a.dev_frac = 0.1
        os.environ.setdefault("BER_FOLDS", "3"); os.environ.setdefault("BER_ROUNDS", "800"); os.environ.setdefault("BER_LR", "0.12")
    train.ROUNDS = int(os.environ.get("BER_ROUNDS", train.ROUNDS)); train.PATIENCE = int(os.environ.get("BER_PATIENCE", train.PATIENCE))
    train.PARAMS["learning_rate"] = float(os.environ.get("BER_LR", train.PARAMS["learning_rate"]))
    if a.dev_frac is None: a.dev_frac = 1.0
    os.makedirs(a.work, exist_ok=True)
    sys.stdout = Tee(os.path.join(a.work, "run.log"))
    steps = STEPS if a.steps == "all" else a.steps.split(",")
    auto = a.steps == "all"
    raw = os.path.join(a.work, "raw"); t0 = time.time()
    log = lambda s: print(f"\n===== {s} [{time.time()-t0:.0f}s]", flush=True)
    done = lambda s: os.path.exists(os.path.join(a.work, f".done_{s}"))
    mark = lambda s: open(os.path.join(a.work, f".done_{s}"), "w").close()
    print(f"settings: dev_frac={a.dev_frac} folds={os.environ.get('BER_FOLDS', '5')} rounds={train.ROUNDS} "
          f"lr={train.PARAMS['learning_rate']} patience={train.PATIENCE}", flush=True)
    checks(a.work)

    def run(step, fn):
        if auto and done(step): print(f"===== {step}: already done (resume)", flush=True); return
        log(step); fn(); mark(step)

    run_all_steps = {
        "parquet": lambda: io_raw.tsv_to_parquet(a.train_dir, a.test_dir, raw),
        "prepare": lambda: prepare.main(raw, a.work),
        "block_train": lambda: block.main(a.work, "train", ["US", "India"], a.dev_frac),
        "train": lambda: train.main(a.work, raw, ["US", "India"]),
        "block_test": lambda: block.main(a.work, "test", ["US", "India", "France"], a.test_frac),
        "predict": lambda: predict.main(a.work, a.out),
    }
    for s in steps:
        if s == "block_train" and auto and not a.no_preflight and not done("preflight"):
            log("preflight (small end-to-end run on real data samples)")
            preflight(a.work); mark("preflight")
        if s in run_all_steps: run(s, run_all_steps[s])
        elif s == "prepare_test": log(s); prepare.main(raw, a.work, splits=("test",))
        elif s == "decide":
            log("decide")
            dec = None
            if a.decision:
                m, t, u = a.decision.split(","); dec = (m, float(t), u)
            predict.decide_only(a.work, a.out, dec)
    log("done")
    if os.path.exists(os.path.join(a.work, "train_report.json")):
        r = json.load(open(os.path.join(a.work, "train_report.json"), encoding="utf-8"))
        print(f"validation (OOF) F0.5: stage1 {r['f1']:.4f}, stage2 {r['f2']:.4f}, by country {r['by_country2']}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        print("\nFAILED - rerun the same command to resume from the last finished step (send me the lines above).", flush=True)
        sys.exit(1)
