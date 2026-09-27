"""Unattended end-to-end run for a cloud sandbox (e.g. Claude Code on the web). From the repo root:

    python code/business_entity_resolution/src/cloud_run.py            # auto profile from detected RAM
    python code/business_entity_resolution/src/cloud_run.py --profile fast|final|full

1. Gets the data into train/ and test/ (from --data-url / DATA_URL, a zip in data_in/, or files already there).
2. Picks settings that fit the machine (RAM decides the profile; small machines also use v2's lighter blocking).
3. Runs run_all.py (resumable: rerun the same command after an interruption and it continues where it stopped).
4. Validates the submission with tools/validate_submission.py and collects everything into results/.
"""
import argparse, gzip, json, os, shutil, subprocess, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
FILES = {"train": ["train_source1.tsv", "train_source2.tsv", "train_source3.tsv", "train_ground_truth.tsv"],
         "test": ["test_source1.tsv", "test_source2.tsv", "test_source3.tsv"]}


def sh(cmd, **kw):
    print("$", " ".join(cmd), flush=True)
    return subprocess.run(cmd, **kw)


def is_pointer(path):
    if not os.path.exists(path) or os.path.getsize(path) > 1024: return False
    return open(path, "rb").read(64).startswith(b"version https://git-lfs")


def download(url, dest):
    """Google Drive share links need gdown (handles the large-file confirmation page); anything else: plain HTTP."""
    if "drive.google.com" in url or "docs.google.com" in url:
        sh([sys.executable, "-m", "pip", "install", "-q", "gdown"], check=True)
        import gdown
        if not gdown.download(url, dest, quiet=False, fuzzy=True):
            raise SystemExit("Google Drive download failed - is the file shared as 'Anyone with the link'?")
    else:
        with urllib.request.urlopen(url) as r, open(dest, "wb") as f:
            shutil.copyfileobj(r, f, 1 << 24)


def ensure_data(data_url=None):
    """Real TSVs must end up in <repo>/train and <repo>/test. Sources, in order: already there; a zip found in
    <repo>/data_in/ (any zip containing the contest TSVs, e.g. the official student_resource.zip, or its
    parts split with `split` - they are joined first); a zip downloaded from --data-url / DATA_URL."""
    import zipfile, glob
    paths = [os.path.join(ROOT, d, f) for d, fs in FILES.items() for f in fs]
    if all(os.path.exists(p) and not is_pointer(p) for p in paths):
        print("data: real TSVs already present", flush=True); return
    inbox = os.path.join(ROOT, "data_in"); os.makedirs(inbox, exist_ok=True)
    parts = sorted(glob.glob(os.path.join(inbox, "*.part*")))
    if parts:   # e.g. dataset.zip.part_aa, dataset.zip.part_ab
        with open(os.path.join(inbox, "joined.zip"), "wb") as fo:
            for p in parts:
                with open(p, "rb") as fi: shutil.copyfileobj(fi, fo, 1 << 24)
        print(f"data: joined {len(parts)} parts", flush=True)
    zips = glob.glob(os.path.join(inbox, "*.zip"))
    url = data_url or os.environ.get("DATA_URL")
    if not zips and url:
        print(f"data: downloading {url}", flush=True)
        download(url, os.path.join(inbox, "dataset.zip")); zips = [os.path.join(inbox, "dataset.zip")]
    if not zips:
        raise SystemExit("DATA MISSING: pass --data-url <link to the dataset zip> (or set DATA_URL), or put the zip in "
                         "data_in/, or put the TSVs in train/ and test/ at the repo root.")
    want = {f: d for d, fs in FILES.items() for f in fs}
    for z in zips:
        with zipfile.ZipFile(z) as zf:
            for info in zf.infolist():
                base = os.path.basename(info.filename)
                if base in want and "__MACOSX" not in info.filename:
                    dest = os.path.join(ROOT, want[base], base); os.makedirs(os.path.dirname(dest), exist_ok=True)
                    with zf.open(info) as fi, open(dest, "wb") as fo: shutil.copyfileobj(fi, fo, 1 << 24)
                    print(f"data: extracted {want[base]}/{base}", flush=True)
    bad = [p for p in paths if not os.path.exists(p) or is_pointer(p)]
    if bad: raise SystemExit("DATA MISSING after extraction: " + ", ".join(bad))
    print("data: OK", flush=True)


def resources():
    import psutil
    return psutil.virtual_memory().total / 2**30, os.cpu_count(), shutil.disk_usage(ROOT).free / 2**30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=["auto", "fast", "final", "full"], default="auto")
    ap.add_argument("--work", default=os.path.join(ROOT, "work"))
    ap.add_argument("--data-url", default=None, help="link to the dataset zip (Google Drive share link or direct URL)")
    a = ap.parse_args()
    t0 = time.time()
    sh([sys.executable, "-m", "pip", "install", "-q", "-r", os.path.join(HERE, "..", "requirements.txt")], check=True)
    ensure_data(a.data_url)
    ram, cpus, disk = resources()
    prof = a.profile
    if prof == "auto":
        prof = "fast" if ram < 28 else ("final" if ram < 100 else "full")
    print(f"machine: RAM {ram:.0f} GB, CPUs {cpus}, disk free {disk:.0f} GB -> profile '{prof}'", flush=True)
    if disk < 25: print("WARNING: <25 GB free disk; the run may fail on disk space.", flush=True)

    env = dict(os.environ, PYTHONUTF8="1")
    env.setdefault("BER_BLOCK", "v2" if ram < 28 else "wide")   # small box: v2's lighter candidate set
    args = [sys.executable, "-W", "ignore", os.path.join(HERE, "run_all.py"),
            "--train-dir", os.path.join(ROOT, "train"), "--test-dir", os.path.join(ROOT, "test"),
            "--work", a.work, "--out", os.path.join(a.work, "out")]
    if prof == "fast":
        args += ["--fast"]
    elif prof == "final":
        args += ["--final"]
    else:   # full: every partition, more rounds; only for big-RAM machines
        args += ["--final", "--dev-frac", "1.0"]
        env.setdefault("BER_ROUNDS", "4000"); env.setdefault("BER_LR", "0.04"); env.setdefault("BER_PATIENCE", "150")
    print(f"BER_BLOCK={env['BER_BLOCK']}", flush=True)
    r = sh(args, cwd=HERE, env=env)
    if r.returncode != 0:
        raise SystemExit(f"pipeline failed (exit {r.returncode}) - see {a.work}/run.log; rerun the same command to resume")

    out = os.path.join(a.work, "out")
    v = sh([sys.executable, os.path.join(ROOT, "tools", "validate_submission.py"),
            "--matching", os.path.join(out, "matching_results.tsv"), "--candidate", os.path.join(out, "candidate_pairs.tsv"),
            "--test-dir", os.path.join(ROOT, "test")], capture_output=True, text=True)
    print(v.stdout, v.stderr, flush=True)

    stamp = time.strftime("%Y%m%d-%H%M")
    res = os.path.join(ROOT, "results", f"{stamp}-{prof}")
    os.makedirs(res, exist_ok=True)
    # only the submission is committable: ~97 MB raw -> ~25 MB gzipped. candidate_pairs.tsv (~2 GB) stays in work/out.
    with open(os.path.join(out, "matching_results.tsv"), "rb") as fi, \
         gzip.open(os.path.join(res, "matching_results.tsv.gz"), "wb") as fo:
        shutil.copyfileobj(fi, fo)
    rep = json.load(open(os.path.join(a.work, "train_report.json"), encoding="utf-8"))
    summary = {"profile": prof, "BER_BLOCK": env["BER_BLOCK"], "ram_gb": round(ram), "cpus": cpus,
               "hours": round((time.time() - t0) / 3600, 2), "validator_pass": v.returncode == 0,
               "oof_f05_stage1": rep.get("f1"), "oof_f05_stage2": rep.get("f2"), "oof_f05_stage2_cal": rep.get("f2cal"),
               "oof_by_country_stage2": rep.get("by_country2"), "oof_by_country_stage2_cal": rep.get("by_country2cal"),
               "training_pairs": rep.get("pairs"), "training_S1": rep.get("S1")}
    json.dump(summary, open(os.path.join(res, "summary.json"), "w"), indent=1)
    shutil.copy(os.path.join(a.work, "train_report.json"), res)
    with open(os.path.join(a.work, "run.log"), encoding="utf-8", errors="replace") as f:
        open(os.path.join(res, "run_log_tail.txt"), "w", encoding="utf-8").write("".join(f.readlines()[-300:]))
    print("\nRESULTS in", res, "\n" + json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
