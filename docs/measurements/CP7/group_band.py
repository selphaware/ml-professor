"""CP7, section B (D62): noise in a table of few-valued columns, run with
`--duplicates group`. The band of `duplicates.group_no_skill`, and the gap to F12.

    python -B group_band.py gate    <gate script> <reference parent> [--seeds 30] [--table]
    python -B group_band.py explore <reference parent> [--seeds 30] [--tables a,b,...]

`gate`: the dataset `few_valued` as the gate's own generator writes it, at each
data seed (the gate's base seed 1115, then 1215, and so on), profiled as the
gate profiles it, by the reference and by F12, in which the sets of
`duplicates="group"` are formed with the target, as built in the fifth pass:

    python -m mlprofessor profile --data few_valued.csv --target outcome --out O
           --profile fast --seed 0 --n-jobs 1 --duplicates group --no-time-order --no-entities

`explore`: the tables that were tried before `few_valued` was chosen, with their
own generator below. The label is drawn at random in all of them, so the right
skill is about zero. <reference parent> holds the reference as the folder
`mlprofessor`; `-` is the repo's own verify/reference, and for <gate script>
verify/smoke_test.py. Two throwaway copies are made outside the repo, one for
each switch. Nothing is written inside the repo.
"""
import argparse
import importlib.util
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[3]
CASES = ("good", "F12")


def gate_module(path):
    sys.path.insert(0, str(REPO / "verify"))
    spec = importlib.util.spec_from_file_location("mlprof_gate", str(path))
    module = importlib.util.module_from_spec(spec)
    saved, sys.argv = sys.argv, [str(path)]
    try:
        spec.loader.exec_module(module)
    finally:
        sys.argv = saved
    return module


def few_valued(seed, n, levels):
    """Columns of a few levels each, and a label that depends on none of them.

    `levels` as ("nested", regions, branches): a region and a branch, each branch
    in one region, so that the branch alone says which combination a row has.
    """
    rng = random.Random(seed)
    if levels[0] == "nested":
        _, regions, branches = levels
        rows = []
        for _ in range(n):
            b = rng.randrange(branches)
            rows.append([f"r{b % regions}", f"b{b:02d}", rng.choice(["no", "yes"])])
        return "\n".join(",".join(r) for r in [["region", "branch", "outcome"]] + rows) + "\n"
    names = [f"c{i + 1}" for i in range(len(levels))]
    rows = [[f"{names[i]}_{rng.randrange(k)}" for i, k in enumerate(levels)]
            + [rng.choice(["no", "yes"])] for _ in range(n)]
    return "\n".join(",".join(r) for r in [names + ["outcome"]] + rows) + "\n"


EXPLORED = {
    "a": ("three columns of 4, 3 and 2 levels: 24 combinations, 1,500 rows", 1500, (4, 3, 2)),
    "b": ("four columns of 4, 3, 3 and 2 levels: 72 combinations, 1,500 rows", 1500, (4, 3, 3, 2)),
    "c": ("four columns of 5, 4, 3 and 3 levels: 180 combinations, 1,500 rows", 1500, (5, 4, 3, 3)),
    "d": ("three columns of 6, 5 and 4 levels: 120 combinations, 1,800 rows", 1800, (6, 5, 4)),
    "e": ("one column of 16 levels, 1,500 rows", 1500, (16,)),
    "f": ("two columns of 8 and 2 levels: 16 combinations, 1,500 rows", 1500, (8, 2)),
    "g": ("three columns of 3, 2 and 2 levels: 12 combinations, 1,500 rows", 1500, (3, 2, 2)),
    "h": ("two columns of 4 and 3 levels: 12 combinations, 2,000 rows", 2000, (4, 3)),
    "i": ("two columns of 5 and 4 levels: 20 combinations, 2,000 rows", 2000, (5, 4)),
    "j": ("two columns of 8 and 2 levels: 16 combinations, 2,000 rows", 2000, (8, 2)),
    "k": ("two columns of 10 and 2 levels: 20 combinations, 2,000 rows", 2000, (10, 2)),
    "l": ("two columns of 12 and 2 levels: 24 combinations, 2,000 rows", 2000, (12, 2)),
    "m": ("one column of 16 levels, 2,000 rows", 2000, (16,)),
    "n": ("two columns of 6 and 2 levels: 12 combinations, 2,000 rows", 2000, (6, 2)),
    "o": ("one column of 12 levels, 2,000 rows", 2000, (12,)),
    "p": ("a region of 4 levels and a branch of 16, each branch in one region, 2,000 rows",
          2000, ("nested", 4, 16)),
    "q": ("a region of 5 levels and a branch of 20, each branch in one region, 2,000 rows",
          2000, ("nested", 5, 20)),
}


def one(parent, data, out):
    env = dict(os.environ)
    for name in ("PYTHONPATH", "PYTHONSTARTUP", "PYTHONHOME"):
        env.pop(name, None)
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    cmd = [sys.executable, "-B", "-m", "mlprofessor", "profile", "--data", str(data),
           "--target", "outcome", "--out", str(out), "--profile", "fast", "--seed", "0",
           "--n-jobs", "1", "--duplicates", "group", "--no-time-order", "--no-entities"]
    proc = subprocess.run(cmd, cwd=str(parent), env=env, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        return {"error": proc.stderr.strip()[:160]}
    doc = json.loads((Path(out) / "profile.json").read_text(encoding="utf-8"))
    v, h = doc["validation"], doc["headline"]
    return {"skill": h["skill"], "model": h["model"], "units": v["n_groups_holdout"],
            "same_features": v["n_rows_same_features"], "copies": v["n_holdout_copies"],
            "interval": h["interval_method"], "holdout": round(v["holdout_share"], 4),
            "low": h["skill_low"], "high": h["skill_high"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["gate", "explore"])
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--tables", default=",".join(EXPLORED))
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--table", action="store_true", help="print every seed")
    args = parser.parse_args()
    if args.mode == "gate":
        gate_path, reference = args.paths
        gate_path = REPO / "verify" / "smoke_test.py" if gate_path == "-" else Path(gate_path)
        gate = gate_module(gate_path)
        tables = {"few_valued": ("the gate's `few_valued`", None)}
    else:
        (reference,) = args.paths
        gate = None
        tables = {key: (EXPLORED[key][0], EXPLORED[key][1:]) for key in args.tables.split(",")}
    reference = REPO / "verify" / "reference" if reference == "-" else Path(reference)
    root = Path(tempfile.mkdtemp(prefix="mlprof-cp7-band-")).resolve()
    assert REPO not in root.parents
    parents = {}
    for case in CASES:
        parents[case] = root / case
        shutil.copytree(reference / "mlprofessor", parents[case] / "mlprofessor",
                        ignore=lambda folder, names: [n for n in names if n == "__pycache__"])
        (parents[case] / "mlprofessor" / "_variant.py").write_text(
            f"BROKEN = {None if case == 'good' else case!r}\n", encoding="utf-8")
    jobs = []
    for key, (_, spec) in tables.items():
        for k in range(args.seeds):
            base = 1115 + 100 * k
            data = root / f"{key}_{k}.csv"
            if spec is None:
                text = gate.dataset_text("few_valued", base)
            else:
                text = few_valued(base, *spec)
            data.write_text(text, encoding="utf-8", newline="")
            for case in CASES:
                jobs.append((key, k, case, data, root / f"out_{key}_{k}_{case}"))
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(lambda j: one(parents[j[2]], j[3], j[4]), jobs))
    by = {}
    for (key, k, case, _, _), res in zip(jobs, results):
        by.setdefault((key, case), []).append((k, res))
    print(f"{args.seeds} data seeds (base 1115, 1215, ...), fast profile, tool seed 0, one core")
    print("| Table | Copy | skill min | mean | sd | max | stops | holdout units | interval "
          "reported | interval holds 0 |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for key, (label, _) in tables.items():
        for case in CASES:
            runs = by[(key, case)]
            good = [r for _, r in runs if "skill" in r]
            skills = [r["skill"] for r in good]
            mean = sum(skills) / len(skills)
            sd = (sum((s - mean) ** 2 for s in skills) / (len(skills) - 1)) ** 0.5
            units = sorted({r["units"] for r in good})
            with_interval = [r for r in good if r["interval"]]
            holds = sum(r["low"] <= 0 <= r["high"] for r in with_interval)
            print(f"| {label} | {case} | {min(skills):+.3f} | {mean:+.3f} | {sd:.3f} | "
                  f"{max(skills):+.3f} | {len(runs) - len(good)} | {units[0]}..{units[-1]} | "
                  f"{len(with_interval)} of {len(good)} | {holds} of {len(with_interval)} |")
            if case == "good" and args.mode == "gate":
                print(f"|  | mean - 4.5 sd, mean + 4.5 sd | {mean - 4.5 * sd:+.3f} |  |  | "
                      f"{mean + 4.5 * sd:+.3f} |  |  |  |  |")
    if args.table:
        print("\n| table | copy | k | skill | model | holdout units | rows that agree with "
              "another | holdout copies | holdout share |")
        print("|---|---|---|---|---|---|---|---|---|")
        for (key, case), runs in by.items():
            for k, r in runs:
                if "skill" in r:
                    print(f"| {key} | {case} | {k} | {r['skill']:+.4f} | {r['model']} | "
                          f"{r['units']} | {r['same_features']} | {r['copies']} | "
                          f"{r['holdout']} |")
                else:
                    print(f"| {key} | {case} | {k} | stopped: {r['error']} | | | | | |")
    shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
