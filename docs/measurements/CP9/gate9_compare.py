"""CP9: what moved between two gate runs, run by run.

    python -B gate9_compare.py <runs folder A> <runs folder B> [--old-runs-only <gate as it was>]

Each folder is a copy of verify/artifacts/runs after a full gate run: one folder per
run, with profile.json and predictions.csv when the run finished. For each run
both hold, every value of profile.json is compared, leaf by leaf, apart from
`meta.timing`, and predictions.csv byte for byte. Lists of records (columns,
models, baselines, rows, warnings, folds) are compared record by record.

Printed: the runs present on one side only, the runs that stopped on both, and
for each run that finished on both, the keys only B has, the keys only A has,
and every value that differs. Warnings are compared as the list of (code,
column, message): a warning added or taken away, or a message that changed.
"""
import json
import sys
from pathlib import Path

a_dir, b_dir = Path(sys.argv[1]), Path(sys.argv[2])
only = None
if "--old-runs-only" in sys.argv:
    import importlib.util
    path = Path(sys.argv[sys.argv.index("--old-runs-only") + 1]).resolve()
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("gate_was", str(path))
    module = importlib.util.module_from_spec(spec)
    saved, sys.argv = sys.argv, [str(path)]
    spec.loader.exec_module(module)
    sys.argv = saved
    only = set(module.RUNS)


def leaves(node, path=""):
    if isinstance(node, dict):
        for key, value in node.items():
            yield from leaves(value, f"{path}.{key}" if path else str(key))
    elif isinstance(node, list):
        if not node:
            yield path, []
        for i, value in enumerate(node):
            yield from leaves(value, f"{path}[{i}]")
    else:
        yield path, node


def load(folder):
    path = folder / "profile.json"
    if not path.is_file():
        return None
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["meta"].pop("timing", None)
    return doc


names = sorted({p.name for p in a_dir.iterdir() if p.is_dir()}
               | {p.name for p in b_dir.iterdir() if p.is_dir()})
same, moved, stopped, added_only = [], [], [], []
for name in names:
    if only is not None and name not in only:
        continue
    fa, fb = a_dir / name, b_dir / name
    if not fa.is_dir() or not fb.is_dir():
        print(f"{name}: present only in {'A' if fa.is_dir() else 'B'}")
        continue
    da, db = load(fa), load(fb)
    if da is None and db is None:
        stopped.append(name)
        continue
    if da is None or db is None:
        print(f"{name}: finished only in {'A' if da else 'B'}")
        moved.append(name)
        continue
    wa = [(w["code"], w["column"], w["message"]) for w in da.pop("warnings")]
    wb = [(w["code"], w["column"], w["message"]) for w in db.pop("warnings")]
    la, lb = dict(leaves(da)), dict(leaves(db))
    only_b = sorted(set(lb) - set(la))
    only_a = sorted(set(la) - set(lb))
    differ = sorted(k for k in set(la) & set(lb) if la[k] != lb[k])
    pa, pb = (fa / "predictions.csv").read_bytes(), (fb / "predictions.csv").read_bytes()
    lines = []
    if differ:
        lines += [f"    {k}: {la[k]!r} -> {lb[k]!r}" for k in differ[:25]]
        if len(differ) > 25:
            lines.append(f"    ... and {len(differ) - 25} more values")
    if pa != pb:
        lines.append("    predictions.csv differs")
    gone = [w for w in wa if w not in wb]
    new = [w for w in wb if w not in wa]
    for w in gone:
        lines.append(f"    warning only in A: {w[0]} [{w[1]}] {w[2][:160]}")
    for w in new:
        lines.append(f"    warning only in B: {w[0]} [{w[1]}] {w[2][:160]}")
    if only_a:
        lines.append(f"    keys only in A: {only_a[:12]}")
    if lines:
        moved.append(name)
        print(f"{name}: MOVED")
        print(chr(10).join(lines))
    elif only_b:
        added_only.append(name)
    else:
        same.append(name)
    if only_b:
        short = sorted({k.split("[")[0] + ("[]." + k.split("].")[1] if "]." in k else "")
                        for k in only_b})
        print(f"{name}: keys only in B: {short}")
print()
print(f"identical in every value: {len(same)} runs: {same}")
print(f"identical in every value A has, with keys only B has: {len(added_only)} runs: {added_only}")
print(f"stopped on both sides (no profile.json): {len(stopped)} runs: {stopped}")
print(f"moved: {len(moved)} runs: {moved}")
