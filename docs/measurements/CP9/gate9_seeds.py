"""CP9: checks of the gate, run as the gate runs them, at many data seeds.

    python -B gate9_seeds.py <gate file> <package folder> <work folder> <out.json>
                             [--only id,id] [--seeds 30] [--first 0]

The gate file is loaded as a module. For k = first .. first + seeds - 1 the base seed
of every dataset is the gate's own plus 100 * k (k = 0 is the gate's seed), the
artifacts go to <work folder>/k<k>/verify/artifacts, and the named checks are run on
the package by the gate's own `run`: the same command lines, the same checks, the
same messages. Nothing is written beside the gate file or inside the package.

One record for each seed: for each check whether it passed, and its message; and for
each run those checks read, what profile.json holds (skill, model, baseline, the
roles, the warnings, the largest row's shares), or the named error of a run that
stopped.
"""
import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True
parser = argparse.ArgumentParser()
parser.add_argument("gate")
parser.add_argument("package")
parser.add_argument("work")
parser.add_argument("out")
parser.add_argument("--only", default=None)
parser.add_argument("--seeds", type=int, default=30)
parser.add_argument("--first", type=int, default=0)
args = parser.parse_args()

gate_file, work = Path(args.gate).resolve(), Path(args.work).resolve()
package = Path(args.package).resolve()
sys.path.insert(0, str(gate_file.parent))
spec = importlib.util.spec_from_file_location("mlprof_gate", str(gate_file))
g = importlib.util.module_from_spec(spec)
sys.argv = [str(gate_file)]
spec.loader.exec_module(g)
only = [i.strip() for i in args.only.split(",")] if args.only else list(g.CHECK_RUNS)
runs = []
for check_id in only:
    runs += [r for r in g.CHECK_RUNS[check_id] if r not in runs]


def of_run(folder, message):
    path = folder / "profile.json"
    if not path.is_file():
        found = re.search(r"error: (\w+):", message or "")
        return {"error": found.group(1) if found else "no profile.json"}
    d = json.loads(path.read_text(encoding="utf-8"))
    h = d["headline"]
    return {
        "skill": h["skill"], "score": h["score"], "model": h["model"],
        "baseline": h["baseline"], "baseline_score": h["baseline_score"],
        "baselines": {b["name"]: b["score"] for b in d["baselines"]},
        "roles": {c["name"]: [c["role"], c["reason"], c["leak_rule"]] for c in d["columns"]},
        "warnings": [[w["code"], w["column"]] for w in d["warnings"]],
        "largest": {k: h.get(k) for k in ("largest_row_share", "largest_row",
                                          "baseline_largest_row_share", "baseline_largest_row")},
        "cv_largest": {k: d["cv"].get(k) for k in ("largest_row_share", "largest_row",
                                                   "baseline_largest_row_share",
                                                   "baseline_largest_row")},
        "models": {m["config"]: [m["cv_mean"], m.get("largest_row_share")] for m in d["models"]},
        "chosen": d["cv"]["chosen"],
        "folds": [[f["training_rows"], f["validation_rows"], f["scored"]]
                  for f in d["validation"]["folds"]],
        "rows": {r["item"]: r["count"] for r in d["rows"]},
    }


records = []
for k in range(args.first, args.first + args.seeds):
    base = g.SEED + 100 * k
    for fn in (g.dataset_seed, g.dataset_text, g.write_dataset):
        fn.__defaults__ = (base,)
    root = work / f"k{k:02d}"
    g.gate.root = root
    sys.argv = [str(root / "verify" / "smoke_test.py"), str(package), "--only", ",".join(only)]
    results = g.gate.run(package, only=only)
    by_check = {r["id"]: r for r in results}
    record = {"k": k, "base": base, "checks": {i: [by_check[i]["ok"], by_check[i]["message"][:700]]
                                               for i in only}, "runs": {}}
    for check_id in only:
        for run in g.CHECK_RUNS[check_id]:
            record["runs"].setdefault(run, of_run(root / "verify" / "artifacts" / "runs" / run,
                                                  by_check[check_id]["message"]))
    records.append(record)
    print(k, {i: by_check[i]["ok"] for i in only}, flush=True)
Path(args.out).write_bytes(json.dumps({"gate": str(gate_file), "package": str(package),
                                       "only": only, "records": records}, indent=1).encode("utf-8"))
print("written", args.out)
