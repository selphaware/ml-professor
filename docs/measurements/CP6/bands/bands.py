"""CP6: the bands of the two new score checks, and what the grouped scheme reports.

    python -B bands.py <gate script> <reference parent> <data seeds> <out.jsonl>
    python -B bands.py table <out.jsonl>

For each data seed (the gate's base seed 1115, then 1215, ...) the datasets
`entities` and `panel` are written by the gate's own generators and profiled as
the gate profiles them (fast profile, tool seed 0, one core), by the reference
and by the broken copy each check must catch: F10 (the split ignores the
entity column) on `entities`, F8b (the entities are not used for the baseline)
on `panel`. Each variant runs in a process of its own, since the switch is read
when the package is imported. Nothing is written inside the repo.
"""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(r"E:\CODE\ml-professor")
CASES = (("entities", {"group_col": "customer"}, ("good", "F10", "M21")),
         ("panel", {"time_col": "date", "group_col": "series"}, ("good", "F8b")))


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


def worker(gate_path, parent, broken, n_seeds):
    """One variant of the reference, every seed. Prints one JSON line per run."""
    sys.path.insert(0, str(parent))
    from mlprofessor import _variant
    _variant.BROKEN = None if broken == "good" else broken
    import mlprofessor
    from mlprofessor import core, errors
    core.BROKEN = _variant.BROKEN
    gate = gate_module(gate_path)
    folder = Path(tempfile.mkdtemp(prefix="cp6-bands-"))
    warnings.simplefilter("ignore")
    for k in range(n_seeds):
        base = gate.SEED + 100 * k
        for dataset, kw, variants in CASES:
            if broken not in variants:
                continue
            path = folder / f"{dataset}.csv"
            path.write_text(gate.dataset_text(dataset, base), encoding="utf-8", newline="")
            line = {"dataset": dataset, "variant": broken, "k": k}
            try:
                result = mlprofessor.profile(str(path), gate.DATASETS[dataset][1], profile="fast",
                                             seed=0, n_jobs=1, **kw)
            except errors.ProfileError as exc:
                line["outcome"] = exc.code
            else:
                d = result.to_dict()
                h, v = d["headline"], d["validation"]
                line.update(outcome="result", skill=h["skill"], model=h["model"],
                            baseline=h["baseline"], scheme=v["scheme"],
                            interval=h["interval_method"], low=h["skill_low"],
                            high=h["skill_high"], p_value=h["p_value"],
                            holdout=len(result.predictions),
                            groups=[v["n_groups_training"], v["n_groups_holdout"]],
                            roles={c["name"]: c["role"] + ("/" + c["reason"] if c["reason"] else "")
                                   for c in d["columns"]},
                            warnings=sorted({w["code"] for w in d["warnings"]}))
            print(json.dumps(line), flush=True)


def measure(gate_path, parent, n_seeds, out):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    procs = [subprocess.Popen([sys.executable, "-B", __file__, "worker", gate_path, parent, variant,
                               str(n_seeds)], stdout=subprocess.PIPE, text=True, env=env)
             for variant in ("good", "F10", "F8b", "M21")]
    with open(out, "w", encoding="utf-8") as fh:
        for proc in procs:
            fh.write(proc.communicate()[0])
    print("wrote", out)


def table(out):
    import statistics
    lines = [json.loads(line) for line in Path(out).read_text(encoding="utf-8").splitlines()]
    for dataset in ("entities", "panel"):
        for variant in ("good", "F10", "F8b", "M21"):
            found = [l for l in lines if l["dataset"] == dataset and l["variant"] == variant]
            if not found:
                continue
            ran = [l for l in found if l["outcome"] == "result"]
            skills = [l["skill"] for l in ran]
            print(f"{dataset} / {variant}: {len(found)} runs, {len(ran)} finished; outcomes "
                  f"{sorted({l['outcome'] for l in found})}")
            if skills:
                print(f"   skill {min(skills):+.4f} to {max(skills):+.4f} (mean "
                      f"{statistics.mean(skills):+.4f}, sd {statistics.pstdev(skills):.4f}); "
                      f"baselines {sorted({l['baseline'] for l in ran})}; models "
                      f"{sorted({l['model'] for l in ran})}; schemes {sorted({l['scheme'] for l in ran})}")
                print(f"   intervals {sorted({str(l['interval']) for l in ran})}; skill inside its "
                      f"interval's zero: "
                      f"{sum(1 for l in ran if l['low'] is not None and l['low'] <= 0 <= l['high'])}"
                      f" of {sum(1 for l in ran if l['low'] is not None)}; p-values below 0.05: "
                      f"{sum(1 for l in ran if l['p_value'] is not None and l['p_value'] < 0.05)} of "
                      f"{sum(1 for l in ran if l['p_value'] is not None)}")
                print(f"   holdout rows {min(l['holdout'] for l in ran)} to "
                      f"{max(l['holdout'] for l in ran)}; holdout entities "
                      f"{min(l['groups'][1] for l in ran)} to {max(l['groups'][1] for l in ran)}; "
                      f"warnings {sorted({w for l in ran for w in l['warnings']})}")
                print(f"   roles {sorted({json.dumps(l['roles'], sort_keys=True) for l in ran})[:2]}")


if __name__ == "__main__":
    if sys.argv[1] == "worker":
        worker(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]))
    elif sys.argv[1] == "table":
        table(sys.argv[2])
    else:
        measure(sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4])
