"""CP9: no seed and no table of the gate moved.

    python -B gate9_tables.py <the gate as it was> <the new gate>

Loads both files as modules and compares, for every dataset, run, band and check
of the gate as it was: the dataset seed, the table byte for byte, the run, the
band and the runs each check reads. Prints what the new gate adds.
"""
import importlib.util
import sys
from pathlib import Path

mods = []
for label, arg in (("was", sys.argv[1]), ("new", sys.argv[2])):
    path = Path(arg).resolve()
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("gate_" + label, str(path))
    module = importlib.util.module_from_spec(spec)
    saved, sys.argv = sys.argv, [str(path)]
    spec.loader.exec_module(module)
    sys.argv = saved
    mods.append(module)
was, new = mods
print("datasets", len(was.DATASETS), "->", len(new.DATASETS), "| runs", len(was.RUNS), "->",
      len(new.RUNS), "| fast runs", sum(r[2] == "fast" for r in was.RUNS.values()), "->",
      sum(r[2] == "fast" for r in new.RUNS.values()), "| checks", len(was.CHECK_RUNS), "->",
      len(new.CHECK_RUNS), "| failure modes", len(was.gate._modes), "->", len(new.gate._modes))
print("order of the datasets and of the runs kept:",
      list(new.DATASETS)[:len(was.DATASETS)] == list(was.DATASETS)
      and list(new.RUNS)[:len(was.RUNS)] == list(was.RUNS))
print("every dataset seed kept:", all(was.dataset_seed(n) == new.dataset_seed(n)
                                      for n in was.DATASETS))
print("every table byte for byte:", all(was.dataset_text(n) == new.dataset_text(n)
                                        for n in was.DATASETS))
print("every run kept:", all(was.RUNS[r] == new.RUNS[r] for r in was.RUNS))
print("every band kept:", all(was.BANDS[b] == new.BANDS[b] for b in was.BANDS),
      "| new bands:", {b: new.BANDS[b] for b in new.BANDS if b not in was.BANDS})
print("the runs of every check kept:", all(was.CHECK_RUNS[c] == new.CHECK_RUNS[c]
                                           for c in was.CHECK_RUNS))
print("new datasets and their seeds:", {n: new.dataset_seed(n) for n in new.DATASETS
                                        if n not in was.DATASETS})
print("new runs:", {r: new.RUNS[r][:2] for r in new.RUNS if r not in was.RUNS})
print("new checks:", {c: (new.CHECK_RUNS[c], new.gate._checks[c]["catches"])
                      for c in new.CHECK_RUNS if c not in was.CHECK_RUNS})
