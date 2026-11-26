"""CP10, section A: on every run whose headline is at or above 0.98, the falls of every
column, in order: where a rule for naming columns would cut.

    python -B falls_near_perfect.py <json> [<json> ...]
"""
import sys
from collections import defaultdict
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lab10                                            # noqa: E402

rows = defaultdict(list)
for path in sys.argv[1:]:
    for r in lab10.load(path):
        if "error" in r or r["skill"] is None or r["skill"] < 0.98:
            continue
        falls = sorted(((v[2], n) for n, v in r["rests"].items() if v[2] is not None), reverse=True)
        rows[(r.get("run") or r.get("make"), r.get("change") or str(r.get("params")))].append(falls)

print("| table | runs at or above 0.98 | largest fall | second | third | fourth | fifth | sixth | "
      "columns with a fall of 0.02 or more (range) | of 0.05 or more | of 0.10 or more | "
      "largest fall below 0.02 |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for key, runs in rows.items():
    cells = []
    for rank in range(6):
        have = [f[rank][0] for f in runs if len(f) > rank]
        cells.append(f"{min(have):+.3f} to {max(have):+.3f}" if have else "-")
    counts = []
    for cut in (0.02, 0.05, 0.10):
        each = [sum(v >= cut for v, _ in f) for f in runs]
        counts.append(f"{min(each)} to {max(each)}")
    below = [max((v for v, _ in f if v < 0.02), default=float("nan")) for f in runs]
    below = [b for b in below if b == b]
    print(f"| {key[0]} {key[1] if key[1] != 'None' else ''} | {len(runs)} | " + " | ".join(cells)
          + " | " + " | ".join(counts)
          + f" | {max(below):+.4f} |" if below else " | - |")
