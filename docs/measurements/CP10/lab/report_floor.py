"""CP10, R4 (D78 point 5): how few scored rows may a reading by levels rest on?

    python -B report_floor.py <b_floor.json> [<more json> ...]

Every column here is noise beside a target of noise, read by its levels with no floor.
For each column: the rows scored (validation rows whose level the fold's training rows
hold) and the reading. For a regression the columns are put by the rows scored; for a
classification by the scored rows of the rarer class. Printed: how many columns, the
largest reading, and how many read at or above 0.90 (a warning) and 0.98 (an
exclusion).
"""
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lab10                                            # noqa: E402

recs = [r for path in sys.argv[1:] for r in lab10.load(path)]
print("runs:", len(recs), "errors:", dict(Counter(r["error"] for r in recs if "error" in r)))
EDGES = {"regression": (0, 1, 5, 10, 15, 20, 30, 50, 100, 10 ** 9),
         "classification": (0, 1, 2, 3, 5, 8, 10, 15, 20, 30, 10 ** 9)}
for task in ("regression", "classification"):
    bins = defaultdict(list)
    for r in recs:
        if "error" in r or bool(r["params"].get("classification")) != (task == "classification"):
            continue
        for name, s in r["log"].get("screen", {}).items():
            raw = s.get("levels_raw")
            if not name.startswith("c") or raw is None:
                continue
            reading, scored = raw[0], raw[1]
            if task == "regression":
                rule2, by = raw[3], scored
            else:
                rule2, by = raw[0], raw[3]
            edge = next(e for e in EDGES[task][1:] if by < e)
            bins[edge].append((reading, rule2, scored, r["params"]["kind"], r["params"]["n"],
                               r["params"].get("share")))
    what = "rows scored" if task == "regression" else "scored rows of the rarer class"
    print(f"\n## {task}: columns of noise by the {what}\n")
    print(f"| {what} | columns | no value | largest reading, every row scored | largest, "
          "rule 2's reading | at or above 0.90 | at or above 0.98 | rows scored there | "
          "where the largest is |")
    print("|---|---|---|---|---|---|---|---|---|")
    low = 0
    for edge in EDGES[task][1:]:
        rows = bins.get(edge, [])
        if rows:
            have = [v for v in rows if v[0] is not None]
            top = max(have, key=lambda v: v[0]) if have else None
            top2 = max((v[1] for v in have if v[1] is not None), default=None)
            scored = [v[2] for v in rows]
            print(f"| {low} to {'more' if edge > 10 ** 8 else edge - 1} | {len(rows)} | "
                  f"{len(rows) - len(have)} | {'-' if top is None else format(top[0], '+.3f')} | "
                  f"{'-' if top2 is None else format(top2, '+.3f')} | "
                  f"{sum(max(v[0], v[1] if v[1] is not None else -9) >= 0.90 for v in have)} | "
                  f"{sum((v[1] if v[1] is not None else v[0]) >= 0.98 for v in have)} | "
                  f"{min(scored)} to {max(scored)} | {'' if top is None else top[3:]} |")
        low = edge
