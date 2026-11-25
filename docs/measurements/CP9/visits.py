"""D74 (3): the one planted cell the extreme-value warning does not name. `messy`
with one cell of `visits` (whole numbers from 1,000 to 100,000) or of `spend`
(about 50, spread 10) set to -999 or 999999, 30 data seeds, three rows each:
whether the value is named, and where skill lands (the band is 0.55 to 0.91).

    python -B visits.py <package folder>
"""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lab9

CHANGES = [None, "feature:visits:-999", "feature:visits:999999", "feature:spend:-999",
           "feature:spend:999999"]
if __name__ == "__main__":
    jobs = [dict(run="messy", k=k, change=c, draw=d) for c in CHANGES for k in range(30)
            for d in ((0,) if c is None else (0, 1, 2))]
    found = lab9.execute(sys.argv[1], jobs, {}, workers=12)
    print("| `messy` | runs | the value named | skill | inside 0.55 to 0.91 |")
    print("|---|---|---|---|---|")
    for c in CHANGES:
        rows = [r for r in found if r["change"] == c and "error" not in r]
        col = c.split(":")[1] if c else None
        named = sum(any(code == "extreme_value" and name == col for code, name in r["warnings"])
                    for r in rows)
        sk = [r["skill"] for r in rows]
        print(f"| {c or 'as it is'} | {len(rows)} | {named} | {min(sk):+.3f} to {max(sk):+.3f} | "
              f"{sum(0.55 <= s <= 0.91 for s in sk)} of {len(rows)} |")
