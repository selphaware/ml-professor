"""CP9 A3(b): what defeats the rows set aside. `leak_reg` (its own leak column is
`amount_scaled`, a rescaled copy of the target) and `random_walk` with a rescaled
copy added, with k target cells set to -999: 1 cell, 0.5%, 1%, 2% and 5% of the
rows. On how many of 30 data seeds the copy is excluded, and whether any other
column is.

    python -B a3b_defeat.py <package folder>
"""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lab9


def cells(share):
    def change(header, rows, target, rng, info):
        usable = lab9._usable(header, rows, target)
        if "amount_scaled" not in header:
            lab9.CHANGES["copy_rescaled"](header, rows, target, rng, info)
        n = 1 if share == 0 else max(1, round(share * len(usable)))
        t = header.index(target)
        for i in rng.sample(usable, n):
            rows[i][t] = "-999"
        info["cells"] = n
    return change


SHARES = {"one cell": 0, "0.5% of the rows": 0.005, "1%": 0.01, "2%": 0.02, "5%": 0.05}
for label, share in SHARES.items():
    lab9.CHANGES["cells:" + label] = cells(share)

if __name__ == "__main__":
    jobs = [dict(run=run, k=k, change="cells:" + label)
            for run in ("leak_reg", "random_walk") for label in SHARES for k in range(30)]
    found = lab9.execute(sys.argv[1], jobs, {}, workers=24)
    print("| run | target cells set to -999 | the copy excluded, of 30 | another column excluded | errors |")
    print("|---|---|---|---|---|")
    for run in ("leak_reg", "random_walk"):
        copy = "amount_scaled" if run == "leak_reg" else "target_copy"
        for label in SHARES:
            rows = [r for r in found if r["run"] == run and r["change"] == "cells:" + label]
            ok = [r for r in rows if "error" not in r]
            out = sum(r["roles"][copy][1] == "suspected_leak" for r in ok)
            other = sum(any(v[1] == "suspected_leak" and n != copy for n, v in r["roles"].items()) for r in ok)
            errors = {}
            for r in rows:
                if "error" in r:
                    errors[r["error"]] = errors.get(r["error"], 0) + 1
            print(f"| {run} | {label} | {out} of {len(ok)} | {other} | {errors or ''} |")
