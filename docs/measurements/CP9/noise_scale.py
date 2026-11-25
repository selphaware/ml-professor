"""D74 (2): how small the noise on a copy of the target must be, under time order, for
the screen as ruled to warn about it or exclude it. `random_walk`, `int_stamps` and
`panel` with a rescaled copy of the target plus normal noise of 0.5%, 1%, 2% and 5%
of the target's spread; 30 data seeds each.

    python -B noise_scale.py <package folder>
"""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lab9

SHARES = (0.005, 0.01, 0.02, 0.05)
for share in SHARES:
    lab9.CHANGES[f"noise:{share}"] = lab9._noisy_copy(share)

if __name__ == "__main__":
    runs = ("random_walk", "int_stamps", "panel")
    jobs = [dict(run=run, k=k, change=f"noise:{share}") for run in runs for share in SHARES
            for k in range(30)]
    found = lab9.execute(sys.argv[1], jobs, {}, workers=20)
    print("| run | noise, share of the target's spread | excluded / warned / silent | the "
          "column's skill against the previous value | headline skill |")
    print("|---|---|---|---|---|")
    for run in runs:
        for share in SHARES:
            rows = [r for r in found if r["run"] == run and r["change"] == f"noise:{share}"]
            x = w = k = 0
            read, sk = [], []
            for r in rows:
                role = r["roles"]["target_copy"]
                warned = any(c == "strong_single_column" and col == "target_copy"
                             for c, col in r["warnings"])
                if role[1] == "suspected_leak":
                    x += 1
                elif warned:
                    w += 1
                else:
                    k += 1
                told = ((r.get("log") or {}).get("screen", {}).get("target_copy") or {}).get("told")
                if told is not None:
                    read.append(told)
                sk.append(r["skill"])
            print(f"| {run} | {100 * share:g}% | {x} / {w} / {k} | {min(read):+.3f} to "
                  f"{max(read):+.3f} | {min(sk):+.3f} to {max(sk):+.3f} |")
