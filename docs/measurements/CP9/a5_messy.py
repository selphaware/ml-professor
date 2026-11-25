"""CP9 A5: the gate's `messy` at 30 data seeds, as it is and with its rows shuffled
or repeated: the role of `record_id` (an identifier, 100001 + row number) and of
`visits` (a distinct whole-number measure, not in order), old rule and proposed.

    python -B a5_messy.py <package folder>
"""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lab9

CHANGES = [None, "rows_shuffled", "row_repeated", "rows_repeated_1pct", "rows_repeated_10pct"]
if __name__ == "__main__":
    jobs = [dict(run="messy", k=k, change=c) for c in CHANGES for k in range(30)]
    print("| `messy` | rule | `record_id` excluded as id_like, of 30 | `visits` a feature | skill | inside 0.55 to 0.91 | stopped |")
    print("|---|---|---|---|---|---|---|")
    for label, env in (("D19 as it is", {"MLP_A5": "old"}), ("proposed", {})):
        found = lab9.execute(sys.argv[1], jobs, env, workers=20)
        for c in CHANGES:
            rows = [r for r in found if r["change"] == c and "error" not in r]
            ids = sum(r["roles"]["record_id"][1] == "id_like" for r in rows)
            vis = sum(r["roles"]["visits"][0] == "feature" for r in rows)
            sk = [r["skill"] for r in rows]
            stops = {}
            for r in found:
                if r["change"] == c and "error" in r:
                    stops[r["error"]] = stops.get(r["error"], 0) + 1
            span = f"{min(sk):+.3f} to {max(sk):+.3f}" if sk else "none"
            print(f"| {c or 'as it is'} | {label} | {ids} of {len(rows)} | {vis} of {len(rows)} | "
                  f"{span} | {sum(0.55 <= s <= 0.91 for s in sk)} of {len(rows)} | {stops or ''} |")
