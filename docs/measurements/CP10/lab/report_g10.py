"""CP10: the runs of the ninth pass's gate checks over 30 data seeds, on the reference and
on each broken copy: on how many seeds each check's condition holds, with the range of
the number it turns on.

    python -B report_g10.py <folder of g10_<case>.json files>

The conditions are the checks' structural assertions, read from the lab's records:
  random_walk        (see a_expected.py for the shuffled score's z)
  pair_together      skill >= 0.98, one near_perfect_headline warning, pair_a and pair_b
                     both in headline.rests_on
  signal_reg_copied  f1 and f1_scaled hold the same three numbers, both in the list
  leak_reg_coded     amount_band excluded as a suspected leak (predicts_target)
  noise_tied         score and noise_rounded features that no leak warning names
  skewed_copy        amount_copy excluded as a suspected leak (predicts_target)
  last_stamp         one holdout_far_above_quarter warning
"""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lab10                                            # noqa: E402

FOLDER = Path(sys.argv[1])
CASES = ["good", "F29", "F30", "F31", "F32", "F33", "F34", "F35"]


def codes(r, code):
    return [c for c, _ in r["warnings"] if c == code]


def holds(run, r):
    listed = {e["column"] for e in r.get("rests_on") or []}
    if run == "pair_together":
        return (r["skill"] is not None and r["skill"] >= 0.98
                and len(codes(r, "near_perfect_headline")) == 1
                and {"pair_a", "pair_b"} <= listed), r["skill"]
    if run == "signal_reg_copied":
        a, b = r["rests"].get("f1"), r["rests"].get("f1_scaled")
        return (a == b and a is not None and {"f1", "f1_scaled"} <= listed), \
            (a[2] if a else None)
    if run in ("leak_reg_coded", "skewed_copy"):
        name = "amount_band" if run == "leak_reg_coded" else "amount_copy"
        role = tuple(r["roles"].get(name, (None, None, None)))
        return role[:3] == ("excluded", "suspected_leak", "predicts_target"), r["skill"]
    if run == "noise_tied":
        named = {col for code, col in r["warnings"]
                 if code in ("strong_single_column", "suspected_leak")}
        ok = all(tuple(r["roles"].get(n, (None, None)))[:2] == ("feature", None)
                 and n not in named for n in ("score", "noise_rounded"))
        return ok, r["skill"]
    if run == "last_stamp":
        return len(codes(r, "holdout_far_above_quarter")) == 1, r.get("rows", {}).get("holdout")
    return None, None


print("| run | " + " | ".join(CASES) + " |")
print("|---|" + "---|" * len(CASES))
for run in ("pair_together", "signal_reg_copied", "leak_reg_coded", "noise_tied",
            "skewed_copy", "last_stamp"):
    cells = []
    for case in CASES:
        recs = [r for r in lab10.load(FOLDER / f"g10_{case}.json") if r["run"] == run]
        got = [holds(run, r) for r in recs if "error" not in r]
        n_ok = sum(1 for ok, _ in got if ok)
        vals = [v for _, v in got if isinstance(v, (int, float))]
        cells.append(f"{n_ok}/{len(recs)}" + (f" ({lab10.span(vals)})" if vals else ""))
    print(f"| {run} | " + " | ".join(cells) + " |")
print("\nper-seed (good): seed k, then for each run whether the condition holds and its number")
recs = {run: {r["k"]: r for r in lab10.load(FOLDER / "g10_good.json") if r["run"] == run}
        for run in ("pair_together", "signal_reg_copied", "leak_reg_coded", "noise_tied",
                    "skewed_copy", "last_stamp")}
print("| k | " + " | ".join(recs) + " |")
print("|---|" + "---|" * len(recs))
for k in range(30):
    row = []
    for run, by_k in recs.items():
        ok, v = holds(run, by_k[k])
        row.append(("yes" if ok else "NO") + (f" {v:+.4f}" if isinstance(v, float) else
                                              f" {v}" if v is not None else ""))
    print(f"| {k} | " + " | ".join(row) + " |")
