"""CP10: the per-seed table of a measurement set (D35), from the JSON a set wrote.

    python -B per_seed10.py <set.json> [<column> ...]

One line per tool run, tab-separated: the case (the gate run or the made table with its
parameters, the change, the scheme tag), the data seed k, the headline skill, the chosen
configuration, for each named column (default: the planted ones the set is about) its
role/reason/rule and the fall of its statement, the columns of `headline.rests_on`, the
floor, and the codes of the ninth pass's warnings that the run raised, with the column
each names. A run that stopped shows the error's code.
"""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lab10                                            # noqa: E402

PLANTED = ("target_band", "target_code", "pair_a", "pair_b", "level", "target_copy",
           "amount_band", "amount_copy", "score", "noise_rounded", "f1", "f1_again",
           "f1_scaled", "f1_negated", "f1_near", "x1", "x1_again1", "tied_noise", "lead",
           "copy", "region_renamed", "seg_renamed")
CODES = ("near_perfect_headline", "holdout_far_above_quarter", "shuffled_score_not_computed",
         "strong_single_column", "suspected_leak")


def fmt(v):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


recs = lab10.load(sys.argv[1])
wanted = sys.argv[2:]
print("\t".join(["case", "k", "skill", "model", "columns: role/reason/rule fall",
                 "rests_on", "floor", "warnings"]))
for r in recs:
    case = r.get("run") or r.get("make")
    if r.get("params"):
        case += " " + " ".join(f"{a}={b}" for a, b in sorted(r["params"].items()))
    if r.get("change"):
        case += " | " + r["change"]
    if r.get("tag"):
        case += " | " + str(r["tag"])
    if "error" in r:
        print("\t".join([case, str(r["k"]), "stop: " + str(r["error"]), "", "", "", "", ""]))
        continue
    roles, rests = r.get("roles", {}), r.get("rests", {})
    names = wanted or [n for n in roles if n in PLANTED]
    cols = []
    for n in names:
        role = roles.get(n)
        fall = (rests.get(n) or [None, None, None])[2]
        cols.append(f"{n}: " + ("/".join(fmt(x) for x in role[:3]) if role else "absent")
                    + f" {fmt(fall)}")
    listed = ",".join(f"{e['column']}" for e in (r.get("rests_on") or []))
    warned = ",".join(f"{c}[{col}]" for c, col in r.get("warnings", []) if c in CODES)
    print("\t".join([case, str(r["k"]), fmt(r.get("skill")), str(r.get("model")),
                     "; ".join(cols), listed or "-", fmt(r.get("floor")), warned or "-"]))
