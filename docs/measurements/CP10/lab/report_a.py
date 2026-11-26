"""CP10, section A: two columns that give the target together, and copies of a strong column.

    python -B report_a.py <a_two.json> <a_dup.json>
"""
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lab10                                            # noqa: E402

two, dup = lab10.load(sys.argv[1]), lab10.load(sys.argv[2])


def named(record, column):
    return any(code == "near_perfect_headline" and repr(column) in message
               for code, _, message in record["texts"])


print("## `pair_a` = target + z and `pair_b` = z added (30 seeds)\n")
print("| run | scheme | model chosen | headline skill | at or above 0.98 | warning raised | both named | "
      "fall of pair_a | fall of pair_b | largest fall of any other column | either excluded or warned |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
by = defaultdict(list)
for r in two:
    by[r["run"]].append(r)
for run, recs in by.items():
    done = [r for r in recs if "error" not in r]
    errors = Counter(r["error"] for r in recs if "error" in r)
    if not done:
        print(f"| {run} | | | {dict(errors)} | | | | | | | |")
        continue
    skills = [r["skill"] for r in done if r["skill"] is not None]
    high = sum(s >= 0.98 for s in skills)
    warned = sum(any(c == "near_perfect_headline" for c, _ in r["warnings"]) for r in done)
    both = sum(named(r, "pair_a") and named(r, "pair_b") for r in done)
    fa = [r["rests"]["pair_a"][2] for r in done if "pair_a" in r["rests"]]
    fb = [r["rests"]["pair_b"][2] for r in done if "pair_b" in r["rests"]]
    other = [max((v[2] for n, v in r["rests"].items() if n not in ("pair_a", "pair_b")
                  and v[2] is not None), default=None) for r in done]
    screened = sum(any(lab10.fate(r, c) in ("excluded", "warned") for c in ("pair_a", "pair_b"))
                   for r in done)
    models = Counter(r["model"].split("-")[0] for r in done)
    print(f"| {run} | {done[0]['scheme']} | {dict(models)} | "
          f"{lab10.span(skills) if skills else 'no value'} | {high} | {warned} | {both} | "
          f"{lab10.span(fa)} | {lab10.span(fb)} | {lab10.span(other)} | {screened} "
          f"{dict(errors) if errors else ''} |")

print("\n## One, two or three copies of a strong legitimate column (30 seeds)\n")
print("| run | columns | model chosen | headline skill | fall of f1 | fall of each copy | "
      "sum of the falls of f1 and its copies |")
print("|---|---|---|---|---|---|---|")
by = defaultdict(list)
for r in dup:
    by[(r["run"], r["change"])].append(r)
for (run, change), recs in by.items():
    done = [r for r in recs if "error" not in r]
    skills = [r["skill"] for r in done]
    f1 = [r["rests"]["f1"][2] for r in done]
    copies = [v[2] for r in done for n, v in r["rests"].items() if n.startswith("f1_again")]
    total = [sum(v[2] for n, v in r["rests"].items() if n == "f1" or n.startswith("f1_again"))
             for r in done]
    models = Counter(r["model"].split("-")[0] for r in done)
    print(f"| {run} | {change or 'as it is'} | {dict(models)} | {lab10.span(skills)} | "
          f"{lab10.span(f1)} | {lab10.span(copies)} | {lab10.span(total)} |")
print("\nper model family, the fall of f1 with one exact copy beside it, as a share of its fall alone:")
alone = {(r["run"], r["k"]): r for r in dup if r["change"] is None and "error" not in r}
ratios = defaultdict(list)
zero = Counter()
for r in dup:
    if r["change"] and r["change"].endswith("f1") and "error" not in r:
        base = alone[(r["run"], r["k"])]
        family = r["model"].split("-")[0]
        ratios[(r["run"], family)].append(r["rests"]["f1"][2] / base["rests"]["f1"][2])
        zero[(r["run"], family)] += (abs(r["rests"]["f1"][2]) < 1e-12
                                     or abs(r["rests"]["f1_again"][2]) < 1e-12)
for key, values in ratios.items():
    print(f"  {key}: {lab10.span(values)} over {len(values)} tables; one of the two shows a fall "
          f"of exactly 0 on {zero[key]}")
