"""CP10: every gate run at 30 seeds, the reference as it stands against the prototype.

    python -B report_gate.py <gate_new.json> <gate_old.json>
"""
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lab10                                            # noqa: E402

new, old = lab10.load(sys.argv[1]), lab10.load(sys.argv[2])
assert len(new) == len(old)
FIELDS = ("error", "skill", "score", "model", "baseline", "baseline_score", "scheme", "baselines",
          "roles", "rows", "folds", "cv", "low", "high")
moved = defaultdict(list)
gained = defaultdict(Counter)
lost = defaultdict(Counter)
for a, b in zip(new, old):
    assert (a["run"], a["k"]) == (b["run"], b["k"])
    for field in FIELDS:
        if a.get(field) != b.get(field):
            moved[a["run"]].append((a["k"], field))
    wa, wb = Counter(map(tuple, a.get("warnings", []))), Counter(map(tuple, b.get("warnings", [])))
    for key, n in (wa - wb).items():
        gained[a["run"]][key] += n
    for key, n in (wb - wa).items():
        lost[a["run"]][key] += n

runs = list(dict.fromkeys(r["run"] for r in new))
print("## What moves, run by run (30 seeds each; every field but the warnings and the new keys)\n")
print("| run | seeds on which a number, a role or a row count moved | warnings gained (seeds) | "
      "warnings lost (seeds) |")
print("|---|---|---|---|")
for run in runs:
    seeds = sorted({k for k, _ in moved[run]})
    fields = sorted({f for _, f in moved[run]})
    g = "; ".join(f"{c}[{col}] x{n}" for (c, col), n in sorted(gained[run].items(), key=str)) or "-"
    lo = "; ".join(f"{c}[{col}] x{n}" for (c, col), n in sorted(lost[run].items(), key=str)) or "-"
    print(f"| {run} | {len(seeds)} {fields if seeds else ''} | {g} | {lo} |")

print("\n## The headline skill, run by run, and which runs reach 0.98\n")
print("| run | finished | skill (new) | at or above 0.98 | near_perfect_headline raised |")
print("|---|---|---|---|---|")
for run in runs:
    recs = [r for r in new if r["run"] == run]
    done = [r for r in recs if "error" not in r]
    skills = [r["skill"] for r in done if r["skill"] is not None]
    high = sum(s >= 0.98 for s in skills)
    warned = sum(any(c == "near_perfect_headline" for c, _ in r["warnings"]) for r in done)
    errors = Counter(r["error"] for r in recs if "error" in r)
    print(f"| {run} | {len(done)}{' ' + str(dict(errors)) if errors else ''} | "
          f"{lab10.span(skills) if skills else 'no value'} | {high} | {warned} |")

print("\n## The statement: the largest fall of each run, and the spread over the shuffles\n")
print("| run | features | largest fall: range over seeds | the column with it (seeds) | "
      "sd over 5 shuffles of the top column's shuffled skill: median, max | smallest fall |")
print("|---|---|---|---|---|---|")
for run in runs:
    done = [r for r in new if r["run"] == run and "error" not in r]
    tops, names, sds, lows, counts = [], Counter(), [], [], []
    for r in done:
        falls = {n: v[2] for n, v in r["rests"].items() if v[2] is not None}
        counts.append(len(r["rests"]))
        if not falls:
            continue
        top = max(falls, key=falls.get)
        tops.append(falls[top])
        lows.append(min(falls.values()))
        names[top] += 1
        scores = r["log"].get("rests", {}).get(top)
        if scores and r["baseline_score"]:
            if r["baselines"].get("chance") is not None:
                each = [2 * s - 1 for s in scores]
            else:
                each = [1 - s / r["baseline_score"] for s in scores]
            sds.append(statistics.stdev(each))
    if not tops:
        print(f"| {run} | {sorted(set(counts))} | no value | | | |")
        continue
    print(f"| {run} | {sorted(set(counts))} | {lab10.span(tops)} | "
          f"{', '.join(f'{n} ({c})' for n, c in names.most_common(3))} | "
          f"{statistics.median(sds):.4f}, {max(sds):.4f} | {min(lows):+.4f} |")

print("\n## The leak screen: the reading by levels on every clean column (new)\n")
tier = Counter()
worst = []
for r in new:
    if "error" in r:
        continue
    for name, s in r["log"].get("screen", {}).items():
        lv = s.get("levels")
        if lv is None:
            continue
        others = [s[k] for k in ("tree", "line", "direct") if s.get(k)]
        best2 = max((o[0] for o in others if o[0] is not None), default=None)
        best3 = max((o[1] for o in others if o[1] is not None), default=None)
        used = s["level_mode"] == 2
        tier[(r["run"], used)] += 1
        if lv[1] is not None:
            worst.append((lv[1], lv[0], best3, r["run"], name, r["k"], used))
worst.sort(key=lambda w: -w[0])
print("columns read by levels (used / logged only), by run:",
      {f"{run}{'' if used else ' (log)'}": n for (run, used), n in sorted(tier.items())})
print("\nthe ten largest level readings (every row scored) on clean columns:")
for lv3, lv2, best3, run, name, k, used in worst[:10]:
    print(f"  {run} k={k} {name}: levels {lv3:+.3f} (rule 2 reading {lv2:+.3f}), other readers "
          f"{best3 if best3 is None else round(best3, 3)}, {'used' if used else 'logged only'}")
changed = [(run, name, k, lv3, best3) for lv3, lv2, best3, run, name, k, used in worst
           if used and best3 is not None and (lv3 > best3) and (lv3 >= 0.90 or (lv2 or 0) >= 0.98)]
print(f"\nclean columns that the level reading moves into a tier: {len(changed)} {changed[:10]}")

print("\n## The order reading on every clean numeric column of a regression (new log)\n")
forms = ("old", "mean", "median", "random", "pairs")
low = {f: [] for f in forms}
for r in new:
    if "error" in r:
        continue
    for name, per in r["log"].get("order", {}).items():
        for f in forms:
            share, before = per[f]
            if share is not None:
                low[f].append((share, before, r["run"], name, r["k"]))
for f in forms:
    vals = sorted(low[f], key=lambda v: v[0])
    fired = [v for v in vals if v[0] <= 0.10 and (v[1] is None or v[1] > 0.10)]
    print(f"{f}: {len(vals)} columns with a share; smallest {vals[0][0]:.3f} "
          f"({vals[0][2]} {vals[0][3]} k={vals[0][4]}); both conditions hold on {len(fired)}: "
          f"{Counter((v[2], v[3]) for v in fired).most_common(6)}")
