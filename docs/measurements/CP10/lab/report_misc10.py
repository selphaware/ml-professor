"""CP10: three compact tables, from the sets' JSON.

    python -B report_misc10.py r3     <r3_bands.json>      two to eight bands (D80 point 3)
    python -B report_misc10.py copies <set.json> ...       copies shuffled together (D78, D80)
    python -B report_misc10.py near   <a_near.json>        a column beside a noisy copy of it

r3: for each scheme (the three random-scheme runs together, each entity run, the three
walks together) and band count, the copy's outcome (excluded / warned / named by the
near-perfect warning / named in the list / none), the level reading and the headline.

copies: for each run, kind of copy and model family chosen, the fall of the column and
of its copy, on how many tables the two are equal, the column's fall as a share of its
fall on the same table without the copy, and on how many tables both are in the list.
Sets: d80_copies.json (exact, rescaled, negated), r2_copies.json (relabelled levels),
d80_curved.json (forest and boosting, the full profile).

near: `f1` beside `f1_near`, a copy with noise of 5% or 20% of its spread added.
"""
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lab10                                            # noqa: E402

what, paths = sys.argv[1], sys.argv[2:]

if what == "r3":
    by, lv, sk = defaultdict(Counter), defaultdict(list), defaultdict(list)
    for r in lab10.load(paths[0]):
        q = int(r["change"].split(":")[1])
        grp = r["run"] if r["tag"] == "grouped" else {"random": "random scheme (3 runs)",
                                                       "time": "walks (3 runs)"}[r["tag"]]
        by[(grp, q)][lab10.fate(r, "target_band")] += 1
        s = r.get("log", {}).get("screen", {}).get("target_band", {})
        if s.get("levels") and s["levels"][1] is not None:
            lv[(grp, q)].append(s["levels"][1])
        if r.get("skill") is not None:
            sk[(grp, q)].append(r["skill"])
    print("| scheme | bands | excluded / warned / named / listed / none | headline |")
    print("|---|---|---|---|")
    for key in sorted(by):
        c = by[key]
        print(f"| {key[0]} | {key[1]} | {c['excluded']} / {c['warned']} / {c['named']} / "
              f"{c['listed']} / {c['silent']} | {lab10.span(sk[key])} |")
elif what == "copies":
    recs = [r for p in paths for r in lab10.load(p) if "error" not in r]
    alone = {}
    for r in recs:
        if not r.get("change") and not (r.get("make") and r["params"].get("copies")):
            alone[(r.get("run") or "curved", r["k"])] = r["rests"]
    by = defaultdict(list)
    for r in recs:
        if r.get("make"):
            if not r["params"].get("copies"):
                continue
            key = ("curved, full", ("negated" if r["params"].get("negate") else "exact")
                   + f" x{r['params']['copies']}")
        elif r.get("change"):
            key = (r["run"], r["change"])
        else:
            continue
        by[key + (r["model"].split("-")[0],)].append(r)
    print("| run | the copy | family | tables | fall of the column | of its copy | one fall on "
          "| share of the fall alone | both named |")
    print("|---|---|---|---|---|---|---|---|---|")
    for key in sorted(by):
        g = by[key]
        col = "x1" if key[0].startswith("curved") else (
            "region" if "region" in key[1] else "seg" if "seg" in key[1] else "f1")
        others = [n for n in g[0]["rests"] if n != col and n.startswith(col)]
        a = [r["rests"][col][2] for r in g]
        b = [r["rests"][n][2] for r in g for n in others]
        same = sum(all(r["rests"][n] == r["rests"][col] for n in others) for r in g)
        share = [r["rests"][col][2] / alone[(r.get("run") or "curved", r["k"])][col][2]
                 for r in g if (alone.get((r.get("run") or "curved", r["k"])) or {}).get(col)
                 and alone[(r.get("run") or "curved", r["k"])][col][2]]
        named = sum({col, *others} <= {e["column"] for e in r["rests_on"]} for r in g)
        print(f"| {key[0]} | {key[1]} | {key[2]} | {len(g)} | {lab10.span(a)} | "
              f"{lab10.span(b) if b else '-'} | {same} | {lab10.span(share, 2) if share else '-'}"
              f" | {named} |")
elif what == "near":
    recs = [r for r in lab10.load(paths[0]) if "error" not in r]
    alone = {(r["run"], r["k"]): r["rests"]["f1"][2] for r in recs if not r.get("change")}
    by = defaultdict(list)
    for r in recs:
        if r.get("change"):
            by[(r["run"], r["change"])].append(r)
    print("| run | the second column | fall of f1, share of its fall alone | fall of f1_near, "
          "share | both named | either named |")
    print("|---|---|---|---|---|---|")
    for key in sorted(by):
        g = by[key]
        a = [r["rests"]["f1"][2] / alone[(r["run"], r["k"])] for r in g]
        b = [r["rests"]["f1_near"][2] / alone[(r["run"], r["k"])] for r in g]
        named = [{e["column"] for e in r["rests_on"]} for r in g]
        print(f"| {key[0]} | {key[1]} | {lab10.span(a, 2)} | {lab10.span(b, 2)} | "
              f"{sum({'f1', 'f1_near'} <= n for n in named)} | "
              f"{sum(bool({'f1', 'f1_near'} & n) for n in named)} |")
else:
    raise SystemExit(__doc__)
