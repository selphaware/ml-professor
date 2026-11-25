"""CP9: the per-seed tables of A1, A2, A3 and A6 and of the gate's own runs (D35).

    python -B per_seed.py a1   <new.json> <before.json>
    python -B per_seed.py a2   <new.json> <before.json>
    python -B per_seed.py a3   <new.json>
    python -B per_seed.py a6   <new.json> <before.json>
    python -B per_seed.py gate <new.json> <before.json>
    python -B per_seed.py same <one.json> <other.json>

Each file is what `measure.py` wrote for one package: `new` for the rules of this
pass, `before` for the reference as it stood. Seeds are in order, k = 0 to 29 (the
data seed is 1115 + 100 k; the tool's seed is 0).

    a1    skill of each run, seed by seed, and the baseline of the headline
    a2    what became of the added column `target_copy`, seed by seed:
          X excluded as a suspected leak, w named by `strong_single_column`,
          - kept in silence, E the run stopped
    a3    one cell changed, three draws of the row for each seed. For each draw two
          letters: where the row fell (H holdout, T training) and what the run
          says: `.` skill inside the band of the run; outside it, `L` a
          `largest_row_dominates` warning, `V` an `extreme_value` warning and no
          other, `o` nothing. Needs the prototype's record of the holdout rows.
    a6    skill, the chosen configuration and the `selection_not_made` warning
    gate  skill of every run of the gate, and every field that differs
    same  compares every field two packages share, record by record
"""
import json
import sys
from collections import defaultdict

BANDS = {"signal_clf": (0.54, 0.91), "signal_reg": (0.76, 0.93), "messy": (0.55, 0.91),
         "noise_reg": (-0.65, -0.05), "random_walk": (-0.30, 0.05), "panel": (-0.04, 0.03),
         "leak_reg": (0.76, 0.93), "binary_numeric": (0.54, 0.91), "int_stamps": (-0.30, 0.05),
         "dominant_entity": (-0.03, 0.02)}
FIELDS = ("error", "skill", "score", "model", "baseline", "baseline_score", "scheme", "baselines",
          "roles", "warnings", "rows", "folds", "cv", "unique", "share", "share_row",
          "base_share", "base_share_row")


def load(path):
    return json.load(open(path, encoding="utf-8"))


def label(r):
    return (r["run"] + (" --no-entities" if r.get("tag") == "no_entities" else ""),
            r["change"] or "(as it is)")


def grouped(records):
    out = defaultdict(list)
    for r in records:
        out[label(r)].append(r)
    for rows in out.values():
        rows.sort(key=lambda r: (r["k"], r.get("draw", 0)))
    return out


def number(r):
    if "error" in r:
        return r["error"]
    return "null" if r["skill"] is None else f"{r['skill']:+.3f}"


def span(rows):
    have = [r["skill"] for r in rows if "error" not in r and r["skill"] is not None]
    stopped = sum("error" in r for r in rows)
    text = f"{min(have):+.4f} to {max(have):+.4f}" if have else "no value"
    return text + (f"; {stopped} stopped" if stopped else "")


def skills(what, new, before, also=None):
    old = grouped(before)
    for key, rows in grouped(new).items():
        print(f"\n{key[0]}, {key[1]}")
        was = old.get(key)
        if was:
            print(f"  before  {span(was)}")
            print("    " + " ".join(number(r) for r in was))
        print(f"  new     {span(rows)}")
        print("    " + " ".join(number(r) for r in rows))
        if also:
            for line in also(rows, was):
                print("  " + line)


def a1_also(rows, was):
    for name, part in (("before", was), ("new", rows)):
        if not part:
            continue
        count = defaultdict(int)
        for r in part:
            if "error" not in r:
                count[f"{r['baseline']} of {'+'.join(sorted(r['baselines']))}"] += 1
        yield f"{name}: headline baseline, of those listed: " + ", ".join(
            f"{k} on {v}" for k, v in sorted(count.items()))
    warned = sum(any(code == "autocorrelated_panel_target" for code, _ in r["warnings"])
                 for r in rows if "error" not in r)
    yield f"new: `autocorrelated_panel_target` on {warned} of {len(rows)}"


def a6_also(rows, was):
    for name, part in (("before", was), ("new", rows)):
        if not part:
            continue
        ok = [r for r in part if "error" not in r]
        yield f"{name}: chosen  " + " ".join(r["model"].split("_")[0][:6] for r in ok)
        said = sum(any(code == "selection_not_made" for code, _ in r["warnings"]) for r in ok)
        yield f"{name}: `selection_not_made` on {said} of {len(ok)}"


def tier(r):
    if "error" in r:
        return "E"
    role = r["roles"].get("target_copy")
    if role and role[1] == "suspected_leak":
        return "X"
    return "w" if any(code == "strong_single_column" and column == "target_copy"
                      for code, column in r["warnings"]) else "-"


def a2(new, before):
    old = grouped(before)
    print("| run | change | before, seeds 0 to 29 | new, seeds 0 to 29 | new: X / w / - / E |")
    print("|---|---|---|---|---|")
    for key, rows in grouped(new).items():
        if key[1] == "(as it is)":
            continue
        now = "".join(tier(r) for r in rows)
        was = "".join(tier(r) for r in old.get(key, []))
        print(f"| {key[0]} | {key[1]} | `{was}` | `{now}` | "
              + " / ".join(str(now.count(c)) for c in "Xw-E") + " |")
    print("\nA column other than `target_copy` (and other than the gate's own leak "
          "`amount_scaled`) excluded as a suspected leak:")
    found = defaultdict(lambda: [0, 0])
    for name, records in (("before", before), ("new", new)):
        for r in records:
            if "error" in r:
                continue
            for column, role in r["roles"].items():
                if role[1] == "suspected_leak" and column not in ("target_copy", "amount_scaled"):
                    found[(*label(r), column)][name == "new"] += 1
    for (run, change, column), (was, now) in sorted(found.items()):
        print(f"  {run}, {change}: `{column}` on {was} seeds before, {now} now")
    if not found:
        print("  none")


def a3(new):
    print("| run | cell changed | seeds 0 to 29, three draws each | row in holdout: runs, out of "
          "band, of those warned | row in training: runs, out of band, of those warned |")
    print("|---|---|---|---|---|")
    for key, rows in grouped(new).items():
        if key[1] == "(as it is)":
            continue
        band = BANDS.get(key[0])
        tokens, tally = defaultdict(list), {"H": [0, 0, 0], "T": [0, 0, 0]}
        for r in rows:
            if "error" in r:
                tokens[r["k"]].append("E" + r["error"][:1])
                continue
            held = r["info"].get("sentinel_row") in set((r.get("log") or {}).get("holdout_rows", []))
            where = "H" if held else "T"
            codes = {code for code, _ in r["warnings"]}
            out = band is not None and not (r["skill"] is not None
                                            and band[0] <= r["skill"] <= band[1])
            said = "L" if "largest_row_dominates" in codes else "V" if "extreme_value" in codes else "o"
            tokens[r["k"]].append(where + (said if out else "."))
            tally[where][0] += 1
            tally[where][1] += out
            tally[where][2] += out and said != "o"
        line = " ".join("".join(tokens[k]) for k in sorted(tokens))
        cells = [f"{n}, {out}, {said}" if band else f"{n} (the run has no band)"
                 for n, out, said in (tally["H"], tally["T"])]
        print(f"| {key[0]} | {key[1]} | `{line}` | {cells[0]} | {cells[1]} |")
    a3_totals(new)


GROUPS = (("regression, random scheme", ("signal_reg", "noise_reg", "leak_reg")),
          ("regression, time order", ("random_walk", "int_stamps", "panel")),
          ("classification", ("signal_clf", "messy", "binary_numeric")))


def a3_totals(new):
    """The same runs added up: where the row fell, whether the skill left the band of
    its run, and what the result then said. A statement over the selection folds is
    a `largest_row_dominates` record beyond those of the two holdout shares."""
    print("\n| cell | runs | n | row in holdout | of those out of band | of those: a "
          "`largest_row_dominates` record, `extreme_value` only, nothing | row in training | of "
          "those out of band | of those: a holdout statement, a selection-fold statement, "
          "`extreme_value` only, nothing |")
    print("|---|---|---|---|---|---|---|---|---|")
    for kind in ("feature", "target"):
        for name, runs in GROUPS:
            t = {"H": [0] * 7, "T": [0] * 7}
            n = 0
            for r in new:
                change = r["change"] or ""
                if r["run"] not in runs or "error" in r or change.endswith(":empty") \
                        or change.startswith("feature") != (kind == "feature") or not change:
                    continue
                n += 1
                held = r["info"].get("sentinel_row") in set((r.get("log") or {})
                                                            .get("holdout_rows", []))
                band = BANDS[r["run"]]
                out = not (r["skill"] is not None and band[0] <= r["skill"] <= band[1])
                codes = [code for code, _ in r["warnings"]]
                records = codes.count("largest_row_dominates")
                on_holdout = sum(isinstance(r.get(k), (int, float)) and r[k] > 0.5
                                 for k in ("share", "base_share"))
                slot = t["H" if held else "T"]
                slot[0] += 1
                slot[1] += out
                if out:
                    slot[2] += on_holdout > 0
                    slot[3] += records > on_holdout
                    slot[4] += records == 0 and "extreme_value" in codes
                    slot[5] += records == 0 and "extreme_value" not in codes
                    slot[6] += records > 0
            if not n:
                continue
            h, tr = t["H"], t["T"]
            print(f"| {kind} | {name} | {n} | {h[0]} | {h[1]} | {h[6]}, {h[4]}, {h[5]} | "
                  f"{tr[0]} | {tr[1]} | {tr[2]}, {tr[3]}, {tr[4]}, {tr[5]} |")


def same(one, other):
    a = {(r["run"], r["k"], r["change"], r.get("tag"), r.get("draw", 0)): r for r in one}
    b = {(r["run"], r["k"], r["change"], r.get("tag"), r.get("draw", 0)): r for r in other}
    differ, shown = 0, 0
    for key in a:
        for field in FIELDS:
            if a[key].get(field) != b.get(key, {}).get(field):
                differ += 1
                if shown < 10:
                    shown += 1
                    print(key, field, str(a[key].get(field))[:90], "|",
                          str(b.get(key, {}).get(field))[:90])
    print(f"records compared: {len(a)} (the other file holds {len(b)}); fields compared in "
          f"each: {len(FIELDS)}; fields that differ: {differ}")
    return differ


def gate(new, before):
    old = grouped(before)
    moved = defaultdict(lambda: defaultdict(int))
    for key, rows in grouped(new).items():
        print(f"\n{key[0]}\n  new     {span(rows)}\n    " + " ".join(number(r) for r in rows))
        was = {r["k"]: r for r in old.get(key, [])}
        for r in rows:
            for field in FIELDS:
                if field in ("share", "share_row", "base_share", "base_share_row"):
                    continue                    # new in this pass
                if r.get(field) != was.get(r["k"], {}).get(field):
                    moved[key[0]][field] += 1
        if moved.get(key[0]):
            ordered = [was[k] for k in sorted(was)]
            print(f"  before  {span(ordered)}\n    " + " ".join(number(r) for r in ordered))
    print("\nFields that differ from the reference as it stood, by run (seeds of 30):")
    for run, fields in moved.items():
        print(f"  {run}: " + ", ".join(f"{f} on {n}" for f, n in fields.items()))
    quiet = [run for run, _ in grouped(new) if not moved.get(run)]
    print(f"  no field differs on {len(quiet)} runs: " + ", ".join(quiet))


if __name__ == "__main__":
    what = sys.argv[1]
    files = [load(p) for p in sys.argv[2:]]
    if what == "a1":
        skills(what, *files, also=a1_also)
    elif what == "a6":
        skills(what, *files, also=a6_also)
    elif what == "a2":
        a2(*files)
    elif what == "a3":
        a3(*files)
    elif what == "gate":
        gate(*files)
    elif what == "same":
        raise SystemExit(1 if same(*files) else 0)
    else:
        raise SystemExit(__doc__)
