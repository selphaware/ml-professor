"""CP10, section B: the reading by levels. Coded copies that must be excluded, honest
columns that must not be.

    python -B report_b.py bands  <b_bands.json>
    python -B report_b.py codes  <b_codes.json>
    python -B report_b.py honest <b_honest.json>
    python -B report_b.py few    <json> [<json> ...]     the level reading of numeric columns by rows a level
"""
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lab10                                            # noqa: E402

what = sys.argv[1]
recs = [r for path in sys.argv[2:] for r in lab10.load(path)]
ORDER = ("excluded", "warned", "named", "silent")


def counts(group, column):
    fates = Counter(lab10.fate(r, column) for r in group)
    other = {k: v for k, v in fates.items() if k not in ORDER}
    return "/".join(str(fates.get(f, 0)) for f in ORDER) + (f" {other}" if other else "")


def reading(group, column, reader, which=1):
    vals = [r["log"]["screen"][column][reader][which] for r in group
            if "error" not in r and column in r.get("log", {}).get("screen", {})
            and r["log"]["screen"][column].get(reader)
            and r["log"]["screen"][column][reader][which] is not None]
    return f"{min(vals):+.3f} to {max(vals):+.3f}" if vals else "-"


if what in ("bands", "codes"):
    column = "target_band" if what == "bands" else "target_code"
    groups = defaultdict(list)
    for r in recs:
        _, n, kind = r["change"].split(":")
        groups[(r["tag"], r["run"], kind, int(n))].append(r)
    print("excluded/warned/named/silent of the coded column, 30 seeds; the level reading with "
          "every row scored; the reading of the other readers; the headline where it is kept\n")
    print("| scheme | run | the column | levels | fate | levels reading | rule 2 reading by levels | "
          "tree | share of validation rows whose level was seen | headline where kept | "
          "another column excluded |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for key in sorted(groups):
        g = groups[key]
        kept = [r["skill"] for r in g if lab10.fate(r, column) in ("warned", "silent", "named")
                and r["skill"] is not None]
        seen = [r["log"]["screen"][column].get("seen") for r in g
                if "error" not in r and column in r.get("log", {}).get("screen", {})]
        seen = [s for s in seen if s is not None]
        others = sum(any(v[1] == "suspected_leak" and n != column for n, v in r["roles"].items())
                     for r in g if "error" not in r)
        print("| " + " | ".join(str(k) for k in key) + f" | {counts(g, column)} | "
              f"{reading(g, column, 'levels')} | {reading(g, column, 'levels', 0)} | "
              f"{reading(g, column, 'tree')} | "
              f"{(format(min(seen), '.2f') + ' to ' + format(max(seen), '.2f')) if seen else '-'} | "
              f"{lab10.span(kept) if kept else '-'} | {others} |")
elif what == "honest":
    groups = defaultdict(list)
    for r in recs:
        p = dict(r["params"])
        scheme, numeric = p.pop("scheme"), p.pop("numeric")
        groups[(r["make"], scheme, "numeric" if numeric else "text",
                " ".join(f"{k}={v}" for k, v in sorted(p.items())))].append(r)
    print("| table | scheme | the column | parameters | fate of `level`: excluded/warned/named/silent | "
          "levels reading (every row scored) | tree | used / logged only | headline |")
    print("|---|---|---|---|---|---|---|---|---|")
    for key in sorted(groups):
        g = groups[key]
        done = [r for r in g if "error" not in r]
        modes = Counter(r["log"]["screen"]["level"]["level_mode"] for r in done
                        if "level" in r.get("log", {}).get("screen", {}))
        skills = [r["skill"] for r in done if r["skill"] is not None]
        errors = Counter(r["error"] for r in g if "error" in r)
        print("| " + " | ".join(str(k) for k in key) + f" | {counts(g, 'level')} | "
              f"{reading(g, 'level', 'levels')} | {reading(g, 'level', 'tree')} | "
              f"{modes.get(2, 0)} / {modes.get(1, 0)} | {lab10.span(skills) if skills else '-'}"
              f"{' ' + str(dict(errors)) if errors else ''} |")
elif what == "few":
    # every numeric column of every run: its rows a level and what the level reading says
    bins = defaultdict(list)
    for r in recs:
        if "error" in r:
            continue
        for name, (distinct, held) in r.get("log", {}).get("levels", {}).items():
            s = r["log"]["screen"].get(name, {})
            if not s.get("levels") or s["levels"][1] is None or not distinct:
                continue
            per = held / distinct
            edge = next(e for e in (1.5, 2, 3, 5, 10, 20, 50, 1e9) if per < e)
            others = [s[k][1] for k in ("tree", "line", "direct") if s.get(k) and s[k][1] is not None]
            bins[edge].append((s["levels"][1], max(others) if others else None, s["levels"][0]))
    print("| training rows for each distinct value | numeric columns | level reading (every row scored) | "
          "the level reading is the best reader on | level reading at or above 0.90 | "
          "rule 2 reading at or above 0.98 |")
    print("|---|---|---|---|---|---|")
    last = 1.0
    for edge in (1.5, 2, 3, 5, 10, 20, 50, 1e9):
        rows = bins.get(edge, [])
        if rows:
            vals = [v[0] for v in rows]
            best = sum(v[1] is None or v[0] > v[1] for v in rows)
            print(f"| {last:g} to {'more' if edge > 1e8 else format(edge, 'g')} | {len(rows)} | "
                  f"{min(vals):+.3f} to {max(vals):+.3f} | {best} | "
                  f"{sum(v >= 0.90 for v in vals)} | {sum((v[2] or -9) >= 0.98 for v in rows)} |")
        last = edge
