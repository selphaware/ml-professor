"""CP10, section C, compact.

    python -B report_c.py warn  <c_warn.json>     copies and curves that must at least be warned about
    python -B report_c.py small <c_small.json>    the small-table cells of CP9, the running total,
                                                   and an honest column of repeated values
    python -B report_c.py e     <e_share.json>    section E: the holdout share and the warning
"""
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lab10                                            # noqa: E402

what, recs = sys.argv[1], lab10.load(sys.argv[2])
FORMS = ("old", "mean", "median", "random", "pairs")


def would_warn(r, column, form):
    """Whether the order reading's two conditions hold for `column` under `form`; None
    when the column was not read by its order (excluded, or warned by its skill)."""
    per = r.get("log", {}).get("order", {}).get(column)
    if per is None:
        return None
    share, before = per[form]
    return share is not None and share <= 0.10 and (before is None or before > 0.10)


if what == "warn":
    groups = defaultdict(list)
    for r in recs:
        groups[(r["tag"], r["change"])].append(r)
    print("the added column over the runs of each scheme, 30 seeds each: excluded / warned / "
          "kept in silence as run (the pairs form); then, among the runs in which the column is "
          "read by its order, on how many the two conditions hold under each form\n")
    print("| scheme | the column | excluded / warned / silent | read by its order | "
          + " | ".join(FORMS) + " |")
    print("|---|---|---|---|" + "---|" * len(FORMS))
    for key in sorted(groups):
        g = groups[key]
        fates = Counter(lab10.fate(r, "target_copy") for r in g)
        read = [r for r in g if would_warn(r, "target_copy", "pairs") is not None]
        cells = [str(sum(bool(would_warn(r, "target_copy", f)) for r in read)) for f in FORMS]
        print(f"| {key[0]} | {key[1]} | {fates.get('excluded', 0)} / {fates.get('warned', 0)} / "
              f"{fates.get('silent', 0)} | {len(read)} | " + " | ".join(cells) + " |")
elif what == "small":
    groups = defaultdict(list)
    for r in recs:
        p = dict(r["params"])
        groups[(r["make"], " ".join(f"{k}={v}" for k, v in sorted(p.items())))].append(r)
    print("| table | parameters | the column | excluded / warned / silent | warned by the order "
          "reading | share against the order, pairs form | old form |")
    print("|---|---|---|---|---|---|---|")
    for key in sorted(groups):
        g = groups[key]
        column = {"strong_small": "x", "running_total": "level_prev", "monotone_levels": "x"}[key[0]]
        done = [r for r in g if "error" not in r]
        fates = Counter(lab10.fate(r, column) for r in g)
        by_order = sum(any(c == "strong_single_column" and col == column and "order" in m
                           for c, col, m in r["texts"]) for r in done)
        shares = {f: [r["log"]["order"][column][f][0] for r in done
                      if column in r.get("log", {}).get("order", {})] for f in ("pairs", "old")}
        print(f"| {key[0]} | {key[1]} | {column} | {fates.get('excluded', 0)} / "
              f"{fates.get('warned', 0)} / {fates.get('silent', 0)}"
              f"{' ' + str({k: v for k, v in fates.items() if k.startswith('error')}) if any(k.startswith('error') for k in fates) else ''} | "
              f"{by_order} | {lab10.span(shares['pairs'])} | {lab10.span(shares['old'])} |")
elif what == "e":
    print("| table | parameters | holdout share | `holdout_far_above_quarter` raised | rows: training, holdout |")
    print("|---|---|---|---|---|")
    for r in recs:
        if "error" in r:
            print(f"| {r['make']} | {r['params']} | {r['error']} | | |")
            continue
        if r["k"] != 0:
            continue
        raised = any(c == "holdout_far_above_quarter" for c, _ in r["warnings"])
        print(f"| {r['make']} | {r['params']} | {r['holdout_share']:.4f} | {raised} | "
              f"{r['rows']['training']}, {r['rows']['holdout']} |")
