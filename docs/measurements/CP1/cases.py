"""Crafted cases behind the cut-offs of phase 1 (CP1 session; D35). One process at a time.

    python docs/measurements/CP1/cases.py <ref_parent> <gate_file> <what> [shares]

    examples   the pattern for a number with thousands separators, value by value
    shares     messy over the 30 bases: how `revenue` and `fee` are written
    roles      crafted columns of 400 rows: role, reason, warnings (D28, D34)
    coerce     the two shapes of the coercion measurement, on 2,000 rows
    neartime   a flipped copy of the target under the time-ordered scheme
    gateseed   where the reference lands at the gate's seed on noise_reg and imbalanced

<ref_parent> and <gate_file> as in lab.py (`-` for the repo's own). `shares` and
`neartime` also write WORK/out/shares.json and WORK/out/neartime.json, which
`lab.py table` reads.
"""
import sys
import os

sys.dont_write_bytecode = True
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import random
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lab  # noqa: E402

PATTERN = re.compile(r"[+-]?[0-9]{1,3}(,[0-9]{3})+(\.[0-9]+)?")      # as written in the spec
YES = ["1,200", "-3,450.75", "+12,345,678", "0,123"]
NO = ["1200", "532.10", "12,34", "1,2345", "1234,567", "1.200,50", "1,200.", "1,200.5.1",
      "$1,200", "(1,200)", "1 200", "1,200 units"]


def parses(cell):
    try:
        float(cell)
        return True
    except ValueError:
        return False


def examples(mlprofessor, core, G):
    import pandas as pd
    ok = True
    for value in YES + NO:
        mine = PATTERN.fullmatch(value) is not None
        theirs = core._separated(value)
        numeric = bool(pd.to_numeric(pd.Series([value], dtype="str"), errors="coerce").notna()[0])
        want = value in YES
        ok &= (mine == theirs == want) and not (want and numeric)
        print(f"{value!r:<16} spec pattern {mine!s:<5} reference {theirs!s:<5} "
              f"to_numeric parses {numeric}")
    print("EXAMPLES OK" if ok else "EXAMPLES DIFFER")


def shares(mlprofessor, core, G):
    out = {}

    def keep(label, value):
        out.setdefault(label, {"values": []})["values"].append(value)

    for k in range(30):
        seed = G.dataset_seed("messy", G.SEED + 100 * k)
        header, rows, _ = G.make_messy(seed)
        target = header.index("outcome")
        usable = [row for row in rows if row[target] != ""]
        cells = [row[header.index("revenue")] for row in usable]
        keep("messy.revenue.share_plain", sum(parses(c) for c in cells) / len(cells))
        keep("messy.revenue.share_separated",
             sum(PATTERN.fullmatch(c) is not None for c in cells) / len(cells))
        keep("messy.revenue.share_either",
             sum(parses(c) or PATTERN.fullmatch(c) is not None for c in cells) / len(cells))
        keep("messy.revenue.share_distinct", len(set(cells)) / len(cells))
        if "fee" in header:
            cells = [row[header.index("fee")] for row in usable]
            keep("messy.fee.share_plain", sum(parses(c) for c in cells) / len(cells))
            keep("messy.fee.n_separated", sum(PATTERN.fullmatch(c) is not None for c in cells))
            keep("messy.fee.n_separated_all_rows",
                 sum(PATTERN.fullmatch(row[header.index("fee")]) is not None for row in rows))
        cells = [row[header.index("spend")] for row in rows]
        keep("messy.spend.n_text_cells",
             sum(not parses(c) and c not in ("", "n/a") for c in cells))
        keep("messy.spend.n_separated_all_rows",
             sum(PATTERN.fullmatch(c) is not None for c in cells))
    for label, entry in out.items():
        lab.show(label, entry["values"])
    header, rows, _ = G.make_messy(G.dataset_seed("messy"))
    for name in ("revenue", "fee", "spend"):
        if name in header:
            cells = [row[header.index(name)] for row in rows]
            odd = [c for c in cells if not parses(c) and c not in ("", "n/a")]
            print(f"{name} at the gate seed: first values {cells[:5]}; "
                  f"{len(odd)} cells do not parse, e.g. {odd[:8]}")
    lab.save(lab.OUT / "shares.json", out)


def roles(mlprofessor, core, G):
    """Crafted columns: role and reason from the step function, then a run's warnings."""
    import pandas as pd
    rng = random.Random(7)
    n = 400
    y = [rng.choice(["no", "yes"]) for _ in range(n)]
    for i in rng.sample(range(n), 5):
        y[i] = None                                   # five rows with no target
    f1 = [rng.gauss(1.0 if v == "yes" else 0.0, 1.0) for v in y]
    with_target = [i for i in range(n) if y[i] is not None]
    without_target = [i for i in range(n) if y[i] is None]

    def money(x, form="sep"):
        if form == "sep":
            return f"{x:,.2f}"
        if form == "euro":                            # 1.200,50
            return f"{x:,.2f}".replace(",", "_").replace(".", ",").replace("_", ".")
        if form == "short":                           # 12,34
            return f"{x / 100:.2f}".replace(".", ",")
        raise ValueError(form)

    def plain(mean=50, sd=10):
        return [f"{rng.gauss(mean, sd):.2f}" for _ in range(n)]

    def with_cells(column, rows, cells):
        column = list(column)
        for i, cell in zip(rows, cells):
            column[i] = cell
        return column

    amounts = [rng.lognormvariate(7.92, 1.2) for _ in range(n)]
    pool = [rng.lognormvariate(8.5, 0.5) for _ in range(240)]         # 60% distinct
    few = ["1,000", "2,500", "10,000", "12,345.50", "250,000"]
    mostly = [money(a) for a in amounts]
    for i in rng.sample(range(n), 60):                                # 15% words: under 90%
        mostly[i] = f"word {i}"
    frame = pd.DataFrame({
        "f1": f1,
        "sep_all_distinct": [money(a) for a in amounts],
        "sep_60pct_distinct": [money(pool[i % 240]) for i in range(n)],
        "sep_few_levels": [few[i % 5] for i in range(n)],
        "sep_constant": ["1,200"] * n,
        "euro": [money(a, "euro") for a in amounts],
        "short": [money(a, "short") for a in amounts],
        "euro_60pct": [money(pool[i % 240], "euro") for i in range(n)],
        "plain_one_sep_cell": with_cells(plain(), rng.sample(with_target, 1), ["1,200"]),
        "plain_few_sep_cells": with_cells(plain(), rng.sample(with_target, 6),
                                          ["1,200", "3,450", "abc", "12 units", "1,200", "-"]),
        "plain_5pct_sep": with_cells(plain(500, 100), rng.sample(with_target, 20),
                                     [money(rng.uniform(1000, 9000)) for _ in range(20)]),
        "plain_words_only": with_cells(plain(), rng.sample(with_target, 4),
                                       ["abc", "unknown", "12 units", "-"]),
        "plain_sep_off_target": with_cells(plain(), without_target[:1], ["1,200"]),
        "plain_other_money_text": with_cells(plain(), rng.sample(with_target, 3),
                                             ["$1,200", "1,200.", "1,200 units"]),
        "sep_85pct_words_15pct": mostly,
        "outcome": y,
    })
    table = mlprofessor.stages.column_roles(frame, "outcome")
    print(table[["name", "role", "kind", "reason", "n_coerced"]].to_string(index=False))
    d = mlprofessor.profile(frame, "outcome", profile="fast", seed=0, n_jobs=1).to_dict()
    print("warnings:", sorted((w["code"], w["column"]) for w in d["warnings"]))
    print("limitations:", len(d["limitations"]), "| interval_method:",
          d["headline"]["interval_method"])
    base = frame[["f1", "outcome"]].dropna().reset_index(drop=True)
    long = pd.concat([base, base], ignore_index=True)
    long["t"] = range(len(long))                      # 790 timestamps, one row each
    short = base.assign(t=[i % 100 for i in range(len(base))])       # 100 timestamps
    for label, timed in (("unique timestamps", long), ("100 timestamps", short)):
        d = mlprofessor.profile(timed, "outcome", time_col="t", profile="fast", seed=0,
                                n_jobs=1).to_dict()
        print(f"time order, {label}: {d['validation']['n_holdout_timestamps']} holdout "
              f"timestamps, method {d['headline']['interval_method']}, warnings "
              f"{sorted((w['code'], w['column']) for w in d['warnings'])}")


def coerce(mlprofessor, core, G):
    """Amounts written with separators throughout, and with half the cells written so."""
    import pandas as pd
    rng = random.Random(11)
    n = 2000
    y = [rng.choice(["no", "yes"]) for _ in range(n)]
    f1 = [rng.gauss(1.0 if v == "yes" else 0.0, 1.0) for v in y]
    for label, mu in (("most values below 1,000", 6.0), ("most values above 1,000", 9.0)):
        amounts = [rng.lognormvariate(mu, 1.0) for _ in range(n)]
        cells = [f"{a:,.2f}" for a in amounts]
        half = [f"{a:,.2f}" if i % 2 else f"{a:.2f}" for i, a in enumerate(amounts)]
        frame = pd.DataFrame({"f1": f1, "all_sep": cells, "half_sep": half, "outcome": y})
        table = mlprofessor.stages.column_roles(frame, "outcome")
        print(f"{label}: written with separators throughout, "
              f"{sum(parses(c) for c in cells) / n:.1%} parse; half written that way, "
              f"{sum(parses(c) for c in half) / n:.1%} parse, "
              f"{sum(not parses(c) for c in half)} cells carry a separator")
        print(table[["name", "role", "kind", "reason", "n_coerced"]].to_string(index=False))


TIME_SHARES = (0.01, 0.015, 0.02, 0.025, 0.03)


def neartime(mlprofessor, core, G):
    """The flipped copy of lab.py's `nearleak`, with a unique numeric time column."""
    seen = lab.instrument(core)
    out = {}
    for n in (1500, 400):
        for share in SHARES:
            values = []
            for seed in lab.near_seeds(G):
                header, rows = lab.flipped_copy(G, seed, n, share, True, time_col=True)
                path = lab.write_table("custom", "near_time", lab.table_text(G, header, rows))
                values.append(lab.screen_once(mlprofessor, seen, path, "outcome",
                                              ["--time-col", "t"])["outcome_noisy"])
            label = f"flip.time.n{n}.exact.{share:.3f}"
            s = lab.show(label, values)
            s.update(at_or_above_098=sum(v >= 0.98 for v in values),
                     below_090=sum(v < 0.90 for v in values), values=values)
            out[label] = s
            print(f"    at or above 0.98: {s['at_or_above_098']} of 30; below 0.90: "
                  f"{s['below_090']} of 30", flush=True)
    lab.save(lab.OUT / "neartime.json", out)


def gateseed(mlprofessor, core, G):
    """The good fixture at the gate's own seed, against the two bands D35 leaves as they are."""
    for check, run, band in (("model.noise_reg", "noise_reg", "noise_reg"),
                             ("model.metric_clf", "imbalanced", "imbalanced")):
        dataset, extra, profile, _ = G.RUNS[run]
        path = lab.dataset_file(G, "gate", dataset, G.dataset_seed(dataset))
        result = mlprofessor.profile(str(path), G.DATASETS[dataset][1], profile=profile, seed=0,
                                     n_jobs=1, **lab.kwargs_of(extra))
        skill = result.headline["skill"]
        low, high = G.BANDS[band]
        print(f"{check} on `{run}` at the gate seed {G.dataset_seed(dataset)}: skill "
              f"{skill:+.4f}; band {low} to {high}; {skill - low:.4f} above the lower edge, "
              f"{high - skill:.4f} below the upper edge")
        if run == "imbalanced":
            labels = [str(v) for v in result.predictions["y_true"]]
            majority = max(labels.count("yes"), labels.count("no")) / len(labels)
            score = result.headline["score"]
            print(f"    its other margin: |score {score:.4f} - majority share {majority:.4f}| = "
                  f"{abs(score - majority):.4f}, required above {G.MAJORITY_MARGIN}")


if __name__ == "__main__":
    ref_parent, gate_file, what = sys.argv[1:4]
    SHARES = tuple(float(s) for s in sys.argv[4].split(",")) if len(sys.argv) > 4 else TIME_SHARES
    G = lab.load_gate(gate_file)
    mlprofessor, core = lab.load_ref(ref_parent)
    {"examples": examples, "shares": shares, "roles": roles, "coerce": coerce,
     "neartime": neartime, "gateseed": gateseed}[what](mlprofessor, core, G)
