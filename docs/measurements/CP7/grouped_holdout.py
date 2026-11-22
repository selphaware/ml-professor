"""CP7, section C (D64): the grouped holdout, as built and under the rule proposed.

    python -B grouped_holdout.py [--seeds 30] [--table]

`old` is the reference's `_split_grouped` as it stood before this pass: the
units in an order drawn from the seed, taken until they hold a quarter of the
rows. `new` is the rule proposed (the function below). For each table and each
of the seeds: the holdout's share of the rows, whether the run would stop with
`too_few_groups`, and which of the two new warnings would be raised (D66): one for
a unit of a quarter or more that is kept among the training rows, one for a unit
that is the holdout because the smaller units cannot fill a quarter. Then the
gate's `entities` dataset at 30 data seeds: is the split the same, row for row?

Nothing here fits a model. No file is written.
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[3]
N_FOLDS = 5
MIN_GROUPS = N_FOLDS + 1


class TooFewGroups(Exception):
    pass


def split_old(groups, y, classification, seed):
    """The rule as built (pass 5): units in seed order until a quarter is held."""
    groups = pd.factorize(np.asarray(groups))[0]
    sizes = np.bincount(groups)
    n_groups = len(sizes)
    if n_groups < MIN_GROUPS:
        raise TooFewGroups("fewer than six units")
    rng = np.random.default_rng([seed, 7])
    if classification:
        positive = np.bincount(groups, weights=y, minlength=n_groups)
        stratum = (2 * positive >= sizes).astype("int64")
    else:
        stratum = np.zeros(n_groups, dtype="int64")
    held = np.zeros(n_groups, dtype=bool)
    fold_of = np.full(n_groups, -1)
    orders = {}
    for s in (0, 1):
        members = np.flatnonzero(stratum == s)
        members = members[rng.permutation(len(members))]
        total, taken = int(sizes[members].sum()), 0
        for g in members.tolist():
            if 4 * taken >= total:
                break
            held[g], taken = True, taken + int(sizes[g])
        orders[s] = members[~held[members]]
    return _folds(groups, sizes, held, fold_of, orders, rng, stratum, [])


def split_new(groups, y, classification, seed):
    """The rule proposed for D64.

    For regression, and for the units of each class apart, with the units in an
    order drawn from the seed:
    1. a unit joins the holdout when, with it, the holdout still holds less than
       a quarter of the rows; any other unit is passed over;
    2. the holdout is completed by the smallest unit passed over, the first in
       the order among equals;
    3. every unit the holdout can spare is returned to the training rows (the
       largest first): one without which it still holds a quarter.
    `too_large` lists every unit that holds a quarter or more of the rows it is
    drawn among, as (rows, rows drawn among, whether it is the holdout). Such a
    unit is never taken in step 1. It is the holdout exactly when the units that
    each hold less than a quarter hold, together, less than a quarter: step 2
    then takes the smallest of the large ones and step 3 returns everything else.
    Otherwise it stays among the training rows.
    """
    groups = pd.factorize(np.asarray(groups))[0]
    sizes = np.bincount(groups)
    n_groups = len(sizes)
    if n_groups < MIN_GROUPS:
        raise TooFewGroups("fewer than six units")
    rng = np.random.default_rng([seed, 7])
    if classification:
        positive = np.bincount(groups, weights=y, minlength=n_groups)
        stratum = (2 * positive >= sizes).astype("int64")
    else:
        stratum = np.zeros(n_groups, dtype="int64")
    held = np.zeros(n_groups, dtype=bool)
    fold_of = np.full(n_groups, -1)
    orders, too_large = {}, []
    for s in (0, 1):
        members = np.flatnonzero(stratum == s)
        members = members[rng.permutation(len(members))]
        total, taken, passed = int(sizes[members].sum()), 0, []
        for g in members.tolist():
            if 4 * (taken + int(sizes[g])) < total:
                held[g], taken = True, taken + int(sizes[g])
            else:
                passed.append(g)
        if passed:
            last = min(passed, key=lambda g: int(sizes[g]))     # first in the order among equals
            held[last], taken = True, taken + int(sizes[last])
            spare = [g for g in members.tolist() if held[g] and g != last]
            for g in sorted(spare, key=lambda g: -int(sizes[g])):   # stable: seed order among equals
                if 4 * (taken - int(sizes[g])) >= total:
                    held[g], taken = False, taken - int(sizes[g])
        too_large += [(int(sizes[g]), total, bool(held[g])) for g in members.tolist()
                      if 4 * int(sizes[g]) >= total]
        orders[s] = members[~held[members]]
    return _folds(groups, sizes, held, fold_of, orders, rng, stratum, too_large)


def _folds(groups, sizes, held, fold_of, orders, rng, stratum, too_large):
    if int((~held).sum()) < N_FOLDS:
        raise TooFewGroups("fewer than five units left for the folds")
    load = np.zeros(N_FOLDS, dtype="int64")
    for s in (0, 1):
        own = np.zeros(N_FOLDS, dtype="int64")
        for g in orders[s][rng.permutation(len(orders[s]))].tolist():
            k = min(range(N_FOLDS), key=lambda f: (own[f], load[f], f))
            fold_of[g], own[k], load[k] = k, own[k] + sizes[g], load[k] + sizes[g]
    row_fold = fold_of[groups]
    return {"holdout": np.flatnonzero(held[groups]), "train": np.flatnonzero(row_fold >= 0),
            "row_fold": row_fold, "held": held, "sizes": sizes, "stratum": stratum,
            "too_large": too_large}


def quarter_unit(sizes):
    """u*: the size of the smallest unit such that the units no larger than it hold,
    together, at least a quarter of the rows."""
    total, run = int(sizes.sum()), 0
    for size in np.sort(sizes).tolist():
        run += size
        if 4 * run >= total:
            return size
    return 0


def check_properties(split):
    """The properties proposed for step 7, property 3, on a split made by `split_new`."""
    sizes, held, stratum = split["sizes"], split["held"], split["stratum"]
    for s in (0, 1):
        members = stratum == s
        total = int(sizes[members].sum())
        if total == 0:
            continue
        h = int(sizes[members & held].sum())
        assert 4 * h >= total, "less than a quarter"
        for g in np.flatnonzero(members & held).tolist():
            assert 4 * (h - int(sizes[g])) < total, "a unit the holdout could spare"
        assert 4 * h < total + 4 * quarter_unit(sizes[members]), "past a quarter plus u*"


def table(kind, seed):
    """`groups` (one entity label per row, rows shuffled) and a 0/1 `y` for classification."""
    rng = np.random.default_rng(seed)
    small = 5
    if kind == "equal: 400 entities of 5 rows":
        sizes = [small] * 400
    elif kind == "one entity of 30% of the rows":
        sizes = [600] + [small] * 280
    elif kind == "one entity of 60% of the rows":
        sizes = [1200] + [small] * 160
    elif kind == "one entity of 74.5% of the rows":
        sizes = [1490] + [2] * 255
    elif kind == "one entity of 90% of the rows":
        sizes = [1800] + [small] * 40
    elif kind == "five entities of 12% each, the rest of 5 rows":
        sizes = [240] * 5 + [small] * 160
    elif kind == "eight large entities and no other":
        sizes = [400, 350, 300, 250, 250, 200, 150, 100]
    elif kind == "two of 40% each and five of 4%":
        sizes = [800, 800] + [80] * 5
    elif kind == "one entity of exactly a quarter of the rows":
        sizes = [500] + [small] * 300
    elif kind == "sizes drawn from a long-tailed law (1 to about 300 rows)":
        sizes = []
        while sum(sizes) < 2000:
            sizes.append(int(min(400, max(1, round(rng.lognormal(1.5, 1.2))))))
    elif kind == "classification: one entity of 40% of the rows, its label 1":
        sizes = [800] + [small] * 240
    else:
        raise ValueError(kind)
    groups = np.repeat(np.arange(len(sizes)), sizes)
    label = rng.integers(0, 2, size=len(sizes))
    label[0] = 1
    y = label[groups].astype(float)
    order = rng.permutation(len(groups))
    return groups[order], y[order], kind.startswith("classification")


KINDS = [
    "equal: 400 entities of 5 rows",
    "one entity of 30% of the rows",
    "one entity of 60% of the rows",
    "one entity of 74.5% of the rows",
    "one entity of 90% of the rows",
    "five entities of 12% each, the rest of 5 rows",
    "eight large entities and no other",
    "two of 40% each and five of 4%",
    "one entity of exactly a quarter of the rows",
    "sizes drawn from a long-tailed law (1 to about 300 rows)",
    "classification: one entity of 40% of the rows, its label 1",
]


def gate_module():
    sys.path.insert(0, str(REPO / "verify"))
    spec = importlib.util.spec_from_file_location("mlprof_gate", str(REPO / "verify" / "smoke_test.py"))
    module = importlib.util.module_from_spec(spec)
    saved = sys.argv
    sys.argv = [str(REPO / "verify" / "smoke_test.py")]
    try:
        spec.loader.exec_module(module)
    finally:
        sys.argv = saved
    return module


def same_as_reference(seeds, rule, broken):
    """One of the two rules above against the reference's own `_split_grouped`, with
    the reference's switch set to `broken`: the same holdout rows and the same folds,
    or the same stop. `split_new` is the reference as changed for this pass, and
    `split_old` its broken copy F13, the rule as built in the fifth pass."""
    sys.path.insert(0, str(REPO / "verify" / "reference"))
    from mlprofessor import core, errors
    core.BROKEN = broken
    agree = total = 0
    for kind in KINDS:
        for seed in range(seeds):
            groups, y, classification = table(kind, seed)
            total += 1
            try:
                mine = rule(groups, y, classification, seed)
            except TooFewGroups:
                mine = None
            try:
                theirs = core._split_grouped(groups, y, classification, seed)
            except errors.TooFewGroupsError:
                theirs = None
            if mine is None or theirs is None:
                agree += mine is None and theirs is None
                continue
            folds = all(np.array_equal(np.flatnonzero(mine["row_fold"] == k), f["valid"])
                        for k, f in enumerate(theirs["folds"]))
            agree += bool(np.array_equal(mine["holdout"], theirs["holdout"]) and folds)
    return agree, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--table", action="store_true", help="print every seed")
    parser.add_argument("--against-reference", action="store_true",
                        help="compare both rules with verify/reference: the good copy and F13")
    args = parser.parse_args()
    if args.against_reference:
        for label, rule, broken in (("proposed", split_new, None), ("as built", split_old, "F13")):
            agree, total = same_as_reference(args.seeds, rule, broken)
            print(f"`{label}` here and the reference's _split_grouped"
                  f"{' as its broken copy F13' if broken else ''} agree in {agree} of {total} "
                  "splits (holdout rows and folds, or the same stop)")
    print(f"holdout share of the rows over {args.seeds} seeds (seed k: table from data seed k, "
          "split from seed k)")
    print("| Table | Rule | min | mean | max | too_few_groups | warned: kept in training | "
          "warned: is the holdout | u* (rows) |")
    print("|---|---|---|---|---|---|---|---|---|")
    detail = []
    for kind in KINDS:
        for name, rule in (("as built", split_old), ("proposed", split_new)):
            shares, stops, kept, alone, ustar = [], 0, 0, 0, set()
            for seed in range(args.seeds):
                groups, y, classification = table(kind, seed)
                try:
                    split = rule(groups, y, classification, seed)
                except TooFewGroups:
                    stops += 1
                    detail.append((kind, name, seed, "too_few_groups", ""))
                    continue
                if rule is split_new:
                    check_properties(split)
                share = len(split["holdout"]) / len(groups)
                shares.append(share)
                kept += any(not is_holdout for _, _, is_holdout in split["too_large"])
                alone += any(is_holdout for _, _, is_holdout in split["too_large"])
                ustar.add(quarter_unit(split["sizes"]) if not classification else
                          tuple(quarter_unit(split["sizes"][split["stratum"] == s]) for s in (0, 1)))
                detail.append((kind, name, seed, f"{share:.4f}", "; ".join(
                    f"{size} of {total} rows {'is the holdout' if is_holdout else 'kept in training'}"
                    for size, total, is_holdout in split["too_large"])))
            shown = (f"{min(shares):.3f} | {np.mean(shares):.3f} | {max(shares):.3f}" if shares
                     else "- | - | -")
            us = sorted(ustar)
            us = us[0] if len(us) == 1 else f"{us[0]}..{us[-1]}"
            print(f"| {kind} | {name} | {shown} | {stops} | "
                  f"{kept if rule is split_new else 'n/a'} | "
                  f"{alone if rule is split_new else 'n/a'} | {us} |")
    if args.table:
        print("\nper seed (seed 0, 1, ...): the holdout's share of the rows; `stop` is "
              "too_few_groups")
        rows = {}
        for kind, name, seed, share, _note in detail:
            rows.setdefault((kind, name), []).append(
                "stop" if share == "too_few_groups" else share[1:])
        for (kind, name), shares in rows.items():
            print(f"{kind} | {name} | " + " ".join(shares))

    gate = gate_module()
    same, shares = 0, []
    for k in range(args.seeds):
        base = gate.SEED + 100 * k
        header, rows, _ = gate.DATASETS["entities"][0](gate.dataset_seed("entities", base))
        frame = pd.DataFrame(rows, columns=header)
        groups = frame["customer"].to_numpy()
        y = (frame["outcome"] == "yes").to_numpy().astype(float)
        old, new = split_old(groups, y, True, 0), split_new(groups, y, True, 0)
        check_properties(new)
        same += bool(np.array_equal(old["holdout"], new["holdout"])
                     and np.array_equal(old["row_fold"], new["row_fold"]))
        shares.append(len(new["holdout"]) / len(frame))
    print(f"\ngate dataset `entities`, {args.seeds} data seeds, tool seed 0: holdout and folds "
          f"identical, row for row, under both rules in {same} of {args.seeds}; holdout share "
          f"{min(shares):.4f} to {max(shares):.4f}")


if __name__ == "__main__":
    main()
