"""Pass 9 (CP10) build: the reading by levels on honest columns of real tables.

For every numeric column the screen sees, the build's rule (training rows, not missing,
at least 5 x the distinct values: what T76 adopted) is compared with the rule T71 first
built (at least 5 rows of every value). Each table is run with the build as it is; where
a column's decision differs between the two rules, the table is run again with the
first rule patched in, and the two outcomes are printed side by side.

    python cp10_levels.py app10      (or ref10 for the reference, without the patch)
"""
import sys
import warnings

import numpy as np
import pandas as pd

which = sys.argv[1]
sys.path.insert(0, f"/home/claude/realdata/{which}")
warnings.simplefilter("ignore")
import mlprofessor  # noqa: E402
from mlprofessor import errors  # noqa: E402

D = "/home/claude/realdata/data/"
NO = dict(time_col=False, group_col=False)
LOG = []
PATCHED = which == "app10"

if PATCHED:
    from mlprofessor import leakage, columns  # noqa: E402

    built_rule = leakage.reads_by_levels
    built_screen = leakage.screen
    state = {"min_rule": False}

    def logged_rule(values, kind, training_rows):
        name = sys._getframe(1).f_locals.get("name")
        mean_rule = built_rule(values, kind, training_rows)
        if kind != "numeric":
            return mean_rule
        col = columns.feature_numbers(values)[np.asarray(training_rows)]
        col = col[~np.isnan(col)]
        counts = pd.Series(col).value_counts()
        min_rule = bool(len(col) and counts.min() >= leakage.LEVEL_ROWS)
        LOG.append(dict(name=name, rows=len(col), distinct=int(counts.size),
                        smallest=int(counts.min()) if len(col) else 0,
                        mean_rule=mean_rule, min_rule=min_rule))
        return min_rule if state["min_rule"] else mean_rule

    def logged_screen(*args, **kw):
        for record in built_screen(*args, **kw):
            LOG.append(dict(record=True, **{k: record[k] for k in
                                             ("name", "skill", "by_levels", "warn",
                                              "leak_rule", "order_share")}))
            yield record

    leakage.reads_by_levels = logged_rule
    leakage.screen = logged_screen
    from mlprofessor import pipeline  # noqa: E402
    if getattr(pipeline, "leakage", None) is not None:
        pipeline.leakage.screen = logged_screen


def outcome(data, target, **kw):
    kw.setdefault("profile", "fast")
    kw.setdefault("n_jobs", 2)
    try:
        d = mlprofessor.profile(data, target, **kw).to_dict()
    except errors.ProfileError as exc:
        return dict(error=exc.code)
    h = d["headline"]
    return dict(skill=None if h["skill"] is None else round(h["skill"], 4),
                excluded=sorted(c["name"] for c in d["columns"] if c["reason"] == "suspected_leak"),
                warned=sorted({w["column"] for w in d["warnings"] if w["code"] == "strong_single_column"}),
                near_perfect=any(w["code"] == "near_perfect_headline" for w in d["warnings"]),
                rests_on=[(e["column"], round(e["skill_fall"], 3)) for e in h.get("rests_on") or []],
                floor=None if h.get("rests_on_floor") is None else round(h["rests_on_floor"], 3))


def table(label, data, target, **kw):
    del LOG[:]
    built = outcome(data, target, **kw)
    print("=" * 100)
    print(f"{label}\n  {which}: {built}")
    if not PATCHED:
        return
    reads = {r["name"]: r for r in LOG if "record" not in r}
    screen = {r["name"]: r for r in LOG if "record" in r}
    for name, r in reads.items():
        s = screen.get(name, {})
        mark = "  <-- the two rules differ" if r["mean_rule"] != r["min_rule"] else ""
        if r["mean_rule"] or r["min_rule"]:
            skill = s.get("skill")
            print(f"    {name:>14}: {r['rows']} rows, {r['distinct']} values, fewest {r['smallest']}; "
                  f"by levels: built {r['mean_rule']}, first rule {r['min_rule']}; screen skill "
                  f"{'None' if skill is None else round(skill, 3)}{mark}")
    if any(r["mean_rule"] != r["min_rule"] for r in reads.values()):
        state["min_rule"] = True
        first = outcome(data, target, **kw)
        state["min_rule"] = False
        print(f"  with the first rule: {first}")


db = __import__("sklearn.datasets", fromlist=["load_diabetes"]).load_diabetes(as_frame=True).frame
table("diabetes (age 58 values, s4)", db, "target", **NO)
mpg = pd.read_csv(D + "mpg.csv").drop(columns=["name"])
table("mpg (cylinders 5 values, model_year 13)", mpg, "mpg", **NO)
tit = pd.read_csv(D + "titanic.csv").drop(columns=["alive"])
table("titanic (pclass, sibsp, parch, age)", tit, "survived", duplicates="keep", **NO)
table("tips (size 6 values)", pd.read_csv(D + "tips.csv"), "tip", duplicates="keep", **NO)
pen = pd.read_csv(D + "penguins.csv")
table("penguins body mass (flipper length)", pen, "body_mass_g", **NO)
table("penguins sex", pen, "sex", **NO)
table("diamonds price (carat under the fast cap)", D + "diamonds.csv", "price", **NO)
table("diamonds price, full profile", D + "diamonds.csv", "price", profile="full", **NO)
pl = pd.read_csv(D + "planets.csv")
table("planets orbital period (number, year)", pl, "orbital_period", duplicates="keep", **NO)
tx = pd.read_csv(D + "taxis.csv")
tx = tx[tx["payment"].notna()].drop(columns=["pickup", "dropoff"])
table("taxis tip (passengers 7 values)", tx.drop(columns=["total"]), "tip", **NO)
