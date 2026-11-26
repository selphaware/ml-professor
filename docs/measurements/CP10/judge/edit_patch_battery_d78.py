r"""patch_battery10.py brought up to D78: four outcomes for a pair that plants a carrier,
the rests-on list checked on every finished run, the three kinds of stated limit that
D78 point 2 names, and one pair for R2 (a feature written twice shares one fall)."""
from pathlib import Path

p = Path(__file__).resolve().parent / "patch_battery10.py"
t = p.read_text(encoding="utf-8")


def sub(old, new, count=1):
    global t
    assert t.count(old) == count, (t.count(old), old[:100])
    t = t.replace(old, new)


# ---- the stated limits of D78 point 2
sub(r'''# ------------------------------------------------------------- the case --''',
    r'''# ------------------------------------------------- the stated limits of D78 --
sub("""    ("panel", "copy_noise_5pct"): "D76",
}""", """    ("panel", "copy_noise_5pct"): "D76",
}
# D78 point 2: three more kinds, each a carrier of the target that the screen neither
# excludes nor warns about and that leaves the headline below 0.98. The result names
# what the headline rests on; where the carrier lifts it, the pair holds by that.
_TIME_REGRESSIONS = ("random_walk", "int_stamps", "flat_tail", "panel", "walk_repeated",
                     "walk_copy", "walk_copy_cell", "walk_late_cell")
for _run in _TIME_REGRESSIONS:
    # a coded copy of a regression target under time order
    STATED_LIMITS[(_run, "coded_copy_text")] = "D78"
    STATED_LIMITS[(_run, "coded_copy_number")] = "D78"
    # two columns that give the target only together, under time order and the fast profile
    STATED_LIMITS[(_run, "two_columns_together")] = "D78"
# a coded copy beside a dominant entity that stays below 0.98
STATED_LIMITS[("dominant_entity", "coded_copy_text")] = "D78"
STATED_LIMITS[("dominant_entity", "coded_copy_number")] = "D78\"""")

# ------------------------------------------------------------- the case --''')

# ---- the must of a carrier pair: four outcomes
sub(r'''    case.must = ("each of " + ", ".join(f"`{n}`" for n in names) + " is excluded as a suspected "
                 "leak, or named by a `strong_single_column` warning, or the headline is at or "
                 "above 0.98 and the `near_perfect_headline` warning names it: " + what)''',
    r'''    case.must = ("each of " + ", ".join(f"`{n}`" for n in names) + " is excluded as a suspected "
                 "leak, or named by a `strong_single_column` warning, or named by the "
                 "`near_perfect_headline` warning, or named in `headline.rests_on`: " + what)''')

# ---- R2: a feature written twice
sub(r'''CHANGES.update({
    "noise_rounded_1dp":''', r'''def ch_feature_copied(case, rng):
    name = case.info.get("num")
    if not name or case.info.get("malformed"):
        return "no numeric feature"
    if case.duplicates == "group":
        return "the run is about rows that agree; a column written twice changes none of that"
    at = case.col(name)
    case.add_column(name + "_again", [row[at] for row in case.rows])
    _added(case, name + "_again", "feature", None)
    _holds(case, f"the numeric feature `{name}` is written twice: `{name}_again` holds the same "
                 "cells. Two exact copies are shuffled together and share one fall (D78 point 3)")
    if not _stops(case) and not case.stop_ok:
        case.claims = [c for c in case.claims if not c.startswith("band_or_named")]
        case.extra.append(("same_fall", name, name + "_again"))
    return None


CHANGES.update({
    "feature_written_twice": ch_feature_copied,
    "noise_rounded_1dp":''')
sub(r'''    "feature_rounded_40": "a numeric feature is put into 40 bands of equal width",''',
    r'''    "feature_written_twice": "a numeric feature is written twice, under a second name",
    "feature_rounded_40": "a numeric feature is put into 40 bands of equal width",''')

# ---- the judging
sub(r'''            elif name in named:
                self.c.outcome[name] = "named"
            else:''', r'''            elif name in named:
                self.c.outcome[name] = "named"
            elif name in listed:
                self.c.outcome[name] = "listed"
            else:''')
sub(r'''        named = self.named_by_near_perfect() if near else []
        silent = []''', r'''        named = self.named_by_near_perfect() if near else []
        listed = [entry.get("column") for entry in self.at("headline.rests_on")]
        silent = []''')
sub(r'''                     "excluded, no `strong_single_column` warning, and "
                     + (f"the `near_perfect_headline` warning names {named}" if near
                        else f"no `near_perfect_headline` warning (skill {skill!r})")''',
    r'''                     "excluded, no `strong_single_column` warning, "
                     + (f"the `near_perfect_headline` warning names {named}" if near
                        else f"no `near_perfect_headline` warning (skill {skill!r})")
                     + f", and `headline.rests_on` names {listed}"''')
sub(r'''    def x_carrier(self, *names):''', r'''    def x_same_fall(self, first, second):
        columns = self.columns()
        a, b = columns.get(self.c.name(first), {}), columns.get(second, {})
        keys = ("shuffled_score", "shuffled_skill", "skill_fall")
        self.require(a.get("role") == b.get("role") == "feature",
                     f"`{first}` and `{second}` are {a.get('role')!r} and {b.get('role')!r}")
        self.require(all(a.get(k) == b.get(k) for k in keys),
                     f"`{first}` and `{second}` hold the same cells and do not share one fall: "
                     f"{[a.get(k) for k in keys]} against {[b.get(k) for k in keys]}")

    def x_carrier(self, *names):''')
sub(r'''        near = isinstance(skill, (int, float)) and skill >= NEAR_PERFECT
        found = self.warned("near_perfect_headline")
        self.require(len(found) == (1 if near else 0),
                     f"{len(found)} `near_perfect_headline` warning(s) with skill {skill!r}")
        if near:
            want = [name for fall, _, name in sorted(falls) if -fall >= NAMED_FALL][:NAMED_MOST]
            got = self.named_by_near_perfect()''',
    r'''        want = [name for fall, _, name in sorted(falls) if -fall >= NAMED_FALL][:NAMED_MOST]
        listed = self.at("headline.rests_on")
        self.require(isinstance(listed, list)
                     and [entry.get("column") for entry in listed] == want,
                     f"`headline.rests_on` names {listed!r}; the columns whose fall is at least "
                     f"{NAMED_FALL}, the first {NAMED_MOST} in order of their fall, are {want}")
        by_name = self.columns()
        for entry in listed:
            self.require(g.close(entry.get("skill_fall"), by_name[entry["column"]]["skill_fall"]),
                         f"`headline.rests_on` gives {entry!r}; the column's record says "
                         f"{by_name[entry['column']]['skill_fall']!r}")
        near = isinstance(skill, (int, float)) and skill >= NEAR_PERFECT
        found = self.warned("near_perfect_headline")
        self.require(len(found) == (1 if near else 0),
                     f"{len(found)} `near_perfect_headline` warning(s) with skill {skill!r}")
        if near:
            got = self.named_by_near_perfect()''')

# ---- the report: the four counts
sub(r'''    tally = {o: sum(o in r["outcome"].values() for r in planted)
             for o in ("excluded", "warned", "named", "silent")}
    whole = {o: sum(set(r["outcome"].values()) == {o} for r in planted)
             for o in ("excluded", "warned", "named")}
    print(f"pairs that plant a carrier of the target: {len(planted)}. Every planted column "
          f"excluded: {whole['excluded']}; warned about by name: {whole['warned']}; named by the "
          f"`near_perfect_headline` warning: {whole['named']}; a mixture of the three: "
          f"{len(planted) - sum(whole.values()) - tally['silent']}; with a column that nothing "
          f"names: {tally['silent']}")''',
    r'''    # a pair counts under the weakest outcome among its planted columns
    weakest = ("silent", "listed", "named", "warned", "excluded")
    tally = {o: 0 for o in weakest}
    for r in planted:
        tally[min(r["outcome"].values(), key=weakest.index)] += 1
    print(f"pairs that plant a carrier of the target: {len(planted)}. Excluded: "
          f"{tally['excluded']}; warned about by name: {tally['warned']}; named by the "
          f"`near_perfect_headline` warning: {tally['named']}; named in `headline.rests_on`: "
          f"{tally['listed']}; none of the four: {tally['silent']} (a pair with two planted "
          "columns counts under the weaker of their outcomes)")''')
p.write_text(t, encoding="utf-8", newline="\n")
print("edited", p)
