r"""patch_battery10.py brought up to D80:

  (1) the rests-on floor: `headline.rests_on_floor` is 0.05 or the half-width of the
      headline skill's interval, the larger; where there is no such interval at least
      0.05 (its stand-in is the tool's own bootstrap, which the judge does not redraw);
      the list holds the columns whose fall is at least the floor;
  (2) a carrier pair holds when it ends in one of the four outcomes, or when the
      headline with the carrier is within that run's floor of the headline without
      it; of two planted columns either one named counts; where the headline skill
      has no value the pair is a stated limit citing D36. The D78 stated limits go,
      and the list is the four kinds of D76 plus D36.
"""
from pathlib import Path

p = Path(__file__).resolve().parent / "patch_battery10.py"
t = p.read_text(encoding="utf-8")


def sub(old, new, count=1):
    global t
    assert t.count(old) == count, (t.count(old), old[:100])
    t = t.replace(old, new)


# ---- the stated limits: D76's four kinds and D36
a = t.index("# ------------------------------------------------- the stated limits of D78 --")
b = t.index("# ------------------------------------------------------------- the case --")
t = t[:a] + '''# ------------------------------------------------- the stated limits of D80 --
sub("""    ("panel", "copy_noise_5pct"): "D76",
}""", """    ("panel", "copy_noise_5pct"): "D76",
    # where the headline skill has no value (a baseline with no error), nothing can be
    # said of how far a carrier moves it, and no fall is computed (D36, D80 point 2)
    ("flat_tail", "two_columns_together"): "D36",
}""")

''' + t[b:]

# ---- the carrier claim
sub('''    def x_carrier(self, *names):
        columns, skill = self.columns(), self.at("headline.skill")''',
    '''    def x_carrier(self, *names):
        columns, skill = self.columns(), self.at("headline.skill")
        base = self.base["doc"]["headline"].get("skill") \\
            if self.base and self.base.get("finished") else None
        floor = self.at("headline.rests_on_floor")
        lift = skill - base if isinstance(skill, (int, float)) \\
            and isinstance(base, (int, float)) else None
        self.c.carrier = {"skill": skill, "base": base, "floor": floor, "lift": lift,
                          "falls": {n: columns.get(n, {}).get("skill_fall") for n in names}}''')
sub('''        self.require(not silent,
                     f"{silent} carr{'ies' if len(silent) == 1 else 'y'} the target and "''',
    '''        if len(silent) < len(names):
            return                  # of two planted columns, either one named counts (D80)
        within = lift is not None and isinstance(floor, (int, float)) and abs(lift) <= floor
        if within:
            for name in names:
                self.c.outcome[name] = "within"
            return
        self.require(not silent,
                     f"the headline moves {lift!r} from the run without it (floor {floor!r}), "
                     f"and {silent} carr{'ies' if len(silent) == 1 else 'y'} the target and "''')

# ---- the list and its floor, on every finished run
sub('''        want = [name for fall, _, name in sorted(falls) if -fall >= NAMED_FALL][:NAMED_MOST]
        listed = self.at("headline.rests_on")''',
    '''        floor = self.at("headline.rests_on_floor")
        low, high = self.at("headline.skill_low"), self.at("headline.skill_high")
        if skill is None:
            self.require(floor is None, f"skill has no value and the floor is {floor!r}")
        elif isinstance(low, (int, float)) and isinstance(high, (int, float)):
            self.require(isinstance(floor, (int, float))
                         and g.close(floor, max(NAMED_FALL, (high - low) / 2), 1e-9),
                         f"`headline.rests_on_floor` is {floor!r}; with the skill interval "
                         f"{low!r} to {high!r} it is {max(NAMED_FALL, (high - low) / 2)!r}")
        else:
            self.require(isinstance(floor, (int, float)) and floor >= NAMED_FALL,
                         f"`headline.rests_on_floor` is {floor!r}, below {NAMED_FALL}")
        cut = floor if isinstance(floor, (int, float)) else NAMED_FALL
        want = [name for fall, _, name in sorted(falls) if -fall >= cut][:NAMED_MOST]
        listed = self.at("headline.rests_on")''')
sub('''                     f"`headline.rests_on` names {listed!r}; the columns whose fall is at least "
                     f"{NAMED_FALL}, the first {NAMED_MOST} in order of their fall, are {want}")''',
    '''                     f"`headline.rests_on` names {listed!r}; the columns whose fall is at least "
                     f"the floor {cut!r}, the first {NAMED_MOST} in order of their fall, are "
                     f"{want}")''')
sub('''                         f"fall is at least {NAMED_FALL}, the first {NAMED_MOST} in order of "
                         f"their fall, are {want}")''',
    '''                         f"fall is at least the floor {cut!r}, the first {NAMED_MOST} in order "
                         f"of their fall, are {want}")''')

# ---- the record and the counts
sub('''        self.outcome = {}                       # a planted carrier of the target -> what became of it
''', '''        self.outcome = {}                       # a planted carrier of the target -> what became of it
        self.carrier = None                     # its skill, the run's without it, the floor
''')
sub('''                  "outcome": dict(case.outcome)}''',
    '''                  "outcome": dict(case.outcome), "carrier": case.carrier}''')
sub('''    # a pair counts under the weakest outcome among its planted columns
    weakest = ("silent", "listed", "named", "warned", "excluded")
    tally = {o: 0 for o in weakest}
    for r in planted:
        tally[min(r["outcome"].values(), key=weakest.index)] += 1
    print(f"pairs that plant a carrier of the target: {len(planted)}. Excluded: "
          f"{tally['excluded']}; warned about by name: {tally['warned']}; named by the "
          f"`near_perfect_headline` warning: {tally['named']}; named in `headline.rests_on`: "
          f"{tally['listed']}; none of the four: {tally['silent']} (a pair with two planted "
          "columns counts under the weaker of their outcomes)")''',
    '''    # a pair counts under the strongest outcome among its planted columns: of two, either
    # one named counts as named (D80 point 2)
    strongest = ("excluded", "warned", "named", "listed", "within", "silent")
    tally = {o: 0 for o in strongest}
    for r in planted:
        tally[min(r["outcome"].values(), key=strongest.index)] += 1
    print(f"pairs that plant a carrier of the target: {len(planted)}. Excluded: "
          f"{tally['excluded']}; warned about by name: {tally['warned']}; named by the "
          f"`near_perfect_headline` warning: {tally['named']}; named in `headline.rests_on`: "
          f"{tally['listed']}; the headline within the run's floor of the run without it: "
          f"{tally['within']}; none of the five: {tally['silent']}")''')
p.write_text(t, encoding="utf-8", newline="\n")
print("edited")
