"""Bring the edited battery (the output of patch_battery.py) up to D74 and D76.

    python -B patch_battery2.py <battery.py made by patch_battery.py>

1. The claim for one extreme cell (D74 point 3): skill inside the band; or the result
   names the row or the value, and the same call on the table without the named
   rows is inside the band. The judge makes that second call itself.
2. The judge's own count of `largest_row_dominates` records, with the selection folds.
3. Under time order (D74 point 2, D76 points 1 and 2): a copy of the target with 2%
   of its cells altered, a curve of the target, and a copy with noise of 1% of the
   target's spread must each be excluded or at least warned about. A copy with
   noise of 5% is asserted the same way and is a stated limit (D76).
4. The table sorted by a whole-number measure is asserted as the spec's intent has it
   (the measure stays a feature) and is a stated limit (D31; D76 point 6).
5. One cell of -999 in `visits` of `messy`: in band, or named (D76 point 3).
"""
import sys
from pathlib import Path

path = Path(sys.argv[1])
s = path.read_text(encoding="utf-8")


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), old[:80])
    s = s.replace(old, new)


# ---- 1. the claim ------------------------------------------------------------
start = s.index("    def c_band_or_row(self, name):")
end = s.index("    def c_fold_rule(self):")
s = s[:start] + '''    def named_rows(self):
        """The rows the result names: the largest row of a squared error where its
        share is above 0.5 (the holdout or the selection folds, a configuration or
        the baseline), and every row that holds a value an `extreme_value` warning
        names."""
        rows = set()
        parts = [self.at("headline"), self.at("cv")] + list(self.at("models"))
        for part in parts:
            for key in ("", "baseline_"):
                share, row = part.get(key + "largest_row_share"), part.get(key + "largest_row")
                if isinstance(share, (int, float)) and share > 0.5 and row is not None:
                    rows.add(int(row))
        header, source = self.r["header"], self.r["source"]
        for w in self.warned("extreme_value"):
            found = re.search(r"holds the value (\\S+) in (\\d+) row", w["message"])
            if not found or w["column"] not in header:
                continue
            value, at = float(found.group(1)), header.index(w["column"])
            for i, line in enumerate(source):
                try:
                    cell = float(line[at])
                except ValueError:
                    continue
                if cell == value:
                    rows.add(i)
        return rows

    def c_band_or_row(self, name):
        """D74 (3): skill inside the band; or the result names the row or the value,
        and the same call on the table without the named rows is inside the band."""
        skill = self.at("headline.skill")
        low, high = BANDS[name]
        if isinstance(skill, (int, float)) and low <= skill <= high:
            return
        shown, where = skill if skill is None else round(skill, 4), self.where_touched()
        named = self.named_rows()
        codes = sorted({w["code"] for w in self.at("warnings")
                        if w["code"] in ("largest_row_dominates", "extreme_value")})
        self.require(named, f"skill is {shown}, outside the band of `{name}` ({low} to {high}), "
                            f"and the result names no row and no value. The changed row(s): "
                            f"{where}. model {self.at('headline.model')}, baseline "
                            f"{self.at('headline.baseline')}")
        self.require(len(named) <= 20, f"skill is {shown}, outside the band; the result names "
                                       f"{len(named)} rows, too many to be the changed cell")
        case = copy.copy(self.c)
        case.change = self.c.change + "__without"
        case.rows = [list(row) for i, row in enumerate(self.c.rows) if i not in named]
        case.blank_after = set()
        again = read_outputs(launch(case, _RERUN["package"], _RERUN["work"]))
        self.require(again.get("finished"),
                     f"skill is {shown}, outside the band; the result names row(s) "
                     f"{sorted(named)} ({codes}); the same call without them did not finish: "
                     f"exit {again.get('rc')}, `{again.get('code')}`")
        without = again["doc"]["headline"]["skill"]
        self.require(isinstance(without, (int, float)) and low <= without <= high,
                     f"skill is {shown}, outside the band of `{name}` ({low} to {high}); the "
                     f"result names row(s) {sorted(named)} ({codes}); the same call without "
                     f"them gives skill {without!r}: still outside the band. The changed "
                     f"row(s): {where}")

''' + s[end:]
rep('''OPEN = {pair: finding for finding, pairs in _OPEN.items() for pair in pairs}''',
    '''OPEN = {pair: finding for finding, pairs in _OPEN.items() for pair in pairs}
_RERUN = {}                 # the package and the work folder, for a second call''')
rep('''    work = Path(args.work).resolve() if args.work else WORK
    try:''', '''    work = Path(args.work).resolve() if args.work else WORK
    _RERUN.update(package=package, work=work)
    try:''')
if "\nimport copy\n" not in s:
    rep("import argparse\n", "import argparse\nimport copy\n")

# ---- 2. the judge's own count of `largest_row_dominates` records --------------
rep("""        self.require(warned == expected,
                     f"{warned} `largest_row_dominates` warning(s); the judge expects "
""", """        cv = self.at("cv")
        absent = [key for key in keys if key not in cv] + [
            f"models[{m.get('config')}].{key}" for m in self.at("models")
            for key in ("largest_row_share", "largest_row") if key not in m]
        self.require(not absent, f"the selection folds' statement is missing: {absent[:6]}")
        ruled = [m["config"] for m in self.at("models")
                 if isinstance(m["largest_row_share"], (int, float))
                 and m["largest_row_share"] > 0.5]
        expected += bool(ruled)
        over = cv["baseline_largest_row_share"]
        expected += bool(isinstance(over, (int, float)) and over > 0.5)
        self.require(warned == expected,
                     f"{warned} `largest_row_dominates` warning(s); the judge expects "
""")

# ---- 3. under time order: altered cells, a curve, noise -----------------------
rep('''CHANGES.update(EXTENSION_CHANGES)''', '''NOISE_LIMIT_RUNS = ("random_walk", "int_stamps", "panel")


def _told_copy(make_cells, what, only=None):
    """Under time order, a column derived from the target that must be excluded as a
    suspected leak or at least named by a `strong_single_column` warning (D74 point
    2; D76 points 1 and 2)."""
    def change(case, rng):
        if not case.regression():
            return "the target is not a regression target"
        if not isinstance(case.time, str):
            return ("asserted under time order, where the previous value is a baseline; "
                    "under the other schemes the screen already excludes or warns")
        if only is not None and case.run not in only:
            return ("asserted on the runs where the previous value is the baseline in use: "
                    + ", ".join(only))
        usable = _usable_rows(case)
        values = [None if g.is_missing(c) else float(c) for c in case.target_cells()]
        cells = make_cells(values, usable, rng, case)
        case.add_column("target_copy", cells)
        case.claims = [c for c in case.claims if not c.startswith("band")
                       and c not in ("walk_baselines", "panel_baselines", "flat", "int_keys")]
        case.extra.append(("copy_told", "target_copy"))
        case.kind = "structure"
        case.must = ("the added column `target_copy` is excluded as a suspected leak, or at "
                     "least named by a `strong_single_column` warning: " + what)
        return None
    return change


def _cells_altered(values, usable, rng, case):
    cells = ["" if v is None else g._f(3.7 * v + 12, 9) for v in values]
    picks = rng.sample(usable, max(1, round(0.02 * len(usable))))
    moved = [cells[i] for i in picks]
    for i, value in zip(picks, moved[1:] + moved[:1]):
        cells[i] = value
    case.touched += picks
    return cells


def _cells_noisy(share):
    def make(values, usable, rng, case):
        present = [values[i] for i in usable]
        mean = sum(present) / len(present)
        spread = math.sqrt(sum((v - mean) ** 2 for v in present) / len(present))
        return ["" if v is None else g._f(3.7 * v + 12 + rng.gauss(0, 3.7 * share * spread), 9)
                for v in values]
    return make


def _cells_curve(values, usable, rng, case):
    return ["" if v is None else g._f((v + 1000.0) ** 2, 6) for v in values]


def ch_measure_sentinel(case, rng):
    """D76 (3): -999 in `visits` of `messy`, whole numbers from 1,000 to 100,000. The
    value is not far outside the column and moves no number: the run is in band, or
    the value is named."""
    if _messy_only(case):
        return _messy_only(case)
    k = case.pick(rng)
    case.rows[k][case.col("visits")] = "-999"
    case.touched.append(k)
    _holds(case, "one cell of the whole-number measure `visits` set to -999, just below "
                 "its smallest value")
    _named_row(case)
    return None


EXTENSION_CHANGES.update({
    "copy_2pct_cells_altered": _told_copy(_cells_altered, "a rescaled copy of the target in "
                                          "which the values of 2% of the rows are rotated "
                                          "among them"),
    "copy_curve": _told_copy(_cells_curve, "the square of the target plus 1,000, a curve that "
                             "rises with it"),
    "copy_noise_1pct": _told_copy(_cells_noisy(0.01), "a rescaled copy of the target with "
                                  "noise of 1% of the target's spread added"),
    "copy_noise_5pct": _told_copy(_cells_noisy(0.05), "a rescaled copy of the target with "
                                  "noise of 5% of the target's spread added: as large as the "
                                  "previous value's own miss. A stated limit (D76): such a "
                                  "column cannot be told from a strong feature measured at "
                                  "the same moment", only=NOISE_LIMIT_RUNS),
    "measure_sentinel_-999": ch_measure_sentinel,
})
CHANGES.update(EXTENSION_CHANGES)''')
rep('''    def x_roles_as_base(self):''', '''    def x_copy_told(self, name):
        got = self.columns().get(name, {})
        out = (got.get("role"), got.get("reason")) == ("excluded", "suspected_leak")
        warned = name in self.names_warned("strong_single_column")
        self.require(out or warned,
                     f"the added column `{name}` is a feature (role {got.get('role')!r}, reason "
                     f"{got.get('reason')!r}) and no `strong_single_column` warning names it; "
                     f"skill {self.at('headline.skill')!r} against "
                     f"{self.at('headline.baseline')!r}, model {self.at('headline.model')}")

    def x_roles_as_base(self):''')
rep('''def show(case, record, result, package, lead):''', '''CHANGE_WORDS.update({
    "copy_2pct_cells_altered": "a column `target_copy` is added: the target rescaled, the "
                               "values of 2% of the rows rotated among them",
    "copy_curve": "a column `target_copy` is added: the square of the target plus 1,000",
    "copy_noise_1pct": "a column `target_copy` is added: the target rescaled, with noise of "
                       "1% of the target's spread",
    "copy_noise_5pct": "a column `target_copy` is added: the target rescaled, with noise of "
                       "5% of the target's spread",
    "measure_sentinel_-999": "one cell of the whole-number measure `visits` is set to -999",
})


def show(case, record, result, package, lead):''')

# ---- 4. the stated limits: exactly four kinds (D76 point 6) --------------------
rep('''    ("int_stamps", "missing_cell_date"): "D55",     # a CSV whole-number column with an
                                                    # empty cell loads as floats
}''', '''    ("int_stamps", "missing_cell_date"): "D55",     # a CSV whole-number column with an
                                                    # empty cell loads as floats
    ("messy", "sorted_by_measure"): "D31",          # a distinct whole-number measure in a
                                                    # table sorted by it is called id-like
    # a copy of the target with noise above 1% of its spread, under time order (D76)
    ("random_walk", "copy_noise_5pct"): "D76",
    ("int_stamps", "copy_noise_5pct"): "D76",
    ("panel", "copy_noise_5pct"): "D76",
}''')
path.write_bytes(s.encode("utf-8"))
print("battery brought up to D74 and D76")
