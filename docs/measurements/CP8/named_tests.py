"""CP8, proof 5: does each broken copy fail the test that is named for it?

    python -B named_tests.py <folder of kept reports> [--new-files]

<folder of kept reports> is the folder a run of `verify/mutation_check.py --keep`
names in its first line: one `<copy>.junit.xml` for each case. For each broken
copy, the cases that fail on it and pass on the good reference are read from
those reports. The names come from two places:

- the table of small faults in CLAUDE.md (M1 to M39): every test a row names
  must be among the cases its copy fails (for a parametrised test, at least
  one case of it; a row that names a file, at least one case of the file);
- the list below: the test of the seventh pass named for each of F12b, F14b
  and F15 (spec section 8), and the second test of each of the eleven faults
  that one test held alone (spec 8.1).

`--new-files` also prints, for every copy, the cases of the seven new test
files that fail on it, so that a guard which fails on a copy it is not named
for shows. Exits 1 when a named test is missing.
"""
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
G = "test_guard_review6.py::"
ALSO = {
    "F12b": ["test_group_after_leak.py::test_a_copy_of_the_target_under_group_stops_the_run"],
    "F14b": ["test_date_sentinels.py::"
             "test_a_column_of_dates_with_a_value_that_is_no_date_is_still_a_date_column"],
    "F15": ["test_fast_cap_groups.py::"
            "test_an_entity_larger_than_the_cap_stops_the_fast_profile"],
    "M1": [G + "test_m1_the_final_fit_is_centred_on_its_own_rows"],
    "M5": [G + "test_m5_a_shuffle_that_ties_counts_where_only_some_of_them_tie"],
    "M6": [G + "test_m6_an_infinite_holdout_prediction_in_a_regression_falls_back"],
    "M8": [G + "test_m8_the_interval_from_whole_entities_is_the_middle_95_percent"],
    "M10": [G + "test_m10_the_fast_cap_is_a_sample_whatever_the_order_of_the_file"],
    "M11": [G + "test_m11_cv_std_of_two_scored_folds_is_their_sample_deviation"],
    "M12": [G + "test_m12_tied_predictions_in_a_resample_of_blocks_and_of_entities"],
    "M17": [G + "test_m17_the_fast_cap_keeps_whole_entities_of_seven_rows"],
    "M21": [G + "test_m21_a_resample_of_whole_entities_has_one_score"],
    "M23": [G + "test_m23_a_panel_between_the_two_cut_offs_is_warned_about_on_the_command_line"],
    "M28": [G + "test_m28_eight_digit_text_orders_a_panel_and_read_time_reads_it"],
}
NEW_FILES = ("test_guard_review6.py", "test_date_sentinels.py", "test_group_after_leak.py",
             "test_fast_cap_groups.py", "test_review6_small.py", "test_readme_review6.py")


def named_in_claude_md():
    """{fault: [test, ...]} from the rows `| M<n> | ... | ... |` of CLAUDE.md."""
    named = {}
    for line in (REPO / "CLAUDE.md").read_text(encoding="utf-8").splitlines():
        row = re.match(r"^\| (M\d+) \|.*\|(.*)\|\s*$", line)
        if not row:
            continue
        tests, last_file = [], None
        for item in re.findall(r"`([^`]+)`", row.group(2)):
            if item.startswith("::"):
                item = last_file + item
            elif "::" not in item:
                item += "::"
            last_file = item.split("::")[0]
            tests.append(item)
        named[row.group(1)] = tests
    return named


def failing_in(report):
    """The cases a kept report shows as failing, named as the mutation check names them."""
    failed = set()
    for item in ET.parse(report).iter("testcase"):
        if {child.tag for child in item} & {"failure", "error", "skipped"}:
            failed.add(f"{item.get('classname', '').split('.')[-1]}.py::{item.get('name', '?')}")
    return failed


def order(copy):
    kind, number, form = re.match(r"([FM])(\d+)([a-z]?)", copy).groups()
    return kind, int(number), form


def main(argv):
    kept = Path(argv[0])
    good = failing_in(kept / "good.junit.xml")
    named = named_in_claude_md()
    for copy, tests in ALSO.items():
        named.setdefault(copy, []).extend(tests)
    copies = sorted((p.name[:-len(".junit.xml")] for p in kept.glob("*.junit.xml")
                     if p.name != "good.junit.xml"), key=order)
    print(f"{len(copies)} broken copies; the good reference fails {len(good)} case(s)"
          + (": " + ", ".join(sorted(good)) if good else ""))
    print("| Copy | Cases failing on it | Test named for it | Cases of that test failing |")
    print("|---|---|---|---|")
    missing = []
    for copy in copies:
        caught = sorted(failing_in(kept / f"{copy}.junit.xml") - good)
        tests = named.get(copy, [])
        if not tests:
            print(f"| {copy} | {len(caught)} | (the gate's; none named) | |")
        for k, test in enumerate(tests):
            hits = [c for c in caught if c == test or c.startswith(test + "[")
                    or (test.endswith("::") and c.startswith(test))]
            print(f"| {copy if k == 0 else ''} | {len(caught) if k == 0 else ''} | `{test}` "
                  f"| {len(hits)} |")
            if not hits:
                missing.append((copy, test))
    if "--new-files" in argv:
        print()
        print("Cases of the new test files that fail on each copy:")
        for copy in copies:
            caught = sorted(failing_in(kept / f"{copy}.junit.xml") - good)
            new = [c for c in caught if c.split("::")[0] in NEW_FILES]
            if new:
                print(f"  {copy}: " + ", ".join(new))
    for copy, test in missing:
        print(f"MISSING: {copy} does not fail {test}")
    print("NAMED TESTS: " + ("FAIL" if missing else
                             "PASS (every broken copy fails each test named for it)"))
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
