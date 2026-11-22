"""CP8, proof 5: is each prototype red or green today, as the test plan lists it?

    python -B red_green.py <junit.xml> [--reference]

<junit.xml> is the report of the whole suite with the prototypes of the seventh
pass laid over it (a folder outside the repo: a copy of `tests/`, the prototype
files copied over it, and a copy of the application beside it), run against the
application as it stands: pytest's `--junitxml`. `tests/test_judge_tools.py` is
left out of that run: it needs `verify/`, and the two tools are run by
themselves. The lists are those of plan/test-plan.md, "Seventh pass (CP8)":

- every case of a file that existed before the pass passes, the one loosened
  test (`test_date_numbers.py`) included;
- every case of `test_guard_review6.py` passes;
- of the five files of T55 to T59, the cases on the green list pass and every
  other case fails.

With `--reference` the report is of the same suite run against the reference
implementation: every case passes but the README's.

Exits 1 when a case is not as listed.
"""
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

GREEN = {
    "test_date_sentinels.py": [
        "test_the_rule_from_both_sides[two_dates_and_nothing_else]",
        "test_the_rule_from_both_sides[four_dates_and_one_other]",
        "test_the_rule_from_both_sides[four_dates_and_two_others]",
        "test_the_rule_from_both_sides[twenty_six_dates_and_three_others]",
        "test_a_few_codes_of_which_some_read_as_dates_are_no_date_column",
        "test_too_many_values_that_are_no_dates_leave_the_column_a_number",
        "test_a_time_column_that_is_no_date_column_is_a_number_as_before",
    ],
    "test_group_after_leak.py": [
        "test_no_combination_of_the_features_is_on_both_sides_in_a_finished_run",
        "test_under_keep_and_error_nothing_changes",
        "test_without_the_leak_the_same_table_runs",
    ],
    "test_fast_cap_groups.py": [
        "test_when_the_full_profile_would_stop_too_its_error_is_raised",
        "test_the_full_profile_runs_the_same_table",
        "test_it_is_for_an_entity_column_with_no_time_column_only",
    ],
    "test_review6_small.py": [],
    "test_readme_review6.py": [],
}
GUARDS = "test_guard_review6.py"


def listed_green(module, name):
    return any(name == g or name.startswith(g + "[") for g in GREEN[module])


def main(argv):
    on_reference = "--reference" in argv
    outcome = {}
    for item in ET.parse(argv[0]).iter("testcase"):
        module = item.get("classname", "").split(".")[-1] + ".py"
        failed = bool({child.tag for child in item} & {"failure", "error", "skipped"})
        outcome[(module, item.get("name", "?"))] = "failed" if failed else "passed"
    wrong, table = [], Counter()
    for (module, name), got in sorted(outcome.items()):
        if module in GREEN:
            want = "passed" if listed_green(module, name) else "failed"
            if on_reference:
                want = "failed" if module == "test_readme_review6.py" else "passed"
            kind = module
        elif module == GUARDS:
            want, kind = "passed", module
        else:
            want, kind = "passed", "the files that existed before the pass"
        table[(kind, got)] += 1
        if got != want:
            wrong.append(f"{module}::{name}: {got}, listed as {want}")
    print(f"{len(outcome)} cases in {argv[0]}")
    print("| File | passed | failed |")
    print("|---|---|---|")
    kinds = ["the files that existed before the pass", GUARDS] + list(GREEN)
    for kind in kinds:
        print(f"| {kind} | {table[(kind, 'passed')]} | {table[(kind, 'failed')]} |")
    new = [k for k in kinds[1:]]
    print(f"| the six new files together | {sum(table[(k, 'passed')] for k in new)} "
          f"| {sum(table[(k, 'failed')] for k in new)} |")
    for line in wrong:
        print("NOT AS LISTED: " + line)
    print("RED AND GREEN: " + ("FAIL" if wrong else "PASS (every case is as the test plan lists it)"))
    return 1 if wrong else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
