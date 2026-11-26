"""CP10, proof 5: a run of the suite with the prototypes of the ninth pass laid over it, on
the application as it stands, against the plan's red and green lists
(plan/test-plan.md, "Ninth pass", "What must fail today").

    python -B red_green10.py <junit.xml>
"""
import sys
import xml.etree.ElementTree as ET
from collections import Counter

NEW = {"test_rests_on", "test_near_perfect", "test_screen_levels", "test_order_ties",
       "test_screen_marked", "test_large_holdout", "test_readme_pass9", "test_guard_review8"}
GREEN = {
    "test_near_perfect::test_no_warning_below_the_level",
    "test_screen_levels::test_a_column_excluded_by_another_reading_has_no_sentence_on_group_col",
    "test_screen_levels::test_few_values_are_counted_over_the_training_rows",
    "test_screen_levels::test_eight_bands_are_warned_about_and_kept",
    "test_screen_levels::test_an_honest_column_of_ten_levels_is_a_quiet_feature",
    "test_screen_levels::test_in_a_classification_too",
    "test_screen_levels::test_a_numeric_column_of_ten_values_is_a_quiet_feature",
    "test_screen_marked::test_whatever_the_size_of_the_change[1.0]",
    "test_screen_marked::test_wild_target_cells_beyond_the_share_make_no_column_a_leak",
    "test_large_holdout::test_below_half_there_is_no_warning",
    "test_large_holdout::test_an_ordinary_time_split_has_no_warning",
    "test_guard_review8::test_m60_a_classification_code_of_six_rows_a_value",
    "test_guard_review8::test_m63_a_target_of_noise_with_seven_sentinel_cells",
    "test_guard_review8::test_m63_under_time_order_with_entities",
}


def green(name):
    return name in GREEN or name.split("[")[0] in GREEN


cases = []
for case in ET.parse(sys.argv[1]).getroot().iter("testcase"):
    mod = case.get("classname", "").split(".")[-1]
    name = f"{mod}::{case.get('name')}"
    failed = any(child.tag in ("failure", "error") for child in case)
    cases.append((mod, name, failed))
wrong = []
count = Counter()
for mod, name, failed in cases:
    if mod in NEW:
        want_red = not green(name)
        count["new red" if failed else "new green"] += 1
        if failed != want_red:
            wrong.append(("should be " + ("red" if want_red else "green"), name))
    else:
        count["existing failed" if failed else "existing passed"] += 1
        if failed:
            wrong.append(("an existing case fails", name))
print(f"{len(cases)} cases: {dict(count)}")
print("not as the plan lists:", len(wrong))
for w in wrong:
    print("  ", w)
