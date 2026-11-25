"""T67: the README on the eighth pass (spec 1; D73, D74, D76).

It judges `README.md`, which the builder owns; it neither edits it nor adds files. The
README is read as `test_readme_examples` reads it. No wording is pinned beyond the words
below.
"""
import test_readme_examples as rx


def _joined():
    return " ".join(rx._readme().split())


def test_the_readme_names_what_the_eighth_pass_added():
    text = rx._readme()
    for needed in ("largest_row_share", "largest_row_dominates", "extreme_value",
                   "selection_not_made"):
        assert needed in text, f"the README does not hold {needed!r}"
    found = rx._UNDER_JUDGE.findall(text)
    assert not found, f"the README names a path under the judge's folders: {found}"


def test_the_readme_says_what_the_baseline_is_at_repeated_timestamps():
    text = _joined()
    assert "timestamp before" in text
    for gone in ("only when timestamps are unique", "when timestamps are unique, so"):
        assert gone not in text, f"the README still says {gone!r}"


def test_the_readme_states_the_limits_of_the_eighth_pass():
    text = _joined().lower()
    for needed in ("strong_single_column", "noise", "sorted"):
        assert needed in text, f"the README does not hold {needed!r}"
