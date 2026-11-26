"""T75: the README on the ninth pass (spec 1, step 12a; D77, D78, D80).

It judges `README.md`, which the builder owns; it neither edits it nor adds files. The
README is read as `test_readme_examples` reads it. No wording is pinned beyond the words
below.
"""
import test_readme_examples as rx


def _joined():
    return " ".join(rx._readme().split())


def test_the_readme_names_what_the_ninth_pass_added():
    text = rx._readme()
    for needed in ("shuffled_score", "shuffled_skill", "skill_fall", "rests_on",
                   "rests_on_floor", "near_perfect_headline", "holdout_far_above_quarter",
                   "shuffled_score_not_computed"):
        assert needed in text, f"the README does not hold {needed!r}"
    found = rx._UNDER_JUDGE.findall(text)
    assert not found, f"the README names a path under the judge's folders: {found}"


def test_the_readme_says_what_the_statement_and_its_floor_are():
    text = _joined().lower()
    for needed in ("shuffled", "fall", "0.05", "half-width", "interval", "five"):
        assert needed in text, f"the README does not hold {needed!r}"


def test_the_readme_says_what_a_near_perfect_headline_asks_of_the_caller():
    text = _joined().lower()
    for needed in ("0.98", "easy", "known before the outcome", "half"):
        assert needed in text, f"the README does not hold {needed!r}"
