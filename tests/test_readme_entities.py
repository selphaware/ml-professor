"""T47: the README on the fifth pass (spec 1; D56 to D60). Red today.

It judges `README.md`, which the builder owns. No wording is pinned beyond
the names below.
"""
import re

import mlp_helpers as h


def _readme():
    path = h.REPO_ROOT / "README.md"
    assert path.is_file(), "README.md is missing"
    return path.read_text(encoding="utf-8")


def test_the_readme_names_the_new_arguments():
    text = _readme()
    for needle in ("--group-col", "--no-entities", "--no-time-order", "group_col",
                   "time_col=False"):
        assert needle in text, needle
    assert re.search(r"duplicates[= ][\"']?group\b", text), "duplicates group"


def test_the_readme_says_that_a_date_column_must_be_declared_or_dismissed():
    text = _readme()
    assert "time_order_undeclared" in text
    assert "entities_undeclared" in text
