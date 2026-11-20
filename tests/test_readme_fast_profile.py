"""T30: the README says which rows the fast profile keeps, under each scheme (spec step 6). Red today.

Under the random scheme the 2,000 rows are drawn from the seed; under time
order they are the most recent whole timestamps. Two ideas are looked for in
the blocks that mention the cap, not a sentence. This judges `README.md`,
which the builder owns; it does not edit it.
"""
import re

import mlp_helpers as h

_ITEM_START = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
_CAP = re.compile(r"\b2,?000\b")
_RANDOM = re.compile(r"\b(?:seed|random|randomly)\b", re.I)
_RECENT = re.compile(r"\b(?:most\s+recent|latest|newest|last)\b", re.I)
_TIME = re.compile(r"\b(?:timestamps?|dates?|days?)\b", re.I)


def _blocks(text):
    """Paragraphs, with each list item a block of its own."""
    blocks = []
    for paragraph in re.split(r"\n[ \t]*\n", text):
        current = []
        for line in paragraph.splitlines():
            if _ITEM_START.match(line) and current:
                blocks.append("\n".join(current))
                current = []
            current.append(line)
        if current:
            blocks.append("\n".join(current))
    return blocks


def test_readme_says_which_rows_the_fast_profile_keeps_under_each_scheme():
    path = h.REPO_ROOT / "README.md"
    assert path.is_file(), "README.md is missing"
    cap_blocks = [b for b in _blocks(path.read_text(encoding="utf-8")) if _CAP.search(b)]
    assert cap_blocks, "no paragraph or list item of README.md mentions the cap of 2,000 rows"
    joined = "\n".join(cap_blocks)
    assert _RANDOM.search(joined), f"the random scheme's rule is not stated:\n{joined}"
    assert _RECENT.search(joined) and _TIME.search(joined), (
        f"the time-ordered rule (the most recent timestamps) is not stated:\n{joined}")
