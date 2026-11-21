"""T38: the README on copies of holdout rows among the training rows (spec 1, 2.1, step 8a; D49). Red today.

Ideas are looked for in blocks (paragraphs, and each list item as a block of
its own), whole words, case ignored. This judges `README.md`, which the
builder owns; it does not edit it.
"""
import re

import mlp_helpers as h

_ITEM_START = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
_OPTION = re.compile(r"--duplicates\b")
_ARGUMENT = re.compile(r"\bduplicates\s*=")
_CODE = re.compile(r"\bholdout_copies_in_training\b")
_KEEP = re.compile(r"\bkeep\b", re.I)
_SHARE = re.compile(r"(?<![\d.])1\s?%")
_COUNT = re.compile(r"\b(?:10|ten)\b", re.I)
_TIME = re.compile(r"\btime\w*", re.I)
_CLAIM = re.compile(r"no\s+fitted\s+step\s+has\s+seen", re.I)
_COPY_WORD = re.compile(r"\b(?:copy|copies|duplicat\w*|repeat\w*)\b", re.I)
_HOLDOUT = re.compile(r"\bholdout\b", re.I)


def _readme():
    path = h.REPO_ROOT / "README.md"
    assert path.is_file(), "README.md is missing"
    return path.read_text(encoding="utf-8")


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


def test_readme_names_the_argument_and_the_error():
    text = _readme()
    assert _OPTION.search(text), "README.md does not name --duplicates"
    assert _ARGUMENT.search(text), "README.md does not show the Python argument duplicates="
    assert _CODE.search(text), "README.md does not name holdout_copies_in_training"
    joined = "\n".join(b for b in _blocks(text) if _OPTION.search(b) or _CODE.search(b))
    assert _KEEP.search(joined), f"the value keep is not named:\n{joined}"
    assert _SHARE.search(joined), f"the share of the rule (1%) is not stated:\n{joined}"
    assert _COUNT.search(joined), f"the count of the rule (10 rows) is not stated:\n{joined}"
    assert _TIME.search(joined), f"the scheme the rule does not apply to is not named:\n{joined}"


def test_the_claim_about_unseen_holdout_rows_is_true_as_worded():
    """D49: the claim that no fitted step has seen the holdout rows says what copies do to it."""
    blocks = _blocks(_readme())
    claims = [b for b in blocks if _CLAIM.search(b)]
    for block in claims:
        assert _COPY_WORD.search(block), f"this block makes the claim and says nothing of copies:\n{block}"
    if not claims:
        assert any(_COPY_WORD.search(b) and _HOLDOUT.search(b) for b in blocks), (
            "no block of README.md speaks of copies of holdout rows")
