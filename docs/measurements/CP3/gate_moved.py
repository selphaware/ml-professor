"""CP3: did any value of the gate move between the judge before CP3 and after it?

    python gate_moved.py <before.json> <after.json>

The two files are written by the `values` command of the CP2 script
(`docs/measurements/CP2/long_rows.py`), which runs every check of a gate on a
reference implementation and keeps every dataset, every check's verdict and
message, every value the gate recorded and every field of every run.

CP3 meant two differences and no others, and this script sorts what it finds
into those two and the rest:

  - the text of the console summary, which now shows more (D39);
  - one new key in `headline`, `n_resamples_skill_used` (D40).

Exit 0 when nothing else differs under a name that existed before.
"""
import json
import sys
from pathlib import Path

NEW_KEY = "headline.n_resamples_skill_used"


def leaves(node, path=""):
    """Every value in a nested structure, under its dotted path."""
    if isinstance(node, dict) and node:
        for key, value in node.items():
            yield from leaves(value, f"{path}.{key}" if path else str(key))
    elif isinstance(node, list) and node:
        for i, value in enumerate(node):
            yield from leaves(value, f"{path}[{i}]")
    else:
        yield path, node


def main(before_file, after_file):
    before = json.loads(Path(before_file).read_text(encoding="utf-8"))
    after = json.loads(Path(after_file).read_text(encoding="utf-8"))
    other = []
    for part in ("datasets", "checks", "records", "runs"):
        same = summary = new_key = 0
        for name, was in before[part].items():
            a = dict(leaves(was))
            b = dict(leaves(after[part][name])) if name in after[part] else {}
            for path in sorted(set(a) | set(b)):
                if path in a and path in b and a[path] == b[path]:
                    same += 1
                elif part == "runs" and path == "stdout" and path in a and path in b:
                    summary += 1
                elif part == "runs" and path.endswith(NEW_KEY) and path not in a:
                    new_key += 1
                else:
                    other.append(f"{part}: {name}: {path}: {a.get(path, '(absent)')!r:.120} -> "
                                 f"{b.get(path, '(absent)')!r:.120}")
        only_after = [name for name in after[part] if name not in before[part]]
        print(f"{part}: {len(before[part])} before, {len(after[part])} after; {same} values "
              f"identical" + (f"; {summary} console summaries reworded" if summary else "")
              + (f"; {new_key} times the new key {NEW_KEY}" if new_key else "")
              + (f"; new: {', '.join(only_after)}" if only_after else ""))
    for line in other:
        print(f"  MOVED  {line}")
    print("NOTHING ELSE MOVED" if not other else f"{len(other)} OTHER VALUES MOVED")
    return 1 if other else 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    sys.exit(main(sys.argv[1], sys.argv[2]))
