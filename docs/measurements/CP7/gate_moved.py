"""CP7, proof 3: what moved between two states of the gate, by kind.

    python -B gate_moved.py <before.json> <after.json>

The two files are written by docs/measurements/CP2/long_rows.py `values`: every
dataset, every check, every value the gate recorded and every field of every
run, at the gate seed. `long_rows.py compare` lists the first twenty of each
kind; this prints every one, grouped by the field that moved, so that the claim
"no score, count or band moved" can be read off.
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

before, after = (json.loads(Path(p).read_text(encoding="utf-8")) for p in sys.argv[1:3])


def flat(node, path=""):
    if isinstance(node, dict):
        for key, value in node.items():
            yield from flat(value, f"{path}.{key}" if path else str(key))
    elif isinstance(node, list):
        for i, value in enumerate(node):
            yield from flat(value, f"{path}[{i}]")
    else:
        yield path, node


for part in before:
    # `state` holds where the two judges were and the hashes of their files
    if part == "state" or not isinstance(before[part], dict) or part not in after:
        continue
    a, b = dict(flat(before[part])), dict(flat(after[part]))
    names_a = {k.split(".")[0] for k in a}
    names_b = {k.split(".")[0] for k in b}
    moved = {k: (a[k], b[k]) for k in a if k in b and a[k] != b[k]}
    gone = [k for k in a if k not in b and k.split(".")[0] in names_b]
    new = [k for k in b if k not in a and k.split(".")[0] in names_a]
    print(f"{part}: {len(a)} values before under {len(names_a)} names; {len(moved)} moved, "
          f"{len(gone)} gone, {len(new)} new under an existing name; "
          f"only after: {', '.join(sorted(names_b - names_a)) or 'none'}")
    kinds = defaultdict(list)
    for key, (x, y) in moved.items():
        name, _, field = key.partition(".")
        kinds[field].append((name, x, y))
    for field, items in sorted(kinds.items()):
        shifts = Counter((repr(x)[:40], repr(y)[:40]) for _, x, y in items)
        shown = "; ".join(f"{x} -> {y} ({n})" for (x, y), n in shifts.most_common(3))
        print(f"    moved  {field}: {len(items)} run(s): {', '.join(sorted(n for n, _, _ in items))}")
        if field not in ("command",):
            print(f"           {shown}")
    for label, keys in (("gone", gone), ("new", new)):
        fields = Counter(k.partition(".")[2] for k in keys)
        for field, n in sorted(fields.items()):
            print(f"    {label}   {field}: {n} run(s)")
