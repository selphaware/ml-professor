"""CP10, proof 3: every value of every gate run that moved between two `long_rows.py values`
files, classified: the console summary's new line, the new keys of the statement, the new
warnings, and anything else (which must be nothing).

    python -B values_moved.py <before.json> <after.json>
"""
import json
import re
import sys
from collections import Counter

a = json.load(open(sys.argv[1], encoding="utf-8"))
b = json.load(open(sys.argv[2], encoding="utf-8"))


def flat(node, prefix=""):
    if isinstance(node, dict):
        for k, v in node.items():
            yield from flat(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from flat(v, f"{prefix}[{i}]")
    else:
        yield prefix, node


ra, rb = a["runs"], b["runs"]
kinds, other = Counter(), []
NEW_KEYS = ("shuffled_score", "shuffled_skill", "skill_fall", "rests_on", "rests_on_floor")
for run in ra:
    fa, fb = dict(flat(ra[run])), dict(flat(rb[run]))
    for key in sorted(set(fa) | set(fb)):
        if fa.get(key) == fb.get(key):
            continue
        if key.endswith("stdout"):
            la = fa.get(key, "").splitlines()
            lb = fb.get(key, "").splitlines()
            extra = [x for x in lb if x not in la]
            gone = [x for x in la if x not in lb]
            if not gone and all(x.startswith("rests on") or "near_perfect_headline" in x
                                for x in extra):
                kinds["stdout: the `rests on` line added" + (" and the near-perfect warning's"
                      if any("near_perfect" in x for x in extra) else "")] += 1
                continue
        if any(re.search(rf"\b{k}\b", key) for k in NEW_KEYS) and key not in fa:
            kinds["a key of the statement, new"] += 1
            continue
        if "warnings" in key and key not in fa and "near_perfect_headline" in json.dumps(
                rb[run].get("profile", {}).get("warnings", [])):
            kinds["the near-perfect warning, new: " + run] += 1
            continue
        if key.endswith("profile_bytes") or key.endswith("profile.json bytes") or "bytes" in key:
            kinds["the bytes of profile.json (it holds the new keys)"] += 1
            continue
        other.append((run, key, fa.get(key), fb.get(key)))
print(f"runs before: {len(ra)}; after: {len(rb)}, of which new: {sorted(set(rb) - set(ra))}")
for k, v in sorted(kinds.items()):
    print(f"  {v:5d}  {k}")
print(f"anything else that moved: {len(other)}")
for o in other[:30]:
    print("   ", str(o)[:300])
