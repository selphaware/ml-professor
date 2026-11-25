"""CP9: do runs of the battery give the same verdict on every pair?

    python -B same_verdicts.py <label>=<the battery's --json file> <label>=<file> ...

The first file is the one the others are compared with. For each pair of a run of
the gate and a change: its class (holds, stated limit, open, fails) and the claims
that failed. Also printed for each run: how many runs side by side, the time, how
many pairs needed a second attempt, and how many second calls the judge made.
"""
import json
import sys

docs = {}
for arg in sys.argv[1:]:
    label, _, path = arg.partition("=")
    docs[label] = json.load(open(path, encoding="utf-8"))


def verdicts(doc):
    return {(p["run"], p["change"]): (p["class"], json.dumps(p["fails"], sort_keys=True))
            for p in doc["pairs"]}


first = next(iter(docs))
base = verdicts(docs[first])
for label, doc in docs.items():
    again = sum(1 for p in doc["pairs"] if isinstance(p.get("attempts"), int) and p["attempts"] > 1)
    print(f"{label}: {doc['jobs']} side by side, {doc['wall_seconds']:.0f} s; {len(doc['pairs'])} "
          f"pairs; {doc['counts']}; pairs that needed a second attempt: {again}; second calls "
          f"by the judge: {len(doc['second_calls'])}")
for label, doc in docs.items():
    if label == first:
        continue
    found = verdicts(doc)
    print(f"{label} against {first}: the same pairs: {set(found) == set(base)}; pairs whose "
          f"class differs: {sum(found[p][0] != base[p][0] for p in base if p in found)}; pairs "
          f"whose verdict differs in any claim: {sum(found[p] != base[p] for p in base if p in found)}")
print("the same stated limits in every run:",
      len({json.dumps(d["stated_limits"], sort_keys=True) for d in docs.values()}) == 1)
print("the same second calls in every run:",
      len({json.dumps(sorted(d["second_calls"])) for d in docs.values()}) == 1)
