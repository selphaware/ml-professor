"""CP10, proof 5: for each broken copy, the number of failing cases of the suite as it
stands (before) and with the prototypes of the ninth pass laid over it (after), from the
two printouts of verify/mutation_check.py.

    python -B mutation_table10.py <printout without overlay> <printout with overlay>
"""
import re
import sys

LINE = re.compile(r"^(good|[FM][0-9]+[a-z]?) +fails ([0-9]+)")


def counts(path):
    out = {}
    for line in open(path, encoding="utf-8"):
        m = LINE.match(line)
        if m:
            out[m.group(1)] = int(m.group(2))
    return out


before, after = counts(sys.argv[1]), counts(sys.argv[2])
print("| case | failing cases before | after | change |")
print("|---|---|---|---|")
for case in before:
    b, a = before[case], after.get(case)
    change = "-" if a is None else f"{a - b:+d}"
    print(f"| {case} | {b} | {a} | {change} |")
lost = [c for c in before if before[c] and after.get(c) is not None and after[c] < before[c]]
new_caught = [c for c in before if before[c] == 0 and (after.get(c) or 0) > 0]
print(f"\ncases caught before: {sum(1 for c in before if before[c] and c != 'good')}; "
      f"of them with fewer failing cases after: {lost or 'none'}")
print(f"caught only with the prototypes: {len(new_caught)}: {', '.join(new_caught)}")
