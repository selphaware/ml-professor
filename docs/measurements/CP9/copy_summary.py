import json, sys
from collections import defaultdict
recs = json.load(open(sys.argv[1], encoding="utf-8"))
g = defaultdict(list)
for r in recs:
    if r["change"]:
        g[(r["run"] + (" --no-entities" if r.get("tag") == "no_entities" else ""), r["change"])].append(r)
print("| run | change | copy excluded | copy warned | copy kept in silence | other column excluded | errors | skill |")
for (run, change), rows in g.items():
    x = w = k = other = 0
    errs = defaultdict(int); sk = []
    for r in rows:
        if "error" in r:
            errs[r["error"]] += 1; continue
        role = r["roles"].get("target_copy")
        warned = any(c == "strong_single_column" and col == "target_copy" for c, col in r["warnings"])
        if role and role[1] == "suspected_leak": x += 1
        elif warned: w += 1
        else: k += 1
        gate_leak = {"leak_reg": "amount_scaled"}.get(r["run"])
        other += any(v[1] == "suspected_leak" and n not in ("target_copy", gate_leak) for n, v in r["roles"].items())
        sk.append(r["skill"])
    sk = [s for s in sk if s is not None]
    span = f"{min(sk):+.3f} to {max(sk):+.3f}" if sk else ""
    print(f"| {run} | {change} | {x} | {w} | {k} | {other} | {dict(errs) or ''} | {span} |")
