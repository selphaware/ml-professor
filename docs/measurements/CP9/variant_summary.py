"""What each tool finds on a broken copy of the reference.

    python -B variant_summary.py <folder with battery_variant_<V>.json and generated_variant_<V>.json>
"""
import json
import sys
from collections import Counter
from pathlib import Path

folder = Path(sys.argv[1])
for v in ("F22", "M13", "M33", "M34", "M37", "M39"):
    print(f"== {v}")
    path = folder / f"battery_variant_{v}.json"
    if path.is_file():
        d = json.loads(path.read_text(encoding="utf-8"))
        fails = [p for p in d["pairs"] if p["class"] == "fails"]
        claims = Counter(name.split(":")[0] for p in fails for name, _ in p["fails"])
        runs = Counter(p["run"] for p in fails)
        print(f"  battery: {d['counts']}; {d['wall_seconds']} s; second calls {len(d.get('second_calls', []))}")
        print(f"    failing claims: {dict(claims)}")
        print(f"    by run: {dict(runs)}")
        for p in fails[:2]:
            print(f"    e.g. {p['run']}:{p['change']}: {p['fails'][0][0]}: {p['fails'][0][1][:260]}")
    else:
        print("  battery: no result yet")
    path = folder / f"generated_variant_{v}.json"
    if path.is_file():
        d = json.loads(path.read_text(encoding="utf-8"))
        fields = Counter()
        first = None
        for e in d["tables_with_findings"]:
            if e["class"] != "fails":
                continue
            for f in e["findings"]:
                if f["class"] in ("raw", "outcome", "exact", "fit", "property"):
                    fields[(f["class"], f["field"].split("[")[0][:80])] += 1
                    first = first or (e["number"], f)
        print(f"  generated tables: {d['tables_by_class']}; {d['wall_seconds']} s")
        for key, count in sorted(fields.items(), key=lambda kv: -kv[1])[:8]:
            print(f"    {count:>4} {key}")
        if first:
            n, f = first
            print(f"    e.g. table {n}: [{f['class']}] {f['field']}: {str(f['application'])[:200]!r} | {str(f['reference'])[:200]!r}")
    else:
        print("  generated tables: no result yet")
