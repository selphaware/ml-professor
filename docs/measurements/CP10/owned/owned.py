"""CP10: the failures the ninth pass owns (D78 point 8), from the `--json` reports of the
judge's two tools run on the application as built by the eighth pass.

    python -B owned.py <battery.json> <generated_tables.json> <out owned_by_pass9.json>
                       [--literal <out .py>]   [--check <cp10_helpers.py>]

Writes the two lists (`battery`: "run:change" names of the pairs that fail; `tables`: the
numbers of the tables that fail), each sorted, and, with --literal, the two frozensets
as `tests/cp10_helpers.py` is to hold them. With --check it reads the two frozensets of
a helper file and says whether they are the same lists.
"""
import ast
import json
import sys
from pathlib import Path

battery = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
tables = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
pairs = sorted(f"{p['run']}:{p['change']}" for p in battery["pairs"] if p["class"] == "fails")
numbers = sorted(int(e["number"]) for e in tables["tables_with_findings"] if e["class"] == "fails")
assert len(pairs) == len(set(pairs)) == battery["counts"]["fails"]
assert len(numbers) == tables["tables_by_class"]["fails"]
Path(sys.argv[3]).write_text(json.dumps({"battery": pairs, "tables": numbers}, indent=0) + "\n",
                             encoding="utf-8")
print(f"{len(pairs)} pairs and {len(numbers)} tables")
if "--literal" in sys.argv:
    out = ["BATTERY_OWNED = frozenset({"]
    line = "   "
    for p in pairs:
        item = f" {p!r},"
        if len(line) + len(item) > 92:
            out.append(line)
            line = "   "
        line += item
    out += [line, "})", "TABLES_OWNED = frozenset({"]
    line = "   "
    for n in numbers:
        item = f" {n},"
        if len(line) + len(item) > 92:
            out.append(line)
            line = "   "
        line += item
    out += [line, "})"]
    Path(sys.argv[sys.argv.index("--literal") + 1]).write_text("\n".join(out) + "\n",
                                                               encoding="utf-8")
if "--check" in sys.argv:
    tree = ast.parse(Path(sys.argv[sys.argv.index("--check") + 1]).read_text(encoding="utf-8"))
    got = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and node.targets[0].id in ("BATTERY_OWNED", "TABLES_OWNED"):
            got[node.targets[0].id] = ast.literal_eval(node.value.args[0])
    print("the helper's lists are the same:",
          got.get("BATTERY_OWNED") == set(pairs) and got.get("TABLES_OWNED") == set(numbers))
