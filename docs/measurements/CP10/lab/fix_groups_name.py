"""The A block of edit_make_proto5.py used the name `groups`, which `_run` already has
for the entity codes: rename it there."""
import re
from pathlib import Path

p = Path(__file__).resolve().parent / "edit_make_proto5.py"
t = p.read_text(encoding="utf-8")
a = t.index("    rests_on = []\n    if P.A:")
b = t.index("        if P.DEBUG:\n            P.LOG[\"a_seconds\"]")
block = re.sub(r"\bgroups\b", "copy_sets", t[a:b])
p.write_text(t[:a] + block + t[b:], encoding="utf-8", newline="\n")
print("renamed", t[a:b].count("groups"), "uses")
