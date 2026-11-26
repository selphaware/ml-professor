"""Print the records of one lab file, one block each."""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lab10                                            # noqa: E402

for r in lab10.load(sys.argv[1]):
    label = r.get("run") or r.get("make")
    print(label, r.get("change") or r.get("params"), "k", r["k"], "|", r.get("error"),
          None if r.get("skill") is None else round(r["skill"], 4), r.get("model"), r.get("scheme"))
    if "error" in r:
        print("   ", r["message"][:300])
        continue
    print("   warnings", r["warnings"])
    print("   excluded", {k: v[1:] for k, v in r["roles"].items() if v[0] == "excluded"})
    print("   falls", {k: (None if v[2] is None else round(v[2], 3)) for k, v in r["rests"].items()})
    for code, col, message in r["texts"]:
        if code in ("near_perfect_headline", "large_holdout"):
            print("   >>", message[:500])
    if len(sys.argv) > 2:
        print("   log", {k: v for k, v in r.get("log", {}).items() if k in sys.argv[2:]})
