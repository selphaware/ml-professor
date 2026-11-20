"""CP4: which gate checks fail on each fixture of a gate, and is every `catches` proved?

    python -B fixtures.py [<gate script>]

Runs the known-good fixture and every broken one of a gate through all its
checks, once each, and prints for every check the failure modes it names and
the fixtures it failed on. A pair of a check and a mode it names that was not
seen failing is marked UNPROVED. `<gate script>` defaults to the repo's
verify/smoke_test.py; give a copy of an older one to see the state before CP4
(its reference implementation is read from the folder beside it).

Since CP4 the gate's own self-test prints this table and fails on an unproved
pair. This script is how the table was measured before that, and how to
measure it for a gate that does not yet.

The gate is loaded as a module with its project root set to a new folder in
the system's temporary folder, so its artifacts go there and nothing is
written inside the repo. About 20 seconds a fixture.
"""
import importlib.util
import sys
import tempfile
import time
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[3]


def main(gate_file):
    root = Path(tempfile.mkdtemp(prefix="mlprof-cp4-fixtures-"))
    (root / "verify").mkdir(parents=True)
    sys.argv = [str(root / "verify" / "smoke_test.py")]   # gatekit takes the root from here
    sys.path.insert(0, str(REPO / "verify"))              # gatekit itself
    spec = importlib.util.spec_from_file_location("gate_under_test", str(gate_file))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    gate = module.gate

    failed = {}
    fixtures = [("good", gate._good)] + [(m, d["build"]) for m, d in gate._modes.items()]
    for name, builder in fixtures:
        started = time.perf_counter()
        results = gate.run(gate._build(builder, root / "fixtures" / name))
        failed[name] = sorted(r["id"] for r in results if not r["ok"])
        print(f"{name}: failed {failed[name] or 'nothing'}  [{time.perf_counter() - started:.0f} s]",
              flush=True)

    unproved = 0
    print("\n| Check | Names | Failed on the fixture of |")
    print("|---|---|---|")
    for entry in gate._checks.values():
        on = [name for name in failed if name != "good" and entry["id"] in failed[name]]
        missing = [mode for mode in entry["catches"] if mode not in on]
        unproved += len(missing)
        print(f"| {entry['id']} | {', '.join(entry['catches']) or '(a reason)'} | "
              f"{', '.join(on) or '-'}"
              + (f" **UNPROVED for {', '.join(missing)}**" if missing else "") + " |")
    print(f"\n{unproved} pair(s) of a check and a mode it names not seen failing; "
          f"{len(failed) - 1} broken fixtures; artifacts under {root}")
    return 1 if unproved or failed["good"] else 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]).resolve() if len(sys.argv) > 1
                  else REPO / "verify" / "smoke_test.py"))
