"""CP4: does the gate's self-test leave the build's run folder alone? (D46)

    python -B runs_untouched.py snapshot <file.json>
    python -B runs_untouched.py compare  <file.json>

`snapshot` records every file under verify/artifacts/runs: its size, its
modification time and the SHA-256 of its bytes. `compare` records them again
and says what differs. Take the snapshot after an ordinary gate run on the
build, run the self-test, then compare: nothing may differ.

It reads the repo and writes only the file named.
"""
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RUNS = REPO / "verify" / "artifacts" / "runs"


def state():
    files = {}
    for path in sorted(RUNS.rglob("*")):
        if path.is_file():
            stat = path.stat()
            files[path.relative_to(RUNS).as_posix()] = {
                "bytes": stat.st_size, "modified_ns": stat.st_mtime_ns,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return files


def main(command, target):
    now = state()
    if command == "snapshot":
        Path(target).write_text(json.dumps(now, indent=1), encoding="utf-8")
        print(f"{len(now)} files under {RUNS} recorded in {target}")
        return 0
    before = json.loads(Path(target).read_text(encoding="utf-8"))
    gone = sorted(set(before) - set(now))
    new = sorted(set(now) - set(before))
    rewritten = sorted(f for f in before if f in now and before[f] != now[f])
    changed = [f for f in rewritten if before[f]["sha256"] != now[f]["sha256"]]
    print(f"{len(before)} files before, {len(now)} now: {len(gone)} gone, {len(new)} new, "
          f"{len(rewritten)} written again ({len(changed)} with other bytes)")
    for label, names in (("gone", gone), ("new", new), ("written again", rewritten)):
        for name in names[:10]:
            print(f"  {label}: {name}")
    untouched = not (gone or new or rewritten)
    print("UNTOUCHED: every file has the bytes and the modification time it had" if untouched
          else "TOUCHED")
    return 0 if untouched else 1


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("snapshot", "compare"):
        raise SystemExit(__doc__)
    sys.exit(main(sys.argv[1], sys.argv[2]))
