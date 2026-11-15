"""Stop hook: an agent may not finish while its acceptance checks fail.

Active only when the session was started for the build stage (LOOP_STAGE=build,
which the driver sets). In a plain interactive session it does nothing, so
asking a question never triggers a forced fix-up loop. Runs the gate checks AND
the tests the current task must satisfy (LOOP_ACCEPT from the driver), or the
full gate and suite if none were given. This also covers changes made through
shell commands, which the PostToolUse hook never sees.

Exit 2 + stderr = keep working; the reason becomes the next instruction.
Exit 0          = allowed to stop.

It blocks at most LOOP_STOP_MAX times per session (default 3), then lets the
session end. That is deliberate: the driver re-judges every run itself and
owns retries and escalation, so this hook never needs to loop forever.

Kit file. Standard library only.
"""
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(os.environ.get("CLAUDE_PROJECT_DIR") or Path(__file__).resolve().parents[2])
sys.path.insert(0, str(ROOT / "loop"))


def _bump(session_id: str) -> int:
    """Count blocks for this session. Returns the count before this one."""
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", session_id or "unknown")[:80]
    folder = ROOT / "loop" / "state" / "stop"
    folder.mkdir(parents=True, exist_ok=True)
    counter = folder / f"{safe}.count"
    try:
        seen = int(counter.read_text(encoding="utf-8").strip() or 0)
    except (OSError, ValueError):
        seen = 0
    counter.write_text(str(seen + 1), encoding="utf-8")
    return seen


def main() -> int:
    try:
        import checks

        payload = json.loads(sys.stdin.buffer.read().decode("utf-8", "replace") or "{}")
        if os.environ.get("LOOP_STAGE", "").strip().lower() != "build":
            return 0  # not a build run: never force a loop
        plan = checks.load_plan(ROOT)
        if not plan or not plan.get("deliverable"):
            return 0
        accept = checks.read_accept()
        gate = checks.run_gate(ROOT, plan["deliverable"],
                               only=accept["gate"] if accept is not None else None)
        if accept is not None:
            tests = checks.run_tests(ROOT, accept["tests"])
        elif any((ROOT / "tests").glob("**/test_*.py")):
            tests = checks.run_tests(ROOT)
        else:
            tests = checks.Result(checks.PASS)
    except Exception as exc:
        sys.stdout.write(json.dumps({"systemMessage":
            f"HARNESS ERROR in stop hook (not a code problem): {type(exc).__name__}: {exc}"}))
        return 0

    if checks.HARNESS in (gate.status, tests.status):
        bad = gate if gate.status == checks.HARNESS else tests
        sys.stdout.write(json.dumps({"systemMessage":
            "Stop hook could not judge the work. " + checks.tail(bad.output, 1200)}))
        return 0
    if gate.ok and tests.ok:
        return 0

    limit = int(os.environ.get("LOOP_STOP_MAX", "3"))
    if _bump(payload.get("session_id", "")) >= limit:
        return 0  # the driver takes it from here

    parts = ["NOT DONE: the acceptance checks for this work still fail."]
    if not gate.ok:
        parts.append(f"Gate checks failing: {', '.join(gate.failing)}\n{gate.output}")
    if not tests.ok:
        parts.append(f"Tests failing: {', '.join(tests.failing[:20])}\n{tests.output}")
    parts.append("Fix the deliverable. The gate and the tests are locked. If one of them "
                 "is wrong, say so plainly with the evidence, then stop.")
    sys.stderr.write("\n\n".join(parts) + "\n")
    return 2


if __name__ == "__main__":
    sys.exit(main())
