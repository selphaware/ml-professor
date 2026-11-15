"""PostToolUse hook: run the gate after every file edit (the inner loop).

Build stage only. Runs the checks the current task must satisfy (exported by
the driver in LOOP_ACCEPT), or the whole gate when no driver is in charge.
It runs the gate's fast profile: checks marked slow are skipped here and run
when the builder tries to finish and when the driver judges.

Exit 2 + stderr  = the deliverable fails a check; the model reads the reason
                   and corrects itself before it can claim to be done.
Exit 0           = pass, or a HARNESS ERROR. A broken environment is reported
                   to the human and is never blamed on the code.

Kit file. Standard library only. Runs on the interpreter that launched it,
which settings.json pins by absolute path.
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("CLAUDE_PROJECT_DIR") or Path(__file__).resolve().parents[2])
sys.path.insert(0, str(ROOT / "loop"))


def warn_human(message: str) -> None:
    """Exit-0 path: surface a harness problem to the person, not the model."""
    sys.stdout.write(json.dumps({"systemMessage": message}))


def main() -> int:
    try:
        import checks
        import policy

        sys.stdin.buffer.read()  # payload not needed; drain it
        if policy.current_stage() != "build":
            return 0
        plan = checks.load_plan(ROOT)
        if not plan or not plan.get("deliverable"):
            return 0  # nothing to judge yet
        accept = checks.read_accept()
        only = accept["gate"] if accept is not None else None
        timeout = int(os.environ.get("LOOP_HOOK_GATE_TIMEOUT")
                      or checks.limit(ROOT, "hook_gate_timeout_sec", 150))
        result = checks.run_gate(ROOT, plan["deliverable"], only=only, timeout=timeout,
                                 fast=True)
    except Exception as exc:
        warn_human(f"HARNESS ERROR in gate hook (not a code problem): {type(exc).__name__}: {exc}")
        return 0

    if result.status == checks.PASS:
        return 0
    if result.status == checks.HARNESS:
        warn_human("Gate hook could not judge this edit. " + checks.tail(result.output, 1200))
        return 0
    scope = "this task's gate checks" if only else "the gate"
    sys.stderr.write(
        f"GATE FAILED after your edit ({scope}; failing: {', '.join(result.failing)}).\n"
        "Fix the deliverable, not the gate.\n\n" + result.output + "\n"
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
