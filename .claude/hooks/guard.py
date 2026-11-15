"""PreToolUse hook: keep agents out of the judge.

Blocks writes to locked paths for the current stage (see loop/policy.py),
blocks git, and blocks agents from running the loop driver.

Exit 0 = no objection. Exit 2 = block, with the reason on stderr for the model.
This hook fails CLOSED: if it cannot decide, it blocks and says HARNESS ERROR,
because a guard that silently stops guarding is the failure this kit exists
to prevent.

Kit file. Standard library only.
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("CLAUDE_PROJECT_DIR") or Path(__file__).resolve().parents[2])
sys.path.insert(0, str(ROOT / "loop"))

FILE_TOOLS = {"Write", "Edit", "MultiEdit", "NotebookEdit"}
SHELL_TOOLS = {"Bash", "PowerShell"}


def main() -> int:
    try:
        import policy

        payload = json.loads(sys.stdin.buffer.read().decode("utf-8", "replace") or "{}")
        tool = payload.get("tool_name", "")
        tool_input = payload.get("tool_input") or {}
        cwd = payload.get("cwd") or str(ROOT)
        stage = policy.current_stage()

        if tool in FILE_TOOLS:
            path = tool_input.get("file_path") or tool_input.get("notebook_path")
            if not path:
                return 0
            allowed, reason = policy.check_file_write(path, stage, ROOT, cwd)
        elif tool in SHELL_TOOLS:
            allowed, reason = policy.check_shell(tool_input.get("command", ""), stage, ROOT, cwd)
        else:
            return 0
        if allowed:
            return 0
        sys.stderr.write(reason + "\n")
        return 2
    except Exception as exc:  # fail closed, and say whose fault it is
        sys.stderr.write(
            "HARNESS ERROR in the guard hook (not a problem with your code): "
            f"{type(exc).__name__}: {exc}. The guard blocks while it is broken. "
            "Stop and report this to the human.\n"
        )
        return 2


if __name__ == "__main__":
    sys.exit(main())
