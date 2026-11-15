#!/usr/bin/env python
"""Loop driver: runs each stage of the build on the model its work calls for.

    python loop/run_loop.py doctor [--live]   preflight; --live makes one tiny real call
    python loop/run_loop.py harness [--open]  interactive: design + gate (fable)
    python loop/run_loop.py tests             unattended: write the test suite (opus)
    python loop/run_loop.py build [--dry-run] unattended: one run per task, at its tier
    python loop/run_loop.py review            unattended, read-only review (fable)
    python loop/run_loop.py go                tests -> build -> review
    python loop/run_loop.py status            what is done, what it cost

Run it with the project's pinned interpreter. That interpreter also runs the
gate and the tests, so there is exactly one Python in play.

The driver is the judge of record. An agent stopping proves nothing; after
every run the driver re-runs the task's acceptance checks itself, checks that
no locked file was touched, and only then moves on. Models come from
loop/loop.json and nowhere else.

Exit codes: 0 done, 1 stage failed, 2 stopped for a human decision,
3 HARNESS ERROR (environment, never the code).

Kit file. Standard library only. Agents never run this; the guard blocks it.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import checks  # noqa: E402
import policy  # noqa: E402

STATE_DIR = HERE / "state"
STATE_FILE = STATE_DIR / "state.json"
EXIT_OK, EXIT_FAILED, EXIT_STOPPED, EXIT_HARNESS = 0, 1, 2, 3
HARNESS_GIT_PATHS = ["verify", "plan", "spec.md", "CLAUDE.md", ".claude", "loop"]
JUDGE_AWARE = re.compile(
    r"PYTEST_CURRENT_TEST|sys\.modules.{0,20}pytest|LOOP_(STAGE|ACCEPT|TASK|DRIVER)|"
    r"smoke_test|gatekit|verify[/\\]|tests[/\\]fixtures", re.I)
CODE_SUFFIXES = {".py", ".js", ".ts", ".jsx", ".tsx", ".html", ".ps1", ".sh", ".sql"}


class HarnessProblem(Exception):
    """Environment or kit problem. Ends the run with exit code 3."""


class Stopped(Exception):
    """A human decision is needed. Ends the run with exit code 2."""


def log(message: str = "") -> None:
    stamp = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{stamp}] {message}" if message else "", flush=True)


# ----------------------------------------------------------- config/state --

def load_config() -> dict:
    try:
        cfg = json.loads((HERE / "loop.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HarnessProblem(f"cannot read loop/loop.json: {exc}")
    override = os.environ.get("LOOP_CLAUDE_CMD")
    if override:
        cfg["claude_cmd"] = json.loads(override)
    return cfg


def claude_cmd(cfg) -> list:
    cmd = cfg.get("claude_cmd")
    if cmd:
        return list(cmd) if isinstance(cmd, list) else [cmd]
    found = shutil.which("claude")
    if not found:
        raise HarnessProblem("'claude' is not on PATH. Install Claude Code, or set "
                             "claude_cmd in loop/loop.json to its full path.")
    return [found]


def load_state() -> dict:
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"tasks": {}, "ratchet": {"gate": [], "tests": []}, "spent_usd": 0.0}


def save_state(state: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(STATE_FILE)


# ------------------------------------------------------------------- plan --

def load_plan(gate_ids=None):
    """Load and validate plan/tasks.json. Returns (plan, errors, warnings)."""
    plan = checks.load_plan(ROOT)
    if plan is None:
        return None, [f"{checks.PLAN_FILE} is missing or is not valid JSON"], []
    cfg_tiers = set(load_config()["tiers"])
    errors, warnings = [], []
    if not plan.get("deliverable"):
        errors.append("plan has no 'deliverable' (the gate's argument)")
    tasks = plan.get("tasks") or []
    if not tasks:
        errors.append("plan has no tasks")
    ids = [t.get("id") for t in tasks]
    for dup in sorted({i for i in ids if ids.count(i) > 1}):
        errors.append(f"duplicate task id {dup}")
    claimed = set()
    for t in tasks:
        tid = t.get("id", "?")
        if t.get("tier") not in cfg_tiers:
            errors.append(f"{tid}: tier '{t.get('tier')}' is not one of {sorted(cfg_tiers)}")
        if not t.get("brief"):
            errors.append(f"{tid}: no brief")
        if not (t.get("gate_checks") or t.get("tests")):
            errors.append(f"{tid}: no acceptance (needs gate_checks and/or tests); "
                          "a task nothing can fail is not verifiable")
        for dep in t.get("depends_on", []):
            if dep not in ids:
                errors.append(f"{tid}: depends on unknown task {dep}")
        for cid in t.get("gate_checks", []):
            claimed.add(cid)
            if gate_ids is not None and cid not in gate_ids:
                errors.append(f"{tid}: gate check '{cid}' does not exist in the gate")
    if gate_ids is not None:
        for cid in sorted(set(gate_ids) - claimed):
            warnings.append(f"gate check '{cid}' belongs to no task (only the final "
                            "full gate will run it)")
    if not errors:
        try:
            topo_order(tasks)
        except ValueError as exc:
            errors.append(str(exc))
    return plan, errors, warnings


def topo_order(tasks) -> list:
    by_id = {t["id"]: t for t in tasks}
    done, order, visiting = set(), [], set()

    def visit(tid):
        if tid in done:
            return
        if tid in visiting:
            raise ValueError(f"dependency cycle through task {tid}")
        visiting.add(tid)
        for dep in by_id[tid].get("depends_on", []):
            visit(dep)
        visiting.discard(tid)
        done.add(tid)
        order.append(by_id[tid])

    for t in tasks:
        visit(t["id"])
    return order


def ladder_from(cfg, tier) -> list:
    ladder = cfg["ladder"]
    return ladder[ladder.index(tier):] if tier in ladder else [tier]


# -------------------------------------------------------- tamper checking --

class Snapshot:
    """Bytes of every locked file, so tampering is detected AND undone."""

    def __init__(self, kinds):
        self.kinds = frozenset(kinds)
        self.files = {rel: path.read_bytes() for rel, path in self._walk()}

    def _walk(self):
        return policy.iter_files(ROOT, lambda rel: policy.classify(rel) in self.kinds)

    def violations(self) -> list:
        now = {rel: path for rel, path in self._walk()}
        out = []
        for rel, data in self.files.items():
            if rel not in now:
                out.append(f"deleted {rel}")
            elif hashlib.sha256(now[rel].read_bytes()).digest() != hashlib.sha256(data).digest():
                out.append(f"modified {rel}")
        out += [f"created {rel}" for rel in now if rel not in self.files]
        return sorted(out)

    def restore(self) -> None:
        now = {rel: path for rel, path in self._walk()}
        for rel, path in now.items():
            if rel not in self.files:
                path.unlink(missing_ok=True)
        for rel, data in self.files.items():
            target = ROOT / rel
            if not target.exists() or target.read_bytes() != data:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)


def tree_signature(kinds) -> dict:
    """Cheap change detector (size + mtime) for files of the given kinds."""
    sig = {}
    for rel, path in policy.iter_files(ROOT, lambda r: policy.classify(r) in kinds):
        st = path.stat()
        sig[rel] = (st.st_size, st.st_mtime_ns)
    return sig


def tree_changes(before: dict, kinds) -> list:
    after = tree_signature(kinds)
    changed = [r for r in after if before.get(r) != after[r]]
    return sorted(changed + [r for r in before if r not in after])


def git_dirty(paths):
    """Uncommitted changes under ``paths``; None if git or a repo is absent.

    Read-only. The driver never writes to git; the human owns that.
    """
    git = shutil.which("git")
    if not git:
        return None
    try:
        inside = subprocess.run([git, "rev-parse", "--is-inside-work-tree"], cwd=ROOT,
                                capture_output=True, text=True, timeout=30)
        if inside.returncode != 0:
            return None
        existing = [p for p in paths if (ROOT / p).exists()]
        out = subprocess.run([git, "status", "--porcelain", "--", *existing], cwd=ROOT,
                             capture_output=True, text=True, timeout=60)
        return [line for line in out.stdout.splitlines() if line.strip()]
    except (OSError, subprocess.SubprocessError):
        return None


def require_committed(args, paths, what) -> None:
    dirty = git_dirty(paths)
    if dirty is None:
        log("note: not a git repository (or git not found); the in-memory snapshot "
            "is the only tamper baseline for this run.")
        return
    if dirty and not args.allow_dirty:
        shown = "\n    ".join(dirty[:15])
        raise Stopped(
            f"{what} has uncommitted changes:\n    {shown}\n"
            "Commit it first, so there is a baseline you reviewed to compare against. "
            "(--allow-dirty overrides this, and you lose that baseline.)")
    if dirty:
        log(f"WARNING: running with uncommitted changes in {what} (--allow-dirty).")


# ------------------------------------------------------------ claude runs --

def run_claude(cfg, role, prompt, stage, label, accept=None, task_id="", resume=None,
               state=None) -> dict:
    """One unattended Claude Code run. Returns a summary dict."""
    cmd = claude_cmd(cfg) + [
        "-p", "--output-format", "json",
        "--model", role["model"], "--effort", role.get("effort", "medium"),
        "--permission-mode", cfg.get("permission_mode", "auto"),
        "--permission-prompts", "none",
    ]
    if role.get("max_turns"):
        cmd += ["--max-turns", str(role["max_turns"])]
    if role.get("max_budget_usd"):
        cmd += ["--max-budget-usd", str(role["max_budget_usd"])]
    if role.get("advisor"):
        cmd += ["--advisor", role["advisor"]]
    if resume:
        cmd += ["--resume", resume]

    env = dict(os.environ)
    env.update({"LOOP_STAGE": stage, "LOOP_DRIVER": "1", "LOOP_TASK": task_id,
                "PYTHONIOENCODING": "utf-8"})
    if accept is not None:
        env["LOOP_ACCEPT"] = json.dumps(accept)
    else:
        env.pop("LOOP_ACCEPT", None)

    timeout = int(cfg["limits"].get("run_timeout_min", 60)) * 60
    started = time.time()
    try:
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env, input=prompt, text=True,
                              encoding="utf-8", errors="replace", capture_output=True,
                              timeout=timeout)
        rc, raw, err = proc.returncode, proc.stdout or "", proc.stderr or ""
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout or ""
        raw = out.decode("utf-8", "replace") if isinstance(out, bytes) else out
        rc, err = None, f"run exceeded {timeout // 60} minutes and was stopped"
    except OSError as exc:
        raise HarnessProblem(f"could not start Claude Code ({cmd[0]}): {exc}")
    duration = round(time.time() - started, 1)

    logs = STATE_DIR / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / f"{label}.json").write_text(raw or "", encoding="utf-8")
    if err.strip():
        (logs / f"{label}.stderr.txt").write_text(err, encoding="utf-8")

    data = checks._last_json_line(raw)
    if data is None:
        try:
            parsed = json.loads(raw)
            data = parsed if isinstance(parsed, dict) else None
        except ValueError:
            data = None
    cost = float((data or {}).get("total_cost_usd") or 0.0)
    turns = (data or {}).get("num_turns")
    subtype = str((data or {}).get("subtype", ""))
    capped = rc is None or "max_turns" in subtype or "budget" in subtype
    # Infrastructure failure: nothing usable came back and it was not a cap.
    infra = (not capped) and (
        data is None or (bool(data.get("is_error")) and (not turns or cost == 0.0)))
    if state is not None:
        state["spent_usd"] = round(state.get("spent_usd", 0.0) + cost, 4)
        save_state(state)
    return {
        "rc": rc, "data": data, "cost": cost, "turns": turns, "duration": duration,
        "session_id": (data or {}).get("session_id"), "capped": capped, "infra": infra,
        "text": str((data or {}).get("result") or ""),
        # When Claude Code reports its own failure, its message is the useful part.
        "error": (str(data.get("result") or "") if data and data.get("is_error") else "")
                 or checks.tail(err or raw, 1500),
    }


def check_budget(cfg, state) -> None:
    cap = cfg["limits"].get("total_budget_usd")
    if cap and state.get("spent_usd", 0.0) >= cap:
        raise Stopped(f"total budget reached: ${state['spent_usd']:.2f} of ${cap} "
                      "(limits.total_budget_usd in loop/loop.json). Costs are Claude "
                      "Code's own estimates.")


def infra_abort(run, label) -> None:
    raise HarnessProblem(
        f"Claude Code run '{label}' returned nothing usable (exit {run['rc']}). This is "
        "an environment problem (sign-in, network, a rejected flag), not a code problem, "
        f"so no attempt was counted.\n{run['error']}\n"
        f"Full output: loop/state/logs/{label}.json. Try: python loop/run_loop.py doctor --live")


def judge(cfg, plan, accept):
    """Run gate + tests for an acceptance set ({'gate': [...]|None, 'tests': [...]|None})."""
    limits = cfg["limits"]
    gate = checks.run_gate(ROOT, plan["deliverable"], only=accept.get("gate"),
                           timeout=limits.get("gate_timeout_sec", 300))
    tests = checks.run_tests(ROOT, accept.get("tests"),
                             timeout=limits.get("tests_timeout_sec", 900))
    for res, name in ((gate, "gate"), (tests, "tests")):
        if res.status == checks.HARNESS:
            raise HarnessProblem(f"the {name} could not run:\n{res.output}")
    return gate, tests


def prompt_file(name: str) -> str:
    return (HERE / "prompts" / name).read_text(encoding="utf-8")


def py() -> str:
    return sys.executable


def test_cmd(targets) -> str:
    return (f"{py()} -m pytest -c tests/pytest.ini --rootdir . --confcutdir tests "
            f"-p no:cacheprovider -q {' '.join(targets)}")


# ------------------------------------------------------------------ build --

def build_prompt(plan, task, tier, role, accept, gate_docs, failure=None) -> str:
    own_gate = task.get("gate_checks", [])
    own_tests = task.get("tests", [])
    kept_gate = [c for c in accept["gate"] if c not in own_gate]
    kept_tests = [t for t in accept["tests"] if t not in own_tests]
    lines = [prompt_file("builder.md"), "\n---\n",
             f"# Task {task['id']}: {task.get('title', '')}",
             f"Tier: {tier} (you are running as `{role['model']}`).",
             f"Project interpreter: `{py()}`", "",
             "## Brief", task["brief"], ""]
    if task.get("writes"):
        lines += ["## Files in scope"] + [f"- `{w}`" for w in task["writes"]] + [""]
    lines += ["## Done means"]
    if own_gate:
        lines += ["Gate checks that must pass:"]
        lines += [f"- `{c}`: {gate_docs.get(c, {}).get('doc', '')}"
                  + (" (slow: not run on each edit, run when you finish)"
                     if gate_docs.get(c, {}).get("slow") else "") for c in own_gate]
        lines += [f"\nRun: `{py()} verify/smoke_test.py {plan['deliverable']} "
                  f"--only {','.join(own_gate)}`"]
    if own_tests:
        lines += ["\nTests that must pass:"] + [f"- `{t}`" for t in own_tests]
        lines += [f"\nRun: `{test_cmd(own_tests)}`"]
    if kept_gate or kept_tests:
        lines += ["\n## Must stay green (from earlier tasks)"]
        if kept_gate:
            lines += ["Gate: " + ", ".join(f"`{c}`" for c in kept_gate)]
        if kept_tests:
            lines += ["Tests: " + ", ".join(f"`{t}`" for t in kept_tests)]
    if failure:
        lines += ["\n## The previous attempt did not pass", failure]
    return "\n".join(lines)


def describe_failure(gate, tests, tamper, run) -> str:
    parts = []
    if tamper:
        parts.append("You changed locked files, which is never allowed. They have been "
                     "restored: " + "; ".join(tamper) + ". Fix the application instead.")
    if run.get("capped"):
        parts.append("The run hit its turn, time or budget cap before finishing.")
    if gate is not None and not gate.ok:
        parts.append(f"Gate checks failing: {', '.join(gate.failing)}\n```\n{gate.output}\n```")
    if tests is not None and not tests.ok:
        parts.append(f"Tests failing: {', '.join(tests.failing[:25])}\n```\n{tests.output}\n```")
    return "\n\n".join(parts)


def run_task(cfg, plan, task, state, gate_docs, snapshot) -> bool:
    """Run one task up its escalation ladder. True if it passed."""
    tid = task["id"]
    record = state["tasks"].setdefault(tid, {"status": "pending", "attempts": []})
    ratchet = state["ratchet"]
    accept = {
        "gate": sorted(set(task.get("gate_checks", [])) | set(ratchet["gate"])),
        "tests": sorted(set(task.get("tests", [])) | set(ratchet["tests"])),
    }
    failure, last_tier = None, None
    for tier in ladder_from(cfg, task["tier"]):
        role = cfg["tiers"][tier]
        session = None
        if last_tier is not None:
            log(f"  {tid}: escalating {last_tier} -> {tier} ({role['model']})")
        last_tier = tier
        for attempt in range(1, int(role.get("attempts", 1)) + 1):
            check_budget(cfg, state)
            n = len(record["attempts"]) + 1
            label = f"{tid}-a{n}-{tier}"
            log(f"  {tid}: attempt {n} on {role['model']} ({tier}, "
                f"{'resume' if session else 'fresh session'})")
            if session:
                prompt = ("The driver re-ran this task's acceptance checks and it is not "
                          "done.\n\n" + (failure or "") + "\n\nFix it and finish the task.")
            else:
                prompt = build_prompt(plan, task, tier, role, accept, gate_docs, failure)
            run = run_claude(cfg, role, prompt, "build", label, accept=accept,
                             task_id=tid, resume=session, state=state)
            if run["infra"]:
                infra_abort(run, label)
            tamper = snapshot.violations()
            if tamper:
                snapshot.restore()
                log(f"  {tid}: TAMPER detected and undone: {'; '.join(tamper)}")
            gate, tests = judge(cfg, plan, accept)
            side = snapshot.violations()  # did running the checks write into locked dirs?
            if side:
                snapshot.restore()
                log(f"  {tid}: running the checks changed locked files: {'; '.join(side)}")
            passed = gate.ok and tests.ok and not tamper and not side
            record["attempts"].append({
                "n": n, "tier": tier, "model": role["model"], "effort": role.get("effort"),
                "cost_usd": run["cost"], "turns": run["turns"], "seconds": run["duration"],
                "capped": run["capped"], "tamper": tamper, "passed": passed,
                "signature": "" if passed else checks.signature(gate, tests),
                "session_id": run["session_id"],
            })
            if passed:
                record.update(status="done", tier_used=tier, model_used=role["model"])
                ratchet["gate"] = accept["gate"]
                ratchet["tests"] = accept["tests"]
                save_state(state)
                log(f"  {tid}: PASS on {role['model']} (attempt {n})")
                return True
            failure = describe_failure(gate, tests, tamper, run)
            if side:
                failure += ("\n\nRunning the gate and tests changed locked files (" +
                            "; ".join(side) + "). Nothing may write into verify/ or tests/ "
                            "at run time. If the application writes next to its inputs, "
                            "make it write elsewhere. If a test or the gate does it, say "
                            "so and stop: that is theirs to fix, not yours.")
            # Resume keeps context for a retry; a capped or tampering run starts clean.
            session = None if (run["capped"] or tamper) else run["session_id"]
            save_state(state)
            log(f"  {tid}: not passing ({checks.signature(gate, tests)})")
    record["status"] = "failed"
    save_state(state)
    escalate(cfg, plan, task, record, failure, state)
    return False


def escalate(cfg, plan, task, record, failure, state) -> None:
    """Three strikes: write the packet, get a diagnosis, stop for the human."""
    tid = task["id"]
    folder = STATE_DIR / "escalations"
    folder.mkdir(parents=True, exist_ok=True)
    sigs = [a["signature"] for a in record["attempts"] if not a["passed"]]
    same = len(set(sigs)) == 1
    packet = [
        f"# Escalation: task {tid} ({task.get('title', '')})", "",
        f"Failed {len(sigs)} attempt(s) across every tier up to the top of the ladder.",
        ("Every attempt failed the SAME checks. That points at the spec, the gate or "
         "the tests rather than the model." if same else
         "Attempts failed in DIFFERENT ways. That points at a task that is too big "
         "or underspecified."),
        "", "| Attempt | Tier | Model | Capped | Failing |", "|---|---|---|---|---|",
    ]
    packet += [f"| {a['n']} | {a['tier']} | {a['model']} | {a['capped']} | "
               f"{a['signature'] or 'passed'} |" for a in record["attempts"]]
    packet += ["", "## Last failure", "", failure or "(none recorded)", "",
               "## Task", "", "```json", json.dumps(task, indent=2), "```"]
    path = folder / f"{tid}.md"
    path.write_text("\n".join(packet), encoding="utf-8")
    if cfg.get("diagnose_on_escalation"):
        log(f"  {tid}: asking {cfg['roles']['diagnose']['model']} for a read-only diagnosis")
        prompt = (prompt_file("diagnose.md") + "\n\n---\n\n" + "\n".join(packet) +
                  f"\n\nProject interpreter: `{py()}`. Gate argument: `{plan['deliverable']}`.")
        guard = Snapshot({"kit", "judge", "tests", "rootlock"})
        run = run_claude(cfg, cfg["roles"]["diagnose"], prompt, "review",
                         f"{tid}-diagnose", task_id=tid, state=state)
        if guard.violations():
            guard.restore()
        with path.open("a", encoding="utf-8") as fh:
            fh.write("\n\n## Diagnosis\n\n" + (run["text"] or "(no diagnosis returned)") + "\n")
    raise Stopped(
        f"task {tid} failed on every tier. Nothing was changed in the spec, plan, gate "
        f"or tests. Read loop/state/escalations/{tid}.md, apply the fix you agree with "
        "(by hand, or in a harness session), commit, then run build again.")


def reverify_done(cfg, plan, state) -> None:
    """Re-check earlier results still hold; reopen any task whose checks now fail."""
    ratchet = state["ratchet"]
    if not (ratchet["gate"] or ratchet["tests"]):
        return
    gate, tests = judge(cfg, plan, {"gate": ratchet["gate"], "tests": ratchet["tests"]})
    if gate.ok and tests.ok:
        return
    broken_gate, broken_tests = set(gate.failing), not tests.ok
    for task in plan["tasks"]:
        rec = state["tasks"].get(task["id"])
        if rec and rec.get("status") == "done":
            if broken_gate & set(task.get("gate_checks", [])) or (broken_tests and task.get("tests")):
                rec["status"] = "pending"
                log(f"  reopening {task['id']}: its checks no longer pass")
    ratchet["gate"] = [c for c in ratchet["gate"] if c not in broken_gate]
    if broken_tests:
        ratchet["tests"] = []
    save_state(state)


def has_test_suite() -> bool:
    tests = ROOT / "tests"
    return any(tests.glob("**/test_*.py")) or any(tests.glob("**/*_test.py"))


def judge_awareness() -> list:
    hits = []
    for rel, path in policy.iter_files(ROOT, lambda r: policy.classify(r) == "app"):
        if path.suffix.lower() not in CODE_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for number, line in enumerate(text.splitlines(), 1):
            if JUDGE_AWARE.search(line):
                hits.append(f"{rel}:{number}: {line.strip()[:160]}")
    return hits


def write_build_report(plan, state, final_ok, warnings) -> Path:
    judged_by = "full gate + full suite" if state.get("has_test_suite", True) else \
        "full gate ONLY: this project has no test suite"
    rows = ["# Build report", "",
            f"Result: {'PASS' if final_ok else 'NOT PASSING'} ({judged_by})",
            f"Estimated spend so far: ${state.get('spent_usd', 0.0):.2f}", "",
            "| Task | Planned tier | Passed on | Attempts | Est. cost | Minutes |",
            "|---|---|---|---|---|---|"]
    names = [t["id"] for t in plan["tasks"]] + [k for k in state["tasks"] if k == "INTEGRATION"]
    tiers = {t["id"]: t["tier"] for t in plan["tasks"]}
    for tid in names:
        rec = state["tasks"].get(tid)
        if not rec:
            continue
        atts = rec["attempts"]
        rows.append(f"| {tid} | {tiers.get(tid, '-')} | {rec.get('model_used', '-')} | "
                    f"{len(atts)} | ${sum(a['cost_usd'] for a in atts):.2f} | "
                    f"{sum(a['seconds'] for a in atts) / 60:.1f} |")
    escalated = [tid for tid in names if state["tasks"].get(tid, {}).get("tier_used")
                 and state["tasks"][tid]["tier_used"] != tiers.get(tid, state["tasks"][tid]["tier_used"])]
    if escalated:
        rows += ["", "Escalated above their planned tier: " + ", ".join(escalated) +
                 ". If a tier keeps under-delivering, plan those kinds of task one tier up."]
    if warnings:
        rows += ["", "## Judge-awareness warnings (for the reviewer and for you)", "",
                 "Application code that mentions the test or gate machinery. Each one "
                 "needs a reason to exist:", ""] + [f"- `{w}`" for w in warnings]
    path = STATE_DIR / "build-report.md"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def preflight(args, need_tests=False):
    cfg = load_config()
    gate_ids, err = checks.gate_list(ROOT)
    if gate_ids is None:
        raise HarnessProblem(f"the gate cannot list its checks (run the harness stage "
                             f"first?):\n{err}")
    plan, errors, warnings = load_plan(gate_ids)
    for w in warnings:
        log(f"plan warning: {w}")
    if errors:
        raise HarnessProblem("the plan is not valid:\n  - " + "\n  - ".join(errors))
    if need_tests:
        missing = sorted({t.split("::")[0] for task in plan["tasks"]
                          for t in task.get("tests", [])
                          if not (ROOT / t.split("::")[0]).exists()})
        if missing:
            raise Stopped("these planned test files do not exist yet: " + ", ".join(missing)
                          + ". Run: python loop/run_loop.py tests")
    return cfg, plan, gate_ids


def cmd_build(args) -> int:
    cfg, plan, gate_docs = preflight(args, need_tests=not args.dry_run)
    order = topo_order(plan["tasks"])
    if args.dry_run:
        log("Dry run. Nothing will be executed.")
        for task in order:
            rungs = " -> ".join(f"{cfg['tiers'][t]['model']} x{cfg['tiers'][t].get('attempts', 1)}"
                                for t in ladder_from(cfg, task["tier"]))
            log(f"  {task['id']:<6} {task['tier']:<8} {rungs:<34} "
                f"gate={len(task.get('gate_checks', []))} tests={len(task.get('tests', []))}  "
                f"{task.get('title', '')}")
        log(f"Then: full gate ({len(gate_docs)} checks) + full suite, on failure an "
            f"integration task at tier '{cfg['integration']['tier']}'.")
        return EXIT_OK
    require_committed(args, HARNESS_GIT_PATHS + ["tests"], "the harness (gate, plan, spec, tests, kit)")
    state = load_state()
    if args.restart:
        state = {"tasks": {}, "ratchet": {"gate": [], "tests": []},
                 "spent_usd": state.get("spent_usd", 0.0)}
    snapshot = Snapshot({"kit", "judge", "tests", "rootlock"})
    log(f"Build: {len(order)} task(s); {len(snapshot.files)} locked file(s) under watch.")
    try:
        return _build(cfg, plan, order, state, gate_docs, snapshot)
    except (Stopped, HarnessProblem):
        state["build_passed"] = False
        save_state(state)
        write_build_report(plan, state, False, judge_awareness())
        raise


def _build(cfg, plan, order, state, gate_docs, snapshot) -> int:
    reverify_done(cfg, plan, state)
    for task in order:
        if state["tasks"].get(task["id"], {}).get("status") == "done":
            log(f"  {task['id']}: already done")
            continue
        log(f"Task {task['id']} [{task['tier']}]: {task.get('title', '')}")
        run_task(cfg, plan, task, state, gate_docs, snapshot)

    # A project may be judged by its gate alone. Say so loudly rather than fail on
    # an empty tests/ folder, and never treat "no suite" as "suite passed" silently.
    suite = None if has_test_suite() else []
    if suite is None:
        log("All tasks passed. Running the full gate and the full suite.")
    else:
        log("All tasks passed. Running the full gate. NOTE: there is no test suite; "
            "the gate is the only judge of this build.")
    gate, tests = judge(cfg, plan, {"gate": None, "tests": suite})
    if not (gate.ok and tests.ok):
        log("Full verification is not passing; running an integration task.")
        integration = {
            "id": "INTEGRATION", "title": "Make the full gate and the full suite pass",
            "tier": cfg["integration"]["tier"], "writes": [],
            "brief": ("Every planned task passed its own checks, but the complete gate "
                      "and test suite do not pass together. Find out why and fix the "
                      "application so everything passes at once.\n\n" +
                      describe_failure(gate, tests, [], {})),
            "gate_checks": sorted(gate_docs), "tests": ["tests"] if suite is None else [],
        }
        state["tasks"].pop("INTEGRATION", None)
        run_task(cfg, plan, integration, state, gate_docs, snapshot)
        gate, tests = judge(cfg, plan, {"gate": None, "tests": suite})
    final_ok = gate.ok and tests.ok
    state["build_passed"] = final_ok
    save_state(state)
    state["has_test_suite"] = suite is None
    save_state(state)
    report = write_build_report(plan, state, final_ok, judge_awareness())
    log(f"Build {'PASSED' if final_ok else 'did NOT pass'}. Report: {report.relative_to(ROOT)}")
    return EXIT_OK if final_ok else EXIT_FAILED


# ------------------------------------------------------------------ tests --

def cmd_tests(args) -> int:
    cfg, plan, _ = preflight(args)
    require_committed(args, HARNESS_GIT_PATHS, "the harness (gate, plan, spec, kit)")
    role = cfg["roles"]["tests"]
    state = load_state()
    planned = sorted({t.split("::")[0] for task in plan["tasks"] for t in task.get("tests", [])})
    if not planned:
        log("The plan names no test files; nothing to write.")
        return EXIT_OK
    snapshot = Snapshot({"kit", "judge", "rootlock"})
    app_before = tree_signature({"app"})
    # Once part of the application exists, new tests may legitimately pass at once.
    incremental = any(r.get("status") == "done" for r in state["tasks"].values())
    feedback, session = None, None
    for attempt in range(1, int(role.get("attempts", 2)) + 1):
        check_budget(cfg, state)
        log(f"Tests: attempt {attempt} on {role['model']}")
        if session:
            prompt = "The driver checked your test suite and it is not acceptable yet.\n\n" + feedback
        else:
            prompt = (prompt_file("test-author.md") + "\n\n---\n\n"
                      f"Project interpreter: `{py()}`\n\nTest files the plan requires:\n" +
                      "\n".join(f"- `{p}`" for p in planned) +
                      f"\n\nRun the suite with: `{test_cmd(['tests'])}`" +
                      (f"\n\n## Previous attempt\n{feedback}" if feedback else ""))
        run = run_claude(cfg, role, prompt, "tests", f"tests-a{attempt}", resume=session,
                         state=state)
        if run["infra"]:
            infra_abort(run, f"tests-a{attempt}")
        session = None if run["capped"] else run["session_id"]
        problems = []
        tamper = snapshot.violations()
        if tamper:
            snapshot.restore()
            problems.append("You changed locked files (restored): " + "; ".join(tamper))
        strays = tree_changes(app_before, {"app"})
        if strays:
            problems.append("You wrote outside tests/, which this stage never does. Remove: "
                            + ", ".join(strays[:20]))
        missing = [p for p in planned if not (ROOT / p).is_file()]
        if missing:
            problems.append("Planned test files missing: " + ", ".join(missing))
        empty = [p for p in planned if (ROOT / p).is_file() and not re.search(
            r"^\s*(async\s+)?def\s+test_", (ROOT / p).read_text(encoding="utf-8", errors="replace"), re.M)]
        if empty:
            problems.append("Test files with no test functions: " + ", ".join(empty))
        result = checks.run_tests(ROOT, timeout=cfg["limits"].get("tests_timeout_sec", 900))
        if result.status == checks.HARNESS:
            raise HarnessProblem(f"the test suite could not run:\n{result.output}")
        if result.skipped:
            problems.append("Skipped or xfailed tests are not allowed: " + ", ".join(result.skipped[:20]))
        if result.ok and not incremental:
            problems.append("The whole suite PASSES before the application exists, so it "
                            "proves nothing. Every test that depends on the application must fail now.")
        if not problems:
            vacuous = result.passed
            lines = ["# Tests stage report", "",
                     f"Test files: {len(planned)}. Tests failing before the build (expected): "
                     f"{len(result.failing)}. Passing before the build: {len(vacuous)}.", ""]
            if vacuous:
                lines += ["## Check these by eye", "",
                          "These tests ALREADY PASS with no application. That is fine for a "
                          "test of fixtures or source data, and a defect for anything else:", ""]
                lines += [f"- `{v}`" for v in vacuous]
            (STATE_DIR / "tests-report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
            state["tests_written"] = True
            save_state(state)
            log(f"Tests written: {len(result.failing)} red, {len(vacuous)} already green. "
                "Report: loop/state/tests-report.md")
            if vacuous:
                log("  Review the already-green tests before you commit tests/.")
            return EXIT_OK
        feedback = "\n".join(f"- {p}" for p in problems)
        log("  not acceptable yet:\n" + feedback)
    raise Stopped("the test author did not produce an acceptable suite. See the log "
                  "above and loop/state/logs/. Nothing outside tests/ was kept changed.")


# ----------------------------------------------------------------- review --

def cmd_review(args) -> int:
    cfg, plan, _ = preflight(args)
    state = load_state()
    if not state.get("build_passed"):
        log("WARNING: the last build did not record a full pass; reviewing anyway.")
    role = cfg["roles"]["review"]
    report = STATE_DIR / "build-report.md"
    matrix = subprocess.run([py(), checks.GATE_SCRIPT, "--matrix"], cwd=ROOT,
                            capture_output=True, text=True).stdout
    prompt = (prompt_file("reviewer.md") + "\n\n---\n\n"
              f"Project interpreter: `{py()}`. Gate: `{py()} verify/smoke_test.py "
              f"{plan['deliverable']}`. Tests: `{test_cmd(['tests'])}`.\n\n"
              "## Failure-mode matrix\n\n" + matrix + "\n\n## Build report\n\n" +
              (report.read_text(encoding="utf-8") if report.exists() else "(none)"))
    snapshot = Snapshot({"kit", "judge", "tests", "rootlock"})
    app_before = tree_signature({"app"})
    check_budget(cfg, state)
    log(f"Review on {role['model']} (read-only)")
    run = run_claude(cfg, role, prompt, "review", "review", state=state)
    if run["infra"]:
        infra_abort(run, "review")
    touched = snapshot.violations()
    if touched:
        snapshot.restore()
    touched += [f"modified {r}" for r in tree_changes(app_before, {"app"})]
    text = run["text"] or "(the reviewer returned no report)"
    if touched:
        text = ("**WARNING: the reviewer changed files, which it must never do. Locked "
                "files were restored; check the rest:** " + "; ".join(touched) + "\n\n" + text)
    out = STATE_DIR / "review.md"
    out.write_text(text + "\n", encoding="utf-8")
    verdict = re.search(r"VERDICT:\s*([A-Z\- ]+)", text)
    log(f"Review written to {out.relative_to(ROOT)}. "
        f"Verdict: {verdict.group(1).strip() if verdict else 'not stated'}")
    return EXIT_OK if verdict and verdict.group(1).strip() == "SHIP" and not touched else EXIT_STOPPED


# -------------------------------------------------------- harness / misc --

def cmd_harness(args) -> int:
    cfg = load_config()
    if not (ROOT / "BOOTSTRAP.md").is_file():
        raise Stopped("BOOTSTRAP.md is not in the project root. Write the brief first.")
    role = cfg["roles"]["harness"]
    cmd = claude_cmd(cfg) + ["--model", role["model"], "--effort", role["effort"]]
    if not args.open:
        cmd.append(f"python is {py()} . Read BOOTSTRAP.md and follow it.")
    env = dict(os.environ, LOOP_STAGE="harness", LOOP_DRIVER="1")
    log(f"Starting an interactive harness session on {role['model']} ({role['effort']})"
        + (" with the gate, spec and plan unlocked." if args.open else "."))
    rc = subprocess.call(cmd, cwd=str(ROOT), env=env)
    log("Harness session ended. Next: python loop/run_loop.py doctor, then the "
        "live-fire test in SETUP-GUIDE.md, then commit.")
    return EXIT_OK if rc == 0 else EXIT_FAILED


def cmd_go(args) -> int:
    if args.dry_run:
        return cmd_build(args)
    state = load_state()
    if not state.get("tests_written"):
        rc = cmd_tests(args)
        if rc != EXIT_OK:
            return rc
        # tests/ is new and uncommitted by design in a single unattended run
        args.allow_dirty = True
        log("note: 'go' continues without a commit of tests/. The in-memory snapshot "
            "still locks them for the build.")
    rc = cmd_build(args)
    if rc != EXIT_OK:
        return rc
    return cmd_review(args)


def cmd_status(args) -> int:
    state = load_state()
    plan = checks.load_plan(ROOT) or {"tasks": []}
    log(f"Estimated spend: ${state.get('spent_usd', 0.0):.2f}   "
        f"tests written: {bool(state.get('tests_written'))}   "
        f"build passed: {bool(state.get('build_passed'))}")
    for task in plan["tasks"]:
        rec = state["tasks"].get(task["id"], {})
        log(f"  {task['id']:<6} {task['tier']:<8} {rec.get('status', 'pending'):<8} "
            f"attempts={len(rec.get('attempts', []))} on={rec.get('model_used', '-')}  "
            f"{task.get('title', '')}")
    return EXIT_OK


def version_tuple(text: str):
    m = re.search(r"(\d+)\.(\d+)\.(\d+)", text or "")
    return tuple(int(x) for x in m.groups()) if m else None


def cmd_doctor(args) -> int:
    results = []

    def note(level, message):
        results.append(level)
        print(f"  {level:<4} {message}")

    print("Loop doctor")
    exe = sys.executable
    if "windowsapps" in exe.lower():
        note("FAIL", f"interpreter is the Microsoft Store stub: {exe}")
    elif sys.version_info < (3, 10):
        note("FAIL", f"Python 3.10+ needed, this is {sys.version.split()[0]}")
    else:
        note("ok", f"interpreter {exe} ({sys.version.split()[0]})")
    if str(ROOT).startswith("\\\\"):
        note("WARN", f"project is on a UNC path ({ROOT}); move it to a local disk")

    for rel in (".claude/hooks/guard.py", ".claude/hooks/gate_hook.py",
                ".claude/hooks/stop_gate.py", "verify/gatekit.py", "tests/pytest.ini",
                "loop/policy.py", "loop/checks.py", "loop/loop.json"):
        if not (ROOT / rel).is_file():
            note("FAIL", f"kit file missing: {rel}")
    try:
        settings = json.loads((ROOT / ".claude/settings.json").read_text(encoding="utf-8"))
        hooks = settings.get("hooks", {})
        for event in ("PreToolUse", "PostToolUse", "Stop"):
            cmds = [h.get("command", "") for group in hooks.get(event, [])
                    for h in group.get("hooks", [])]
            if not cmds:
                note("FAIL", f".claude/settings.json registers no {event} hook")
            for c in cmds:
                if not Path(c).is_absolute() or not Path(c).exists():
                    note("FAIL", f"{event} hook interpreter is not an existing absolute path: {c}")
                elif os.path.normcase(os.path.realpath(c)) != os.path.normcase(os.path.realpath(exe)):
                    note("WARN", f"{event} hook uses {c}, but you ran doctor with {exe}")
        if not any(r in ("FAIL",) for r in results):
            note("ok", "hooks registered with an absolute interpreter path")
    except (OSError, ValueError) as exc:
        note("FAIL", f".claude/settings.json unreadable: {exc}")

    try:
        cfg = load_config()
        out = subprocess.run(claude_cmd(cfg) + ["--version"], capture_output=True,
                             text=True, timeout=60).stdout
        have, need = version_tuple(out), version_tuple(cfg.get("min_claude_version", "0.0.0"))
        if have and need and have < need:
            note("FAIL", f"Claude Code {out.strip()} is older than {cfg['min_claude_version']}; run: claude update")
        else:
            note("ok", f"Claude Code {out.strip()}")
        for name, role in list(cfg["roles"].items()) + list(cfg["tiers"].items()):
            caps = f"turns<={role.get('max_turns', '-')}, ${role.get('max_budget_usd', '-')}/run"
            adv = f", advisor {role['advisor']}" if role.get("advisor") else ""
            print(f"       {name:<9} -> {role['model']:<7} effort={role.get('effort', '-'):<6} {caps}{adv}")
        print(f"       total budget ${cfg['limits'].get('total_budget_usd', '-')}, "
              f"permission mode '{cfg.get('permission_mode')}'")
    except (HarnessProblem, OSError, subprocess.SubprocessError, ValueError) as exc:
        cfg = None
        note("FAIL", str(exc))

    probe = subprocess.run([exe, "-c", "import pytest"], capture_output=True, text=True)
    note("ok" if probe.returncode == 0 else "FAIL",
         "pytest importable" if probe.returncode == 0 else
         f"pytest is not installed in {exe} (a gate dependency: install it yourself)")

    if not (ROOT / checks.GATE_SCRIPT).is_file():
        note("info", "no gate yet (verify/smoke_test.py): run the harness stage")
    else:
        gate_ids, err = checks.gate_list(ROOT)
        if gate_ids is None:
            note("FAIL", f"gate --list failed: {err.strip()[:300]}")
        else:
            note("ok", f"gate lists {len(gate_ids)} check(s)")
            st = subprocess.run([exe, checks.GATE_SCRIPT, "--selftest"], cwd=ROOT,
                                capture_output=True, text=True)
            lines = st.stdout.strip().splitlines() or ["(no output)"]
            note("ok" if st.returncode == 0 else "FAIL", f"gate self-test: {lines[-1]}")
            for line in lines:
                if line.startswith("SELFTEST DEFECT") and "NOT DETERMINISTIC" in line:
                    note("FAIL", line[len("SELFTEST DEFECT: "):])
            timing = re.search(r"TIMING: full gate ([\d.]+)s, per-edit \(--fast\) ([\d.]+)s",
                               st.stdout)
            if timing and cfg:
                full, fast = float(timing.group(1)), float(timing.group(2))
                budget = cfg["limits"].get("hook_gate_budget_sec", 20)
                if fast > budget:
                    note("WARN", f"the gate takes {fast:.0f}s on every edit (budget {budget}s). "
                                 "Mark heavy checks slow=True or shrink the fast profile")
                else:
                    note("ok", f"gate timing: {fast:.1f}s per edit, {full:.1f}s in full")
                if full > cfg["limits"].get("gate_timeout_sec", 300):
                    note("FAIL", f"the full gate takes {full:.0f}s, over limits.gate_timeout_sec")
            plan, errors, warnings = load_plan(gate_ids)
            for e in errors:
                note("FAIL", f"plan: {e}")
            for w in warnings:
                note("WARN", f"plan: {w}")
            if plan and not errors:
                tiers = [t["tier"] for t in plan["tasks"]]
                note("ok", f"plan valid: {len(tiers)} task(s) (" + ", ".join(
                    f"{tiers.count(t)} {t}" for t in sorted(set(tiers))) + ")")

    dirty = git_dirty(HARNESS_GIT_PATHS + ["tests"])
    if dirty is None:
        note("WARN", "not a git repository, or git not found: no committed baseline")
    elif dirty:
        note("WARN", f"{len(dirty)} uncommitted change(s) in the harness; commit before an unattended run")
    else:
        note("ok", "harness is committed")

    if args.live and cfg:
        print("  Live check: one tiny real call on the support tier...")
        role = dict(cfg["tiers"][cfg["ladder"][0]], max_turns=2, max_budget_usd=1)
        try:
            role["max_turns"] = 4
            run = run_claude(cfg, role, f"Run this command: {exe} --version\nThen reply with "
                             "exactly LOOP-OK followed by the command's output, nothing else.",
                             "review", "doctor-live")
            if "LOOP-OK" in run["text"] and "python" in run["text"].lower():
                note("ok", f"live call worked on {role['model']} (est. ${run['cost']:.4f}): "
                           "the driver's flags were accepted and a command ran unattended")
            elif "LOOP-OK" in run["text"]:
                note("FAIL", "the model answered but could not run a command. Permission mode "
                             f"'{cfg.get('permission_mode')}' may not be available on your "
                             "account; see MODEL-ROUTING.md, 'Permission mode'")
            else:
                note("FAIL", f"live call failed. Claude Code said: {run['error'][:400]}")
                if re.search(r"authenticat|log ?in|oauth|api key", run["error"], re.I):
                    note("info", "sign in again: run `claude auth login` (or `claude`, then "
                                 "/login), then repeat this check")
        except HarnessProblem as exc:
            note("FAIL", str(exc))

    bad = results.count("FAIL")
    print(f"Doctor: {'OK' if not bad else str(bad) + ' problem(s)'}"
          + (f", {results.count('WARN')} warning(s)" if "WARN" in results else ""))
    return EXIT_HARNESS if bad else EXIT_OK


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    parser = argparse.ArgumentParser(description="Loop driver. See SETUP-GUIDE.md.")
    sub = parser.add_subparsers(dest="command", required=True)
    commands = {"doctor": cmd_doctor, "harness": cmd_harness, "tests": cmd_tests,
                "build": cmd_build, "review": cmd_review, "go": cmd_go, "status": cmd_status}
    for name in commands:
        p = sub.add_parser(name)
        p.add_argument("--allow-dirty", action="store_true",
                       help="run even if the harness has uncommitted changes")
        if name == "doctor":
            p.add_argument("--live", action="store_true", help="make one tiny real call")
        if name == "harness":
            p.add_argument("--open", action="store_true",
                           help="open a harness-stage session without the BOOTSTRAP kickoff "
                                "(to change the gate, spec or plan after a review or escalation)")
        if name in ("build", "go"):
            p.add_argument("--dry-run", action="store_true", help="show routing, run nothing")
            p.add_argument("--restart", action="store_true", help="forget earlier task results")
    args = parser.parse_args(argv)
    for flag in ("dry_run", "restart", "live", "open"):
        if not hasattr(args, flag):
            setattr(args, flag, False)
    try:
        return commands[args.command](args)
    except HarnessProblem as exc:
        log(f"HARNESS ERROR - this is not a code problem:\n{exc}")
        return EXIT_HARNESS
    except Stopped as exc:
        log(f"STOPPED - needs you:\n{exc}")
        return EXIT_STOPPED
    except KeyboardInterrupt:
        log("Interrupted. State is saved; run the same command to continue.")
        return EXIT_STOPPED


if __name__ == "__main__":
    sys.exit(main())
