"""gatekit: the fixed half of every project gate.

The project gate (verify/smoke_test.py) declares WHAT to check. gatekit owns
HOW a gate behaves: the command line, exit codes, output format, and the
self-test that proves the gate can tell good from broken. Keeping that half
identical on every project is what lets one driver and one set of hooks work
everywhere.

Kit file. Do not edit per project. Standard library only.

    from gatekit import Gate

    gate = Gate("sales-pipeline")

    @gate.setup
    def start(ctx): ...            # e.g. start the server; ctx.state is yours
    @gate.teardown
    def stop(ctx): ...             # always runs

    @gate.check("api.rows", catches=["F1"], doc="GET /api/deals returns >= 1 row")
    def api_rows(ctx):
        ctx.require(len(rows) >= 1, "0 rows from /api/deals: loader produced nothing")

    @gate.check("model.lift", catches=["F2"], slow=True, doc="lift on planted signal")
    def model_lift(ctx):           # slow=True: skipped on each edit, run at stop
        ctx.require_between(lift, 0.25, 0.45, "lift on the planted-signal fixture")

    @gate.good_fixture
    def good(tmp): ...             # build a known-good deliverable, return its path
    @gate.failure_mode("F1", "API returns [] after a silent parse failure")
    def broken_empty(tmp): ...     # build a deliverable broken in exactly this way

    if __name__ == "__main__":
        gate.main()

Command line
    smoke_test.py <deliverable> [--only id,id] [--fast] [--json]
    smoke_test.py --list [--json]
    smoke_test.py --matrix
    smoke_test.py --selftest

Slow checks and the fast profile
    A check marked slow=True is skipped when the gate runs with --fast, which is
    how the per-edit hook calls it. The Stop hook and the driver always run
    everything. Setup can read ctx.fast to skip heavy preparation too.

Determinism
    --selftest runs the known-good fixture twice and compares the outcome and
    every value passed to ctx.record / ctx.require_between / ctx.require_close.
    A gate that disagrees with itself is reported as a defect.

Exit codes
    0  every selected check passed
    1  at least one check failed (a deliverable problem)
    3  HARNESS ERROR (environment or gate problem, never blamed on the code)
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

EXIT_PASS, EXIT_FAIL, EXIT_HARNESS = 0, 1, 3


class CheckFailed(Exception):
    """The deliverable is wrong. The message is fed back to the builder."""


class HarnessError(Exception):
    """The environment or the gate is wrong. Never blamed on the deliverable."""


class Context:
    """Passed to setup, teardown and every check."""

    def __init__(self, deliverable: Path, root: Path, fast: bool = False):
        self.deliverable = Path(deliverable)
        self.root = Path(root)
        self.artifacts = self.root / "verify" / "artifacts"
        self.fast = fast            # True on the per-edit run: skip heavy setup
        self.state: dict = {}
        self.records: dict = {}     # values compared across runs by --selftest

    def fail(self, message: str):
        raise CheckFailed(message)

    def require(self, condition, message: str):
        if not condition:
            raise CheckFailed(message)

    def harness_error(self, message: str):
        raise HarnessError(message)

    def record(self, name: str, value):
        """Note a value the gate measured. --selftest fails if it drifts between runs."""
        self.records[name] = value
        return value

    def require_between(self, value, low, high, label: str):
        """Assert low <= value <= high. For scores that vary within a known band."""
        self.record(label, value)
        if not (low <= value <= high):
            raise CheckFailed(f"{label} is {value:.6g}, expected between {low} and {high}")

    def require_close(self, value, expected, tolerance, label: str):
        """Assert |value - expected| <= tolerance."""
        self.record(label, value)
        if abs(value - expected) > tolerance:
            raise CheckFailed(f"{label} is {value:.6g}, expected {expected} "
                              f"(tolerance {tolerance})")


class Gate:
    def __init__(self, name: str):
        self.name = name
        self._checks: dict = {}
        self._modes: dict = {}
        self._good = None
        self._setup = None
        self._teardown = None
        self.root = Path(sys.argv[0]).resolve().parent.parent

    # ---------------------------------------------------- registration --

    def check(self, check_id: str, catches=(), doc: str = "", why: str = "",
              slow: bool = False):
        """Register a check.

        ``slow=True`` keeps the check out of the per-edit run (--fast). It still
        runs when a builder tries to finish and whenever the driver judges.

        ``catches`` names the failure modes this check exists to catch. A check
        that catches none must say ``why`` it exists; a check nobody can
        justify is decoration.
        """
        if check_id in self._checks:
            raise HarnessError(f"duplicate check id '{check_id}'")
        if not catches and not why:
            raise HarnessError(
                f"check '{check_id}' names no failure mode (catches=[...]) and gives "
                "no reason (why='...'). Say what silent failure it catches."
            )

        def register(fn):
            self._checks[check_id] = {
                "id": check_id, "fn": fn, "catches": list(catches),
                "doc": doc or (fn.__doc__ or "").strip(), "why": why,
                "slow": bool(slow),
            }
            return fn
        return register

    def failure_mode(self, mode_id: str, description: str):
        """Register a builder that produces a deliverable broken in one way."""
        def register(fn):
            self._modes[mode_id] = {"id": mode_id, "description": description, "build": fn}
            return fn
        return register

    def good_fixture(self, fn):
        self._good = fn
        return fn

    def setup(self, fn):
        self._setup = fn
        return fn

    def teardown(self, fn):
        self._teardown = fn
        return fn

    # ---------------------------------------------------------- running --

    def run(self, deliverable, only=None, fast=False):
        """Run checks. Returns a list of {id, ok, message, catches, seconds}.

        With ``fast`` the checks marked slow are left out (see ``last_skipped``).
        Raises HarnessError for environment problems.
        """
        ids = list(self._checks) if only is None else list(only)
        unknown = [i for i in ids if i not in self._checks]
        if unknown:
            raise HarnessError("unknown check id(s): " + ", ".join(unknown))
        self.last_skipped = [i for i in ids if fast and self._checks[i]["slow"]]
        ids = [i for i in ids if i not in self.last_skipped]
        self.last_records = {}
        if not ids:
            return []
        ctx = Context(Path(deliverable), self.root, fast=fast)
        results = []
        setup_error = None
        try:
            if self._setup:
                try:
                    self._setup(ctx)
                except HarnessError:
                    raise
                except Exception as exc:  # the deliverable would not start
                    setup_error = f"setup failed: {_describe(exc)}"
            for check_id in ids:
                entry = self._checks[check_id]
                ok, message = True, ""
                started = time.perf_counter()
                if setup_error:
                    ok, message = False, setup_error
                else:
                    try:
                        entry["fn"](ctx)
                    except HarnessError:
                        raise
                    except (CheckFailed, AssertionError) as exc:
                        ok, message = False, str(exc) or "assertion failed (no message)"
                    except Exception as exc:
                        ok, message = False, f"unexpected error: {_describe(exc)}"
                results.append({"id": check_id, "ok": ok, "message": message,
                                "catches": entry["catches"],
                                "seconds": round(time.perf_counter() - started, 3)})
        finally:
            self.last_records = dict(ctx.records)
            if self._teardown:
                try:
                    self._teardown(ctx)
                except Exception as exc:
                    print(f"warning: teardown raised {_describe(exc)}", file=sys.stderr)
        return results

    # --------------------------------------------------------- selftest --

    def _static_problems(self):
        problems = []
        if not self._checks:
            problems.append("the gate declares no checks")
        if self._good is None:
            problems.append("no @gate.good_fixture: nothing proves the gate can pass")
        if not self._modes:
            problems.append("no @gate.failure_mode: nothing proves the gate can fail")
        for mode_id in self._modes:
            if not any(mode_id in c["catches"] for c in self._checks.values()):
                problems.append(f"failure mode {mode_id} has no check that catches it")
        for entry in self._checks.values():
            for mode_id in entry["catches"]:
                if mode_id not in self._modes:
                    problems.append(
                        f"check '{entry['id']}' claims to catch {mode_id}, "
                        "which has no @gate.failure_mode fixture to prove it"
                    )
        return problems

    def selftest(self) -> int:
        """Prove the gate discriminates. A gate never watched failing is not a gate."""
        problems = self._static_problems()
        if problems:
            for p in problems:
                print(f"SELFTEST DEFECT: {p}")
            return EXIT_FAIL
        project_base = self.root / "verify" / "artifacts" / "_selftest"
        bases = [("temp dir", None), ("project dir", project_base)]
        outcomes = {}
        defects = []
        timing = None
        for label, base in bases:
            if base is not None:
                shutil.rmtree(base, ignore_errors=True)
                base.mkdir(parents=True, exist_ok=True)
            work = Path(tempfile.mkdtemp(prefix="gate-selftest-", dir=base))
            try:
                outcome = {}
                good_path = self._build(self._good, work / "good")
                started = time.perf_counter()
                good = self.run(good_path)
                full_seconds = time.perf_counter() - started
                first = (sorted(r["id"] for r in good if not r["ok"]), dict(self.last_records))
                bad_good = [r for r in good if not r["ok"]]
                outcome["good"] = not bad_good
                for r in bad_good:
                    defects.append(f"[{label}] known-GOOD fixture failed '{r['id']}': {r['message']}")
                if base is None:
                    # Same fixture, same place, second run: the gate must agree with itself.
                    again = self.run(good_path)
                    second = (sorted(r["id"] for r in again if not r["ok"]),
                              dict(self.last_records))
                    defects.extend(_drift(first, second))
                    started = time.perf_counter()
                    self.run(good_path, fast=True)
                    timing = (full_seconds, time.perf_counter() - started,
                              len(self.last_skipped))
                for mode_id, mode in self._modes.items():
                    res = self.run(self._build(mode["build"], work / mode_id))
                    failed = [r for r in res if not r["ok"]]
                    right = [r["id"] for r in failed if mode_id in r["catches"]]
                    outcome[mode_id] = sorted(r["id"] for r in failed)
                    if not failed:
                        defects.append(
                            f"[{label}] {mode_id} ({mode['description']}): the broken "
                            "fixture PASSED every check. The gate cannot see this failure."
                        )
                    elif not right:
                        defects.append(
                            f"[{label}] {mode_id}: fixture failed only "
                            f"{', '.join(r['id'] for r in failed)}, none of which claims "
                            f"to catch {mode_id}. Fix the catches=[...] mapping or the check."
                        )
                    else:
                        print(f"ok  [{label}] {mode_id} caught by {', '.join(right)}")
                outcomes[label] = outcome
            finally:
                shutil.rmtree(work, ignore_errors=True)
        if outcomes.get("temp dir") != outcomes.get("project dir"):
            defects.append(
                "the gate gave different results from a temp dir and from the project "
                "dir (often a network drive or path-length issue): "
                f"{json.dumps(outcomes, sort_keys=True)}"
            )
        shutil.rmtree(project_base, ignore_errors=True)
        if defects:
            for d in defects:
                print(f"SELFTEST DEFECT: {d}")
            print(f"SELFTEST: FAIL ({len(defects)} defect(s))")
            return EXIT_FAIL
        if timing:
            print(f"TIMING: full gate {timing[0]:.1f}s, per-edit (--fast) {timing[1]:.1f}s "
                  f"on the good fixture ({timing[2]} slow check(s) skipped per edit)")
        print(f"SELFTEST: PASS (good fixture passes twice with identical results; "
              f"{len(self._modes)} failure mode(s) each caught by a check that names it; "
              "temp and project dirs agree)")
        return EXIT_PASS

    @staticmethod
    def _build(builder, target: Path) -> Path:
        target.mkdir(parents=True, exist_ok=True)
        built = builder(target)
        if built is None:
            raise HarnessError(f"fixture builder {builder.__name__} returned None; "
                               "it must return the deliverable path")
        return Path(built)

    # -------------------------------------------------------------- CLI --

    def matrix(self) -> str:
        lines = ["| Failure mode | Description | Caught by |", "|---|---|---|"]
        for mode_id, mode in self._modes.items():
            by = [c["id"] for c in self._checks.values() if mode_id in c["catches"]]
            lines.append(f"| {mode_id} | {mode['description']} | "
                         f"{', '.join(by) or '**NOTHING**'} |")
        return "\n".join(lines)

    def main(self, argv=None) -> None:
        sys.exit(self._main(argv))

    def _main(self, argv=None) -> int:
        _utf8_streams()
        parser = argparse.ArgumentParser(prog="smoke_test.py", description=f"{self.name} gate")
        parser.add_argument("deliverable", nargs="?")
        parser.add_argument("--only", default=None, help="comma-separated check ids")
        parser.add_argument("--fast", action="store_true",
                            help="skip checks marked slow (the per-edit profile)")
        parser.add_argument("--json", action="store_true", help="last line is a JSON summary")
        parser.add_argument("--list", action="store_true")
        parser.add_argument("--matrix", action="store_true")
        parser.add_argument("--selftest", action="store_true")
        try:
            args = parser.parse_args(argv)
        except SystemExit as exc:
            if exc.code in (0, None):  # --help
                return EXIT_PASS
            print("HARNESS ERROR: bad gate arguments (this is not a deliverable problem)")
            return EXIT_HARNESS
        try:
            if args.list:
                checks = [{"id": c["id"], "doc": c["doc"], "catches": c["catches"],
                           "slow": c["slow"]} for c in self._checks.values()]
                if args.json:
                    print(json.dumps({"gate": self.name, "checks": checks}))
                else:
                    for c in checks:
                        print(f"{c['id']}  [{', '.join(c['catches']) or '-'}]"
                              f"{'  (slow)' if c['slow'] else ''}  {c['doc']}")
                return EXIT_PASS
            if args.matrix:
                print(self.matrix())
                return EXIT_PASS
            if args.selftest:
                return self.selftest()
            if not args.deliverable:
                print("HARNESS ERROR: no deliverable argument given")
                return EXIT_HARNESS
            only = [i.strip() for i in args.only.split(",") if i.strip()] if args.only else None
            results = self.run(args.deliverable, only, fast=args.fast)
        except HarnessError as exc:
            print(f"HARNESS ERROR - this is not a deliverable problem: {exc}")
            if args.json:
                print(json.dumps({"gate": self.name, "status": "harness_error",
                                  "message": str(exc), "checks": []}))
            return EXIT_HARNESS
        failed = [r for r in results if not r["ok"]]
        seen = set()
        for r in results:
            if r["ok"]:
                print(f"PASS {r['id']}")
            elif r["message"] in seen:   # e.g. one setup failure behind every check
                print(f"FAIL {r['id']}: (same cause as above)")
            else:
                seen.add(r["message"])
                print(f"FAIL {r['id']}: {r['message']}")
        for skipped in self.last_skipped:
            print(f"SKIP {skipped} (slow: runs when you finish, not on each edit)")
        if failed:
            print(f"GATE: FAIL ({len(failed)} of {len(results)} checks failed)")
        else:
            print(f"GATE: PASS ({len(results)} checks)")
        if args.json:
            print(json.dumps({"gate": self.name, "status": "fail" if failed else "pass",
                              "checks": results, "skipped_slow": self.last_skipped}))
        return EXIT_FAIL if failed else EXIT_PASS


def _same(a, b) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) \
            and not isinstance(a, bool) and not isinstance(b, bool):
        return abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b))
    return a == b


def _drift(first, second) -> list:
    """Defects for two runs of the same fixture that did not agree."""
    advice = ("Pin every seed, run single-threaded in the gate, and remove anything "
              "that depends on the clock, dict order or the network.")
    if first[0] != second[0]:
        return [f"the gate is NOT DETERMINISTIC: two runs on the same known-GOOD fixture "
                f"failed different checks ({first[0] or 'none'} then {second[0] or 'none'}). "
                + advice]
    out = []
    for name in sorted(set(first[1]) | set(second[1])):
        a, b = first[1].get(name), second[1].get(name)
        if not _same(a, b):
            out.append(f"the gate is NOT DETERMINISTIC: '{name}' was {a!r} then {b!r} on "
                       "two runs of the same known-GOOD fixture. " + advice)
    return out


def _describe(exc: Exception) -> str:
    last = traceback.extract_tb(exc.__traceback__)[-1:] if exc.__traceback__ else []
    where = f" at {Path(last[0].filename).name}:{last[0].lineno}" if last else ""
    return f"{type(exc).__name__}: {exc}{where}"


def _utf8_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
