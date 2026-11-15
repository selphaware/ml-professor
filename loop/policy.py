"""Stage policy for the loop kit: what an agent may write, and when.

Shared by the hooks (.claude/hooks/) and the driver (loop/run_loop.py).
Standard library only, so it runs on the pinned interpreter with nothing added.

Two layers live here:

* ``check_file_write`` is exact. File tools report an absolute path, so the
  answer is a lookup.
* ``check_shell`` is best effort. Shell text cannot be parsed reliably, so
  it catches the obvious cases and the driver's tamper check (a hash
  snapshot of every locked file) is the backstop that cannot be talked
  round.
"""
from __future__ import annotations

import os
import posixpath
import re
from pathlib import Path

STAGES = ("harness", "tests", "build", "review")
DEFAULT_STAGE = "build"  # an unlabelled session gets the strict policy

# Kit files: identical on every project, never edited by an agent.
KIT_PATHS = (
    ".claude/hooks/",
    ".claude/agents/",
    ".claude/settings.json",
    ".claude/settings.local.json",
    "loop/",
    "verify/gatekit.py",
    "tests/pytest.ini",
)
# The judge: written in the harness stage, locked from then on.
JUDGE_PATHS = (
    "verify/",
    "plan/",
    "spec.md",
    "CLAUDE.md",
    "BOOTSTRAP.md",
    ".claude/skills/",
)
# The test suite: written in the tests stage, locked from then on.
TEST_PATHS = ("tests/",)
# Files that would let app code reconfigure the test run from outside tests/.
ROOT_LOCKED_NAMES = (
    "conftest.py",
    "pytest.ini",
    "tox.ini",
    "sitecustomize.py",
    "usercustomize.py",
)
# Written by the driver and the gate themselves; never part of a snapshot.
SNAPSHOT_IGNORE = ("loop/state/", "verify/artifacts/")
IGNORE_DIR_NAMES = frozenset(
    {
        "__pycache__", ".pytest_cache", ".git", ".venv", "venv", "env",
        "node_modules", ".mypy_cache", ".ruff_cache", ".idea", ".vscode",
    }
)

_DRIVE = re.compile(r"^[a-zA-Z]:/")


def project_root() -> Path:
    """Project root: CLAUDE_PROJECT_DIR if set, else the parent of loop/."""
    env = os.environ.get("CLAUDE_PROJECT_DIR")
    return Path(env) if env else Path(__file__).resolve().parents[1]


def current_stage() -> str:
    stage = os.environ.get("LOOP_STAGE", "").strip().lower()
    return stage if stage in STAGES else DEFAULT_STAGE


# ---------------------------------------------------------------- paths --

def _slash(path) -> str:
    return str(path).replace("\\", "/")


def _is_windows_root(root_s: str) -> bool:
    return bool(_DRIVE.match(root_s)) or os.name == "nt"


def _norm(path_s: str, windows: bool) -> str:
    if windows:
        # Git Bash spells E:\x as /e/x
        m = re.match(r"^/([a-zA-Z])(/.*)?$", path_s)
        if m and not _DRIVE.match(path_s):
            path_s = f"{m.group(1)}:{m.group(2) or '/'}"
    return posixpath.normpath(path_s)


def rel_posix(path, root, cwd=None):
    """Return ``path`` relative to ``root`` in posix form, or None if outside.

    Works on Windows-style strings on any OS (the tests rely on that), and
    compares case-insensitively when the root is a Windows path.
    """
    root_s = _slash(root)
    windows = _is_windows_root(root_s)
    r = _norm(root_s, windows).rstrip("/")
    p = _slash(path).strip()
    is_abs = p.startswith("/") or bool(_DRIVE.match(p))
    if not is_abs:
        base = _slash(cwd) if cwd else root_s
        p = base.rstrip("/") + "/" + p
    n = _norm(p, windows)
    key_n, key_r = (n.lower(), r.lower()) if windows else (n, r)
    if key_n == key_r:
        return ""
    if key_n.startswith(key_r + "/"):
        return n[len(r) + 1:]
    return None


def _resolved_rel(path, root, cwd=None):
    """rel_posix, plus a second opinion through realpath to defeat symlinks."""
    rel = rel_posix(path, root, cwd)
    try:
        if Path(root).exists():
            p = Path(path)
            if not p.is_absolute():
                p = Path(cwd or root) / p
            real = rel_posix(os.path.realpath(p), os.path.realpath(root))
            if real is not None and real != rel:
                # Prefer whichever lands on a locked path.
                if classify(real) != "app" or rel is None:
                    return real
    except (OSError, ValueError):
        pass
    return rel


def _under(rel: str, prefixes) -> bool:
    key = rel.lower()
    for prefix in prefixes:
        p = prefix.lower()
        if p.endswith("/"):
            if key == p[:-1] or key.startswith(p):
                return True
        elif key == p:
            return True
    return False


def classify(rel: str) -> str:
    """Kind of a project-relative path: kit, judge, tests, rootlock or app."""
    if _under(rel, KIT_PATHS):
        return "kit"
    if _under(rel, JUDGE_PATHS):
        return "judge"
    if _under(rel, TEST_PATHS):
        return "tests"
    if "/" not in rel and rel.lower() in ROOT_LOCKED_NAMES:
        return "rootlock"
    return "app"


def locked_kinds(stage: str) -> frozenset:
    """Kinds an agent may not write in ``stage``."""
    if stage == "harness":
        return frozenset({"kit"})
    if stage == "tests":
        return frozenset({"kit", "judge", "rootlock", "app"})
    if stage == "review":
        return frozenset({"kit", "judge", "tests", "rootlock", "app"})
    return frozenset({"kit", "judge", "tests", "rootlock"})  # build


_WHY = {
    "kit": "it is a loop-kit file, which no agent edits",
    "judge": "it belongs to the judge (gate, spec, plan), locked after the harness stage",
    "tests": "the test suite is locked once the tests stage ends",
    "rootlock": "a file with that name at the project root would reconfigure the test run",
    "app": "this stage does not write application files",
}


def _deny(rel, kind, stage) -> str:
    return (
        f"BLOCKED by the harness guard ({stage} stage): cannot write '{rel}' because "
        f"{_WHY[kind]}. Do not look for another way to change it. If you believe the "
        "check, test or spec is itself wrong, stop and say so plainly, with the evidence."
    )


def check_file_write(path, stage, root, cwd=None):
    """Decide a Write/Edit/NotebookEdit. Returns (allowed, reason)."""
    rel = _resolved_rel(path, root, cwd)
    if rel is None:
        if stage in ("tests", "review"):
            return False, (
                f"BLOCKED by the harness guard ({stage} stage): '{path}' is outside "
                "the project, and this stage does not write there."
            )
        return True, ""  # outside the project: Claude Code's own rules decide
    kind = classify(rel)
    if kind in locked_kinds(stage):
        return False, _deny(rel, kind, stage)
    return True, ""


# ---------------------------------------------------------------- shell --

_GIT = re.compile(r"(?<![\w.\-])git(?:\.exe)?(?=\s|$|[\"'])", re.I)
_DRIVER = re.compile(r"(run_loop|install_kit)\.py", re.I)
# The harness stage may run the driver's read-only checks to prove its own work.
_DRIVER_READONLY = re.compile(
    r"^\s*(?:&\s*)?[^;&|<>`$()]*run_loop\.py[\"']?\s+(doctor|status|build\s+--dry-run)\s*$",
    re.I)
_REDIRECT = re.compile(r"(?:\d?>>?|>\|)\s*[\"']?([^\s\"'|&;<>]+)")
_SPLIT = re.compile(r"\|\||&&|[;|\n]")
_CANDIDATE = re.compile(r"[A-Za-z0-9_.\-:/\\~]+")
_ENV_ASSIGN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=\S*$")
_DEST = re.compile(r"-dest(?:ination)?\s+[\"']?([^\s\"']+)", re.I)

_WRITE_CMDS = frozenset(
    {
        # POSIX and cmd.exe
        "rm", "del", "erase", "rmdir", "rd", "mv", "move", "ren", "rename",
        "tee", "touch", "truncate", "dd", "ln", "chmod", "chown", "attrib",
        "icacls", "mklink",
        # PowerShell cmdlets and their common aliases
        "remove-item", "move-item", "rename-item", "new-item", "set-content",
        "add-content", "clear-content", "out-file", "set-itemproperty",
        "ri", "mi", "rni", "ni", "sc", "ac", "clc",
    }
)
_COPY_CMDS = frozenset({"cp", "copy", "copy-item", "cpi", "xcopy", "robocopy"})
_WRITE_API = re.compile(
    r"open\s*\(|write_text|write_bytes|\.unlink|os\.remove|os\.rename|os\.replace|"
    r"shutil\.|rmtree|writeFile|unlinkSync|renameSync|\[IO\.File\]::|"
    r"Out-File|Set-Content|Add-Content|sed\s+(-\w*i|--in-place)|perl\s+-\w*i",
    re.I,
)
_WRAPPERS = frozenset({"sudo", "command", "env", "nohup", "timeout", "nice", "&", "("})


def _first_word(segment: str) -> str:
    words = segment.strip().lstrip("&( ").split()
    for word in words:
        bare = word.strip("\"'")
        if _ENV_ASSIGN.match(bare) or bare.lower() in _WRAPPERS:
            continue
        name = _slash(bare).rsplit("/", 1)[-1].lower()
        return name[:-4] if name.endswith(".exe") else name
    return ""


def _locked_hits(text: str, stage: str, root, cwd):
    hits = []
    locked = locked_kinds(stage) - {"app"}
    for cand in _CANDIDATE.findall(text):
        cand = cand.rstrip(".:")
        if not cand or cand.startswith("-"):
            continue
        rel = rel_posix(cand, root, cwd)
        if rel and classify(rel) in locked:
            hits.append(rel)
    return hits


def check_shell(command: str, stage: str, root, cwd=None):
    """Decide a Bash/PowerShell command. Best effort. Returns (allowed, reason)."""
    if not command or not command.strip():
        return True, ""
    if _GIT.search(command):
        return False, (
            "BLOCKED by the harness guard: agents do not run git in this project. "
            "The human owns every git operation. Carry on without it."
        )
    if stage == "harness" and _DRIVER_READONLY.match(command.strip()):
        return True, ""
    if _DRIVER.search(command):
        return False, (
            "BLOCKED by the harness guard: the loop driver and the kit installer are "
            "run by the human, never by an agent."
        )

    def blocked(rel):
        kind = classify(rel)
        return False, (
            f"BLOCKED by the harness guard ({stage} stage): this command looks like it "
            f"writes to '{rel}', and {_WHY[kind]}. Reading and running locked files is "
            "fine; use the Read tool to read them. If the check, test or spec is "
            "itself wrong, stop and say so plainly, with the evidence."
        )

    locked = locked_kinds(stage) - {"app"}
    for target in _REDIRECT.findall(command):
        rel = rel_posix(target, root, cwd)
        if rel and classify(rel) in locked:
            return blocked(rel)

    for segment in _SPLIT.split(command):
        if not segment.strip():
            continue
        word = _first_word(segment)
        if word in _COPY_CMDS:
            dest = _DEST.search(segment)
            if dest:
                target = dest.group(1)
            else:
                cands = [c for c in _CANDIDATE.findall(segment) if not c.startswith("-")]
                target = cands[-1] if len(cands) > 1 else ""
            rel = rel_posix(target, root, cwd) if target else None
            if rel and classify(rel) in locked:
                return blocked(rel)
            continue
        if word in _WRITE_CMDS or _WRITE_API.search(segment):
            hits = _locked_hits(segment, stage, root, cwd)
            if hits:
                return blocked(hits[0])
    return True, ""


# -------------------------------------------------------------- listing --

def iter_files(root, include=None):
    """Yield (rel_posix, Path) for project files, skipping caches and state."""
    root = Path(root)
    for dirpath, dirnames, filenames in os.walk(root):
        # Skip caches, and any virtual environment whatever it is called: a venv
        # kept inside the project is thousands of files that are not the project.
        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIR_NAMES
                       and not os.path.isfile(os.path.join(dirpath, d, "pyvenv.cfg"))]
        for name in filenames:
            path = Path(dirpath) / name
            rel = path.relative_to(root).as_posix()
            if _under(rel, SNAPSHOT_IGNORE):
                continue
            if include is None or include(rel):
                yield rel, path
