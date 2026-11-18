"""T23: the README's commands use a table kept in the repo (spec 1). Red today.

These judge `README.md`, which the builder owns; they neither edit it nor add
files under `examples/`.
"""
import re
import shlex
from pathlib import PurePosixPath

import mlp_helpers as h

_UNDER_JUDGE = re.compile(r"(?<![\w.-])(verify|tests)[\\/]", re.I)


def _readme():
    path = h.REPO_ROOT / "README.md"
    assert path.is_file(), "README.md is missing"
    return path.read_text(encoding="utf-8")


def _fenced_lines(text):
    """Lines of fenced blocks, with continuation lines (\\, `, ^) joined."""
    lines = []
    for block in re.findall(r"```[^\n]*\n(.*?)```", text, re.S):
        joined = re.sub(r"[ \t]*[\\`^][ \t]*\r?\n[ \t]*", " ", block)
        lines.extend(line.strip() for line in joined.splitlines() if line.strip())
    return lines


def _profile_commands(text):
    """The arguments after `-m mlprofessor` of every line that runs `profile`."""
    commands = []
    for line in _fenced_lines(text):
        match = re.search(r"-m\s+mlprofessor\s+(profile\b.*)$", line)
        if match:
            tokens = shlex.split(match.group(1), posix=False)
            commands.append([t.strip("\"'") for t in tokens])
    return commands


def _option(args, name):
    for i, token in enumerate(args):
        if token == name and i + 1 < len(args):
            return args[i + 1]
        if token.startswith(name + "="):
            return token.split("=", 1)[1]
    return None


def _is_kept_in_repo(path_text):
    """A relative path, not under verify/ or tests/, to a file under the repo root."""
    text = path_text.replace("\\", "/")
    if re.match(r"^[A-Za-z]:", text) or text.startswith("/") or text.startswith("~"):
        return False
    parts = [p for p in PurePosixPath(text).parts if p not in ("", ".")]
    if not parts or ".." in parts or parts[0].lower() in ("verify", "tests"):
        return False
    return (h.REPO_ROOT.joinpath(*parts)).is_file()


def _tables_named(text):
    tables = [_option(args, "--data") for args in _profile_commands(text)]
    tables += [m.group(2) for m in re.finditer(r"\bprofile\(\s*(['\"])([^'\"]+)\1", text)]
    return [t for t in tables if t is not None]


def test_readme_names_nothing_under_verify_or_tests():
    found = [m.group(0) for m in _UNDER_JUDGE.finditer(_readme())]
    assert found == [], f"README.md names paths under verify/ or tests/: {found}"


def test_every_table_the_readme_names_is_kept_in_the_repo():
    tables = _tables_named(_readme())
    assert tables, "the README names at least one table"
    for table in tables:
        assert _is_kept_in_repo(table), f"{table!r} is not a file kept in the repo"


def _forced(args, out):
    """The command as written, with --out into tmp_path and fast, one-core runs forced."""
    kept, skip = [], False
    for token in args:
        if skip:
            skip = False
            continue
        name = token.split("=", 1)[0]
        if name in ("--out", "--profile", "--n-jobs"):
            skip = "=" not in token
            continue
        kept.append(token)
    return kept + ["--out", str(out), "--profile", "fast", "--n-jobs", "1"]


def test_every_readme_command_runs(tmp_path):
    commands = _profile_commands(_readme())
    assert commands, "the README shows at least one `-m mlprofessor profile` command"
    for k, args in enumerate(commands):
        data = _option(args, "--data")
        assert data is not None, f"command {k} has no --data: {args}"
        assert _is_kept_in_repo(data), f"command {k} reads {data!r}, not a table kept in the repo"
        out = tmp_path / f"run{k}"
        proc = h.run_cli(_forced(args, out))
        assert proc.returncode == 0, f"command {k} failed: {args}\n{proc.stderr}"
        assert h.files_in(out) == ["predictions.csv", "profile.json"]
