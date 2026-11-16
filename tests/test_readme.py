"""T17: README.md: every command shown names things that exist."""
import re
from pathlib import Path

import mlp_helpers as h

INTERPRETER = r"E:\CODE\ml-professor\mlprof_env\Scripts\python.exe"


def _readme():
    path = h.REPO_ROOT / "README.md"
    assert path.is_file(), "README.md is missing"
    return path.read_text(encoding="utf-8")


def _command_lines(text):
    """Lines of fenced blocks, with continuation lines joined."""
    lines = []
    for block in re.findall(r"```[^\n]*\n(.*?)```", text, re.S):
        joined = re.sub(r"[ \t]*[\\`^][ \t]*\r?\n[ \t]*", " ", block)
        lines.extend(ln.strip() for ln in joined.splitlines() if ln.strip())
    return lines


def test_readme_states_interpreter_and_that_the_package_is_not_installed():
    text = _readme()
    assert INTERPRETER.lower() in text.lower()
    assert re.search(r"\b(not|never)\b[^.\n]{0,60}\binstall", text, re.I)


def test_every_command_names_things_that_exist():
    text = _readme()
    commands = [ln for ln in _command_lines(text) if re.search(r"-m\s+mlprofessor\b", ln)]
    assert any(re.search(r"-m\s+mlprofessor\s+profile\b", ln) for ln in commands)

    help_out = h.run_cli(["profile", "--help"])
    assert help_out.returncode == 0
    known = set(re.findall(r"--[a-z][a-z-]*", help_out.stdout)) | {"--help"}
    for line in commands:
        for option in re.findall(r"(?<![\w-])--[a-z][a-z-]*", line):
            assert option in known, (option, line)
        for exe in re.findall(r"[A-Za-z]:\\[^\s`'\"]+?\.exe", line):
            assert Path(exe).is_file(), exe
        for raw in re.findall(r"--data[ =]+(\"[^\"]+\"|'[^']+'|\S+)", line):
            data = Path(raw.strip("\"'"))
            if not data.is_absolute():
                data = h.REPO_ROOT / data
            assert data.is_file(), (raw, line)
