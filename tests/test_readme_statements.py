"""T53: the README on the sixth pass (spec 1; D62 to D66).

These judge `README.md`, which the builder owns; they neither edit it nor add files
under `examples/`. The README's commands are read as `test_readme_examples` reads them,
and each is run as written, with no flag added.
"""
import re

import cp7_helpers as s
import mlp_helpers as h
import test_readme_examples as rx


def _has(args, *flags):
    return any(token in flags or token.split("=", 1)[0] in flags for token in args)


def test_every_readme_command_states_both_and_runs_as_written(tmp_path):
    commands = rx._profile_commands(rx._readme())
    assert commands, "the README shows at least one `-m mlprofessor profile` command"
    for k, args in enumerate(commands):
        assert _has(args, "--time-col", "--no-time-order"), f"command {k}: {args}"
        assert _has(args, "--group-col", "--no-entities"), f"command {k}: {args}"
        out = tmp_path / f"run{k}"
        proc = s.run_cli(rx._forced(args, out))
        assert proc.returncode == 0, f"command {k} failed: {args}\n{proc.stderr}"
        assert h.files_in(out) == ["predictions.csv", "profile.json"]


def _profile_calls(text):
    """The text of every `profile(...)` call in a fenced block, parentheses balanced."""
    calls = []
    for block in re.findall(r"```[^\n]*\n(.*?)```", text, re.S):
        for match in re.finditer(r"\bprofile\(", block):
            depth, end = 0, None
            for i in range(match.end() - 1, len(block)):
                if block[i] == "(":
                    depth += 1
                elif block[i] == ")":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            assert end is not None, f"unbalanced call: {block[match.start():][:80]}"
            calls.append(block[match.start():end])
    return calls


def test_every_python_example_states_both():
    calls = _profile_calls(rx._readme())
    assert calls, "the README shows at least one `profile(...)` call"
    for call in calls:
        assert re.search(r"\btime_col\s*=", call), call
        assert re.search(r"\bgroup_col\s*=", call), call


def test_the_readme_names_what_this_pass_added():
    text = rx._readme()
    for name in ("YYYYMMDD", "n_rows_same_features", "same_features_grouped",
                 s.KEPT, s.IS_HOLDOUT):
        assert name in text, f"the README does not name {name}"
    for old in ("n_rows_with_copies", "copies_grouped"):
        assert old not in text, f"the README still names {old}"
