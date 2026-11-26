"""T76: the judge's two other tools exit 0 on the application, strictly (spec 8.2; D73,
D74 point 7, D76, D78 point 8, D80, D81).

The strict form, as `tests/test_judge_tools_pass8.py` was until the ninth pass: each test
asks `test_judge_tools.run_tool` for the tool's one run of the session, so the three
files together cost one run of each tool, and asserts exit code 0. Nothing is open: a
pair or a table that fails, on neither list of stated limits, fails the tool. Red until
the tasks of the ninth pass before T76 are built (D78 point 8).
"""
import test_judge_tools as j


def _assert_exit_0(tool, tmp_path_factory):
    script, timeout = tool
    code, _, tail = j.run_tool(script, timeout, tmp_path_factory)
    assert code == 0, f"{script} exited {code}; its last lines:\n{tail}"


def test_the_perturbation_battery_exits_0(tmp_path_factory):
    """Every pair of the battery holds or is a stated limit (D76, D80, D81)."""
    _assert_exit_0(j.BATTERY, tmp_path_factory)


def test_the_generated_tables_exit_0(tmp_path_factory):
    """Every generated table agrees with the reference or is a stated limit."""
    _assert_exit_0(j.TABLES, tmp_path_factory)
