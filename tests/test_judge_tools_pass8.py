"""T68: the judge's two other tools on the application, in the form the eighth pass left
(spec 8.2; D73, D74 point 7, D76, D78 point 8).

Until the ninth pass this file was strict; since D78 point 8 it takes the accepting form
of `tests/test_judge_tools.py`, on the same session run: each test asks that file's
`run_tool` for the tool's one run of the session, so the files together cost one run of
each tool. The strict form is `tests/test_judge_tools_pass9.py` (T76).
"""
import test_judge_tools as j


def test_the_perturbation_battery_exits_0(tmp_path_factory):
    """Every pair of the battery holds or is a stated limit (D74, D76), or is one the ninth pass owns."""
    j.test_the_perturbation_battery_holds(tmp_path_factory)


def test_the_generated_tables_exit_0(tmp_path_factory):
    """Every generated table agrees with the reference or is a stated limit (D74), or is one the ninth pass owns."""
    j.test_the_generated_tables_agree_with_the_reference(tmp_path_factory)
