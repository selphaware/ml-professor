"""T58: the text of the `date_column_undeclared` warning (spec 2.5, 5.4; D69).

Its message holds the column's name as `repr` writes it, `time_col=False`, and the
caveat of the scheme the run used, exactly as `validation.caveat` has it.
"""
import pandas as pd
import pytest

import cp5_helpers as p
import cp6_helpers as g
import cp7_helpers as s
import cp8_helpers as e
import mlp_helpers as h

NAME = "signed up"


def _random_case():
    frame = p.noise_frame(300, seed=5801)
    frame.insert(0, NAME, h.dates(300, start="2021-03-01"))
    return frame, dict(s.SAID), "random", e.RANDOM_CAVEAT


def _entity_case():
    frame = g.entity_frame(60, sizes=5, seed=5802)
    frame.insert(1, NAME, pd.array([d.strftime("%Y-%m-%d") for d in
                                    h.dates(len(frame), start="2021-03-01")], dtype="str"))
    return frame, {"time_col": False, "group_col": "who"}, "grouped", e.GROUPED_CAVEAT


def _agreeing_rows_case():
    frame = p.noise_frame(300, seed=5803)
    frame.insert(0, NAME, pd.array(e.days(300), dtype="int64"))
    return (p.twice(frame, seed=5803), {**s.SAID, "duplicates": "group"}, "grouped",
            e.GROUPED_CAVEAT)


CASES = {
    "random": _random_case,
    "grouped_by_an_entity_column": _entity_case,
    "grouped_by_rows_that_agree": _agreeing_rows_case,
}


@pytest.mark.parametrize("case", list(CASES))
def test_the_date_warning_holds_the_caveat_of_its_scheme(case):
    """M36: `date_column_undeclared` carries the caveat of the random scheme under the grouped scheme."""
    frame, said, scheme, caveat = CASES[case]()
    log, families = s.rec()
    result = s.run(frame, families=families, **said)
    validation = result.to_dict()["validation"]
    assert validation["scheme"] == scheme, validation["scheme"]
    assert validation["caveat"] == caveat, validation["caveat"]
    found = s.warning_records(result, "date_column_undeclared")
    assert [r["column"] for r in found] == [NAME], s.codes(result)
    message = str(found[0]["message"])
    assert repr(NAME) in message, message
    assert "time_col=False" in message, message
    assert validation["caveat"] in message, message
    other = e.GROUPED_CAVEAT if scheme == "random" else e.RANDOM_CAVEAT
    assert other not in message, message
