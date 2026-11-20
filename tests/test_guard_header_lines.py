"""T24: a line pandas reads as the header, before a header of two columns (spec 6, 9; F6d).

Green today. A failure here means the application is wrong, not the test.

pandas skips a leading line only when it holds nothing but spaces and tabs. A
form feed, a no-break space or two quotes round a space are content to it: it
reads that line as a header of one column, and every line after it, the real
header among them, then has more fields than the header. With two fields a
line, pandas makes the first field a plain index, so the file loads as one
column and the second real column is lost without a word. The loader must stop
with `input_error` (data rows longer than the header).
"""
import pandas as pd
import pytest

import mlp_helpers as h

N_ROWS = 160

FIRST_LINES = {
    "form_feed": b"\f\n",
    "no_break_space": " \n".encode("utf-8"),
    "two_quotes_round_a_space": b'" "\n',
}


def _two_column_body():
    rows = "".join(f"{(i % 13) * 0.5 + 0.25},{i % 2}\n" for i in range(N_ROWS))
    return ("a,y\n" + rows).encode("utf-8")


@pytest.mark.parametrize("case", list(FIRST_LINES))
def test_a_line_pandas_reads_as_the_header_before_a_two_column_header_is_an_input_error(tmp_path, case):
    """F6d: a line pandas takes for the header, before a two-column header, must stop the run, not lose a column."""
    import mlprofessor
    from mlprofessor import stages
    from mlprofessor.errors import InputError

    path = tmp_path / "lead.csv"
    path.write_bytes(FIRST_LINES[case] + _two_column_body())

    read = pd.read_csv(path, low_memory=False)
    assert len(read.columns) == 1, "pandas reads the first line as a header of one column"
    assert len(read) == N_ROWS + 1, "the real header becomes a data row"

    with pytest.raises(InputError):
        stages.load_table(str(path))

    out = tmp_path / "out"
    with pytest.raises(InputError):
        mlprofessor.profile(str(path), "y", out_dir=out, **h.FAST)
    assert h.files_in(out) == []
