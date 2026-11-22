"""T18: two CSV edges and header cells of white space (spec 6, 7, 2.4; D39, D42).

Green today. A failure here means the application is wrong, not the test.
"""
import numpy as np
import pandas as pd
import pytest

import mlp_helpers as h


def _rows(n=120, start=0):
    return "".join(f"{i},{(i % 7) + 0.5},{i % 2}\n" for i in range(start, start + n))


def _good_bytes(n=120):
    return ("a,b,y\n" + _rows(n)).encode("utf-8")


def _bad_header():
    return b"a,b\xff,y\n" + _rows().encode("utf-8")


def _bad_first_row():
    return b"a,b,y\n1,2\xff.5,0\n" + _rows().encode("utf-8")


def _bad_third_row():
    return b"a,b,y\n" + _rows(2).encode("utf-8") + b"2,\xff,0\n" + _rows(120, 3).encode("utf-8")


def _bad_deep():
    good = _rows(25_000).encode("utf-8")
    assert len(good) > 200_000
    return b"a,b,y\n" + good + b"25000,9.5\xff,1\n" + _rows(10, 25_001).encode("utf-8")


def _bad_last_cell():
    return b"a,b,y\n" + _rows().encode("utf-8") + b"999,3.5,1\xff"


def _latin1():
    lines = ["a,b,y"] + [f"{i},{'café' if i % 3 == 0 else 'thé'},{i % 2}" for i in range(120)]
    return ("\n".join(lines) + "\n").encode("latin-1")


BAD_FILES = {
    "header_line": _bad_header,
    "first_data_row": _bad_first_row,
    "third_data_row": _bad_third_row,
    "past_200_kB": _bad_deep,
    "last_cell_no_line_end": _bad_last_cell,
    "latin1_text": _latin1,
}


@pytest.mark.parametrize("case", list(BAD_FILES))
def test_a_file_that_is_not_utf8_is_an_input_error_wherever_the_bad_byte_sits(tmp_path, case):
    import mlprofessor
    from mlprofessor import stages
    from mlprofessor.errors import InputError

    content = BAD_FILES[case]()
    with pytest.raises(UnicodeDecodeError):
        content.decode("utf-8")
    path = tmp_path / "bad.csv"
    path.write_bytes(content)

    with pytest.raises(InputError):
        stages.load_table(str(path))
    out = tmp_path / "out"
    with pytest.raises(InputError):
        mlprofessor.profile(str(path), "y", out_dir=out, **h.FAST)
    assert h.files_in(out) == []


LEADING = {
    "three_spaces": b"   \n",
    "a_tab": b"\t\n",
    "spaces_tabs_and_an_empty_line": b"  \t \n\n\t\t\n   \n",
    "bom_then_spaces": b"\xef\xbb\xbf   \n",
    "two_empty_then_one_space": b"\n\n \n",
}


@pytest.mark.parametrize("case", list(LEADING))
def test_leading_lines_of_spaces_and_tabs_are_skipped_as_pandas_skips_them(tmp_path, case):
    from mlprofessor import stages

    path = tmp_path / "lead.csv"
    path.write_bytes(LEADING[case] + _good_bytes())
    expected = pd.read_csv(path, low_memory=False)
    assert list(expected.columns) == ["a", "b", "y"]

    frame, info = stages.load_table(str(path))
    assert list(frame.columns) == ["a", "b", "y"]
    assert len(frame) == len(expected) == 120
    for name in ["a", "b", "y"]:
        assert np.array_equal(frame[name].to_numpy(dtype=float), expected[name].to_numpy(dtype=float))
    assert info["warnings"] == []


def test_duplicate_names_are_still_found_after_leading_blank_lines(tmp_path):
    from mlprofessor import stages
    from mlprofessor.errors import DuplicateColumnsError

    path = tmp_path / "dup.csv"
    body = "a,b,a\n" + "".join(f"{i},{i + 1},{i % 2}\n" for i in range(120))
    path.write_bytes(b"  \n\t\n" + body.encode("utf-8"))
    with pytest.raises(DuplicateColumnsError):
        stages.load_table(str(path))


@pytest.mark.parametrize("first_line", [b'""\n', b"\f\n"], ids=["two_quotes", "form_feed"])
def test_a_line_pandas_reads_as_the_header_is_not_skipped(tmp_path, first_line):
    from mlprofessor import stages
    from mlprofessor.errors import InputError

    path = tmp_path / "content_first.csv"
    path.write_bytes(first_line + _good_bytes())
    assert len(pd.read_csv(path, low_memory=False).columns) == 1, "pandas reads a header of one column"
    with pytest.raises(InputError):
        stages.load_table(str(path))


def test_profile_runs_on_a_file_with_leading_blank_lines(tmp_path):
    import mlprofessor

    table = h.clf_frame(n=160, seed=5)
    path = tmp_path / "lead.csv"
    path.write_bytes(b" \t\n\n  \n" + table.to_csv(index=False).encode("utf-8"))
    result = h.stated_profile(str(path), "y", **h.FAST)
    assert list(h.columns_of(result)) == list(table.columns)
    assert h.rows_of(result)["loaded"] == len(table)
    assert "blank_header" not in [code for code, _ in h.warnings_of(result)]


def test_header_cells_of_only_white_space_are_names_like_any_other(tmp_path):
    import mlprofessor
    from mlprofessor import stages
    from mlprofessor.errors import DuplicateColumnsError

    rows = "".join(f"{i},{i % 5},{(i * 7) % 11},{i % 2}\n" for i in range(120))

    same = tmp_path / "same.csv"
    same.write_text("a, , ,y\n" + rows, encoding="utf-8")
    with pytest.raises(DuplicateColumnsError):
        stages.load_table(str(same))
    out = tmp_path / "out"
    with pytest.raises(DuplicateColumnsError):
        mlprofessor.profile(str(same), "y", out_dir=out, **h.FAST)
    assert h.files_in(out) == []

    differ = tmp_path / "differ.csv"
    differ.write_text("a, ,  ,y\n" + rows, encoding="utf-8")
    names = ["a", " ", "  ", "y"]
    assert list(pd.read_csv(differ, low_memory=False).columns) == names
    frame, info = stages.load_table(str(differ))
    assert list(frame.columns) == names
    assert info["warnings"] == []
