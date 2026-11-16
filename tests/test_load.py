"""T2: `stages.load_table` and `stages.usable_rows` (spec steps 1-2, section 6)."""
import numpy as np
import pandas as pd
import pytest

MISSING_TOKENS = ["", "n/a", "N/A", "NA", "NaN", "nan", "-NaN", "-nan", "null", "NULL",
                  "None", "#N/A", "#N/A N/A", "#NA", "<NA>", "1.#IND", "-1.#IND",
                  "1.#QNAN", "-1.#QNAN"]


def _write(path, text):
    path.write_text(text, encoding="utf-8", newline="")
    return str(path)


def _csv_types_frame():
    rng = np.random.default_rng(3)
    n = 40
    frame = pd.DataFrame({
        "ints": rng.integers(-50, 50, size=n),
        "floats": rng.normal(size=n),
        "floats_missing": np.where(rng.random(n) < 0.2, np.nan, rng.normal(size=n)),
        "text": [f"level_{k}" for k in rng.integers(0, 5, size=n)],
    })
    frame.index = rng.permutation(np.arange(100, 100 + n))
    return frame


def test_dataframe_and_csv_load_equal(tmp_path):
    from mlprofessor import stages

    frame = _csv_types_frame()
    before = frame.copy(deep=True)
    path = tmp_path / "t.csv"
    frame.to_csv(path, index=False)

    from_df, info_df = stages.load_table(frame)
    from_csv, info_csv = stages.load_table(str(path))

    n = len(frame)
    assert list(from_df.index) == list(range(n))
    assert list(from_csv.index) == list(range(n))
    pd.testing.assert_frame_equal(from_df, from_csv, check_dtype=False)
    # the caller's frame, index included, is untouched
    pd.testing.assert_frame_equal(frame, before)
    assert list(frame.index) == list(before.index)


def test_column_names_become_strings():
    from mlprofessor import stages

    frame = pd.DataFrame({0: [1, 2, 3], 1: ["a", "b", "c"]})
    loaded, _ = stages.load_table(frame)
    assert list(loaded.columns) == ["0", "1"]


def test_f6_blank_lines_are_not_rows(tmp_path):
    """F6: blank lines must not shift or invent rows."""
    from mlprofessor import stages

    path = _write(tmp_path / "b.csv", "a,y\n1,0\n\n2,1\n\n\n3,0\n\n")
    loaded, _ = stages.load_table(path)
    assert len(loaded) == 3
    assert loaded["a"].tolist() == [1, 2, 3]
    assert loaded["y"].tolist() == [0, 1, 0]


def test_line_of_commas_is_a_row_of_missing_values(tmp_path):
    from mlprofessor import stages

    path = _write(tmp_path / "c.csv", "a,y\n1,0\n,\n2,1\n")
    loaded, _ = stages.load_table(path)
    assert len(loaded) == 3
    assert loaded.iloc[1].isna().all()


def test_missing_spellings_load_as_missing(tmp_path):
    from mlprofessor import stages

    lines = ["k,v,w"]
    for i, tok in enumerate(MISSING_TOKENS):
        lines.append(f"{i},{tok},{i}")
    base = len(MISSING_TOKENS)
    for i in range(10):
        lines.append(f"{base + i},{i}.5,{i}")
    lines.append(f'{base + 10},7.25,"1,200"')
    lines.append(f"{base + 11},8.25,abc")
    loaded, _ = stages.load_table(_write(tmp_path / "m.csv", "\n".join(lines) + "\n"))

    assert len(loaded) == base + 12
    assert loaded["v"].iloc[:base].isna().all()
    assert loaded["v"].iloc[base:].notna().all()
    assert loaded["w"].notna().all()
    assert str(loaded["w"].iloc[base + 10]).strip() == "1,200"
    assert str(loaded["w"].iloc[base + 11]).strip() == "abc"


def test_duplicate_columns_in_csv_raise(tmp_path):
    from mlprofessor import errors, stages

    path = _write(tmp_path / "d.csv", "a,b,a\n1,2,3\n4,5,6\n")
    with pytest.raises(errors.DuplicateColumnsError):
        stages.load_table(path)


def test_duplicate_columns_in_dataframe_raise():
    from mlprofessor import errors, stages

    with pytest.raises(errors.DuplicateColumnsError):
        stages.load_table(pd.DataFrame([[1, 2], [3, 4]], columns=["a", "a"]))
    with pytest.raises(errors.DuplicateColumnsError):
        stages.load_table(pd.DataFrame([[1, 2], [3, 4]], columns=[1, "1"]))


def test_blank_header_kept_and_warned(tmp_path):
    from mlprofessor import stages

    path = _write(tmp_path / "h.csv", "a,,c\n1,2,3\n4,5,6\n")
    loaded, info = stages.load_table(path)
    assert list(loaded.columns) == ["a", "Unnamed: 1", "c"]
    found = [w for w in info["warnings"] if w["code"] == "blank_header"]
    assert len(found) == 1
    assert found[0]["column"] == "Unnamed: 1"
    assert "message" in found[0]


@pytest.mark.parametrize("case", ["missing_file", "empty_file", "long_rows"])
def test_bad_files_raise_input_error(tmp_path, case):
    from mlprofessor import errors, stages

    if case == "missing_file":
        path = str(tmp_path / "nowhere.csv")
    elif case == "empty_file":
        path = _write(tmp_path / "e.csv", "")
    else:
        path = _write(tmp_path / "l.csv", "a,b\n1,2,3\n4,5,6\n")
    with pytest.raises(errors.InputError):
        stages.load_table(path)


@pytest.mark.parametrize("value", [123, [1, 2, 3], None, {"a": [1]}])
def test_neither_dataframe_nor_path_raises_input_error(value):
    from mlprofessor import errors, stages

    with pytest.raises(errors.InputError):
        stages.load_table(value)


def test_info_kind_and_source(tmp_path):
    from mlprofessor import stages

    frame = pd.DataFrame({"a": [1, 2], "y": [0, 1]})
    path = tmp_path / "k.csv"
    frame.to_csv(path, index=False)
    _, info = stages.load_table(frame)
    assert info["kind"] == "dataframe"
    assert info["source"] == "<DataFrame>"
    assert info["warnings"] == []
    _, info = stages.load_table(str(path))
    assert info["kind"] == "csv"
    assert info["source"] == str(path)


# ------------------------------------------------------------- usable_rows

def test_f6_usable_rows_count_missing_targets():
    """F6: every dropped row is counted, and the counts add up."""
    from mlprofessor import stages

    y = np.array([0, 1, np.nan, 1, 0, np.nan, 1, np.inf, -np.inf, 0, 1, 0], dtype=float)
    frame = pd.DataFrame({"x": np.arange(len(y), dtype=float), "y": y})
    expected = [i for i, v in enumerate(y) if np.isfinite(v)]

    res = stages.usable_rows(frame, "y")
    assert np.asarray(res["usable"]).tolist() == expected
    assert res["dropped_missing_target"] == len(y) - len(expected)
    assert res["dropped_missing_time"] == 0
    assert len(res["usable"]) + res["dropped_missing_target"] + res["dropped_missing_time"] == len(frame)


def test_usable_rows_text_target():
    from mlprofessor import stages

    frame = pd.DataFrame({"x": range(6), "y": ["a", None, "b", "a", np.nan, "b"]})
    res = stages.usable_rows(frame, "y")
    assert np.asarray(res["usable"]).tolist() == [0, 2, 3, 5]
    assert res["dropped_missing_target"] == 2


def test_f6_usable_rows_with_time_column():
    """F6: rows without a time value are counted apart; a row missing both counts under the target."""
    from mlprofessor import stages

    y = [1.0, 2.0, np.nan, 4.0, 5.0, np.nan, 7.0, 8.0, 9.0, 10.0]
    t = [1.0, np.nan, np.nan, 4.0, np.inf, 6.0, 7.0, np.nan, 9.0, 10.0]
    frame = pd.DataFrame({"t": t, "y": y})
    target_missing = [i for i in range(10) if not np.isfinite(y[i])]
    time_missing = [i for i in range(10) if np.isfinite(y[i]) and not np.isfinite(t[i])]
    usable = [i for i in range(10) if i not in target_missing and i not in time_missing]

    res = stages.usable_rows(frame, "y", time_col="t")
    assert np.asarray(res["usable"]).tolist() == usable
    assert res["dropped_missing_target"] == len(target_missing) == 2
    assert res["dropped_missing_time"] == len(time_missing) == 3
    assert len(res["usable"]) + res["dropped_missing_target"] + res["dropped_missing_time"] == 10


def test_usable_rows_datetime_time_column_with_nat():
    from mlprofessor import stages

    t = pd.Series(pd.date_range("2024-01-01", periods=6, freq="D"))
    t.iloc[3] = pd.NaT
    frame = pd.DataFrame({"t": t, "y": [0, 1, 0, 1, 0, 1]})
    res = stages.usable_rows(frame, "y", time_col="t")
    assert np.asarray(res["usable"]).tolist() == [0, 1, 2, 4, 5]
    assert res["dropped_missing_time"] == 1
    assert res["dropped_missing_target"] == 0


def test_usable_rows_missing_column_raises():
    from mlprofessor import errors, stages

    frame = pd.DataFrame({"x": [1, 2], "y": [0, 1]})
    with pytest.raises(errors.ColumnNotFoundError):
        stages.usable_rows(frame, "nope")
    with pytest.raises(errors.ColumnNotFoundError):
        stages.usable_rows(frame, "y", time_col="nope")
