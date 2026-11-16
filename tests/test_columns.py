"""T3: `stages.column_roles`, the rule table of step 5 (spec 3, 4.3, 4.5, D19, D28, D34)."""
import numpy as np
import pandas as pd
import pytest

import mlp_helpers as h

N = 120


def _target(n=N):
    return np.arange(n) % 2


def _roles(frame, target="y", time_col=None):
    from mlprofessor import stages

    roles = stages.column_roles(frame, target, time_col=time_col)
    return {r["name"]: r for r in h.records(roles)}, roles


def _via_csv(tmp_path, frame, name="t.csv"):
    """Write the frame to CSV and load it as the tool would."""
    from mlprofessor import stages

    path = tmp_path / name
    frame.to_csv(path, index=False)
    loaded, _ = stages.load_table(str(path))
    return loaded


def _rule_frame():
    rng = np.random.default_rng(5)
    when = pd.date_range("2023-03-01", periods=N, freq="D")
    words = list(rng.permutation([f"w{k}" for k in range(90)] + [f"w{k}" for k in range(30)]))
    frame = pd.DataFrame({
        "const_num": np.full(N, 7),
        "const_text": ["a", " a", "a "] * (N // 3),
        "all_missing": np.full(N, np.nan),
        "dt": when,
        "dt_tz": pd.date_range("2023-03-01", periods=N, freq="h", tz="UTC"),
        "free_text": words,
        "text_id": [f"ID{k:05d}" for k in rng.permutation(N)],
        "lists": [[k, k + 1] for k in range(N)],
        "mixed": [k if k % 2 else f"s{k}" for k in range(N)],
        "duration": pd.to_timedelta(np.arange(N), unit="D"),
        "complex": np.arange(N) + 1j,
        "category": pd.Categorical(["lo", "mid", "hi"] * (N // 3)),
        "colour": ["red", "green", "blue", "red"] * (N // 4),
        "flag": rng.random(N) < 0.5,
        "flag_nullable": pd.array([True, False, None, True] * (N // 4), dtype="boolean"),
        "count_nullable": pd.array([int(v) if v < 8 else None for v in rng.integers(0, 9, size=N)],
                                   dtype="Int64"),
        "measure": rng.normal(size=N),
    })
    frame["y"] = _target()
    return frame


EXPECTED = {
    "const_num": ("excluded", "constant", None),
    "const_text": ("excluded", "constant", None),
    "all_missing": ("excluded", "all_missing", None),
    "dt": ("excluded", "datetime", None),
    "dt_tz": ("excluded", "datetime", None),
    "free_text": ("excluded", "free_text", None),
    "text_id": ("excluded", "id_like", None),
    "lists": ("excluded", "unsupported_type", None),
    "mixed": ("excluded", "unsupported_type", None),
    "duration": ("excluded", "unsupported_type", None),
    "complex": ("excluded", "unsupported_type", None),
    "category": ("feature", None, "categorical"),
    "colour": ("feature", None, "categorical"),
    "flag": ("feature", None, "numeric"),
    "flag_nullable": ("feature", None, "numeric"),
    "count_nullable": ("feature", None, "numeric"),
    "measure": ("feature", None, "numeric"),
    "y": ("target", None, None),
}


def test_f6_every_column_once_in_file_order():
    """F6: no column is lost or reordered; each carries the eleven keys."""
    frame = _rule_frame()
    by_name, roles = _roles(frame)
    assert roles["name"].tolist() == list(frame.columns)
    assert roles["position"].tolist() == list(range(frame.shape[1]))
    assert set(h.COLUMN_KEYS) <= set(roles.columns)


@pytest.mark.parametrize("column", sorted(EXPECTED))
def test_rule_table_one_column_per_rule(column):
    by_name, _ = _roles(_rule_frame())
    role, reason, kind = EXPECTED[column]
    rec = by_name[column]
    assert rec["role"] == role
    assert rec["reason"] == reason
    if role == "feature":
        assert rec["kind"] == kind


def test_leak_rule_null_and_nothing_imputed():
    _, roles = _roles(_rule_frame())
    assert all(v is None for v in h.plain(roles["leak_rule"].tolist()))
    assert all(int(v) == 0 for v in roles["n_imputed"])


def test_time_column_has_role_time():
    frame = pd.DataFrame({"t": np.arange(N, dtype=float), "x": np.random.default_rng(0).normal(size=N),
                          "y": _target()})
    by_name, _ = _roles(frame, time_col="t")
    assert by_name["t"]["role"] == "time"
    assert by_name["y"]["role"] == "target"


def test_missing_column_raises():
    from mlprofessor import errors

    with pytest.raises(errors.ColumnNotFoundError):
        _roles(pd.DataFrame({"x": range(N), "y": _target()}), target="nope")


DATE_FORMATS = ["%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%d.%m.%Y",
                "%d-%b-%Y", "%d %B %Y", "%b %d, %Y", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M"]


@pytest.mark.parametrize("fmt", DATE_FORMATS)
def test_text_dates_are_datetime(tmp_path, fmt):
    when = pd.date_range("2022-01-03", periods=N, freq="37h")
    frame = pd.DataFrame({"d": when.strftime(fmt), "y": _target()})
    by_name, _ = _roles(_via_csv(tmp_path, frame))
    assert by_name["d"]["role"] == "excluded"
    assert by_name["d"]["reason"] == "datetime"


@pytest.mark.parametrize("values", [
    ["1,200", "3,400", "15,000", "2,750"],
    ["May", "June", "July"],
    ["09:30", "10:45", "13:15", "16:00"],
    ["2024Q1", "2024Q2", "2024Q3", "2024Q4"],
    ["1/2", "3/4", "5/8"],
])
def test_not_dates(tmp_path, values):
    frame = pd.DataFrame({"c": (values * N)[:N], "y": _target()})
    by_name, _ = _roles(_via_csv(tmp_path, frame))
    assert by_name["c"]["reason"] != "datetime"


# ------------------------------------------------------------- coercion

def test_f6_coerced_cells_counted(tmp_path):
    """F6: text cells turned into missing are counted, and n/a is missing, not coerced."""
    rng = np.random.default_rng(1)
    cells = [f"{v:.3f}" for v in rng.normal(size=N)]
    words = {3: "abc", 17: "unknown", 40: "12 units", 77: "-"}
    nas = [9, 55, 101]
    for i, w in words.items():
        cells[i] = w
    for i in nas:
        cells[i] = "n/a"
    loaded = _via_csv(tmp_path, pd.DataFrame({"v": cells, "y": _target()}))
    by_name, _ = _roles(loaded)
    rec = by_name["v"]
    assert rec["role"] == "feature"
    assert rec["kind"] == "numeric"
    assert rec["n_coerced"] == len(words)
    assert rec["n_missing"] == len(nas)


def test_f6_half_text_column_is_not_numeric(tmp_path):
    """F6: a column half made of words is not silently read as numbers."""
    rng = np.random.default_rng(2)
    cells = [f"{v:.3f}" if i % 2 else ["low", "high", "medium"][i % 3]
             for i, v in enumerate(rng.normal(size=N))]
    by_name, _ = _roles(_via_csv(tmp_path, pd.DataFrame({"v": cells, "y": _target()})))
    rec = by_name["v"]
    assert not (rec["role"] == "feature" and rec["kind"] == "numeric")
    assert rec["n_coerced"] == 0


# ------------------------------------------------- thousands separators

def _amounts(seed, low=10.0, high=50_000.0, n=N):
    return np.round(np.random.default_rng(seed).uniform(low, high, size=n), 2)


def test_f6_separators_throughout_excluded(tmp_path):
    """F6: a column written with separators is excluded, never parsed or half-read."""
    cells = [f"{v:,.2f}" for v in _amounts(3)]
    cells[0], cells[1], cells[2] = "-3,450.75", "+12,345,678", "532.10"
    assert any("," in c for c in cells) and any("," not in c for c in cells)
    by_name, _ = _roles(_via_csv(tmp_path, pd.DataFrame({"rev": cells, "y": _target()})))
    rec = by_name["rev"]
    assert (rec["role"], rec["reason"]) == ("excluded", "numeric_with_separators")
    assert rec["n_coerced"] == 0


@pytest.mark.parametrize("n_sep", [1, 3])
def test_f6_plain_numbers_with_a_few_separator_cells_excluded(tmp_path, n_sep):
    """F6: the cells lost would be the column's largest values, so the column is excluded."""
    cells = [f"{v:.2f}" for v in _amounts(4, 10, 900)]
    for i, text in zip([5, 50, 90][:n_sep], ["1,200", "2,450.50", "13,000"]):
        cells[i] = text
    by_name, _ = _roles(_via_csv(tmp_path, pd.DataFrame({"fee": cells, "y": _target()})))
    rec = by_name["fee"]
    assert (rec["role"], rec["reason"]) == ("excluded", "numeric_with_separators")
    assert rec["n_coerced"] == 0


def _european(v):
    whole, cents = int(v), int(round((v - int(v)) * 100))
    return f"{whole:,}".replace(",", ".") + f",{cents:02d}"


def test_european_and_short_groups_are_not_separators(tmp_path):
    amounts = _amounts(6, 1_000, 900_000)
    distinct = [_european(v) for v in amounts]
    sixty = [_european(v) for v in amounts[:72]] + [_european(v) for v in amounts[:48]]
    short = [f"{k},{int(v) % 100:02d}" for k, v in enumerate(amounts)]
    frame = pd.DataFrame({"eu_all": distinct, "eu_60": sixty, "short": short, "y": _target()})
    assert frame["eu_all"].nunique() == N and frame["short"].nunique() == N
    by_name, _ = _roles(_via_csv(tmp_path, frame))
    assert by_name["eu_all"]["reason"] == "id_like"
    assert by_name["eu_60"]["reason"] == "free_text"
    assert by_name["short"]["reason"] == "id_like"


def test_other_text_around_numbers_is_coerced(tmp_path):
    cells = [f"{v:.2f}" for v in _amounts(7, 10, 900)]
    odd = {8: "$1,200", 60: "1,200.", 99: "1,200 units"}
    for i, text in odd.items():
        cells[i] = text
    by_name, _ = _roles(_via_csv(tmp_path, pd.DataFrame({"v": cells, "y": _target()})))
    rec = by_name["v"]
    assert (rec["role"], rec["kind"]) == ("feature", "numeric")
    assert rec["n_coerced"] == len(odd)


def test_eighty_five_percent_separators_does_not_get_the_reason(tmp_path):
    amounts = _amounts(8, 1_000, 900_000)
    cells = [f"{v:,.2f}" for v in amounts]
    for i in range(0, N, 7)[:18]:
        cells[i] = ["pending", "tbc", "unknown"][i % 3]
    share = sum("," in c for c in cells) / N
    assert 0.8 < share < 0.88
    by_name, _ = _roles(_via_csv(tmp_path, pd.DataFrame({"v": cells, "y": _target()})))
    assert by_name["v"]["reason"] != "numeric_with_separators"


def test_separator_rule_takes_precedence(tmp_path):
    amounts = _amounts(9, 1_000, 900_000)
    frame = pd.DataFrame({
        "sep_60": [f"{v:,.2f}" for v in amounts[:72]] + [f"{v:,.2f}" for v in amounts[:48]],
        "sep_all": [f"{v:,.2f}" for v in amounts],
        "sep_five": [f"{v:,.2f}" for v in amounts[:5]] * (N // 5),
        "sep_const": ["1,200"] * N,
        "y": _target(),
    })
    by_name, _ = _roles(_via_csv(tmp_path, frame))
    for col in ["sep_60", "sep_all", "sep_five"]:
        assert by_name[col]["reason"] == "numeric_with_separators", col
    assert by_name["sep_const"]["reason"] == "constant"


def test_separator_cell_only_in_row_without_target_is_coerced(tmp_path):
    cells = [f"{v:.2f}" for v in _amounts(10, 10, 900)]
    cells[11] = "1,200"
    y = _target().astype(float)
    y[11] = np.nan
    loaded = _via_csv(tmp_path, pd.DataFrame({"v": cells, "y": y}))
    by_name, _ = _roles(loaded)
    rec = by_name["v"]
    assert (rec["role"], rec["kind"]) == ("feature", "numeric")
    assert rec["n_coerced"] == 1


# ------------------------------------------------------ counts and ids

def test_infinite_values_counted():
    x = np.random.default_rng(11).normal(size=N)
    x[[4, 30, 90]] = [np.inf, -np.inf, np.inf]
    by_name, _ = _roles(pd.DataFrame({"x": x, "y": _target()}))
    rec = by_name["x"]
    assert (rec["role"], rec["kind"]) == ("feature", "numeric")
    assert rec["n_infinite"] == 3
    assert rec["n_missing"] == 0


def test_counts_are_over_every_row_including_rows_without_target():
    rng = np.random.default_rng(12)
    x = rng.normal(size=N)
    y = _target().astype(float)
    no_target = [2, 8, 15, 33]
    y[no_target] = np.nan
    x[[2, 8, 50]] = np.nan          # two in rows with no target, one usable
    by_name, _ = _roles(pd.DataFrame({"x": x, "y": y}))
    assert by_name["x"]["n_missing"] == int(np.isnan(x).sum()) == 3


def test_roles_judged_on_usable_rows():
    y = _target().astype(float)
    c = np.full(N, 5.0)
    no_target = [1, 20, 41, 70]
    y[no_target] = np.nan
    c[no_target] = [1.0, 2.0, 3.0, 4.0]
    by_name, _ = _roles(pd.DataFrame({"c": c, "y": y}))
    assert (by_name["c"]["role"], by_name["c"]["reason"]) == ("excluded", "constant")


def test_id_like_rule():
    rng = np.random.default_rng(13)
    in_order = np.arange(1, N + 1)
    with_blanks = in_order.astype(float)
    with_blanks[[3, 40, 77, 100, 118]] = np.nan
    unordered = rng.choice(100_000, size=N, replace=False)
    assert not (np.all(np.diff(unordered) > 0) or np.all(np.diff(unordered) < 0))
    assert unordered.max() - unordered.min() + 1 != N
    frame = pd.DataFrame({
        "id_order": in_order,
        "id_shuffled": rng.permutation(in_order),
        "id_blanks": with_blanks,
        "id_gaps": np.cumsum(rng.integers(1, 6, size=N)),
        "measure_distinct": unordered,
        "measure_repeats": rng.integers(0, 50, size=N),
        "float_distinct": rng.normal(size=N),
        "y": _target(),
    })
    by_name, _ = _roles(frame)
    for col in ["id_order", "id_shuffled", "id_blanks", "id_gaps"]:
        assert (by_name[col]["role"], by_name[col]["reason"]) == ("excluded", "id_like"), col
    for col in ["measure_distinct", "measure_repeats", "float_distinct"]:
        assert (by_name[col]["role"], by_name[col]["kind"]) == ("feature", "numeric"), col
