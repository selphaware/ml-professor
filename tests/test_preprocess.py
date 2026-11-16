"""T8: `stages.Preprocessor` (spec step 8, 'Preprocessing'; spec 2.7)."""
import numpy as np
import pandas as pd
import pytest

NAN = np.nan


def _affine(raw, out):
    """Recover (centre, scale) of out = (raw - centre) / scale from two rows."""
    scale = (raw[0] - raw[1]) / (out[0] - out[1])
    return raw[0] - scale * out[0], scale


def _fitted(kinds, frame):
    from mlprofessor import stages

    pre = stages.Preprocessor(kinds)
    assert pre.fit(frame) is pre
    return pre


def test_f1_statistics_come_from_the_fit_rows_only():
    """F1: the median, mean, deviation are those of the rows given to fit."""
    rows_a = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, NAN, 10.0]}, index=[0, 1, 2, 3, 4, 5])
    rows_b = pd.DataFrame({"a": [1e6, NAN, -5.0, 2.0]}, index=[10, 11, 12, 13])
    pre = _fitted({"a": "numeric"}, rows_a)

    out_a = pre.transform(rows_a)["a"].to_numpy(dtype=float)
    imputed_a = np.array([1.0, 2.0, 3.0, 4.0, 3.0, 10.0])        # median of A is 3
    centre, scale = _affine(imputed_a, out_a)
    # The spec does not say which degrees of freedom the deviation uses, nor whether
    # mean and deviation are taken before or after imputation; accept each reading,
    # all of them computed from A alone.
    observed_a = imputed_a[[0, 1, 2, 3, 5]]
    readings = [(vals.mean(), vals.std(ddof=d)) for vals in (imputed_a, observed_a) for d in (0, 1)]
    assert any(centre == pytest.approx(m) and scale == pytest.approx(s) for m, s in readings)
    assert np.allclose(out_a, (imputed_a - centre) / scale)

    out_b = pre.transform(rows_b)["a"].to_numpy(dtype=float)
    raw_b = np.array([1e6, 3.0, -5.0, 2.0])                      # B's missing cell gets A's median
    assert np.allclose(out_b, (raw_b - centre) / scale)

    # transforming B alongside A changes nothing for A
    both = pre.transform(pd.concat([rows_a, rows_b]))
    assert np.allclose(both.loc[rows_a.index, "a"].to_numpy(dtype=float), out_a)
    changed_b = rows_b.assign(a=[-3e7, 4.0, NAN, 9.0])
    both2 = pre.transform(pd.concat([rows_a, changed_b]))
    assert np.allclose(both2.loc[rows_a.index, "a"].to_numpy(dtype=float), out_a)


def test_f1_levels_come_from_the_fit_rows_only():
    """F1: a level seen only outside the fit rows gets no column."""
    rows_a = pd.DataFrame({"c": ["x", "y", "x", "z"]}, index=[0, 1, 2, 3])
    rows_b = pd.DataFrame({"c": ["w", "w", "y"]}, index=[7, 8, 9])
    pre = _fitted({"c": "categorical"}, rows_a)
    out_b = pre.transform(rows_b)
    assert list(out_b.columns) == ["c=x", "c=y", "c=z"]
    assert out_b.to_numpy(dtype=float).tolist() == [[0, 0, 0], [0, 0, 0], [0, 1, 0]]


def test_column_all_missing_in_fit_is_zero():
    rows_a = pd.DataFrame({"d": [NAN, NAN, NAN]}, index=[0, 1, 2])
    rows_b = pd.DataFrame({"d": [4.0, NAN]}, index=[5, 6])
    pre = _fitted({"d": "numeric"}, rows_a)
    assert pre.transform(rows_a)["d"].tolist() == [0.0, 0.0, 0.0]
    assert pre.transform(rows_b)["d"].tolist() == [4.0, 0.0]


def test_zero_deviation_read_as_one():
    rows_a = pd.DataFrame({"c": [5.0, 5.0, 5.0, 5.0]}, index=[0, 1, 2, 3])
    rows_b = pd.DataFrame({"c": [7.0, NAN, 2.5]}, index=[4, 5, 6])
    pre = _fitted({"c": "numeric"}, rows_a)
    assert pre.transform(rows_a)["c"].tolist() == [0.0, 0.0, 0.0, 0.0]
    assert pre.transform(rows_b)["c"].to_numpy(dtype=float).tolist() == [2.0, 0.0, -2.5]


def test_missing_categorical_is_a_level_and_unseen_is_zeros():
    rows_a = pd.DataFrame({"c": ["x", "y", None, "x"]}, index=[0, 1, 2, 3])
    rows_b = pd.DataFrame({"c": ["new", None, "(missing)", "y"]}, index=[4, 5, 6, 7])
    assert rows_b["c"].isna().tolist() == [False, True, False, False]
    pre = _fitted({"c": "categorical"}, rows_a)
    cols = sorted(["c=(missing)", "c=x", "c=y"])
    out_a = pre.transform(rows_a)
    assert list(out_a.columns) == cols
    assert out_a.to_numpy(dtype=float).tolist() == [[0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0]]
    out_b = pre.transform(rows_b)
    assert out_b.to_numpy(dtype=float).tolist() == [[0, 0, 0], [1, 0, 0], [1, 0, 0], [0, 0, 1]]


def test_at_most_thirty_most_frequent_levels():
    levels = [f"L{i:02d}" for i in range(35)]
    values = [lv for i, lv in enumerate(levels) for _ in range(40 - i)]
    rows = pd.DataFrame({"c": np.random.default_rng(0).permutation(values)})
    pre = _fitted({"c": "categorical"}, rows)
    out = pre.transform(rows.iloc[:5])
    assert list(out.columns) == [f"c={lv}" for lv in levels[:30]]


def test_level_ties_broken_by_text():
    levels = [f"K{i:02d}" for i in range(32)]
    rows = pd.DataFrame({"c": np.random.default_rng(1).permutation(levels * 2)})
    pre = _fitted({"c": "categorical"}, rows)
    out = pre.transform(rows.iloc[:3])
    assert list(out.columns) == [f"c={lv}" for lv in sorted(levels)[:30]]


def test_output_index_names_and_order():
    rows = pd.DataFrame({
        "n1": [1.0, 2.0, NAN, 4.0],
        "c1": ["b", "a", "b", None],
        "n2": [10.0, 20.0, 30.0, 40.0],
        "c2": ["q", "p", "p", "q"],
    }, index=[17, 3, 41, 8])
    kinds = {"n1": "numeric", "c1": "categorical", "n2": "numeric", "c2": "categorical"}
    out = _fitted(kinds, rows).transform(rows)
    assert list(out.index) == [17, 3, 41, 8]
    assert list(out.columns) == ["n1", "n2", "c1=(missing)", "c1=a", "c1=b", "c2=p", "c2=q"]
    dummies = out[["c1=(missing)", "c1=a", "c1=b", "c2=p", "c2=q"]].to_numpy(dtype=float)
    assert set(np.unique(dummies).tolist()) <= {0.0, 1.0}
    assert dummies.tolist() == [[0, 0, 1, 0, 1], [0, 1, 0, 1, 0], [0, 0, 1, 1, 0], [1, 0, 0, 0, 1]]
    assert not out.isna().any().any()
