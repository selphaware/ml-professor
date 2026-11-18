"""Shared helpers of the second pass (CP3). Builds on `mlp_helpers`; holds no test.

Nothing here imports `mlprofessor` at module level: each test imports the
application inside its own body.
"""
import json
import math
import re

import numpy as np

import mlp_helpers as h


# ------------------------------------------------------------ recording fakes

class KeepLog(h.Log):
    """Like `h.Log`, and also keeps what each call was handed.

    For a model: a float copy of the matrix (`values`). For a step: a deep copy
    of the frame (`frame`), since a step receives text columns too.
    """

    def add(self, owner, action, X, y=None):
        super().add(owner, action, X, y)
        event = self.events[-1]
        if owner.kind == "model":
            event["values"] = X.to_numpy(dtype=float).copy()
        else:
            event["frame"] = X.copy(deep=True)

    def calls(self, kind, action):
        return [ev for ev in self.events if ev["kind"] == kind and ev["action"] == action]


class MatrixEstimator(h.WrapEstimator):
    """Records rows, columns and values of every fit and predict (through a
    `KeepLog`) and otherwise behaves like `h.WrapEstimator`."""


class KeepStep(h.RecordingStep):
    """A step that changes nothing and keeps a copy of every frame it is handed
    (through a `KeepLog`)."""


def keep_step_factory(log):
    return lambda: KeepStep(log)


class FailOnCommand(h.LookupEstimator):
    """Predicts values the test chose, by source row position; fails on command.

    `fail_fit_rows`: raise from `fit` when the rows given are exactly this set.
    `fail_predict_rows`: raise from `predict` / `predict_proba` when the rows
    given include any of this set. The call is recorded before it raises.
    """

    def __init__(self, *args, fail_fit_rows=None, fail_predict_rows=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.fail_fit_rows = (None if fail_fit_rows is None
                              else frozenset(int(r) for r in fail_fit_rows))
        self.fail_predict_rows = (None if fail_predict_rows is None
                                  else frozenset(int(r) for r in fail_predict_rows))

    @staticmethod
    def message_for(family_name):
        return f"deliberate failure of family {family_name}"

    def _fail(self):
        raise RuntimeError(self.message_for(self.family))

    def fit(self, X, y):
        self.log.add(self, "fit", X, y)
        if self.fail_fit_rows is not None and {int(i) for i in X.index} == self.fail_fit_rows:
            self._fail()
        return self

    def _check_predict(self, X):
        if self.fail_predict_rows is not None and {int(i) for i in X.index} & self.fail_predict_rows:
            self._fail()

    def predict_proba(self, X):
        self.log.add(self, "predict", X)
        self._check_predict(X)
        return self._proba(self.values[np.asarray(X.index, dtype=int)])

    def predict(self, X):
        self.log.add(self, "predict", X)
        self._check_predict(X)
        return self.values[np.asarray(X.index, dtype=int)].copy()


# ------------------------------------------------------------------ running

def run(data, target="y", families=None, steps=None, seed=0, **kw):
    """`profile()` inside `extensions.use`, fast profile and one core by default."""
    import mlprofessor
    from mlprofessor import extensions

    args = dict(h.FAST)
    args.update(kw)
    with extensions.use(families=families, steps=steps):
        return mlprofessor.profile(data, target, seed=seed, **args)


def assert_strict_json(result):
    """The whole result serialises as JSON with no NaN or infinity."""
    json.dumps(h.plain(result.to_dict()), allow_nan=False)


def codes(result):
    return [code for code, _ in h.warnings_of(result)]


def count_code(result, code):
    return codes(result).count(code)


# ------------------------------------------------------- the console summary

_NUMBER = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")


def numbers_in(text):
    out = []
    for token in _NUMBER.findall(text):
        try:
            out.append(float(token))
        except ValueError:
            pass
    return out


def shown(value, text):
    """A value is shown when some decimal number in `text` is within
    0.0005 x max(1, |value|) of it (spec 2.2)."""
    value = float(value)
    assert math.isfinite(value)
    tolerance = 0.0005 * max(1.0, abs(value))
    return any(abs(number - value) <= tolerance for number in numbers_in(text))


def unmatched_warning_lines(warning_records, text):
    """Warnings that cannot each be given a line of their own.

    A line serves a warning when it holds the warning's `code` and, when it has
    one, its `column`. No line serves two warnings (a bipartite matching).
    Returns the records left without a line; empty means every warning has one.
    """
    lines = text.splitlines()
    candidates = []
    for record in warning_records:
        code, column = record["code"], record.get("column")
        candidates.append([
            i for i, line in enumerate(lines)
            if code in line and (column is None or str(column) in line)
        ])
    owner = {}

    def place(k, seen):
        for i in candidates[k]:
            if i in seen:
                continue
            seen.add(i)
            if i not in owner or place(owner[i], seen):
                owner[i] = k
                return True
        return False

    return [record for k, record in enumerate(warning_records) if not place(k, set())]
