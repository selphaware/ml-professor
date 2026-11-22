"""In which order does an implementation hand the rows of one fit to a model?

    python -B row_order.py <package parent> <seed> <table number> <data dir>

Runs the generated table through `profile()` with a recording family (the seam of
spec 2.6) and prints, for each fit, whether the index of X is in increasing
position order, in time order, or neither. Prototype evidence only.
"""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
parent, seed, number, data_dir = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import generated_tables as gt  # noqa: E402

sys.path.insert(0, parent)
import numpy as np  # noqa: E402
import mlprofessor  # noqa: E402
from mlprofessor import extensions  # noqa: E402

case = gt.make_table(seed, number)
data = str(gt.csv_path(data_dir, number)) if case["kind"] == "csv" else case["frame"]
t = case["oracle"]["t"]
seen = []


class Recorder:
    def fit(self, X, y):
        index = np.asarray(X.index)
        by_position = bool(np.all(index[1:] > index[:-1]))
        by_time = t is not None and bool(np.all(t[index][1:] >= t[index][:-1]))
        seen.append((len(index), by_position, by_time, list(X.columns)))
        self.mean = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean)

    def predict_proba(self, X):
        return np.column_stack([np.full(len(X), 0.5), np.full(len(X), 0.5)])


family = {"name": "rec", "grid": [{}], "make": lambda task, params, seed, n_jobs: Recorder()}
args = dict(case["args"])
with extensions.use(families=[family]):
    mlprofessor.profile(data, **args)
print(Path(mlprofessor.__file__).parent)
for rows, by_position, by_time, columns in seen:
    print(f"  fit on {rows} rows: index in increasing position: {by_position}; in time order: "
          f"{by_time}; columns {columns}")
