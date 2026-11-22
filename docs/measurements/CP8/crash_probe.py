"""CP8: a date beyond the nanosecond range among ordinary dates (found by the
perturbation battery on 2026-11-22; spec section 9).

    python -B -X faulthandler crash_probe.py pandas
    python -B -X faulthandler crash_probe.py <package folder> [rows]

`pandas`: how pandas 3.1.1 holds an ISO 8601 text column with one cell of
9999-12-31, and a check of the range that needs no conversion.
<package folder>: the package `mlprofessor`, imported from its parent folder, run
on such a column as the time column, `rows` rows (1,500 by default: the size at
which the reference as it stood ended with an access violation; 400 rows raised
as they should). Run it in a process of its own: a build with the fault does not
raise, it ends the interpreter with exit code 3221225477.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
what = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
days = pd.date_range("2018-01-01", periods=n, freq="D").strftime("%Y-%m-%d").tolist()
days[7] = "9999-12-31"
rng = np.random.default_rng(0)

if what == "pandas":
    for label, values in (("with one cell of 9999-12-31", days), ("ordinary dates", days[8:])):
        parsed = pd.to_datetime(pd.Series(values), format="ISO8601")
        beyond = bool(((parsed < pd.Timestamp.min) | (parsed > pd.Timestamp.max)).any())
        print(f"{label}: parsed as {parsed.dtype}, latest {parsed.max()}; beyond the "
              f"nanosecond range, asked without converting: {beyond}")
else:
    sys.path.insert(0, str(Path(what).resolve().parent))
    import mlprofessor
    from mlprofessor import errors

    frame = pd.DataFrame({"date": days, "x": rng.normal(size=n), "y": rng.normal(size=n)})
    frame = frame.sample(frac=1, random_state=0).reset_index(drop=True)
    print(f"package: {Path(mlprofessor.__file__).resolve().parent}; {n} rows", flush=True)
    try:
        mlprofessor.profile(frame, "y", time_col="date", group_col=False, profile="fast",
                            n_jobs=1)
        print("the run finished: the date was ordered or dropped")
    except errors.ProfileError as exc:
        print(f"{exc.code}: {str(exc)[:200]}")
