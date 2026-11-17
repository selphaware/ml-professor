"""CP3, part E: two CSV edges, in pandas, in the application and in the reference.

    python csv_edges.py app|ref

`app` is the package at the repo root, `ref` the judge's reference
implementation. For each crafted file: what `pandas.read_csv` does alone, and
what `stages.load_table` does. Files go under MLPROF_LAB_WORK, or `mlprof-lab`
in the system's temporary folder; nothing is written inside the repo.
"""
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
WORK = Path(os.environ.get("MLPROF_LAB_WORK") or Path(tempfile.gettempdir()) / "mlprof-lab") / "cp3"

if len(sys.argv) != 2 or sys.argv[1] not in ("app", "ref"):
    raise SystemExit(__doc__)
sys.dont_write_bytecode = True
sys.path.insert(0, str(REPO if sys.argv[1] == "app" else REPO / "verify" / "reference"))

import pandas as pd  # noqa: E402

from mlprofessor import errors, stages  # noqa: E402

ROWS = "".join(f"{i * 0.5:.1f},{i % 7},{i % 2}\n" for i in range(150))
LONG = "".join(f"{i * 0.25:.2f},{i % 11},{i % 2}\n" for i in range(20000))     # about 280 kB

BAD_BYTES = {
    "in the header line": b"a,b\xff,y\n" + ROWS.encode(),
    "in the first data row": b"a,b,y\n0.5,\xff1,1\n" + ROWS.encode(),
    "in the third data row": b"a,b,y\n" + ROWS.encode()[:16] + b"0.5,\xff1,1\n" + ROWS.encode()[16:],
    "past 200 kB": b"a,b,y\n" + LONG.encode() + b"0.5,\xff,1\n" + ROWS.encode(),
    "in the last cell": b"a,b,y\n" + ROWS.encode() + b"0.5,1,\xff",
    "Latin-1 text": b"a,b,y\n" + "0.5,café,1\n".encode("latin-1") + ROWS.encode(),
}
FIRST_LINES = {
    "empty": "", "spaces": "   ", "a tab": "\t", "spaces and tabs": " \t ",
    "a form feed": "\x0c", "a vertical tab": "\x0b", "a no-break space": " ",
    "an em space": " ", "two quotes": '""', "a quoted space": '" "',
    "commas": ",,", "spaces between commas": " , ,",
}


def by_pandas(path):
    try:
        frame = pd.read_csv(path, low_memory=False)
        return f"header {list(frame.columns)}, {len(frame)} rows"
    except Exception as exc:  # noqa: BLE001
        return type(exc).__name__


def by_tool(path):
    try:
        frame, info = stages.load_table(str(path))
        return (f"header {list(frame.columns)}, {len(frame)} rows, "
                f"{len(info['warnings'])} warning(s)")
    except errors.ProfileError as exc:
        return f"named error {exc.code}"
    except Exception as exc:  # noqa: BLE001
        return f"UNNAMED {type(exc).__name__}"


WORK.mkdir(parents=True, exist_ok=True)
path = WORK / "edge.csv"
print(f"{sys.argv[1]}: {Path(stages.__file__).parent}")
print("\nA byte that is not valid UTF-8")
for where, data in BAD_BYTES.items():
    path.write_bytes(data)
    print(f"  {where:<22} pandas: {by_pandas(path):<20} tool: {by_tool(path)}")
print("\nOne line before the header `a,b,y` and two data rows")
for what, line in FIRST_LINES.items():
    path.write_bytes((line + "\na,b,y\n1,2,0\n3,4,1\n").encode("utf-8"))
    print(f"  {what:<22} pandas: {by_pandas(path):<50} tool: {by_tool(path)}")
