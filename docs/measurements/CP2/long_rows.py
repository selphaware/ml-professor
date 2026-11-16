"""CP2: data rows longer than the header. What pandas does with them, what it
costs to find them, and whether anything else the gate measures moved.

    python -B long_rows.py forms   <reference>                    the table of the finding, and more
    python -B long_rows.py timing  <reference>                    100,000 rows x 100 columns
    python -B long_rows.py nat                                    ways to build an array of NaT
    python -B long_rows.py values  <gate> <reference> <out.json>  every gate run, every value
    python -B long_rows.py compare <before.json> <after.json>     what moved between two states

<reference> is the folder that holds the package `mlprofessor` to measure and
<gate> a gate script; `-` is the repo's own verify/reference or
verify/smoke_test.py. Files are written under MLPROF_LAB_WORK, or `mlprof-lab`
in the system's temporary folder; never inside the repo.
"""
import csv
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import time
import warnings
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
WORK = Path(os.environ.get("MLPROF_LAB_WORK") or Path(tempfile.gettempdir()) / "mlprof-lab") / "cp2"
CSV_FIELD_LIMIT = 2 ** 31 - 1       # csv refuses a cell over 131,072 characters unless told

BIG = "x" * 200_000                 # one cell, longer than csv's own limit

# name -> (file text, is some data row longer than the header)
FORMS = {
    # the four forms of the finding: every row one field longer than the header
    "first column uneven numbers (1, 4, 9)": ("a,b\n1,2,3\n4,5,6\n9,8,7\n", True),
    "first column text (x, y)": ("a,b\nx,2,3\ny,5,6\n", True),
    "first column evenly spaced (1, 4)": ("a,b\n1,2,3\n4,5,6\n", True),
    "first column row numbers (1, 2, 3, 4)": ("a,b\n1,2,3\n2,5,6\n3,8,9\n4,1,1\n", True),
    # which first columns come back as a RangeIndex
    "first column 1, 4, 7": ("a,b\n1,2,3\n4,5,6\n7,8,9\n", True),
    "first column 3, 2, 1": ("a,b\n3,2,3\n2,5,6\n1,8,9\n", True),
    "first column 10, 20, 30": ("a,b\n10,2,3\n20,5,6\n30,8,9\n", True),
    "first column 1, 2, 4": ("a,b\n1,2,3\n2,5,6\n4,8,9\n", True),
    "first column 5, 5, 5": ("a,b\n5,2,3\n5,5,6\n5,8,9\n", True),
    "a single row": ("a,b\n7,2,3\n", True),
    # more of the same fault
    "row numbers from 0": ("a,b\n0,2,3\n1,5,6\n2,8,9\n", True),
    "evenly spaced floats (0.5, 1.0, 1.5)": ("a,b\n0.5,2,3\n1.0,5,6\n1.5,8,9\n", True),
    "extra field last, and blank": ("a,b\n1,2,\n4,5,\n", True),
    "extra field in the middle (x, y)": ("a,b,c\n1,x,2,3\n4,y,5,6\n", True),
    "two extra fields": ("a,b\n1,2,3,4\n2,6,7,8\n", True),
    "only the first row is longer": ("a,b\n1,2,3\n4,5\n6,7\n", True),
    "first longer by one, a later row by two": ("a,b\n1,2,3\n2,5,6,7\n", True),
    "blank line, then longer rows": ("a,b\n\n1,2,3\n2,5,6\n", True),
    "line of spaces, then longer rows": ("a,b\n   \n1,2,3\n2,5,6\n", True),
    "line of a tab, then longer rows": ("a,b\n\t\n1,2,3\n2,5,6\n", True),
    "line of two quotes, then longer rows": ('a,b\n""\n1,2,3\n2,5,6\n', True),
    "line of commas, then longer rows": ("a,b\n,\n1,2,3\n2,5,6\n", True),
    "quoted comma in a longer row": ('a,b\n1,"x,y",3\n2,"z",6\n', True),
    "quoted line break in a longer row": ('a,b\n1,"x\ny",3\n2,"z",6\n', True),
    "one name, two fields a row": ("a\n1,2\n2,3\n", True),
    "a blank name, and longer rows": ("a,,c\n1,2,3,4\n2,6,7,8\n", True),
    "longer rows, lines ending CR LF": ("a,b\r\n1,2,3\r\n2,5,6\r\n", True),
    "longer rows, byte-order mark": ("﻿a,b\n1,2,3\n2,5,6\n", True),
    "longer rows, a cell of 200,000 characters": (f"a,b\n1,{BIG},3\n2,y,6\n", True),
    # the first row fits; a later one does not
    "second row longer": ("a,b\n1,2\n3,4,5\n6,7\n", True),
    "last row longer": ("a,b\n1,2\n3,4\n6,7,8\n", True),
    "last row longer by a blank field": ("a,b\n1,2\n3,4\n6,7,\n", True),
    "a short row, then a longer one": ("a,b\n5\n1,2,3\n", True),
    # no fault: these must load
    "well formed": ("a,b\n1,2\n3,4\n", False),
    "a short row": ("a,b\n1,2\n3\n5,6\n", False),
    "quoted comma, well formed": ('a,b\n1,"x,y"\n2,"z"\n', False),
    "blank lines, well formed": ("a,b\n\n1,2\n\n3,4\n", False),
    "line of commas, well formed": ("a,b\n1,2\n,\n3,4\n", False),
    "a cell of 200,000 characters, well formed": (f"a,b\n{BIG},2\ny,4\n", False),
}


def _or_repo(arg, *parts):
    return REPO.joinpath(*parts) if arg == "-" else Path(arg).resolve()


def reference(arg):
    """Import the package under `arg` and return its loader and its errors."""
    sys.path.insert(0, str(_or_repo(arg, "verify", "reference")))
    from mlprofessor import core, errors
    return core._load, errors


def _short(values, most=4, width=12):
    shown = [repr(v) if len(repr(v)) <= width else repr(v)[:width - 3] + "..." for v in values]
    return "[" + ", ".join(shown[:most]) + (", ..." if len(shown) > most else "") + "]"


def forms(arg):
    load, errors = reference(arg)
    WORK.mkdir(parents=True, exist_ok=True)
    print(f"pandas {pd.__version__}; reference at {Path(load.__code__.co_filename).parent}\n")
    print("| File | Long row | pandas: index | pandas: columns hold | Reference |")
    print("|---|---|---|---|---|")
    wrong = 0
    for i, (name, (text, long_row)) in enumerate(FORMS.items()):
        path = WORK / f"form_{i:02d}.csv"
        path.write_text(text, encoding="utf-8", newline="")
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                frame = pd.read_csv(path, low_memory=False)
            index = f"`{type(frame.index).__name__}` {_short(list(frame.index))}"
            holds = "; ".join(f"{c}={_short(frame[c].tolist())}" for c in frame.columns)
            if caught:
                holds += " (" + ", ".join(type(w.message).__name__ for w in caught) + ")"
        except Exception as exc:  # what pandas itself refuses
            index, holds = f"raises `{type(exc).__name__}`", str(exc).strip()[:70]
        try:
            load(str(path), [])
            outcome = "loads"
        except errors.ProfileError as exc:
            outcome = f"raises `{exc.code}`"
        except Exception as exc:  # not one of the tool's own errors
            outcome = f"raises `{type(exc).__name__}`, not a named error"
        right = outcome == ("raises `input_error`" if long_row else "loads")
        wrong += not right
        print(f"| {name} | {'yes' if long_row else 'no'} | {index} | {holds} | "
              f"{outcome}{'' if right else ' **WRONG**'} |")
    print(f"\n{wrong} of {len(FORMS)} files handled wrongly by the reference")


# ---------------------------------------------------------------- timing --

def first_data_row(path):
    """Method A: the field counts of the header and of the first data row."""
    limit = csv.field_size_limit(CSV_FIELD_LIMIT)
    try:
        with open(path, newline="", encoding="utf-8-sig") as fh:
            rows = csv.reader(fh)
            header = next((row for row in rows if row), [])
            # pandas skips a line that is empty or only white space; a line of commas is a row
            first = next((row for row in rows if len(row) > 1 or "".join(row).strip()), [])
    finally:
        csv.field_size_limit(limit)
    return len(header), len(first)


def count_every_row(path):
    """Method B: the field count of the header and the largest of any data row."""
    limit = csv.field_size_limit(CSV_FIELD_LIMIT)
    try:
        with open(path, newline="", encoding="utf-8-sig") as fh:
            rows = (len(row) for row in csv.reader(fh) if row)
            header = next(rows, 0)
            return header, max(rows, default=0)
    finally:
        csv.field_size_limit(limit)


def _best(fn, repeats=5):
    seconds = []
    for _ in range(repeats):
        started = time.perf_counter()
        fn()
        seconds.append(time.perf_counter() - started)
    return min(seconds), sorted(seconds)[len(seconds) // 2]


def big_file(n_rows, n_columns=100, long_rows=False, seed=7):
    """n_rows x n_columns: 89 numeric columns, 10 text, one yes/no target."""
    path = WORK / f"big_{n_rows}x{n_columns}_{'long' if long_rows else 'good'}.csv"
    if path.is_file():
        return path
    rng = np.random.default_rng(seed)
    n_text = n_columns // 10
    n_numeric = n_columns - n_text - 1
    frame = pd.DataFrame(np.round(rng.normal(size=(n_rows, n_numeric)), 4),
                         columns=[f"x{i + 1:02d}" for i in range(n_numeric)])
    levels = np.array(["north", "south", "east", "west", "a b", "c, d"])
    for i in range(n_text):
        frame[f"t{i + 1:02d}"] = levels[rng.integers(0, len(levels), n_rows)]
    frame["outcome"] = np.where(rng.random(n_rows) < 0.5, "yes", "no")
    WORK.mkdir(parents=True, exist_ok=True)
    # index=True, index_label=False: a row number leads each row and the header does not name it
    if long_rows:
        frame.index = frame.index + 1
        frame.to_csv(path, index=True, index_label=False, lineterminator="\n")
    else:
        frame.to_csv(path, index=False, lineterminator="\n")
    return path


def _loads(load, errors, path):
    try:
        load(str(path), [])
        return "loads"
    except errors.ProfileError as exc:
        return f"raises {exc.code}"


def timing(arg):
    load, errors = reference(arg)
    print(f"pandas {pd.__version__}; reference at {Path(load.__code__.co_filename).parent}")
    print("seconds, best and median of 5")
    for n_rows in (1_000, 10_000, 100_000):
        for label, long_rows in (("well formed", False), ("a row number leads every row", True)):
            path = big_file(n_rows, long_rows=long_rows)
            print(f"\n{n_rows:,} rows x 100 columns, {label}: {path.stat().st_size / 1e6:.1f} MB; "
                  f"fields (header, first data row) {first_data_row(path)}, "
                  f"(header, longest row) {count_every_row(path)}; "
                  f"the reference {_loads(load, errors, path)}")
            for name, fn in (("pandas.read_csv(path, low_memory=False)",
                              lambda: pd.read_csv(path, low_memory=False)),
                             ("A: read the header and the first data row", lambda: first_data_row(path)),
                             ("B: count the fields of every row", lambda: count_every_row(path)),
                             ("the reference's loader", lambda: _loads(load, errors, path))):
                best, median = _best(fn)
                print(f"  {name:<42} best {best:8.4f}   median {median:8.4f}")


# ------------------------------------------------------------------- NaT --

def nat():
    """Which ways of building an array of NaT warn, and whether they give the same array."""
    ways = {
        'np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")   as it was':
            lambda n: np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]"),
        'np.full(n, np.datetime64("NaT", "ns"), dtype="datetime64[ns]")   as it is':
            lambda n: np.full(n, np.datetime64("NaT", "ns"), dtype="datetime64[ns]"),
        'np.full(n, np.datetime64("NaT", "ns"))':
            lambda n: np.full(n, np.datetime64("NaT", "ns")),
        'np.full(n, "NaT", dtype="datetime64[ns]")':
            lambda n: np.full(n, "NaT", dtype="datetime64[ns]"),
    }
    print("numpy", np.__version__)
    first = None
    for text, make in ways.items():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = make(5)
        first = out if first is None else first
        same = out.dtype == first.dtype and out.tobytes() == first.tobytes() \
            and bool(np.isnat(out).all())
        said = "; ".join(f"{type(w.message).__name__}: {str(w.message)[:60]}"
                         for w in caught) or "no warning"
        print(f"{text}\n    dtype {out.dtype}, same bytes as the first: {same}, {said}")


# ------------------------------------------------- every value of the gate --

def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _without(root, value):
    """`value` with the work folder's path taken out of every string in it."""
    if isinstance(value, dict):
        return {k: _without(root, v) for k, v in value.items()}
    if isinstance(value, list):
        return [_without(root, v) for v in value]
    return value.replace(root, "<work>") if isinstance(value, str) else value


def values(gate_arg, ref_arg, out):
    """Run every check of a gate on a reference; keep all it recorded and all the tool wrote."""
    gate_file = _or_repo(gate_arg, "verify", "smoke_test.py")
    package = _or_repo(ref_arg, "verify", "reference") / "mlprofessor"
    sources = {p.name: _sha(p.read_bytes()) for p in sorted(package.glob("*.py"))}
    root = WORK / "gate" / _sha((_sha(gate_file.read_bytes()) + json.dumps(sources)).encode())[:12]
    (root / "verify").mkdir(parents=True, exist_ok=True)
    # gatekit takes the project root, and with it the artifacts folder, from sys.argv[0]
    sys.argv = [str(root / "verify" / "smoke_test.py")]
    sys.path.insert(0, str(REPO / "verify"))    # gatekit, for a gate script copied without it
    spec = importlib.util.spec_from_file_location("gate_under_test", str(gate_file))
    G = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(G)

    launched, launch = {}, G._launch

    def keep(ctx, name):
        launched[name] = launch(ctx, name)
        return launched[name]

    G._launch = keep
    started = time.perf_counter()
    results = G.gate.run(str(package))
    print(f"{len(results)} checks, {sum(not r['ok'] for r in results)} failed, "
          f"{len(launched)} runs, {time.perf_counter() - started:.0f} s; files under {root}")
    for r in results:
        if not r["ok"]:
            print(f"  FAIL {r['id']}: {r['message'][:200]}")

    runs = {}
    for name, result in launched.items():
        folder = Path(result["out"])
        doc, predictions = None, folder / "predictions.csv"
        if (folder / "profile.json").is_file():
            doc = json.loads((folder / "profile.json").read_text(encoding="utf-8"))
            doc.get("meta", {}).pop("timing", None)     # the one part that differs run to run
        stdout, mark = result["stdout"], "@@RESULT@@"
        if mark in stdout:      # the run through the Python function prints its result as JSON
            head, _, tail = stdout.partition(mark)
            printed = json.loads(tail)
            printed.get("to_dict", {}).get("meta", {}).pop("timing", None)
            stdout = {"before the mark": head, "result": printed}
        runs[name] = {
            "dataset": result["dataset"], "command": result["cmd"],
            "exit_code": result["rc"], "stdout": stdout, "stderr": result["stderr"],
            "files": sorted(p.name for p in folder.iterdir()),
            "profile": doc,
            "predictions_sha256": _sha(predictions.read_bytes()) if predictions.is_file() else None,
        }
    payload = {
        "state": {"gate": str(gate_file), "gate_sha256": _sha(gate_file.read_bytes()),
                  "reference": str(package), "reference_sha256": sources},
        "datasets": {name: {"seed": G.dataset_seed(name), "target": G.DATASETS[name][1],
                            "sha256": _sha(G.dataset_text(name).encode("utf-8"))}
                     for name in G.DATASETS},
        "checks": {r["id"]: {"ok": r["ok"], "message": r["message"], "catches": r["catches"]}
                   for r in results},
        "records": dict(G.gate.last_records),
        "runs": runs,
    }
    for part in ("checks", "records", "runs"):
        payload[part] = _without(str(root), payload[part])
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"wrote {out}")


def _leaves(node, path=""):
    """Every value in a nested structure, under its dotted path."""
    if isinstance(node, dict) and node:
        for key, value in node.items():
            yield from _leaves(value, f"{path}.{key}" if path else str(key))
    elif isinstance(node, list) and node:
        for i, value in enumerate(node):
            yield from _leaves(value, f"{path}[{i}]")
    else:
        yield path, node


def compare(before_file, after_file):
    """Every value of the state before, against the state after. Exit 1 if one moved."""
    before = json.loads(Path(before_file).read_text(encoding="utf-8"))
    after = json.loads(Path(after_file).read_text(encoding="utf-8"))
    for label, doc in (("before", before), ("after", after)):
        state = doc["state"]
        print(f"{label}: gate {state['gate_sha256'][:12]}, core.py "
              f"{state['reference_sha256']['core.py'][:12]}")
    bad = 0
    for part, what in (("datasets", "datasets (seed, target, bytes)"),
                       ("checks", "checks (verdict, message, failure modes named)"),
                       ("records", "values the gate recorded"),
                       ("runs", "runs (exit code, both streams, files, every field of "
                                "profile.json but meta.timing, bytes of predictions.csv)")):
        n_values, moved, gone, added = 0, [], [], []
        for name in before[part]:
            a = dict(_leaves(before[part][name]))
            b = dict(_leaves(after[part][name])) if name in after[part] else {}
            n_values += len(a)
            moved += [(f"{name}: {p}", a[p], b[p]) for p in a if p in b and a[p] != b[p]]
            gone += [(f"{name}: {p}", a[p], None) for p in a if p not in b]
            added += [(f"{name}: {p}", None, b[p]) for p in b if p not in a]
        bad += len(moved) + len(gone) + len(added)
        print(f"\n{what}: {len(before[part])} before, {len(after[part])} after")
        print(f"  {n_values} values before; {len(moved)} moved, {len(gone)} gone, "
              f"{len(added)} new under an existing name")
        for label, found in (("moved", moved), ("gone", gone), ("new", added)):
            for path, was, now in found[:20]:
                print(f"    {label}  {path}: {repr(was)[:200]} -> {repr(now)[:200]}")
        only_after = [name for name in after[part] if name not in before[part]]
        if only_after:
            print(f"  only after: {', '.join(only_after)}")
    print("\nNOTHING MOVED" if not bad else f"\n{bad} VALUES MOVED")
    return 1 if bad else 0


if __name__ == "__main__":
    command, args = sys.argv[1], sys.argv[2:]
    sys.exit({"forms": forms, "timing": timing, "nat": nat, "values": values,
              "compare": compare}[command](*(args or (["-"] if command in ("forms", "timing")
                                                    else []))))
