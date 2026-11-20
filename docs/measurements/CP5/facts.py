"""CP5: the pandas and numpy facts behind D50, D51 and D52 (measured, not assumed).

    python -B facts.py

Prints one line per fact: how read_csv loads whole numbers of each size, where
64-bit floats merge them, what an object column of numbers looks like to
pandas' type tests, how equal rows are compared, fractions of a second, and
dates beyond the nanosecond range. spec.md section 9 quotes from it. Reads and
writes no file.
"""
import decimal
import io
import math
import sys
import warnings

import numpy as np
import pandas as pd
from pandas.api import types as pt

D = decimal.Decimal
warnings.simplefilter("ignore")


def show(label, value):
    print(f"{label:<62} {value}")


def csv(text):
    return pd.read_csv(io.StringIO(text), low_memory=False)


print("-- how read_csv loads whole numbers")
for label, cells in [
    ("int64 max", ["9223372036854775807", "1"]),
    ("int64 max + 1 (fits uint64)", ["9223372036854775808", "1"]),
    ("uint64 max", ["18446744073709551615", "1"]),
    ("uint64 max + 1", ["18446744073709551616", "1"]),
    ("26 digits", ["12345678901234567890123456", "1"]),
    ("26 digits and a blank", ["12345678901234567890123456", "", "1"]),
    ("26 digits and a negative", ["12345678901234567890123456", "-5"]),
    ("above int64 and a negative", ["9223372036854775808", "-5"]),
    ("below int64 and small ones", ["-9223372036854775809", "5"]),
    ("above and below int64 together", ["9223372036854775808", "-9223372036854775809"]),
    ("2**53 + 1 and a blank", ["9007199254740993", "", "9007199254740992"]),
    ("2**53 + 1, no blank", ["9007199254740993", "9007199254740992"]),
    ("26 digits and text", ["12345678901234567890123456", "abc"]),
    ("1e400", ["1e400", "1.5"]),
]:
    frame = csv("a,b\n" + "\n".join(f"{c},0" for c in cells) + "\n")
    col = frame["a"]
    kinds = sorted({type(v).__name__ for v in col})
    show(label, f"dtype {col.dtype}; python types {kinds}; values {list(col)[:3]}")

print("-- floats merge whole numbers above 2**53")
ints = np.arange(2 ** 53 + 1, 2 ** 53 + 201, dtype="int64")
show("200 consecutive int64 from 2**53 + 1, as float64: distinct", len(np.unique(ints.astype("float64"))))
ns = 1_700_000_000_000_000_000 + 100 * np.arange(1500, dtype="int64")
show("1500 int64 nanosecond stamps 100 ns apart, as float64: distinct", len(np.unique(ns.astype("float64"))))
show("float64 spacing at 1.7e18", np.spacing(1.7e18))
u = np.array([2 ** 63 + 7 * k for k in range(50)], dtype="uint64")
show("50 distinct uint64 near 2**63, as float64: distinct", len(np.unique(u.astype("float64"))))
show("2**53 as float is exact; 2**53 + 1 is not", (float(2 ** 53) == 2 ** 53, float(2 ** 53 + 1) == 2 ** 53 + 1))

print("-- object columns as a database driver returns them")
show("pd.isna(Decimal('NaN'))", pd.isna(D("NaN")))
show("Series([Decimal('NaN'), Decimal(1)]).isna()", list(pd.Series([D("NaN"), D(1)], dtype=object).isna()))
show("float(Decimal('1E+400')), float(Decimal('-Infinity'))", (float(D("1E+400")), float(D("-Infinity"))))
show("float(Decimal('1E-400'))", float(D("1E-400")))
try:
    float(10 ** 400)
except OverflowError as exc:
    show("float(10**400)", f"OverflowError: {exc}")
show("hash(Decimal(1)) == hash(1) == hash(1.0)", hash(D(1)) == hash(1) == hash(1.0))
for label, values in [("ints and None", [1, 2, None]), ("bools and None", [True, False, None]),
                      ("Decimals", [D("1.5"), D("2.5")]), ("ints beyond 64 bits", [10 ** 25, 1]),
                      ("numpy ints in object", [np.int64(1), np.int64(2)])]:
    s = pd.Series(values, dtype=object)
    show(f"object {label}: is_numeric / is_bool / is_integer / is_string",
         (pt.is_numeric_dtype(s.dtype), pt.is_bool_dtype(s.dtype), pt.is_integer_dtype(s.dtype),
          pt.is_string_dtype(s.dtype)))
show("infer_dtype ints+None / bools+None / Decimals / floats+None",
     [pd.api.types.infer_dtype(pd.Series(v, dtype=object), skipna=True)
      for v in ([1, 2, None], [True, None], [D("1.5")], [1.5, None])])
try:
    np.array([10 ** 25, 1], dtype=object).astype("int64")
except OverflowError as exc:
    show("object array with 10**25 .astype('int64')", f"OverflowError: {str(exc)[:50]}")
try:
    show("object array [-5, 2**63] .astype('uint64')", np.array([-5, 2 ** 63], dtype=object).astype("uint64"))
except OverflowError as exc:
    show("object array [-5, 2**63] .astype('uint64')", f"OverflowError: {str(exc)[:50]}")
try:
    np.isinf(10 ** 25)
except TypeError as exc:
    show("np.isinf(10**25)", f"TypeError: {str(exc)[:60]}")

print("-- integer dtypes")
for name in ("int64", "uint64", "Int64", "UInt64"):
    s = pd.Series([1, 2, 3], dtype=name)
    show(f"{name}: is_integer / is_unsigned / kind", (pt.is_integer_dtype(s.dtype),
                                                      pt.is_unsigned_integer_dtype(s.dtype),
                                                      getattr(s.dtype, "kind", None)))
sp = pd.Series(pd.arrays.SparseArray(np.array([0, 2 ** 53 + 1, 0, 5], dtype="int64"), fill_value=0))
show("Sparse[int64]: is_integer / is_numeric / dtype", (pt.is_integer_dtype(sp.dtype), pt.is_numeric_dtype(sp.dtype), str(sp.dtype)))
show("Sparse[int64].to_numpy(dtype='int64')", sp.to_numpy(dtype="int64"))
n = pd.Series([2 ** 63 + 5, None, 7], dtype="UInt64")
show("UInt64 with NA .to_numpy(dtype='uint64', na_value=0)", n.to_numpy(dtype="uint64", na_value=0))
show("UInt64 with NA .to_numpy(dtype='float64', na_value=nan)", n.to_numpy(dtype="float64", na_value=np.nan))
show("np.unique on uint64 keeps values exact", np.unique(np.array([2 ** 64 - 1, 2 ** 64 - 2], dtype="uint64")))
show("int64 min/max diff overflows silently", np.diff(np.array([np.iinfo('int64').min, np.iinfo('int64').max], dtype="int64")))

print("-- equality of rows: missing cells and negative zero")
a = np.array([0.0, -0.0, np.nan, np.nan, 1.5])
show("np.unique([0.0, -0.0, nan, nan, 1.5], return_inverse)", np.unique(a, return_inverse=True))
show("pd.factorize of the same", pd.factorize(a))
show("DataFrame.duplicated on [0.0],[-0.0],[nan],[nan]", list(pd.DataFrame({"a": [0.0, -0.0, np.nan, np.nan]}).duplicated()))

print("-- ISO 8601 fractions")
for text in ("2024-01-05 09:15:00.5", "2024-01-05 09:15:00.123456", "2024-01-05T09:15:00.123456789",
             "2024-01-05 09:15:00.1234567891"):
    try:
        show(text, pd.to_datetime(pd.Series([text]), format="ISO8601").iloc[0].isoformat())
    except Exception as exc:  # noqa: BLE001
        show(text, f"{type(exc).__name__}: {str(exc)[:70]}")
stamp = pd.Timestamp("2024-01-05 09:15:00.123456")
show("how pandas writes a microsecond timestamp to CSV", pd.DataFrame({"t": [stamp]}).to_csv(index=False).split()[1:])

print("-- datetime columns beyond the nanosecond range")
far = pd.Series(np.array(["9999-12-31T23:59:59", "2020-01-01T00:00:00"], dtype="datetime64[s]"))
show("datetime64[s] year 9999: dtype", far.dtype)
try:
    far.to_numpy("datetime64[ns]")
except Exception as exc:  # noqa: BLE001
    show("  .to_numpy('datetime64[ns]')", f"{type(exc).__name__}: {str(exc)[:60]}")
print("-- what the reference relies on to read objects as typed columns")


def tried(label, fn):
    try:
        show(label, repr(fn()))
    except Exception as exc:  # noqa: BLE001
        show(label, f"RAISES {type(exc).__name__}: {str(exc)[:60]}")


tried("Int64 from [None, 5, 2**63-1, 2**53+1]",
      lambda: list(pd.array([None, 5, 2 ** 63 - 1, 2 ** 53 + 1], dtype="Int64")))
tried("UInt64 from [2**64-1, None, 2**63+5]",
      lambda: list(pd.array([2 ** 64 - 1, None, 2 ** 63 + 5], dtype="UInt64")))
tried("pd.isna(Decimal('sNaN'))", lambda: pd.isna(D("sNaN")))
tried("Series([Decimal('sNaN'), Decimal(1)]).isna()",
      lambda: list(pd.Series([D("sNaN"), D(1)], dtype=object).isna()))
tried("float(Decimal('sNaN'))", lambda: float(D("sNaN")))
tried("Decimal('1E+3') == its integral value; int() of it",
      lambda: (D("1E+3") == D("1E+3").to_integral_value(), int(D("1E+3"))))
nullable = pd.Series([True, None, False, True], dtype="boolean")
tried("(boolean == True).to_numpy() with a missing cell",
      lambda: (nullable == True).to_numpy())  # noqa: E712
tried("the same, .fillna(False).to_numpy(dtype=bool)",
      lambda: (nullable == True).fillna(False).to_numpy(dtype=bool))  # noqa: E712
ends = np.array([np.iinfo("int64").min, np.iinfo("int64").max], dtype="int64")
tried("int64 ends compared, not subtracted: x[1:] > x[:-1]", lambda: ends[1:] > ends[:-1])
tried("their range in Python integers", lambda: int(ends.max()) - int(ends.min()) + 1)
cplx = pd.Series(np.array([1 + 1j, 2 + 2j]))
tried("complex: is_numeric_dtype / is_complex_dtype",
      lambda: (pt.is_numeric_dtype(cplx.dtype), pt.is_complex_dtype(cplx.dtype)))
show("pandas, numpy", (pd.__version__, np.__version__))
