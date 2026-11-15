"""Example gate, built on gatekit. Copy the SHAPE, not the checks.

Project: a small report module, `app/report.py`, that loads `data/deals.csv`
and computes a total and a probability-weighted total.

Gate argument (the deliverable): the `app` folder.

Named silent failures this gate exists to catch:
  F1  the loader swallows a parse error and returns no rows, so every total is 0
  F2  the weighted total always equals the plain total, because the
      stage-probability lookup silently falls back to 1.0

Every check names the failure it catches. Every failure has a fixture that is
broken in exactly that way, and `--selftest` proves the right check fires.
The gate reads the source data itself and compares: it never trusts a number
the application hands it.
"""
import csv
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gatekit import Gate  # noqa: E402

gate = Gate("deals-report")

DATA = "data/deals.csv"
CSV = "deal_id,stage,value_gbp\n1,Proposal,100\n2,Negotiation,300\n3,Closed Won,50\n"


@gate.setup
def load_module(ctx):
    """Import the deliverable fresh. If it will not import, every check fails."""
    data = ctx.deliverable.parent / DATA
    if not data.is_file():
        ctx.harness_error(f"source data missing: {data}")
    with data.open(newline="", encoding="utf-8") as fh:
        ctx.state["source"] = list(csv.DictReader(fh))  # the gate's own reading
    ctx.state["data_path"] = data
    spec = importlib.util.spec_from_file_location("report_under_test",
                                                  ctx.deliverable / "report.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ctx.state["report"] = module


@gate.check("load.rows", catches=["F1"],
            doc="load() returns one row per data line in the CSV")
def load_rows(ctx):
    rows = ctx.state["report"].load(ctx.state["data_path"])
    want = len(ctx.state["source"])
    ctx.require(len(rows) == want,
                f"load() returned {len(rows)} rows, the CSV has {want}: check the "
                "loader is not swallowing a parse error")


@gate.check("total.reconciles", catches=["F1"],
            doc="total() equals the sum of value_gbp computed directly from the CSV")
def total_reconciles(ctx):
    report = ctx.state["report"]
    got = report.total(report.load(ctx.state["data_path"]))
    want = sum(float(r["value_gbp"]) for r in ctx.state["source"])
    ctx.require(abs(got - want) < 0.005, f"total() is {got}, the CSV sums to {want}")


@gate.check("weighted.differs", catches=["F2"],
            doc="weighted_total() is strictly below total() when open deals exist")
def weighted_differs(ctx):
    report = ctx.state["report"]
    rows = report.load(ctx.state["data_path"])
    total, weighted = report.total(rows), report.weighted_total(rows)
    ctx.require(0 < weighted < total,
                f"weighted_total() is {weighted} and total() is {total}: the "
                "stage-probability lookup is not being applied")


# ------------------------------------------------------------- fixtures --

GOOD = '''
import csv
PROB = {"Proposal": 0.5, "Negotiation": 0.75, "Closed Won": 1.0}
def load(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return [dict(r, value_gbp=float(r["value_gbp"])) for r in csv.DictReader(fh)]
def total(rows):
    return sum(r["value_gbp"] for r in rows)
def weighted_total(rows):
    return sum(r["value_gbp"] * PROB[r["stage"]] for r in rows)
'''


def _write(tmp: Path, source: str) -> Path:
    (tmp / "app").mkdir(parents=True, exist_ok=True)
    (tmp / "data").mkdir(exist_ok=True)
    (tmp / "app" / "report.py").write_text(source, encoding="utf-8")
    (tmp / DATA).write_text(CSV, encoding="utf-8")
    return tmp / "app"


@gate.good_fixture
def good(tmp):
    return _write(tmp, GOOD)


@gate.failure_mode("F1", "loader swallows a parse error and returns no rows")
def broken_empty_loader(tmp):
    return _write(tmp, GOOD.replace(
        "return [dict(r, value_gbp=float(r[\"value_gbp\"])) for r in csv.DictReader(fh)]",
        "return []"))


@gate.failure_mode("F2", "probability lookup falls back to 1.0, weighted == total")
def broken_dead_lookup(tmp):
    return _write(tmp, GOOD.replace('PROB[r["stage"]]', 'PROB.get(r["stage"] + "?", 1.0)'))


if __name__ == "__main__":
    gate.main()
