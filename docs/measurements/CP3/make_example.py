"""CP3: writes the example table the README's commands use, `examples/customers.csv`.

    python make_example.py [output path]

Synthetic and fixed: the same bytes every time. 400 customers, one per sign-up
date, and whether each one left. Standard library only.

    customer_id     C0001 to C0400: an identifier, so the tool excludes it
    signup_date     2023-01-02 onwards, one row per day, year first: usable as --time-col
    plan            basic, standard or premium
    region          north, south, east or west: carries nothing
    monthly_spend   a number with two decimals; six cells are blank
    support_calls   a whole number, 0 to 9
    age             a whole number, 18 to 79: carries nothing
    churned         yes or no, the target; three cells are blank

Leaving is more likely with more support calls, on the basic plan and at a
lower spend. Nothing drifts over time, so the random and the time-ordered
scheme tell the same story.
"""
import csv
import datetime
import math
import random
import sys
from pathlib import Path

SEED = 2026
N_ROWS = 400
PLAN_EFFECT = {"basic": 0.6, "standard": 0.0, "premium": -0.7}
DEFAULT = Path(__file__).resolve().parents[3] / "examples" / "customers.csv"


def rows():
    rng = random.Random(SEED)
    start = datetime.date(2023, 1, 2)
    blank_spend = set(rng.sample(range(N_ROWS), 6))
    blank_target = set(rng.sample(range(N_ROWS), 3))
    for i in range(N_ROWS):
        plan = rng.choices(["basic", "standard", "premium"], weights=[5, 3, 2])[0]
        region = rng.choice(["north", "south", "east", "west"])
        spend = max(5.0, rng.gauss(52, 14) + {"basic": -12, "standard": 0, "premium": 18}[plan])
        calls = min(9, int(rng.expovariate(0.55)))
        age = rng.randint(18, 79)
        z = -1.1 + 0.6 * (calls - 1.5) + PLAN_EFFECT[plan] - 0.025 * (spend - 50)
        left = rng.random() < 1 / (1 + math.exp(-z))
        yield [f"C{i + 1:04d}", (start + datetime.timedelta(days=i)).isoformat(), plan, region,
               "" if i in blank_spend else f"{spend:.2f}", calls, age,
               "" if i in blank_target else ("yes" if left else "no")]


def main(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        out = csv.writer(fh, lineterminator="\n")
        out.writerow(["customer_id", "signup_date", "plan", "region", "monthly_spend",
                      "support_calls", "age", "churned"])
        out.writerows(rows())
    print(f"wrote {path}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT)
