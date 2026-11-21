import warnings, json, sys
import numpy as np, pandas as pd
import repeat_rules as rr
from real_tables import tables
warnings.simplefilter("ignore")
T = tables()
for k in ["titanic (891)", "diamonds (53,940)", "VIX daily OHLC (9,277)", "NYC taxis (6,389)"]:
    f, y, task = T[k]
    r = rr.count_repeats(f, y, task, seed=0)
    c = r["calibrated"]
    print("=" * 100); print(k, "| columns:", r["columns"] if not isinstance(r["columns"], dict) else {a: (len(b) if hasattr(b, "__len__") and not isinstance(b, str) else b) for a, b in r["columns"].items()})
    print("  stop", c["stop"], "count", c["count"], "share", round(c["share"], 4), "counted", c["counted"], "chance", round(c["chance"], 1), "spread", round(c["spread"], 2) if c["spread"] is not None else None, "level", c["level"], "exact_only", c["exact_only"])
    print("  settled:", c["settled"])
    print("  facts:", {a: (round(b, 3) if isinstance(b, float) else b) for a, b in c["facts"].items()} if isinstance(c["facts"], dict) else c["facts"])
    for lv in (c["levels"] if isinstance(c["levels"], list) else list(c["levels"].items())):
        print("   level:", lv)
