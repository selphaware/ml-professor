"""CP10, item 6: what the statement of section A costs when a forest is the chosen
configuration, the costliest family to predict with.

    python -B forest_cost.py [--rows 100000] [--columns 100]

Fits the full profile's default forest (100 trees, no row cap) on the 75,000 training
rows of the CP3 timing table, and times: one prediction of the 25,000 holdout rows on
one thread; and 99 columns x 5 shuffles of them, 64 predictions side by side, each on
one thread (what the prototype does).
"""
import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, r"E:\CODE\ml-professor\docs\measurements\CP3")
import built_library                                    # noqa: E402
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--rows", type=int, default=100_000)
parser.add_argument("--columns", type=int, default=100)
args = parser.parse_args()
cores = os.cpu_count()
for task in ("classification", "regression"):
    frame = built_library.make_table(args.rows, args.columns, task)
    target = built_library.TARGETS[task]
    y = frame[target]
    x = frame.drop(columns=[target])
    x = np.column_stack([x[c].fillna(0).to_numpy(dtype=float) if x[c].dtype.kind in "fi"
                         else (x[c].to_numpy()[:, None] == np.array(list("abcdef"))).astype(float)
                         for c in x.columns])
    cut = int(0.75 * len(x))
    if task == "classification":
        model = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=cores,
                                       max_features="sqrt")
        yy = (y == "yes").to_numpy().astype(int)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=cores,
                                      max_features=1.0 / 3.0)
        yy = y.to_numpy(dtype=float)
    started = time.perf_counter()
    model.fit(x[:cut], yy[:cut])
    fit = time.perf_counter() - started
    model.set_params(n_jobs=1)
    held = x[cut:]
    predict = model.predict_proba if task == "classification" else model.predict
    started = time.perf_counter()
    predict(held)
    one = time.perf_counter() - started
    rng = np.random.default_rng(0)

    def shuffled(j):
        for _ in range(5):
            moved = held.copy()
            moved[:, j] = held[rng.permutation(len(held)), j]
            predict(moved)

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=cores) as pool:
        list(pool.map(shuffled, range(99)))
    side = time.perf_counter() - started
    print(f"{task}: forest of 100 trees fitted on {cut} rows x {x.shape[1]} columns in {fit:.0f} s; "
          f"one prediction of {len(held)} rows on one thread {one:.2f} s; 99 columns x 5 shuffles: "
          f"{495 * one:.0f} s one after another, {side:.1f} s at {cores} side by side", flush=True)
