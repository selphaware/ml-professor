"""T18 guards: the block bootstrap and ties in the p-value (spec 3 step 12, 8.1, M4, M5).

Green today. A failure here means the application is wrong, not the test.
"""
import numpy as np
import pandas as pd

import cp3_helpers as c
import mlp_helpers as h


def test_m4_block_bootstrap_draws_whole_blocks():
    """M4: the block bootstrap draws whole blocks of consecutive holdout timestamps."""
    n = 256
    rng = np.random.default_rng(61)
    frame = pd.DataFrame({"t": np.arange(n, dtype=float), "x1": rng.normal(size=n)})
    frame["y"] = np.round(rng.normal(size=n) * 8.0) / 8.0 + 3.0
    target = frame["y"].to_numpy()
    values = target + rng.normal(size=n)
    values[192:] = target[192:] + (np.arange(64) % 2)
    # Squared errors on the holdout alternate 0, 1: any four consecutive
    # timestamps hold two of each, so every block resample's mean is 0.5.
    assert np.array_equal((values[192:] - target[192:]) ** 2, np.arange(64) % 2)

    log = h.Log()
    result = c.run(frame, families=[h.family("pick", log, h.LookupEstimator, values=values)],
                   time_col="t", task="regression")
    assert h.holdout_rows(result) == list(range(192, 256))
    head = result.headline
    assert head["interval_method"] == "bootstrap_blocks"
    assert head["block_length"] == 4
    assert head["score"] == 0.5
    assert head["n_resamples_used"] == 200
    assert abs(head["interval_low"] - 0.5) < 1e-12
    assert abs(head["interval_high"] - 0.5) < 1e-12


def test_m5_ties_count_in_the_p_value():
    """M5: a shuffle whose skill equals the observed skill counts in the p-value."""
    rng = np.random.default_rng(62)
    x1 = rng.normal(size=240)
    y = (rng.random(240) < 1.0 / (1.0 + np.exp(-2.5 * x1))).astype(int)
    frame = pd.DataFrame({"x1": x1, "y": y})

    log = h.Log()
    result = c.run(frame, families=[h.family("half", log, h.LookupEstimator,
                                             values=np.full(240, 0.5))])
    head = result.headline
    assert head["score"] == 0.5
    assert head["skill"] == 0.0
    assert head["n_permutations"] == 200
    assert head["p_value"] == 1.0
