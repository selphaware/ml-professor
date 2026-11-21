"""Shared helpers of the fifth pass (CP6). Builds on `mlp_helpers`,
`cp3_helpers` and `cp5_helpers`; holds no test.

Nothing here imports `mlprofessor` at module level: each test imports the
application inside its own body.
"""
import numpy as np
import pandas as pd

import cp3_helpers as c
import cp5_helpers as p
import mlp_helpers as h

GROUPED_CAVEAT = ("Random split of whole entities: this number is optimistic if the rows "
                  "are ordered in time.")

run = c.run
raised = p.raised
count_code = c.count_code


# ------------------------------------------------------------------ tables

def entity_frame(n_entities=60, sizes=5, seed=0, task="classification"):
    """Rows in entities: `who` (text), `m1` and `m2` alike within an entity, a target `y`
    that belongs to the entity. Rows in an order shuffled from `seed`."""
    rng = np.random.default_rng(seed)
    if np.ndim(sizes) == 0:
        sizes = [int(sizes)] * n_entities
    sizes = [int(s) for s in sizes]
    assert len(sizes) == n_entities
    names = [f"e{k:04d}" for k in range(n_entities)]
    centre1 = rng.normal(size=n_entities)
    centre2 = rng.normal(size=n_entities)
    if task == "classification":
        labels = np.array([0] * (n_entities // 2) + [1] * (n_entities - n_entities // 2))
        labels = labels[rng.permutation(n_entities)]
    else:
        labels = rng.normal(size=n_entities)
    who, m1, m2, y = [], [], [], []
    for k, size in enumerate(sizes):
        who.extend([names[k]] * size)
        m1.extend(centre1[k] + 0.05 * rng.normal(size=size))
        m2.extend(centre2[k] + 0.05 * rng.normal(size=size))
        if task == "classification":
            y.extend([int(labels[k])] * size)
        else:
            y.extend(labels[k] + 0.1 * rng.normal(size=size))
    frame = pd.DataFrame({"who": who, "m1": m1, "m2": m2, "y": y})
    order = rng.permutation(len(frame))
    return frame.iloc[order].reset_index(drop=True)


def walks(n_series=6, n_times=60, seed=0, start=0):
    """A panel of random walks on the timestamps `t` (`start`, `start + 1`, ... as floats):
    `who` (`s00`, ...), `prev` (the level one step before, same series) and `y` (the level).
    Each series starts at a level of its own, ten apart. Rows in shuffled order."""
    rng = np.random.default_rng(seed)
    parts = []
    for s in range(n_series):
        level = 10.0 * s + np.cumsum(rng.normal(size=n_times + 1))
        parts.append(pd.DataFrame({
            "t": np.arange(start, start + n_times, dtype=float),
            "who": f"s{s:02d}",
            "prev": level[:-1],
            "y": level[1:],
        }))
    frame = pd.concat(parts, ignore_index=True)
    order = rng.permutation(len(frame))
    return frame.iloc[order].reset_index(drop=True)


# ----------------------------------------------------------------- readers

def sides(result, frame, column="who"):
    """(entities of the training rows, entities of the holdout rows); no row was dropped."""
    rows = h.rows_of(result)
    for item in ("dropped_missing_target", "dropped_missing_time", "dropped_gap",
                 "dropped_row_cap"):
        assert rows[item] == 0, rows
    assert rows["loaded"] == len(frame)
    holdout = set(h.holdout_rows(result))
    names = frame[column].tolist()
    training = {names[r] for r in range(len(frame)) if r not in holdout}
    held = {names[r] for r in holdout}
    return training, held


def fits(log):
    """For an `h.Log`: `(fit rows, predicted rows)` of every model the run made, in order."""
    out = []
    for record in log.objects("model"):
        out.append((set(record["fit"]), set(record["predict"])))
    return out


def previous_by_hand(frame, gap=0, time="t", who="who", target="y"):
    """For each row: the target of the same entity's latest row at least `gap + 1`
    distinct timestamps earlier (over all rows of `frame`), or None."""
    stamps = sorted(set(frame[time].tolist()))
    rank = {value: k for k, value in enumerate(stamps)}
    ranks = [rank[v] for v in frame[time].tolist()]
    names = frame[who].tolist()
    values = frame[target].tolist()
    by_entity = {}
    for r, name in enumerate(names):
        by_entity.setdefault(name, []).append(r)
    out = []
    for r in range(len(frame)):
        limit = ranks[r] - (gap + 1)
        earlier = [q for q in by_entity[names[r]] if ranks[q] <= limit]
        if not earlier:
            out.append(None)
        else:
            out.append(values[max(earlier, key=lambda q: ranks[q])])
    return out
