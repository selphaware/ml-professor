"""CP10: tables made for the plan's measurements. Each maker returns
(frame, target, the call's statements, info). Seeds: numpy.random.default_rng([tag, ..., seed])
with the tag given in each maker."""
import numpy as np
import pandas as pd

RANDOM = dict(time_col=False, group_col=False)


def _scheme(frame, scheme, rng):
    """Put the columns a scheme needs in front of `frame`, and give the statements."""
    n = len(frame)
    if scheme == "random":
        return frame, dict(RANDOM)
    if scheme == "grouped":
        who = np.array([f"E{i // 5:04d}" for i in range(n)])
        frame.insert(0, "who", who[rng.permutation(n)])
        return frame, dict(time_col=False, group_col="who")
    frame.insert(0, "t", np.arange(n))
    return frame, dict(time_col="t", group_col=False)


def _labels(codes, n_levels, rng, numeric=False, prefix="lv"):
    names = rng.permutation(n_levels)
    if numeric:
        return (100 + names[codes]).astype("float64")
    return np.array([f"{prefix}_{v:04d}" for v in names])[codes]


def honest_levels(seed, scheme="random", levels=24, r2=0.9, n=1500, numeric=False):
    """A column of `levels` levels that honestly explains `r2` of the target: each level
    has an effect of its own, the labels are in random order. Tag 10100."""
    rng = np.random.default_rng([10100, levels, round(1000 * r2), seed, int(numeric)])
    codes = rng.integers(0, levels, n)
    effect = rng.normal(size=levels)
    signal = effect[codes]
    signal = (signal - signal.mean()) / signal.std()
    noise = rng.normal(size=n)
    noise = (noise - noise.mean()) / noise.std() * np.sqrt((1 - r2) / r2)
    frame = pd.DataFrame({"level": _labels(codes, levels, rng, numeric), "n1": rng.normal(size=n),
                          "n2": rng.normal(size=n), "y": signal + noise})
    frame, call = _scheme(frame, scheme, rng)
    return frame, "y", call, {}


def noise_levels(seed, scheme="random", levels=60, n=1500, numeric=False, signal=True):
    """A column of noise with `levels` levels beside a target that two other columns
    explain (or nothing does). Tag 10200."""
    rng = np.random.default_rng([10200, levels, seed, int(numeric), int(signal)])
    f1, f2 = rng.normal(size=n), rng.normal(size=n)
    y = (2.0 * f1 - 1.2 * f2 if signal else 0.0) + rng.normal(size=n)
    frame = pd.DataFrame({"f1": f1, "f2": f2,
                          "level": _labels(rng.integers(0, levels, n), levels, rng, numeric),
                          "y": y})
    frame, call = _scheme(frame, scheme, rng)
    return frame, "y", call, {}


def few_rows_a_level(seed, scheme="random", r2=0.0, n=1500, numeric=False, classification=False):
    """A column with two to four rows a level: noise (`r2` 0) or honest. Tag 10300."""
    rng = np.random.default_rng([10300, round(1000 * r2), seed, int(numeric), int(classification)])
    sizes = []
    while sum(sizes) < n:
        sizes.append(int(rng.integers(2, 5)))
    codes = np.repeat(np.arange(len(sizes)), sizes)[:n]
    codes = codes[rng.permutation(n)]
    levels = len(sizes)
    effect = rng.normal(size=levels)
    noise = rng.normal(size=n)
    if r2 > 0:
        signal = effect[codes]
        signal = (signal - signal.mean()) / signal.std()
        y = signal + (noise - noise.mean()) / noise.std() * np.sqrt((1 - r2) / r2)
    else:
        y = noise
    if classification:
        y = np.where(y > np.median(y), "yes", "no")
    frame = pd.DataFrame({"level": _labels(codes, levels, rng, numeric), "n1": rng.normal(size=n),
                          "n2": rng.normal(size=n), "y": y})
    frame, call = _scheme(frame, scheme, rng)
    return frame, "y", call, {"levels": levels}


def honest_levels_clf(seed, scheme="random", levels=24, strength=2.0, n=1500, numeric=False):
    """Classification: each level has a log-odds of its own, sd `strength`. Tag 10400."""
    rng = np.random.default_rng([10400, levels, round(100 * strength), seed, int(numeric)])
    codes = rng.integers(0, levels, n)
    logit = strength * rng.normal(size=levels)[codes]
    y = np.where(rng.random(n) < 1 / (1 + np.exp(-logit)), "yes", "no")
    frame = pd.DataFrame({"level": _labels(codes, levels, rng, numeric), "n1": rng.normal(size=n),
                          "n2": rng.normal(size=n), "y": y})
    frame, call = _scheme(frame, scheme, rng)
    return frame, "y", call, {}


def heavy_series(seed, n=1500, kind="vol", cells=0, size=7.0, lags=True, where="any"):
    """One series under time order whose steps have heavy tails, with its own lags, and
    optionally a copy with `cells` training cells changed. Tag 10500.

    vol   a mean-reverting level with bursts of volatility (the shape of a volatility
          index): log level AR(1), the size of a step itself a slow random process
    t2    a random walk with steps of a t distribution with 2 degrees of freedom
    walk  a random walk with normal steps (nothing is marked)
    """
    rng = np.random.default_rng([10500, seed, {"vol": 0, "t2": 1, "walk": 2}[kind]])
    if kind == "vol":
        log_vol = np.zeros(n)
        x = np.zeros(n)
        for t in range(1, n):
            log_vol[t] = 0.97 * log_vol[t - 1] + 0.25 * rng.normal()
            jump = 0.35 * rng.exponential() if rng.random() < 0.02 else 0.0
            x[t] = 0.985 * x[t - 1] + 0.04 * np.exp(log_vol[t]) * rng.normal() + jump
        level = np.round(17 * np.exp(x), 2)
    elif kind == "t2":
        level = np.round(np.cumsum(rng.standard_t(2, n)), 4)
    else:
        level = np.round(np.cumsum(rng.normal(size=n)), 4)
    frame = pd.DataFrame({"t": np.arange(n), "y": level})
    if lags:
        for lag in (1, 2, 5):
            frame[f"lag{lag}"] = frame["y"].shift(lag)
        frame = frame.dropna().reset_index(drop=True)
    info = {}
    if cells:
        copy = frame["y"].to_numpy().copy()
        m = len(frame)
        # anywhere; or among the training rows: the first three quarters of the series,
        # less its start
        picks = rng.choice(np.arange(m // 6, int(0.74 * m)) if where == "train" else m, cells,
                           replace=False)
        step = float(np.median(np.abs(np.diff(frame["y"].to_numpy()))))
        if size == "other":
            copy[picks] = frame["y"].to_numpy()[rng.choice(m, cells)]
        elif isinstance(size, str) and size.startswith("x"):
            copy[picks] = copy[picks] * float(size[1:])
        elif isinstance(size, str) and size.startswith("+"):
            copy[picks] = copy[picks] + float(size[1:])
        else:
            copy[picks] = copy[picks] + float(size) * step
        frame.insert(len(frame.columns) - 0, "target_copy", copy)
        info = {"differ": [int(p) for p in picks], "step": step}
    cols = [c for c in frame.columns if c != "y"] + ["y"]
    return frame[cols], "y", dict(time_col="t", group_col=False), info


def skewed_reg(seed, n=1500, sigma=1.0, cells=0, size=7.0, scheme="random"):
    """A regression whose target has a long right tail (log-normal), two honest features,
    and optionally a rescaled copy with `cells` cells changed by `size` times the
    target's spread. Tag 10600."""
    rng = np.random.default_rng([10600, seed, round(100 * sigma)])
    f1, f2 = rng.normal(size=n), rng.normal(size=n)
    y = np.round(np.exp(sigma * (0.6 * f1 - 0.4 * f2 + 0.7 * rng.normal(size=n))) * 100, 2)
    frame = pd.DataFrame({"f1": f1, "f2": f2, "n1": rng.normal(size=n)})
    info = {}
    if cells:
        copy = 1.07 * y + 3
        picks = rng.choice(n, cells, replace=False)
        if size == "other":
            copy[picks] = 1.07 * y[rng.choice(n, cells)] + 3
        elif isinstance(size, str) and size.startswith("x"):
            copy[picks] = copy[picks] * float(size[1:])
        else:
            copy[picks] = copy[picks] + 1.07 * float(size) * float(np.std(y))
        frame["target_copy"] = copy
        info = {"differ": [int(p) for p in picks]}
    frame["y"] = y
    frame, call = _scheme(frame, scheme, rng)
    return frame, "y", call, info


def wild_targets(seed, n=1500, share=0.012, value=-999.0, scheme="random"):
    """Noise features and a noise target, with a share of the target cells set to one
    wild value: nothing here predicts anything, and no column may be excluded. Tag 10700."""
    rng = np.random.default_rng([10700, seed, round(10000 * share)])
    frame = pd.DataFrame({f"n{i + 1}": rng.normal(size=n) for i in range(4)})
    frame["r1"] = rng.normal(size=n).round(1)
    y = rng.normal(size=n)
    picks = rng.choice(n, max(1, round(share * n)), replace=False)
    y[picks] = value
    frame["y"] = y
    frame, call = _scheme(frame, scheme, rng)
    return frame, "y", call, {"wild": len(picks)}


def last_stamp(seed, n=1000, share=0.6, stamps=200):
    """Time order: `share` of the rows on the last timestamp, the others spread evenly
    over `stamps` earlier timestamps. Noise. Tag 10800."""
    rng = np.random.default_rng([10800, seed, round(1000 * share), stamps])
    late = round(share * n)
    t = np.r_[np.sort(rng.integers(0, stamps, n - late)), np.full(late, stamps)]
    # every earlier timestamp present, so the count of timestamps is as stated
    t[: min(stamps, n - late)] = np.arange(min(stamps, n - late))
    frame = pd.DataFrame({"t": np.sort(t), "n1": rng.normal(size=n), "n2": rng.normal(size=n),
                          "y": rng.normal(size=n)})
    return frame, "y", dict(time_col="t", group_col=False), {}


def even_stamps(seed, stamps=9, per=40):
    """Time order: `stamps` timestamps of `per` rows each. Noise. Tag 10900."""
    rng = np.random.default_rng([10900, seed, stamps, per])
    n = stamps * per
    frame = pd.DataFrame({"t": np.repeat(np.arange(stamps), per), "n1": rng.normal(size=n),
                          "n2": rng.normal(size=n), "y": rng.normal(size=n)})
    return frame, "y", dict(time_col="t", group_col=False), {}


def strong_small(seed, shape="independent", n=100, r2=0.95):
    """CP9's `order_strong.py` table: one legitimately strong feature under time order."""
    rng = np.random.default_rng([9900, n, round(1000 * r2), seed, 0 if shape == "independent" else 1])
    x = rng.normal(size=n) if shape == "independent" else np.cumsum(rng.normal(size=n))
    noise = rng.normal(size=n)
    noise *= np.std(x) * np.sqrt((1 - r2) / r2) / np.std(noise)
    frame = pd.DataFrame({"t": np.arange(n), "x": x, "n1": rng.normal(size=n),
                          "n2": rng.normal(size=n), "y": x + noise})
    return frame, "y", dict(time_col="t", group_col=False), {}


def running_total(seed, n=1500):
    """CP9's control: a series that only rises, with its own lag."""
    rng = np.random.default_rng([9800, seed])
    level = np.cumsum(rng.exponential(1.0, n))
    frame = pd.DataFrame({"t": np.arange(n), "level_prev": np.r_[0.0, level[:-1]], "level": level})
    return frame, "level", dict(time_col="t", group_col=False), {}


def monotone_levels(seed, n=1500, distinct=40, r2=0.2, scheme="random"):
    """An honest feature of `distinct` whole-number values on which the target's mean
    rises steadily, explaining `r2` of it: years of service against pay. Tag 11000."""
    rng = np.random.default_rng([11000, seed, n, distinct, round(1000 * r2)])
    x = rng.integers(1, distinct + 1, n).astype("float64")
    signal = (x - x.mean()) / x.std()
    y = signal + rng.normal(size=n) * np.sqrt((1 - r2) / r2)
    frame = pd.DataFrame({"x": x, "n1": rng.normal(size=n), "y": y})
    frame, call = _scheme(frame, scheme, rng)
    return frame, "y", call, {}


def diabetes(seed, variant="clean"):
    """scikit-learn's diabetes table, as it is or with a column added."""
    from sklearn.datasets import load_diabetes
    frame = load_diabetes(as_frame=True).frame.copy()
    g = np.random.default_rng(seed)
    if variant == "bands":
        band = pd.qcut(frame["target"], 24, labels=False, duplicates="drop")
        names = np.array([f"tier_{i:02d}" for i in g.permutation(int(band.max()) + 1)])
        frame.insert(0, "tier", names[band.to_numpy()])
    elif variant == "two":
        z = np.random.default_rng(9).normal(0, frame["target"].std(), len(frame))
        frame.insert(0, "a", frame["target"] + z)
        frame.insert(1, "b", z)
    return frame, "target", dict(RANDOM), {}


def customers(seed):
    """`examples/customers.csv`, the README's table."""
    import os
    path = os.path.join(os.environ.get("MLPROF_REPO", r"E:\CODE\ml-professor"), "examples",
                        "customers.csv")
    frame = pd.read_csv(path)
    return frame, "churned" if "churned" in frame.columns else frame.columns[-1], dict(RANDOM), {}


def curved(seed, copies=0, n=1500, scheme="random", negate=False):
    """A target that bends with `x1` and `x2`, so that a tree, a forest or boosting is
    chosen over the linear model; `copies` exact copies of `x1` beside it. Tag 11100."""
    rng = np.random.default_rng([11100, seed, n])
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    y = 2.0 * np.sin(2.5 * x1) + 0.8 * np.abs(x2) + 0.3 * rng.normal(size=n)
    frame = pd.DataFrame({"x1": x1})
    for c in range(copies):
        frame[f"x1_again{c + 1}"] = -x1 if negate else x1
    frame["x2"], frame["n1"], frame["y"] = x2, rng.normal(size=n), y
    frame, call = _scheme(frame, scheme, rng)
    return frame, "y", call, {}


def floor_noise(seed, n=150, kind="sparse", classification=False, share=0.5, columns=12):
    """Columns of noise read by their levels, built so that few validation rows have a
    level the fold's training rows hold; the target is noise too. Tag 11300.

    sparse   text labels drawn at random among 0.45 x n names: most levels hold one to
             three rows (random scheme)
    pairs    text labels of two rows each, and one row in three with a label of its own
    batches  time order: a label for each run of `w` consecutive rows (w drawn from 4
             to 12 for each column, with its own offset): only a batch that straddles
             the end of a fold's training rows is seen
    `share`: the share of the positive class of a classification.
    """
    rng = np.random.default_rng([11300, seed, n, ["sparse", "pairs", "batches"].index(kind),
                                 int(classification), round(100 * share)])
    frame = pd.DataFrame({"n1": rng.normal(size=n)})
    for c in range(columns):
        if kind == "sparse":
            labels = rng.integers(0, max(2, int(0.45 * n)), n)
        elif kind == "pairs":
            labels = np.repeat(np.arange(n), 2)[:n]
            alone = rng.random(n) < 1 / 3
            labels = np.where(alone, n + np.arange(n), labels)
            labels = labels[rng.permutation(n)]
            # keep it a categorical column: under half of its values distinct
            labels = np.where(rng.random(n) < 0.25, -1, labels)
        else:
            width = int(rng.integers(4, 13))
            labels = (np.arange(n) + int(rng.integers(0, width))) // width
        frame[f"c{c + 1:02d}"] = np.array([f"L{v}" for v in labels])
    if classification:
        frame["y"] = np.where(rng.random(n) < share, "yes", "no")
    else:
        frame["y"] = rng.normal(size=n)
    if kind == "batches":
        frame.insert(0, "t", np.arange(n))
        return frame, "y", dict(time_col="t", group_col=False), {}
    return frame, "y", dict(RANDOM), {}


def no_interval(seed, kind="grouped", classification=False, signal=False, carrier="none"):
    """A table whose headline has no interval: under the grouped scheme 20 entities of 25
    rows (about five in the holdout, fewer than 30), or under time order 100 timestamps
    of 5 rows (25 holdout timestamps, fewer than 30). Two features that carry the target
    when `signal`, two of noise; and a carrier of the target when asked: `code` (30
    equal-count bands as text, labels in random order; for a class, a code of 40 levels
    each of one class) or `two` (the target plus a noise, and that noise). Tag 11400."""
    rng = np.random.default_rng([11400, seed, ["grouped", "stamps"].index(kind),
                                 int(classification), int(signal),
                                 ["none", "code", "two"].index(carrier)])
    n = 500
    f1, f2 = rng.normal(size=n), rng.normal(size=n)
    z = (1.6 * f1 - 1.2 * f2) if signal else np.zeros(n)
    if kind == "grouped":
        who = np.repeat(np.arange(20), 25)
        z = z + (0.0 if signal else 0.0)
        y = z + rng.normal(size=n) + rng.normal(size=20)[who]
    else:
        who = None
        y = z + rng.normal(size=n)
    if classification:
        y = (y > np.median(y)).astype(int)
    frame = pd.DataFrame({"f1": f1, "f2": f2, "n1": rng.normal(size=n), "n2": rng.normal(size=n)})
    if carrier == "code":
        if classification:
            labels, half = rng.permutation(40), 20
            code = np.where(y == 1, labels[half + rng.integers(0, 20, n)], labels[rng.integers(0, 20, n)])
        else:
            ranks = np.argsort(np.argsort(y, kind="stable"), kind="stable")
            code = rng.permutation(30)[np.minimum(29, ranks * 30 // n)]
        frame["target_code"] = np.array([f"code_{int(v):03d}" for v in code])
    elif carrier == "two":
        noise = rng.normal(0, float(np.std(y)) or 1.0, size=n)
        frame["pair_a"], frame["pair_b"] = y + noise, noise
    frame["y"] = np.where(y == 1, "yes", "no") if classification else y
    if kind == "grouped":
        frame.insert(0, "who", np.array([f"E{v:02d}" for v in who])[rng.permutation(n)]
                     if False else np.array([f"E{v:02d}" for v in who]))
        return frame, "y", dict(time_col=False, group_col="who"), {}
    frame.insert(0, "t", np.repeat(np.arange(100), 5))
    return frame, "y", dict(time_col="t", group_col=False), {}
