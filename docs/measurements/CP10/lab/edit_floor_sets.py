"""Add the tables and the set that place the floor of R4 (D78 point 5): columns of noise
whose levels are seldom seen in a fold's training rows."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
p = HERE / "tables10.py"
t = p.read_text(encoding="utf-8")
if "def floor_noise" not in t:
    t += '''

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
'''
    p.write_text(t, encoding="utf-8", newline="\n")

p = HERE / "measure10.py"
t = p.read_text(encoding="utf-8")
if '"b_floor"' not in t:
    old = '''    if what == "b_bands":'''
    new = '''    if what == "b_floor":
        out = []
        for kind in ("sparse", "pairs", "batches"):
            for n in (100, 150, 300, 600):
                out += [dict(make="floor_noise", k=k, tag=kind,
                             params=dict(n=n, kind=kind, classification=False)) for k in ks]
                for share in (0.5, 0.2, 0.1):
                    out += [dict(make="floor_noise", k=k, tag=kind,
                                 params=dict(n=n, kind=kind, classification=True, share=share))
                            for k in ks]
        return out
    if what == "b_bands":'''
    assert t.count(old) == 1
    p.write_text(t.replace(old, new), encoding="utf-8", newline="\n")
print("ok")
