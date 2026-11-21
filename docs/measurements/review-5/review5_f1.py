"""Review 5, finding 1: duplicates="group" on a table of few-valued columns. As built (sets formed on
features and target) against sets formed on the features alone (the build's own function, patched here)."""
import sys, warnings; sys.path.insert(0, "/home/claude/realdata/app6")
import numpy as np, pandas as pd, mlprofessor
from mlprofessor import pipeline, errors
warnings.simplefilter("ignore")
built = pipeline.copy_sets
def features_only(features, rows, y_rows):
    return built(features, rows, np.zeros(len(np.asarray(y_rows))))
def table(seed, signal, n=4000, shape=(2, 3, 4)):
    g = np.random.default_rng(seed)
    d = pd.DataFrame({f"c{i}": g.integers(0, k, n).astype(str) for i, k in enumerate(shape)})
    p = 0.5 + (0.3 * ((d.c0 == "1").astype(float) - 0.5) * 2 * 0.5 + 0.25 * (d.c1 == "2") - 0.2 * (d.c2 == "0") if signal else 0.0)
    d["y"] = (g.random(n) < np.clip(p, 0.02, 0.98)).astype(int); return d
def go(d, **kw):
    try:
        r = mlprofessor.profile(d, "y", profile="fast", n_jobs=2, **kw).to_dict(); h, v = r["headline"], r["validation"]
        lo, hi = h.get("skill_low"), h.get("skill_high")
        return f"{h['skill']:+.3f} " + (f"[{lo:+.2f},{hi:+.2f}]" if lo is not None else "[no interval]") + f" units {v.get('n_groups_training')}/{v.get('n_groups_holdout')}"
    except errors.ProfileError as e: return f"{e.code}"
for name, signal, shape in [("noise, 24 combinations", False, (2, 3, 4)), ("signal, 24 combinations", True, (2, 3, 4)),
                            ("signal, 3,125 combinations (five 5-level columns)", True, (5, 5, 5, 5, 5))]:
    print(name)
    for seed in (1, 2, 3):
        d = table(seed, signal, shape=shape)
        pipeline.copy_sets = built; a = go(d, duplicates="keep"); b = go(d, duplicates="group")
        pipeline.copy_sets = features_only; c = go(d, duplicates="group"); pipeline.copy_sets = built
        print(f"   seed {seed}: keep {a:38s} | group as built {b:38s} | group on features alone {c}")
tit = pd.read_csv("data/titanic.csv").drop(columns=["alive"])
sh = tit.copy(); sh["survived"] = np.random.default_rng(5).permutation(sh["survived"].to_numpy()); tw = pd.concat([sh, sh], ignore_index=True)
for name, d, y in [("titanic", tit, "survived"), ("titanic shuffled, every row twice", tw, "survived")]:
    d = d.rename(columns={y: "y"})
    pipeline.copy_sets = built; a = go(d, duplicates="keep"); b = go(d, duplicates="group")
    pipeline.copy_sets = features_only; c = go(d, duplicates="group"); pipeline.copy_sets = built
    print(f"{name}: keep {a} | group as built {b} | group on features alone {c}")
