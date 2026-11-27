"""Pass 9 (CP10) build: what the near-perfect warning and the rests-on list say on the
two-column carrier of review 7, on planets (a heavy-tailed target) and on the s1 fall of
diabetes. python cp10_extra.py app10|ref10"""
import sys, warnings
which = sys.argv[1]; sys.path.insert(0, f"/home/claude/realdata/{which}")
warnings.simplefilter("ignore")
import numpy as np, pandas as pd, mlprofessor
from sklearn.datasets import load_diabetes
NO = dict(time_col=False, group_col=False, n_jobs=2)
def show(label, d, y, **kw):
    r = mlprofessor.profile(d, y, **NO, **kw).to_dict(); h = r["headline"]
    print("=" * 90); print(label)
    print(f"  skill {h['skill']:.4f} [{h['skill_low']}, {h['skill_high']}] model {h['model']}; floor {h['rests_on_floor']}; rests_on {[(e['column'], round(e['skill_fall'], 3)) for e in h['rests_on']]}")
    print(f"  excluded {sorted(c['name'] for c in r['columns'] if c['role'] == 'excluded')}")
    for w in r["warnings"]:
        if w["code"] in ("near_perfect_headline", "strong_single_column", "suspected_leak", "largest_row_dominates", "extreme_value"):
            print(f"  >> {w['code']} [{w['column']}]: {w['message'][:330]}")
    falls = {c["name"]: (c.get("skill_fall"), c.get("shuffled_skill")) for c in r["columns"] if c["role"] == "feature"}
    return falls
db = load_diabetes(as_frame=True).frame
for prof in ("full", "fast"):
    e = db.copy(); z = np.random.default_rng(9).normal(0, e["target"].std(), len(e)); e["a"] = e["target"] + z; e["b"] = z
    f = show(f"diabetes + a = y + z, b = z ({prof})", e, "target", profile=prof)
    print("  falls of a and b:", {k: f[k] for k in ("a", "b")})
f = show("diabetes clean (fast): every fall", db, "target", profile="fast")
print("  ", {k: round(v[0], 3) for k, v in f.items()})
pl = pd.read_csv("/home/claude/realdata/data/planets.csv")
show("planets orbital period (fast)", pl, "orbital_period", duplicates="keep", profile="fast")
