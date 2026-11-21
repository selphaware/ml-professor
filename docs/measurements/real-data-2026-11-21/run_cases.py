"""Run the built mlprofessor on real public tables and print a compact summary per case."""
import json
import sys
import time
import traceback
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/claude/realdata/app")
import mlprofessor  # noqa: E402
from mlprofessor import errors  # noqa: E402

D = "/home/claude/realdata/data/"
RESULTS = []


def run(name, data, target, expect, **kw):
    kw.setdefault("profile", "fast")
    kw.setdefault("n_jobs", 2)
    t0 = time.time()
    rec = {"case": name, "expect": expect}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            res = mlprofessor.profile(data, target, **kw)
            d = res.to_dict()
            h = d["headline"]
            v = d["validation"]
            rec.update(
                outcome="ran",
                task=d["task"]["type"],
                scheme=v.get("scheme"),
                metric=h["metric"], score=h["score"], baseline=h["baseline"],
                baseline_score=h["baseline_score"], skill=h["skill"],
                skill_ci=[h.get("skill_low"), h.get("skill_high")],
                p=h.get("p_value"), model=h["model"],
                rows={r["item"]: r["count"] for r in d["rows"]},
                excluded={c["name"]: c["reason"] for c in d["columns"] if c["role"] == "excluded"},
                n_features=sum(1 for c in d["columns"] if c["role"] == "feature"),
                warnings=sorted({(w["code"], w["column"]) for w in d["warnings"]}),
                copies=[v.get("n_holdout_copies"), v.get("holdout_copy_share")],
                baselines=d.get("baselines"),
            )
            rec["_res"] = res
        except errors.ProfileError as exc:
            rec.update(outcome="named error", error=type(exc).__name__, code=exc.code, message=str(exc)[:300])
        except Exception as exc:  # a raw exception is itself a finding
            rec.update(outcome="RAW EXCEPTION", error=type(exc).__name__, message=str(exc)[:300],
                       trace=traceback.format_exc()[-600:])
    rec["seconds"] = round(time.time() - t0, 1)
    rec["py_warnings"] = sorted({f"{w.category.__name__}: {str(w.message)[:90]}" for w in caught})
    RESULTS.append(rec)
    show(rec)
    return rec


def show(r):
    print("=" * 100)
    print(f"{r['case']}   [{r['seconds']}s]   expect: {r['expect']}")
    if r["outcome"] == "ran":
        f = lambda x: "None" if x is None else f"{x:.4g}"  # noqa: E731
        print(f"  ran: {r['task']}, {r['scheme']}; {r['metric']} {f(r['score'])} vs {r['baseline']} {f(r['baseline_score'])}"
              f" -> skill {f(r['skill'])} [{f(r['skill_ci'][0])}, {f(r['skill_ci'][1])}], p {f(r['p'])}, model {r['model']}")
        print(f"  rows: {r['rows']}")
        print(f"  features {r['n_features']}; excluded: {r['excluded']}")
        print(f"  warnings: {r['warnings']}")
        if r["copies"][0]:
            print(f"  holdout copies: {r['copies']}")
    else:
        print(f"  {r['outcome']}: {r.get('error')} ({r.get('code')}): {r.get('message')}")
        if r.get("trace"):
            print(r["trace"])
    if r["py_warnings"]:
        print(f"  python warnings: {r['py_warnings']}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    import cases  # noqa: E402
    cases.main(run, D, which)
    slim = [{k: v for k, v in r.items() if k != "_res"} for r in RESULTS]
    json.dump(slim, open(f"/home/claude/realdata/out/results_{which}.json", "w"), indent=1, default=str)
