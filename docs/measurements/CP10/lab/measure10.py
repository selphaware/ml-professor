"""CP10 measurements on the prototype of the changed reference: 30 data seeds each.

    python -B measure10.py <set> <package folder> <out.json> [--seeds 30] [--workers 28]
                           [--env NAME=VALUE ...] [--old]

`--old` switches every proposed rule off: the reference as it stands.
"""
import argparse
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lab10                                            # noqa: E402

OLD = {"MLP_A": "0", "MLP_B": "none", "MLP_C": "old", "MLP_D": "old", "MLP_E": "0"}
# the regression runs of the gate, by scheme
REG = {"random": ["signal_reg", "noise_reg", "leak_reg"],
       "grouped": ["quarter_entity", "dominant_entity"],
       "time": ["random_walk", "int_stamps", "flat_tail", "panel"]}
CLF = {"random": ["signal_clf", "noise_clf", "imbalanced", "binary_numeric", "messy"],
       "grouped": ["entities"],
       "time": ["time_blocks"]}
REG_RUNS = [run for runs in REG.values() for run in runs]
CLF_RUNS = [run for runs in CLF.values() for run in runs]
SCHEMES = ("random", "grouped", "time")
SIZES = ("0.1", "1", "5", "20", "1000", "other", "x1.5")


def jobs_for(what, seeds):
    g = lab10.lab9.gate()
    ks = range(seeds)
    if what == "g10":         # the runs of the ninth pass's gate checks
        return [dict(run=run, k=k) for run in ("random_walk", "pair_together",
                                               "signal_reg_copied", "leak_reg_coded",
                                               "noise_tied", "skewed_copy", "last_stamp")
                for k in ks]
    if what == "gate":
        return [dict(run=run, k=k) for run in g.RUNS if run != "api_signal_clf" for k in ks]
    if what == "gate_fast":
        return [dict(run=run, k=k) for run in g.RUNS
                if run != "api_signal_clf" and g.RUNS[run][2] == "fast" for k in ks]
    if what == "a_two":
        runs = REG_RUNS + CLF_RUNS + ["walk_repeated", "full_signal_reg_j1", "full_signal_clf_j1",
                                      "full_noise_small"]
        return [dict(run=run, k=k, change="two") for run in runs for k in ks]
    if what == "a_near":      # A3: a column and another that carries the same information
        out = []
        for run in ("signal_reg", "signal_clf", "full_signal_reg_j1", "full_signal_clf_j1"):
            for change in (None, "dupnoise:f1:0.05", "dupnoise:f1:0.2"):
                out += [dict(run=run, k=k, change=change) for k in ks]
        return out
    if what == "a_dup":
        out = []
        for run, col in (("signal_reg", "f1"), ("full_signal_reg_j1", "f1"), ("signal_clf", "f1"),
                         ("full_signal_clf_j1", "f1")):
            for change in (None, f"dup:{col}", f"dup:{col}:2"):
                out += [dict(run=run, k=k, change=change) for k in ks]
        return out
    if what == "a_curved":
        return [dict(make="curved", k=k, params=dict(copies=copies, n=n), args={"profile": prof},
                     tag=prof)
                for prof, n in (("fast", 1500), ("full", 600)) for copies in (0, 1, 2) for k in ks]
    if what == "b_few":     # numeric codes of many levels: where "few" ends
        out = []
        for run in ("signal_reg", "noise_reg"):
            out += [dict(run=run, k=k, change=f"bands:{q}:num_random", tag="bands")
                    for q in (100, 150, 225, 300, 450) for k in ks]
        for run in ("signal_clf", "noise_clf"):
            out += [dict(run=run, k=k, change=f"codes:{q}:num", tag="codes")
                    for q in (100, 150, 224, 300, 450) for k in ks]
        for levels in (100, 150, 225, 300, 450):
            for r2 in (0.9, 0.99):
                out += [dict(make="honest_levels", k=k, tag="honest",
                             params=dict(scheme="random", levels=levels, r2=r2, numeric=True))
                        for k in ks]
            out += [dict(make="noise_levels", k=k, tag="noise",
                         params=dict(scheme="random", levels=levels, numeric=True)) for k in ks]
        return out
    if what == "a_two_full":
        return [dict(run=run, k=k, change="two", args={"profile": "full"})
                for run in ("random_walk", "int_stamps", "panel") for k in ks]
    if what == "b_floor":
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
    if what == "r2_copies":    # R2: copies as values, rescaled, relabelled
        out = []
        for run in ("signal_reg", "signal_clf", "full_signal_reg_j1", "full_signal_clf_j1"):
            for change in ("dup:f1", "dupscaled:f1", "duprelabel:region" if "reg" in run
                           else "duprelabel:seg"):
                out += [dict(run=run, k=k, change=change) for k in ks]
        for copies in (0, 1, 2):
            out += [dict(make="curved", k=k, params=dict(copies=copies, n=600),
                         args={"profile": "full"}, tag="full") for k in ks]
        return out
    if what == "r3_bands":     # R3: two to eight bands
        return [dict(run=run, k=k, change=f"bands:{q}:{kind}", tag=scheme)
                for scheme, runs in (("random", ["signal_reg", "noise_reg", "leak_reg"]),
                                     ("grouped", ["quarter_entity", "dominant_entity"]),
                                     ("time", ["random_walk", "int_stamps", "panel"]))
                for run in runs for q in (2, 3, 4, 5, 6, 7, 8)
                for kind in ("text_random", "num_random") for k in ks]
    if what == "g_cand":       # the candidate runs of the new gate checks
        out = []
        for change in ("bands:30:text_random", "bands:30:num_random",
                       "tied:levels:40+tied:round:1", "two", "dup:f1"):
            out += [dict(run="signal_reg", k=k, change=change) for k in ks]
        out += [dict(run="signal_clf", k=k, change="codes:40:text") for k in ks]
        out += [dict(make="skewed_reg", k=k, tag="random",
                     params=dict(scheme="random", sigma=1.0, cells=5, size=20.0)) for k in ks]
        out += [dict(make="last_stamp", k=k, params=dict(share=0.6)) for k in ks]
        return out
    if what == "d80_noint":
        return [dict(make="no_interval", k=k, tag=kind,
                     params=dict(kind=kind, classification=clf, signal=signal, carrier=carrier))
                for kind in ("grouped", "stamps") for clf in (False, True)
                for signal in (False, True) for carrier in ("none", "code", "two") for k in ks]
    if what == "d80_curved":
        return [dict(make="curved", k=k, params=dict(copies=c, n=600, negate=neg),
                     args={"profile": "full"}, tag="full")
                for c, neg in ((0, False), (1, False), (1, True)) for k in ks]
    if what == "d80_copies":
        out = []
        for run in ("signal_reg", "signal_clf", "full_signal_reg_j1", "full_signal_clf_j1"):
            for change in ("dup:f1", "dupscaled:f1", "dupneg:f1"):
                out += [dict(run=run, k=k, change=change) for k in ks]
            out += [dict(run=run, k=k) for k in ks]
        return out
    if what == "b_bands":
        return [dict(run=run, k=k, change=f"bands:{q}:{kind}", tag=scheme)
                for scheme, runs in REG.items() for run in runs
                for q in (8, 16, 24, 30, 40, 60)
                for kind in ("text_random", "text_ordered", "num_random") for k in ks]
    if what == "b_codes":
        return [dict(run=run, k=k, change=f"codes:{levels}:{kind}", tag=scheme)
                for scheme, runs in CLF.items() for run in runs
                for levels in (10, 40, 100) for kind in ("text", "num") for k in ks]
    if what == "b_honest":
        out = []
        for scheme in SCHEMES:
            for numeric in (False, True):
                for levels in (10, 24, 60):
                    for r2 in (0.5, 0.7, 0.9, 0.95, 0.97, 0.98, 0.99):
                        out += [dict(make="honest_levels", k=k, tag=scheme,
                                     params=dict(scheme=scheme, levels=levels, r2=r2, numeric=numeric))
                                for k in ks]
                    for strength in (1.0, 2.0, 4.0, 8.0):
                        out += [dict(make="honest_levels_clf", k=k, tag=scheme,
                                     params=dict(scheme=scheme, levels=levels, strength=strength,
                                                 numeric=numeric)) for k in ks]
                for levels in (10, 60, 300):
                    for signal in (True, False):
                        out += [dict(make="noise_levels", k=k, tag=scheme,
                                     params=dict(scheme=scheme, levels=levels, numeric=numeric,
                                                 signal=signal)) for k in ks]
                for r2 in (0.0, 0.5, 0.9, 0.99):
                    for classification in (False, True):
                        out += [dict(make="few_rows_a_level", k=k, tag=scheme,
                                     params=dict(scheme=scheme, r2=r2, numeric=numeric,
                                                 classification=classification)) for k in ks]
        return out
    if what == "c_tied":
        changes = [f"tied:levels:{d}" for d in (5, 10, 40, 100, 400)] + ["tied:round:1", "tied:round:2"]
        return [dict(run=run, k=k, change=change, tag=scheme)
                for scheme, runs in REG.items() for run in runs for change in changes for k in ks]
    if what == "c_warn":
        changes = ["copy_2pct_cells_differ", "copy_squared", "copy_exp", "copy_noise_5pct",
                   "round:2pct:40", "round:squared:40", "round:exp:40", "round:exact:40",
                   "round:2pct:100", "round:squared:100", "round:exp:100"]
        return [dict(run=run, k=k, change=change, tag=scheme)
                for scheme, runs in REG.items() for run in runs for change in changes for k in ks]
    if what == "c_small":
        out = []
        for shape in ("independent", "persistent"):
            for n in (100, 200):
                for r2 in (0.90, 0.95, 0.99):
                    out += [dict(make="strong_small", k=k, params=dict(shape=shape, n=n, r2=r2))
                            for k in ks]
        out += [dict(make="running_total", k=k, params={}) for k in ks]
        for scheme in SCHEMES:
            for distinct in (25, 40, 100):
                for r2 in (0.05, 0.2, 0.5, 0.8):
                    out += [dict(make="monotone_levels", k=k, tag=scheme,
                                 params=dict(scheme=scheme, distinct=distinct, r2=r2)) for k in ks]
        return out
    if what == "d_cells":
        return [dict(run=run, k=k, change=f"cells:{cells}:{size}", tag=scheme)
                for scheme, runs in REG.items() for run in runs
                for cells in (1, 5, 15) for size in SIZES for k in ks]
    if what == "d_heavy":
        out = []
        for kind in ("vol", "t2", "walk"):
            out += [dict(make="heavy_series", k=k, params=dict(kind=kind)) for k in ks]
            for cells in (1, 5, 15):
                for size in ("0.1", "1", "5", "20", "1000", "other", "x1.5", "+7", "+0.5"):
                    out += [dict(make="heavy_series", k=k,
                                 params=dict(kind=kind, cells=cells,
                                             size=size if size[0] in "ox+" else float(size)))
                            for k in ks]
        for scheme in SCHEMES:
            for sigma in (0.5, 1.0, 1.5):
                out += [dict(make="skewed_reg", k=k, tag=scheme, params=dict(scheme=scheme, sigma=sigma))
                        for k in ks]
                for cells in (1, 5, 15):
                    for size in ("0.1", "1", "5", "20", "1000", "other", "x1.5"):
                        out += [dict(make="skewed_reg", k=k, tag=scheme,
                                     params=dict(scheme=scheme, sigma=sigma, cells=cells,
                                                 size=size if size[0] in "ox" else float(size)))
                                for k in ks]
        return out
    if what == "d_lead":        # the lead's check: five cells among the training rows
        return [dict(make="heavy_series", k=k, params=dict(kind=kind, cells=5, size=size, where="train"))
                for kind in ("vol", "t2", "walk") for size in ("+7", "x1.5", "other", "+0.5")
                for k in ks]
    if what == "d_wild":
        out = []
        for scheme in ("random", "time", "grouped"):
            for share in (0.001, 0.005, 0.009, 0.0105, 0.012, 0.015, 0.02, 0.03, 0.05):
                for value in (-999.0, 999999.0, 30.0):
                    out += [dict(make="wild_targets", k=k, tag=scheme,
                                 params=dict(scheme=scheme, share=share, value=value)) for k in ks]
        return out
    if what == "d_defeat":      # CP9's a3b: target cells of -999 beside a rescaled copy
        return [dict(run=run, k=k, change="target_-999_and_copy") for run in REG_RUNS for k in ks]
    if what == "e_share":
        out = []
        for share in (0.2, 0.26, 0.3, 0.34, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7):
            out += [dict(make="last_stamp", k=k, params=dict(share=share)) for k in range(min(seeds, 5))]
        for stamps in range(7, 41):
            out.append(dict(make="even_stamps", k=0, params=dict(stamps=stamps, per=40)))
        return out
    if what == "real":
        out = [dict(make="customers", k=0, params={}),
               dict(make="customers", k=0, params={}, args={"profile": "full"})]
        for variant in ("clean", "bands", "two"):
            for prof in ("fast", "full"):
                out += [dict(make="diabetes", k=k, params=dict(variant=variant),
                             args={"profile": prof}) for k in (1, 2, 3)]
        return out
    raise SystemExit(f"unknown: {what}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("what")
    parser.add_argument("package")
    parser.add_argument("out")
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--workers", type=int, default=28)
    parser.add_argument("--env", action="append", default=[])
    parser.add_argument("--old", action="store_true")
    args = parser.parse_args()
    env = dict(OLD if args.old else {})
    for pair in args.env:
        name, _, value = pair.partition("=")
        env[name] = value
    jobs = jobs_for(args.what, args.seeds)
    started = time.time()
    print(f"{len(jobs)} jobs on {args.package} with {env}", flush=True)
    records = lab10.execute(args.package, jobs, env, workers=args.workers)
    lab10.save(args.out, records)
    print(f"wrote {args.out}: {sum('error' in r for r in records)} of {len(records)} ended in an "
          f"error; {time.time() - started:.0f} s")
