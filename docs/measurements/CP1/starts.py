"""Start N tool processes at the same instant, for R rounds; say which rounds blocked (D32).

    python docs/measurements/CP1/starts.py <n> <rounds>

Each process is the reference implementation's fast profile on signal_clf, started
the way the gate starts a run. A barrier releases N threads together and each calls
Popen once, so the starts fall within about a millisecond of each other. A round is
called blocked when any process takes 20 seconds or more: an unblocked run takes
about 3 seconds, and the stall measured before D32 was 35 or 70 seconds.

Run it on a quiet machine: any other Python process started meanwhile (a hook, a
gate run) is one more simultaneous start. Nothing is written inside the repo.
"""
import sys
import os

sys.dont_write_bytecode = True

import subprocess
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lab  # noqa: E402

BLOCKED_SECONDS = 20.0


def main():
    n, rounds = int(sys.argv[1]), int(sys.argv[2])
    G = lab.load_gate("-")
    data = lab.write_table("starts", "signal_clf", G.dataset_text("signal_clf"))
    env = dict(os.environ)
    for name in ("PYTHONPATH", "PYTHONSTARTUP", "PYTHONHOME"):
        env.pop(name, None)
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    print(f"python {sys.version.split()[0]}, base {sys.base_prefix}", flush=True)
    blocked, all_times = 0, []
    for r in range(1, rounds + 1):
        barrier = threading.Barrier(n)
        started, procs = [None] * n, [None] * n

        def launch(i):
            out = lab.OUT / "starts" / f"p{i}"
            cmd = [sys.executable, "-m", "mlprofessor", "profile", "--data", str(data),
                   "--target", "outcome", "--out", str(out), "--profile", "fast", "--seed", "0",
                   "--n-jobs", "1"]
            barrier.wait()
            started[i] = time.perf_counter()
            procs[i] = subprocess.Popen(cmd, cwd=str(lab.REFERENCE_PARENT), env=env,
                                        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

        threads = [threading.Thread(target=launch, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        spread = (max(started) - min(started)) * 1000
        took, codes = [None] * n, [None] * n
        while any(t is None for t in took):
            for i, p in enumerate(procs):
                if took[i] is None and p.poll() is not None:
                    took[i] = time.perf_counter() - started[i]
                    codes[i] = p.returncode
            time.sleep(0.02)
        errors = [p.stderr.read().decode("utf-8", "replace").strip()[-200:] for p in procs]
        is_blocked = max(took) >= BLOCKED_SECONDS
        blocked += is_blocked
        all_times += took
        print(f"round {r:>2}: starts within {spread:6.1f} ms; run times {min(took):5.1f} to "
              f"{max(took):5.1f} s; exit codes {sorted(set(codes))}"
              f"{'  BLOCKED' if is_blocked else ''}", flush=True)
        if any(codes):
            print("   stderr:", [e for e in errors if e][:2], flush=True)
        time.sleep(1.0)
    print(f"RESULT: {blocked} of {rounds} rounds of {n} simultaneous starts blocked "
          f"(threshold {BLOCKED_SECONDS:.0f} s). Run times over all processes: "
          f"{min(all_times):.1f} to {max(all_times):.1f} s, mean "
          f"{sum(all_times) / len(all_times):.1f} s", flush=True)


if __name__ == "__main__":
    main()
