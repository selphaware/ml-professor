# Review 5, finding 1: the lead's check

Run by the technical lead (the claude.ai Project) on 21 November 2026, on the
application at commit f3115f0, fast profile, Python 3.13 on Linux with the
pinned libraries. Not part of the gate; no score here is asserted anywhere.

`review5_f1.py` builds tables of few-valued columns (the fifth review's
case) and runs `duplicates="keep"`, `duplicates="group"` as built (sets of
rows equal in every feature and the target), and `duplicates="group"` with
the sets formed on the features alone (the build's own function, called with
a constant in place of the target). `output.txt` is its output.

- The finding is confirmed. On noise, as built: skill -0.28, -0.27, -0.56,
  where `keep` gives about 0. On signal: -0.08 on one seed of three.
- With the sets formed on the features alone, noise scores -0.07, +0.06,
  -0.04, and signal 0.41, 0.42, 0.63 against 0.51, 0.55, 0.46 with `keep`.
  With 24 combinations there are 24 units, so no interval is reported.
- Titanic: `keep` 0.745, as built 0.741, features alone 0.645. Keeping every
  identical-looking passenger on one side asks the model about combinations
  it has not seen, which is cautious and not the same question as `keep`.
- Titanic with the target shuffled and every row twice: 0.733 with `keep`,
  -0.154 as built, -0.108 on the features alone.
