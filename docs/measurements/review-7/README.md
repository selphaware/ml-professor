# Review 7, findings 1 and 2: the lead's check

Run by the technical lead (the claude.ai Project) on 25 November 2026, on the
application as built by pass 8, full profile, Python 3.13 on Linux with the
pinned libraries. Not part of the gate; no score here is asserted anywhere.
`review7_check.py` uses two public regression tables (scikit-learn's
diabetes, and seaborn's mpg); `output.txt` is its output.

- **Finding 1 is confirmed.** A text column of 24 bands of the target, with
  labels in random order, added to each table on three seeds: on diabetes
  the headline goes from 0.490 to 0.997 on all three, the column excluded on
  none, warned about on two and passed in silence on one. On mpg it goes
  from 0.856 to 0.993 in silence on two seeds and is excluded on the third.
- **Finding 2 is confirmed.** With `a = target + z` and `b = z` added, both
  tables report skill 1.000 and nothing names either column.
- **Seen beside them.** On diabetes the only warnings in the two-column run
  are the order reading's false alarms on `age` and `s6`
  (`..\CP9\build-on-real-tables\README.md`).
