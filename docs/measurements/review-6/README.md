# Review 6, findings 1 and 2: the lead's check

Run by the technical lead (the claude.ai Project) on 22 November 2026, on the
application at commit e917ba8, fast profile, Python 3.13 on Linux with the
pinned libraries. Not part of the gate; no score here is asserted anywhere.
`review6_check.py` builds its own tables; `output.txt` is its output.

- **Finding 1 is confirmed.** A day column written `YYYYMMDD`, three rows a
  day, the label random for each block of five days, both statements `False`.
  Clean: the column is excluded as a date and skill is +0.04, +0.00, -0.02.
  With one cell of 1,800 set to `99991231`, or to `0`: the column is a
  feature, skill is +0.99, +0.97, +0.98, and there is no warning. With 5% of
  the cells set to `99991231`: +0.92, +0.92, +0.95. So a share of 95%, as the
  text-date rule has, would not be enough on its own: an open-ended date
  column can hold its sentinel in far more than 5% of its rows.
- **Finding 2 is confirmed.** Noise in 16 feature combinations with
  `duplicates="group"`: -0.04, -0.05, -0.02. With a `status` column that
  copies the target added: `status` is excluded as a suspected leak, the
  run goes ahead, and skill is -0.79, -0.97, -0.26.
