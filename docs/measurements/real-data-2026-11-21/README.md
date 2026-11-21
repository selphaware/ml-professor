# Real-data check of the built library, 2026-11-21

Run by the technical lead (Claude, in the claude.ai Project), outside the
build loop, on commit `d14301e` of the application (after tasks T1 to T38).
Nothing here is a gate check and no score here is asserted anywhere: on real
data there is no known right score. Each case states what a correct tool
should conclude, and the result is read by eye.

**Environment:** Linux, Python 3.13, 2 cores, with pandas 3.1.1, numpy 2.5.4,
scikit-learn 1.9.1 and threadpoolctl 3.8.0. The owner's machine runs Python
3.12 on Windows, so timings and the last digits may differ there.

**Data, all public:**

| File | Source |
|---|---|
| `titanic.csv`, `diamonds.csv`, `penguins.csv` | github.com/mwaskom/seaborn-data |
| `stocks.csv`, `sp500.csv`, `seattle-weather.csv` | github.com/vega/vega-datasets |
| `vix-daily.csv` | github.com/datasets/finance-vix |
| breast cancer, diabetes | bundled with scikit-learn |
| `customers.csv` | this repo, `examples/` |

**Files:** `cases.py` (the cases and what each should conclude),
`run_cases.py` (the runner), `verify.py` (independent recomputation and the
near-copy measurement), `results.json` (every run's outcome).

## Outcome

32 runs: 26 finished, 6 stopped with a named error, none raised a raw
exception, and no library warning reached the caller.

Independent recomputation matched on every figure checked: Titanic ROC AUC
by scikit-learn (0.872433), `y_true` against the source rows, the VIX time
split (the holdout is exactly the latest rows), the previous-value baseline
(3.946753) and the model's squared error (3.854534). Results at `n_jobs` 1
and 2 were identical.

## What a reader should take from it

| Case | Result | Reading |
|---|---|---|
| Titanic, `alive` left in | `alive` excluded as a suspected leak | a real leak, caught |
| Titanic, target shuffled | skill 0.07, interval spans zero, p 0.21 | no signal found in noise |
| Breast cancer | skill 0.99, nothing flagged | a truly predictable table is not called a leak |
| VIX next-day change, time-ordered | skill 0.007 | honest: about zero |
| S&P 500 monthly next return | skill 0.03, interval spans zero | honest: about zero |
| VIX level, time-ordered | skill 0.023 against the previous value | honest: the level is persistence |
| **VIX level, time column not declared** | **skill 0.939 against the mean, with warnings** | **forty times the honest figure; a warning is the only signal** |
| **Stocks panel, price level** | **skill 0.979 against the mean, with a warning** | **a panel has no per-instrument baseline yet** |
| **Noise, every row twice, plus a batch column** | **skill 0.35 (0.44 with a 1e-9 change), no warning; minus 0.13 without the copies** | **fourth review, finding 2, reproduced on real features** |
| Titanic as shipped | stops: 37 of 223 holdout rows have a copy | the copies rule fires on a standard table; these are different passengers with the same recorded values (ROC AUC 0.888 on those rows, 0.854 on the rest) |
| `sp500.csv` dates like `Jan 1 2000` | stops, naming the time column | as designed |
| Penguins, three species | stops, task ambiguous | the message suggests declaring the task, which would not help |
