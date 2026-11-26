# CP10: the proofs of the change prompt, as run

Each was run alone, one after another, on 2026-11-26, with nothing else on the
machine. Commands from `E:\CODE\ml-professor`, with
`E:\CODE\ml-professor\mlprof_env\Scripts\python.exe -B` before each.
`<old>` is a copy of the repo taken before the first edit of the pass, kept
outside it; `<overlay>` the folder of the prototype tests, kept outside it.

| # | Proof | Command | File | Result |
|---|---|---|---|---|
| 1 | the gate's self-test | `verify/smoke_test.py --selftest` | `selftest.txt` | `SELFTEST: PASS`: 58 of 58 pairs of a check and its mode seen failing, over 44 broken copies; no check listed as never seen failing; 1,547 s |
| 2 | each new check fails on the unfixed reference and passes on the fixed one | `verify/smoke_test.py <old>\verify\reference\mlprofessor --only <the seven>`, then the same on `verify\reference\mlprofessor` | `new_checks_on_the_reference_as_it_stood.txt`, `new_checks_on_the_reference.txt` | 7 of 7 fail; 7 of 7 pass |
| 3 | no other gate value moved | `docs\measurements\CP2\long_rows.py values` on the gate and the reference before and after, then `compare`; `lab\values_moved.py` classifies every value of every run | `gate_values_compared.txt`, `gate_values_classified.txt` | the 35 older datasets, 53 older checks and 143 recorded values: none moved. Of the runs' own outputs, all that moved: 2,450 values that are the new keys of the statement; the console summary's new `rests on` line on 33 runs; the `near_perfect_headline` warning on `time_blocks_undeclared`. Nothing else |
| 4 | the mutation check against the existing suite | `verify/mutation_check.py --jobs 16` | `mutation_check_existing_suite.txt` | the good reference fails 5 of 2,119 cases, exactly the five the plan loosens under D59; every one of the 88 older variants is caught; of the 25 new ones, M52 is caught by an existing test and the other 24 by no test yet (the tests stage writes them). `MUTATION CHECK: FAIL`, as it must be before that stage |
| 5 | the prototypes of every new and changed test | the suite with `<overlay>` laid over it, on the application (`lab\red_green10.py`); `verify/mutation_check.py --jobs 16 --overlay <overlay>`; `lab\mutation_table10.py` | `prototypes_on_the_application.txt`, `mutation_check_with_prototypes.txt`, `mutation_before_after.txt` | on the application: 2,136 existing cases pass (the five changed files among them), the 76 new cases are 57 red and 19 green, exactly as the plan lists. The mutation check: `PASS`, the good reference failing nothing and each of 113 variants failing at least one test. No broken copy that the suite caught before has fewer failing cases after |
| 6 | both tools on the fixed reference, and on the application | `verify/battery.py <package> --json <f>`, `verify/generated_tables.py <package> --json <f>`, then `owned\owned.py` | `battery_on_the_reference.txt`, `generated_tables_on_the_reference.txt`, `battery_on_the_application.txt`, `generated_tables_on_the_application.txt`, `owned_lists_checked.txt` | the reference: battery `PASS` (holds 1,555, stated limit 7, open 0; carriers excluded 103, warned 2, named 20, listed 7, within 8, lowered 5, the D36 pair 1), generated tables `PASS` (1,200 agree). The application: 1,196 pairs and 704 tables fail, exactly `owned\owned_by_pass9.json`, the lists of `cp10_helpers` |
| 8 | the loop's doctor | `loop/run_loop.py doctor` | `doctor.txt` | `SELFTEST: PASS`; 14.1 s per edit, 16.6 s in full (budget 20); plan valid, 76 tasks; one warning, the uncommitted changes |
| 9 | the matrix and the build's dry run | `verify/smoke_test.py --matrix`; `loop/run_loop.py build --dry-run` | `matrix.txt`, `build_dry_run.txt` | 44 modes; T69 to T76 with their checks |

The gate on the application from the command line, three rounds:
`gate_timing_command_line.txt` (per edit 14.1 to 14.2 s, in full 24.3 to 25.3
s), and its last full run, `gate_on_the_application.txt`: the seven new checks
fail, as the plan lists, and every other check passes.

Proof 7, the measurements of A to E with their ranges, is the README of this
folder's parent and spec 10.11.
