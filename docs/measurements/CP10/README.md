# CP10: the measurements behind the ninth pass (D77, D78, D80, D81)

The scripts and the values behind spec section 10.11, sections 7, 8, 8.2 and 11
and the plan of the pass (D35). Measured from 2026-11-25 to 2026-11-26 in the
harness session of CP10, on this machine (Threadripper PRO 5975WX, 64 threads,
Python 3.12.13, pandas 3.1.1, numpy 2.5.4, scikit-learn 1.9.1). Nothing here is
part of the tool, the gate or the suite. No script writes inside the repo: lab
copies and work files go to `MLPROF_LAB_WORK`, or to `mlprof-lab` in the
system's temporary folder, and the JSON a script writes goes to the path its
command names, outside the repo (`<out>` below). The JSON files themselves
are not kept (the largest is over 200 MB): each set's per-seed table is
(`per_seed\`), and every table below is the text a script printed (`reports\`).

Four packages are measured:

- **the application**: `mlprofessor` at the repo root, as built by the eighth
  pass (commit `54ad137`, unchanged at `f2e43c5`);
- **the reference**: `verify\reference\mlprofessor`, as changed by this pass;
- **the reference as it stood**: the same folder at `f2e43c5`, from a copy of
  the repo taken before the first edit of the pass and kept outside it
  (`<old reference>` below);
- **the prototype**: `prototype\mlprofessor` in this folder, the reference as
  it stood with every proposed rule written in behind a switch of `_proto.py`
  (one environment variable each, `MLP_...`), on which every rule was measured
  before the owner ruled on it. Its defaults are the rules adopted. It is the
  judge's material, kept so that the measurements can be made again: the
  guardrail on `verify\reference\` applies to it, and it is not a starting
  point for anything. Section 0 shows that it and the reference give the same
  output.

`judge\` holds the scripts that wrote the judge of the pass from the judge as
it stood: `make_reference10.py` and `make_variant10.py` (the reference and its
variants), `make_gate10.py` (the gate), `patch_battery10.py` then
`finish_battery10.py` (the battery), `patch_generated10.py` then
`finish_generated10.py` (the generated tables), `make_mutation10.py`,
`make_tasks10.py`, and `edit_spec10a.py`, `edit_spec10b.py`, `edit_spec10c.py`,
`edit_claude10.py`, `edit_skill10.py` with `fix_bands_wording.py` (the texts).
Each applies exact replacements and stops if one does not match exactly once.
The `edit_*` and `*_d78`, `*_d80` files are the steps by which the battery's
patch was brought up to each ruling.

Run each command from `E:\CODE\ml-professor`, with
`E:\CODE\ml-professor\mlprof_env\Scripts\python.exe -B` before it.

## Seeds

- **Data seed *k*** of a gate dataset: the gate's base seed 1115 + 100 × *k*,
  *k* = 0 to 29 (*k* = 0 is the gate's own data), each dataset at its own
  position, as `verify\smoke_test.py` makes them. The ninth pass's six
  datasets were added last, so no older one moved.
- **Tool seed**: 0, in every run.
- **A change to a gate table** (`lab10.py`'s changes, and CP9's `lab9.py`):
  drawn from `random.Random` seeded with the names of the run, the change and
  *k*, so every package sees the same table.
- **Tables made here** (`tables10.py`): `numpy.random.default_rng` seeded with
  the tag its docstring gives and the table's parameters and seed (tags 10100
  to 11400).
- **The battery** changes each gate table from `random.Random` seeded with the
  names of the run and the change. **The generated tables**: table *i* of seed
  20261122, tables 0 to 999 and 1,000,000 to 1,000,199.

## The scripts

All in `lab\`. `measure10.py` runs a named set (`<set>` below) at 30 data seeds
in a pool of processes that each import one package; every other script reads
the JSON it wrote.

| Script | What it does | Command (after `docs\measurements\CP10\lab\`) |
|---|---|---|
| `lab10.py`, `tables10.py` | the lab (runs, changes, fates) and the tables made for this pass | (imported) |
| `measure10.py` | one set: `gate`, `g10`, `a_two`, `a_two_full`, `a_dup`, `a_near`, `a_curved`, `r2_copies`, `d80_copies`, `d80_curved`, `d80_noint`, `b_bands`, `r3_bands`, `b_codes`, `b_honest`, `c_tied`, `c_warn`, `c_small`, `d_cells`, `d_heavy`, `d_lead`, `d_wild`, `e_share`, `real`, `g_cand` | `measure10.py <set> <package folder> <out.json> --workers 24` (`--env MLP_X=v` sets a prototype switch) |
| `same10.py` | two packages' outputs, record by record | `same10.py <one.json> <other.json> [--a-free]` |
| `per_seed10.py` | the per-seed table of a set | `per_seed10.py <set.json>` |
| `report_g10.py` | the seven new gate runs, the reference and each broken copy | `report_g10.py <folder of g10_<case>.json>` |
| `a_expected.py` | the shuffled score of a one-feature run against its exact mean (z) | `a_expected.py <package folder> --seeds 30` |
| `a_shuffles.py` | the spread over shuffles, from runs with 25 shuffles (`MLP_A_K=25`, prototype) | `a_shuffles.py <a_two_k25.json>` |
| `report_a_two.py` | A4: two columns that give the target together | `report_a_two.py <a_two.json> <a_two_full.json>` |
| `report_misc10.py` | two to eight bands; copies; a noisy copy | `report_misc10.py r3 <r3_bands.json>`, `copies <d80_copies.json> <r2_copies.json> <d80_curved.json> <gate.json>`, `near <a_near.json>` |
| `report_floor_ref.py` | the floor of the list, D78 against D80 | `report_floor_ref.py <gate.json>` |
| `report_floor80.py` | the stand-in for the floor, from the prototype's log (`MLP_DEBUG=1`) | `report_floor80.py <gate.json> <d80_noint.json>` |
| `report_carriers80.py` | every carrier pair of the battery: outcome, lift, both floors, under four readings | `report_carriers80.py <battery --json file>` |
| `report_b.py` | B: coded copies, codes of a class, honest columns | `report_b.py bands|codes|honest <json>` |
| `report_b_controls.py` | B under D78 point 5 against the first proposal | `report_b_controls.py <b_honest.json> <b_honest first proposal.json>` |
| `report_floor.py` | the floor of the reading by levels (`MLP_B_MIN=0`, prototype) | `report_floor.py <b_floor.json>` |
| `order_scale.py` | C: each form of the order reading as the table grows | `order_scale.py <package folder>` |
| `report_c.py` | C: what must still be warned about; the small tables; E | `report_c.py warn|small|e <json>` |
| `report_d.py`, `d_compact.py` | D: by the size of the change, before and after | `report_d.py cells old=<json> new=<json>`; `d_compact.py heavy|wild|lead old=<json> new=<json>` |
| `report_gate.py` | every gate run at 30 seeds, two packages | `report_gate.py <new gate.json> <old gate.json>` |
| `timing10.py`, `forest_cost.py` | the cost at 100,000 rows by 100 columns, full profile | `timing10.py <package folder>`; `forest_cost.py` |
| `suite_on.py` | the existing suite against a package, in a throwaway project | `suite_on.py <package folder> <work folder> --parts 6` |
| `make_proto.py`, `make_proto5.py`, `edit_*.py`, `_proto.py` | the prototype, from the reference as it stood | `make_proto5.py <old reference>\mlprofessor <out>\mlprofessor` |

## 0. The prototype and the reference give the same output

Every set below was run on the prototype (defaults, D80 and D81) and on the
reference as placed in the repo, and compared with `same10.py`: every field of
every record (error, skill, score, model, baselines, roles, rows, folds, the
selection, the intervals, warnings with their columns, and the statement: the
three keys of every column, the list and the floor). On the sets measured
before D80 the statement is left out (`--a-free`), since its floor and copies
were ruled after them; on the sets after D80 it is compared too.

| Set | Records | Fields that differ |
|---|---|---|
| `b_bands`, `b_codes`, `b_honest`, `r3_bands` | 4,860, 1,260, 8,460, 3,360 | none |
| `c_tied`, `c_warn`, `c_small` | 1,890, 2,970, 1,470 | none |
| `d_cells`, `d_heavy`, `d_lead`, `d_wild` | 5,670, 8,460, 360, 2,430 | none |
| `e_share`, `real`, `g_cand` | 84, 20, 240 | none |
| `a_two`, `a_two_full`, `a_dup`, `a_curved`, `r2_copies` | 600, 90, 360, 180, 450 | none |
| `d80_noint`, `d80_copies`, `d80_curved` (statement compared) | 720, 480, 90 | none |
| `gate`, the older runs (statement compared) | 1,260 | `flat_tail`'s floor: 0.05 in the prototype, null in the reference, which is what spec 5.2 pins where skill is null |

## A. What the headline rests on (step 12a)

- The seven new gate runs, 30 seeds, the reference and each broken copy:
  `reports\gate_runs_ninth.txt` (per seed: `per_seed\gate_runs_ninth_<case>.tsv`).
  Each check's condition holds on 30 of 30 seeds of the reference and fails on
  its broken copy on 30 of 30, but F32 on `leak_reg_coded`, which the tree
  happens to exclude on 3 of 30 (the check also asks for the `group_col`
  sentence, which F32 never writes).
- The shuffled score of the one feature of `random_walk` against its exact
  mean: `reports\shuffle_z_reference.txt`, `reports\shuffle_z_F29.txt`.
- The spread over shuffles: `reports\a_shuffles.txt` (prototype, 25 shuffles).
- Two columns that give the target together: `reports\a_two.txt`.
- Copies: `reports\a_copies.txt`; a noisy copy: `reports\a_near_copies.txt`.
- The floor: `reports\floor_ref.txt` (the reference), and the stand-in's
  candidates `reports\floor_standin_prototype.txt` (prototype, `MLP_DEBUG=1`).
- The carrier pairs: `reports\carrier_pairs.txt` (below).
- The cost: `reports\a_timing_100k.txt`, `reports\a_forest_cost.txt`.

### The carrier pairs of the battery (D80 point 2, D81)

`verify\battery.py` on the reference, `--json`, then `report_carriers80.py`. Of
146 pairs: excluded 103, warned about by name 2, named by the near-perfect
warning 20, named in `headline.rests_on` 7, within the unperturbed run's floor
8, lowered by more 5, and `flat_tail:two_columns_together`, the D36 stated
limit. Under the four readings of "within that run's floor": the pair's own
floor both ways, 7 fail; own floor, a rise only, 2; the unperturbed run's
floor both ways, 6; the unperturbed run's floor, a rise only (D81), none but
the D36 pair. Every pair, its outcome under D78 and under D80/D81, its lift
over the unperturbed run and both floors: `reports\carrier_pairs.txt`.

## B. The reading by levels (step 8)

- Coded copies of a regression target (8 to 60 bands, three forms, three
  schemes): `reports\b_bands.txt`; two to eight bands: `reports\b_two_to_eight_bands.txt`.
- Codes of a class: `reports\b_codes.txt`.
- Honest columns, noise, few rows a level, and where they meet:
  `reports\b_honest.txt`, and against the first proposal (unseen levels given
  the baseline): `reports\b_controls_unseen_not_scored.txt`.
- The floor of the reading: `reports\b_floor.txt` (no floor, `MLP_B_MIN=0`),
  `reports\b_floor_with_floor.txt`.
- The gate's clean columns and `examples\customers.csv`: `reports\gate_runs_30_seeds.txt`
  and `per_seed\real.tsv`.

## C. The order reading (step 8)

`reports\c_order_scale.txt` (each form as the table grows),
`reports\c_warn.txt`, `reports\c_small.txt`.

## D. A copy with a handful of cells changed (step 8)

`reports\d_cells.txt` (the gate's regressions, old against new),
`reports\d_heavy.txt` (long tails), `reports\d_wild.txt` (wild targets: no
other column excluded), `reports\d_lead.txt` (the lead's case). Without the
median baseline (the broken variant M63): `per_seed\d_wild_M63.tsv`, 186 of
2,430 tables exclude a noise column.

## E. A large time-ordered holdout (step 7)

`reports\e.txt`.

## Per-seed tables

`per_seed\<set>.tsv`, one line per tool run: the case, *k*, the skill, the chosen
configuration, the planted columns' role and fall, the list, the floor, and the
ninth pass's warnings with their columns. The reference's runs, but
`gate_runs_ninth_<case>.tsv` (each broken copy) and `d_wild_M63.tsv`.

## The proofs of the change prompt

See `output\README.md`.
