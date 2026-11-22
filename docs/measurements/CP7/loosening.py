"""CP7: the existing test files as the tests stage is to loosen them (D59, D65).

    python -B loosening.py <folder>     write the loosened files into <folder>
    python -B loosening.py --table      print every replacement, as a table

Each file named below is copied from the repo's tests/ with the replacements
listed, each of which must match exactly once; every other line stays as it
is. <folder> is laid over a copy of tests/ outside the repo
(`verify/mutation_check.py --overlay <folder>`), so that the loosening can be
proved before the tests stage makes it. Nothing is written inside the repo.

Why each group exists is in plan/test-plan.md, "Sixth pass (CP7)", rule 3:
- the statements (D63): a helper makes the call as the test wrote it, and makes
  it again with the missing statement only when the run stops for one;
- five tests that pin the stop itself opt out of that;
- five tests that pinned "nothing said is recorded as null" accept `false`;
- seven tests that pinned the two names D66 changed read either name.
"""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[3]

EDITS = {}


def edit(name, old, new, count=1):
    EDITS.setdefault(name, []).append((old, new, count))


# ------------------------------------------------------------ mlp_helpers.py
edit("mlp_helpers.py", '''def run_cli(args, timeout=900):
    """Run `python -m mlprofessor` in a new process from the repo root."""
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [PYTHON, "-m", "mlprofessor", *[str(a) for a in args]],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=timeout, env=env,
    )
''', '''def run_cli(args, timeout=900, as_given=False):
    """Run `python -m mlprofessor` in a new process from the repo root.

    D59, D63: when the run stops only because nothing was said about time order or
    about entities, it is run once more with `--no-time-order` and `--no-entities`
    added for whichever was not said, and that run is returned. `as_given=True`
    returns the first run whatever it is.
    """
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    def once(arguments):
        return subprocess.run(
            [PYTHON, "-m", "mlprofessor", *[str(a) for a in arguments]],
            cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=timeout, env=env,
        )

    proc = once(args)
    more = [] if as_given else missing_flags(proc, args)
    return once([*args, *more]) if more else proc


# ------------------------------------------------- the two statements (D63)

def missing_statements(code, said):
    """The statements a call left unsaid when it stopped with `code`, as arguments.

    `said` maps an argument's name to what the call passed. Empty unless the run
    stopped because nothing was said about time order or about entities (D63).
    """
    if code == "time_order_undeclared" and said.get("time_col") is None:
        more = {"time_col": False}
        if said.get("group_col") is None:
            more["group_col"] = False
        return more
    if code == "entities_undeclared" and said.get("group_col") is None:
        return {"group_col": False}
    return {}


def missing_flags(proc, args):
    """The same for a run of the command line: the flags to add, or []."""
    if proc.returncode != 3:
        return []
    words = [str(a) for a in args]

    def said(*flags):
        return any(w in flags or w.startswith(flags[0] + "=") for w in words) or None

    code = proc.stderr.strip().partition("error: ")[2].partition(":")[0]
    more = missing_statements(code, {"time_col": said("--time-col", "--no-time-order"),
                                     "group_col": said("--group-col", "--no-entities")})
    return [flag for name, flag in (("time_col", "--no-time-order"),
                                    ("group_col", "--no-entities")) if name in more]


def stated_profile(*args, **kw):
    """`mlprofessor.profile(*args, **kw)`, for a test written before every call had to
    say whether the rows are ordered in time and whether rows belong together (D59, D63).

    The call is made as the test wrote it. Only when it stops because a statement
    was not made is it made again, with each missing statement made as "no":
    `time_col=False`, `group_col=False`. Nothing else is changed, and any other
    error is raised as it is.
    """
    import mlprofessor
    from mlprofessor import errors

    try:
        return mlprofessor.profile(*args, **kw)
    except errors.ProfileError as exc:
        more = missing_statements(exc.code, kw)
        if not more:
            raise
    return mlprofessor.profile(*args, **{**kw, **more})
''')

# ---------------------------------------------------------------- conftest.py
edit("conftest.py", '''    out["clf"] = mlprofessor.profile(clf, "y", profile="full", n_jobs=1, seed=0,''',
     '''    out["clf"] = h.stated_profile(clf, "y", profile="full", n_jobs=1, seed=0,''')
edit("conftest.py", '''    out["reg"] = mlprofessor.profile(str(reg_csv), "y", profile="full", n_jobs=1, seed=0,''',
     '''    out["reg"] = h.stated_profile(str(reg_csv), "y", profile="full", n_jobs=1, seed=0,''')

# ------------------------------------------------------------- cp3_helpers.py
edit("cp3_helpers.py", '''        return mlprofessor.profile(data, target, seed=seed, **args)''',
     '''        return h.stated_profile(data, target, seed=seed, **args)''')

# ------------------------------------------------------------- cp5_helpers.py
edit("cp5_helpers.py", '''def raised(frame, target="y", families=None, steps=None, **kw):
    """The `ProfileError` that `profile()` raises (fast, one core by default).

    Fails if it returns a result; any other exception passes through.
    """
    import mlprofessor
    from mlprofessor import errors, extensions

    args = dict(h.FAST)
    args.update(kw)
    try:
        with extensions.use(families=families, steps=steps):
            result = mlprofessor.profile(frame, target, **args)
    except errors.ProfileError as exc:
        return exc
    pytest.fail(f"profile() returned a result; a named error was expected "
                f"(rows {h.rows_of(result)})")
''', '''def raised(frame, target="y", families=None, steps=None, as_called=False, **kw):
    """The `ProfileError` that `profile()` raises (fast, one core by default).

    Fails if it returns a result; any other exception passes through.

    D59, D63: when the call stops because nothing was said about time order or
    about entities, it is made again with each missing statement made as "no".
    If that run ends in a named error, that error is returned: it is the one a
    test written before D63 is after. If it returns a result, the first error
    is. `as_called=True` returns the error of the call as written, and makes no
    second call.
    """
    import mlprofessor
    from mlprofessor import errors, extensions

    args = dict(h.FAST)
    args.update(kw)
    try:
        with extensions.use(families=families, steps=steps):
            result = mlprofessor.profile(frame, target, **args)
    except errors.ProfileError as exc:
        more = {} if as_called else h.missing_statements(exc.code, args)
        if not more:
            return exc
        asked = exc
    else:
        pytest.fail(f"profile() returned a result; a named error was expected "
                    f"(rows {h.rows_of(result)})")
    try:
        with extensions.use(families=families, steps=steps):
            mlprofessor.profile(frame, target, **{**args, **more})
    except errors.ProfileError as exc:
        return exc
    return asked
''')

# ------------------------------------------------- file-local run helpers
for name in ("test_pipeline.py", "test_uncertainty.py", "test_time_pipeline.py"):
    edit(name, '''        return mlprofessor.profile(frame, target, seed=seed, **args)''',
         '''        return h.stated_profile(frame, target, seed=seed, **args)''')
edit("test_edges.py", '''        return mlprofessor.profile(frame, target, **args)''',
     '''        return h.stated_profile(frame, target, **args)''')
edit("test_leakage.py", '''        res = mlprofessor.profile(frame, target, seed=0, **args)''',
     '''        res = h.stated_profile(frame, target, seed=0, **args)''')
edit("test_result_schema.py", '''    return mlprofessor.profile(data, "y", out_dir=out_dir, **args)''',
     '''    return h.stated_profile(data, "y", out_dir=out_dir, **args)''')
edit("test_model_families.py", '''    return mlprofessor.profile(frame, "y", **args)''',
     '''    return h.stated_profile(frame, "y", **args)''')
edit("test_run_hygiene.py", '''    return mlprofessor.profile(str(path), "y", **h.FAST)''',
     '''    return h.stated_profile(str(path), "y", **h.FAST)''')

# ------------------------------------------------- test functions that call profile()
for name in ("test_type_matrix.py", "test_type_matrix_group.py"):
    edit(name, '''            result = mlprofessor.profile(data, profile="fast", seed=0, n_jobs=1,
                                         out_dir=str(out), **args)''',
         '''            result = h.stated_profile(data, profile="fast", seed=0, n_jobs=1,
                                      out_dir=str(out), **args)''')
edit("test_csv_edges.py", '''    result = mlprofessor.profile(str(path), "y", **h.FAST)''',
     '''    result = h.stated_profile(str(path), "y", **h.FAST)''')
edit("test_full_profile.py",
     '''    mlprofessor.profile(full_runs["reg_csv"], "y", profile="full", n_jobs=2, seed=0,''',
     '''    h.stated_profile(full_runs["reg_csv"], "y", profile="full", n_jobs=2, seed=0,''')
edit("test_full_profile.py",
     '''        res = mlprofessor.profile(h.clf_frame(150, seed=3), "y", profile="fast", n_jobs=-1)''',
     '''        res = h.stated_profile(h.clf_frame(150, seed=3), "y", profile="fast", n_jobs=-1)''')
edit("test_non_finite.py", '''        result = mlprofessor.profile(frame, "y", **h.FAST)''',
     '''        result = h.stated_profile(frame, "y", **h.FAST)''')
edit("test_seed_range.py", '''    first = mlprofessor.profile(frame, "y", seed=LARGEST_SEED, **h.FAST)
    second = mlprofessor.profile(frame, "y", seed=LARGEST_SEED, **h.FAST)''',
     '''    first = h.stated_profile(frame, "y", seed=LARGEST_SEED, **h.FAST)
    second = h.stated_profile(frame, "y", seed=LARGEST_SEED, **h.FAST)''')

# ------------------------------------------------- tests that pin the stop itself
edit("test_time_order_stated.py", '''    err = g.raised(_table(as_text), families=[h.family("rec", log)],
                   steps=[h.step_factory(log)], out_dir=str(out))''',
     '''    err = g.raised(_table(as_text), families=[h.family("rec", log)],
                   steps=[h.step_factory(log)], out_dir=str(out), as_called=True)''')
edit("test_time_order_stated.py", '''    err = g.raised(frame, families=p.flat_family(h.Log(), n))
    assert err.code == CODE, f"{type(err).__name__}: {err}"
''', '''    err = g.raised(frame, families=p.flat_family(h.Log(), n), as_called=True)
    assert err.code == CODE, f"{type(err).__name__}: {err}"
''')
edit("test_time_order_stated.py", '''    out = tmp_path / "plain"
    proc = h.run_cli([*common, "--out", out])''', '''    out = tmp_path / "plain"
    proc = h.run_cli([*common, "--out", out], as_given=True)''')
edit("test_panel_stop.py", '''                   time_col="date", out_dir=str(out))''',
     '''                   time_col="date", out_dir=str(out), as_called=True)''')
edit("test_panel_stop.py", '''    out = tmp_path / "asked"
    proc = h.run_cli([*common, "--out", out])''', '''    out = tmp_path / "asked"
    proc = h.run_cli([*common, "--out", out], as_given=True)''')

# Read by make_overlay.py: the six assertions that pin an unsaid statement as null.
# D63: neither statement is ever null in a finished run. Each is loosened to "null,
# as today, or false, the statement the helper made for the test".
edit("test_groups_random.py", '''    assert v["scheme"] == "random"
    assert doc["input"]["group_col"] is None
''', '''    assert v["scheme"] == "random"
    assert doc["input"]["group_col"] is None or doc["input"]["group_col"] is False
''')
edit("test_groups_random.py", '''    assert plain["input"]["group_col"] is None
    assert stated["input"]["group_col"] is False
''', '''    assert plain["input"]["group_col"] is None or plain["input"]["group_col"] is False
    assert stated["input"]["group_col"] is False
''')
edit("test_panel_stop.py", '''    assert result.to_dict()["input"]["group_col"] is None
''', '''    said = result.to_dict()["input"]["group_col"]
    assert said is None or said is False
''')
edit("test_result_schema.py",
     '''    assert inp["target"] == "y" and inp["time_col"] is None and inp["task"] == "auto"
''', '''    assert inp["target"] == "y" and inp["task"] == "auto"
    assert inp["time_col"] is None or inp["time_col"] is False
''')
edit("test_time_order_stated.py", '''    assert plain["input"]["time_col"] is None
    assert stated["input"]["time_col"] is False
''', '''    assert plain["input"]["time_col"] is None or plain["input"]["time_col"] is False
    assert stated["input"]["time_col"] is False
''')

# ---- D66, ruling 5: the count and the warning of duplicates="group" are renamed.
# A test that reads either reads it under whichever name the application gives it.
edit("mlp_helpers.py", '''def files_in(folder):
''', '''# D66: `validation.n_rows_with_copies` became `n_rows_same_features`, and the warning
# `copies_grouped` became `same_features_grouped`. A test written before that reads
# whichever of the two the application gives (D59).
SAME_FEATURES_CODES = ("same_features_grouped", "copies_grouped")


def same_features_rows(validation):
    """`validation.n_rows_same_features`, or `n_rows_with_copies` where it is still named so."""
    if "n_rows_same_features" in validation:
        return validation["n_rows_same_features"]
    return validation["n_rows_with_copies"]


def count_same_features_warnings(result):
    """How many warnings say that rows with the same features were kept together."""
    return sum(code in SAME_FEATURES_CODES for code, _ in warnings_of(result))


def files_in(folder):
''')
edit("test_duplicates_group.py", '''    assert v["n_rows_with_copies"] == 600
''', '''    assert h.same_features_rows(v) == 600
''')
edit("test_duplicates_group.py", '''    assert g.count_code(result, "copies_grouped") == 1
''', '''    assert h.count_same_features_warnings(result) == 1
''')
edit("test_duplicates_group.py", '''    assert v["n_rows_with_copies"] == 0
    assert g.count_code(grouped, "copies_grouped") == 0
    assert plain.to_dict()["validation"]["n_rows_with_copies"] is None
''', '''    assert h.same_features_rows(v) == 0
    assert h.count_same_features_warnings(grouped) == 0
    assert h.same_features_rows(plain.to_dict()["validation"]) is None
''')
edit("test_duplicates_group.py", '''    assert v["n_rows_with_copies"] == 14
''', '''    assert h.same_features_rows(v) == 14
''')
edit("test_duplicates_group.py", '''    assert v["n_rows_with_copies"] == 100
''', '''    assert h.same_features_rows(v) == 100
''')
edit("test_duplicates_group.py", '''    assert grouped["validation"]["n_rows_with_copies"] is None
''', '''    assert h.same_features_rows(grouped["validation"]) is None
''')
edit("test_duplicates_group.py", '''    assert "copies_grouped" in proc.stdout, proc.stdout
''', '''    assert any(code in proc.stdout for code in h.SAME_FEATURES_CODES), proc.stdout
''')
edit("test_groups_random.py", '''    for key in ("group_col", "n_groups_training", "n_groups_holdout", "n_rows_with_copies"):
        assert v[key] is None, key
''', '''    for key in ("group_col", "n_groups_training", "n_groups_holdout"):
        assert v[key] is None, key
    assert h.same_features_rows(v) is None
''')

# The helper or the test each replacement is in, in the order they are made above.
WHERE = {
    "mlp_helpers.py": ["run_cli (and three new functions after it)",
                       "three new names before files_in"],
    "conftest.py": ["full_runs", "full_runs"],
    "cp3_helpers.py": ["run"],
    "cp5_helpers.py": ["raised"],
    "test_pipeline.py": ["_run"],
    "test_uncertainty.py": ["_run"],
    "test_time_pipeline.py": ["_run"],
    "test_edges.py": ["_run"],
    "test_leakage.py": ["_run"],
    "test_result_schema.py": ["_profile", "test_profile_json_has_every_key_with_its_type"],
    "test_model_families.py": ["_profile"],
    "test_run_hygiene.py": ["_call_profile"],
    "test_type_matrix.py": ["test_a_result_or_a_named_error"],
    "test_type_matrix_group.py": ["test_a_result_or_a_named_error_as_the_entity_column"],
    "test_csv_edges.py": ["test_profile_runs_on_a_file_with_leading_blank_lines"],
    "test_full_profile.py": ["test_same_result_at_one_and_two_jobs",
                             "test_n_jobs_minus_one_records_cpu_count"],
    "test_non_finite.py": [
        "test_a_target_on_a_very_large_scale_gives_finite_numbers_or_the_named_error"],
    "test_seed_range.py": ["test_the_largest_seed_runs_and_the_next_one_is_refused"],
    "test_time_order_stated.py": [
        "test_a_date_column_and_no_word_about_time_stops_the_run",
        "test_the_date_error_comes_before_the_copies_error",
        "test_the_command_line_states_it",
        "test_the_statement_on_a_table_with_no_dates_changes_only_its_record"],
    "test_panel_stop.py": ["test_a_persistent_panel_with_no_entity_column_stops",
                           "test_the_command_line_says_there_are_no_entities",
                           "test_a_panel_that_does_not_persist_runs_without_a_word"],
    "test_groups_random.py": ["test_without_an_entity_column_the_keys_are_null",
                              "test_group_col_false_is_recorded_and_changes_nothing_else",
                              "test_without_an_entity_column_the_keys_are_null"],
    "test_duplicates_group.py": [
        "test_each_set_of_copies_stays_on_one_side_and_in_one_fold",
        "test_each_set_of_copies_stays_on_one_side_and_in_one_fold",
        "test_a_table_with_no_copies_is_split_as_by_default",
        "test_a_few_copies_are_grouped_and_counted",
        "test_copies_and_declared_entities_are_kept_together_both",
        "test_under_time_order_it_changes_nothing",
        "test_the_command_line_takes_group"],
}


def loosened(name):
    """The file with its replacements made, and with the line endings it has in the
    repo: five of the test files end their lines with CR LF, and they still do."""
    raw = (REPO / "tests" / name).read_bytes().decode("utf-8")
    windows = "\r\n" in raw
    text = raw.replace("\r\n", "\n")
    for old, new, count in EDITS[name]:
        assert text.count(old) == count, (name, text.count(old), old[:70])
        text = text.replace(old, new)
    return text.replace("\n", "\r\n") if windows else text


def main(argv):
    assert {k: len(v) for k, v in WHERE.items()} == {k: len(v) for k, v in EDITS.items()}
    if argv == ["--table"]:
        print("| File | In | Lines before | Lines after |")
        print("|---|---|---|---|")
        for name, edits in EDITS.items():
            for (old, new, _), where in zip(edits, WHERE[name]):
                if old.count("\n") + new.count("\n") > 8:
                    old = new = "(a whole function: see the test plan)"

                def cell(block):
                    return "<br>".join("`" + line.strip() + "`"
                                       for line in block.strip("\n").split("\n") if line.strip())
                print(f"| `{name}` | `{where}` | {cell(old)} | {cell(new)} |")
        return
    overlay = Path(argv[0])
    overlay.mkdir(parents=True, exist_ok=True)
    for name in EDITS:
        (overlay / name).write_bytes(loosened(name).encode("utf-8"))
    print(f"{len(EDITS)} files, {sum(len(v) for v in EDITS.values())} replacements -> {overlay}")


if __name__ == "__main__":
    main(sys.argv[1:])
