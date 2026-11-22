"""T27: the seed's range (spec 2.1, 2.2, 2.4, 2.7; D45). Red today.

`seed` is a whole number from 0 to 4,294,967,295 (2**32 - 1). A seed outside
the range is an `InputError` raised with the other arguments, before anything
is loaded; it is never folded down.
"""
import pytest

import cp3_helpers as c
import mlp_helpers as h

LARGEST_SEED = 2**32 - 1
TOO_LARGE = {"two_to_the_32": 2**32, "two_to_the_63": 2**63, "timestamp_style": 20261119123456}


def _duplicate_names_csv(path):
    rows = "".join(f"{i},{(i * 7) % 11},{i % 2}\n" for i in range(160))
    path.write_text("a,a,y\n" + rows, encoding="utf-8")
    return str(path)


@pytest.mark.parametrize("seed", list(TOO_LARGE.values()), ids=list(TOO_LARGE))
def test_a_seed_above_the_range_is_an_input_error_before_anything_is_loaded(tmp_path, seed):
    import mlprofessor
    from mlprofessor import stages
    from mlprofessor.errors import DuplicateColumnsError, InputError

    dup = _duplicate_names_csv(tmp_path / "dup.csv")
    with pytest.raises(DuplicateColumnsError):
        stages.load_table(dup)
    with pytest.raises(InputError):
        mlprofessor.profile(dup, "y", seed=seed, **h.FAST)

    log = h.Log()
    out = tmp_path / "out"
    with pytest.raises(InputError):
        c.run(h.reg_frame(n=160, seed=31), families=[h.family("rec", log)],
              steps=[h.step_factory(log)], seed=seed, out_dir=out)
    assert log.events == []
    assert h.files_in(out) == []


def test_the_largest_seed_runs_and_the_next_one_is_refused():
    import mlprofessor
    from mlprofessor.errors import InputError

    frame = h.reg_frame(n=160, seed=32)
    first = h.stated_profile(frame, "y", seed=LARGEST_SEED, **h.FAST)
    second = h.stated_profile(frame, "y", seed=LARGEST_SEED, **h.FAST)
    doc_a, doc_b = first.to_dict(), second.to_dict()
    assert doc_a["meta"]["seed"] == LARGEST_SEED
    doc_a["meta"].pop("timing")
    doc_b["meta"].pop("timing")
    assert h.plain(doc_a) == h.plain(doc_b)

    with pytest.raises(InputError):
        mlprofessor.profile(frame, "y", seed=LARGEST_SEED + 1, **h.FAST)


def test_command_line_refuses_a_seed_out_of_range(tmp_path):
    path = tmp_path / "table.csv"
    h.reg_frame(n=160, seed=33).to_csv(path, index=False)
    out = tmp_path / "out"
    proc = h.run_cli(["profile", "--data", path, "--target", "y", "--out", out,
                      "--profile", "fast", "--n-jobs", "1", "--seed", str(2**32)])
    assert proc.returncode == 3, proc.stderr
    lines = [line for line in proc.stderr.splitlines() if line.strip()]
    assert len(lines) == 1, proc.stderr
    assert lines[0].startswith("error: input_error:"), proc.stderr
    assert "Traceback" not in proc.stderr
    assert h.files_in(out) == []


@pytest.mark.parametrize("seed", [2**32, -1], ids=["two_to_the_32", "minus_one"])
def test_split_random_refuses_a_seed_out_of_range(seed):
    from mlprofessor import stages
    from mlprofessor.errors import InputError

    with pytest.raises(InputError):
        stages.split_random(120, None, seed)
