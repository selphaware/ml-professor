"""mlprofessor: how predictable one target column of one table is (spec 2.1)."""
from mlprofessor.result import ProfileResult


def profile(data, target, task="auto", time_col=None, gap=0,
            profile="full", seed=0, out_dir=None, n_jobs=-1, duplicates="error",
            group_col=None):
    """Measure how predictable `target` is in `data`. Returns a ProfileResult.

    Every call says two things: `time_col` (a column's name or False) and
    `group_col` (a column's name or False). None, the default of each, is "not
    said", and the run stops with a named error (spec 2.1, step 5).

    `profile="fast"` keeps at most 2,000 usable rows (step 6). With an entity
    column and no time column it keeps whole entities, always every entity of
    a quarter or more of the usable rows; when it cannot, the run stops with
    `fast_profile_group_too_large` and the full profile is the answer.
    """
    from mlprofessor.pipeline import run

    return run(data, target, task=task, time_col=time_col, gap=gap, profile=profile,
               seed=seed, out_dir=out_dir, n_jobs=n_jobs, duplicates=duplicates,
               group_col=group_col)


__all__ = ["profile", "ProfileResult"]
