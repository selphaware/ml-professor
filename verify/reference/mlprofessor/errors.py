"""Named errors of the mlprofessor contract (spec.md, "Errors")."""


class ProfileError(Exception):
    """Base class. `code` is the stable name the command line prints."""

    code = "profile_error"


class InputError(ProfileError):
    code = "input_error"


class ColumnNotFoundError(ProfileError):
    code = "column_not_found"


class DuplicateColumnsError(ProfileError):
    code = "duplicate_columns"


class TargetConstantError(ProfileError):
    code = "target_constant"


class TaskAmbiguousError(ProfileError):
    code = "task_ambiguous"


class TaskUnsupportedError(ProfileError):
    code = "task_unsupported"


class TimeColumnError(ProfileError):
    code = "time_column_invalid"


class TooFewRowsError(ProfileError):
    code = "too_few_rows"


class FastProfileError(ProfileError):
    code = "fast_profile_too_few_timestamps"


class GapError(ProfileError):
    code = "gap_too_large"


class FeatureNameCollisionError(ProfileError):
    code = "feature_name_collision"


class NoFeaturesError(ProfileError):
    code = "no_features"


class OneClassError(ProfileError):
    code = "one_class"


class AllModelsFailedError(ProfileError):
    code = "all_models_failed"


class NonFiniteScoreError(ProfileError):
    """A number the result rests on is infinite or not-a-number (D44)."""

    code = "non_finite_score"
