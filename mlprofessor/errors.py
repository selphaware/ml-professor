"""Named errors (spec 2.4). Each carries the `code` the command line prints."""


class ProfileError(Exception):
    """Base class of every named error of the package."""

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


class FastProfileRowsError(ProfileError):
    code = "fast_profile_too_few_rows"


class GapError(ProfileError):
    code = "gap_too_large"


class NoFeaturesError(ProfileError):
    code = "no_features"


class OneClassError(ProfileError):
    code = "one_class"


class FeatureNameCollisionError(ProfileError):
    code = "feature_name_collision"


class AllModelsFailedError(ProfileError):
    code = "all_models_failed"


class NonFiniteScoreError(ProfileError):
    """A number the result rests on is infinite or not-a-number (D44)."""

    code = "non_finite_score"


class HoldoutCopiesError(ProfileError):
    """Too many holdout rows have a copy among the training rows (step 8a)."""

    code = "holdout_copies_in_training"
