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


class FastProfileRowsError(ProfileError):
    """The fast profile's cap left too few rows, or too few of a class, and the
    full profile would run (D52, D53)."""

    code = "fast_profile_too_few_rows"


class GapError(ProfileError):
    code = "gap_too_large"


class FeatureNameCollisionError(ProfileError):
    code = "feature_name_collision"


class NoFeaturesError(ProfileError):
    code = "no_features"


class HoldoutCopiesError(ProfileError):
    """Under the random scheme, too many holdout rows have a copy among the
    training rows, and the caller has not said that they are genuine (D49, D53)."""

    code = "holdout_copies_in_training"


class TimeOrderUndeclaredError(ProfileError):
    """A column holds dates and the caller has neither declared a time column nor
    said that the rows have no time order (D58)."""

    code = "time_order_undeclared"


class EntitiesUndeclaredError(ProfileError):
    """Under time order the timestamps repeat, the target is strongly
    autocorrelated, and the caller has neither declared an entity column nor said
    that there are no entities (D58)."""

    code = "entities_undeclared"


class GroupColumnError(ProfileError):
    """The entity column cannot be used as it is: an empty cell in a usable row,
    values that cannot be compared, or two rows of one entity at one timestamp (D57)."""

    code = "group_column_invalid"


class TooFewGroupsError(ProfileError):
    """Too few entities to form the holdout and five folds of whole entities (D57)."""

    code = "too_few_groups"


class OneClassError(ProfileError):
    code = "one_class"


class AllModelsFailedError(ProfileError):
    code = "all_models_failed"


class NonFiniteScoreError(ProfileError):
    """A number the result rests on is infinite or not-a-number (D44)."""

    code = "non_finite_score"
