"""Fixed public source split declared before any new policy optimization."""

TRAIN_MOTION_GLOB = "soccer-standard-00[12345689]_*.npz"
TRAIN_MOTION_NAMES = (
    "soccer-standard-001_right.npz",
    "soccer-standard-002_left.npz",
    "soccer-standard-003_left.npz",
    "soccer-standard-004_right.npz",
    "soccer-standard-005_right.npz",
    "soccer-standard-006_right.npz",
    "soccer-standard-008_left.npz",
    "soccer-standard-009_right.npz",
)
DEVELOPMENT_MOTION_NAMES = ("soccer-standard-007_left.npz", "soccer-standard-010_right.npz")


def motion_names_for_pattern(pattern: str) -> tuple[str, ...]:
    if type(pattern) is not str:
        raise ValueError("declared source pattern required")
    if pattern == TRAIN_MOTION_GLOB:
        return TRAIN_MOTION_NAMES
    if pattern in DEVELOPMENT_MOTION_NAMES:
        return (pattern,)
    raise ValueError("fixed eight-source training or one explicit development source required")
