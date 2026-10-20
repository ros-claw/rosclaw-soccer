"""Split is fixed before optimization; filenames are not unseen-world proof."""

from pathlib import PurePath

import pytest

from rosclaw_soccer.rsi.mjlab_motion_curriculum import (
    DEVELOPMENT_MOTION_NAMES,
    TRAIN_MOTION_GLOB,
    TRAIN_MOTION_NAMES,
    motion_names_for_pattern,
)


def test_eight_training_sources_and_balanced_two_development_sources_are_disjoint():
    assert len(TRAIN_MOTION_NAMES) == 8
    assert len(DEVELOPMENT_MOTION_NAMES) == 2
    assert not set(TRAIN_MOTION_NAMES) & set(DEVELOPMENT_MOTION_NAMES)
    assert motion_names_for_pattern(TRAIN_MOTION_GLOB) == TRAIN_MOTION_NAMES
    assert all(PurePath(x).match(TRAIN_MOTION_GLOB) for x in TRAIN_MOTION_NAMES)
    assert not any(PurePath(x).match(TRAIN_MOTION_GLOB) for x in DEVELOPMENT_MOTION_NAMES)
    for name in DEVELOPMENT_MOTION_NAMES:
        assert motion_names_for_pattern(name) == (name,)


@pytest.mark.parametrize("pattern", ["*", "soccer-standard-*.npz", "../*", None, True])
def test_no_future_expansion_of_training_split(pattern):
    with pytest.raises(ValueError):
        motion_names_for_pattern(pattern)
