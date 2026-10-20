import numpy as np
import pytest
from test_g1_velocity_policy import parameters

from rosclaw_soccer.rsi.bounded_reference_imitation import build_imitation_rows


def raw_fixture():
    obs = np.zeros((3, 8, 168), dtype=np.float32)
    obs[:, :, 98:127] = 2.0
    return dict(
        full_observation=obs,
        action=np.zeros((3, 8, 29), dtype=np.float32),
        target=np.zeros((3, 8, 29)),
        reference_joint=np.full((3, 8, 29), 2, dtype=np.float32),
        measured_joint=np.zeros((3, 8, 29)),
        active_before=np.ones((3, 8), dtype=np.bool_),
        source_id=np.tile(np.arange(8), (3, 1)),
        source_phase=np.repeat(np.arange(3)[:, None], 8, axis=1),
    )


def test_correction_is_bounded_and_first_episode_rows_are_not_repeated():
    raw = raw_fixture()
    raw["active_before"][2, 0] = False
    before = {k: v.copy() for k, v in raw.items()}
    rows = build_imitation_rows(raw, parameters())
    assert len(rows["observation"]) == 23
    np.testing.assert_allclose(rows["target_delta_rad"], 0.15, atol=1e-7)
    np.testing.assert_allclose(rows["target_action"], 0.6, atol=1e-7)
    assert all(np.array_equal(raw[k], v) for k, v in before.items())


@pytest.mark.parametrize(
    "kind", ["source", "phase", "reentry", "reference", "current", "target", "nan"]
)
def test_unbound_or_post_reset_labels_rejected(kind):
    raw = raw_fixture()
    if kind == "source":
        raw["source_id"][1, 0] = 7
    elif kind == "phase":
        raw["source_phase"][1, 0] = 2
    elif kind == "reentry":
        raw["active_before"][1, 0] = False
    elif kind == "reference":
        raw["reference_joint"][0, 0, 0] = 1
    elif kind == "current":
        raw["measured_joint"][0, 0, 0] = 1
    elif kind == "target":
        raw["target"][0, 0, 0] = 1
    else:
        raw["action"][0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        build_imitation_rows(raw, parameters())
