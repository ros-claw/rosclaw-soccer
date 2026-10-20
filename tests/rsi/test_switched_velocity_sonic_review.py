"""Synthetic input/history/PD reconstruction, never native motor qualification."""

from types import SimpleNamespace

import numpy as np
import pytest
from test_sonic_external_history import fixture as tracker_fixture
from test_velocity_policy_native_evidence import fixture as velocity_fixture

from rosclaw_soccer.providers.g1.sonic_runup import _VARIANTS
from rosclaw_soccer.rsi.switched_velocity_sonic_review import review_switched


def fixture():
    full, p, _, _ = velocity_fixture()
    raw = {
        k: v
        for k, v in full.items()
        if k
        not in {
            "velocity_policy_observation",
            "velocity_policy_action_hg",
            "command_body_mps",
            "command_world_mps",
            "angular_velocity_body_radps",
            "gait_phase",
        }
    }
    c = tracker_fixture()[0]
    c._variant = _VARIANTS["low_latency"]
    raw["canonical_qpos"][:, 7:36] = c.default_angles
    raw["target_body_canonical_qpos_after_substep"][:, :, 7:36] = c.default_angles
    raw["joint_target_rad"][:] = c.default_angles
    c.reference = np.repeat(raw["canonical_qpos"][0, None, :36], 309, axis=0)
    enc = c._encoder_observation(SimpleNamespace(qpos=raw["canonical_qpos"][0]), 0)[0]
    dec = np.zeros(994, dtype=np.float32)
    dec[964:994] = np.tile([0.0, 0.0, -1.0], 10)
    neural = dict(
        encoder_input=np.repeat(enc[None], 270, axis=0),
        encoder_output=np.zeros((270, 64), dtype=np.float32),
        decoder_input=np.repeat(dec[None], 270, axis=0),
        decoder_output=np.zeros((270, 29), dtype=np.float32),
    )
    names = {
        "pelvis": 1,
        "left_ankle_roll_link": 2,
        "left_ankle_pitch_link": 3,
        "right_ankle_roll_link": 4,
        "right_ankle_pitch_link": 5,
    }
    world = SimpleNamespace(
        nbody=46, body_rootid=np.array([0] + [1] * 45), body=lambda n: SimpleNamespace(id=names[n])
    )
    return raw, neural, p, c, world


def test_reconstructs_all_history_and_common_pd_without_claiming_football():
    raw, n, p, c, w = fixture()
    r = review_switched(raw, n, p, c, w, handoff=30)
    assert r["all_common_pd_and_policy_inputs_reconstructed"]
    assert not r["score"]["qualified"]
    assert not r["score"]["official_football_exam_pass"]
    assert not r["body_reset_performed"]
    assert not r["gain_switch_performed"]


@pytest.mark.parametrize(
    "kind", ["encoder", "history", "action", "pd", "continuity", "dtype", "missing"]
)
def test_mixed_codec_or_corrupt_actual_evidence_fails(kind):
    raw, n, p, c, w = fixture()
    if kind == "encoder":
        n["encoder_input"][4, 4] += 0.1
    elif kind == "history":
        n["decoder_input"][4, 674] += 0.1
    elif kind == "action":
        n["decoder_output"][4, 0] += 0.1
    elif kind == "pd":
        raw["pd_requested_torque_nm"][40, 3, 0] += 1
    elif kind == "continuity":
        raw["target_body_canonical_qpos_after_substep"][40, -1, 0] += 0.1
    elif kind == "dtype":
        n["decoder_output"] = n["decoder_output"].astype(np.float64)
    else:
        del raw["force_n"]
    with pytest.raises(ValueError):
        review_switched(raw, n, p, c, w, handoff=30)
