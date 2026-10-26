from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.rsi.cftrack_foot_frame_reference import condition_cftrack_roll_reference


class Kinematics:
    def __init__(self):
        import mujoco

        self.model = mujoco.MjModel.from_xml_string("<mujoco/>")
        self.data = mujoco.MjData(self.model)
        self.data.time = 7.3
        self.defect = None

    def forward(self, joints, base, quat):
        self.data.time += 1  # Detect accidentally using donor data, without physics.
        result = {}
        for side, y in [("left", 0.1), ("right", -0.1)]:
            pitch = base + np.array([0.0, y, -0.65])
            roll = pitch + np.array([0.01, 0.0, -0.017])
            if self.defect == "workspace" and side == "right":
                roll[0] += 2
            if self.defect == "speed":
                roll[0] += joints[1]
            if self.defect == "floor":
                roll[2] -= 0.2
            angle = joints[0] if self.defect != "angular" else joints[1]
            result[side + "_ankle_pitch_link"] = dict(pos=pitch, quat=quat.copy())
            result[side + "_ankle_roll_link"] = dict(
                pos=roll, quat=np.array([np.cos(angle / 2), np.sin(angle / 2), 0.0, 0.0])
            )
        if self.defect == "nan_fk":
            result["left_ankle_pitch_link"]["quat"][0] = np.nan
        return result


def fixture():
    base = np.tile(np.array([0.0, 0.0, 0.77], np.float32), (10, 1))
    p = SimpleNamespace(
        task="kickball",
        reference_source="CFgen",
        counter_step=0,
        async_stage_thread=None,
        obs_history_buffer=np.zeros((5, 141)),
        action=np.zeros(29),
        ref_base_pos=base,
        ref_base_quat=np.tile(np.array([1, 0, 0, 0], np.float32), (10, 1)),
        dof_pos=np.zeros((10, 29), np.float32),
        ref_torso_future_pos=base.copy(),
        kinematics=Kinematics(),
        unrelated=np.array([13.0]),
    )
    for side, y in [("left", 0.1), ("right", -0.1)]:
        setattr(p, f"ref_{side}_ankle_future_pos", base + np.array([0, y, -0.65], np.float32))
        setattr(p, f"ref_{side}_ankle_future_quat", p.ref_base_quat.copy())
    return p


def test_only_foot_reference_changes_without_donor_fk_or_history_writes():
    p = fixture()
    p.dof_pos[:, 0] = np.linspace(0, 0.05, 10)
    originals = {k: v.copy() for k, v in vars(p).items() if isinstance(v, np.ndarray)}
    donor = p.kinematics.data
    receipt = condition_cftrack_roll_reference(p)
    assert receipt["correct_training_frame_not_proven"]
    assert receipt["frames"] == 10 and receipt["source_frame"] == "ankle_pitch"
    assert receipt["metrics"]["right"]["maximum_extra_angular_speed_rad_s"] < 8
    assert p.kinematics.data is donor and donor.time == 7.3
    for k, v in originals.items():
        if "ankle_future" not in k:
            np.testing.assert_array_equal(getattr(p, k), v)
    np.testing.assert_allclose(
        p.ref_right_ankle_future_pos - originals["ref_right_ankle_future_pos"],
        np.tile([0.01, 0, -0.017], (10, 1)),
        atol=1e-7,
    )


@pytest.mark.parametrize(
    "defect",
    [
        "workspace",
        "speed",
        "angular",
        "floor",
        "nan_fk",
        "history",
        "advanced",
        "npz",
        "mismatch",
        "shape",
    ],
)
def test_failure_does_not_partially_install_or_mutate_scratch_donor(defect):
    p = fixture()
    if defect == "history":
        p.obs_history_buffer[0, 0] = 1
    elif defect == "advanced":
        p.counter_step = 1
    elif defect == "npz":
        p.reference_source = "NPZmotion"
    elif defect == "mismatch":
        p.ref_right_ankle_future_pos[0, 0] = 1
    elif defect == "shape":
        p.dof_pos = p.dof_pos[:9]
    else:
        p.kinematics.defect = defect
        p.dof_pos[5, 1] = 1
    originals = {k: v.copy() for k, v in vars(p).items() if isinstance(v, np.ndarray)}
    with pytest.raises(ValueError):
        condition_cftrack_roll_reference(p)
    for k, v in originals.items():
        np.testing.assert_array_equal(getattr(p, k), v)
    assert p.kinematics.data.time == 7.3
