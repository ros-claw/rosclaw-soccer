import numpy as np
import pytest

from rosclaw_soccer.rsi.cftrack_transfer_preflight import compare_cftrack_transfer


def test_same_target_gain_change_is_not_motion_proof():
    z = np.zeros(29)
    one = np.ones(29)
    target = z.copy()
    target[5] = 0.5
    report = compare_cftrack_transfer(
        previous_target=target,
        joint_position=z,
        joint_velocity=z,
        old_stiffness=one,
        old_damping=one,
        cf_stiffness=2 * one,
        cf_damping=one,
        default_lab=z,
        scale_lab=one,
        mj2lab=np.arange(29),
        lab2mj=np.arange(29),
    )
    assert report["max_same_target_raw_pd_change_nm"] == pytest.approx(0.5)
    assert report["limiting_joint_index"] == 5
    assert report["input_compatibility_is_not_policy_qualification"]


def test_nonfinite_rejected():
    z = np.zeros(29)
    bad = z.copy()
    bad[0] = np.nan
    with pytest.raises(ValueError):
        compare_cftrack_transfer(
            previous_target=bad,
            joint_position=z,
            joint_velocity=z,
            old_stiffness=np.ones(29),
            old_damping=np.ones(29),
            cf_stiffness=np.ones(29),
            cf_damping=np.ones(29),
            default_lab=z,
            scale_lab=np.ones(29),
            mj2lab=np.arange(29),
            lab2mj=np.arange(29),
        )
