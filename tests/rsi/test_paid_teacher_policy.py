import numpy as np
import pytest

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.rsi.paid_teacher_policy import (
    OBSERVATION_NAMES,
    PaidResearchPolicy,
    strict_parameters,
)


def metadata():
    return dict(
        joint_names=",".join(reversed(G1_DDS_JOINT_NAMES)),
        observation_names=",".join(OBSERVATION_NAMES),
        command_names="motion",
        anchor_body_name="torso_link",
        joint_stiffness=",".join(["40"] * 29),
        joint_damping=",".join(["2"] * 29),
        default_joint_pos=",".join(["0"] * 29),
        action_scale=",".join(["0.25"] * 29),
    )


def test_named_permutation_and_exact_metadata_no_fallback():
    names, params = strict_parameters(metadata())
    assert names == list(reversed(G1_DDS_JOINT_NAMES))
    assert params["action_scale"].shape == (29,)


@pytest.mark.parametrize(
    "key,value",
    [
        ("joint_names", "hip"),
        ("action_scale", "0.2"),
        ("joint_stiffness", ",".join(["nan"] * 29)),
        ("joint_damping", ",".join(["0"] * 29)),
        ("observation_names", "obs"),
    ],
)
def test_rejects_incomplete_wrong_order_or_unsafe_metadata(key, value):
    m = metadata()
    m[key] = value
    with pytest.raises(ValueError):
        strict_parameters(m)


def test_no_padding_or_implicit_restart_after_bad_frame_before_inference():
    policy = PaidResearchPolicy.__new__(PaidResearchPolicy)
    policy.frame, policy.horizon, policy.faulted = 0, 260, False
    with pytest.raises(ValueError, match="consecutive"):
        policy.step(np.zeros(159, dtype=np.float32), frame=0)
    assert policy.faulted
    with pytest.raises(ValueError, match="latched"):
        policy.step(np.zeros(160, dtype=np.float32), frame=0)
