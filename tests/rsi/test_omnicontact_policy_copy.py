import threading
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
from test_shared_nn_forecast import setup

from rosclaw_soccer.rsi.omnicontact_policy_copy import cftrack_state_hash, copy_cftrack_policy


class Kinematics:
    pass


class Record:
    pass


Kinematics.__module__ = "common.mujoco_kinematics"


def policy(tmp_path):
    p, _ = setup(tmp_path)
    obj = p.policy
    obj.history = list(obj.history)
    for key in ("state_cmd", "policy_output"):
        record = Record()
        vars(record).update(vars(getattr(obj, key)))
        setattr(obj, key, record)
    obj.task, obj.reference_source = "kickball", "CFgen"
    obj.async_stage_thread = obj.async_stage_pending = obj.async_stage_error = None
    obj.async_stage_lock = threading.Lock()
    # A robot-only FK model, distinct from the 43-q native ball/body world.
    joints = (
        "".join('<body><joint/><geom type="sphere" size=".01" mass=".01"/>' for _ in range(29))
        + "</body>" * 29
    )
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body><freejoint/><geom type="sphere" size=".1" mass="1"/>'
        + joints
        + "</body></worldbody></mujoco>"
    )
    obj.kinematics = Kinematics()
    obj.kinematics.model, obj.kinematics.data = model, mujoco.MjData(model)
    mujoco.mj_forward(model, obj.kinematics.data)
    return obj


def test_private_fk_data_history_and_lock_with_only_ort_and_model_shared(tmp_path):
    p = policy(tmp_path)
    before = cftrack_state_hash(p)
    c = copy_cftrack_policy(p)
    assert c is not p and cftrack_state_hash(c) == before
    assert c.ort_session is p.ort_session and c.kinematics.model is p.kinematics.model
    assert c.kinematics.data is not p.kinematics.data
    assert c.async_stage_lock is not p.async_stage_lock
    assert c.state_cmd is not p.state_cmd and c.policy_output is not p.policy_output
    c.last_action_il[0] += 1
    c.kinematics.data.qpos[0] += 1
    assert cftrack_state_hash(p) == before and cftrack_state_hash(c) != before


@pytest.mark.parametrize("defect", ["resource", "cycle", "nan", "active", "locked", "world_fk"])
def test_unsupported_or_concurrent_resources_are_rejected(tmp_path, defect):
    p = policy(tmp_path)
    if defect == "resource":
        p.injected = object()
    elif defect == "cycle":
        p.loop = []
        p.loop.append(p.loop)
    elif defect == "nan":
        p.last_action_il[0] = np.nan
    elif defect == "active":
        p.async_stage_thread = SimpleNamespace()
    elif defect == "locked":
        p.async_stage_lock.acquire()
    else:
        p.kinematics.model = mujoco.MjModel.from_xml_string("<mujoco/>")
    with pytest.raises(ValueError):
        copy_cftrack_policy(p)


def test_dictionary_keys_and_fk_model_parameters_are_fingerprinted(tmp_path):
    p = policy(tmp_path)
    p.metadata = {"one": 1}
    before = cftrack_state_hash(p)
    p.metadata = {"two": 1}
    assert cftrack_state_hash(p) != before
    before = cftrack_state_hash(p)
    p.kinematics.model.opt.gravity[0] = 1
    assert cftrack_state_hash(p) != before


def test_npz_motion_requires_explicit_reference_opt_in(tmp_path):
    p = policy(tmp_path)
    p.reference_source = "NPZmotion"
    with pytest.raises(ValueError):
        copy_cftrack_policy(p)
    with pytest.raises(ValueError):
        cftrack_state_hash(p, reference_source="unknown")
    before = cftrack_state_hash(p, reference_source="NPZmotion")
    copied = copy_cftrack_policy(p, reference_source="NPZmotion")
    assert cftrack_state_hash(copied, reference_source="NPZmotion") == before
    copied.last_action_il[0] += 1
    assert cftrack_state_hash(p, reference_source="NPZmotion") == before
