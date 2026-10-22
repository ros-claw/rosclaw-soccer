import xml.etree.ElementTree as ET

import numpy as np
import pytest

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.rsi.shared_native_motor_world import NativePDParameters, SharedNativeMotorWorld
from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene


def source_xml():
    joints = (
        "".join(
            f'<body name="link{i}"><joint name="{n}" armature=".1"/>'
            '<geom type="sphere" size=".01" mass=".01" contype="0" conaffinity="0"/>'
            for i, n in enumerate(G1_DDS_JOINT_NAMES)
        )
        + "</body>" * 29
    )
    act = "".join(f'<motor name="{n}" joint="{n}"/>' for n in reversed(G1_DDS_JOINT_NAMES))
    return (
        '<mujoco><option timestep=".002" gravity="0 0 0"/><worldbody>'
        '<body name="ball"><freejoint/><geom type="sphere" size=".11" mass=".43"/></body>'
        '<body name="pelvis" pos="0 0 1"><freejoint name="root"/>'
        '<geom type="sphere" size=".05" mass="1"/>'
        + joints
        + "</body></worldbody><actuator>"
        + act
        + "</actuator></mujoco>"
    )


def world():
    import mujoco

    xml = shared_native_scene(source_xml(), prefixes=("a/", "b/"))
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    b = m.joint("b/root").qposadr[0]
    d.qpos[b] = 2
    params = NativePDParameters(np.full(29, 20.0), np.ones(29), np.full(29, 25.0))
    w = SharedNativeMotorWorld(m, d, players={"a/": params, "b/": params}, ball_body_name="ball")
    return w


def test_two_players_share_ten_steps_and_correct_noncanonical_addresses():
    import mujoco

    w = world()
    m, d = w.model, w.data
    own = mujoco.MjData(m)
    mujoco.mj_copyData(own, m, d)
    targets = {"a/": np.linspace(-0.02, 0.02, 29), "b/": np.linspace(0.03, -0.03, 29)}
    for _ in range(10):
        for p, layout in w.layouts.items():
            own.ctrl[layout.ai] = np.clip(
                20 * (targets[p] - own.qpos[layout.qi]) - own.qvel[layout.vi], -25, 25
            )
        mujoco.mj_step(m, own)
    row = w.advance(targets)
    assert w.native_steps == 10 and w.control_steps == 1
    assert d.time == pytest.approx(0.02)
    np.testing.assert_array_equal(d.qpos, own.qpos)
    np.testing.assert_array_equal(d.qvel, own.qvel)
    assert row["body_failed"] == {"a/": False, "b/": False}
    assert row["ball_forces"].shape == (10, 2, 3)
    assert not row["ball_forces"].any()
    for layout in w.layouts.values():
        q, v = layout.canonical(d)
        assert q.shape == (43,) and v.shape == (41,)
        np.testing.assert_array_equal(q[36:], d.qpos[layout.bq : layout.bq + 7])
        assert len(layout.ai) == 29


@pytest.mark.parametrize("invalid", ["missing", "nonfinite", "wrongsize"])
def test_all_player_targets_validated_before_any_mutation(invalid):
    w = world()
    targets = {p: np.zeros(29) for p in w.layouts}
    if invalid == "missing":
        targets.pop("b/")
    elif invalid == "nonfinite":
        targets["b/"][4] = np.nan
    else:
        targets["b/"] = np.zeros(28)
    before = w.data.qpos.copy(), w.data.qvel.copy(), w.data.ctrl.copy()
    with pytest.raises(ValueError):
        w.advance(targets)
    assert w.native_steps == 0 and w.data.time == 0
    for a, b in zip(before, (w.data.qpos, w.data.qvel, w.data.ctrl), strict=True):
        np.testing.assert_array_equal(a, b)


def test_force_or_clock_intervention_rejected_without_automatic_reset():
    w = world()
    w.data.xfrc_applied[1, 0] = 1
    with pytest.raises(ValueError, match="unmodified"):
        w.advance({p: np.zeros(29) for p in w.layouts})
    assert w.native_steps == 0


def test_native_ball_contact_is_owned_by_actual_body_not_nearest_player():
    import mujoco

    w = world()
    layout = w.layouts["a/"]
    w.data.qpos[layout.bq : layout.bq + 3] = [0.12, 0.0, 1.0]
    mujoco.mj_forward(w.model, w.data)
    row = w.advance({p: np.zeros(29) for p in w.layouts})
    # This synthetic body has no ankle-roll links, so its root contact is
    # correctly NONFOOT. A visually nearby foot must never be credited.
    assert row["ball_forces"][:, 0, 2].max() > 1
    assert not row["ball_forces"][:, 0, :2].any()
    assert not row["ball_forces"][:, 1].any()


def test_clone_preserves_shared_floor_assets_and_contact_parameters():
    source = (
        source_xml()
        .replace("</worldbody>", '<geom name="floor" type="plane" size="0 0 .1"/></worldbody>')
        .replace(
            "</mujoco>",
            '<contact><pair name="test" geom1="floor" geom2="floor" '
            'friction=".8 .8"/></contact></mujoco>',
        )
    )
    root = ET.fromstring(shared_native_scene(source, prefixes=("a/", "b/")))
    assert len(root.findall(".//body[@name='ball']")) == 1
    assert len(root.findall(".//geom[@name='floor']")) == 1
    assert len(root.findall("./actuator/motor")) == 58
    assert root.find("./contact/pair").get("friction") == ".8 .8"


def test_hidden_scene_section_and_duplicate_player_rejected():
    with pytest.raises(ValueError):
        shared_native_scene(source_xml(), prefixes=("a/", "a/"))
    with pytest.raises(ValueError, match="hidden"):
        shared_native_scene(
            source_xml().replace("</mujoco>", "<equality/></mujoco>"), prefixes=("a/",)
        )
