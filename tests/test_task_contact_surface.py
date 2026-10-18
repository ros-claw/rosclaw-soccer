import mujoco
import pytest

from rosclaw_soccer.physics.task_contact_surface import with_mesh_task_contacts


def specification(ball_x=0.18):
    return mujoco.MjSpec.from_string(f'''<mujoco><option timestep=".002"/>
    <asset><mesh name="foot_mesh" vertex="
    -.06 -.04 -.035 -.06 .04 -.035 .14 -.04 -.035 .14 .04 -.035
    -.06 -.04 .025 -.06 .04 .025 .14 -.04 .025 .14 .04 .025"/></asset>
    <worldbody><geom name="floor" type="plane" size="4 4 .1"/>
    <body name="foot" pos="0 0 .04"><freejoint/>
    <inertial mass=".608" pos="0 0 0" diaginertia=".003 .003 .003"/>
    <geom type="mesh" mesh="foot_mesh" contype="0" conaffinity="0" density="0"/>
    <geom name="heel_l" size=".005" pos="-.05 .025 -.03"/>
    <geom name="heel_r" size=".005" pos="-.05 -.025 -.03"/>
    <geom name="toe_l" size=".005" pos=".12 .03 -.03"/>
    <geom name="toe_r" size=".005" pos=".12 -.03 -.03"/></body>
    <body name="ball_body" pos="{ball_x} 0 .065"><freejoint/>
    <geom name="ball" size=".05" mass=".43"/></body></worldbody></mujoco>''')


def test_visible_surface_contact_not_present_in_four_point_source():
    source = specification()
    names = [g.name for g in source.geoms]
    modified, receipt = with_mesh_task_contacts(
        source, task_geom_name="ball", surface_body_names=("foot",)
    )
    assert names == [g.name for g in source.geoms]
    before, after = source.compile(), modified.compile()
    old, new = mujoco.MjData(before), mujoco.MjData(after)
    mujoco.mj_forward(before, old)
    mujoco.mj_forward(after, new)
    ball = int(after.geom("ball").id)
    assert not any(ball in c.geom for c in old.contact[: old.ncon])
    contacts = [c for c in new.contact[: new.ncon] if ball in c.geom]
    assert contacts
    mesh = int(after.geom("task_contact_surface_0").id)
    assert all(set(c.geom) == {ball, mesh} for c in contacts)
    assert after.ngeom == before.ngeom
    assert not receipt["contact_geometry_rescaled"]
    assert not receipt["physics_qualified"]


def test_explicit_pair_bypasses_foot_body_exclusion_but_support_spheres_do_not():
    source = specification(ball_x=0.16)
    # This pose overlaps a toe support sphere as well as the full foot mesh.
    source.body("ball_body").pos[:] = [0.16, 0.03, 0.025]
    modified, _ = with_mesh_task_contacts(
        source, task_geom_name="ball", surface_body_names=("foot",)
    )
    before, after = source.compile(), modified.compile()
    old, new = mujoco.MjData(before), mujoco.MjData(after)
    mujoco.mj_forward(before, old)
    mujoco.mj_forward(after, new)
    ball = int(after.geom("ball").id)
    assert any(ball in c.geom for c in old.contact[: old.ncon])
    contacts = [c for c in new.contact[: new.ncon] if ball in c.geom]
    assert contacts
    foot = int(after.body("foot").id)
    foot_contacts = [c for c in contacts if any(after.geom_bodyid[g] == foot for g in c.geom)]
    assert foot_contacts
    assert all(
        set(c.geom) == {ball, int(after.geom("task_contact_surface_0").id)} for c in foot_contacts
    )
    assert sum(int(after.geom("floor").id) in c.geom for c in contacts) == 1


def test_no_task_contact_ground_dynamics_are_byte_equal_for_1000_steps():
    source = specification(ball_x=3)
    modified, _ = with_mesh_task_contacts(
        source, task_geom_name="ball", surface_body_names=("foot",)
    )
    before, after = source.compile(), modified.compile()
    old, new = mujoco.MjData(before), mujoco.MjData(after)
    for _ in range(1000):
        mujoco.mj_step(before, old)
        mujoco.mj_step(after, new)
        for field in ("qpos", "qvel", "qacc", "qacc_warmstart", "qfrc_constraint"):
            assert getattr(old, field).tobytes() == getattr(new, field).tobytes()


def test_legacy_point_features_cannot_silently_claim_new_mesh_contact_geometry():
    from rosclaw_soccer.rsi.foot_ball_contact_features import measured_foot_ball_contact_features

    source = specification()
    source.body("foot").name = "left_ankle_roll_link"
    source.body("ball_body").name = "ball"
    source.geom("ball").name = "ball_geom"
    modified, _ = with_mesh_task_contacts(
        source, task_geom_name="ball_geom", surface_body_names=("left_ankle_roll_link",)
    )
    model = modified.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    with pytest.raises(ValueError, match="legacy four-point"):
        measured_foot_ball_contact_features(model, data)


@pytest.mark.parametrize("bodies", [(), ("missing",), ("foot", "foot"), ("world",), ("ball_body",)])
def test_invalid_surface_definition_rejected_without_source_mutation(bodies):
    source = specification()
    names = [g.name for g in source.geoms]
    with pytest.raises(ValueError):
        with_mesh_task_contacts(source, task_geom_name="ball", surface_body_names=bodies)
    assert names == [g.name for g in source.geoms]


def test_duplicate_installation_is_not_silently_accepted():
    modified, _ = with_mesh_task_contacts(
        specification(), task_geom_name="ball", surface_body_names=("foot",)
    )
    with pytest.raises(ValueError):
        with_mesh_task_contacts(modified, task_geom_name="ball", surface_body_names=("foot",))


def test_inflated_margin_or_ambiguous_mesh_is_refused():
    source = specification()
    source.geom("ball").margin = 0.01
    with pytest.raises(ValueError, match="inflation"):
        with_mesh_task_contacts(source, task_geom_name="ball", surface_body_names=("foot",))
    source = specification()
    source.body("foot").add_geom(
        type=mujoco.mjtGeom.mjGEOM_MESH, meshname="foot_mesh", contype=0, conaffinity=0
    )
    with pytest.raises(ValueError, match="one original"):
        with_mesh_task_contacts(source, task_geom_name="ball", surface_body_names=("foot",))
