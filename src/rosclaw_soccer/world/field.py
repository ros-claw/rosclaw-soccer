"""Deterministic native-MuJoCo football goal for qualified G1 scenes.

The qualified RoboNaldo scene ships with a heavy free box behind the target.
This module removes that box from a transient :class:`mujoco.MjSpec` and adds
a collision-capable training goal.  No external mesh or rendered pixel is
used for task scoring.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

_SCENE_REL = Path("g1_description/scene_with_ball.xml")
_MODEL_REL = Path("g1_description/g1_liao.xml")

# Conservative collision envelope for a visible adult goalkeeper glove.  The
# ellipsoid stays inside the 19 x 10 x 6.5 cm rendered glove instead of using
# the football radius as a hidden reach multiplier.
G1_GOALKEEPER_GLOVE_CENTER_M = (0.090, 0.0, 0.0)
G1_GOALKEEPER_GLOVE_HALF_EXTENTS_M = (0.095, 0.050, 0.0325)


@dataclass
class G1CompliantGoalNetState:
    """Target-independent state of a deformable three-axis goal-net pocket."""

    engaged: bool = False
    anchor_xyz_m: tuple[float, float, float] | None = None
    engagement_count: int = 0
    peak_force_n: float = 0.0
    peak_anchor_displacement_m: float = 0.0
    last_force_xyz_n: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def reset(self) -> None:
        self.engaged = False
        self.anchor_xyz_m = None
        self.last_force_xyz_n = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class G1TrainingGoalSpec:
    """Geometry and scoring contract for a humanoid-scale training goal."""

    plane_x_m: float = 5.0
    width_m: float = 2.4
    height_m: float = 1.6
    depth_m: float = 1.0
    post_radius_m: float = 0.035
    net_strand_radius_m: float = 0.00125
    target_y_m: float = 1.0
    target_z_m: float = 0.115
    precision_radius_m: float = 0.16
    ball_free_joint_damping_n_s_m: float = 0.02
    ball_angular_damping_n_m_s_rad: float = 0.00002
    ball_radius_m: float = 0.115
    ball_mass_kg: float = 0.41
    ball_contact_sliding_friction: float = 0.05
    ball_sliding_friction: float = 0.05
    ball_torsional_friction: float = 0.0003
    ball_rolling_friction: float = 0.00002
    regulation_field_enabled: bool = False
    field_length_m: float = 105.0
    field_width_m: float = 68.0
    field_line_width_m: float = 0.10
    goal_area_depth_m: float = 5.50
    penalty_area_depth_m: float = 16.50
    penalty_mark_distance_m: float = 11.0
    schema_version: str = "rosclaw.simforge.g1_training_goal_spec.v8"

    def __post_init__(self) -> None:
        values = (
            self.plane_x_m,
            self.width_m,
            self.height_m,
            self.depth_m,
            self.post_radius_m,
            self.net_strand_radius_m,
            self.target_y_m,
            self.target_z_m,
            self.precision_radius_m,
            self.ball_free_joint_damping_n_s_m,
            self.ball_angular_damping_n_m_s_rad,
            self.ball_radius_m,
            self.ball_mass_kg,
            self.ball_contact_sliding_friction,
            self.ball_sliding_friction,
            self.ball_torsional_friction,
            self.ball_rolling_friction,
            self.field_length_m,
            self.field_width_m,
            self.field_line_width_m,
            self.goal_area_depth_m,
            self.penalty_area_depth_m,
            self.penalty_mark_distance_m,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("training goal values must be finite")
        if not 4.0 <= self.plane_x_m <= 12.0:
            raise ValueError("training goal plane must be in [4, 12] m")
        if not 1.5 <= self.width_m <= 7.32:
            raise ValueError("training goal width must be in [1.5, 7.32] m")
        if not 1.0 <= self.height_m <= 2.44:
            raise ValueError("training goal height must be in [1.0, 2.44] m")
        if not 0.4 <= self.depth_m <= 2.0:
            raise ValueError("training goal depth must be in [0.4, 2.0] m")
        if not 0.02 <= self.post_radius_m <= 0.08:
            raise ValueError("training goal post radius must be in [0.02, 0.08] m")
        if not 0.001 <= self.net_strand_radius_m <= 0.006:
            raise ValueError("training goal net strand radius must be in [0.001, 0.006] m")
        if abs(self.target_y_m) > self.width_m / 2.0 - self.ball_radius_m:
            raise ValueError("training target must keep the whole ball inside the goal posts")
        if not self.ball_radius_m <= self.target_z_m <= self.height_m - self.ball_radius_m:
            raise ValueError("training target must keep the whole ball inside the goal height")
        if not 0.05 <= self.precision_radius_m <= 0.30:
            raise ValueError("precision radius must be in [0.05, 0.30] m")
        if not 0.001 <= self.ball_free_joint_damping_n_s_m <= 0.10:
            raise ValueError("ball free-joint damping must be in [0.001, 0.10] N s/m")
        if not 0.0 <= self.ball_angular_damping_n_m_s_rad <= 0.001:
            raise ValueError("ball angular damping must be in [0, 0.001] N m s/rad")
        if not 0.105 <= self.ball_radius_m <= 0.115:
            raise ValueError("football radius must be in [0.105, 0.115] m")
        if not 0.40 <= self.ball_mass_kg <= 0.46:
            raise ValueError("football mass must be in [0.40, 0.46] kg")
        if not 0.03 <= self.ball_contact_sliding_friction <= 0.80:
            raise ValueError("football contact sliding friction must be in [0.03, 0.80]")
        if not 0.03 <= self.ball_sliding_friction <= 0.80:
            raise ValueError("football-ground sliding friction must be in [0.03, 0.80]")
        if not 0.0 <= self.ball_torsional_friction <= 0.02:
            raise ValueError("football torsional friction must be in [0, 0.02]")
        if not 0.0 <= self.ball_rolling_friction <= 0.001:
            raise ValueError("football rolling friction must be in [0, 0.001]")
        if not isinstance(self.regulation_field_enabled, bool):
            raise ValueError("regulation field flag must be boolean")
        if self.regulation_field_enabled:
            if not 100.0 <= self.field_length_m <= 110.0:
                raise ValueError("international field length must be in [100, 110] m")
            if not 64.0 <= self.field_width_m <= 75.0:
                raise ValueError("international field width must be in [64, 75] m")
            if not 0.0 < self.field_line_width_m <= 0.12:
                raise ValueError("field line width must be in (0, 0.12] m")
            if (
                self.goal_area_depth_m != 5.50
                or self.penalty_area_depth_m != 16.50
                or self.penalty_mark_distance_m != 11.0
            ):
                raise ValueError("regulation penalty geometry does not match IFAB dimensions")

    @property
    def spec_hash(self) -> str:
        return str(hash_json(asdict(self)))

    @property
    def target_corner(self) -> str:
        """Declared corner using the kicker-facing +x, +y-is-left convention."""

        side = "left" if self.target_y_m >= 0.0 else "right"
        level = "upper" if self.target_z_m >= self.height_m / 2.0 else "lower"
        return f"{side}_{level}"

    @property
    def target_corner_center_m(self) -> tuple[float, float, float]:
        y = math.copysign(self.width_m / 2.0 - self.ball_radius_m, self.target_y_m)
        z = (
            self.height_m - self.ball_radius_m
            if "upper" in self.target_corner
            else self.ball_radius_m
        )
        return (self.plane_x_m, y, z)


def build_g1_stadium_model(asset_root: Path, spec: G1TrainingGoalSpec | None = None) -> Any:
    """Compile a qualified G1 scene with the wall replaced by a native goal."""

    import mujoco

    goal = spec or G1TrainingGoalSpec()
    parent = _stadium_spec(asset_root, goal)
    _add_goalkeeper_hand_envelopes(
        parent,
        body_prefix="",
        geom_prefix="",
        mujoco=mujoco,
    )
    model = parent.compile()
    _configure_ball_dof_damping(model, goal)
    _require_stadium_model(model)
    return model


def build_g1_coupled_stadium_model(
    asset_root: Path,
    *,
    passer_origin_m: tuple[float, float, float],
    passer_yaw_rad: float = math.pi,
    spec: G1TrainingGoalSpec | None = None,
) -> Any:
    """Compile the two-G1 replay scene with the same native football goal.

    This builder is intended for evidence-downstream visualization.  It keeps
    the original shooter and ball qpos ordering, then attaches the passer body
    exactly as the coupled physics model does.  Stored trajectories therefore
    remain the source of every rendered pose; the derived stadium does not
    change or rescore their physics.
    """

    goal = spec or G1TrainingGoalSpec()
    root = asset_root.expanduser().resolve()
    parent = _stadium_spec(root, goal)

    import mujoco

    child = mujoco.MjSpec.from_file(str(root / _MODEL_REL))
    frame = parent.worldbody.add_frame(
        name="passer_frame",
        pos=passer_origin_m,
        quat=(math.cos(0.5 * passer_yaw_rad), 0.0, 0.0, math.sin(0.5 * passer_yaw_rad)),
    )
    first_body = child.worldbody.first_body()
    if first_body is None:
        raise ValueError("qualified G1 model does not contain a root body")
    frame.attach_body(first_body, prefix="passer_")
    model = parent.compile()
    _configure_ball_dof_damping(model, goal)
    _require_stadium_model(model)
    if model.nu != 58:
        raise ValueError(f"coupled stadium model has {model.nu} actuators, expected 58")
    return model


def build_g1_three_player_stadium_model(
    asset_root: Path,
    *,
    passer_origin_m: tuple[float, float, float],
    passer_yaw_rad: float = math.pi,
    goalkeeper_origin_m: tuple[float, float, float],
    spec: G1TrainingGoalSpec | None = None,
) -> Any:
    """Compile one shared pitch for passer, shooter, goalkeeper and ball.

    The unprefixed source body remains the shooter.  The two attached bodies
    use the same qualified G1 model and face back toward the shooter.  This is
    a physical three-body scene, not a video compositing helper.
    """

    goal = spec or G1TrainingGoalSpec()
    root = asset_root.expanduser().resolve()
    parent = _stadium_spec(root, goal)

    import mujoco

    _attach_g1(
        parent,
        root=root,
        frame_name="passer_frame",
        prefix="passer_",
        origin_m=passer_origin_m,
        yaw_rad=passer_yaw_rad,
        mujoco=mujoco,
    )
    _attach_g1(
        parent,
        root=root,
        frame_name="goalkeeper_frame",
        prefix="goalkeeper_",
        origin_m=goalkeeper_origin_m,
        yaw_rad=math.pi,
        mujoco=mujoco,
    )
    _add_goalkeeper_hand_envelopes(
        parent,
        body_prefix="goalkeeper_",
        geom_prefix="goalkeeper_",
        mujoco=mujoco,
    )
    model = parent.compile()
    _configure_ball_dof_damping(model, goal)
    _require_stadium_model(model)
    if model.nu != 87:
        raise ValueError(f"three-player stadium model has {model.nu} actuators, expected 87")
    return model


def build_g1_four_player_two_ball_stadium_model(
    asset_root: Path,
    *,
    passer_origin_m: tuple[float, float, float],
    passer_yaw_rad: float = math.pi,
    goalkeeper_origin_m: tuple[float, float, float],
    second_striker_origin_m: tuple[float, float, float],
    first_ball_origin_m: tuple[float, float, float],
    second_ball_origin_m: tuple[float, float, float],
    second_ball_mass_kg: float | None = None,
    spec: G1TrainingGoalSpec | None = None,
) -> Any:
    """Compile a no-reset four-G1 stadium with a second physical football.

    This builder is infrastructure for a future second-striker exam. It does
    not claim that the fourth G1 has kicked the ball: that claim belongs to a
    rollout with ordered foot contact, post-contact speed gain and goal/save
    scoring. Both balls exist from model compilation onward, so consumers do
    not need to teleport or replace the first save's live ball.
    """

    goal = spec or G1TrainingGoalSpec()
    secondary_mass = goal.ball_mass_kg if second_ball_mass_kg is None else second_ball_mass_kg
    if not math.isfinite(secondary_mass) or not 0.40 <= secondary_mass <= 0.46:
        raise ValueError("second football mass is outside the regulation interval")
    root = asset_root.expanduser().resolve()
    parent = _stadium_spec(root, goal)

    import mujoco

    _attach_g1(
        parent,
        root=root,
        frame_name="passer_frame",
        prefix="passer_",
        origin_m=passer_origin_m,
        yaw_rad=passer_yaw_rad,
        mujoco=mujoco,
    )
    _attach_g1(
        parent,
        root=root,
        frame_name="goalkeeper_frame",
        prefix="goalkeeper_",
        origin_m=goalkeeper_origin_m,
        yaw_rad=math.pi,
        mujoco=mujoco,
    )
    _attach_g1(
        parent,
        root=root,
        frame_name="second_striker_frame",
        prefix="second_striker_",
        origin_m=second_striker_origin_m,
        yaw_rad=0.0,
        mujoco=mujoco,
    )
    _add_goalkeeper_hand_envelopes(
        parent,
        body_prefix="goalkeeper_",
        geom_prefix="goalkeeper_",
        mujoco=mujoco,
    )
    if (
        len(first_ball_origin_m) != 3
        or not all(math.isfinite(value) for value in first_ball_origin_m)
        or abs(first_ball_origin_m[1]) > goal.field_width_m / 2.0
        or first_ball_origin_m[2] < goal.ball_radius_m
    ):
        raise ValueError("first football origin is outside the declared pitch")
    first_ball = parent.body("ball")
    if first_ball is None:
        raise ValueError("four-player stadium source football is unavailable")
    first_ball.pos = first_ball_origin_m
    _add_secondary_football(
        parent,
        origin_m=second_ball_origin_m,
        spec=goal,
        mass_kg=secondary_mass,
        mujoco=mujoco,
    )
    model = parent.compile()
    _configure_ball_dof_damping(model, goal)
    _configure_ball_dof_damping(model, goal, joint_name="second_ball_free")
    _require_stadium_model(model)
    if model.nu != 116:
        raise ValueError(f"four-player stadium model has {model.nu} actuators, expected 116")
    if model.body("second_ball").id < 0 or model.geom("second_ball_geom").id < 0:
        raise ValueError("four-player stadium model lacks its second physical ball")
    return model


def _attach_g1(
    parent: Any,
    *,
    root: Path,
    frame_name: str,
    prefix: str,
    origin_m: tuple[float, float, float],
    yaw_rad: float,
    mujoco: Any,
) -> None:
    """Attach one qualified G1 without changing source ball/qpos ordering."""

    child = mujoco.MjSpec.from_file(str(root / _MODEL_REL))
    half_yaw = 0.5 * yaw_rad
    frame = parent.worldbody.add_frame(
        name=frame_name,
        pos=origin_m,
        quat=(math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)),
    )
    first_body = child.worldbody.first_body()
    if first_body is None:
        raise ValueError("qualified G1 model does not contain a root body")
    frame.attach_body(first_body, prefix=prefix)


def _add_secondary_football(
    parent: Any,
    *,
    origin_m: tuple[float, float, float],
    spec: G1TrainingGoalSpec,
    mass_kg: float,
    mujoco: Any,
) -> None:
    """Add one regulation-mass physical ball at a declared initial pose."""

    if (
        len(origin_m) != 3
        or not all(math.isfinite(value) for value in origin_m)
        or abs(origin_m[1]) > spec.field_width_m / 2.0
        or origin_m[2] < spec.ball_radius_m
    ):
        raise ValueError("second football origin is outside the declared pitch")
    body = parent.worldbody.add_body(name="second_ball", pos=origin_m)
    joint = body.add_freejoint(name="second_ball_free")
    try:
        joint.damping = 0.0
    except TypeError:
        joint.damping = (0.0, 0.0, 0.0)
    inertia = 0.4 * mass_kg * spec.ball_radius_m**2
    body.mass = mass_kg
    body.inertia = (inertia, inertia, inertia)
    # MjSpec otherwise infers the explicit inertial position from the body's
    # world spawn coordinates when mass/inertia are assigned programmatically.
    # That creates a metre-scale COM offset and turns a resting floor force
    # into a large rolling torque. The source ball's COM is at its body origin.
    body.ipos = (0.0, 0.0, 0.0)
    body.add_geom(
        name="second_ball_geom",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=(spec.ball_radius_m, 0.0, 0.0),
        rgba=(0.93, 0.93, 0.91, 1.0),
        density=0.0,
        contype=1,
        conaffinity=1,
        condim=3,
        solref=(0.001, 0.05),
        solimp=(0.95, 0.99, 0.001, 0.5, 2.0),
        friction=(
            spec.ball_contact_sliding_friction,
            spec.ball_torsional_friction,
            spec.ball_rolling_friction,
        ),
    )
    # Match the qualified source football's explicit six-dimensional floor
    # pair. Relying on an auto-generated condim=6 contact causes small
    # tangential solver noise to self-excite a resting sphere in MuJoCo; the
    # named pair keeps both footballs under the same rolling contract.
    parent.add_pair(
        name="second_ball_floor",
        geomname1="second_ball_geom",
        geomname2="floor",
        condim=6,
        solref=(0.001, 1.0),
        solimp=(0.95, 0.99, 0.001, 0.5, 2.0),
        friction=(
            spec.ball_sliding_friction,
            spec.ball_sliding_friction,
            spec.ball_torsional_friction,
            spec.ball_rolling_friction,
            spec.ball_rolling_friction,
        ),
    )


def _add_goalkeeper_hand_envelopes(
    parent: Any,
    *,
    body_prefix: str,
    geom_prefix: str,
    mujoco: Any,
) -> None:
    """Add visible, collision-faithful goalkeeper gloves to both wrists.

    The 0.19 m palm-to-fingertip length, 0.10 m width and 0.065 m thickness
    conservatively cover the source G1 hand while remaining inside the visible
    blue glove.  They are attached to the real wrist bodies, so the physics
    scorer can distinguish a hand save from a torso/leg block without adding
    an invisible reach advantage.
    """

    for side in ("left", "right"):
        body = parent.body(f"{body_prefix}{side}_wrist_yaw_link")
        if body is None:
            raise ValueError(f"qualified G1 is missing the goalkeeper {side} wrist")
        body.add_geom(
            name=f"{geom_prefix}{side}_goalkeeper_glove",
            type=mujoco.mjtGeom.mjGEOM_ELLIPSOID,
            pos=G1_GOALKEEPER_GLOVE_CENTER_M,
            # ``size`` stores ellipsoid half-extents.  These dimensions are a
            # collision inset, not a ball-radius-wide reach paddle.
            size=G1_GOALKEEPER_GLOVE_HALF_EXTENTS_M,
            rgba=(0.03, 0.22, 0.92, 1.0),
            friction=(0.8, 0.005, 0.0001),
            contype=1,
            conaffinity=1,
        )


def _stadium_spec(asset_root: Path, spec: G1TrainingGoalSpec) -> Any:
    import mujoco

    scene = asset_root.expanduser().resolve() / _SCENE_REL
    parent = mujoco.MjSpec.from_file(str(scene))
    wall = parent.body("box")
    if wall is None:
        raise ValueError("qualified G1 scene does not contain the replaceable box body")
    # MuJoCo renamed the MjSpec removal operation: 3.3 exposes
    # ``detach_body`` while current releases expose generic ``delete``.
    # Keep the qualified scene valid at both the declared floor and the
    # modern MJWarp toolchain instead of pinning either side to an obsolete
    # editor API.
    if hasattr(parent, "detach_body"):
        parent.detach_body(wall)
    elif hasattr(parent, "delete"):
        parent.delete(wall)
    else:
        raise RuntimeError("MuJoCo MjSpec cannot remove the replaceable box body")
    _style_pitch_and_ball(parent, spec)
    _add_goal(parent, spec)
    return parent


def _require_stadium_model(model: Any) -> None:
    import mujoco

    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box") >= 0:
        raise AssertionError("stadium scene retained the original wall")
    for name in ("goal_left_post", "goal_right_post", "goal_crossbar", "goal_back_net"):
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) < 0:
            raise AssertionError(f"stadium scene is missing {name}")


def _configure_ball_dof_damping(
    model: Any,
    spec: G1TrainingGoalSpec,
    *,
    joint_name: str = "ball_free",
) -> None:
    """Separate translational drag from rotational drag after compilation.

    MuJoCo's free-joint ``damping`` shortcut expands one scalar across all six
    degrees of freedom.  Applying the football's linear value to rotation is
    dimensionally wrong and erases rolling spin in roughly a tenth of a
    second.  The compiled model exposes the six physical DOFs explicitly, so
    bind their two units here and fail closed if the qualified scene changes.
    """

    ball_joint = model.joint(joint_name)
    addresses = tuple(int(value) for value in ball_joint.dofadr)
    if len(addresses) != 1:
        raise ValueError(f"qualified stadium {joint_name} does not have one DOF address")
    dof_address = addresses[0]
    if dof_address < 0 or dof_address + 6 > int(model.nv):
        raise ValueError(f"qualified stadium {joint_name} does not expose six DOFs")
    model.dof_damping[dof_address : dof_address + 3] = spec.ball_free_joint_damping_n_s_m
    model.dof_damping[dof_address + 3 : dof_address + 6] = spec.ball_angular_damping_n_m_s_rad


def g1_stadium_scene_hash(asset_root: Path, spec: G1TrainingGoalSpec | None = None) -> str:
    """Bind the derived scene to both its source XML and declarative goal spec."""

    goal = spec or G1TrainingGoalSpec()
    scene = asset_root.expanduser().resolve() / _SCENE_REL
    return str(
        hash_json(
            {
                "source_scene_hash": hash_bytes(scene.read_bytes()),
                "goal_spec_hash": goal.spec_hash,
                "builder_hash": hash_bytes(Path(__file__).read_bytes()),
            }
        )
    )


def g1_goal_net_contact_plane_x(
    spec: G1TrainingGoalSpec,
    *,
    capture_depth_m: float,
    ball_z_m: float,
) -> float:
    """Return the ball-centre contact plane aligned to the sloped visible net."""

    if not math.isfinite(capture_depth_m) or not 0.20 <= capture_depth_m <= spec.depth_m:
        raise ValueError("goal net capture depth must be inside the visible net")
    normalized_height = min(1.0, max(0.0, ball_z_m / spec.height_m))
    visible_depth = spec.depth_m * (1.0 - 0.32 * normalized_height)
    selected_depth = min(capture_depth_m, visible_depth)
    return spec.plane_x_m + selected_depth - spec.ball_radius_m


def g1_ball_inside_goal_mouth(
    spec: G1TrainingGoalSpec,
    *,
    ball_y_m: float,
    ball_z_m: float,
) -> bool:
    """Return whether the complete ball fits through the scoring aperture."""

    if not math.isfinite(ball_y_m) or not math.isfinite(ball_z_m):
        return False
    return bool(
        abs(ball_y_m) <= spec.width_m / 2.0 - spec.ball_radius_m
        and spec.ball_radius_m <= ball_z_m <= spec.height_m - spec.ball_radius_m
    )


def apply_g1_compliant_goal_net_force(
    data: Any,
    *,
    ball_body_id: int,
    ball_qpos: int,
    ball_qvel: int,
    spec: G1TrainingGoalSpec,
    capture_depth_m: float,
    stiffness_n_m: float,
    damping_n_s_m: float,
    state: G1CompliantGoalNetState | None = None,
) -> None:
    """Apply a bounded spring-damper force matching the visible goal net.

    Passing ``state`` upgrades the legacy depth-only damping to a deformable
    three-axis pocket.  Its anchor is the first physical net-contact point and
    is independent of the requested scoring target, so capture cannot improve
    accuracy.  Omitting ``state`` preserves the stateless compatibility path.
    """

    if not 10.0 <= stiffness_n_m <= 250.0:
        raise ValueError("goal net stiffness must be in [10, 250] N/m")
    if not 2.0 <= damping_n_s_m <= 30.0:
        raise ValueError("goal net damping must be in [2, 30] N s/m")
    data.xfrc_applied[ball_body_id, :] = 0.0
    if state is not None:
        state.last_force_xyz_n = (0.0, 0.0, 0.0)
    x, y, z = (float(value) for value in data.qpos[ball_qpos : ball_qpos + 3])
    vx, vy, vz = (float(value) for value in data.qvel[ball_qvel : ball_qvel + 3])
    capture_x = g1_goal_net_contact_plane_x(
        spec,
        capture_depth_m=capture_depth_m,
        ball_z_m=z,
    )
    if state is not None and state.engaged and x < spec.plane_x_m - spec.ball_radius_m:
        state.reset()
    if state is not None and state.engaged:
        if state.anchor_xyz_m is None:
            raise RuntimeError("engaged goal net is missing its physical anchor")
        anchor_x, anchor_y, anchor_z = state.anchor_xyz_m
        displacement = (x - anchor_x, y - anchor_y, z - anchor_z)
        fx = -stiffness_n_m * displacement[0] - damping_n_s_m * vx
        fy = -0.35 * stiffness_n_m * displacement[1] - 0.55 * damping_n_s_m * vy
        fz = -0.35 * stiffness_n_m * displacement[2] - 0.55 * damping_n_s_m * vz
        force = (
            max(-250.0, min(250.0, fx)),
            max(-250.0, min(250.0, fy)),
            max(-250.0, min(250.0, fz)),
        )
        data.xfrc_applied[ball_body_id, :3] = force
        state.last_force_xyz_n = force
        state.peak_force_n = max(
            state.peak_force_n,
            math.sqrt(sum(value * value for value in force)),
        )
        state.peak_anchor_displacement_m = max(
            state.peak_anchor_displacement_m,
            math.sqrt(sum(value * value for value in displacement)),
        )
        return
    if not g1_ball_inside_goal_mouth(spec, ball_y_m=y, ball_z_m=z):
        return
    if x <= capture_x:
        if x > spec.plane_x_m and vx < 0.0:
            data.xfrc_applied[ball_body_id, :3] = (
                min(250.0, -damping_n_s_m * vx),
                max(-250.0, min(250.0, -0.12 * damping_n_s_m * vy)),
                max(-250.0, min(250.0, -0.08 * damping_n_s_m * vz)),
            )
        return

    if state is not None:
        state.engaged = True
        state.anchor_xyz_m = (capture_x, y, z)
        state.engagement_count += 1
        return

    penetration = x - capture_x
    engagement = min(1.0, max(0.0, penetration / 0.10))
    fx = -stiffness_n_m * penetration - engagement * damping_n_s_m * vx
    fy = -0.20 * engagement * damping_n_s_m * vy
    fz = -0.12 * engagement * damping_n_s_m * vz
    side_limit = spec.width_m / 2.0 - spec.ball_radius_m
    side_penetration = abs(y) - side_limit
    if x > spec.plane_x_m and side_penetration > 0.0:
        fy -= math.copysign(stiffness_n_m * side_penetration, y) + damping_n_s_m * vy
    roof_z = spec.height_m - spec.ball_radius_m
    if x > spec.plane_x_m and z > roof_z:
        fz -= stiffness_n_m * (z - roof_z) + damping_n_s_m * max(0.0, vz)
    data.xfrc_applied[ball_body_id, :3] = tuple(
        max(-250.0, min(250.0, value)) for value in (fx, fy, fz)
    )


def _add_goal(parent: Any, spec: G1TrainingGoalSpec) -> None:
    import mujoco

    world = parent.worldbody
    x = spec.plane_x_m
    half_width = spec.width_m / 2.0
    radius = spec.post_radius_m
    white = (0.98, 0.98, 0.96, 1.0)

    def capsule(
        name: str,
        start: tuple[float, float, float],
        end: tuple[float, float, float],
        *,
        rgba: tuple[float, float, float, float] = white,
        collision: bool = True,
        strand: bool = False,
        custom_radius: float | None = None,
    ) -> None:
        world.add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            fromto=(*start, *end),
            size=(
                custom_radius
                if custom_radius is not None
                else spec.net_strand_radius_m
                if strand
                else radius,
                0.0,
                0.0,
            ),
            rgba=rgba,
            contype=1 if collision else 0,
            conaffinity=1 if collision else 0,
        )

    capsule("goal_left_post", (x, -half_width, 0.0), (x, -half_width, spec.height_m))
    capsule("goal_right_post", (x, half_width, 0.0), (x, half_width, spec.height_m))
    capsule("goal_crossbar", (x, -half_width, spec.height_m), (x, half_width, spec.height_m))
    rear_bottom_x = x + spec.depth_m
    rear_top_x = x + spec.depth_m * 0.68
    support_radius = radius * 0.58
    capsule(
        "goal_left_back_post",
        (rear_bottom_x, -half_width, 0.0),
        (rear_top_x, -half_width, spec.height_m),
        custom_radius=support_radius,
    )
    capsule(
        "goal_right_back_post",
        (rear_bottom_x, half_width, 0.0),
        (rear_top_x, half_width, spec.height_m),
        custom_radius=support_radius,
    )
    capsule(
        "goal_left_depth_bar",
        (x, -half_width, spec.height_m),
        (rear_top_x, -half_width, spec.height_m),
        custom_radius=support_radius,
    )
    capsule(
        "goal_right_depth_bar",
        (x, half_width, spec.height_m),
        (rear_top_x, half_width, spec.height_m),
        custom_radius=support_radius,
    )
    capsule(
        "goal_back_crossbar",
        (rear_top_x, -half_width, spec.height_m),
        (rear_top_x, half_width, spec.height_m),
        custom_radius=support_radius,
    )
    capsule(
        "goal_back_ground_bar",
        (rear_bottom_x, -half_width, 0.025),
        (rear_bottom_x, half_width, 0.025),
        custom_radius=support_radius,
    )

    # A fine sloped net is visual geometry. A deterministic compliant force
    # field in the rollout models capture without the rigid-wall rebound of a
    # transparent box.
    # Thin half-metre visual mesh makes depth legible without creating any
    # collision wall. Ball retention comes only from the compliant force field.
    net = (0.91, 0.94, 0.92, 0.22)
    vertical_count = max(7, int(math.ceil(spec.width_m / 0.50)) + 1)
    horizontal_count = max(5, int(math.ceil(spec.height_m / 0.42)) + 1)
    for index in range(vertical_count):
        y = -half_width + spec.width_m * index / (vertical_count - 1)
        capsule(
            "goal_back_net" if index == vertical_count // 2 else f"goal_back_net_v_{index}",
            (rear_bottom_x, y, 0.03),
            (rear_top_x, y, spec.height_m),
            rgba=net,
            collision=False,
            strand=True,
        )
    for index in range(horizontal_count):
        z = spec.height_m * index / (horizontal_count - 1)
        rear_x = rear_bottom_x + (rear_top_x - rear_bottom_x) * z / spec.height_m
        capsule(
            f"goal_back_net_h_{index}",
            (rear_x, -half_width, max(0.025, z)),
            (rear_x, half_width, max(0.025, z)),
            rgba=net,
            collision=False,
            strand=True,
        )
    for side, y in (("left", -half_width), ("right", half_width)):
        side_horizontal_count = max(4, int(math.ceil(spec.height_m / 0.42)) + 1)
        for index in range(side_horizontal_count):
            z = spec.height_m * index / (side_horizontal_count - 1)
            rear_x = rear_bottom_x + (rear_top_x - rear_bottom_x) * z / spec.height_m
            capsule(
                f"goal_{side}_net_h_{index}",
                (x, y, max(0.025, z)),
                (rear_x, y, max(0.025, z)),
                rgba=net,
                collision=False,
                strand=True,
            )
        side_vertical_count = max(3, int(math.ceil(spec.depth_m / 0.50)) + 1)
        for index in range(side_vertical_count):
            alpha = index / (side_vertical_count - 1)
            net_x_bottom = x + alpha * (rear_bottom_x - x)
            net_x_top = x + alpha * (rear_top_x - x)
            capsule(
                f"goal_{side}_net_v_{index}",
                (net_x_bottom, y, 0.025),
                (net_x_top, y, spec.height_m),
                rgba=net,
                collision=False,
                strand=True,
            )
    for index in range(vertical_count):
        y = -half_width + spec.width_m * index / (vertical_count - 1)
        capsule(
            f"goal_roof_net_{index}",
            (x, y, spec.height_m),
            (rear_top_x, y, spec.height_m),
            rgba=net,
            collision=False,
            strand=True,
        )


def _style_pitch_and_ball(parent: Any, spec: G1TrainingGoalSpec) -> None:
    """Replace the blue grid with a pitch and add a lightweight ball pattern."""

    import mujoco

    floor = parent.geom("floor")
    floor.material = ""
    floor.rgba = (0.055, 0.24, 0.075, 1.0)
    world = parent.worldbody
    pitch_start_x = spec.plane_x_m - spec.field_length_m if spec.regulation_field_enabled else -5.0
    pitch_end_x = (
        spec.plane_x_m if spec.regulation_field_enabled else spec.plane_x_m + spec.depth_m + 1.0
    )
    pitch_half_width = spec.field_width_m / 2.0 if spec.regulation_field_enabled else 4.2
    stripe_count = 20 if spec.regulation_field_enabled else 10
    stripe_width = (pitch_end_x - pitch_start_x) / stripe_count
    for index in range(stripe_count):
        shade = 0.105 if index % 2 == 0 else 0.082
        world.add_geom(
            name=f"pitch_mowing_stripe_{index}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=(pitch_start_x + (index + 0.5) * stripe_width, 0.0, 0.0015),
            size=(stripe_width / 2.0, pitch_half_width, 0.0015),
            rgba=(0.035, shade + 0.12, 0.052, 1.0),
            contype=0,
            conaffinity=0,
        )
    line = (0.93, 0.94, 0.90, 0.92)
    line_half = spec.field_line_width_m / 2.0 if spec.regulation_field_enabled else 0.018
    box_depth = (
        spec.goal_area_depth_m
        if spec.regulation_field_enabled
        else min(2.2, max(1.4, spec.plane_x_m - 2.0))
    )
    box_half_width = (
        spec.width_m / 2.0 + spec.goal_area_depth_m if spec.regulation_field_enabled else 2.8
    )
    for name, pos, size in (
        (
            "pitch_goal_line",
            (spec.plane_x_m, 0.0, 0.004),
            (line_half, pitch_half_width, 0.003),
        ),
        (
            "pitch_box_front",
            (spec.plane_x_m - box_depth, 0.0, 0.004),
            (line_half, box_half_width, 0.003),
        ),
        (
            "pitch_box_left",
            (spec.plane_x_m - box_depth / 2.0, -box_half_width, 0.004),
            (box_depth / 2.0, line_half, 0.003),
        ),
        (
            "pitch_box_right",
            (spec.plane_x_m - box_depth / 2.0, box_half_width, 0.004),
            (box_depth / 2.0, line_half, 0.003),
        ),
    ):
        world.add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=pos,
            size=size,
            rgba=line,
            contype=0,
            conaffinity=0,
        )
    if spec.regulation_field_enabled:
        penalty_half_width = spec.width_m / 2.0 + spec.penalty_area_depth_m
        for name, pos, size in (
            (
                "pitch_penalty_front",
                (spec.plane_x_m - spec.penalty_area_depth_m, 0.0, 0.004),
                (line_half, penalty_half_width, 0.003),
            ),
            (
                "pitch_penalty_left",
                (
                    spec.plane_x_m - spec.penalty_area_depth_m / 2.0,
                    -penalty_half_width,
                    0.004,
                ),
                (spec.penalty_area_depth_m / 2.0, line_half, 0.003),
            ),
            (
                "pitch_penalty_right",
                (
                    spec.plane_x_m - spec.penalty_area_depth_m / 2.0,
                    penalty_half_width,
                    0.004,
                ),
                (spec.penalty_area_depth_m / 2.0, line_half, 0.003),
            ),
        ):
            world.add_geom(
                name=name,
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=pos,
                size=size,
                rgba=line,
                contype=0,
                conaffinity=0,
            )
        world.add_geom(
            name="pitch_penalty_mark",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            pos=(spec.plane_x_m - spec.penalty_mark_distance_m, 0.0, 0.006),
            size=(0.11, 0.004, 0.0),
            rgba=line,
            contype=0,
            conaffinity=0,
        )
    ball = parent.body("ball")
    ball_geom = parent.geom("ball_geom")
    if ball_geom is None:
        raise ValueError("qualified stadium is missing ball_geom")
    ball_joints = list(ball.joints)
    if len(ball_joints) != 1 or ball_joints[0].name != "ball_free":
        raise ValueError("qualified stadium ball must expose exactly one ball_free joint")
    # Clear the upstream scalar shortcut.  Linear and angular damping have
    # different units and are assigned to compiled DOFs by
    # ``_configure_ball_dof_damping``.
    try:
        ball_joints[0].damping = 0.0
    except TypeError:
        # MuJoCo >= 3.10 exposes the joint default triple as an ndarray,
        # whereas the 3.3 API accepts the scalar XML shortcut directly.
        ball_joints[0].damping = (0.0, 0.0, 0.0)
    radius = spec.ball_radius_m
    inertia = 0.4 * spec.ball_mass_kg * radius * radius
    ball.mass = spec.ball_mass_kg
    ball.inertia = (inertia, inertia, inertia)
    ball_geom.size = (radius, 0.0, 0.0)
    ball_geom.friction = (
        spec.ball_contact_sliding_friction,
        spec.ball_torsional_friction,
        spec.ball_rolling_friction,
    )
    ball_floor = parent.pair("ball_floor")
    if ball_floor is None:
        raise ValueError("qualified stadium is missing ball_floor contact pair")
    ball_floor.condim = 6
    ball_floor.friction = (
        spec.ball_sliding_friction,
        spec.ball_sliding_friction,
        spec.ball_torsional_friction,
        spec.ball_rolling_friction,
        spec.ball_rolling_friction,
    )
    patch_distance = radius - 0.003
    patch_radius = radius * 0.19
    patch_depth = radius * 0.025
    patches = (
        ((patch_distance, 0.0, 0.0), (math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0)),
        ((-patch_distance, 0.0, 0.0), (math.sqrt(0.5), 0.0, -math.sqrt(0.5), 0.0)),
        ((0.0, patch_distance, 0.0), (math.sqrt(0.5), -math.sqrt(0.5), 0.0, 0.0)),
        ((0.0, -patch_distance, 0.0), (math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0)),
        ((0.0, 0.0, patch_distance), (1.0, 0.0, 0.0, 0.0)),
    )
    for index, (position, quaternion) in enumerate(patches):
        ball.add_geom(
            name=f"ball_patch_{index}",
            type=mujoco.mjtGeom.mjGEOM_ELLIPSOID,
            pos=position,
            quat=quaternion,
            size=(patch_radius, patch_radius, patch_depth),
            rgba=(0.025, 0.025, 0.025, 1.0),
            density=0.0,
            contype=0,
            conaffinity=0,
        )


__all__ = [
    "G1CompliantGoalNetState",
    "G1TrainingGoalSpec",
    "apply_g1_compliant_goal_net_force",
    "build_g1_coupled_stadium_model",
    "build_g1_four_player_two_ball_stadium_model",
    "build_g1_stadium_model",
    "build_g1_three_player_stadium_model",
    "g1_stadium_scene_hash",
    "g1_goal_net_contact_plane_x",
]
