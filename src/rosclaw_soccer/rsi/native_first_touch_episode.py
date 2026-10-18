"""Native CPU first-touch physics kernel, separate from exam admission.

Consumed/development diagnostics and explicitly declared public training are
accepted. This module cannot open a private Fresh pool or authorize promotion. Its caller must pin
and validate entry arguments, source and policy factories before allocation.
The legacy CLI retains its original consumed-course guard.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.providers.g1.sonic_navigation import G1SonicNavigation, SonicNavigationConfig
from rosclaw_soccer.providers.g1.sonic_torch import FrozenSonicG1Torch
from rosclaw_soccer.providers.g1.sonic_vector import BatchedSonicTracker
from rosclaw_soccer.rsi.bounded_first_touch_control import BoundedFirstTouchControl
from rosclaw_soccer.rsi.contact_motor_contract import load_policy, motor_delta
from rosclaw_soccer.rsi.contact_time_phase_features import (
    current_context,
    gait_phase_features,
    predict_contact_time,
)
from rosclaw_soccer.rsi.fast_intercept_control import FastInterceptControl
from rosclaw_soccer.rsi.late_swing_memory import load_late_swing_actor
from rosclaw_soccer.rsi.online_motor_actor_critic import configure_preview, validate_model
from rosclaw_soccer.rsi.postcontact_navigation_probe import PostcontactNavigationProbe
from rosclaw_soccer.rsi.public_first_touch_training_plan import PublicFirstTouchTrainingPlan
from rosclaw_soccer.rsi.reviewed_contact_world import (
    ReviewedContactWorld,
    verify_contact_world_request,
)
from rosclaw_soccer.rsi.sampling_model_io import load_sampling_model
from rosclaw_soccer.rsi.taskspace_gate_memory import select_taskspace_gate
from rosclaw_soccer.rsi.taskspace_swing_evidence import LEG_NAMES
from rosclaw_soccer.rsi.taskspace_swing_probe import (
    choose_swing_side,
    continuous_swing_joint_delta,
    release_joint_delta,
    swing_joint_delta,
)
from rosclaw_soccer.sim.contracts import G1_HARD_TORQUE_LIMITS, hash_bytes, hash_json
from rosclaw_soccer.sim.current_kinematic_observation import (
    CurrentKinematicObserver,
)
from rosclaw_soccer.sim.current_kinematic_observation import (
    observation_contract as make_observation_contract,
)
from rosclaw_soccer.sim.root_velocity_reference import root_velocity_world
from rosclaw_soccer.skills.team.motor_option import TeamMotorObservation


def run_native_first_touch_episode(
    args: argparse.Namespace,
    *,
    course: tuple[float, float, float],
    partition: str,
    entry_source_path: Path,
    consumed_bank_hash: str | None,
    sampling_factory: Any = None,
    extended_factory: Any = None,
    proposal_factory: Any = None,
    imitation_factory: Any = None,
    recurrent_factory: Any = None,
    recurrent_sampling_factory: Any = None,
    recurrent_clipped_factory: Any = None,
    approach_mode: str = "legacy",
    swing_mode: str = "legacy",
    training_plan: PublicFirstTouchTrainingPlan | None = None,
    training_group: int | None = None,
    control_profile: BoundedFirstTouchControl | None = None,
    postcontact_profile: PostcontactNavigationProbe | None = None,
    locomotion_mode_probe: int | None = None,
    foundation_approach_probe: bool = False,
    reference_height_probe_m: float | None = None,
    record_default_planner_events: bool = False,
    cerebellar_model: dict[str, Any] | None = None,
    cerebellar_sampling_seed: int | None = None,
    mesh_task_contact_world_hash: str | None = None,
    mesh_world_review: ReviewedContactWorld | None = None,
    generated_seed_episode_binding: bool = False,
    suppress_postcontact_swing_release: bool = False,
    use_contact_exit_torque_mpc: bool = False,
    use_directional_contact_compliance: bool = False,
) -> None:
    """Execute a pinned diagnostic; no RNG course generation or admission here."""
    from rosclaw_soccer.rsi.native_cerebellar_torque import validate_native_request

    if type(use_directional_contact_compliance) is not bool:
        raise ValueError("explicit boolean directional contact compliance required")
    if use_directional_contact_compliance:
        from rosclaw_soccer.rsi.recurrent_sampling_episode_factory import (
            RecurrentSamplingEpisodeFactory,
        )

        if (
            partition != "DECLARED_DEVELOPMENT_DIAGNOSTIC"
            or getattr(args, "record_foundation_observation", False) is not True
            or getattr(args, "foundation_only", False) is not False
            or getattr(args, "proposal_decoder", None) != "reference"
            or getattr(args, "step_model", None) is None
            or getattr(args, "body_response_bundle", None) is not None
            or getattr(args, "cached_proposal_envelope", False)
            or getattr(args, "neural_model", None) is not None
            or getattr(args, "motor_policy", None) is not None
            or type(control_profile) is not BoundedFirstTouchControl
            or approach_mode != "bounded_approach_diagnostic"
            or swing_mode != "bounded_contact_diagnostic"
            or mesh_task_contact_world_hash is None
            or mesh_world_review is None
            or type(recurrent_sampling_factory) is not RecurrentSamplingEpisodeFactory
            or cerebellar_model is not None
            or cerebellar_sampling_seed is not None
            or postcontact_profile is not None
            or locomotion_mode_probe is not None
            or reference_height_probe_m is not None
            or foundation_approach_probe is not False
            or record_default_planner_events is not True
            or generated_seed_episode_binding is not False
            or suppress_postcontact_swing_release is not False
            or use_contact_exit_torque_mpc is not False
            or any(
                factory is not None
                for factory in (
                    sampling_factory,
                    extended_factory,
                    proposal_factory,
                    imitation_factory,
                    recurrent_factory,
                    recurrent_clipped_factory,
                )
            )
        ):
            raise ValueError("unmixed captured reviewed-mesh directional compliance study required")
        recurrent_sampling_factory._stable()

    if type(use_contact_exit_torque_mpc) is not bool:
        raise ValueError("explicit boolean contact-exit torque teacher required")
    if use_contact_exit_torque_mpc:
        from rosclaw_soccer.rsi.recurrent_sampling_episode_factory import (
            RecurrentSamplingEpisodeFactory,
        )

        if (
            partition != "DECLARED_DEVELOPMENT_DIAGNOSTIC"
            or getattr(args, "record_foundation_observation", False) is not True
            or getattr(args, "foundation_only", False) is not False
            or getattr(args, "proposal_decoder", None) != "reference"
            or getattr(args, "step_model", None) is None
            or getattr(args, "body_response_bundle", None) is not None
            or getattr(args, "cached_proposal_envelope", False)
            or getattr(args, "neural_model", None) is not None
            or getattr(args, "motor_policy", None) is not None
            or type(control_profile) is not BoundedFirstTouchControl
            or approach_mode != "bounded_approach_diagnostic"
            or swing_mode != "bounded_contact_diagnostic"
            or mesh_task_contact_world_hash is None
            or mesh_world_review is None
            or type(recurrent_sampling_factory) is not RecurrentSamplingEpisodeFactory
            or cerebellar_model is not None
            or cerebellar_sampling_seed is not None
            or postcontact_profile is not None
            or locomotion_mode_probe is not None
            or reference_height_probe_m is not None
            or foundation_approach_probe is not False
            or generated_seed_episode_binding is not False
            or suppress_postcontact_swing_release is not False
            or any(
                factory is not None
                for factory in (
                    sampling_factory,
                    extended_factory,
                    proposal_factory,
                    imitation_factory,
                    recurrent_factory,
                    recurrent_clipped_factory,
                )
            )
        ):
            raise ValueError("unmixed captured reviewed-mesh bounded contact-exit study required")
        recurrent_sampling_factory._stable()

    if type(suppress_postcontact_swing_release) is not bool:
        raise ValueError("explicit boolean postcontact swing ablation required")
    if suppress_postcontact_swing_release:
        from rosclaw_soccer.rsi.recurrent_sampling_episode_factory import (
            RecurrentSamplingEpisodeFactory,
        )

        if (
            partition != "DECLARED_DEVELOPMENT_DIAGNOSTIC"
            or getattr(args, "record_foundation_observation", False) is not True
            or getattr(args, "foundation_only", False) is not False
            or getattr(args, "proposal_decoder", None) != "reference"
            or getattr(args, "step_model", None) is None
            or getattr(args, "body_response_bundle", None) is not None
            or getattr(args, "cached_proposal_envelope", False)
            or type(control_profile) is not BoundedFirstTouchControl
            or swing_mode != "bounded_contact_diagnostic"
            or mesh_task_contact_world_hash is None
            or mesh_world_review is None
            or type(recurrent_sampling_factory) is not RecurrentSamplingEpisodeFactory
            or cerebellar_model is not None
            or cerebellar_sampling_seed is not None
            or postcontact_profile is not None
            or locomotion_mode_probe is not None
            or reference_height_probe_m is not None
            or foundation_approach_probe is not False
            or generated_seed_episode_binding is not False
            or any(
                factory is not None
                for factory in (
                    sampling_factory,
                    extended_factory,
                    proposal_factory,
                    imitation_factory,
                    recurrent_factory,
                    recurrent_clipped_factory,
                )
            )
        ):
            raise ValueError("unmixed captured reviewed-mesh bounded release ablation required")
        recurrent_sampling_factory._stable()

    if type(generated_seed_episode_binding) is not bool:
        raise ValueError("explicit boolean private generated episode binding required")
    if generated_seed_episode_binding:
        from rosclaw_soccer.rsi.recurrent_sampling_episode_factory import (
            RecurrentSamplingEpisodeFactory,
        )

        if (
            type(recurrent_sampling_factory) is not RecurrentSamplingEpisodeFactory
            or partition != "DECLARED_DEVELOPMENT_DIAGNOSTIC"
            or getattr(args, "record_foundation_observation", False) is not True
            or getattr(args, "foundation_only", False) is not False
            or getattr(args, "proposal_decoder", None) != "reference"
            or getattr(args, "step_model", None) is None
            or getattr(args, "cached_proposal_envelope", False)
            or getattr(args, "body_response_bundle", None) is not None
            or any(
                factory is not None
                for factory in (
                    sampling_factory,
                    extended_factory,
                    proposal_factory,
                    imitation_factory,
                    recurrent_factory,
                    recurrent_clipped_factory,
                )
            )
        ):
            raise ValueError("captured unmixed development private generated binding required")
        try:
            recurrent_sampling_factory._stable()
        except AttributeError as error:
            raise ValueError("allocated original private factory required") from error

    if mesh_task_contact_world_hash is not None and (
        type(mesh_task_contact_world_hash) is not str
        or re.fullmatch(r"sha256:[0-9a-f]{64}", mesh_task_contact_world_hash) is None
        or partition != "DECLARED_DEVELOPMENT_DIAGNOSTIC"
        or getattr(args, "record_foundation_observation", False) is not True
        or (
            (cerebellar_model is not None or cerebellar_sampling_seed is not None)
            and mesh_world_review is None
        )
    ):
        raise ValueError("new mesh contact world requires an explicit captured SIM diagnostic hash")
    mesh_review_binding = verify_contact_world_request(
        mesh_world_review,
        world_hash=mesh_task_contact_world_hash,
        neural_requested=cerebellar_model is not None or cerebellar_sampling_seed is not None,
    )
    validate_native_request(
        cerebellar_model,
        cerebellar_sampling_seed,
        partition=partition,
        captured=getattr(args, "record_foundation_observation", False),
        foundation_only=getattr(args, "foundation_only", False),
        recurrent_factory=recurrent_sampling_factory,
        mixed_probe=(
            postcontact_profile is not None
            or locomotion_mode_probe is not None
            or reference_height_probe_m is not None
            or foundation_approach_probe is not False
            or getattr(args, "body_response_bundle", None) is not None
        ),
    )
    if (
        mesh_task_contact_world_hash is not None
        and cerebellar_model is not None
        and cerebellar_model["world_hash"] != mesh_task_contact_world_hash
    ):
        raise ValueError("neural model cannot inherit or retag a different physical world")
    if type(record_default_planner_events) is not bool or (
        record_default_planner_events and locomotion_mode_probe is not None
    ):
        raise ValueError("default reference capture cannot mix with a locomotion intervention")
    fast_intercept = type(control_profile) is FastInterceptControl
    if fast_intercept and (
        foundation_approach_probe is not True
        or locomotion_mode_probe != 3
        or reference_height_probe_m != 0.75
    ):
        raise ValueError("fast interception requires captured pure Foundation run/height probe")
    if reference_height_probe_m is not None and (
        type(reference_height_probe_m) is not float
        or not math.isfinite(reference_height_probe_m)
        or not 0.65 <= reference_height_probe_m <= 0.8
        or foundation_approach_probe is not True
        or locomotion_mode_probe != 3
    ):
        raise ValueError("reference-height intervention requires a captured pure run ablation")
    if type(foundation_approach_probe) is not bool:
        raise ValueError("explicit boolean foundation approach ablation required")
    if foundation_approach_probe and (
        getattr(args, "foundation_only", False) is not True
        or getattr(args, "record_foundation_observation", False) is not True
        or control_profile is None
        or approach_mode != "bounded_approach_diagnostic"
        or swing_mode != "legacy"
        or partition != "DECLARED_DEVELOPMENT_DIAGNOSTIC"
        or postcontact_profile is not None
    ):
        raise ValueError("unmixed captured foundation-only bounded approach diagnostic required")
    if locomotion_mode_probe is not None and (
        type(locomotion_mode_probe) is not int
        or locomotion_mode_probe not in (2, 3)
        or control_profile is None
        or postcontact_profile is not None
    ):
        raise ValueError("run-mode probe requires an unmixed bounded SIM control profile")
    if type(swing_mode) is not str or swing_mode not in (
        "legacy",
        "memory_gate_both_directions_diagnostic",
        "bounded_contact_diagnostic",
    ):
        raise ValueError("explicit SIM swing eligibility diagnostic mode required")
    if type(approach_mode) is not str or approach_mode not in (
        "legacy",
        "continuous_lateral_diagnostic",
        "bounded_approach_diagnostic",
    ):
        raise ValueError("explicit SIM approach diagnostic mode required")
    if type(partition) is not str or partition not in (
        "CONSUMED_TRANSFER_DIAGNOSTIC",
        "DECLARED_DEVELOPMENT_DIAGNOSTIC",
        "DECLARED_PUBLIC_TRAINING",
    ):
        raise ValueError("native kernel cannot open or authorize a private Fresh exam")
    if control_profile is not None:
        if (
            type(control_profile) not in (BoundedFirstTouchControl, FastInterceptControl)
            or approach_mode != "bounded_approach_diagnostic"
            or partition != "DECLARED_DEVELOPMENT_DIAGNOSTIC"
            or (recurrent_sampling_factory is None and not foundation_approach_probe)
        ):
            raise ValueError("explicit bounded SIM diagnostic and recurrent sampler required")
        control_profile.__post_init__()
        if postcontact_profile is not None:
            if type(postcontact_profile) is not PostcontactNavigationProbe:
                raise ValueError("typed SIM postcontact diagnostic required")
            postcontact_profile.__post_init__()
    elif (
        approach_mode == "bounded_approach_diagnostic" or swing_mode == "bounded_contact_diagnostic"
    ):
        raise ValueError("bounded interface requires a complete declared control profile")
    if control_profile is None and postcontact_profile is not None:
        raise ValueError("postcontact probe requires bounded SIM control profile")
    if (
        type(course) is not tuple
        or len(course) != 3
        or any(type(value) is not float or not math.isfinite(value) for value in course)
    ):
        raise ValueError("explicit finite three-float native course required")
    if not isinstance(entry_source_path, Path) or not entry_source_path.is_file():
        raise ValueError("pinned native diagnostic entry source required")
    if partition == "DECLARED_PUBLIC_TRAINING":
        if (
            type(training_plan) is not PublicFirstTouchTrainingPlan
            or approach_mode != "legacy"
            or swing_mode != "legacy"
            or consumed_bank_hash is not None
            or any(
                factory is not None
                for factory in (
                    sampling_factory,
                    extended_factory,
                    proposal_factory,
                    imitation_factory,
                    recurrent_factory,
                    recurrent_clipped_factory,
                )
            )
            or any(
                getattr(args, key, None) is not None
                for key in (
                    "neural_model",
                    "motor_policy",
                    "body_response_bundle",
                )
            )
            or getattr(args, "foundation_only", None) is not False
            or getattr(args, "cached_proposal_envelope", None) is not False
        ):
            raise ValueError("explicit unmixed public training declaration required")
        training_plan.require_draw(
            group=training_group, seed=args.seed, lane=args.lane, course=course
        )
    elif training_plan is not None or training_group is not None:
        raise ValueError(
            "training declarations cannot be attached to diagnostic or exam partitions"
        )
    foundation_only = getattr(args, "foundation_only", False)
    if type(foundation_only) is not bool:
        raise ValueError("explicit boolean foundation-only mode required")
    if foundation_only and (
        any(
            getattr(args, key, None) is not None
            for key in ("neural_model", "motor_policy", "step_model", "body_response_bundle")
        )
        or any(
            factory is not None
            for factory in (
                sampling_factory,
                extended_factory,
                proposal_factory,
                imitation_factory,
                recurrent_factory,
                recurrent_sampling_factory,
                recurrent_clipped_factory,
            )
        )
        or getattr(args, "cached_proposal_envelope", False) is not False
        or getattr(args, "proposal_decoder", "reference") != "reference"
    ):
        raise ValueError("foundation-only cannot include residual models, factories or guidance")
    import mujoco
    import torch

    from rosclaw_soccer.rsi.native_lateral_experiment import native_lateral_command
    from rosclaw_soccer.rsi.native_swing_experiment import native_swing_eligible

    torch.set_num_threads(1)
    scene = args.scene.resolve()
    spec = mujoco.MjSpec.from_file(str(scene))
    model = spec.compile()
    native_ball_added = False
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball") < 0:
        native_ball_added = True
        body = spec.worldbody.add_body(name="ball", pos=(2.5, 0.0, 0.13))
        body.add_freejoint(name="ball_free")
        body.add_geom(
            name="ball_geom",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=(0.11, 0.0, 0.0),
            mass=0.43,
            friction=(0.6, 0.005, 0.0001),
        )
        if not np.any(model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE):
            spec.worldbody.add_geom(
                name="transfer_floor",
                type=mujoco.mjtGeom.mjGEOM_PLANE,
                size=(0.0, 0.0, 0.01),
                friction=(1.0, 0.005, 0.0001),
            )
        model = spec.compile()
    mesh_contact_construction = None
    if mesh_task_contact_world_hash is not None:
        from rosclaw_soccer.physics.task_contact_surface import with_mesh_task_contacts

        spec, mesh_contact_construction = with_mesh_task_contacts(
            spec,
            task_geom_name="ball_geom",
            surface_body_names=("left_ankle_roll_link", "right_ankle_roll_link"),
        )
        model = spec.compile()
    data = mujoco.MjData(model)
    model.opt.timestep = 0.002

    def identifier(kind: Any, name: str) -> int:
        value = int(mujoco.mj_name2id(model, kind, name))
        if value < 0:
            raise ValueError(f"missing simulation element {name}")
        return value

    names = list(G1_DDS_JOINT_NAMES)
    joints = [identifier(mujoco.mjtObj.mjOBJ_JOINT, name) for name in names]
    qi = np.asarray([model.jnt_qposadr[j] for j in joints], dtype=int)
    vi = np.asarray([model.jnt_dofadr[j] for j in joints], dtype=int)
    actuator_ids = []
    for joint in joints:
        matching = np.flatnonzero(model.actuator_trnid[:, 0] == joint)
        if len(matching) != 1:
            raise ValueError("exactly one simulated actuator per canonical joint required")
        actuator_ids.append(int(matching[0]))
    ai = np.asarray(actuator_ids, dtype=int)
    if model.nu != 29 or not np.array_equal(model.actuator_trnid[ai, 0], joints):
        raise ValueError("canonical named 29-joint torque contract required")
    pelvis = identifier(mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    body_names = [
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_knee_link",
        "right_knee_link",
    ]
    bodies = [identifier(mujoco.mjtObj.mjOBJ_BODY, n) for n in body_names]
    contact_bodies = (
        bodies[:2]
        + [
            identifier(mujoco.mjtObj.mjOBJ_BODY, "left_ankle_pitch_link"),
            identifier(mujoco.mjtObj.mjOBJ_BODY, "right_ankle_pitch_link"),
        ]
        + bodies[2:]
    )
    ball = identifier(mujoco.mjtObj.mjOBJ_BODY, "ball")
    ball_geom = identifier(mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")
    bj = int(model.body_jntadr[ball])
    bq, bv = int(model.jnt_qposadr[bj]), int(model.jnt_dofadr[bj])
    # Explicit physics adaptation, not an assertion that the XML equals USD.
    model.geom_size[ball_geom, 0] = 0.11
    model.body_mass[ball] = 0.43
    model.body_inertia[ball] = 0.4 * 0.43 * 0.11**2
    model.dof_damping[bv : bv + 6] = 0
    box = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box")
    if box >= 0:
        mask = model.geom_bodyid == box
        model.geom_contype[mask] = 0
        model.geom_conaffinity[mask] = 0
    mujoco.mj_setConst(model, data)
    if mesh_task_contact_world_hash is not None:
        snapshot_bytes = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
        mujoco.mj_saveModel(model, None, snapshot_bytes)
        if hash_bytes(snapshot_bytes.tobytes()) != mesh_task_contact_world_hash:
            raise ValueError("new task contact world differs from its predeclared compiled hash")
    navigation = G1SonicNavigation(
        args.model_root,
        "cpu.first_touch.0",
        SonicNavigationConfig(
            maximum_frames=300,
            planner_seed=30300,
            model_variant="low_latency",
            experimental_maximum_speed_mps=1.5,
            inference_threads=1,
            record_default_planner_events=record_default_planner_events,
        ),
    )
    if control_profile is not None and locomotion_mode_probe is not None:
        navigation = G1SonicNavigation(
            args.model_root,
            "cpu.first_touch.0",
            SonicNavigationConfig(
                maximum_frames=300,
                planner_seed=30300,
                model_variant="low_latency",
                experimental_maximum_speed_mps=2.1 if fast_intercept else 1.5,
                inference_threads=1,
                experimental_locomotion_mode=locomotion_mode_probe,
                experimental_reference_height_m=reference_height_probe_m,
                experimental_fast_intercept=fast_intercept,
            ),
        )
    navigation.backend.qualification.require_eligible()
    foundation = FrozenSonicG1Torch(args.model_root, variant="low_latency", device="cpu")
    late = load_late_swing_actor(args.late_swing_policy)
    neural = json.loads(args.neural_model.read_text()) if args.neural_model else None
    if neural:
        validate_model(neural)
    policy, knots = load_policy(args.motor_policy) if args.motor_policy else (None, None)
    # Restore the WHOLE original numerical model before preview validation.
    # Ordinary JSON/gzip policies remain unchanged; no distribution is changed.
    if args.cached_proposal_envelope:
        from rosclaw_soccer.rsi.json_artifact_io import load_json_artifact

        step_model = sampling_factory.restore_envelope(load_json_artifact(args.step_model))
    else:
        step_model = (
            (
                load_sampling_model(args.step_model)
                if recurrent_sampling_factory is None
                else load_sampling_model(
                    args.step_model, recurrent_sampling_factory=recurrent_sampling_factory
                )
            )
            if args.step_model
            else None
        )
    if training_plan is not None:
        training_plan.require_sampling_view(
            training_plan.require_draw(
                group=training_group, seed=args.seed, lane=args.lane, course=course
            ),
            step_model,
        )
    if imitation_factory is not None and (
        step_model is None
        or step_model.get("schema") != "soccer.rsi.verified_success_imitation_motor.v1"
    ):
        raise ValueError("fixed imitation factory cannot bind another model family")
    if proposal_factory is not None and (
        step_model is None or step_model.get("schema") != "soccer.rsi.proposal_memory_motor.v1"
    ):
        raise ValueError("fixed proposal factory cannot bind another model family")
    if extended_factory is not None and (
        step_model is None or step_model.get("schema") != "soccer.rsi.extended_proposal_motor.v1"
    ):
        raise ValueError("fixed extended factory cannot bind another model family")
    if recurrent_factory is not None and (
        step_model is None
        or step_model.get("schema") != "soccer.rsi.recurrent_success_imitation_motor.v1"
    ):
        raise ValueError("fixed sequence factory cannot bind another model family")
    if recurrent_sampling_factory is not None and (
        step_model is None or step_model.get("schema") != "soccer.rsi.recurrent_motor_sampling.v1"
    ):
        raise ValueError("fixed recurrent sampling factory cannot bind another model family")
    if recurrent_clipped_factory is not None and (
        step_model is None or step_model.get("schema") != "soccer.rsi.recurrent_clipped_mc_motor.v1"
    ):
        raise ValueError("fixed clipped factory cannot bind another model family")
    sampling_compilation = None
    recurrent_decoder: Any = None
    recurrent_sampling = False
    if sampling_factory is not None:
        from rosclaw_soccer.rsi.owned_proposal_sampling_factory import (
            OwnedProposalSamplingEpisodeFactory,
        )
        from rosclaw_soccer.rsi.proposal_sampling_episode_factory import (
            ProposalSamplingEpisodeFactory,
        )

        expected_sampling_schema = (
            "soccer.rsi.proposal_memory_sampling.v1"
            if type(sampling_factory)
            in (ProposalSamplingEpisodeFactory, OwnedProposalSamplingEpisodeFactory)
            else "soccer.rsi.smooth_memory_sampling.v1"
        )
        if step_model is None or step_model.get("schema") != expected_sampling_schema:
            raise ValueError("shared sampling factory cannot compile other motor families")
    if args.proposal_decoder != "reference" and (
        step_model is None or step_model.get("schema") != "soccer.rsi.proposal_memory_motor.v1"
    ):
        raise ValueError("owned proposal decoder only accepts the sealed proposal family")
    if step_model is not None:
        from rosclaw_soccer.rsi.step_motor_execution import delta_at_frame
        from rosclaw_soccer.rsi.step_motor_execution import make_preview as legacy_preview

        make_preview: Callable[[dict[str, Any]], dict[str, Any]] = legacy_preview

        if step_model.get("schema") == "soccer.rsi.recurrent_motor_sampling.v1":
            from rosclaw_soccer.rsi.recurrent_sampling_motor import (
                CompiledRecurrentSamplingMotor,
            )
            from rosclaw_soccer.rsi.recurrent_sampling_motor import (
                make_preview as recurrent_sampling_preview,
            )

            if (
                args.proposal_decoder != "reference"
                or any(
                    factory is not None
                    for factory in (
                        sampling_factory,
                        extended_factory,
                        proposal_factory,
                        imitation_factory,
                        recurrent_factory,
                    )
                )
                or args.body_response_bundle
            ):
                raise ValueError("recurrent exploration requires the original reference law")
            make_preview = recurrent_sampling_preview
            if generated_seed_episode_binding:
                policy, recurrent_decoder = recurrent_sampling_factory.generate_and_bind_seed(
                    seed=step_model["seed"]
                )
                if policy["step_motor_proof"]["model"] != step_model:
                    raise ValueError("private generated model differs from fully validated input")
            else:
                policy = (
                    make_preview(step_model)
                    if recurrent_sampling_factory is None
                    else recurrent_sampling_factory.preview(step_model)
                )
                recurrent_decoder = (
                    CompiledRecurrentSamplingMotor(policy)
                    if recurrent_sampling_factory is None
                    else recurrent_sampling_factory.bind(policy)
                )
            recurrent_sampling = True
            delta_at_frame = recurrent_decoder.delta_at_frame
        elif step_model.get("schema") == "soccer.rsi.recurrent_clipped_mc_motor.v1":
            from rosclaw_soccer.rsi.recurrent_clipped_motor import (
                CompiledRecurrentClippedMotor,
            )
            from rosclaw_soccer.rsi.recurrent_clipped_motor import (
                make_preview as clipped_preview,
            )

            if (
                args.proposal_decoder != "reference"
                or any(
                    factory is not None
                    for factory in (
                        sampling_factory,
                        proposal_factory,
                        extended_factory,
                        imitation_factory,
                        recurrent_factory,
                    )
                )
                or args.body_response_bundle
            ):
                raise ValueError("clipped recurrent motor requires the independent reference law")
            make_preview = clipped_preview
            policy = (
                make_preview(step_model)
                if recurrent_clipped_factory is None
                else recurrent_clipped_factory.preview(step_model)
            )
            recurrent_decoder = (
                CompiledRecurrentClippedMotor(policy)
                if recurrent_clipped_factory is None
                else recurrent_clipped_factory.new_episode()
            )
            delta_at_frame = recurrent_decoder.delta_at_frame
        elif step_model.get("schema") == "soccer.rsi.recurrent_success_imitation_motor.v1":
            from rosclaw_soccer.rsi.recurrent_success_motor import (
                CompiledRecurrentSuccessMotor,
            )
            from rosclaw_soccer.rsi.recurrent_success_motor import (
                make_preview as recurrent_preview,
            )

            if (
                args.proposal_decoder != "reference"
                or any(
                    factory is not None
                    for factory in (
                        sampling_factory,
                        extended_factory,
                        proposal_factory,
                        imitation_factory,
                    )
                )
                or args.body_response_bundle
            ):
                raise ValueError("sequence candidate requires its original bounded execution law")
            make_preview = recurrent_preview
            policy = (
                make_preview(step_model)
                if recurrent_factory is None
                else recurrent_factory.preview(step_model)
            )
            recurrent_decoder = (
                CompiledRecurrentSuccessMotor(policy)
                if recurrent_factory is None
                else recurrent_factory.new_episode()
            )
            delta_at_frame = recurrent_decoder.delta_at_frame
        elif step_model.get("schema") == "soccer.rsi.verified_success_imitation_motor.v1":
            from rosclaw_soccer.rsi.imitation_proposal_motor import CompiledImitationProposalMotor
            from rosclaw_soccer.rsi.imitation_proposal_motor import (
                make_preview as imitation_preview,
            )

            if (
                args.proposal_decoder != "reference"
                or sampling_factory is not None
                or args.body_response_bundle
            ):
                raise ValueError("imitation candidate requires its original bounded execution law")
            make_preview = imitation_preview
            policy = (
                make_preview(step_model)
                if imitation_factory is None
                else imitation_factory.preview(step_model)
            )
            imitation_decoder = (
                CompiledImitationProposalMotor(policy)
                if imitation_factory is None
                else imitation_factory.new_episode()
            )
            delta_at_frame = imitation_decoder.delta_at_frame
        elif step_model.get("schema") == "soccer.rsi.proposal_memory_sampling.v1":
            from rosclaw_soccer.rsi.owned_proposal_sampling_factory import (
                OwnedProposalSamplingEpisodeFactory,
            )
            from rosclaw_soccer.rsi.proposal_sampling_motor import CompiledProposalSamplingMotor
            from rosclaw_soccer.rsi.proposal_sampling_motor import make_preview as sampling_preview

            make_preview = sampling_preview
            prepared_proposal_policy = (
                sampling_factory.preview(step_model)
                if type(sampling_factory) is OwnedProposalSamplingEpisodeFactory
                else make_preview(step_model)
            )
            if sampling_factory is None:
                proposal_sampling_decoder = CompiledProposalSamplingMotor(prepared_proposal_policy)
            else:
                from rosclaw_soccer.rsi.proposal_sampling_episode_factory import (
                    ProposalSamplingEpisodeFactory,
                )
                from rosclaw_soccer.rsi.proposal_sampling_episode_factory import (
                    compilation_contract as proposal_sampling_contract,
                )

                if type(sampling_factory) not in (
                    ProposalSamplingEpisodeFactory,
                    OwnedProposalSamplingEpisodeFactory,
                ):
                    raise ValueError("only the fixed proposal sampling factory is accepted")
                proposal_sampling_decoder = sampling_factory.bind(prepared_proposal_policy)
                if type(sampling_factory) is OwnedProposalSamplingEpisodeFactory:
                    from rosclaw_soccer.rsi.owned_proposal_sampling_factory import (
                        compilation_contract as owned_proposal_sampling_contract,
                    )

                    sampling_compilation = owned_proposal_sampling_contract(
                        prepared_proposal_policy
                    )
                else:
                    sampling_compilation = proposal_sampling_contract(prepared_proposal_policy)
            delta_at_frame = proposal_sampling_decoder.delta_at_frame
        elif step_model.get("schema") == "soccer.rsi.extended_proposal_motor.v1":
            from rosclaw_soccer.rsi.extended_proposal_motor import CompiledExtendedProposalMotor
            from rosclaw_soccer.rsi.extended_proposal_motor import make_preview as extended_preview

            if args.proposal_decoder != "reference" or sampling_factory is not None:
                raise ValueError("extended proposal requires its explicit original bounded law")
            make_preview = extended_preview
            policy = (
                make_preview(step_model)
                if extended_factory is None
                else extended_factory.preview(step_model)
            )
            extended_decoder = (
                CompiledExtendedProposalMotor(policy)
                if extended_factory is None
                else extended_factory.new_episode()
            )
            delta_at_frame = extended_decoder.delta_at_frame
        elif step_model.get("schema") == "soccer.rsi.proposal_memory_motor.v1":
            from rosclaw_soccer.rsi.proposal_decoder_selection import select_proposal_decoder
            from rosclaw_soccer.rsi.proposal_memory_motor import make_preview as proposal_preview

            make_preview = proposal_preview
            policy = make_preview(step_model)
            proposal_decoder = (
                select_proposal_decoder(policy, implementation=args.proposal_decoder)
                if proposal_factory is None
                else proposal_factory.bind(policy)
            )
            delta_at_frame = proposal_decoder.delta_at_frame
        elif step_model.get("schema") == "soccer.rsi.advantage_memory_motor.v1":
            from rosclaw_soccer.rsi.advantage_memory_motor import CompiledAdvantageMemoryMotor
            from rosclaw_soccer.rsi.advantage_memory_motor import make_preview as advantage_preview

            make_preview = advantage_preview
            delta_at_frame = CompiledAdvantageMemoryMotor(make_preview(step_model)).delta_at_frame
        elif step_model.get("schema") == "soccer.rsi.current_memory_guarded_motor.v1":
            from rosclaw_soccer.rsi.current_memory_motor import CompiledCurrentMemoryMotor
            from rosclaw_soccer.rsi.current_memory_motor import make_preview as current_preview

            make_preview = current_preview
            delta_at_frame = CompiledCurrentMemoryMotor(make_preview(step_model)).delta_at_frame
        elif step_model.get("schema") == "soccer.rsi.consolidated_smooth_motor.v1":
            from rosclaw_soccer.rsi.consolidated_smooth_motor import CompiledConsolidatedSmoothMotor
            from rosclaw_soccer.rsi.consolidated_smooth_motor import (
                make_preview as consolidated_preview,
            )

            make_preview = consolidated_preview
            delta_at_frame = CompiledConsolidatedSmoothMotor(
                make_preview(step_model)
            ).delta_at_frame
        elif step_model.get("schema") in (
            "soccer.rsi.smooth_memory_motor.v1",
            "soccer.rsi.smooth_memory_sampling.v1",
        ):
            from rosclaw_soccer.rsi.owned_smooth_sampling_factory import OwnedSmoothSamplingFactory
            from rosclaw_soccer.rsi.smooth_decoder_selection import select_smooth_decoder
            from rosclaw_soccer.rsi.smooth_memory_motor import make_preview as smooth_preview

            make_preview = smooth_preview
            prepared_smooth_policy = (
                sampling_factory.preview(step_model)
                if type(sampling_factory) is OwnedSmoothSamplingFactory
                else make_preview(step_model)
            )
            decoder, sampling_compilation = select_smooth_decoder(
                prepared_smooth_policy, sampling_factory=sampling_factory
            )
            delta_at_frame = decoder.delta_at_frame
        elif step_model.get("schema") in (
            "soccer.rsi.output_memory_step_motor.v1",
            "soccer.rsi.output_memory_step_sampling.v1",
        ):
            from rosclaw_soccer.rsi.output_memory_step_motor import CompiledOutputMemoryMotor
            from rosclaw_soccer.rsi.output_memory_step_motor import make_preview as output_preview

            make_preview = output_preview
            delta_at_frame = CompiledOutputMemoryMotor(make_preview(step_model)).delta_at_frame
        elif step_model.get("schema") == "soccer.rsi.kernel_replay_motor.v1":
            from rosclaw_soccer.rsi.kernel_replay_motor import CompiledReplayStepMotor
            from rosclaw_soccer.rsi.kernel_replay_motor import make_preview as replay_preview

            make_preview = replay_preview
            delta_at_frame = CompiledReplayStepMotor(make_preview(step_model)).delta_at_frame
        elif step_model.get("schema") == "soccer.rsi.selective_phase_memory.v1":
            from rosclaw_soccer.rsi.selective_phase_memory import CompiledSelectivePhaseMemory
            from rosclaw_soccer.rsi.selective_phase_memory import make_preview as selective_preview

            make_preview = selective_preview
            delta_at_frame = CompiledSelectivePhaseMemory(make_preview(step_model)).delta_at_frame
        elif step_model.get("schema") == "soccer.rsi.memory_guarded_phase_transfer.v1":
            from rosclaw_soccer.rsi.memory_guarded_phase_transfer import CompiledMemoryPhaseMotor
            from rosclaw_soccer.rsi.memory_guarded_phase_transfer import (
                make_preview as memory_preview,
            )

            make_preview = memory_preview
            delta_at_frame = CompiledMemoryPhaseMotor(make_preview(step_model)).delta_at_frame
        elif step_model.get("schema") in (
            "soccer.rsi.kernel_guarded_step_actor_critic.v1",
            "soccer.rsi.kernel_guarded_step_sampling.v1",
        ):
            from rosclaw_soccer.rsi.kernel_guarded_step_execution import (
                CompiledKernelStepMotor,
            )
            from rosclaw_soccer.rsi.kernel_guarded_step_execution import (
                make_preview as kernel_preview,
            )

            make_preview = kernel_preview
            delta_at_frame = CompiledKernelStepMotor(make_preview(step_model)).delta_at_frame
        elif step_model.get("schema") == "soccer.rsi.protected_phase_step_actor_critic.v1":
            from rosclaw_soccer.rsi.protected_phase_step_execution import (
                CompiledProtectedPhaseMotor,
            )
            from rosclaw_soccer.rsi.protected_phase_step_execution import (
                make_preview as protected_phase_preview,
            )

            make_preview = protected_phase_preview
            delta_at_frame = CompiledProtectedPhaseMotor(make_preview(step_model)).delta_at_frame
        elif step_model.get("schema") == "soccer.rsi.compiled_step_motor_decoder.v1":
            from rosclaw_soccer.rsi.compiled_step_inference import CompiledStepMotor
            from rosclaw_soccer.rsi.compiled_step_inference import make_preview as compiled_preview

            make_preview = compiled_preview
            delta_at_frame = CompiledStepMotor(make_preview(step_model)).delta_at_frame
        elif step_model.get("schema") == "soccer.rsi.online_step_motor_mc_ppo.v1":
            from rosclaw_soccer.rsi.online_step_execution import (
                delta_at_frame as online_delta_at_frame,
            )
            from rosclaw_soccer.rsi.online_step_execution import make_preview as online_preview

            delta_at_frame = online_delta_at_frame
            make_preview = online_preview
        if step_model.get("schema") in (
            "soccer.rsi.smooth_memory_motor.v1",
            "soccer.rsi.smooth_memory_sampling.v1",
        ):
            policy = prepared_smooth_policy
        elif step_model.get("schema") == "soccer.rsi.proposal_memory_sampling.v1":
            # This complete preview already passed the selected constructor
            # or source-bound factory. Recomputing it only repeats large
            # canonical graph validation and discards the identical result.
            policy = prepared_proposal_policy
        elif step_model.get("schema") not in (
            "soccer.rsi.proposal_memory_motor.v1",
            "soccer.rsi.extended_proposal_motor.v1",
            "soccer.rsi.verified_success_imitation_motor.v1",
            "soccer.rsi.recurrent_success_imitation_motor.v1",
            "soccer.rsi.recurrent_motor_sampling.v1",
            "soccer.rsi.recurrent_clipped_mc_motor.v1",
        ):
            policy = make_preview(step_model)
    x, y, vx = course
    data.qpos[:7] = (0, 0, 0.793, 1, 0, 0, 0)
    data.qpos[qi] = navigation.backend.default_angles
    data.qpos[bq : bq + 7] = (x, y, 0.13, 1, 0, 0, 0)
    data.qvel[bv : bv + 6] = (vx, 0, 0, 0, vx / 0.11, 0)
    mujoco.mj_forward(model, data)
    args.output_root.mkdir(parents=True, exist_ok=False)
    model_snapshot = args.output_root / "compiled_model.mjb"
    if args.shared_evidence:
        from rosclaw_soccer.rsi.shared_cpu_evidence import save_shared_compiled_model

        save_shared_compiled_model(model, model_snapshot)
    else:
        mujoco.mj_saveModel(model, str(model_snapshot), None)
    assets = {
        str(p.relative_to(scene.parent)): hash_bytes(p.read_bytes())
        for p in sorted(scene.parent.rglob("*"))
        if p.is_file() and p.suffix.lower() in {".xml", ".stl", ".obj", ".png", ".jpg"}
    }
    body_guidance = None
    body_guidance_bundle = None
    if args.body_response_bundle:
        from rosclaw_soccer.rsi.body_response_guidance_execution import (
            BodyResponseGuidanceExecution,
        )
        from rosclaw_soccer.rsi.json_artifact_io import load_json_artifact

        if step_model is None or step_model.get("schema") != "soccer.rsi.proposal_memory_motor.v1":
            raise ValueError("body guidance only accepts the explicit proposal mean parent")
        body_guidance_bundle = load_json_artifact(args.body_response_bundle)
        body_guidance = BodyResponseGuidanceExecution(body_guidance_bundle, policy)
    foot_forward_cap = 0.08 if control_profile is None else control_profile.foot_forward_cap_m
    foot_lateral_cap = 0.15 if control_profile is None else control_profile.foot_lateral_cap_m
    foot_vertical_offset = (
        0.04 if control_profile is None else control_profile.foot_vertical_offset_m
    )
    commitment = dict(
        schema="soccer.rsi.cpu_motor_transfer.v1",
        seed=args.seed,
        lane=args.lane,
        course=[x, y, vx],
        partition=partition,
        approach_mode=approach_mode,
        entry_source_hash=hash_bytes(entry_source_path.read_bytes()),
        source_hash=hash_bytes(Path(__file__).read_bytes()),
        assets=assets,
        foundation_hash=navigation.backend.qualification.qualification_hash,
        late_actor_hash=late["actor_hash"],
        execution_profile="foundation_only" if args.foundation_only else "taskspace_plus_motor",
        swing_mode=swing_mode,
        public_training_declaration=training_plan.contract() if training_plan is not None else None,
        model_hash=neural["model_hash"] if neural else None,
        step_model_hash=step_model["model_hash"] if step_model else None,
        consumed_bank_hash=consumed_bank_hash,
        motor_policy_hash=policy["policy_hash"] if policy else None,
        taskspace_contract=dict(
            forward_cap_m=foot_forward_cap,
            lateral_cap_m=foot_lateral_cap,
            acquisition_max_gap_m=0.95,
            revalidate_swing_side=True,
            vertical_offset_m=foot_vertical_offset,
        ),
        compiled_model_hash=hash_bytes(model_snapshot.read_bytes()),
        physics=dict(
            engine="MuJoCo",
            version=mujoco.__version__,
            device="cpu",
            dt_s=0.002,
            control_dt_s=0.02,
            ball_radius_m=0.11,
            ball_mass_kg=0.43,
            ball_free_damping=0.0,
            auxiliary_box_collision_disabled=True,
            robot_xml_not_usd=True,
            contact_settings="original XML pairs retained",
            native_ball_added=native_ball_added,
            canonical_joint_names=names,
            canonical_pd_kp=navigation.backend.kp.tolist(),
            canonical_pd_kd=navigation.backend.kd.tolist(),
        ),
        activation_ceiling="SIM_ONLY",
        promotion_authorized=False,
        hardware_authorized=False,
    )
    if generated_seed_episode_binding:
        commitment["generated_seed_episode_binding"] = dict(
            schema="soccer.rsi.native_private_generated_episode_binding.v1",
            seed=step_model["seed"],
            full_external_sampling_view_restored_and_validated=True,
            complete_generated_model_equals_external_view=True,
            full_original_factory_constructor_retained=True,
            sampling_law_changed=False,
        )
    if mesh_contact_construction is not None:
        commitment["mesh_task_contact_protocol"] = dict(
            expected_world_hash=mesh_task_contact_world_hash,
            construction=mesh_contact_construction,
            legacy_qualifications_transferred=False,
        )
        commitment["physics"]["contact_settings"] = "explicit existing foot meshes vs ball only"
        if mesh_review_binding is not None:
            commitment["mesh_task_contact_protocol"]["numerical_review"] = mesh_review_binding
    if suppress_postcontact_swing_release:
        from rosclaw_soccer.rsi.swing_release_ablation import release_ablation_contract

        commitment["swing_release_ablation"] = release_ablation_contract(
            commitment["compiled_model_hash"]
        )
    if use_contact_exit_torque_mpc:
        from rosclaw_soccer.rsi.contact_exit_torque_mpc import ContactExitTorqueMPC

        exit_teacher = ContactExitTorqueMPC(
            model,
            world_hash=commitment["compiled_model_hash"],
            kp=navigation.backend.kp,
            kd=navigation.backend.kd,
        )
        commitment["contact_exit_torque_mpc"] = exit_teacher.contract()
    if use_directional_contact_compliance:
        from rosclaw_soccer.rsi.directional_contact_compliance import DirectionalContactCompliance

        exit_compliance = DirectionalContactCompliance(
            model,
            world_hash=commitment["compiled_model_hash"],
            kp=navigation.backend.kp,
            kd=navigation.backend.kd,
        )
        commitment["directional_contact_compliance"] = exit_compliance.contract()
    cerebellar = None
    if cerebellar_model is not None:
        from rosclaw_soccer.rsi.native_cerebellar_torque import NativeCerebellarTorque

        cerebellar = NativeCerebellarTorque(
            cerebellar_model,
            world_hash=commitment["compiled_model_hash"],
            sampling_seed=cerebellar_sampling_seed,
        )
        commitment["schema"] = "soccer.rsi.cpu_motor_transfer.v2"
        commitment["cerebellar_torque_contract"] = cerebellar.contract()
    observation_contract = make_observation_contract(
        args.root_velocity_reference, args.observation_snapshot
    )
    if record_default_planner_events:
        commitment["default_locomotion_reference_capture"] = dict(
            schema="soccer.rsi.default_locomotion_reference_capture.v1",
            navigation_contract_hash=navigation.contract_hash,
            numerical_intervention=False,
            hardware_authorized=False,
            promotion_authorized=False,
        )
    if args.proposal_decoder != "reference":
        from rosclaw_soccer.rsi.proposal_decoder_selection import compilation_contract

        commitment["numeric_compilation"] = compilation_contract(args.proposal_decoder)
    if sampling_compilation is not None:
        commitment["numeric_sampling_compilation"] = sampling_compilation
    if observation_contract is not None:
        commitment["observation_contract"] = observation_contract
    if args.record_foundation_observation:
        from rosclaw_soccer.rsi.foundation_observation_capture import capture_contract

        commitment["foundation_observation_capture"] = capture_contract()
    if body_guidance is not None:
        commitment["body_response_guidance"] = body_guidance_bundle
    if extended_factory is not None:
        commitment["extended_proposal_factory"] = extended_factory.contract()
    if proposal_factory is not None:
        commitment["fixed_proposal_factory"] = proposal_factory.contract()
    if imitation_factory is not None:
        commitment["imitation_proposal_factory"] = imitation_factory.contract()
    if recurrent_factory is not None:
        commitment["recurrent_success_factory"] = recurrent_factory.contract()
    if recurrent_sampling_factory is not None:
        commitment["recurrent_sampling_factory"] = recurrent_sampling_factory.contract()
    if recurrent_clipped_factory is not None:
        commitment["recurrent_clipped_factory"] = recurrent_clipped_factory.contract()
    if control_profile is not None:
        commitment["bounded_control_profile"] = control_profile.contract()
        if foundation_approach_probe:
            commitment["foundation_approach_ablation"] = dict(
                schema="soccer.rsi.foundation_only_bounded_approach_ablation.v1",
                motor_residual_enabled=False,
                taskspace_ik_enabled=False,
                navigation_rule_unchanged=True,
                closed_loop_commands_may_differ_after_body_divergence=True,
                alone_does_not_separate_residual_from_ik_effect=True,
                teacher_qualified=False,
                promotion_authorized=False,
                hardware_authorized=False,
            )
        if locomotion_mode_probe is not None:
            commitment["locomotion_mode_probe"] = dict(
                schema="soccer.rsi.sonic_locomotion_mode_probe.v1",
                navigation_contract_hash=navigation.contract_hash,
                planner_hash=navigation.backend.qualification.planner_hash,
                planner_seed=30300,
                maximum_navigation_speed_m_s=1.5,
                high_speed_threshold_m_s=0.8,
                high_speed_planner_mode=locomotion_mode_probe,
                low_speed_modes_unchanged=True,
                physical_joint_and_torque_limits_unchanged=True,
                teacher_qualified=False,
                promotion_authorized=False,
                hardware_authorized=False,
            )
            if reference_height_probe_m is not None:
                commitment["locomotion_mode_probe"].update(
                    schema="soccer.rsi.sonic_locomotion_mode_probe.v2",
                    high_speed_reference_height_m=reference_height_probe_m,
                )
            if fast_intercept:
                commitment["locomotion_mode_probe"].update(
                    schema="soccer.rsi.sonic_locomotion_mode_probe.v3",
                    maximum_navigation_speed_m_s=2.1,
                    fast_precontact_forward_cap_m_s=2.0,
                )
        if postcontact_profile is not None:
            commitment["postcontact_navigation_probe"] = postcontact_profile.contract()
    (args.output_root / "commitment.json").write_text(json.dumps(commitment, indent=2))
    history: dict[str, list[Any]] = {
        k: []
        for k in (
            "root_pose_xyzw_m",
            "root_velocity_world",
            "ball_position_before_step_m",
            "ball_linear_velocity_before_step_m_s",
            "foot_geometry_position_before_step_m",
            "joint_position_rad",
            "joint_velocity_rad_s",
            "joint_target_rad",
            "force_n",
            "ball_position_after_step_m",
            "torque_nm",
            "pre_motor_joint_target_rad",
            "motor_delta_rad",
            "navigation_command",
            "actual_actuator_force_nm",
            "pelvis_z_per_substep_m",
            "canonical_qpos",
            "canonical_qvel",
        )
    }
    if recurrent_decoder is not None:
        from rosclaw_soccer.rsi.recurrent_success_motor import STATE_FIELD

        history[STATE_FIELD] = []
        if recurrent_sampling:
            from rosclaw_soccer.rsi.recurrent_sampling_motor import TRACE_FIELDS

            history.update({name: [] for name in TRACE_FIELDS})
    if args.record_foundation_observation:
        from rosclaw_soccer.rsi.foundation_observation_capture import FIELDS, PREFIX

        history.update({PREFIX + name: [] for name in FIELDS})
    if body_guidance is not None:
        history.update({k: [] for k in ("body_response_applied_increment", "body_response_status")})
    tracker = None
    if cerebellar is not None:
        CEREBELLAR_WIDTHS = cerebellar.capture_widths()
        history.update({name: [] for name in CEREBELLAR_WIDTHS})
    if control_profile is not None:
        from rosclaw_soccer.rsi.foundation_only_action_audit import require_bounded_capture_decoder

        require_bounded_capture_decoder(
            recurrent_decoder, recurrent_sampling, foundation_approach_probe
        )
        history.update(
            {
                k: []
                for k in (
                    "approach_requested_navigation_command",
                    "foundation_navigation_joint_target_rad",
                    "taskspace_joint_increment_rad",
                    "motor_pre_projection_requested_rad",
                    "pd_requested_torque_nm",
                )
            }
        )
        previous_navigation_forward = 0.0
    if use_contact_exit_torque_mpc:
        history.update(
            {
                name: []
                for name in (
                    "contact_exit_mpc_requested_offset_nm",
                    "contact_exit_mpc_applied_offset_nm",
                    "contact_exit_mpc_forecast_steps",
                )
            }
        )
        mpc_plans = []
        previous_mpc_offset = np.zeros(29)
    if use_directional_contact_compliance:
        history.update(
            {
                name: []
                for name in (
                    "directional_compliance_stiffness_nm_per_rad",
                    "directional_compliance_requested_delta_nm",
                    "directional_compliance_applied_delta_nm",
                    "directional_compliance_forecast_steps",
                )
            }
        )
        compliance_plans = []
    tracking = y < 0
    gate = False
    side, contact_frame = -1, None
    swing_contact_delta = np.zeros(6)
    previous, motor_contact_delta = np.zeros(12), np.zeros(12)
    leg_ids = [np.asarray([names.index(n) for n in row]) for row in LEG_NAMES]
    motor_ids = np.concatenate(leg_ids)
    limits = model.jnt_range[joints].copy()
    minimum_pelvis = float("inf")
    first_contact_frame = None
    first_contact_bodies: list[int] = []
    observer = (
        CurrentKinematicObserver(model)
        if args.observation_snapshot == "current-kinematic"
        else None
    )
    for frame in range(300):
        if observer is None:
            positions, quaternions = data.xpos, data.xquat
            root_vel = root_velocity_world(model, data, pelvis, args.root_velocity_reference)
            ball_vel = np.zeros(6)
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, ball, ball_vel, 0)
            ball_linear_vel = ball_vel[3:]
        else:
            current = observer.sample(data)
            positions, quaternions = current.body_position_m, current.body_quaternion_wxyz
            root_vel = current.body_origin_velocity_world[pelvis]
            ball_linear_vel = current.body_origin_velocity_world[ball, :3]
        root_pose = np.concatenate((positions[pelvis], quaternions[pelvis][[1, 2, 3, 0]]))
        for key, value in (
            ("root_pose_xyzw_m", root_pose),
            ("root_velocity_world", root_vel),
            ("ball_position_before_step_m", positions[ball]),
            ("ball_linear_velocity_before_step_m_s", ball_linear_vel),
            ("foot_geometry_position_before_step_m", positions[bodies]),
            ("joint_position_rad", data.qpos[qi]),
            ("joint_velocity_rad_s", data.qvel[vi]),
        ):
            history[key].append(value.copy()[None])
        gap = positions[ball] - positions[pelvis]
        if control_profile is None:
            lateral = native_lateral_command(
                gap_x=float(gap[0]),
                gap_y=float(gap[1]),
                tracking=tracking,
                before_contact=contact_frame is None,
                mode=approach_mode,
            )
            navigation_command = (1.4, lateral, 0.0)
        else:
            requested_navigation, navigation_command = control_profile.navigation(
                gap_x=float(gap[0]),
                gap_y=float(gap[1]),
                ball_vx=float(ball_linear_vel[0]),
                before_contact=contact_frame is None,
            )
            if postcontact_profile is not None and contact_frame is not None:
                raw_forward, actual_forward = postcontact_profile.navigation(
                    previous_forward=previous_navigation_forward,
                    gap_x=float(gap[0]),
                    ball_vx=float(ball_linear_vel[0]),
                    forward_cap=control_profile.forward_cap_m_s,
                )
                requested_navigation = (raw_forward, 0.0, 0.0)
                navigation_command = (actual_forward, 0.0, 0.0)
            previous_navigation_forward = navigation_command[0]
            history["approach_requested_navigation_command"].append(
                np.asarray(requested_navigation)[None]
            )
        qpos = np.concatenate((data.qpos[:7], data.qpos[qi], data.qpos[bq : bq + 7]))
        qvel = np.concatenate((data.qvel[:6], data.qvel[vi], data.qvel[bv : bv + 6]))
        history["canonical_qpos"].append(qpos.copy()[None])
        history["canonical_qvel"].append(qvel.copy()[None])
        observation = TeamMotorObservation(
            agent_id=navigation.agent_id,
            frame=frame,
            time_sec=frame * 0.02,
            intent="other",
            prospective_owner=False,
            qpos=tuple(float(v) for v in qpos),
            qvel=tuple(float(v) for v in qvel),
            target_position_m=(0.0, 0.0, 0.0),
            navigation_command=navigation_command,
            navigation_envelope=navigation.navigation_envelope,
        )
        if frame == 0:
            navigation.start_from_observation(observation)
        navigation.prepare_batched_proposal(observation)
        if tracker is None:
            tracker = BatchedSonicTracker(
                foundation,
                navigation.backend.reference[None],
                low_latency_legacy_encoder_layout=True,
                capture_neural_observation=args.record_foundation_observation,
            )
            tracker.reset(qpos[None], qvel[None])
        else:
            tracker.observe(qpos[None], qvel[None])
            if frame % 20 == 0:
                tracker.refresh_unexecuted_reference(
                    frame, navigation.backend.reference[None], unchanged_lookahead_frames=10
                )
        tracker.update(frame, qpos[None], qvel[None])
        if args.record_foundation_observation:
            for name, value in tracker.neural_observation().items():
                history[PREFIX + name].append(value.cpu().numpy().copy())
        proposal = navigation.commit_batched_action(tracker.action[0].numpy())
        target = np.asarray(proposal.target_rad).copy()
        if control_profile is not None:
            navigation_target = target.copy()
            history["foundation_navigation_joint_target_rad"].append(navigation_target[None])
        if frame == 30:
            raw = current_context(
                root_pose[None], root_vel[None], positions[ball][None], ball_linear_vel[None]
            )
            features = gait_phase_features(
                raw, predict_contact_time(raw, np.asarray(late["contact_time_weights"]))
            )
            gate = native_swing_eligible(
                memory_gate=bool(
                    select_taskspace_gate(
                        features,
                        np.asarray(late["memory_features"]),
                        np.asarray(late["memory_clean"]),
                        np.asarray(late["memory_reward"]),
                        np.asarray(late["memory_groups"]),
                        neighbors=late["neighbors"],
                        confidence=late["confidence"],
                        baseline_clean_ceiling=late["baseline_clean_ceiling"],
                    )[0]
                ),
                initial_vx=float(vx),
                foundation_only=args.foundation_only,
                mode=swing_mode,
            )
            if neural:
                policy = configure_preview(neural, history, contact_frame)
                knots = np.asarray(policy["knots_rad"])
        swing_delta = np.zeros(6)
        if gate:
            if contact_frame is None:
                side = choose_swing_side(
                    positions[bodies[:2]],
                    positions[ball],
                    side,
                    acquisition_max_gap_m=0.95,
                    revalidate_swing_side=True,
                )
            if side >= 0:
                ids = leg_ids[side]
                if contact_frame is not None:
                    if suppress_postcontact_swing_release:
                        swing_delta = np.zeros(6)
                    else:
                        swing_delta = release_joint_delta(
                            swing_contact_delta, frame - contact_frame
                        )
                else:
                    jac = np.zeros((3, model.nv))
                    mujoco.mj_jacBody(model, data, jac, None, bodies[side])
                    if control_profile is None:
                        swing_delta = swing_joint_delta(
                            positions[bodies[side]],
                            positions[ball],
                            jac[:, vi[ids]],
                            target[ids],
                            limits[ids],
                            forward_cap_m=foot_forward_cap,
                            lateral_cap_m=foot_lateral_cap,
                            lateral_lead_m=0.0,
                            vertical_offset_m=foot_vertical_offset,
                        )
                    else:
                        swing_delta = continuous_swing_joint_delta(
                            positions[bodies[side]],
                            positions[ball],
                            jac[:, vi[ids]],
                            target[ids],
                            limits[ids],
                            forward_cap_m=foot_forward_cap,
                            lateral_cap_m=foot_lateral_cap,
                            vertical_offset_m=foot_vertical_offset,
                        )
                target[ids] += swing_delta
        history["pre_motor_joint_target_rad"].append(target.copy()[None])
        if control_profile is not None:
            history["taskspace_joint_increment_rad"].append((target - navigation_target)[None])
        if step_model is not None:
            if policy is None:
                raise ValueError("per-frame actor requires a sealed preview")
            prior_final = previous.copy()
            previous = delta_at_frame(
                policy,
                history,
                frame=frame,
                nominal_target=target,
                baseline=target[motor_ids],
                limits=limits[motor_ids],
                previous=previous,
                previous_contact_forces=history["force_n"][frame - 1][0] if frame else np.zeros(6),
            )
            if body_guidance is not None:
                if not np.array_equal(motor_ids, np.arange(12)):
                    raise ValueError("guidance requires the canonical twelve DDS leg joints")
                previous, guidance_status = body_guidance.advance(
                    proposal_decoder,
                    history,
                    frame=frame,
                    nominal_target=target,
                    parent_delta=previous,
                    previous_final=prior_final,
                    limits=(limits if body_guidance.action_dimensions == 29 else limits[motor_ids]),
                )
                history["body_response_applied_increment"].append(
                    np.asarray(guidance_status["applied_increment"])[None]
                )
                history["body_response_status"].append(
                    np.asarray(
                        [
                            guidance_status["contact_phase"],
                            guidance_status["protected"],
                            guidance_status["active"],
                            guidance_status["fallback"],
                        ],
                        dtype=np.int64,
                    )[None]
                )
            if body_guidance is not None and body_guidance.action_dimensions == 29:
                target += np.asarray(guidance_status["final_target_increment"])
            else:
                target[motor_ids] += previous
        elif policy is not None:
            if knots is None:
                raise ValueError("motor policy requires authenticated knots")
            previous = motor_delta(
                policy,
                knots,
                float(gap[0]),
                target[motor_ids],
                limits[motor_ids],
                previous,
                motor_contact_delta,
                frame - contact_frame if contact_frame is not None else None,
            )
            target[motor_ids] += previous
        if recurrent_decoder is not None:
            history[STATE_FIELD].append(recurrent_decoder.hidden_state[None])
            if recurrent_sampling:
                for name, value in recurrent_decoder.sampled_transition.items():
                    history[name].append(value[None])
        history["motor_delta_rad"].append(previous.copy()[None])
        history["navigation_command"].append(np.asarray(navigation_command)[None])
        if control_profile is not None:
            from rosclaw_soccer.rsi.contact_motor_primitive import CAP_RAD

            raw_motor = (
                np.zeros(12)
                if frame < 30 or foundation_approach_probe
                else CAP_RAD
                * np.tanh(recurrent_decoder.sampled_transition["recurrent_sampling_latent_action"])
            )
            history["motor_pre_projection_requested_rad"].append(raw_motor[None])
        if use_contact_exit_torque_mpc:
            past_forces = np.asarray(history["force_n"], dtype=np.float64).reshape(frame, 6)
            foot_frames = np.flatnonzero(np.any(past_forces[:, :2] > 1, axis=1))
            mpc_plan = exit_teacher.plan(
                data,
                target,
                frame=frame,
                first_foot_contact_frame=int(foot_frames[0]) if len(foot_frames) else None,
                force_history=past_forces[-3:],
                previous_offset_nm=previous_mpc_offset,
                prior_nonfoot_contact=bool(np.any(past_forces[:, 2:] > 1)),
            )
            previous_mpc_offset = mpc_plan.offset_nm.copy()
            mpc_plans.append(mpc_plan.report)
            history["contact_exit_mpc_requested_offset_nm"].append(previous_mpc_offset[None])
            history["contact_exit_mpc_forecast_steps"].append(
                np.asarray([[mpc_plan.report["forecast_substeps"]]], dtype=np.int64)
            )
            mpc_applied = []
        if use_directional_contact_compliance:
            compliance_past = np.asarray(history["force_n"], dtype=np.float64).reshape(frame, 6)
            compliance_foot_frames = np.flatnonzero(np.any(compliance_past[:, :2] > 1, axis=1))
            compliance_plan = exit_compliance.plan(
                data,
                target,
                frame=frame,
                first_foot_contact_frame=int(compliance_foot_frames[0])
                if len(compliance_foot_frames)
                else None,
                force_history=compliance_past[-3:],
                prior_nonfoot_contact=bool(np.any(compliance_past[:, 2:] > 1)),
            )
            compliance_plans.append(compliance_plan.report)
            history["directional_compliance_stiffness_nm_per_rad"].append(
                compliance_plan.stiffness_nm_per_rad[None].copy()
            )
            history["directional_compliance_forecast_steps"].append(
                np.asarray([[compliance_plan.report["forecast_substeps"]]], dtype=np.int64)
            )
            compliance_requested, compliance_applied = [], []
        history["joint_target_rad"].append(target[None].copy())
        forces = np.zeros(6)
        torques = []
        requested_torques = []
        actual_forces = []
        pelvis_samples = []
        cerebellar_samples: list[dict[str, np.ndarray]] = []
        for _ in range(10):
            requested_torque = (target - data.qpos[qi]) * navigation.backend.kp - data.qvel[
                vi
            ] * navigation.backend.kd
            torque = np.clip(
                requested_torque,
                -np.asarray(G1_HARD_TORQUE_LIMITS),
                np.asarray(G1_HARD_TORQUE_LIMITS),
            )
            if use_contact_exit_torque_mpc:
                baseline_torque = torque.copy()
                if np.any(previous_mpc_offset):
                    torque = np.clip(
                        requested_torque + previous_mpc_offset,
                        -np.asarray(G1_HARD_TORQUE_LIMITS),
                        np.asarray(G1_HARD_TORQUE_LIMITS),
                    )
                mpc_applied.append(torque - baseline_torque)
            if use_directional_contact_compliance:
                parent_torque = torque.copy()
                compliant_request = requested_torque
                if not compliance_plan.identity:
                    compliant_request = (
                        compliance_plan.stiffness_nm_per_rad @ (target - data.qpos[qi])
                        - data.qvel[vi] * navigation.backend.kd
                    )
                    torque = np.clip(
                        compliant_request,
                        -np.asarray(G1_HARD_TORQUE_LIMITS),
                        np.asarray(G1_HARD_TORQUE_LIMITS),
                    )
                compliance_requested.append(compliant_request - requested_torque)
                compliance_applied.append(torque - parent_torque)
            if cerebellar is not None:
                current_qpos = np.concatenate(
                    (data.qpos[:7], data.qpos[qi], data.qpos[bq : bq + 7])
                )
                current_qvel = np.concatenate(
                    (data.qvel[:6], data.qvel[vi], data.qvel[bv : bv + 6])
                )
                torque, cerebellar_sample = cerebellar.advance(
                    current_qpos, current_qvel, target, requested_torque
                )
                cerebellar_samples.append(cerebellar_sample)
            data.ctrl[ai] = torque
            torques.append(torque.copy())
            if control_profile is not None:
                requested_torques.append(requested_torque.copy())
            mujoco.mj_step(model, data)
            actual_forces.append(data.actuator_force[ai].copy())
            for c in range(data.ncon):
                contact = data.contact[c]
                g1, g2 = int(contact.geom1), int(contact.geom2)
                if ball_geom not in (g1, g2):
                    continue
                other = g2 if g1 == ball_geom else g1
                body = int(model.geom_bodyid[other])
                if body not in contact_bodies:
                    continue
                force = np.zeros(6)
                mujoco.mj_contactForce(model, data, c, force)
                index = contact_bodies.index(body)
                forces[index] = max(forces[index], float(np.linalg.norm(force[:3])))
            minimum_pelvis = min(minimum_pelvis, float(data.xpos[pelvis, 2]))
            pelvis_samples.append(float(data.xpos[pelvis, 2]))
        if use_contact_exit_torque_mpc:
            exit_teacher.verify_actual_interval(data, mpc_plan)
            history["contact_exit_mpc_applied_offset_nm"].append(np.asarray(mpc_applied)[None])
        if use_directional_contact_compliance:
            exit_compliance.verify_actual_interval(data, compliance_plan)
            history["directional_compliance_requested_delta_nm"].append(
                np.asarray(compliance_requested)[None]
            )
            history["directional_compliance_applied_delta_nm"].append(
                np.asarray(compliance_applied)[None]
            )
        if contact_frame is None and np.any(forces > 1):
            contact_frame = frame
            first_contact_frame = frame
            first_contact_bodies = np.flatnonzero(forces > 1).tolist()
            swing_contact_delta = swing_delta.copy()
            motor_contact_delta = previous.copy()
        if control_profile is not None:
            history["pd_requested_torque_nm"].append(np.asarray(requested_torques)[None])
        if cerebellar is not None:
            for name in CEREBELLAR_WIDTHS:
                history[name].append(
                    np.asarray([sample[name] for sample in cerebellar_samples])[None]
                )
        history["force_n"].append(forces[None])
        history["ball_position_after_step_m"].append(data.xpos[ball].copy()[None])
        history["torque_nm"].append(np.asarray(torques)[None])
        history["actual_actuator_force_nm"].append(np.asarray(actual_forces)[None])
        history["pelvis_z_per_substep_m"].append(np.asarray(pelvis_samples)[None])
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise ValueError("nonfinite physical MuJoCo state")
    if (
        control_profile is not None
        and control_profile.contract() != commitment["bounded_control_profile"]
    ):
        raise ValueError("bounded control profile changed during the episode")
    if use_contact_exit_torque_mpc:  # noqa: SIM102 - retain explicit OFF-path projection
        if exit_teacher.contract() != commitment["contact_exit_torque_mpc"]:
            raise ValueError("contact-exit torque teacher changed during episode")
    if use_directional_contact_compliance:  # noqa: SIM102 - explicit OFF-path projection
        if exit_compliance.contract() != commitment["directional_contact_compliance"]:
            raise ValueError("directional compliance controller changed during episode")
    trace = args.output_root / "physical_trace.npz"
    if cerebellar is not None and (
        cerebellar.calls != 3000
        or cerebellar.contract() != commitment["cerebellar_torque_contract"]
    ):
        raise ValueError("whole unchanged neural torque episode and 3000 substep calls required")
    arrays: dict[str, Any] = {k: np.asarray(v) for k, v in history.items()}
    if mesh_world_review is not None:
        assert mesh_task_contact_world_hash is not None
        if mesh_world_review.verify(mesh_task_contact_world_hash) != mesh_review_binding:
            raise ValueError("contact world numerical review changed during the physical episode")
    if args.record_foundation_observation:
        from rosclaw_soccer.rsi.foundation_observation_capture import validate_capture

        validate_capture(commitment["foundation_observation_capture"], arrays, frames=300, lanes=1)
    np.savez_compressed(trace, **arrays)
    result = dict(
        commitment,
        commitment_hash=hash_json(commitment),
        physical_trace_hash=hash_bytes(trace.read_bytes()),
        first_contact_frame=first_contact_frame,
        first_contact_bodies=first_contact_bodies,
        clean_foot_only=bool(
            first_contact_bodies
            and all(i < 2 for i in first_contact_bodies)
            and not np.any(np.asarray(history["force_n"])[:, :, 2:] > 1)
        ),
        minimum_pelvis_z_m=minimum_pelvis,
        final_ball_position_m=data.xpos[ball].tolist(),
        taskspace_gate_selected=gate,
        executed_motor_policy=policy,
        qualification="UNQUALIFIED_CPU_TRANSFER_DIAGNOSTIC",
    )
    if control_profile is not None and locomotion_mode_probe is not None:
        result["locomotion_mode_planner_events"] = navigation.backend.events
    if record_default_planner_events:
        result["default_locomotion_planner_events"] = navigation.backend.events
    if use_contact_exit_torque_mpc:
        result["contact_exit_mpc_plans"] = mpc_plans
        result["contact_exit_mpc_counters"] = exit_teacher.counters()
    if use_directional_contact_compliance:
        result["directional_compliance_plans"] = compliance_plans
        result["directional_compliance_counters"] = exit_compliance.counters()
    result["report_hash"] = hash_json(result)
    from scripts.rsi_atomic_artifacts import write_once, write_shared_physical_report

    report_name = "report.json.gz" if args.compressed_report else "report.json"
    if args.shared_evidence:
        write_shared_physical_report(args.output_root / report_name, result)
    else:
        write_once(args.output_root / report_name, result)
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "report_hash",
                    "first_contact_frame",
                    "clean_foot_only",
                    "minimum_pelvis_z_m",
                )
            }
        )
    )
