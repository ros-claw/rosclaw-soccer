"""Native SIM-only adapter for a prospective local impact search.

Owns a checked mirror of the original IK state, not another motion primitive.
Raw candidate windows are written before the caller applies any correction.
This adapter never qualifies itself or grants hardware/learning permission.
"""

from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from rosclaw_soccer.rsi.feedback_branch_control import BODY_SHAPES
from rosclaw_soccer.rsi.foot_heading_evidence import FootHeadingCpuReconstruction
from rosclaw_soccer.rsi.impact_feedback_teacher import (
    MAX_BASELINE_WINDOWS,
    MAX_FORECAST_STEPS,
    ImpactCandidate,
    ImpactFeedbackTeacher,
)
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def require_native_impact_request(v: dict[str, Any]) -> None:
    from rosclaw_soccer.rsi.bounded_first_touch_control import BoundedFirstTouchControl
    from rosclaw_soccer.rsi.foot_heading_control import (
        FootHeadingContactProbe,
        require_heading_request,
    )
    from rosclaw_soccer.rsi.original_ik_planner_pulse import OriginalIkPlannerPulse
    from rosclaw_soccer.rsi.recurrent_sampling_episode_factory import (
        RecurrentSamplingEpisodeFactory,
    )

    allowed = {
        "args",
        "course",
        "partition",
        "entry_source_path",
        "recurrent_sampling_factory",
        "approach_mode",
        "swing_mode",
        "control_profile",
        "record_default_planner_events",
        "mesh_task_contact_world_hash",
        "mesh_world_review",
        "generated_seed_episode_binding",
        "original_ik_planner_pulse",
        "foot_heading_control",
        "use_impact_feedback_teacher",
    }
    none = {
        "consumed_bank_hash",
        "training_plan",
        "training_group",
        "sampling_factory",
        "extended_factory",
        "proposal_factory",
        "imitation_factory",
        "recurrent_factory",
        "recurrent_clipped_factory",
        "postcontact_profile",
        "locomotion_mode_probe",
        "reference_height_probe_m",
        "cerebellar_model",
        "cerebellar_sampling_seed",
        "reference_student_model",
        "planner_navigation_pulse",
    }
    off = {
        "foundation_approach_probe",
        "suppress_postcontact_swing_release",
        "use_contact_exit_torque_mpc",
        "use_directional_contact_compliance",
        "use_mesh_surface_swing",
    }
    args = v.get("args")
    if (
        set(v) != allowed | none | off
        or any(v[k] is not None for k in none)
        or any(v[k] is not False for k in off)
        or v["use_impact_feedback_teacher"] is not True
        or v["generated_seed_episode_binding"] is not True
        or v["record_default_planner_events"] is not True
        or v["partition"] != "DECLARED_DEVELOPMENT_DIAGNOSTIC"
        or v["approach_mode"] != "bounded_approach_diagnostic"
        or v["swing_mode"] != "bounded_contact_diagnostic"
        or type(v["control_profile"]) is not BoundedFirstTouchControl
        or type(v["recurrent_sampling_factory"]) is not RecurrentSamplingEpisodeFactory
        or type(v["original_ik_planner_pulse"]) is not OriginalIkPlannerPulse
        or type(v["foot_heading_control"]) is not FootHeadingContactProbe
        or v["mesh_task_contact_world_hash"] is None
        or v["mesh_world_review"] is None
        or getattr(args, "record_foundation_observation", False) is not True
        or getattr(args, "foundation_only", False) is not False
        or getattr(args, "proposal_decoder", None) != "reference"
        or getattr(args, "step_model", None) is None
        or getattr(args, "body_response_bundle", None) is not None
        or getattr(args, "cached_proposal_envelope", False) is not False
        or getattr(args, "neural_model", None) is not None
        or getattr(args, "motor_policy", None) is not None
        or getattr(args, "observation_snapshot", None) != "cached"
        or getattr(args, "root_velocity_reference", None) != "body-com"
    ):
        raise ValueError(
            "captured unmixed positive original-IK full-feedback SIM diagnostic required"
        )
    if v["original_ik_planner_pulse"].lateral_delta_m_s != 0.08:
        raise ValueError("same fixed positive original-IK predecessor required")
    require_heading_request(
        v["foot_heading_control"],
        pulse=v["original_ik_planner_pulse"],
        reference_model=None,
        profile=v["control_profile"],
        record_planner_events=True,
    )


def close(actual: Any, expected: Any, name: str) -> None:
    a, e = np.asarray(actual), np.asarray(expected)
    if (
        a.shape != e.shape
        or not np.isfinite(a).all()
        or not np.isfinite(e).all()
        or not np.allclose(a, e, atol=1e-8, rtol=0)
    ):
        raise ValueError("native impact original-controller mirror differs: " + name)


class NativeImpactFeedback:
    def __init__(
        self,
        model: Any,
        *,
        world_hash: str,
        kp: np.ndarray,
        kd: np.ndarray,
        heading_probe: Any,
        output_root: Path,
    ) -> None:
        self.model = model
        self.teacher = ImpactFeedbackTeacher(model, world_hash=world_hash, kp=kp, kd=kd)
        self.ik = FootHeadingCpuReconstruction(model, heading_probe)
        self.root = output_root / "impact-feedback-windows"
        self.root.mkdir()
        self.world_hash = world_hash
        self.reports: list[dict[str, Any]] = []

    def contract(self) -> dict[str, Any]:
        import rosclaw_soccer.physics.g1_feedback_forecast as forecast
        import rosclaw_soccer.physics.g1_joint_task_kinematics as geometry
        import rosclaw_soccer.rsi.feedback_branch_control as control
        import rosclaw_soccer.rsi.feedback_impact_score as score
        import rosclaw_soccer.rsi.impact_feedback_teacher as teacher
        import rosclaw_soccer.rsi.impact_torque_screen as actions

        source_hashes = {}
        for m in (forecast, geometry, control, score, teacher, actions):
            filename = m.__file__
            if not isinstance(filename, str):
                raise ValueError("reviewable filesystem source required")
            source_hashes[m.__name__] = hash_bytes(Path(filename).read_bytes())
        c = dict(
            schema="soccer.rsi.native_prospective_impact_feedback.v1",
            world_hash=self.world_hash,
            source_hashes=source_hashes,
            adapter_source_hash=hash_bytes(Path(__file__).read_bytes()),
            max_baseline_windows=MAX_BASELINE_WINDOWS,
            maximum_forecast_steps=MAX_FORECAST_STEPS,
            current_surface_gap_interval_m=[-0.02, 0.06],
            only_current_precontact_swing_leg=True,
            maximum_action_duration_s=0.02,
            source_future_contact_frame_forbidden=True,
            full_native_h300_required=True,
            neural_policy_trained=False,
            teacher_qualified=False,
            hardware_authorized=False,
            promotion_authorized=False,
        )
        return dict(c, contract_hash=hash_json(c))

    def _save(
        self, frame: int, candidates: list[ImpactCandidate] | tuple[ImpactCandidate, ...]
    ) -> list[dict[str, Any]]:
        result = []
        for i, candidate in enumerate(candidates):
            p = self.root / f"frame-{frame:03d}-candidate-{i:02d}.npz"
            f = candidate.forecast
            values: dict[str, Any] = dict(
                f.arrays,
                end_integration_state=f.end_integration_state,
                offset_nm=candidate.offset_nm,
                **{"next_" + k: v for k, v in f.next_target_record.items()},
                **{"neural_next_" + k: v for k, v in candidate.next_neural_capture.items()},
            )
            with p.open("xb") as stream:
                np.savez_compressed(stream, **values)
            result.append(
                dict(
                    path=str(p),
                    sha256=hash_bytes(p.read_bytes()),
                    name=candidate.name,
                    donor_hash=f.donor_integration_state_hash,
                )
            )
        return result

    def before_frame(
        self,
        data: Any,
        *,
        frame: int,
        navigation_target: np.ndarray,
        native_increment: np.ndarray,
        target: np.ndarray,
        previous: np.ndarray,
        tracker: Any,
        decoder: Any,
        policy: dict[str, Any],
        history: dict[str, Any],
    ) -> np.ndarray:
        try:
            increment = self.ik.before_step(data, frame, data.xpos, navigation_target)
            close(increment, native_increment, "original IK+heading increment")
            expected = navigation_target + increment
            expected[:12] += previous
            close(expected, target, "original Foundation+IK+residual target")
            plan = self.teacher.plan(
                data,
                target,
                frame=frame,
                tracker=tracker,
                decoder=decoder,
                ik=self.ik,
                policy=policy,
                body_prefix={k: np.asarray(history[k], dtype=np.float64) for k in BODY_SHAPES},
                previous_motor_delta_rad=previous,
            )
        except BaseException as error:
            # Preserve completed predictions and the attempted integrator state.
            witnesses = []
            if self.teacher.last_frame == frame:
                witnesses = self._save(frame, self.teacher.last_candidates)
            d = self.teacher.forecaster._scratch
            spec = mujoco.mjtState.mjSTATE_INTEGRATION
            state = np.empty(mujoco.mj_stateSize(self.model, spec))
            mujoco.mj_getState(self.model, d, state, spec)
            with (self.root / f"frame-{frame:03d}-failure-state.npz").open("xb") as stream:
                np.savez_compressed(stream, integration_state=state)
            native_state = np.empty_like(state)
            mujoco.mj_getState(self.model, data, native_state, spec)
            prefix: dict[str, Any] = {k: np.asarray(v) for k, v in history.items()}
            prefix["native_integration_state"] = native_state
            with (self.root / f"frame-{frame:03d}-failure-prefix.npz").open("xb") as stream:
                np.savez_compressed(stream, **prefix)
            from scripts.rsi_atomic_artifacts import write_once

            f = self.teacher.forecaster
            failure = dict(
                frame=frame,
                error_type=type(error).__name__,
                error=str(error),
                completed_witnesses=witnesses,
                native_substeps_before_failed_decision=frame * 10,
                forecast_attempted_steps=f.attempted_steps,
                forecast_integrated_steps=f.integrated_steps,
                branch_target_attempts=f.next_target_attempts,
                branch_target_completed=f.next_target_completed,
                hardware_authorized=False,
                promotion_authorized=False,
            )
            failure["report_hash"] = hash_json(failure)
            write_once(self.root / f"frame-{frame:03d}-failure.json", failure)
            raise
        witnesses = self._save(frame, plan.candidates)
        report = dict(plan.report, raw_witnesses=witnesses)
        report["report_hash"] = hash_json(report)
        from scripts.rsi_atomic_artifacts import write_once

        write_once(self.root / f"frame-{frame:03d}.json", report)
        self.reports.append(report)
        return plan.offset_nm.copy()

    def after_frame(self, frame: int, forces: np.ndarray) -> None:
        self.ik.after_step(frame, forces)

    def review(self) -> dict[str, Any]:
        if len(self.reports) != 300 or self.teacher.last_frame != 299:
            raise ValueError("full300-frame prospective feedback capture required")
        f = self.teacher.forecaster
        return dict(
            frames_completed=300,
            original_ik_mirror=self.ik.review(),
            forecast_attempted_steps=f.attempted_steps,
            forecast_integrated_steps=f.integrated_steps,
            branch_target_attempts=f.next_target_attempts,
            branch_target_completed=f.next_target_completed,
            baseline_windows=self.teacher.baseline_windows,
            candidate_bank_evaluated=self.teacher.screened,
            hardware_authorized=False,
            promotion_authorized=False,
            teacher_qualified=False,
        )
