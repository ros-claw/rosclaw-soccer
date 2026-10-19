"""Twelve paired full native H300 diagnostic bodies; independent audit still required."""

import argparse
import gc
import json
import shutil
import time
from pathlib import Path

import numpy as np

from rosclaw_soccer.rsi.bounded_first_touch_control import BoundedFirstTouchControl
from rosclaw_soccer.rsi.first_touch_outcome import label_first_touch
from rosclaw_soccer.rsi.foot_heading_control import FootHeadingContactProbe
from rosclaw_soccer.rsi.json_artifact_io import load_json_artifact
from rosclaw_soccer.rsi.late_swing_memory import load_late_swing_actor
from rosclaw_soccer.rsi.native_first_touch_episode import run_native_first_touch_episode
from rosclaw_soccer.rsi.original_ik_planner_pulse import OriginalIkPlannerPulse
from rosclaw_soccer.rsi.recurrent_sampling_episode_factory import RecurrentSamplingEpisodeFactory
from rosclaw_soccer.rsi.reviewed_contact_world import ReviewedContactWorld
from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_feedback_branch_inputs import prepare as original_prepare
from scripts.rsi_stage_release_contact_replay import digest, require_terminal


def require_scope(cfg):
    if (
        cfg.get("mode") != "SIM_ONLY"
        or cfg.get("profile") != "twelve-native-prospective-impact-v1"
        or any(
            type(cfg.get(k)) is not int or cfg[k] != n
            for k, n in (
                ("source_cases", 6),
                ("new_native_episodes", 12),
                ("native_step_budget", 36000),
                ("forecast_step_budget", 5400),
                ("combined_step_budget", 41400),
                ("new_independent_cpu_episodes", 0),
                ("official_horizon_frames", 300),
                ("post_contact_frames", 60),
                ("new_optimizer_updates", 0),
            )
        )
        or any(
            cfg.get(k) is not False
            for k in (
                "hardware_authorized",
                "promotion_authorized",
                "training_authorized",
                "teacher_qualified",
                "private_fresh_accessed",
                "independent_qualification_claimed",
            )
        )
    ):
        raise ValueError(
            "exact twelve-body41400-step unqualified native prospective diagnostic required"
        )


def prepare(path):
    p = original_prepare(path, scope_validator=require_scope)
    if Path(__file__).resolve() != p.source / "scripts/rsi_native_prospective_impact.py":
        raise ValueError("exact frozen prospective runner required")
    require_terminal(p.cfg["screen_readback_unit"])
    require_terminal(p.cfg["recorded_replay_unit"])

    def sealed(name, expected):
        path = Path(name)
        h = digest(path)
        if path.is_symlink() or h != expected or (str(path) in p.pins and p.pins[str(path)] != h):
            raise ValueError("original complete predecessor bytes required")
        d = json.loads(path.read_text())
        if d["report_hash"] != hash_json({k: v for k, v in d.items() if k != "report_hash"}):
            raise ValueError("original predecessor seal required")
        p.pins[str(path)] = h
        return d

    readback = sealed(p.cfg["screen_readback"], p.cfg["screen_readback_file_hash"])
    replay_root = Path(p.cfg["recorded_replay_root"])
    replay = sealed(replay_root / "complete.json", p.cfg["recorded_replay_complete_file_hash"])
    declaration = json.loads((replay_root / "declaration.json").read_text())
    if (
        readback["report_hash"] != p.cfg["screen_readback_hash"]
        or readback["schema"] != "soccer.rsi.feedback_impact_screen_readback.v1"
        or replay["report_hash"] != p.cfg["recorded_replay_complete_hash"]
        or replay["schema"] != "soccer.rsi.recorded_feedback_replay_complete.v1"
        or replay["all_original_six_and_all_nonzero_branches_physically_replayed"] is not True
        or replay["branch_windows_completed"] != 84
        or len(readback["rows"]) != 6
        or len(replay["rows"]) != 6
        or any(
            replay[k] is not False or readback[k] is not False
            for k in (
                "hardware_authorized",
                "promotion_authorized",
                "teacher_qualified",
                "full_m0_gain_proven",
            )
        )
    ):
        raise ValueError(
            "whole completed six-case non-promoting screen and physical replay required"
        )
    if (
        declaration["report_hash"]
        != hash_json({k: v for k, v in declaration.items() if k != "report_hash"})
        or replay["declaration_hash"] != declaration["report_hash"]
        or declaration["readback_hash"] != readback["report_hash"]
    ):
        raise ValueError("same original whole independent replay declaration required")
    p.pins[str(replay_root / "declaration.json")] = digest(replay_root / "declaration.json")
    for proof in (readback, declaration):
        for name, h in proof["input_file_hashes"].items():
            if digest(name) != h or (name in p.pins and p.pins[name] != h):
                raise ValueError("whole prior source/model/raw proof changed")
            p.pins[name] = h
    p.predecessors = dict(
        readback_hash=readback["report_hash"], recorded_replay_hash=replay["report_hash"]
    )
    p.scene = Path(p.cfg["scene_xml"])
    p.late = Path(p.cfg["late_swing_policy"])
    for name in (p.scene, p.late):
        if str(name) not in p.pins or digest(name) != p.pins[str(name)]:
            raise ValueError("same original source-authenticated scene and late actor required")
    p.body_inputs = []
    late_actor_hash = load_late_swing_actor(p.late)["actor_hash"]
    for case, old, rr in zip(p.cases, readback["rows"], replay["rows"], strict=True):
        r = case["receipt"]
        if (
            r["receipt_hash"] != old["source_receipt_hash"]
            or r["receipt_hash"] != rr["source_receipt_hash"]
        ):
            raise ValueError("every original ordered case including failures required")
        w = r["planning_context_witness"]
        context = sealed(w["path"], w["sha256"])
        h = context["source_sampling_view_hash"]
        views = [
            Path(n)
            for n, expected in p.pins.items()
            if expected == h and Path(n).parent.name == "models"
        ]
        if len(views) != 1:
            raise ValueError("unambiguous original whole local sampling view required")
        c = json.loads(Path(r["witnesses"]["commitment"]["path"]).read_text())
        review = c["mesh_task_contact_protocol"]["numerical_review"]
        p.review = ReviewedContactWorld(Path(review["path"]), review["file_hash"])
        if (
            p.review.verify(p.world_hash) != review
            or p.pins.get(str(p.review.path)) != p.review.expected_file_hash
        ):
            raise ValueError("same original numerically reviewed world required")
        if c["late_actor_hash"] != late_actor_hash:
            raise ValueError("same source late actor required")
        p.body_inputs.append(dict(view=views[0], source_commitment=c))
    return p


def require_prefix(trace, original, intervention_frame):
    frames = 300 if intervention_frame is None else intervention_frame
    if type(frames) is not int or not 0 <= frames <= 300:
        raise ValueError("measured actual intervention boundary required")
    pre_step = {
        "canonical_qpos",
        "canonical_qvel",
        "joint_position_rad",
        "joint_velocity_rad_s",
        "root_pose_xyzw_m",
        "root_velocity_world",
        "ball_position_before_step_m",
        "ball_linear_velocity_before_step_m_s",
        "foot_geometry_position_before_step_m",
        "joint_target_rad",
        "pre_motor_joint_target_rad",
        "motor_delta_rad",
        "foundation_navigation_joint_target_rad",
        "taskspace_joint_increment_rad",
        "navigation_command",
        "recurrent_residual_hidden_state",
    }
    for key, value in original.items():
        n = frames + int(
            frames < 300
            and (key in pre_step or key.startswith(("foundation_neural_", "recurrent_sampling_")))
        )
        if (
            key not in trace
            or trace[key].shape != value.shape
            or trace[key].dtype != value.dtype
            or trace[key][:n].tobytes() != value[:n].tobytes()
        ):
            raise ValueError("whole original OFF or causal pre-intervention prefix differs: " + key)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    p = prepare(args.config)
    if args.preflight:
        print("TWELVE_NATIVE_PROSPECTIVE_READY_NO_FULL_MODELS_OR_PHYSICS", flush=True)
        return
    available = next(
        int(s.split()[1]) * 1024
        for s in Path("/proc/meminfo").read_text().splitlines()
        if s.startswith("MemAvailable:")
    )
    if available < 56 * 1024**3:
        raise ValueError("56GiB available before full frozen mean allocation required")
    import torch

    torch.set_num_threads(1)
    p.root.mkdir()
    declaration = dict(
        schema="soccer.rsi.native_prospective_impact_declaration.v1",
        config=p.cfg,
        predecessors=p.predecessors,
        original_jobs=[c["job"] for c in p.cases],
        input_file_hashes=p.pins,
        independent_qualification_claimed=False,
        teacher_qualified=False,
        hardware_authorized=False,
        promotion_authorized=False,
    )
    declaration["report_hash"] = hash_json(declaration)
    write_once(p.root / "declaration.json", declaration)
    begun, rows = 0, []
    current = None
    try:
        print("ALLOCATING_FULL_FIXED_MEAN", flush=True)
        mean = load_json_artifact(Path(p.cfg["mean_model_path"]))
        if mean["model_hash"] != p.cfg["mean_model_hash"]:
            raise ValueError("same whole frozen mean required")
        factory = RecurrentSamplingEpisodeFactory(mean)
        del mean
        gc.collect()
        shared = p.root / ".shared-models"
        shared.mkdir()
        w = p.cases[0]["receipt"]["witnesses"]["proof_payload"]
        shutil.copyfile(w["path"], shared / Path(w["path"]).name)
        if digest(shared / Path(w["path"]).name) != w["sha256"]:
            raise ValueError("whole original physical proof payload copy differs")
        for index, (case, inputs) in enumerate(zip(p.cases, p.body_inputs, strict=True)):
            for enabled in (False, True):
                if time.time() >= p.cfg["deadline_unix"] or begun >= 12:
                    raise RuntimeError("fixed native body/deadline budget exhausted")
                c = inputs["source_commitment"]
                folder = p.root / f"case-{index}-impact-{int(enabled)}"
                current = dict(index=index, enabled=enabled, folder=str(folder))
                write_once(
                    p.root / f"intent-{begun:02d}.json",
                    dict(
                        current,
                        source_receipt_hash=case["receipt"]["receipt_hash"],
                        source_job=case["job"],
                        maximum_forecast_steps=900 if enabled else 0,
                        hardware_authorized=False,
                        promotion_authorized=False,
                    ),
                )
                native_args = argparse.Namespace(
                    scene=p.scene,
                    model_root=p.model_root,
                    late_swing_policy=p.late,
                    output_root=folder,
                    neural_model=None,
                    motor_policy=None,
                    step_model=inputs["view"],
                    body_response_bundle=None,
                    cached_proposal_envelope=False,
                    foundation_only=False,
                    compressed_report=True,
                    record_foundation_observation=True,
                    shared_evidence=True,
                    proposal_decoder="reference",
                    root_velocity_reference="body-com",
                    observation_snapshot="cached",
                    seed=c["seed"],
                    lane=c["lane"],
                )
                begun += 1
                print("NATIVE_BODY_STARTED", index, enabled, flush=True)
                run_native_first_touch_episode(
                    native_args,
                    course=tuple(c["course"]),
                    partition="DECLARED_DEVELOPMENT_DIAGNOSTIC",
                    entry_source_path=Path(__file__),
                    consumed_bank_hash=None,
                    recurrent_sampling_factory=factory,
                    control_profile=BoundedFirstTouchControl(
                        **p.declaration["candidate"]["parameters"]
                    ),
                    approach_mode="bounded_approach_diagnostic",
                    swing_mode="bounded_contact_diagnostic",
                    record_default_planner_events=True,
                    mesh_task_contact_world_hash=p.world_hash,
                    mesh_world_review=p.review,
                    generated_seed_episode_binding=True,
                    original_ik_planner_pulse=OriginalIkPlannerPulse(0.08),
                    foot_heading_control=FootHeadingContactProbe(case["job"]["heading_enabled"]),
                    use_impact_feedback_teacher=enabled,
                )
                with np.load(folder / "physical_trace.npz", allow_pickle=False) as archive:
                    trace = dict(archive)
                envelope = load_json_artifact(folder / "report.json.gz")
                report = envelope["stripped_document"]
                if (
                    report["motor_policy_hash"] != c["motor_policy_hash"]
                    or report["late_actor_hash"] != c["late_actor_hash"]
                    or report["foundation_hash"] != c["foundation_hash"]
                    or report["compiled_model_hash"] != p.world_hash
                    or report["physical_trace_hash"] != digest(folder / "physical_trace.npz")
                ):
                    raise ValueError("same policy/Foundation/world and raw trace required")
                if not enabled:
                    intervention = None
                else:
                    active = np.flatnonzero(
                        np.any(trace["impact_feedback_requested_offset_nm"][:, 0] != 0, axis=1)
                    )
                    intervention = int(active[0]) if len(active) else None
                require_prefix(trace, case["trace"], intervention)
                result = dict(
                    current,
                    source_receipt_hash=case["receipt"]["receipt_hash"],
                    source_job=case["job"],
                    native_outcome=label_first_touch(trace).to_dict(),
                    original_outcome=case["receipt"]["outcome"],
                    whole_original_off_or_preintervention_prefix_matched=True,
                    native_substeps=3000,
                    forecast_counters=report.get("impact_feedback_counters"),
                    trace=dict(
                        path=str(folder / "physical_trace.npz"),
                        sha256=digest(folder / "physical_trace.npz"),
                    ),
                    report_file=dict(
                        path=str(folder / "report.json.gz"),
                        sha256=digest(folder / "report.json.gz"),
                    ),
                    independent_physics_qualified=False,
                    teacher_qualified=False,
                    hardware_authorized=False,
                    promotion_authorized=False,
                )
                result["report_hash"] = hash_json(result)
                write_once(p.root / f"case-{len(rows):02d}-complete.json", result)
                rows.append(result)
                print("NATIVE_BODY_COMPLETE", index, enabled, result["native_outcome"], flush=True)
                del report, envelope, trace
                gc.collect()
        if any(digest(n) != h for n, h in p.pins.items()):
            raise ValueError("whole original source/model/data changed during native pilot")
        total_forecast = sum(
            r["forecast_counters"]["forecast_integrated_steps"] for r in rows if r["enabled"]
        )
        if len(rows) != 12 or total_forecast > 5400:
            raise ValueError("complete bounded paired native pilot required")
        result = dict(
            schema="soccer.rsi.native_prospective_impact_complete.v1",
            declaration_hash=declaration["report_hash"],
            rows=rows,
            native_integrated_steps=36000,
            forecast_integrated_steps=total_forecast,
            independent_physics_qualified=False,
            new_optimizer_updates=0,
            teacher_qualified=False,
            hardware_authorized=False,
            promotion_authorized=False,
        )
        result["report_hash"] = hash_json(result)
        write_once(p.root / "complete.json", result)
    except BaseException as error:
        failure = dict(
            schema="soccer.rsi.native_prospective_impact_failure.v1",
            current=current,
            native_bodies_begun=begun,
            native_bodies_completed=len(rows),
            charged_native_step_upper_bound=begun * 3000,
            error_type=type(error).__name__,
            error=str(error),
            automatic_retry_authorized=False,
            hardware_authorized=False,
            promotion_authorized=False,
        )
        failure["report_hash"] = hash_json(failure)
        write_once(p.root / "failure.json", failure)
        raise


if __name__ == "__main__":
    main()
