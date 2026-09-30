"""Export actual successful motor teachers and failed-outcome trajectories.

Outcome-labelled offline teacher selection is allowed only in consumed training
data. It is NOT a deployed selector or an assertion that one actor wins 41/52.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.rsi.contact_motor_primitive import JOINT_NAMES
from rosclaw_soccer.rsi.failure_curriculum_evidence import _outcome, _sealed
from rosclaw_soccer.rsi.online_motor_actor_critic import terminal_return
from rosclaw_soccer.rsi.step_motor_features import FEATURE_NAMES, feature_vector
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    bank = _sealed(args.bank)
    if (
        bank["schema"] != "soccer.rsi.progressive_motor_learning_bank.v312"
        or bank["partition"] != "TRAIN_CONSUMED"
        or len(bank["courses"]) != 52
        or bank["fresh_holdout_open_authorized"] is not False
    ):
        parser.error("complete sealed consumed physical course bank required")
    observations, actions, returns, actor_mask, group_ids = [], [], [], [], []
    trajectories = []
    for index, course in enumerate(bank["courses"]):
        predecessor = Path(course["predecessor_folder"])
        root = predecessor.parent
        commitment = json.loads((root / "commitment.json").read_text())
        alternatives = [predecessor]
        # Only the previously completed/consumed neural exam has champion labels.
        if root.name == "rsi-neural-fresh-exam-v310":
            alternatives.append(root / f"seed{course['seed']}-lane{course['lane']}-champion-actor")
        measured = []
        for folder in alternatives:
            report = _sealed(folder / "report.json")
            result = _outcome(folder, report["contact_motor_policy_hash"], commitment)
            measured.append((folder, result["report"], result["outcome"]))
        folder, report, outcome = max(
            measured,
            key=lambda entry: (
                entry[2]["high_quality"],
                entry[2]["clean_foot_only"],
                entry[2]["minimum_pelvis_z_m"] >= 0.65,
                entry[2]["reward"],
            ),
        )
        order = report["taskspace_joint_order"]
        dds_ids = [order.index(n) for n in G1_DDS_JOINT_NAMES]
        motor_dds = [list(G1_DDS_JOINT_NAMES).index(n) for n in JOINT_NAMES]
        with (
            np.load(folder / "body_trace.npz", allow_pickle=False) as body,
            np.load(folder / "contact_motor_trace.npz", allow_pickle=False) as motor,
            np.load(folder / "late_swing_action_trace.npz", allow_pickle=False) as swing,
            np.load(folder / "trace.npz", allow_pickle=False) as physics,
        ):
            nominal = swing["executed_taskspace_joint_target_rad"][:, :, dds_ids]
            delta = motor["applied_joint_delta_rad"]
            forces = physics["ball_body_contact_force_peak_n"]
            if (
                nominal.shape != (300, 1, 29)
                or delta.shape != (300, 1, 12)
                or forces.shape != (300, 1, 6)
                or not np.allclose(
                    nominal[:, :, motor_dds], motor["baseline_joint_target_rad"], atol=2e-5, rtol=0
                )
            ):
                raise ValueError("teacher nominal target or observed action contract changed")
            composed = nominal.copy()
            composed[:, :, motor_dds] += delta
            if not np.allclose(composed, body["joint_target_rad"], atol=2e-5, rtol=0):
                raise ValueError("teacher labels do not match physically executed body targets")
            for frame in range(300):
                observation = feature_vector(
                    joint_position=body["joint_position_rad"][frame, 0],
                    joint_velocity=body["joint_velocity_rad_s"][frame, 0],
                    root_pose_xyzw=body["root_pose_xyzw_m"][frame, 0],
                    root_velocity_world=body["root_velocity_world"][frame, 0],
                    ball_position_world=body["ball_position_before_step_m"][frame, 0],
                    ball_velocity_world=body["ball_linear_velocity_before_step_m_s"][frame, 0],
                    geometry_position_world=body["foot_geometry_position_before_step_m"][frame, 0],
                    nominal_target=nominal[frame, 0],
                    previous_motor_delta=delta[frame - 1, 0] if frame else np.zeros(12),
                    previous_contact_forces=forces[frame - 1, 0] if frame else np.zeros(6),
                    frame=frame,
                )
                observations.append(observation)
                actions.append(delta[frame, 0].copy())
                returns.append(terminal_return(outcome))
                actor_mask.append(outcome["high_quality"])
                group_ids.append(index)
        trajectories.append(
            dict(
                index=index,
                seed=course["seed"],
                lane=course["lane"],
                folder=str(folder),
                report_hash=report["report_hash"],
                outcome=outcome,
                policy_hash=report["contact_motor_policy_hash"],
            )
        )
    if len(observations) != 15600 or sum(r["outcome"]["high_quality"] for r in trajectories) != 41:
        raise ValueError("complete consumed teacher frontier required")
    args.output_root.mkdir(parents=True, exist_ok=False)
    data_path = args.output_root / "step_motor_teachers.npz"
    arrays: dict[str, Any] = dict(
        observation=np.asarray(observations),
        applied_delta_rad=np.asarray(actions),
        terminal_return_label=np.asarray(returns),
        actor_supervision_mask=np.asarray(actor_mask),
        trajectory_index=np.asarray(group_ids, dtype=np.int64),
    )
    np.savez_compressed(data_path, **arrays)
    manifest = dict(
        schema="soccer.rsi.causal_step_motor_teacher_bank.v1",
        partition="TRAIN_CONSUMED",
        source_hash=hash_bytes(Path(__file__).read_bytes()),
        source_bank_hash=bank["report_hash"],
        feature_names=list(FEATURE_NAMES),
        action_joint_names=list(JOINT_NAMES),
        trajectories=trajectories,
        observation_feature_count=134,
        sample_count=len(observations),
        distinct_physical_course_count=52,
        independent_seed_count=len({r["seed"] for r in trajectories}),
        successful_teacher_trajectories=41,
        successful_supervision_frame_count=sum(actor_mask),
        failed_outcome_trajectories=11,
        labels=(
            "actually executed bounded bilateral residual; "
            "terminal return is learning label only"
        ),
        data_hash=hash_bytes(data_path.read_bytes()),
        runtime_teacher_selection_authorized=False,
        promotion_authorized=False,
        hardware_authorized=False,
        fresh_holdout_open_authorized=False,
        online_rl_qualified=False,
    )
    manifest["report_hash"] = hash_json(manifest)
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False))
    print(
        json.dumps(
            dict(
                report_hash=manifest["report_hash"],
                samples=len(observations),
                contexts=52,
                successful_teachers=41,
                failed_trajectories=11,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
