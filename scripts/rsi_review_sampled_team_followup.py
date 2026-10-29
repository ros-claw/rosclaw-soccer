"""Measure actual teammate contacts after replayed receiver exploration."""

import argparse
from pathlib import Path

import mujoco

from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean
from scripts.rsi_review_receiver_generation_pair import team_followup
from scripts.rsi_review_relay_balance_counterfactual import report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--audit-hash", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    clean(a.expected_head)
    if a.output.exists() or a.output.resolve().is_relative_to(Path.cwd()):
        raise ValueError("unused external output required")
    source, _, source_hash = report(
        a.source,
        a.audit,
        a.audit_hash,
        "NEURAL_RELAY_CHILD_VELOCITY_CANDIDATE_SAMPLING32_PENDING_OWN_REPLAY",
        worlds=32,
    )
    scene = a.source / "scene.xml"
    scene_hash = digest(scene)
    model = mujoco.MjModel.from_xml_path(str(scene))
    layout = NativeVelocityLayout(
        model, robot_prefix="a/", ball_body_name="soccer_ball", player_count=2
    )
    rows = []
    for row in source["records"]:
        seed = row["course"]["arm"]["exploration_seed"]
        if seed is None:
            continue
        if seed not in (92117, 92143, 92171):
            raise ValueError("complete predeclared stochastic seeds required")
        rows.append(
            dict(
                index=row["index"],
                course=row["course"],
                controls=row["controls"],
                termination=row["termination"],
                result=row["result"],
                team=team_followup(a.source, row, row["controls"], model, layout),
                physics_hash=row["artifacts"]["physics"]["hash"],
            )
        )
    if len(rows) != 24 or digest(scene) != scene_hash:
        raise ValueError("all sampled trajectories and unchanged scene required")
    clean(a.expected_head)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    write_once(
        a.output,
        dict(
            status="REPLAYED_TEAM_CONTACT_EXPLORATION_NOT_POSSESSION",
            source_hash=source_hash,
            audit_hash=a.audit_hash,
            scene_hash=scene_hash,
            code_head=a.expected_head,
            rows=rows,
            team_pass_proven=False,
            physical_improvement_proven=False,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
