"""Review actual contact outcomes for four paired lateral approach commands."""

import argparse
import json
import subprocess
import time
from pathlib import Path

import mujoco
import numpy as np

from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
from rosclaw_soccer.rsi.paired_precontact_authority import paired_precontact_authority
from rosclaw_soccer.rsi.passer_curriculum_outcome import strict_pass_receive
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean
from scripts.rsi_review_relay_balance_counterfactual import bind, measured, report
from scripts.rsi_review_whole_receiver_identity import artifact_equal
from scripts.rsi_watch_passer_residual_pair import worker_ready


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("producer-head", "expected-head"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--wait-unit")
    p.add_argument("--timeout-sec", type=int, default=7200)
    a = p.parse_args()
    clean(a.expected_head)
    if (
        a.output.exists()
        or a.output.resolve().is_relative_to(Path.cwd())
        or not 1 <= a.timeout_sec <= 7200
        or a.wait_unit is not None
        and (
            not a.wait_unit.startswith("rosclaw-receiver-lateral-authority-")
            or not a.wait_unit.endswith(".service")
            or "/" in a.wait_unit
        )
    ):
        raise ValueError("bounded exact SIM job and unused external output required")
    if a.wait_unit:
        deadline = time.monotonic() + a.timeout_sec
        while True:
            q = subprocess.run(
                [
                    "systemctl",
                    "--user",
                    "show",
                    a.wait_unit,
                    "-p",
                    "LoadState",
                    "-p",
                    "ActiveState",
                    "-p",
                    "Result",
                    "-p",
                    "ExecMainStatus",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            state = dict(line.split("=", 1) for line in q.stdout.splitlines() if "=" in line)
            if worker_ready(a.source, state):
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("physical experiment observation expired; do not restart")
            time.sleep(15)
    root, audit = a.source / "source", a.source / "audit/complete.json"
    ah = digest(audit)
    source, _, sh = report(
        root, audit, ah, "NEURAL_RELAY_RECEIVER_LATERAL_AUTHORITY32_PENDING_OWN_REPLAY", worlds=32
    )
    declaration = json.loads((root / "declaration.json").read_text())
    if (
        source["source_head"] != a.producer_head
        or declaration.get("development_context_block") != 0
        or declaration.get("lateral_intercept_is_unlearned_teacher") is not True
    ):
        raise ValueError("exact isolated development-only command authority required")
    model = mujoco.MjModel.from_xml_path(str(root / "scene.xml"))
    layout = NativeVelocityLayout(
        model, robot_prefix="b/", ball_body_name="soccer_ball", player_count=2
    )
    indices = {
        (
            r["course"]["context"],
            r["course"]["lateral_m"],
            r["course"]["arm"]["receiver_lateral_offset_m"],
        ): r
        for r in source["records"]
    }
    offsets = (-0.35, -0.12, 0.12, 0.35)
    if len(indices) != 32:
        raise ValueError("complete paired four-arm source factorial required")
    rows = []
    for context, side in sorted({(c, s) for c, s, _ in indices}):
        baseline = indices[(context, side, 0.12)]
        base_velocity, base_physics = (
            bind(root, baseline, "receiver_velocity"),
            bind(root, baseline, "physics"),
        )
        if base_velocity is None or base_physics is None:
            raise ValueError("baseline actual actions and physics required")
        for offset in offsets:
            row = indices[(context, side, offset)]
            if not artifact_equal(root, row, root, baseline, "initial"):
                raise ValueError("all four commands require identical physical initial state")
            velocity, physical = bind(root, row, "receiver_velocity"), bind(root, row, "physics")
            if velocity is None or physical is None:
                raise ValueError("actual executed motor and physics history required")
            with (
                np.load(base_velocity, allow_pickle=False) as base_v,
                np.load(base_physics, allow_pickle=False) as base_p,
                np.load(velocity, allow_pickle=False) as v,
                np.load(physical, allow_pickle=False) as raw,
            ):
                outcome = strict_pass_receive(
                    raw["ball_forces"],
                    raw["body_failed"],
                    handoff=row["result"]["handoff_executed"],
                )
                n = min(
                    len(v["control"]), len(base_v["control"]), len(raw["qpos"]), len(base_p["qpos"])
                )
                if (
                    n < 1
                    or not np.array_equal(v["control"][:n], np.arange(n))
                    or not np.array_equal(base_v["control"][:n], np.arange(n))
                ):
                    raise ValueError("complete time-aligned own control histories required")
                command_difference = v["command"][:n] - base_v["command"][:n]
                target_difference = raw["target_b"][:n] - base_p["target_b"][:n]
                joint_difference = (
                    raw["qpos"][:n, -1, layout.qi] - base_p["qpos"][:n, -1, layout.qi]
                )
                ball_difference = (
                    raw["qpos"][:n, -1, layout.bq : layout.bq + 3]
                    - base_p["qpos"][:n, -1, layout.bq : layout.bq + 3]
                )
                if any(
                    not np.isfinite(x).all()
                    for x in (
                        command_difference,
                        target_difference,
                        joint_difference,
                        ball_difference,
                    )
                ):
                    raise ValueError("finite physical authority history required")
                effect = dict(
                    common_controls=n,
                    max_velocity_command_delta=float(np.abs(command_difference).max()),
                    max_joint_target_delta_rad=float(np.abs(target_difference).max()),
                    max_joint_position_delta_rad=float(np.abs(joint_difference).max()),
                    max_ball_position_delta_m=float(np.linalg.norm(ball_difference, axis=1).max()),
                )
                precontact = paired_precontact_authority(
                    v["command"],
                    base_v["command"],
                    raw["qpos"],
                    base_p["qpos"],
                    raw["ball_forces"],
                    joint_indices=layout.qi,
                    root_index=layout.rq,
                )
            rows.append(
                dict(
                    context=context,
                    side=side,
                    offset_m=offset,
                    outcome=outcome,
                    original=measured(root, row),
                    physical_effect_vs_baseline=effect,
                    precontact_authority=precontact,
                )
            )
    if len(rows) != 32 or digest(root / "complete.json") != sh or digest(audit) != ah:
        raise ValueError("unchanged complete paired rollout and independent replay required")
    summaries = {}
    for offset in offsets:
        subset = [r for r in rows if r["offset_m"] == offset]
        summaries[str(offset)] = dict(
            conditions=len(subset),
            strict_clean=sum(r["outcome"]["strict_both_clean_handoff"] for r in subset),
            foot_after_pass=sum(r["outcome"]["passer_foot_precedes_receiver"] for r in subset),
            receiver_nonfoot=sum(
                r["outcome"]["receiver_nonfoot_substep"] is not None for r in subset
            ),
            healthy_both=sum(r["outcome"]["both_bodies_healthy"] for r in subset),
            precise_healthy_return=sum(r["original"]["return_0p1m_and_healthy"] for r in subset),
            median_max_target_delta_rad=float(
                np.median(
                    [r["physical_effect_vs_baseline"]["max_joint_target_delta_rad"] for r in subset]
                )
            ),
        )
    clean(a.expected_head)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "complete.json",
        dict(
            status="RECEIVER_LATERAL_AUTHORITY_DEVELOPMENT_REVIEWED",
            source_head=a.expected_head,
            source_hash=sh,
            audit_hash=ah,
            conditions=8,
            summaries=summaries,
            records=rows,
            geometric_teacher_not_learned=True,
            closed_loop_trajectories_diverge=True,
            independent_fresh=False,
            promotion_authorized=False,
            hardware_authorized=False,
            goal_complete=False,
        ),
    )
    print(json.dumps(summaries))


if __name__ == "__main__":
    main()
