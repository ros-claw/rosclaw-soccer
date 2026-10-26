"""Preserve two completed H300 arms, record a guard refusal, run only the missing arm."""

import argparse
import hashlib
import json
import re
import subprocess
from dataclasses import asdict, replace
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--partial", type=Path, required=True)
    p.add_argument("--partial-unit", required=True)
    p.add_argument("--partial-head", required=True)
    for name in ("declaration", "forward-json", "forward-npz", "backward-json", "backward-npz"):
        p.add_argument("--" + name + "-hash", required=True)
    p.add_argument("--pilot", type=Path, required=True)
    p.add_argument("--pilot-hash", required=True)
    p.add_argument("--review", type=Path, required=True)
    p.add_argument("--review-hash", required=True)
    p.add_argument("--external-source", type=Path, required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    if (
        a.output.exists()
        or re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.@-]{0,190}\.service", a.partial_unit) is None
        or a.partial.is_symlink()
        or (a.partial / "complete.json").exists()
        or any(
            (a.partial / x).exists()
            for x in ("left-before.json", "left.npz", "right-before.json", "right.npz")
        )
    ):
        raise ValueError("stopped incomplete original and unattempted right arm required")
    status = subprocess.check_output(
        [
            "systemctl",
            "--user",
            "show",
            a.partial_unit,
            "-p",
            "ActiveState",
            "-p",
            "Result",
            "-p",
            "ExecMainStatus",
            "-p",
            "MainPID",
        ],
        text=True,
    )
    if not all(
        s in status.splitlines()
        for s in ("ActiveState=failed", "Result=exit-code", "ExecMainStatus=1", "MainPID=0")
    ):
        raise ValueError("original unit must be confirmed terminated with its failed status")
    journal = subprocess.check_output(
        ["journalctl", "--user", "-u", a.partial_unit, "-n", "80", "--no-pager", "--output=cat"],
        text=True,
    )
    if "direct approach only qualified for aligned default-pose near-ball starts" not in journal:
        raise ValueError("original recorded reference guard rejection required")
    paths = {
        "declaration.json": a.declaration_hash,
        "forward.json": a.forward_json_hash,
        "forward.npz": a.forward_npz_hash,
        "backward.json": a.backward_json_hash,
        "backward.npz": a.backward_npz_hash,
    }
    for name, expected in paths.items():
        path = a.partial / name
        if path.is_symlink() or digest(path) != expected:
            raise ValueError("exact immutable original partial evidence required")
    pilot_path = a.pilot / "complete.json"
    if (
        pilot_path.is_symlink()
        or a.review.is_symlink()
        or digest(pilot_path) != a.pilot_hash
        or digest(a.review) != a.review_hash
    ):
        raise ValueError("exact prior pilot and independently reviewed baselines required")
    pilot, proof = json.loads(pilot_path.read_text()), json.loads(a.review.read_text())
    declaration = json.loads((a.partial / "declaration.json").read_text())
    if (
        pilot["status"] != "CFTRACK_RIGHT_REFERENCE_AIM_DEVELOPMENT_RECORDED"
        or pilot["local_planning_margin_accepted"]
        or pilot["review_hash"] != a.review_hash
        or proof["status"] != "CFTRACK_INTERCEPT_ALL8_NN_AND_MANUAL_NATIVE_EXACT"
        or declaration["source_head"] != a.partial_head
        or declaration["frozen_reference_goal_offset_y_m"] != -0.4
        or declaration["maximum_nn_calls"] != 1200
    ):
        raise ValueError(
            "original frozen posthoc proposal, budget and strict-margin abstention required"
        )
    probe = next(r for r in pilot["rows"] if r["name"] == "aim-minus-0.4")
    pins = dict(declaration["input_file_hashes"])
    pins[str(pilot_path)] = a.pilot_hash
    pins[str(a.review)] = a.review_hash
    for name, expected in paths.items():
        pins[str(a.partial / name)] = expected
    for name in ("forward", "backward"):
        row = json.loads((a.partial / (name + ".json")).read_text())
        if row["case"] != name or row["artifact"]["hash"] != paths[name + ".npz"]:
            raise ValueError("completed original arm and native artifact identity required")
    external = a.external_source.resolve()
    if str(external / "policy/omnicontact/OmniContact.py") not in pins:
        raise ValueError("qualified original policy source required")
    source_pins(pins)
    check_pins(pins)
    scenes = [Path(x) for x in pins if x.endswith("/cached-standard/declared-scene.xml")]
    if len(scenes) != 1:
        raise ValueError("unique canonical physical body required")
    from rosclaw_soccer.rsi.cftrack_torque_intent import CFtrackTorqueIntentPlan

    value = dict(probe["plan"])
    value["fraction_effort_hg"] = tuple(value["fraction_effort_hg"])
    frozen = CFtrackTorqueIntentPlan(**value)
    frozen.offsets()
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            original_unit=a.partial_unit,
            original_failure_status=status.splitlines(),
            original_journal_sha256="sha256:" + hashlib.sha256(journal.encode()).hexdigest(),
            inherited_complete_arms=["forward", "backward"],
            left_geometry_diagnostic_only=True,
            new_physical_arm="right",
            max_nn_calls=300,
            max_native_steps=3000,
            no_original_completed_arm_rerun=True,
            no_guard_relaxation=True,
            original_h300_and_force_caps_unchanged=True,
            neural_optimizer_updates=0,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    import numpy as np

    from rosclaw_soccer.rsi.cftrack_alignment_diagnostic import direct_approach_geometry
    from rosclaw_soccer.rsi.cftrack_torque_trial import complete_torque_trial, run_cold_torque_trial
    from rosclaw_soccer.rsi.cftrack_velocity_intercept import (
        condition_velocity_intercept,
        predict_initial_intercept,
    )
    from rosclaw_soccer.rsi.cold_cftrack_trial import cold_cftrack_trial
    from rosclaw_soccer.rsi.first_touch_outcome import label_first_touch, outcome_contract
    from rosclaw_soccer.rsi.omnicontact_policy_copy import cftrack_state_hash
    from rosclaw_soccer.sim.contracts import hash_json
    from scripts.rsi_cftrack_native_context import build_cold_cftrack

    template, _, _ = build_cold_cftrack(scenes[0].read_text(), external, direct_approach=True)
    before = cftrack_state_hash(template.policy)
    left_velocity = (0.0, 0.1, 0.0)
    left = cold_cftrack_trial(
        template,
        ball_xy=(1.0, 0.0),
        ball_velocity=left_velocity,
        ball_angular_velocity=(-0.1 / 0.11, 0.0, 0.0),
    )
    s = left.policy.state_cmd
    predicted, prediction = predict_initial_intercept(
        np.asarray(s.ball_pos),
        left.world.data.qvel[left.layout.bv : left.layout.bv + 3].copy(),
        np.asarray(left.policy.goal_pos),
    )
    planning_goal = np.asarray(left.policy.goal_pos, np.float64).copy()
    planning_goal[1] -= 0.4
    geometry = direct_approach_geometry(
        np.asarray(s.base_pos), np.asarray(s.base_quat), predicted, planning_goal
    )
    if (
        geometry["lateral_within_guard"]
        or not geometry["heading_within_guard"]
        or not geometry["forward_within_guard"]
    ):
        raise ValueError("original left rejection not explained by declared lateral guard")
    baseline_left = next(
        r
        for r in proof["records"]
        if r["case"] == "left" and r["arm"] == "intercept-plus-fixed-nominal-torque"
    )
    if not baseline_left["complete"]:
        raise ValueError("verified retained left baseline required")
    left_row = dict(
        case="left",
        origin="REFERENCE_GUARD_REJECTED_BEFORE_ACTUATION",
        prediction=prediction,
        alignment_geometry=geometry,
        outcome=dict(event_status="REFERENCE_ALIGNMENT_GUARD_REJECTED", task_pass=False, safe=None),
        verified_baseline_outcome=baseline_left["outcome"],
        initial_state_hash=hash_json(
            dict(qpos=left.world.data.qpos.tolist(), qvel=left.world.data.qvel.tolist())
        ),
        physical_steps=0,
        neural_calls=0,
    )
    write_once(a.output / "left-guard-rejection.json", left_row)
    velocity = (0.0, -0.1, 0.0)
    player = cold_cftrack_trial(
        template,
        ball_xy=(1.0, 0.0),
        ball_velocity=velocity,
        ball_angular_velocity=(0.1 / 0.11, 0.0, 0.0),
    )
    reference = condition_velocity_intercept(player, reference_goal_offset_y_m=-0.4)
    plan = replace(frozen, parent_policy_hash=cftrack_state_hash(player.policy))
    write_once(a.output / "right-before.json", dict(reference=reference, plan=asdict(plan)))
    raw, run = run_cold_torque_trial(player, plan, controls=300)
    path = a.output / "right.npz"
    np.savez_compressed(path, **raw)
    pins[str(path)] = digest(path)
    outcome = dict(event_status="INCOMPLETE_NATIVE_EPISODE", task_pass=False, safe=False)
    if complete_torque_trial(run, controls=300):
        forces = np.zeros((300, 1, 6))
        forces[:, 0, :3] = raw["ball_forces"][:, 0].reshape(300, 10, 3).max(axis=1)
        q = raw["qpos"].reshape(300, 10, 43)
        lo = player.layout
        outcome = label_first_touch(
            dict(
                force_n=forces,
                pelvis_z_per_substep_m=q[:, :, lo.rq + 2][:, None],
                ball_position_after_step_m=q[:, -1, lo.bq : lo.bq + 3][:, None],
            )
        ).to_dict()
    baseline_right = next(
        r
        for r in proof["records"]
        if r["case"] == "right" and r["arm"] == "intercept-plus-fixed-nominal-torque"
    )
    right_row = dict(
        case="right",
        origin="RECOVERY_UNATTEMPTED_ARM",
        reference=reference,
        plan=asdict(plan),
        run=run,
        outcome=outcome,
        verified_baseline_outcome=baseline_right["outcome"],
        initial_state_hash=hash_json(
            dict(qpos=raw["initial_qpos"].tolist(), qvel=raw["initial_qvel"].tolist())
        ),
        artifact=dict(path=path.name, hash=pins[str(path)]),
    )
    write_once(a.output / "right.json", right_row)
    old_path = a.pilot / probe["artifact"]["path"]
    if old_path.is_symlink() or pins.get(str(old_path)) != digest(old_path):
        raise ValueError("bound earlier nominated right probe required after execution")
    with np.load(old_path, allow_pickle=False) as saved:
        prefix = len(raw["joint_targets"]) >= 160 and all(
            np.array_equal(raw[k][: len(saved[k])], saved[k])
            for k in (
                "qpos",
                "qvel",
                "torque",
                "parent_torque",
                "ball_forces",
                "joint_targets",
                "nn_decoded_targets",
                "neural_obs",
                "neural_actions",
            )
        )
    records = []
    for name in ("forward", "backward"):
        row = json.loads((a.partial / (name + ".json")).read_text())
        row["origin"] = "VERIFIED_ORIGINAL_PARTIAL_ARTIFACT"
        records.append(row)
    records.extend([left_row, right_row])
    baseline_sources = [
        Path(path)
        for path, value in pins.items()
        if value == proof["source_hash"] and path.endswith("/complete.json")
    ]
    if len(baseline_sources) != 1 or digest(baseline_sources[0]) != proof["source_hash"]:
        raise ValueError("unique independently reviewed baseline source required")
    baseline_source = json.loads(baseline_sources[0].read_text())
    for row in records:
        old = next(
            r
            for r in baseline_source["records"]
            if r["case"] == row["case"] and r["arm"] == "intercept-plus-fixed-nominal-torque"
        )
        if row["initial_state_hash"] != old["initial_state_hash"]:
            raise ValueError("paired previous and candidate initial physical states differ")
    unchanged = cftrack_state_hash(template.policy) == before and template.world.native_steps == 0
    check_pins(pins)
    clean(a.expected_head)
    write_once(
        a.output / "complete.json",
        dict(
            status="CFTRACK_REFERENCE_AIM_RETENTION_RECOVERED_WITH_GUARD_REJECTION",
            source_head=a.expected_head,
            input_file_hashes=pins,
            original_partial_unit=a.partial_unit,
            pilot_hash=a.pilot_hash,
            review_hash=a.review_hash,
            outcome_contract_hash=outcome_contract()["contract_hash"],
            records=records,
            candidate_passes=sum(r["outcome"]["task_pass"] for r in records),
            verified_baseline_passes=sum(
                r["verified_baseline_outcome"]["task_pass"] for r in records
            ),
            rescues=[
                r["case"]
                for r in records
                if not r["verified_baseline_outcome"]["task_pass"] and r["outcome"]["task_pass"]
            ],
            regressions=[
                r["case"]
                for r in records
                if r["verified_baseline_outcome"]["task_pass"] and not r["outcome"]["task_pass"]
            ],
            guard_rejected_cases=["left"],
            nominated_right_probe_prefix_exact=prefix,
            cold_template_unchanged=unchanged,
            inherited_neural_calls=600,
            inherited_native_steps=6000,
            new_neural_calls=run["neural_calls"],
            new_native_steps=run["native_steps"],
            neural_optimizer_updates=0,
            posthoc_development_nomination=True,
            original_stricter_planning_margin_not_relaxed=True,
            independent_candidate_review_pending=True,
            fresh_not_evaluated=True,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
