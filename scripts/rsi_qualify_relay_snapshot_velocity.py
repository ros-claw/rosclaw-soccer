"""Recompute own receiver neural inference from qualified SIM state snapshots."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from rosclaw_soccer.providers.g1.velocity_policy import G1VelocityPolicy, observation98, target29
from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean
from scripts.rsi_review_relay_balance_counterfactual import bind, report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("source", "snapshots", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("source-hash", "audit-hash", "snapshot-hash", "expected-head"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--nondefault", action="store_true")
    a = p.parse_args()
    clean(a.expected_head)
    root, audit, receipt = (
        a.source / "source",
        a.source / "audit/complete.json",
        a.snapshots / "complete.json",
    )
    if (
        a.output.exists()
        or a.output.resolve().is_relative_to(Path.cwd())
        or receipt.is_symlink()
        or digest(receipt) != a.snapshot_hash
    ):
        raise ValueError("exact complete snapshots and unused external output required")
    source, _, sh = report(
        root,
        audit,
        a.audit_hash,
        "NEURAL_RELAY_RECEIVER_LATERAL_AUTHORITY32_PENDING_OWN_REPLAY",
        worlds=32,
    )
    snapshots = json.loads(receipt.read_text())
    if (
        sh != a.source_hash
        or snapshots.get("status")
        != (
            "RELAY_NONDEFAULT_PRECONTACT_NATIVE_STATE_RECONSTRUCTION_NOT_POLICY"
            if a.nondefault
            else "RELAY_PRECONTACT_NATIVE_STATE_RECONSTRUCTION_NOT_POLICY"
        )
        or snapshots.get("source_hash") != sh
        or snapshots.get("audit_hash") != a.audit_hash
        or len(snapshots.get("rows", [])) != (24 if a.nondefault else 8)
        or snapshots.get("saved_torque_is_not_policy") is not True
        or any(
            snapshots.get(k) is not False for k in ("promotion_authorized", "hardware_authorized")
        )
    ):
        raise ValueError("complete exact unpromoted qualified snapshot bank required")
    pins = source["input_file_hashes"]
    models = [Path(x) for x in pins if x.endswith("/velocity/v0/exported/policy.onnx")]
    configs = [Path(x) for x in pins if x.endswith("/velocity/v0/params/deploy.yaml")]
    scene = root / "scene.xml"
    if len(models) != 1 or len(configs) != 1 or digest(scene) != snapshots["scene_hash"]:
        raise ValueError("exact model, codec and scene required")
    policy = G1VelocityPolicy(
        models[0],
        configs[0],
        expected_model_hash=pins[str(models[0])],
        expected_config_hash=pins[str(configs[0])],
    )
    model = mujoco.MjModel.from_xml_path(str(scene))
    layout = NativeVelocityLayout(
        model, robot_prefix="b/", ball_body_name="soccer_ball", player_count=2
    )
    by_index = {r["index"]: r for r in source["records"]}
    rows = []
    for item in snapshots["rows"]:
        row = by_index[item["index"]]
        if (
            row["course"]["arm"]["receiver_lateral_offset_m"]
            not in ((-0.35, -0.12, 0.35) if a.nondefault else (0.12,))
            or (
                a.nondefault
                and item.get("receiver_lateral_offset_m")
                != row["course"]["arm"]["receiver_lateral_offset_m"]
            )
            or item["replayed_steps"] != 1400
            or item["restored_validation_steps"] != 10
        ):
            raise ValueError("exact original baseline and validated boundary required")
        snapshot = a.snapshots / item["snapshot"]
        if (
            snapshot.is_symlink()
            or snapshot.resolve().parent != a.snapshots.resolve()
            or digest(snapshot) != item["snapshot_hash"]
        ):
            raise ValueError("exact confined own snapshot required")
        velocity, physical = bind(root, row, "receiver_velocity"), bind(root, row, "physics")
        if velocity is None or physical is None:
            raise ValueError("bound actual velocity and physical history required")
        data = mujoco.MjData(model)
        with np.load(snapshot, allow_pickle=False) as state:
            mujoco.mj_setState(model, data, state["state"], mujoco.mjtState.mjSTATE_INTEGRATION)
        mujoco.mj_forward(model, data)
        with (
            np.load(velocity, allow_pickle=False) as v,
            np.load(physical, allow_pickle=False) as physics,
        ):
            if not np.array_equal(data.qpos, physics["qpos"][139, -1]) or not np.array_equal(
                data.qvel, physics["qvel"][139, -1]
            ):
                raise ValueError("restored state must match actual preceding substep")
            q, dv = layout.canonical(data)
            obs, phase = observation98(
                q,
                dv,
                layout.angular_velocity_body(data),
                v["command"][140],
                v["action"][139],
                v["phase"][139],
                policy.params,
            )
            action = policy.session.run(["actions"], {"obs": obs[None]})[0][0]
            if (
                not np.array_equal(obs, v["obs"][140])
                or not np.array_equal(action, v["action"][140])
                or phase != v["phase"][140]
                or not np.array_equal(target29(action, policy.params), physics["target_b"][140])
            ):
                raise ValueError("restored own neural observation, action, phase or target differs")
        if (
            digest(snapshot) != item["snapshot_hash"]
            or digest(velocity) != row["artifacts"]["receiver_velocity"]["hash"]
            or digest(physical) != row["artifacts"]["physics"]["hash"]
        ):
            raise ValueError("source changed during independent inference")
        rows.append(
            dict(
                index=item["index"],
                context=item["context"],
                side=item["side"],
                receiver_lateral_offset_m=row["course"]["arm"]["receiver_lateral_offset_m"],
                snapshot_hash=item["snapshot_hash"],
                observation_exact=True,
                action_exact=True,
                phase_exact=True,
                target_exact=True,
            )
        )
    if (
        len(rows) != (24 if a.nondefault else 8)
        or len({(r["context"], r["side"], r["receiver_lateral_offset_m"]) for r in rows})
        != (24 if a.nondefault else 8)
        or digest(receipt) != a.snapshot_hash
        or digest(root / "complete.json") != sh
        or digest(audit) != a.audit_hash
        or digest(scene) != snapshots["scene_hash"]
        or digest(models[0]) != pins[str(models[0])]
        or digest(configs[0]) != pins[str(configs[0])]
    ):
        raise ValueError("complete unique unchanged snapshot bank required")
    clean(a.expected_head)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "complete.json",
        dict(
            status=(
                "RECEIVER_NONDEFAULT_SNAPSHOT_OWN_VELOCITY_INFERENCE_MATCH_NOT_POLICY_RESUME"
                if a.nondefault
                else "RECEIVER_SNAPSHOT_OWN_VELOCITY_INFERENCE_MATCH_NOT_POLICY_RESUME"
            ),
            source_head=a.expected_head,
            source_hash=sh,
            audit_hash=a.audit_hash,
            snapshot_hash=a.snapshot_hash,
            rows=rows,
            source_command_replayed_for_validation=True,
            future_commands_used_by_planner=False,
            role_selector_recomputed=False,
            passer_neural_resume_verified=False,
            native_steps=0,
            optimizer_updates=0,
            independent_fresh=False,
            promotion_authorized=False,
            hardware_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
