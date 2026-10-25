"""Authenticate and rescore88 saved trajectories; inspect nonfoot geometry, no training."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("audit", "rollouts", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("audit-hash", "rollouts-hash", "expected-head"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--shard-root", type=Path, nargs=4, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    rp = a.rollouts / "complete.json"
    if (
        a.output.exists()
        or a.audit.is_symlink()
        or rp.is_symlink()
        or (digest(a.audit) != a.audit_hash or digest(rp) != a.rollouts_hash)
    ):
        raise ValueError("new review output and exact completed physical sources required")
    audit, rollouts = (json.loads(p.read_text()) for p in (a.audit, rp))
    if audit["status"] != "ALL80_KICK_HEAD_PPO_CANDIDATE_TRIALS_NATIVE_VERIFIED" or (
        rollouts["status"]
        != "EIGHT_OWN_POLICY_KICK_TRAJECTORIES_COMMITTED_AND_RAW_LIKELIHOODS_REPLAYED"
        or audit["worlds"] != 80
        or rollouts["worlds"] != 8
        or audit["promotion_authorized"] is not False
        or rollouts["promotion_authorized"] is not False
    ):
        raise ValueError("complete native80 paired study and8 actual own-policy worlds required")
    pins = dict(audit["input_file_hashes"])
    for name, h in rollouts["input_file_hashes"].items():
        if name in pins and pins[name] != h:
            raise ValueError("conflicting physical lineage")
        pins[name] = h
    pins.update({str(a.audit): a.audit_hash, str(rp): a.rollouts_hash})
    sources = []
    for shard, root in enumerate(a.shard_root):
        cp = root / "complete.json"
        if cp.is_symlink() or pins.get(str(cp)) != digest(cp):
            raise ValueError("exact audited physical producer shard required")
        report = json.loads(cp.read_text())
        if [r["course_index"] for r in report["results"]] != list(range(shard, 40, 4)):
            raise ValueError("all40 paired source contexts required")
        for item in report["results"]:
            for arm in ("onnx", "torch"):
                path = root / f"course-{item['course_index']:02d}" / (arm + ".json")
                sources.append((path, item["rows"][arm], arm))
    sources.extend(
        (a.rollouts / f"course-{r['course_index']:02d}.json", r, "own") for r in rollouts["results"]
    )
    for path, row, _ in sources:
        if path.is_symlink():
            raise ValueError("ordinary physical record required")
        actual_hash = digest(path)
        if str(path) in pins and pins[str(path)] != actual_hash:
            raise ValueError("inherited physical record hash changed")
        pins[str(path)] = actual_hash
        artifact = path.parent / row["artifact"]["path"]
        if artifact.is_symlink() or artifact.parent.resolve() != path.parent.resolve():
            raise ValueError("confined native artifact required")
        if str(artifact) in pins and pins[str(artifact)] != row["artifact"]["hash"]:
            raise ValueError("conflicting inherited physical artifact hash")
        pins[str(artifact)] = row["artifact"]["hash"]
    source_pins(pins)
    check_pins(pins)
    import mujoco
    import numpy as np
    from rosclaw.simforge.tasks.g1_goalforge.concepts import G1_HARD_TORQUE_LIMITS

    from rosclaw_soccer.rsi.kick_contact_inspection import inspect_ball_geometry
    from rosclaw_soccer.rsi.kick_event_reward import event_reward_contract, kick_event_rewards
    from rosclaw_soccer.rsi.native_kick_reward import finite_returns
    from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
    from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
    from rosclaw_soccer.rsi.walk_kick_teacher_audit import relabel_arrays
    from rosclaw_soccer.sim.contracts import hash_json

    scenes = [Path(k) for k in pins if k.endswith("/cached-standard/declared-scene.xml")]
    if len(scenes) != 1:
        raise ValueError("unique bound actual scene required")
    model = mujoco.MjModel.from_xml_string(
        shared_native_scene(scenes[0].read_text(), prefixes=("a/",))
    )
    layout = NativeVelocityLayout(model, robot_prefix="a/", ball_body_name="soccer_ball")
    a.output.mkdir(parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            protocol="soccer.g1.causal_reward88_and_contact_geometry_review.v1",
            source_head=a.expected_head,
            input_file_hashes=pins,
            expected_trajectories=88,
            reward_contract=event_reward_contract(),
            source_exam_labels_must_be_unchanged=True,
            new_physical_steps=0,
            optimizer_updates=0,
            geometry_is_not_force_replay=True,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )
    results = []
    try:
        for i, (path, row, arm) in enumerate(sources):
            if hash_json(json.loads(path.read_text())) != hash_json(row):
                raise ValueError("physical record differs from its completed source")
            with np.load(path.parent / row["artifact"]["path"]) as saved:
                arrays = dict(saved)
            n = row["controls"]
            if (
                row["ball_qpos_address"] != layout.bq
                or row["robot_qpos_address"] != layout.rq
                or (
                    np.any(
                        abs(arrays["torque"][:, layout.ai])
                        > 0.85 * np.array(G1_HARD_TORQUE_LIMITS) + 1e-12
                    )
                )
            ):
                raise ValueError("unchanged actual body and physical effort limits required")
            label = relabel_arrays(
                arrays, controls=n, ball_address=layout.bq, root_address=layout.rq
            )
            if hash_json(label) != hash_json(row["outcome"]):
                raise ValueError("formal source task label changed")
            pose = arrays["qpos"][:, layout.rq : layout.rq + 7].reshape(n, 10, 7)
            physical = dict(
                ball_after=arrays["ball_position_after_step_m"],
                pelvis_native=pose,
                force_native=arrays["ball_forces"][:, 0].reshape(n, 10, 3),
            )
            reward = kick_event_rewards(**physical)
            for cut in (50, 100, 160, 240, 299, 300, 320):
                if cut < n:
                    prefix = kick_event_rewards(**{k: v[:cut] for k, v in physical.items()})
                    if not np.array_equal(prefix["rewards"], reward["rewards"][:cut]):
                        raise ValueError("actual reward prefix invariance failed")
            events = {
                k: v
                for k, v in reward.items()
                if k not in ("rewards", "components", "component_names")
            }
            witness = None
            if arm == "torch" and row["course_index"] in audit["safety_regression_indices"]:
                native = np.flatnonzero(arrays["ball_forces"][:3000, 0, 2] > 1)
                if not len(native):
                    raise ValueError("expected physical nonfoot witness absent")
                witness = inspect_ball_geometry(model, arrays["qpos"], native_index=int(native[0]))
                witness["recorded_aggregate_nonfoot_force_n"] = float(
                    arrays["ball_forces"][native[0], 0, 2]
                )
                if not any(not x["is_allowed_foot"] for x in witness["geometric_contacts"]):
                    raise ValueError(
                        "preintegration geometry did not witness the recorded nonfoot contact"
                    )
            target = a.output / f"reward-{i:03d}.npz"
            returns = finite_returns(reward["rewards"], gamma=1.0)
            np.savez_compressed(
                target, rewards=reward["rewards"], returns=returns, components=reward["components"]
            )
            result = dict(
                source_record=str(path),
                source_record_hash=digest(path),
                arm=arm,
                course_index=row["course_index"],
                actual_controls=n,
                original_exam_task_pass=label["task_pass"],
                formal_exam_label_unchanged=True,
                causal_total_return=float(reward["rewards"].sum(dtype=np.float64)),
                events=events,
                reward_event_frames=np.flatnonzero(reward["rewards"]).tolist(),
                nonfoot_geometry=witness,
                artifact=dict(path=target.name, hash=digest(target)),
            )
            write_once(a.output / f"record-{i:03d}.json", result)
            results.append(result)
        if len(results) != 88:
            raise ValueError("all88 real source trajectories must be retained")
        check_pins(pins)
        clean(a.expected_head)
        write_once(
            a.output / "complete.json",
            dict(
                status="ALL88_CAUSAL_EVENT_REWARDS_PREFIX_VERIFIED_AND_NONFOOT_GEOMETRY_INSPECTED",
                source_head=a.expected_head,
                input_file_hashes=pins,
                reward_contract=event_reward_contract(),
                results=results,
                trajectories=88,
                new_physical_steps=0,
                optimizer_updates=0,
                physical_football_gain_proven=False,
                promotion_authorized=False,
                hardware_authorized=False,
                goal_complete=False,
            ),
        )
    except BaseException as error:
        write_once(
            a.output / "failed.json",
            dict(
                error_type=type(error).__name__,
                error=str(error),
                completed_records=len(results),
                no_automatic_retry=True,
            ),
        )
        raise


if __name__ == "__main__":
    main()
