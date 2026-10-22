"""Independent whole-world replay of the first coupled Dual256 transport test.

Does NOT use SharedNativeMotorWorld or SharedNativeNeuralPlayer. Reconstructs
each player's measured160 observation and own private256 history, computes all
native PD torques directly, and advances the shared world ONCE per2ms. Failed
body outcomes must replay too; an audit pass is not a football success.
"""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("source-root", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("source-head", "source-hash", "expected-head"):
        p.add_argument("--" + key, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, receipt, source_pins

    clean(a.expected_head)
    rp, dp = a.source_root / "complete.json", a.source_root / "declaration.json"
    if (
        a.output.exists()
        or rp.is_symlink()
        or dp.is_symlink()
        or (a.source_root / "failed.json").exists()
        or digest(rp) != a.source_hash
    ):
        raise ValueError("exact completed source and unused own-replay output required")
    source, declaration = [json.loads(path.read_text()) for path in (rp, dp)]
    if (
        source["status"] != "SHARED_CLOCK_ONE_BODY_EQUIVALENT_TWO_BODY_SMOKE_CLOSED_NOT_TEAM_GAME"
        or source["source_head"] != a.source_head
        or source["declaration_hash"] != digest(dp)
        or declaration["protocol"] != "dual256.shared_clock_transport.v1"
        or declaration["contexts"] != [1, 4, 20, 21]
        or declaration["candidate"]["seed"] != 65023
        or [r["context"] for r in source["records"]] != [1, 4, 20, 21]
    ):
        raise ValueError("complete declared four-context coupled transport source required")
    pins = dict(source["input_file_hashes"])
    pins[str(rp)], pins[str(dp)] = a.source_hash, digest(dp)
    for name, h in source["artifacts"].items():
        path = a.source_root / name
        if path.is_symlink() or path.parent != a.source_root or digest(path) != h:
            raise ValueError("all exact confined single and coupled artifacts required")
        pins[str(path)] = h
    source_pins(pins)
    check_pins(pins)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            producer_hash=a.source_hash,
            input_file_hashes=pins,
            use_shared_motor_clock=False,
            use_shared_player_driver=False,
            teacher_state_borrowed=False,
            source_actions_executed=False,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    counts = dict(native_steps=0, rollout_decisions=0)
    records = []
    try:
        from types import SimpleNamespace

        import mujoco
        import numpy as np
        import torch

        from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
        from rosclaw_soccer.rsi.dual_onpolicy_checkpoint import load_dual_candidate_for_sim_audit
        from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
        from scripts.rsi_stroke_search_common import make_episode, open_motor

        # Read original consumed TRAIN declaration through its baseline proof.
        _, bank = receipt(pins, declaration["candidate"]["training_hash"])
        _, trainbank = receipt(pins, bank["data_hash"])
        _, basebank = receipt(pins, trainbank["base_data_hash"])
        native_audits = []
        for path in pins:
            if path.endswith("/native-audit/complete.json"):
                report = json.loads(Path(path).read_text())
                if report.get("training_hash") == basebank["warm_start_hash"]:
                    native_audits.append(report)
        if len(native_audits) != 1:
            raise ValueError("one original W6 consumed-case witness required")
        npth, native = receipt(pins, native_audits[0]["source_hash"])
        old_dp = npth.parent / "declaration.json"
        if pins[str(old_dp)] != digest(old_dp) or native["declaration_hash"] != digest(old_dp):
            raise ValueError("original exact case data required")
        contexts = json.loads(old_dp.read_text())["contexts"]
        indices = declaration["contexts"]
        bench, baseline = open_motor(
            pins, declaration["baseline_training_hash"], {i: contexts[i] for i in indices}
        )
        c = declaration["candidate"]
        actor, _ = load_dual_candidate_for_sim_audit(
            baseline.parent,
            parent_lineage_hash=bank["parent_lineage_hash"],
            root=Path(c["training_root"]),
            expected_receipt_hash=c["training_hash"],
            checkpoint_kind=c["kind"],
        )
        xmls = [Path(path) for path in pins if path.endswith("/cached-standard/declared-scene.xml")]
        xml = shared_native_scene(xmls[0].read_text(), prefixes=("a/", "b/"))
        if (a.source_root / "two-player-scene.xml").read_text() != xml:
            raise ValueError("generated shared scene differs from pinned original asset")
        m = mujoco.MjModel.from_xml_string(xml)
        episodes = [make_episode(bench, contexts[i]) for i in indices]
        for index, ep, claim in zip(indices, episodes, source["records"], strict=True):
            with np.load(a.source_root / f"context{index}-coupled.npz", allow_pickle=False) as z:
                raw = {k: z[k].copy() for k in z.files}
            d = mujoco.MjData(m)
            ball = m.body("soccer_ball").id
            bj = m.body_jntadr[ball]
            bq, bv = int(m.jnt_qposadr[bj]), int(m.jnt_dofadr[bj])
            q, v = ep.layout.canonical(ep.env.data)
            d.qpos[bq : bq + 7], d.qvel[bv : bv + 6] = q[36:], v[35:]
            states, addresses, references, actions, goals = {}, {}, {}, {}, {}
            for prefix in ("a/", "b/"):
                body = m.body(prefix + "pelvis").id
                rj = m.body_jntadr[body]
                rq, rv = int(m.jnt_qposadr[rj]), int(m.jnt_dofadr[rj])
                joints = [m.joint(prefix + name).id for name in G1_DDS_JOINT_NAMES]
                qi, vi = m.jnt_qposadr[joints], m.jnt_dofadr[joints]
                ai = np.array([m.actuator(prefix + name).id for name in G1_DDS_JOINT_NAMES])
                shift = np.array([0.0, 0.0, 0.0]) if prefix == "a/" else np.array([0.0, 8.0, 0.0])
                d.qpos[rq : rq + 7] = q[:7]
                d.qpos[rq : rq + 3] += shift
                d.qpos[qi] = q[7:36]
                d.qvel[rv : rv + 6], d.qvel[vi] = v[:6], v[6:35]
                addresses[prefix] = dict(body=body, rq=rq, rv=rv, qi=qi, vi=vi, ai=ai)
                states[prefix] = actor.initial_state()
                references[prefix] = [
                    np.zeros(29, np.float32),
                    np.zeros(29, np.float32),
                    np.zeros(3, np.float32),
                ]
                actions[prefix] = np.zeros(29, np.float32)
                goals[prefix] = ep.goal + shift
            mujoco.mj_forward(m, d)
            n = len(raw["qpos"])
            if n != claim["coupled_controls"] or not 1 <= n <= ep.clip.length + 100:
                raise ValueError("exact bounded coupled trace required")
            failed = {prefix: False for prefix in states}
            for step in range(n):
                mujoco.mj_forward(m, d)
                targets = {}
                for prefix, adr in addresses.items():
                    if step == ep.clip.length:
                        references[prefix][1] = np.zeros(29, np.float32)
                        references[prefix][2] = np.zeros(3, np.float32)
                    view = SimpleNamespace(
                        pelvis_pos=d.xpos[adr["body"]].copy(),
                        pelvis_quat=d.xquat[adr["body"]].copy(),
                        joint_pos=d.qpos[adr["qi"]].copy(),
                        joint_vel=d.qvel[adr["vi"]].copy(),
                        base_ang_vel=d.qvel[adr["rv"] + 3 : adr["rv"] + 6].copy(),
                    )
                    obs = ep.build_observation(
                        view,
                        None,
                        0,
                        actions[prefix],
                        *references[prefix],
                        d.xpos[ball].copy(),
                        goals[prefix],
                        ep.offset,
                        obs_dim=None,
                    )
                    with torch.no_grad():
                        mean, after = actor(
                            torch.from_numpy(obs).reshape(1, 1, 160), states[prefix]
                        )
                    counts["rollout_decisions"] += 1
                    action = mean[0, 0].numpy().copy()
                    evidence = dict(
                        NN_obs=obs,
                        NN_action=action,
                        NN_h_before=states[prefix][0].numpy(),
                        NN_c_before=states[prefix][1].numpy(),
                        NN_h_after=after[0].numpy(),
                        NN_c_after=after[1].numpy(),
                    )
                    for key, value in evidence.items():
                        if not np.array_equal(value, raw[prefix[0] + "_" + key][step]):
                            raise ValueError(
                                f"own neural stream differs {index}:{prefix}:{step}:{key}"
                            )
                    states[prefix], actions[prefix] = after, action
                    targets[prefix] = ep.offset + action[ep.order] * ep.scale
                    if step < ep.clip.length:
                        references[prefix] = [
                            ep.clip.joint_pos[step].copy(),
                            ep.clip.joint_vel[step].copy(),
                            ep.clip.body_ang_vel_w[step, ep.torso_index].copy(),
                        ]
                for substep in range(10):
                    torque = np.zeros(m.nu)
                    for prefix, adr in addresses.items():
                        torque[adr["ai"]] = np.clip(
                            ep.kp * (targets[prefix] - d.qpos[adr["qi"]])
                            - ep.kd * d.qvel[adr["vi"]],
                            -ep.effort,
                            ep.effort,
                        )
                    d.ctrl[:] = torque
                    mujoco.mj_step(m, d)
                    counts["native_steps"] += 1
                    for key, value in dict(qpos=d.qpos, qvel=d.qvel, torque=torque).items():
                        if not np.array_equal(value, raw[key][step, substep]):
                            raise ValueError(
                                f"own native physics differs {index}:{step}:{substep}:{key}"
                            )
                    if not np.array_equal(d.actuator_force, torque) or any(
                        int(w.number) for w in d.warning
                    ):
                        raise ValueError("actual motor law or native warning differs")
                    for prefix, adr in addresses.items():
                        root = d.qpos[adr["rq"] : adr["rq"] + 7]
                        failed[prefix] |= bool(
                            root[2] < 0.65 or 1 - 2 * (root[4] ** 2 + root[5] ** 2) < 0.9
                        )
                if any(failed.values()) and step != n - 1:
                    raise ValueError("source continued after actual shared body failure")
            if (
                failed != claim["coupled_body_failed"]
                or (not any(failed.values()) and n != ep.clip.length + 100)
                or claim["coupled_native_steps"] != n * 10
                or claim["coupled_neural_calls"] != {prefix: n for prefix in states}
                or not np.isclose(d.time, n * 0.02, rtol=0, atol=1e-10)
            ):
                raise ValueError("source clock, failure or work outcome differs")
            row = dict(
                context=index,
                controls=n,
                native_steps=n * 10,
                body_failed=failed,
                own_history_verified=True,
                all_native_substeps_equal=True,
            )
            records.append(row)
            write_once(a.output / f"context{index}.json", row)
            print(json.dumps(dict(record=row, counts=counts)), flush=True)
        check_pins(pins)
        clean(a.expected_head)
        write_once(
            a.output / "complete.json",
            dict(
                status="COUPLED_DUAL256_ALL_OWN_HISTORY_NATIVE_REPLAY_PASS_NOT_TEAM_SUCCESS",
                source_head=a.expected_head,
                producer_hash=a.source_hash,
                records=records,
                counts=counts,
                input_file_hashes=pins,
                goal_complete=False,
                team_game_verified=False,
                hardware_authorized=False,
                promotion_authorized=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failed.json",
            dict(
                status="COUPLED_OWN_REPLAY_FAILED_PRESERVE_NO_RETRY",
                source_head=a.expected_head,
                error_type=type(error).__name__,
                error=str(error),
                counts=counts,
                completed_records=records,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
