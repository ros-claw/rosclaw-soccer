"""Qualify Dual256 shared-clock transport before teammate passing research.

First require bit-identical one-player observations, decisions and ALL native
substeps against the qualified episode+neural-standing implementation. Then
run two separated real neural bodies over ONE shared ball/world. The latter
is a smoke test, NOT a successful pass, team game or policy promotion.
"""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("frontier-root", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("frontier-head", "frontier-hash", "expected-head"):
        p.add_argument("--" + key, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import admit_frontier, clean, receipt, source_pins

    clean(a.expected_head)
    if a.output.exists():
        raise ValueError("unused shared-world qualification output required")
    pins, base_hash, candidates, contexts = admit_frontier(
        a.frontier_root, a.frontier_head, a.frontier_hash
    )
    if [v["seed"] for v in candidates] != [65023]:
        raise ValueError("this protocol declares exactly the historical-loss-free seed65023")
    indices = [1, 4, 20, 21]
    source_pins(pins)
    check_pins(pins)
    a.output.mkdir(parents=True)
    declaration = dict(
        source_head=a.expected_head,
        protocol="dual256.shared_clock_transport.v1",
        contexts=indices,
        candidate=candidates[0],
        baseline_training_hash=base_hash,
        second_robot_translation_m=[0, 8, 0],
        declared_standing_controls=100,
        shared_ball_count=1,
        physics_timestep_s=0.002,
        control_substeps=10,
        maximum_native_steps=48000,
        maximum_rollout_decisions=6400,
        test_is_not_team_game=True,
        hardware_authorized=False,
        promotion_authorized=False,
        input_file_hashes=pins,
    )
    write_once(a.output / "declaration.json", declaration)
    counts = dict(qualified_native=0, shared_native=0, rollout_decisions=0)
    records = []
    try:
        import mujoco
        import numpy as np
        import torch

        from rosclaw_soccer.rsi.dual_onpolicy_checkpoint import load_dual_candidate_for_sim_audit
        from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
        from rosclaw_soccer.rsi.paid_neural_stand import NeuralStandingTask
        from rosclaw_soccer.rsi.shared_native_motor_world import (
            NativePDParameters,
            SharedNativeMotorWorld,
        )
        from rosclaw_soccer.rsi.shared_native_neural_player import SharedNativeNeuralPlayer
        from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
        from scripts.rsi_stroke_search_common import make_episode, open_motor

        bench, baseline = open_motor(pins, base_hash, {i: contexts[i] for i in indices})
        c = candidates[0]
        _, tr = receipt(pins, c["training_hash"])
        actor, _ = load_dual_candidate_for_sim_audit(
            baseline.parent,
            parent_lineage_hash=tr["parent_lineage_hash"],
            root=Path(c["training_root"]),
            expected_receipt_hash=c["training_hash"],
            checkpoint_kind=c["kind"],
        )
        xmls = [Path(path) for path in pins if path.endswith("/cached-standard/declared-scene.xml")]
        xml = shared_native_scene(xmls[0].read_text(), prefixes=("a/", "b/"))
        # Compile all source and two-player declarations before ANY physics.
        model = mujoco.MjModel.from_xml_string(xml)
        source_episodes = [make_episode(bench, contexts[i]) for i in indices]
        for ep in source_episodes:
            if ep.native_steps:
                raise ValueError("zero-physics source preflight required")
        scene_path = a.output / "two-player-scene.xml"
        # Raw experimental asset, never a source-code edit or canonical world.
        scene_path.write_text(xml)

        def player(world, prefix, ep, goal):
            return SharedNativeNeuralPlayer(
                world,
                prefix,
                actor,
                clip=ep.clip,
                torso_index=ep.torso_index,
                build_observation=ep.build_observation,
                goal=goal,
                offset=ep.offset,
                scale=ep.scale,
                order=ep.order,
            )

        for index, ep in zip(indices, source_episodes, strict=True):
            param = NativePDParameters(ep.kp, ep.kd, ep.effort)
            data = mujoco.MjData(ep.env.model)
            mujoco.mj_copyData(data, ep.env.model, ep.env.data)
            world = SharedNativeMotorWorld(
                ep.env.model, data, players={"": param}, ball_body_name="soccer_ball"
            )
            own = player(world, "", ep, ep.goal)
            state = actor.initial_state()
            stand = None
            rows = []
            for step in range(ep.clip.length + 100):
                if step == ep.clip.length:
                    ep.admit_reference_recovery()
                    stand = NeuralStandingTask(
                        ep,
                        actor,
                        state=state,
                        memory_mode="carry",
                        count_inference=lambda: None,
                        hidden_width=256,
                    )
                    own.begin_standing()
                mujoco.mj_forward(world.model, world.data)
                target, trace = own.propose()
                counts["rollout_decisions"] += 1
                if stand is None:
                    obs = ep.observation()
                    with torch.no_grad():
                        mean, state = actor(torch.from_numpy(obs).reshape(1, 1, 160), state)
                    old = ep.step(mean[0, 0].numpy().copy())
                    old_obs, old_action = obs, mean[0, 0].numpy()
                else:
                    old = stand.step()
                    old_obs, old_action = old["NN_obs"], old["NN_action"]
                counts["qualified_native"] += 10
                counts["rollout_decisions"] += 1
                new = world.advance({"": target})
                counts["shared_native"] += 10
                own.commit()
                layout = world.layouts[""]
                qi = [
                    *range(layout.rq, layout.rq + 7),
                    *layout.qi,
                    *range(layout.bq, layout.bq + 7),
                ]
                vi = [
                    *range(layout.rv, layout.rv + 6),
                    *layout.vi,
                    *range(layout.bv, layout.bv + 6),
                ]
                for x, y in (
                    (trace["NN_obs"], old_obs),
                    (trace["NN_action"], old_action),
                    (new["qpos"][:, qi], old["qpos"]),
                    (new["qvel"][:, vi], old["qvel"]),
                    (new["torque"], old["torque"][:, np.argsort(ep.layout.ai)]),
                ):
                    if not np.array_equal(x, y):
                        raise ValueError(
                            f"one-body shared transport is NOT bit-identical at {index}:{step}"
                        )
                rows.append(new)
                if old["body_failed"]:
                    break
            np.savez_compressed(
                a.output / f"context{index}-single.npz",
                **{k: np.stack([r[k] for r in rows]) for k in ("qpos", "qvel", "torque")},
            )
            record = dict(
                context=index, one_player_controls=len(rows), one_player_bit_identical=True
            )
            # New coupled world; declare two actual body initial states once.
            data2 = mujoco.MjData(model)
            first = make_episode(bench, contexts[index])
            layouts = {
                prefix: NativeVelocityLayout(
                    model, robot_prefix=prefix, ball_body_name="soccer_ball", player_count=2
                )
                for prefix in ("a/", "b/")
            }
            q, v = first.layout.canonical(first.env.data)
            for prefix, layout in layouts.items():
                shift = np.array([0.0, 0.0, 0.0]) if prefix == "a/" else np.array([0.0, 8.0, 0.0])
                data2.qpos[layout.rq : layout.rq + 7] = q[:7]
                data2.qpos[layout.rq : layout.rq + 3] += shift
                data2.qpos[layout.qi] = q[7:36]
                data2.qvel[layout.rv : layout.rv + 6] = v[:6]
                data2.qvel[layout.vi] = v[6:35]
            layout = layouts["a/"]
            data2.qpos[layout.bq : layout.bq + 7] = q[36:]
            data2.qvel[layout.bv : layout.bv + 6] = v[35:]
            mujoco.mj_forward(model, data2)
            world2 = SharedNativeMotorWorld(
                model, data2, players={"a/": param, "b/": param}, ball_body_name="soccer_ball"
            )
            players = {
                prefix: player(
                    world2,
                    prefix,
                    first,
                    first.goal + (np.zeros(3) if prefix == "a/" else np.array([0.0, 8.0, 0.0])),
                )
                for prefix in world2.layouts
            }
            coupled_rows, traces = [], {prefix: [] for prefix in players}
            for step in range(first.clip.length + 100):
                if step == first.clip.length:
                    for pl in players.values():
                        pl.begin_standing()
                mujoco.mj_forward(model, data2)
                proposals = {prefix: pl.propose() for prefix, pl in players.items()}
                counts["rollout_decisions"] += len(players)
                row = world2.advance({prefix: value[0] for prefix, value in proposals.items()})
                counts["shared_native"] += 10
                for prefix, pl in players.items():
                    pl.commit()
                    traces[prefix].append(proposals[prefix][1])
                coupled_rows.append(row)
                if any(row["body_failed"].values()):
                    break
            np.savez_compressed(
                a.output / f"context{index}-coupled.npz",
                **{k: np.stack([r[k] for r in coupled_rows]) for k in ("qpos", "qvel", "torque")},
                **{
                    prefix[0] + "_" + k: np.stack([r[k] for r in rows_])
                    for prefix, rows_ in traces.items()
                    for k in rows_[0]
                },
            )
            record.update(
                coupled_controls=world2.control_steps,
                coupled_native_steps=world2.native_steps,
                coupled_body_failed=coupled_rows[-1]["body_failed"],
                coupled_neural_calls={p: pl.decisions for p, pl in players.items()},
                shared_ball_no_reposition=True,
                initializations_per_player=1,
            )
            records.append(record)
            write_once(a.output / f"context{index}.json", record)
            print(json.dumps(dict(record=record, counts=counts)), flush=True)
        check_pins(pins)
        clean(a.expected_head)
        write_once(
            a.output / "complete.json",
            dict(
                status="SHARED_CLOCK_ONE_BODY_EQUIVALENT_TWO_BODY_SMOKE_CLOSED_NOT_TEAM_GAME",
                source_head=a.expected_head,
                declaration_hash=digest(a.output / "declaration.json"),
                records=records,
                counts=counts,
                candidate=candidates[0],
                independent_coupled_replay_verified=False,
                team_game_verified=False,
                promotion_authorized=False,
                hardware_authorized=False,
                goal_complete=False,
                artifacts={f.name: digest(f) for f in a.output.glob("*.npz")},
                input_file_hashes=pins,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failed.json",
            dict(
                status="SHARED_NATIVE_QUALIFICATION_FAILED_NO_RETRY",
                error_type=type(error).__name__,
                error=str(error),
                counts=counts,
                completed_records=records,
                source_head=a.expected_head,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
