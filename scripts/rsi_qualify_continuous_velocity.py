"""SIM explicit60s velocity profile and whole-native own replay.

Collect uses the shared native motor clock; audit uses an independent manual
PD/native clock. Both run their own frozen ONNX on own states, no saved action
execution. First300 outputs are checked against the UNMODIFIED H300 provider.
This is commanded locomotion, not learned tactics, passing or full football.
"""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("collect", "audit"), required=True)
    for key in ("frontier-root", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("frontier-head", "frontier-hash", "expected-head"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--source-root", type=Path)
    p.add_argument("--source-hash")
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import admit_frontier, clean, source_pins

    clean(a.expected_head)
    if a.output.exists():
        raise ValueError("unused continuous native evidence root required")
    pins, _, _, contexts = admit_frontier(a.frontier_root, a.frontier_head, a.frontier_hash)
    source = None
    if a.mode == "audit":
        if a.source_root is None or a.source_hash is None:
            raise ValueError("exact completed producer required for independent native replay")
        path = a.source_root / "complete.json"
        if (
            path.is_symlink()
            or (a.source_root / "failed.json").exists()
            or digest(path) != a.source_hash
        ):
            raise ValueError("exact completed long-horizon producer required")
        source = json.loads(path.read_text())
        if (
            source["status"] != "CONTINUOUS_VELOCITY60S_COLLECTED_NOT_TEAM_GAME"
            or [r["context"] for r in source["records"]] != [1, 4, 20, 21]
            or source["horizon_controls"] != 3000
            or source["hardware_authorized"] is not False
        ):
            raise ValueError("complete original four-context60s study required")
        pins[str(path)] = a.source_hash
        for record in source["records"]:
            path = a.source_root / record["artifact"]["path"]
            if (
                path.is_symlink()
                or path.parent != a.source_root
                or digest(path) != record["artifact"]["hash"]
            ):
                raise ValueError("exact confined long-horizon native artifacts required")
            pins[str(path)] = record["artifact"]["hash"]
    source_pins(pins)
    check_pins(pins)
    a.output.mkdir(parents=True)
    schedule = [
        [0, 300, [0.0, 0.0, 0.0]],
        [300, 800, [0.8, 0.0, 0.0]],
        [800, 1100, [0.4, 0.2, 0.8]],
        [1100, 1400, [0.0, 0.4, 0.0]],
        [1400, 1900, [1.0, 0.0, 0.0]],
        [1900, 2200, [0.4, -0.2, -0.8]],
        [2200, 2500, [-0.3, 0.0, 0.0]],
        [2500, 3000, [0.0, 0.0, 0.0]],
    ]
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            mode=a.mode,
            source_hash=a.source_hash,
            horizon_controls=3000,
            contexts=[1, 4, 20, 21],
            schedule=schedule,
            original_H300_unchanged=True,
            shared_clock_advance_used=a.mode == "collect",
            independent_manual_native_clock=a.mode == "audit",
            shared_codec_on_own_states=True,
            source_actions_executed=False,
            gait_reprime_inside_stream=False,
            ball_or_body_reset_inside_stream=False,
            optimizer_updates=0,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    counts = dict(native_steps=0, stream_calls=0, legacy_prefix_calls=0)
    records = []
    try:
        import mujoco
        import numpy as np

        from rosclaw_soccer.providers.g1.continuous_velocity_stream import ContinuousVelocityStream
        from rosclaw_soccer.providers.g1.velocity_policy import G1VelocityPolicy
        from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
        from rosclaw_soccer.rsi.paid_research_setup import open_paid_bench
        from rosclaw_soccer.rsi.relay_audit_clock import RelayAuditClock
        from rosclaw_soccer.rsi.shared_native_motor_world import (
            NativePDParameters,
            SharedNativeMotorWorld,
        )
        from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
        from scripts.rsi_stroke_search_common import make_episode

        indices = [1, 4, 20, 21]
        bench, _, _ = open_paid_bench(pins, [contexts[i]["case"] for i in indices])
        xmls = [Path(path) for path in pins if path.endswith("/cached-standard/declared-scene.xml")]
        xml = shared_native_scene(xmls[0].read_text(), prefixes=("a/",))
        m = mujoco.MjModel.from_xml_string(xml)
        layout = NativeVelocityLayout(
            m, robot_prefix="a/", ball_body_name="soccer_ball", player_count=1
        )
        models = [Path(path) for path in pins if path.endswith("/velocity/v0/exported/policy.onnx")]
        configs = [Path(path) for path in pins if path.endswith("/velocity/v0/params/deploy.yaml")]
        if len(models) != 1 or len(configs) != 1:
            raise ValueError("one exact qualified velocity model/config required")
        kwargs = dict(
            expected_model_hash=pins[str(models[0])], expected_config_hash=pins[str(configs[0])]
        )
        episodes = {i: make_episode(bench, contexts[i]) for i in indices}

        # Every initialization is validated before ANY trial advances physics.
        def initialize(i, params):
            ep = episodes[i]
            q, v = ep.layout.canonical(ep.env.data)
            d = mujoco.MjData(m)
            d.qpos[layout.rq : layout.rq + 7], d.qvel[layout.rv : layout.rv + 6] = q[:7], v[:6]
            d.qpos[layout.qi], d.qvel[layout.vi] = q[7:36], v[6:35]
            d.qpos[layout.bq : layout.bq + 7], d.qvel[layout.bv : layout.bv + 6] = q[36:], v[35:]
            mujoco.mj_forward(m, d)
            world = SharedNativeMotorWorld(
                m,
                d,
                players={
                    "a/": NativePDParameters(
                        np.array(params.stiffness), np.array(params.damping), ep.effort
                    )
                },
                ball_body_name="soccer_ball",
            )
            return world if a.mode == "collect" else RelayAuditClock(world)

        preflight = G1VelocityPolicy(models[0], configs[0], **kwargs)
        for i in indices:
            w = initialize(i, preflight.params)
            if w.native_steps:
                raise ValueError("whole four-course preflight must have zero steps")
        for i in indices:
            stream = ContinuousVelocityStream(
                models[0], configs[0], horizon_controls=3000, **kwargs
            )
            legacy = G1VelocityPolicy(models[0], configs[0], **kwargs)
            w = initialize(i, stream.params)
            initial_q, initial_v = w.data.qpos.copy(), w.data.qvel.copy()
            raw = None
            if source is not None:
                previous = next(r for r in source["records"] if r["context"] == i)
                with np.load(a.source_root / previous["artifact"]["path"], allow_pickle=False) as z:
                    raw = {k: z[k].copy() for k in z.files}
                if not np.array_equal(raw["initial_qpos"], initial_q) or not np.array_equal(
                    raw["initial_qvel"], initial_v
                ):
                    raise ValueError("own continuous initial state differs")
            rows = []
            for step in range(3000):
                mujoco.mj_forward(m, w.data)
                q, v = layout.canonical(w.data)
                angular = layout.angular_velocity_body(w.data)
                command = np.array(
                    next(cmd for start, end, cmd in schedule if start <= step < end), np.float32
                )
                target = stream.propose(q, v, angular, command, frame=step)
                counts["stream_calls"] += 1
                if step < 300:
                    old = legacy.propose(q, v, angular, command, frame=step)
                    counts["legacy_prefix_calls"] += 1
                    if (
                        not np.array_equal(old, target)
                        or not np.array_equal(legacy.last_observation, stream.last_observation)
                        or legacy.phase != stream.phase
                    ):
                        raise ValueError(
                            "first300 measured neural codec differs from unmodified legacy"
                        )
                elif step == 300:
                    try:
                        legacy.propose(q, v, angular, command, frame=step)
                    except ValueError as error:
                        if "H300" not in str(error):
                            raise
                    else:
                        raise ValueError("legacy actor unexpectedly accepted frame300")
                row = w.advance({"a/": target})
                counts["native_steps"] += 10
                row = {k: row[k] for k in ("qpos", "qvel", "torque", "ball_forces")} | dict(
                    body_failed=np.array(w.failed),
                    NN_obs=stream.last_observation.copy(),
                    NN_action=stream.last_action.copy(),
                    phase=np.array(stream.phase),
                    command=command,
                )
                if raw is not None:
                    for key, value in row.items():
                        if not np.array_equal(value, raw[key][step]):
                            raise ValueError(
                                f"whole own continuous stream differs:{i}:{step}:{key}"
                            )
                rows.append(row)
                if w.failed:
                    break
            record = dict(
                context=i,
                controls=w.control_steps,
                native_steps=w.native_steps,
                healthy_full60s=not w.failed and w.control_steps == 3000,
                body_failed=w.failed,
                prefix_frames_verified=min(300, w.control_steps),
                stream_neural_calls=stream.neural_calls,
                legacy_neural_calls=legacy.neural_calls,
            )
            if source is None:
                path = a.output / f"context{i}.npz"
                np.savez_compressed(
                    path,
                    initial_qpos=initial_q,
                    initial_qvel=initial_v,
                    **{k: np.stack([r[k] for r in rows]) for k in rows[0]},
                )
                record["artifact"] = dict(path=path.name, hash=digest(path))
            elif any(record[k] != previous[k] for k in record):
                raise ValueError("own complete continuous outcome differs")
            records.append(record)
            write_once(a.output / f"context{i}.json", record)
            print(json.dumps(dict(record=record, counts=counts)), flush=True)
        if source is not None and counts != source["counts"]:
            raise ValueError("whole continuous own count mismatch")
        check_pins(pins)
        clean(a.expected_head)
        write_once(
            a.output / "complete.json",
            dict(
                status="CONTINUOUS_VELOCITY60S_COLLECTED_NOT_TEAM_GAME"
                if source is None
                else "CONTINUOUS_VELOCITY60S_WHOLE_OWN_NATIVE_REPLAY_MATCH",
                source_head=a.expected_head,
                source_hash=a.source_hash,
                horizon_controls=3000,
                counts=counts,
                records=records,
                input_file_hashes=pins,
                optimizer_updates=0,
                whole_game_verified=False,
                goal_complete=False,
                promotion_authorized=False,
                hardware_authorized=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failed.json",
            dict(
                status="CONTINUOUS_NATIVE_FAILED_PRESERVE_NO_RETRY",
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
