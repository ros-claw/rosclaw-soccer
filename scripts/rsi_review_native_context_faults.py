"""Reintegrate all three preserved native prefixes and reconstruct guard rejection.

Direct raw planner ONNX replay; no production Navigation/plan/proxy is called.
Recorded encoder/decoder actions are NOT mathematically re-inferred here.
"""

import argparse
import json
import math
import subprocess
from pathlib import Path


def literal_feed(np, context, q, command, seed):
    vx, vy, yaw_rate = command
    speed = math.hypot(vx, vy)
    w, x, y, z = q[3:7]
    facing = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    direction = (vx / speed, vy / speed) if speed > 1e-8 else (math.cos(facing), math.sin(facing))
    heading = facing + yaw_rate * 0.4
    return dict(
        context_mujoco_qpos=context[None].astype(np.float32),
        target_vel=np.array([speed], dtype=np.float32),
        mode=np.array([0 if speed < 0.02 else 1 if speed <= 0.8 else 2], dtype=np.int64),
        movement_direction=np.array([[*direction, 0.0]], dtype=np.float32),
        facing_direction=np.array([[math.cos(heading), math.sin(heading), 0.0]], dtype=np.float32),
        random_seed=np.array([seed], dtype=np.int64),
        has_specific_target=np.zeros((1, 1), dtype=np.int64),
        specific_target_positions=np.zeros((1, 4, 3), dtype=np.float32),
        specific_target_headings=np.zeros((1, 4), dtype=np.float32),
        allowed_pred_num_tokens=np.array([[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]], dtype=np.int64),
        height=np.array([-1.0], dtype=np.float32),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "inventory",
        "bank",
        "actor-source",
        "world",
        "model-root",
        "core-source",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("expected-head", "expected-actor-head", "expected-inventory-hash"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if head != args.expected_head:
        raise ValueError("sealed review source required")
    for name in (Path(__file__).name, "rsi_native_context_recovery_common.py"):
        if (
            subprocess.check_output(["git", "show", f"{head}:scripts/{name}"])
            != (Path.cwd() / "scripts" / name).read_bytes()
        ):
            raise ValueError("review bootstrap must equal its sealed Git blob before helper import")

    from scripts.rsi_native_context_recovery_common import (
        POLICY_REASONS,
        admit_inventory,
        check_pins,
        digest,
    )

    inventory, bank, pins, courses = admit_inventory(args)
    for path in (args.world, args.model_root / "planner_sonic.onnx"):
        if pins.get(str(path)) != digest(path):
            raise ValueError("same bound world and planner required")
    faults = [s for s in inventory["slots"] if s["status"] == "FAULT_REQUIRES_REVIEW"]
    if len(faults) != 3 or any(s["arm"] != "measured_blend" for s in faults):
        raise ValueError("all three original measured-context faults required")

    import mujoco
    import numpy as np
    import onnxruntime as ort

    from rosclaw_soccer.physics.g1_target_episode import G1TargetEpisode
    from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
    from rosclaw_soccer.providers.g1.sonic_measured_context import measured_joint_context
    from rosclaw_soccer.providers.g1.sonic_navigation import future_reference_context
    from rosclaw_soccer.providers.g1.sonic_reference_blend import blend_planner_segment
    from rosclaw_soccer.providers.g1.sonic_runup import (
        G1SonicRunupController,
        _resample_segments_30_to_50,
        _sonic_control_parameters,
    )
    from rosclaw_soccer.rsi.ball_entry_teacher import SkillEntryAffordance, ball_entry_decision
    from rosclaw_soccer.sim.contracts import hash_bytes
    from scripts.rsi_atomic_artifacts import write_once

    args.output.mkdir(parents=True, exist_ok=False)
    write_once(
        args.output / "declaration.json",
        dict(
            schema="soccer.rsi.native_context_fault_review.v1",
            input_file_hashes=pins,
            inventory_hash=digest(args.inventory),
            fault_slots=[s["slot"] for s in faults],
            expected_additional_native_steps=1600,
            expected_additional_raw_planner_calls=8,
            source_controller_modules_byte_exact_to_sealed_actor=True,
            encoder_decoder_replayed=False,
            production_navigation_plan_or_proxy_called=False,
            shared_native_context_resampling_and_blend_helpers=True,
            optimizer_steps=0,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )
    total = calls = 0
    body = None
    current_steps = 0
    try:
        model = mujoco.MjModel.from_binary_path(str(args.world))
        limits = model.jnt_range[[model.joint(n).id for n in G1_DDS_JOINT_NAMES]].copy()
        kp, kd, _ = _sonic_control_parameters(1.0, (1.0,) * 29)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = opts.inter_op_num_threads = 1
        session = ort.InferenceSession(
            str(args.model_root / "planner_sonic.onnx"),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        choices = [Path(p) for p in pins if Path(p).name == "entry-affordance.json"]
        if len(choices) != 1:
            raise ValueError("one bound historical affordance required")
        stored = json.loads(choices[0].read_text())
        fields = {
            k: stored[k]
            for k in (
                "ball_offset_xy_m",
                "historical_frame",
                "historical_forward_speed_mps",
                "historical_artifact_hash",
            )
        }
        fields["ball_offset_xy_m"] = tuple(fields["ball_offset_xy_m"])
        affordance = SkillEntryAffordance(**fields)
        if affordance.contract_hash != stored["contract_hash"]:
            raise ValueError("affordance identity differs")
        results = []
        for slot in faults:
            folder = Path(slot["source_artifact_dir"])
            failure = json.loads((folder / "failure.json").read_text())
            ticks = failure["completed_control_ticks"]
            if (
                type(ticks) is not int
                or not 20 <= ticks < 300
                or ticks % 20
                or failure["attempted_native_steps"] != ticks * 10
                or failure["integrated_native_steps"] != ticks * 10
                or failure["trace_persistence_error"] is not None
                or failure["attempted_source_planner_calls"] != ticks // 20
                or failure["returned_source_planner_calls"] != ticks // 20
            ):
                raise ValueError(
                    "complete finite source control prefixes and no pending model call required"
                )
            with np.load(folder / "observed-control-ticks.npz", allow_pickle=False) as a:
                raw = {k: a[k] for k in a.files}
            with np.load(folder / "planner-calls.npz", allow_pickle=False) as a:
                recorded = {k: a[k] for k in a.files}
            with np.load(folder / "last-native-state.npz", allow_pickle=False) as a:
                last = {k: a[k] for k in a.files}
            if (
                len(raw) != 18
                or any(len(a) != ticks or not np.isfinite(a).all() for a in raw.values())
                or any(not np.isfinite(a).all() for a in last.values())
            ):
                raise ValueError(
                    "all18 finite complete prefix fields and last native state required"
                )
            events = json.loads((folder / "planning-events.json").read_text())["events"]
            if [e["frame"] for e in events] != list(range(0, ticks, 20)):
                raise ValueError("all source planner events required")
            item = next(
                c
                for c in courses
                if c.parent_index == slot["cluster"] and c.variant_index == slot["variant"]
            )
            body = G1TargetEpisode(
                model,
                world_hash=digest(args.world),
                kp=kp,
                kd=kd,
                course=item.course,
                defaults=G1SonicRunupController.default_angles.copy(),
                record_all_ball_contacts=True,
            )
            current_steps = 0
            reference = None
            for frame in range(ticks):
                q, v = body.canonical()
                d = ball_entry_decision(q, v, frame=frame, affordance=affordance)
                if d.ready:
                    raise ValueError("these source faults must precede any fixed-pose handoff")
                np.testing.assert_array_equal(d.command, raw["entry_command"][frame])
                np.testing.assert_array_equal(
                    d.target_root_xy_m, raw["entry_target_root_xy_m"][frame]
                )
                if raw["entry_error_m"][frame] != d.horizontal_error_m or raw["entry_stage"][
                    frame
                ] != (2 if d.timed_out else 0):
                    raise ValueError("source nominal entry trace differs")
                if frame % 20 == 0:
                    index = frame // 20
                    context = (
                        np.repeat(q[:36][None], 4, axis=0)
                        if frame == 0
                        else future_reference_context(reference, frame + 10)
                    )
                    original = context.copy()
                    old_current = None if frame == 0 else reference[frame].copy()
                    delta = np.zeros(29)
                    if frame:
                        context, delta = measured_joint_context(context, q, old_current, limits)
                    feed = literal_feed(np, context, q, d.command, 30300 + index)
                    event = events[index]
                    for key, value in feed.items():
                        np.testing.assert_array_equal(value, recorded[f"call{index}__feed__{key}"])
                        np.testing.assert_array_equal(
                            value,
                            np.asarray(
                                event["planner_inputs"][key],
                                dtype=event["planner_input_dtypes"][key],
                            ),
                        )
                    calls += 1
                    prediction, count = session.run(None, feed)
                    np.testing.assert_array_equal(prediction, recorded[f"call{index}__raw_output"])
                    np.testing.assert_array_equal(count, recorded[f"call{index}__raw_count"])
                    np.testing.assert_array_equal(
                        prediction, np.asarray(event["planner_raw_output"], dtype=np.float32)
                    )
                    valid = np.asarray(prediction[0, : int(count[0])], dtype=np.float64)
                    if event["reference_hash"] != hash_bytes(valid.tobytes()):
                        raise ValueError("raw valid planner-segment identity differs")
                    segment = _resample_segments_30_to_50([valid], 0.02)
                    if frame == 0:
                        reference = np.concatenate(
                            (segment, np.repeat(segment[-1:], 410 - len(segment), axis=0))
                        )
                    else:
                        start = frame + 10
                        n = min(len(segment), 410 - start)
                        old = reference[start : start + n].copy()
                        generated = segment[:n].copy()
                        mixed = blend_planner_segment(old, generated)
                        expected = dict(
                            schema="soccer.sonic.measured_context_reference_stitch.v1",
                            start_frame=start,
                            measured_qpos=q.tolist(),
                            current_reference=old_current.tolist(),
                            original_context=original.tolist(),
                            used_context=context.tolist(),
                            joint_context_delta=delta.tolist(),
                            old_future=old.tolist(),
                            generated_future=generated.tolist(),
                            blended_future=mixed.tolist(),
                            physical_state_or_targets_modified=False,
                        )
                        if expected != event["reference_stitch"]:
                            raise ValueError("entire measured reference stitch differs")
                        reference[start : start + n] = mixed
                        reference[start + n :] = mixed[-1]
                np.testing.assert_array_equal(
                    reference[frame : frame + 10], raw["encoder_reference_poses"][frame]
                )
                physical = body.step(frame, raw["joint_target_rad"][frame].copy())
                current_steps = body.integrated_steps
                for key, value in physical.items():
                    np.testing.assert_array_equal(value[0], raw[key][frame])
            np.testing.assert_array_equal(body.data.qpos, last["qpos"])
            np.testing.assert_array_equal(body.data.qvel, last["qvel"])
            np.testing.assert_array_equal([body.data.time], last["time_sec"])
            q, v = body.canonical()
            if ball_entry_decision(q, v, frame=ticks, affordance=affordance).ready:
                raise ValueError("failed frame would enter fixed reference rather than replan")
            context = future_reference_context(reference, ticks + 10)
            try:
                measured_joint_context(context, q, reference[ticks], limits)
            except ValueError as error:
                reason = str(error)
                if reason not in POLICY_REASONS:
                    raise ValueError("not a recognized finite reference rejection") from error
            else:
                raise ValueError("reference rejection cannot be reproduced")
            gap = q[7:36] - reference[ticks, 7:36]
            delta = gap * min(1.0, 0.12 / max(float(abs(gap).max()), 1e-12))
            proposed = context[:, 7:] + delta
            violations = []
            for j, name in enumerate(G1_DDS_JOINT_NAMES):
                low = float((limits[j, 0] - proposed[:, j]).max())
                high = float((proposed[:, j] - limits[j, 1]).max())
                if max(low, high) > 0:
                    violations.append(
                        dict(
                            joint=name,
                            native_range_rad=limits[j].tolist(),
                            requested_delta_rad=float(delta[j]),
                            proposed_min_rad=float(proposed[:, j].min()),
                            proposed_max_rad=float(proposed[:, j].max()),
                            maximum_excess_rad=max(low, high),
                        )
                    )
            total += current_steps
            current_steps = 0
            result = dict(
                slot=slot["slot"],
                failed_control_frame=ticks,
                reason=reason,
                all_prefix_physics_bit_exact=True,
                all_recorded_planner_outputs_bit_exact=True,
                entire_reference_history_reconstructed=True,
                finite_policy_guard_rejection_reproduced=True,
                native_steps_replayed=body.integrated_steps,
                planner_calls_replayed=len(events),
                original_future_legal=bool(
                    np.all(context[:, 7:] >= limits[:, 0])
                    and np.all(context[:, 7:] <= limits[:, 1])
                ),
                proposed_violations=violations,
                original_case_retry_authorized=False,
                full_h300_quality_observed=False,
                promotion_authorized=False,
                hardware_authorized=False,
            )
            results.append(result)
            write_once(
                args.output / f"c{item.parent_index}v{item.variant_index}-complete.json", result
            )
            print("NATIVE_CONTEXT_FAULT_REPLAY_EXACT", slot["slot"], ticks, reason, flush=True)
            body = None
        if total != 1600 or calls != 8:
            raise ValueError("all three recorded prefixes and model calls required")
        check_pins(pins)
        write_once(
            args.output / "complete.json",
            dict(
                results=results,
                inventory_hash=digest(args.inventory),
                additional_recorded_native_steps=total,
                additional_real_planner_calls=calls,
                encoder_decoder_replayed=False,
                production_navigation_plan_or_proxy_called=False,
                new_h300_attempts=0,
                optimizer_steps=0,
                promotion_authorized=False,
                hardware_authorized=False,
            ),
        )
    except Exception as error:
        write_once(
            args.output / "failure.json",
            dict(
                error=repr(error),
                completed_native_steps=total,
                incomplete_native_steps=body.integrated_steps if body is not None else 0,
                attempted_planner_calls=calls,
                promotion_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
