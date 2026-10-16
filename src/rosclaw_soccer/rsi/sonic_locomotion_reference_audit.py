"""Independent replay of opt-in SONIC planner segments, not physics admission.

Reconstructs reference history without the producer navigation backend. This
does not independently prove the tracker's encoder consumed that reference;
actual CPU physics and Foundation qualification remain separate requirements.
"""

import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def _inputs(
    trace: dict[str, Any],
    events: Any,
    mode: int,
    reference_height_probe_m: float | None = None,
    fast_intercept_probe: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    if type(mode) is not int or mode not in (2, 3):
        raise ValueError("explicit walk/run planner mode required")
    if type(fast_intercept_probe) is not bool or (
        fast_intercept_probe and (mode != 3 or reference_height_probe_m != 0.75)
    ):
        raise ValueError("explicit pure run/height fast-intercept reference probe required")
    if reference_height_probe_m is not None and (
        type(reference_height_probe_m) is not float
        or not math.isfinite(reference_height_probe_m)
        or not 0.65 <= reference_height_probe_m <= 0.8
        or mode != 3
    ):
        raise ValueError("bounded explicit run reference-height intervention required")
    arrays = []
    for key, shape in (("canonical_qpos", (300, 1, 43)), ("navigation_command", (300, 1, 3))):
        a = np.asarray(trace.get(key))
        if (
            a.shape != shape
            or a.dtype != np.float64
            or not np.isfinite(a).all()
            or np.max(np.abs(a)) > 1e6
        ):
            raise ValueError("complete finite planner audit observations required")
        arrays.append(a[:, 0])
    qpos, command = arrays
    if (
        not np.allclose(np.linalg.norm(qpos[:, 3:7], axis=1), 1, atol=1e-6, rtol=0)
        or np.any(np.linalg.norm(command[:, :2], axis=1) > (2.1 if fast_intercept_probe else 1.5))
        or np.any(command[:, 2] != 0)
    ):
        raise ValueError("unit body poses and original translational envelope required")
    if type(events) is not list or len(events) != 15:
        raise ValueError("all fifteen fixed-cadence planner events required")
    keys = {"frame", "command", "reference_hash", "planner_mode"}
    if reference_height_probe_m is not None:
        keys.add("reference_height_input_m")
    for event, frame in zip(events, range(0, 300, 20), strict=True):
        if (
            type(event) is not dict
            or set(event) != keys
            or type(event["frame"]) is not int
            or event["frame"] != frame
            or type(event["planner_mode"]) is not int
            or not np.array_equal(np.asarray(event["command"]), command[frame])
        ):
            raise ValueError("timed measured planner event binding required")
        if reference_height_probe_m is not None and (
            type(event["reference_height_input_m"]) is not float
            or event["reference_height_input_m"]
            != (reference_height_probe_m if np.linalg.norm(command[frame, :2]) > 0.8 else -1.0)
        ):
            raise ValueError("reference-height input event differs from declared intervention")
    return qpos, command


def replay_locomotion_segments(
    session: Any,
    trace: dict[str, Any],
    events: list[dict[str, Any]],
    mode: int,
    *,
    reference_observer: Callable[[int, np.ndarray], None] | None = None,
    reference_height_probe_m: float | None = None,
    fast_intercept_probe: bool = False,
) -> list[dict[str, Any]]:
    """Pure segment replay over a supplied session; caller authenticates assets."""
    qpos, commands = _inputs(trace, events, mode, reference_height_probe_m, fast_intercept_probe)
    if reference_observer is not None and not callable(reference_observer):
        raise ValueError("optional reference observer must be callable")
    reference = None
    results = []
    for index, event in enumerate(events):
        frame = event["frame"]
        if reference is None:
            context = np.repeat(qpos[0, :36][None], 4, axis=0)
        else:
            samples = []
            for ordinal in range(4):
                location = frame + 10 + ordinal * 50.0 / 30.0
                left = int(location)
                fraction = location - left
                a, b = reference[left].copy(), reference[left + 1].copy()
                if float(a[3:7] @ b[3:7]) < 0:
                    b[3:7] *= -1
                value = (1 - fraction) * a + fraction * b
                value[3:7] /= np.linalg.norm(value[3:7])
                samples.append(value)
            context = np.asarray(samples)
        vx, vy, wz = (float(v) for v in commands[frame])
        speed = math.hypot(vx, vy)
        w, x, y, z = (float(v) for v in qpos[frame, 3:7])
        facing = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        heading = facing + wz * 0.4
        direction = (
            (vx / speed, vy / speed) if speed > 1e-8 else (math.cos(facing), math.sin(facing))
        )
        expected_mode = 0 if speed < 0.02 else 1 if speed <= 0.8 else mode
        if event["planner_mode"] != expected_mode:
            raise ValueError("recorded planner mode differs from fixed intervention")
        feed = dict(
            context_mujoco_qpos=context[None].astype(np.float32),
            target_vel=np.asarray([speed], dtype=np.float32),
            mode=np.asarray([expected_mode], dtype=np.int64),
            movement_direction=np.asarray([[*direction, 0.0]], dtype=np.float32),
            facing_direction=np.asarray(
                [[math.cos(heading), math.sin(heading), 0.0]], dtype=np.float32
            ),
            random_seed=np.asarray([30300 + index], dtype=np.int64),
            has_specific_target=np.zeros((1, 1), dtype=np.int64),
            specific_target_positions=np.zeros((1, 4, 3), dtype=np.float32),
            specific_target_headings=np.zeros((1, 4), dtype=np.float32),
            allowed_pred_num_tokens=np.asarray([[1] * 6 + [0] * 5], dtype=np.int64),
            height=np.asarray([-1.0], dtype=np.float32),
        )
        if reference_height_probe_m is not None and expected_mode == 3:
            feed["height"] = np.asarray([reference_height_probe_m], dtype=np.float32)
        output, count = session.run(None, feed)
        count = np.asarray(count)
        output = np.asarray(output)
        if (
            count.shape != (1,)
            or count.dtype.kind not in "iu"
            or not 4 <= count[0] <= 64
            or output.ndim != 3
            or output.shape[0] != 1
            or output.shape[2] != 36
            or output.shape[1] < count[0]
        ):
            raise ValueError("bounded complete planner output required")
        segment = np.asarray(output[0, : int(count[0])], dtype=np.float64)
        if (
            not np.isfinite(segment).all()
            or np.max(np.abs(segment)) > 1e6
            or np.any(np.abs(np.linalg.norm(segment[:, 3:7], axis=1) - 1) > 0.01)
            or hash_bytes(segment.tobytes()) != event["reference_hash"]
        ):
            raise ValueError("independently reconstructed planner reference differs")
        # Independent 30->50 Hz interpolation and fixed future-tail insertion.
        source_time = np.arange(len(segment), dtype=np.float64) / 30.0
        target_time = np.arange(0.0, source_time[-1] + 1e-12, 0.02, dtype=np.float64)
        resampled = np.stack(
            [np.interp(target_time, source_time, segment[:, c]) for c in range(36)], axis=1
        )
        resampled[:, 3:7] /= np.linalg.norm(resampled[:, 3:7], axis=1)[:, None]
        if reference is None:
            reference = np.concatenate(
                (resampled, np.repeat(resampled[-1:], 410 - len(resampled), axis=0))
            )
        else:
            start = frame + 10
            length = min(len(resampled), len(reference) - start)
            reference[start : start + length] = resampled[:length]
            reference[start + length :] = resampled[length - 1]
        if reference_observer is not None:
            # A diagnostic cannot mutate the independently reconstructed history.
            reference_observer(frame, reference.copy())
        results.append(
            dict(frame=frame, planner_mode=expected_mode, reference_hash=event["reference_hash"])
        )
    return results


def audit_locomotion_reference(
    report: dict[str, Any], trace: dict[str, Any], planner_path: Path
) -> dict[str, Any]:
    """Caller must authenticate the complete physical report and raw trace."""
    contract = report.get("locomotion_mode_probe")
    height = contract.get("high_speed_reference_height_m") if type(contract) is dict else None
    fast = (
        type(contract) is dict
        and contract.get("schema") == "soccer.rsi.sonic_locomotion_mode_probe.v3"
    )
    fixed = dict(
        schema="soccer.rsi.sonic_locomotion_mode_probe.v1",
        planner_seed=30300,
        maximum_navigation_speed_m_s=1.5,
        high_speed_threshold_m_s=0.8,
        low_speed_modes_unchanged=True,
        physical_joint_and_torque_limits_unchanged=True,
        teacher_qualified=False,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    if height is not None:
        fixed["schema"] = "soccer.rsi.sonic_locomotion_mode_probe.v2"
    if fast:
        fixed.update(
            schema="soccer.rsi.sonic_locomotion_mode_probe.v3",
            maximum_navigation_speed_m_s=2.1,
            fast_precontact_forward_cap_m_s=2.0,
        )
    extra = {"high_speed_planner_mode", "planner_hash", "navigation_contract_hash"}
    if height is not None:
        extra.add("high_speed_reference_height_m")
    if (
        type(contract) is not dict
        or set(contract) != set(fixed) | extra
        or any(type(contract.get(k)) is not type(v) or contract[k] != v for k, v in fixed.items())
    ):
        raise ValueError("complete explicit SIM locomotion intervention contract required")
    mode = contract.get("high_speed_planner_mode")
    events = report.get("locomotion_mode_planner_events")
    if type(mode) is not int or type(events) is not list:
        raise ValueError("explicit planner mode and complete event list required")
    _inputs(trace, events, mode, height, fast)
    binding: dict[str, Any] = dict(
        schema="rosclaw_soccer.g1_sonic_navigation.v1",
        agent_id="cpu.first_touch.0",
        config=dict(
            maximum_frames=300,
            planner_seed=30300,
            replan_frames=20,
            lookahead_frames=10,
            experimental_maximum_speed_mps=1.5,
            model_variant="low_latency",
            experimental_locomotion_mode=mode,
            inference_threads=1,
        ),
        foundation=report.get("foundation_hash"),
        command="explicit experimental envelope; post-clearance command only; yaw <= 1.5",
        activation_ceiling="SIM_ONLY",
    )
    if height is not None:
        binding["config"]["experimental_reference_height_m"] = height
    if fast:
        binding["config"].update(
            experimental_fast_intercept=True, experimental_maximum_speed_mps=2.1
        )
    if contract["navigation_contract_hash"] != hash_json(binding):
        raise ValueError("original navigation body/foundation/configuration binding required")
    import hashlib

    with planner_path.open("rb") as stream:
        digest = "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != contract.get("planner_hash"):
        raise ValueError("original planner asset identity required")
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(planner_path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    rows = replay_locomotion_segments(
        session, trace, events, mode, reference_height_probe_m=height, fast_intercept_probe=fast
    )
    with planner_path.open("rb") as stream:
        after = "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()
    if after != digest:
        raise ValueError("planner asset changed during audit")
    result = dict(
        schema="soccer.rsi.locomotion_reference_replay.v1",
        source_hash=hash_bytes(Path(__file__).read_bytes()),
        physical_report_hash=report["report_hash"],
        physical_trace_hash=report["physical_trace_hash"],
        planner_hash=digest,
        high_speed_planner_mode=mode,
        actual_cpu_onnx_planner_calls=len(rows),
        events=rows,
        reference_history_reconstructed_without_producer_backend=True,
        foundation_encoder_reference_consumption_independently_checked=False,
        physical_report_and_trace_authentication_required_from_caller=True,
        physics_replayed_here=False,
        neural_optimizer_updates=0,
        teacher_qualified=False,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    if height is not None:
        result["high_speed_reference_height_m"] = height
    if fast:
        result["fast_intercept_probe"] = True
    return dict(result, report_hash=hash_json(result))
