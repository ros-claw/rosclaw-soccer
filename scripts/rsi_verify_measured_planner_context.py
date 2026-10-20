"""Independent raw-ONNX replay and recorded-PD prefix review of the16 audit.

No G1SonicNavigation, planner method, recording proxy or encoder/decoder replay.
Shares native PD,30->50 resampling, future-context and bounded-context helpers.
"""

import argparse
import hashlib
import json
import math
import subprocess
from dataclasses import asdict
from pathlib import Path


def digest(path):
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("producer", "world", "model-root", "core-source", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--previous-producer", type=Path, action="append", required=True)
    for name in (
        "expected-head",
        "expected-producer-head",
        "expected-producer-hash",
        "producer-unit",
    ):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    if (
        args.output.exists()
        or len(args.previous_producer) != 4
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != args.expected_head
    ):
        raise ValueError("sealed reader, unused output and all previous clusters required")
    terminal = dict(
        line.split("=", 1)
        for line in subprocess.check_output(
            [
                "systemctl",
                "--user",
                "show",
                args.producer_unit,
                "-p",
                "MainPID",
                "-p",
                "Result",
                "-p",
                "SubState",
                "-p",
                "ExecMainStatus",
            ],
            text=True,
        ).splitlines()
    )
    if terminal != {
        "MainPID": "0",
        "Result": "success",
        "SubState": "exited",
        "ExecMainStatus": "0",
    }:
        raise ValueError("successfully closed whole-bank producer required")
    if digest(args.producer / "complete.json") != args.expected_producer_hash:
        raise ValueError("audit completion commitment differs")
    pd = json.loads((args.producer / "declaration.json").read_text())
    pc = json.loads((args.producer / "complete.json").read_text())
    if (
        pd["schema"] != "soccer.rsi.measured_joint_planner_input16.v1"
        or pd["source_head"] != args.expected_producer_head
        or pc["actual_native_steps"] != 3200
        or pc["actual_prefix_policy_ticks"] != 320
        or pc["prefix_initial_planner_calls"] != 16
        or len(pc["results"]) != 16
        or pc["post20_physics_steps"] != 0
        or pc["optimizer_steps"] != 0
        or any(args.producer.rglob("failure.json"))
    ):
        raise ValueError("authentic complete16 prefix/planner audit required")
    for p, h in pd["input_file_hashes"].items():
        if digest(Path(p)) != h:
            raise ValueError("producer source/assets/evidence changed")
    for p in [args.world, args.model_root / "planner_sonic.onnx"]:
        if pd["input_file_hashes"].get(str(p)) != digest(p):
            raise ValueError("bound world and planner required")
    previous = {}
    for root in args.previous_producer:
        p = root / "complete.json"
        if pd["input_file_hashes"].get(str(p)) != digest(p):
            raise ValueError("bound preceding comparator required")
        record = json.loads(p.read_text())
        if record["cluster"] in previous:
            raise ValueError("one original cluster each required")
        previous[record["cluster"]] = (root, record)
    if set(previous) != set(range(4)):
        raise ValueError("all four prior clusters required")
    paths = [
        p
        for folder in (Path.cwd() / "src", Path.cwd() / "scripts", args.core_source)
        for p in folder.rglob("*.py")
    ]
    paths += [p for p in args.producer.rglob("*") if p.is_file()]
    pins = {**pd["input_file_hashes"], **{str(p): digest(p) for p in paths}}

    import mujoco
    import numpy as np
    import onnxruntime as ort

    from rosclaw_soccer.physics.g1_target_episode import G1TargetEpisode
    from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
    from rosclaw_soccer.providers.g1.sonic_measured_context import measured_joint_context
    from rosclaw_soccer.providers.g1.sonic_navigation import future_reference_context
    from rosclaw_soccer.providers.g1.sonic_runup import (
        G1SonicRunupController,
        _resample_segments_30_to_50,
        _sonic_control_parameters,
    )
    from rosclaw_soccer.training.close_contact_courses import close_contact_perturbations
    from rosclaw_soccer.training.receiving_course_preflight import preflight_receiving_courses
    from scripts.rsi_atomic_artifacts import write_once

    bank = close_contact_perturbations()
    if preflight_receiving_courses(bank) != pd["bank_hash"]:
        raise ValueError("same complete admitted bank required")
    for item, row in zip(bank, pc["results"], strict=True):
        if row["course"] != asdict(item) | {"course": list(item.course)}:
            raise ValueError("every exact ordered public condition required")
        folder = args.producer / f"c{item.parent_index}v{item.variant_index}"
        if row != json.loads((folder / "complete.json").read_text()):
            raise ValueError("individual audit receipt differs")
        if (
            digest(folder / "prefix20.npz") != row["prefix_artifact_hash"]
            or digest(folder / "planner-inputs-outputs.npz") != row["artifact_hash"]
        ):
            raise ValueError("audit raw artifact identity differs")
    args.output.mkdir(parents=True, exist_ok=False)
    write_once(
        args.output / "declaration.json",
        dict(
            schema="soccer.rsi.measured_joint_context_review.v1",
            input_file_hashes=pins,
            producer_terminal=terminal,
            expected_additional_recorded_pd_native_steps=3200,
            expected_additional_initial_planner_calls=16,
            expected_additional_captured_planner_calls=pc["additional_real_planner_calls"],
            encoder_decoder_replayed=False,
            production_navigation_and_proxy_used=False,
            shared_native_resampling_and_context_helpers=True,
            optimizer_steps=0,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )
    total = calls = 0
    body = None
    body_start = 0
    try:
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        session = ort.InferenceSession(
            str(args.model_root / "planner_sonic.onnx"),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        model = mujoco.MjModel.from_binary_path(str(args.world))
        kp, kd, _ = _sonic_control_parameters(1.0, (1.0,) * 29)
        limits = model.jnt_range[[model.joint(n).id for n in G1_DDS_JOINT_NAMES]].copy()
        results = []

        def segment(raw):
            poses, count = raw
            if (
                count.shape != (1,)
                or count.dtype.kind not in "iu"
                or not 4 <= int(count[0]) <= poses.shape[1]
            ):
                raise ValueError("valid planner output length required")
            return _resample_segments_30_to_50(
                [np.asarray(poses[0, : int(count[0])], dtype=np.float64)], 0.02
            )

        def set_command(feed, command, qpos):
            # Literal feed construction; does not call production backend.plan().
            vx, vy, yaw_rate = command
            speed = math.hypot(vx, vy)
            w, x, y, z = qpos[3:7]
            facing = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            direction = (
                (vx / speed, vy / speed) if speed > 1e-8 else (math.cos(facing), math.sin(facing))
            )
            heading = facing + yaw_rate * 0.4
            feed["target_vel"] = np.array([speed], dtype=np.float32)
            feed["mode"] = np.array(
                [0 if speed < 0.02 else 1 if speed <= 0.8 else 2], dtype=np.int64
            )
            feed["movement_direction"] = np.array([[*direction, 0.0]], dtype=np.float32)
            feed["facing_direction"] = np.array(
                [[math.cos(heading), math.sin(heading), 0.0]], dtype=np.float32
            )
            feed["has_specific_target"] = np.zeros((1, 1), dtype=np.int64)
            feed["specific_target_positions"] = np.zeros((1, 4, 3), dtype=np.float32)
            feed["specific_target_headings"] = np.zeros((1, 4), dtype=np.float32)
            feed["allowed_pred_num_tokens"] = np.array(
                [[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]], dtype=np.int64
            )
            feed["height"] = np.array([-1.0], dtype=np.float32)

        for item, row in zip(bank, pc["results"], strict=True):
            folder = args.producer / f"c{item.parent_index}v{item.variant_index}"
            with np.load(folder / "prefix20.npz", allow_pickle=False) as archive:
                prefix = {k: archive[k] for k in archive.files}
            with np.load(folder / "planner-inputs-outputs.npz", allow_pickle=False) as archive:
                arrays = {k: archive[k] for k in archive.files}
            root, original = previous[item.parent_index]
            prior_row = original["results"][item.variant_index]["results"][0]
            with np.load(
                root / f"variant{item.variant_index}" / prior_row["artifact"], allow_pickle=False
            ) as archive:
                gold = {k: archive[k] for k in archive.files}
            if set(prefix) != set(gold) or any(len(a) != 20 for a in prefix.values()):
                raise ValueError("whole matching18-field H20 prefix required")
            body = G1TargetEpisode(
                model,
                world_hash=digest(args.world),
                kp=kp,
                kd=kd,
                course=item.course,
                defaults=G1SonicRunupController.default_angles.copy(),
                record_all_ball_contacts=True,
            )
            body_start = body.integrated_steps
            for frame in range(20):
                physical = body.step(frame, prefix["joint_target_rad"][frame].copy())
                for k, a in physical.items():
                    np.testing.assert_array_equal(a[0], prefix[k][frame])
            q, v = body.canonical()
            if body.integrated_steps != 200 or body.attempted_steps != 200:
                raise ValueError("whole recorded prefix native accounting differs")
            np.testing.assert_array_equal(q, arrays["measured_qpos"])
            np.testing.assert_array_equal(q, gold["canonical_qpos"][20])
            np.testing.assert_array_equal(v, gold["canonical_qvel"][20])
            for k, a in prefix.items():
                np.testing.assert_array_equal(a, gold[k][:20])
            total += 200
            body = None
            np.testing.assert_array_equal(limits, arrays["native_joint_limits"])
            feed_template = {
                k[len("off0__feed__") :]: a.copy()
                for k, a in arrays.items()
                if k.startswith("off0__feed__")
            }
            # Rebuild the initial plan independently; do not trust a supplied future context.
            feed0 = {k: a.copy() for k, a in feed_template.items()}
            feed0["context_mujoco_qpos"] = np.repeat(
                prefix["canonical_qpos"][0, :36][None], 4, axis=0
            )[None].astype(np.float32)
            feed0["random_seed"] = np.array([30300], dtype=np.int64)
            set_command(feed0, tuple(prefix["entry_command"][0]), prefix["canonical_qpos"][0])
            calls += 1
            reference = segment(session.run(None, feed0))
            for frame in range(20):
                np.testing.assert_array_equal(
                    reference[frame : frame + 10], prefix["encoder_reference_poses"][frame]
                )
            np.testing.assert_array_equal(reference[20], arrays["current_reference"])
            np.testing.assert_array_equal(reference[20], gold["encoder_reference_poses"][20, 0])
            context = future_reference_context(reference, 30)
            np.testing.assert_array_equal(context, arrays["original_context"])
            denial = None
            try:
                corrected, delta = measured_joint_context(context, q, reference[20], limits)
            except ValueError as error:
                denial = str(error)
                corrected, delta = None, None
            if denial != row["original_or_modified_context_rejected"]:
                raise ValueError("bounded-context admissibility differs")
            if corrected is not None:
                np.testing.assert_array_equal(corrected, arrays["modified_context"])
                np.testing.assert_array_equal(delta, arrays["joint_context_delta"])
            segments = {}
            for arm in ("off", "on"):
                if arm == "on" and denial is not None:
                    continue
                repeats = []
                for repeat in range(2):
                    tag = f"{arm}{repeat}__"
                    feed = {
                        k[len(tag + "feed__") :]: a
                        for k, a in arrays.items()
                        if k.startswith(tag + "feed__")
                    }
                    expected = {k: a.copy() for k, a in feed_template.items()}
                    expected["context_mujoco_qpos"] = (context if arm == "off" else corrected)[
                        None
                    ].astype(np.float32)
                    expected["random_seed"] = np.array([30301], dtype=np.int64)
                    set_command(expected, tuple(gold["entry_command"][20]), q)
                    if set(feed) != set(expected) or set(feed) != {
                        i.name for i in session.get_inputs()
                    }:
                        raise ValueError("exact complete model feed set required")
                    for k in feed:
                        np.testing.assert_array_equal(feed[k], expected[k])
                    with np.load(
                        folder / f"{arm}{repeat}-call.npz", allow_pickle=False
                    ) as captured:
                        for k in feed:
                            np.testing.assert_array_equal(feed[k], captured["feed__" + k])
                        np.testing.assert_array_equal(
                            arrays[tag + "raw_onnx_output"], captured["raw_onnx_output"]
                        )
                        np.testing.assert_array_equal(
                            arrays[tag + "num_pred_frames"], captured["num_pred_frames"]
                        )
                    calls += 1
                    raw = session.run(None, feed)
                    np.testing.assert_array_equal(raw[0], arrays[tag + "raw_onnx_output"])
                    np.testing.assert_array_equal(raw[1], arrays[tag + "num_pred_frames"])
                    value = segment(raw)
                    np.testing.assert_array_equal(value, arrays[tag + "segment"])
                    repeats.append(value)
                    if arm == "off":
                        np.testing.assert_array_equal(
                            value[:20], gold["encoder_reference_poses"][30:50, 0]
                        )
                np.testing.assert_array_equal(*repeats)
                segments[arm] = repeats[0]
            changed = denial is None and (
                segments["off"].shape != segments["on"].shape
                or not np.array_equal(segments["off"], segments["on"])
            )
            if row["model_output_changed"] != changed:
                raise ValueError("valid planner-output sensitivity differs")
            if denial is None:
                common = min(len(segments["off"]), len(segments["on"]))
                difference = abs(segments["on"][:common] - segments["off"][:common])
                if (
                    float(difference[:, 7:].max()) != row["common_segment_maximum_joint_effect_rad"]
                    or float(difference[:, :3].max())
                    != row["common_segment_maximum_root_xyz_effect_m"]
                ):
                    raise ValueError("sensitivity magnitudes differ")
            seams = {
                arm: dict(
                    first_joint_jump_rad=float(abs(a[0, 7:] - reference[29, 7:]).max()),
                    first10_maximum_adjacent_joint_change_rad=float(
                        abs(np.diff(a[:10, 7:], axis=0)).max()
                    ),
                )
                for arm, a in segments.items()
            }
            if seams != row["seam_diagnostic_not_dynamic_qualification"]:
                raise ValueError("new-reference continuity diagnostic differs")
            result = dict(
                course=row["course"],
                recorded_prefix_native_bit_exact=True,
                initial_and_all_recorded_planner_outputs_bit_exact=True,
                measured_context_and_all_other_feeds_reconstructed=True,
                original_context_reconstructed_from_initial_model_output=True,
                model_output_changed=changed,
                rejected_context=denial,
                seam_diagnostic_not_dynamic_qualification=seams,
                physics_qualified=False,
                hardware_authorized=False,
            )
            results.append(result)
            write_once(
                args.output / f"c{item.parent_index}v{item.variant_index}-complete.json", result
            )
            print(
                "MEASURED_CONTEXT_NATIVE_AND_ONNX_REPLAY_EXACT",
                item.parent_index,
                item.variant_index,
                flush=True,
            )
        if total != 3200 or calls != 16 + pc["additional_real_planner_calls"]:
            raise ValueError("whole16 independent replay work differs")
        if pc["model_sensitive_conditions"] != sum(
            r["model_output_changed"] for r in results
        ) or pc["rejected_context_conditions"] != sum(
            r["rejected_context"] is not None for r in results
        ):
            raise ValueError("whole-bank sensitivity/admission counts differ")
        for p, h in pins.items():
            if digest(Path(p)) != h:
                raise ValueError("sealed sources/assets/raw evidence changed")
        write_once(
            args.output / "complete.json",
            dict(
                results=results,
                additional_recorded_pd_native_steps=total,
                additional_real_planner_calls=calls,
                initial_model_output_reconstructions=16,
                captured_call_replays=pc["additional_real_planner_calls"],
                encoder_decoder_replayed=False,
                production_navigation_and_proxy_used=False,
                post20_physics_steps=0,
                optimizer_steps=0,
                physics_qualified=False,
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
                incomplete_native_steps=body.integrated_steps - body_start
                if body is not None
                else 0,
                attempted_real_planner_calls=calls,
                promotion_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
