"""Whole16 measured-joint planner-input audit; no execution after frame20."""

import argparse
import hashlib
import json
import math
import subprocess
from dataclasses import asdict
from pathlib import Path


def digest(path):
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


class RecordingPlanner:
    """Owned per-branch proxy. NEVER changes the shared ORT session or methods."""

    def __init__(self, session):
        self.session = session
        self.feeds = []
        self.outputs = []

    def run(self, names, feed):
        import numpy as np

        before = {k: v.copy() for k, v in feed.items()}
        self.feeds.append(before)
        result = self.session.run(names, feed)
        for key, value in feed.items():
            np.testing.assert_array_equal(value, before[key])
        self.outputs.append(tuple(a.copy() for a in result))
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("summary", "world", "model-root", "core-source", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--producer", type=Path, action="append", required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--expected-summary-hash", required=True)
    args = parser.parse_args()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if head != args.expected_head or args.output.exists() or len(args.producer) != 4:
        raise ValueError("sealed source, unused output and full four-cluster evidence required")
    if digest(args.summary / "complete.json") != args.expected_summary_hash:
        raise ValueError("reviewed whole16 outcome identity differs")
    sd = json.loads((args.summary / "declaration.json").read_text())
    sc = json.loads((args.summary / "complete.json").read_text())
    if (
        sd["schema"] != "soccer.rsi.close_contact_reviewed_summary.v1"
        or sc["conditions"] != 16
        or sc["parent_clusters"] != 4
        or sc["nominal_qualified"] != 2
        or sc["teacher_qualified"] != 6
        or sc["regressed"] != 0
        or sc["added_unsafe"] != 0
        or sc["total_charged_native_steps"] != 729600
        or sc["optimizer_steps"] != 0
        or any(args.summary.rglob("failure.json"))
    ):
        raise ValueError("authentic full paired reviewed predecessor required")
    for p, h in sd["input_file_hashes"].items():
        if digest(Path(p)) != h:
            raise ValueError("preceding source/assets/evidence changed")
    for p in [args.world] + [
        args.model_root / name
        for name in (
            "planner_sonic.onnx",
            "low_latency/model_encoder.onnx",
            "low_latency/model_decoder.onnx",
            "low_latency/observation_config.yaml",
        )
    ]:
        if sd["input_file_hashes"].get(str(p)) != digest(p):
            raise ValueError("same physical world and foundation assets required")
    producers = {}
    for root in args.producer:
        p = root / "complete.json"
        if sd["input_file_hashes"].get(str(p)) != digest(p):
            raise ValueError("all exact producer closures must be bound by reviewed summary")
        record = json.loads(p.read_text())
        if record["cluster"] in producers or record["total_new_native_steps"] != 91200:
            raise ValueError("unique full cluster required")
        producers[record["cluster"]] = (root, record)
    if set(producers) != set(range(4)):
        raise ValueError("all four parent clusters required")
    paths = [
        p
        for folder in (Path.cwd() / "src", Path.cwd() / "scripts", args.core_source)
        for p in folder.rglob("*.py")
    ]
    paths += [p for p in args.summary.iterdir() if p.is_file()]
    pins = {**sd["input_file_hashes"], **{str(p): digest(p) for p in paths}}

    import mujoco
    import numpy as np

    from rosclaw_soccer.physics.g1_target_episode import G1TargetEpisode
    from rosclaw_soccer.physics.sonic_navigation_forecast import (
        clone_navigation,
        navigation_state_hash,
    )
    from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
    from rosclaw_soccer.providers.g1.sonic_measured_context import measured_joint_context
    from rosclaw_soccer.providers.g1.sonic_navigation import (
        G1SonicNavigation,
        SonicNavigationConfig,
        future_reference_context,
    )
    from rosclaw_soccer.providers.g1.sonic_runup import G1SonicRunupController
    from rosclaw_soccer.rsi.ball_entry_teacher import SkillEntryAffordance, ball_entry_decision
    from rosclaw_soccer.rsi.full_entry_forecast import nominal_prefix20
    from rosclaw_soccer.rsi.full_task_branch import integration_hash
    from rosclaw_soccer.training.close_contact_courses import close_contact_perturbations
    from rosclaw_soccer.training.receiving_course_preflight import preflight_receiving_courses
    from scripts.rsi_atomic_artifacts import write_once

    bank = close_contact_perturbations()
    bank_hash = preflight_receiving_courses(bank)  # ALL16 before any model/body allocation.
    if bank_hash != sd["bank_hash"]:
        raise ValueError("same complete diagnostic bank required")
    for item in bank:
        _, record = producers[item.parent_index]
        if record["results"][item.variant_index]["course"] != asdict(item) | {
            "course": list(item.course)
        }:
            raise ValueError("ordered exact current-state nominal comparator required")
    args.output.mkdir(parents=True, exist_ok=False)
    declaration = dict(
        schema="soccer.rsi.measured_joint_planner_input16.v1",
        source_head=head,
        input_file_hashes=pins,
        bank_hash=bank_hash,
        courses=[asdict(x) for x in bank],
        planned_new_native_steps=3200,
        planned_actual_prefix_policy_ticks=320,
        planned_additional_planner_calls_if_all_contexts_legal=64,
        joint_context_maximum_increment_rad=0.12,
        actual_state_or_executed_target_modified=False,
        post20_physics_performed=False,
        original_rsi_or_private_fresh=False,
        optimizer_steps=0,
        physics_qualified=False,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    write_once(args.output / "declaration.json", declaration)
    total = planner_calls = 0
    body = None
    results = []
    try:
        choices = [Path(p) for p in pins if Path(p).name == "entry-affordance.json"]
        if len(choices) != 1:
            raise ValueError("one authenticated historical affordance required")
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
            raise ValueError("historical affordance identity differs")
        for item in bank:
            folder = args.output / f"c{item.parent_index}v{item.variant_index}"
            folder.mkdir(exist_ok=False)
            root, record = producers[item.parent_index]
            nominal = record["results"][item.variant_index]["results"][0]
            if nominal["handoff_frame"] is not None and nominal["handoff_frame"] < 50:
                raise ValueError("OFF comparison window must precede historical skill entry")
            with np.load(
                root / f"variant{item.variant_index}" / nominal["artifact"], allow_pickle=False
            ) as archive:
                gold = {k: archive[k] for k in archive.files}
            nav = G1SonicNavigation(
                args.model_root,
                item.agent_id,
                SonicNavigationConfig(
                    maximum_frames=300,
                    model_variant="low_latency",
                    inference_threads=1,
                    record_default_planner_events=True,
                ),
            )
            model = mujoco.MjModel.from_binary_path(str(args.world))
            body = G1TargetEpisode(
                model,
                world_hash=digest(args.world),
                kp=nav.backend.kp.copy(),
                kd=nav.backend.kd.copy(),
                course=item.course,
                defaults=G1SonicRunupController.default_angles.copy(),
                record_all_ball_contacts=True,
            )
            prefix = nominal_prefix20(nav, body, affordance)
            for key in prefix:
                np.testing.assert_array_equal(prefix[key], gold[key][:20])
            q, v = body.canonical()
            np.testing.assert_array_equal(q, gold["canonical_qpos"][20])
            np.testing.assert_array_equal(v, gold["canonical_qvel"][20])
            if body.integrated_steps != 200 or body.attempted_steps != 200:
                raise ValueError("whole actual prefix accounting differs")
            total += 200
            prefix_file = folder / "prefix20.npz"
            np.savez_compressed(prefix_file, **prefix)
            donor_nav, donor_body = navigation_state_hash(nav), integration_hash(model, body.data)
            if nav.backend.planner_calls != 1 or nav._next_frame != 20:
                raise ValueError(
                    "prefix must contain exactly one initial planner and20 policy ticks"
                )
            shared_session = nav.backend._planner
            shared_method = shared_session.run
            context = future_reference_context(nav.backend.reference, 30)
            reference_now = nav.backend.reference[20].copy()
            qi = np.asarray([model.joint(name).id for name in G1_DDS_JOINT_NAMES])
            limits = model.jnt_range[qi].copy()
            rejection = None
            try:
                corrected, delta = measured_joint_context(context, q, reference_now, limits)
            except ValueError as error:
                rejection = str(error)
                corrected, delta = None, None
            outputs = {}
            arrays = dict(
                original_context=context,
                measured_qpos=q,
                current_reference=reference_now,
                native_joint_limits=limits,
            )
            if corrected is not None:
                arrays.update(modified_context=corrected, joint_context_delta=delta)
            for arm in ("off", "on"):
                if arm == "on" and rejection is not None:
                    continue
                arm_results = []
                for repeat in range(2):
                    branch = clone_navigation(nav)
                    command = ball_entry_decision(q, v, frame=20, affordance=affordance).command
                    branch.backend.command = command
                    w, x, y, z = q[3:7]
                    branch.backend.facing = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
                    proxy = RecordingPlanner(branch.backend._planner)
                    branch.backend._planner = (
                        proxy  # Owned branch attribute, NOT shared ORT mutation.
                    )
                    planner_calls += 1
                    try:
                        segment = branch.backend.plan(context if arm == "off" else corrected, 20)
                    finally:
                        # Preserve even a model/output-validation exception's consumed inputs.
                        captured = {
                            "feed__" + k: a
                            for k, a in (proxy.feeds[0].items() if proxy.feeds else ())
                        }
                        if proxy.outputs:
                            captured["raw_onnx_output"] = proxy.outputs[0][0]
                            captured["num_pred_frames"] = proxy.outputs[0][1]
                        np.savez_compressed(folder / f"{arm}{repeat}-call.npz", **captured)
                    if len(proxy.feeds) != 1 or len(proxy.outputs) != 1:
                        raise ValueError("exactly one captured real planner call required")
                    feed, raw_output = proxy.feeds[0], proxy.outputs[0]
                    np.testing.assert_array_equal(
                        feed["context_mujoco_qpos"],
                        (context if arm == "off" else corrected)[None].astype(np.float32),
                    )
                    arm_results.append((segment, feed, raw_output))
                    if arm == "off":
                        np.testing.assert_array_equal(
                            segment[:20], gold["encoder_reference_poses"][30:50, 0]
                        )
                    tag = f"{arm}{repeat}__"
                    arrays.update({tag + "feed__" + k: a for k, a in feed.items()})
                    arrays[tag + "segment"] = segment
                    arrays[tag + "raw_onnx_output"] = raw_output[0]
                    arrays[tag + "num_pred_frames"] = raw_output[1]
                a, b = arm_results
                for key in a[1]:
                    np.testing.assert_array_equal(a[1][key], b[1][key])
                for aa, bb in zip((a[0], *a[2]), (b[0], *b[2]), strict=True):
                    np.testing.assert_array_equal(aa, bb)
                outputs[arm] = a
            if (
                navigation_state_hash(nav) != donor_nav
                or integration_hash(model, body.data) != donor_body
                or nav.backend._planner is not shared_session
                or shared_session.run != shared_method
                or body.integrated_steps != 200
            ):
                raise ValueError("planner-input audit changed donor/session/native state")
            changed = False
            joint_effect = root_effect = None
            if rejection is None:
                off, on = outputs["off"], outputs["on"]
                if set(off[1]) != set(on[1]):
                    raise ValueError("planner feed set differs")
                for key in off[1]:
                    if key != "context_mujoco_qpos":
                        np.testing.assert_array_equal(off[1][key], on[1][key])
                np.testing.assert_array_equal(
                    off[1]["context_mujoco_qpos"][..., :7], on[1]["context_mujoco_qpos"][..., :7]
                )
                changed = off[0].shape != on[0].shape or not np.array_equal(off[0], on[0])
                common = min(len(off[0]), len(on[0]))
                joint_effect = float(abs(on[0][:common, 7:] - off[0][:common, 7:]).max())
                root_effect = float(abs(on[0][:common, :3] - off[0][:common, :3]).max())
            seam = {
                arm: dict(
                    first_joint_jump_rad=float(
                        abs(value[0][0, 7:] - nav.backend.reference[29, 7:]).max()
                    ),
                    first10_maximum_adjacent_joint_change_rad=float(
                        abs(np.diff(value[0][:10, 7:], axis=0)).max()
                    ),
                )
                for arm, value in outputs.items()
            }
            artifact = folder / "planner-inputs-outputs.npz"
            np.savez_compressed(artifact, **arrays)
            result = dict(
                course=asdict(item),
                original_or_modified_context_rejected=rejection,
                model_output_changed=changed,
                common_segment_maximum_joint_effect_rad=joint_effect,
                common_segment_maximum_root_xyz_effect_m=root_effect,
                seam_diagnostic_not_dynamic_qualification=seam,
                current_state_gap_maximum_rad=float(abs(q[7:36] - reference_now[7:36]).max()),
                additional_planner_calls=4 if rejection is None else 2,
                prefix_all18_exact_to_own_archived_nominal=True,
                off_segment_exact_to_own_archived_nominal=True,
                repeats_bit_exact=True,
                donor_navigation_hash=donor_nav,
                donor_native_hash=donor_body,
                original_shared_session_object_and_run_method_unchanged=True,
                prefix_artifact_hash=digest(prefix_file),
                artifact_hash=digest(artifact),
                actual_native_steps=200,
                post20_physics_steps=0,
                optimizer_steps=0,
                physics_qualified=False,
                promotion_authorized=False,
                hardware_authorized=False,
            )
            write_once(folder / "complete.json", result)
            results.append(result)
            print(
                "MEASURED_JOINT_CONTEXT_AUDIT",
                item.parent_index,
                item.variant_index,
                "rejected",
                rejection,
                "changed",
                changed,
                "seam",
                seam,
                flush=True,
            )
            body = None
        if (
            total != 3200
            or len(results) != 16
            or planner_calls != sum(r["additional_planner_calls"] for r in results)
        ):
            raise ValueError("whole16 audit native/planner accounting differs")
        for p, h in pins.items():
            if digest(Path(p)) != h:
                raise ValueError("source/assets/evidence changed during audit")
        write_once(
            args.output / "complete.json",
            dict(
                results=results,
                actual_native_steps=total,
                actual_prefix_policy_ticks=320,
                actual_prefix_encoder_calls=320,
                actual_prefix_decoder_calls=320,
                additional_real_planner_calls=planner_calls,
                prefix_initial_planner_calls=16,
                model_sensitive_conditions=sum(r["model_output_changed"] for r in results),
                rejected_context_conditions=sum(
                    r["original_or_modified_context_rejected"] is not None for r in results
                ),
                post20_physics_steps=0,
                optimizer_steps=0,
                physics_qualified=False,
                learned_student_qualified=False,
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
                incomplete_native_steps=body.integrated_steps
                if body is not None and body.integrated_steps != 200
                else 0,
                incomplete_attempted_native_steps=body.attempted_steps
                if body is not None and body.integrated_steps != 200
                else 0,
                attempted_additional_planner_calls=planner_calls,
                promotion_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
