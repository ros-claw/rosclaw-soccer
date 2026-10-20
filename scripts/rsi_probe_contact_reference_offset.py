"""Paired contact-authority pilot: two fixed entry laws, zero vs toe offset.

Four own-neural H300 executions,12000 native steps, no search/training. The
failed far and successful near contexts both remain, never an official exam.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def digest(path):
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "prior",
        "prior-review",
        "world",
        "motion-csv",
        "model-root",
        "core-source",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("expected-head", "expected-prior-review-hash"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    source = Path.cwd().resolve()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if args.output.exists() or head != args.expected_head:
        raise ValueError("pinned source and unused output required")
    if digest(args.prior_review / "complete.json") != args.expected_prior_review_hash:
        raise ValueError("complete preceding physics proof differs")
    previous = json.loads((args.prior / "declaration.json").read_text())
    closed = json.loads((args.prior / "complete.json").read_text())
    reviewed = json.loads((args.prior_review / "complete.json").read_text())
    if (
        previous["schema"] != "soccer.rsi.full_remaining_entry_teacher4.v2"
        or previous["law_family"] != "feedback"
        or closed["total_new_native_steps"] != 91200
        or reviewed["additional_total_native_steps"] != 91200
        or len(reviewed["results"]) != 8
        or not all(
            r["actual_physics_bit_exact"] and r["all_candidate_physics_bit_exact"]
            for r in reviewed["results"]
        )
        or any((p / "failure.json").exists() for p in (args.prior, args.prior_review))
    ):
        raise ValueError(
            "entire closed feedback family required before contact-interface experiment"
        )
    for p, h in previous["input_file_hashes"].items():
        if digest(Path(p)) != h:
            raise ValueError("prior source/assets/evidence changed")
    for path in (args.world, args.motion_csv):
        if previous["input_file_hashes"].get(str(path)) != digest(path):
            raise ValueError("same source and world required")
    for row in closed["results"]:
        if digest(args.prior / (row["job"] + ".npz")) != row["artifact_hash"]:
            raise ValueError("prior actual trajectory differs")
        for c in row["candidates"]:
            if digest(args.prior / c["artifact"]) != c["artifact_hash"]:
                raise ValueError("prior candidate trajectory differs")
    paths = [
        p
        for folder in (source / "src", source / "scripts", args.core_source)
        for p in folder.rglob("*.py")
    ]
    paths += [p for root in (args.prior, args.prior_review) for p in root.iterdir() if p.is_file()]
    paths += [
        args.world,
        args.motion_csv,
        args.model_root / "planner_sonic.onnx",
        args.model_root / "low_latency/model_encoder.onnx",
        args.model_root / "low_latency/model_decoder.onnx",
        args.model_root / "low_latency/observation_config.yaml",
    ]
    pins = {str(p): digest(p) for p in paths}

    import mujoco
    import numpy as np

    from rosclaw_soccer.physics.g1_target_episode import G1TargetEpisode
    from rosclaw_soccer.providers.g1.sonic_navigation import (
        G1SonicNavigation,
        SonicNavigationConfig,
    )
    from rosclaw_soccer.providers.g1.sonic_reference_preparation import prepare_sonic_reference
    from rosclaw_soccer.providers.g1.sonic_runup import G1SonicRunupController
    from rosclaw_soccer.providers.g1.sonic_wrist_quarantine import quarantine_wrist_reference
    from rosclaw_soccer.rsi.ball_entry_teacher import SkillEntryAffordance
    from rosclaw_soccer.rsi.contact_reference_offset import offset_contact_reference
    from rosclaw_soccer.rsi.full_entry_forecast import (
        feedback_entry_laws,
        nominal_prefix20,
        run_entry_suffix,
        score_full_entry,
    )
    from rosclaw_soccer.rsi.reference_ball_strike_metrics import summarize_reference_ball_strike
    from rosclaw_soccer.sim.contracts import hash_json
    from scripts.rsi_atomic_artifacts import write_once

    args.output.mkdir(parents=True, exist_ok=False)
    declaration = dict(
        schema="soccer.rsi.contact_reference_offset_pair2.v1",
        source_head=head,
        input_file_hashes=pins,
        contexts=[dict(course_index=1, law_index=3), dict(course_index=2, law_index=0)],
        arms=dict(off=[0.0, 0.0, 0.0], lower_inward=[0.0, -0.025, -0.025]),
        actual_bodies=4,
        planned_native_steps=12000,
        forecast_native_steps=0,
        optimizer_steps=0,
        reused_consumed_courses=2,
        independent_new_courses=0,
        horizon=300,
        latest_handoff=90,
        source_interval=[286, 767],
        entry_intervals=50,
        reference_only=True,
        no_phase_change=True,
        no_actual_target_torque_or_state_adoption=True,
        publicity_license_authorized=False,
        official_football_exam=False,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    write_once(args.output / "declaration.json", declaration)
    completed_native_steps = 0
    current_body = None
    try:
        stored = json.loads((args.prior / "entry-affordance.json").read_text())
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
            raise ValueError("original past entry-affordance differs")
        poses = np.loadtxt(args.motion_csv, delimiter=",", skiprows=1, dtype=np.float64)
        # Both historical entries must pass source-kinematics admission before
        # any actual neural actor/body is allocated. This is not task success.
        preflight_model = mujoco.MjModel.from_binary_path(str(args.world))
        geometry_admissions = []
        for context in declaration["contexts"]:
            i, index = context["course_index"], context["law_index"]
            parent = next(r for r in closed["results"] if r["job"] == f"course{i}-full_teacher")
            candidate = parent["candidates"][index]
            h = candidate["report"]["handoff_frame"]
            if h is None:
                raise ValueError("contact-authority contexts must have observed skill entry")
            with np.load(args.prior / candidate["artifact"], allow_pickle=False) as archive:
                entry = archive["canonical_qpos"][h - 20, :36].copy()
            held, _ = quarantine_wrist_reference(poses, entry)
            prepared = prepare_sonic_reference(
                held[286:767],
                entry,
                source_frequency_hz=120,
                entry_intervals=50,
                source_evidence_hash=hash_json(declaration),
            )
            for arm, offset in declaration["arms"].items():
                modified, geometry = offset_contact_reference(
                    preflight_model, prepared.reference, offset_root_m=tuple(offset)
                )
                modified.initialize(entry, frames=410)
                geometry_admissions.append(dict(course_index=i, arm=arm, geometry=geometry))
        write_once(
            args.output / "geometry-admission.json",
            dict(
                results=geometry_admissions,
                new_native_steps=0,
                teacher_qualified=False,
                promotion_authorized=False,
            ),
        )
        results = []
        for context in declaration["contexts"]:
            i, index = context["course_index"], context["law_index"]
            course = previous["courses"][i]
            parent = next(r for r in closed["results"] if r["job"] == f"course{i}-full_teacher")
            candidate = parent["candidates"][index]
            with (
                np.load(args.prior / f"course{i}-nominal.npz", allow_pickle=False) as a,
                np.load(args.prior / candidate["artifact"], allow_pickle=False) as b,
            ):
                gold = {key: np.concatenate((a[key][:20], b[key])) for key in a.files}
            off_arrays = None
            for arm, offset in declaration["arms"].items():
                name = f"course{i}-{arm}"
                nav = G1SonicNavigation(
                    args.model_root,
                    "tracker",
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
                    course=tuple(course),
                    defaults=G1SonicRunupController.default_angles.copy(),
                    record_all_ball_contacts=True,
                )
                current_body = body
                prefix = nominal_prefix20(nav, body, affordance)
                suffix, report = run_entry_suffix(
                    nav,
                    body,
                    model_root=args.model_root,
                    source_poses=poses,
                    source_evidence_hash=hash_json(declaration),
                    affordance=affordance,
                    law=feedback_entry_laws()[index],
                    contact_reference_offset_m=tuple(offset),
                )
                arrays = {k: np.concatenate((prefix[k], suffix[k])) for k in prefix}
                if arm == "off":
                    for key in gold:
                        np.testing.assert_array_equal(arrays[key], gold[key])
                    off_arrays = arrays
                if report["handoff_frame"] != candidate["report"]["handoff_frame"]:
                    raise ValueError("contact-only adaptation changed pre-contact entry timing")
                expected_geometry = next(
                    g["geometry"]
                    for g in geometry_admissions
                    if g["course_index"] == i and g["arm"] == arm
                )
                if report["contact_reference_adaptation"] != expected_geometry:
                    raise ValueError("actual measured entry reference differs from preflight")
                artifact = args.output / (name + ".npz")
                np.savez_compressed(artifact, **arrays)
                report.update(
                    job=name,
                    course=list(course),
                    law_index=index,
                    offset_root_m=offset,
                    artifact_hash=digest(artifact),
                    native_steps=3000,
                    score=score_full_entry(model, suffix),
                    strike=summarize_reference_ball_strike(model, arrays),
                    off_all_arrays_exact_to_prior_own_neural_forecast=arm == "off",
                )
                if arm != "off":
                    assert off_arrays is not None
                    report["maximum_array_deltas_from_paired_off"] = {
                        k: float(np.max(abs(arrays[k] - off_arrays[k]))) for k in arrays
                    }
                    report["first_reference_change_frame"] = next(
                        (
                            f
                            for f in range(300)
                            if not np.array_equal(
                                arrays["encoder_reference_poses"][f],
                                off_arrays["encoder_reference_poses"][f],
                            )
                        ),
                        None,
                    )
                results.append(report)
                write_once(args.output / (name + "-complete.json"), report)
                print("CONTACT_REFERENCE_AUTHORITY_CLOSED", name, report["score"], flush=True)
                completed_native_steps += 3000
                current_body = None
        for p, h in {**previous["input_file_hashes"], **pins}.items():
            if digest(Path(p)) != h:
                raise ValueError("source/assets/evidence changed during experiment")
        write_once(
            args.output / "complete.json",
            dict(
                results=results,
                actual_native_steps=12000,
                forecast_native_steps=0,
                optimizer_steps=0,
                reused_consumed_courses=2,
                independent_new_courses=0,
                promotion_authorized=False,
                hardware_authorized=False,
            ),
        )
    except Exception as error:
        write_once(
            args.output / "failure.json",
            dict(
                error=repr(error),
                completed_actual_native_steps=completed_native_steps,
                incomplete_body_attempted_native_steps=current_body.attempted_steps
                if current_body is not None
                else 0,
                incomplete_body_integrated_native_steps=current_body.integrated_steps
                if current_body is not None
                else 0,
                promotion_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
