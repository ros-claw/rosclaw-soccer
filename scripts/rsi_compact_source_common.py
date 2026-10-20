"""Admission/resources for one source-lead hypothesis, not an outcome sweep."""

import json
import subprocess
from pathlib import Path


def admit(a):
    from scripts.rsi_collect_shadow_motor import wait_for_terminal_collection
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    if (
        a.output.exists()
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
    ):
        raise ValueError("clean frozen source and unused output required")
    wait_for_terminal_collection(a.old_review_unit)
    wait_for_terminal_collection(a.default_parity_unit)
    if digest(a.old_review / "complete.json") != a.expected_old_review_hash:
        raise ValueError("exact completed actual30 recovery24 raw audit required")
    old = json.loads((a.old_review / "complete.json").read_text())
    parity = json.loads((a.default_parity / "complete.json").read_text())
    if (
        old["status"] != "ACTUAL_STUDENT30_RECOVERY24_RAW_REPRODUCTION_AUTHENTICATED"
        or old["reviewed_native_steps"] != 1310400
        or old["summary"]["contexts"] != 24
        or parity["status"] != "NOMINAL_ZERO_TEACHER_NEURAL_NATIVE_REPRODUCED"
        or parity["source_head"] != a.expected_default_source
        or parity["all18_array_fields_byte_exact_to_original"] is not True
        or parity["new_native_steps"] != 62000
        or parity["optimizer_steps"] != 0
    ):
        raise ValueError("closed native/NN default and actual30 prerequisites required")
    pins = dict(old["input_file_hashes"])
    for f in (a.inputs, a.near_plan / "complete.json", a.old_launch):
        if pins.get(str(f)) != digest(f):
            raise ValueError("exact old inputs/near roster/launch required: " + str(f))
    for f in (
        a.old_review / "complete.json",
        a.default_parity / "complete.json",
        a.default_parity / "declaration.json",
    ):
        pins[str(f)] = digest(f)
    spec = json.loads(a.inputs.read_text())
    a.world, a.model_root = Path(spec["world"]), Path(spec["model_root"])
    from rosclaw_soccer.rsi.source_lead_protocol import matched_old_six_selection
    from rosclaw_soccer.rsi.teacher_coverage_cohort import teacher_coverage_indices
    from rosclaw_soccer.training.categorical_motion_holdout import (
        categorical_motion_holdout_courses,
    )
    from rosclaw_soccer.training.receiving_course_preflight import preflight_receiving_courses

    indices = teacher_coverage_indices()
    courses = categorical_motion_holdout_courses()
    near = json.loads((a.near_plan / "complete.json").read_text())
    if preflight_receiving_courses(courses) != near["preflight_hash"]:
        raise ValueError("unchanged whole96 population preflight required")
    roster = near["roster"]
    if [r["variant"] for r in roster] != ["anchor", "seed17", "seed29", "seed43"]:
        raise ValueError("original fixed roster required")
    for root in (a.warm, a.warm_repeat):
        if pins.get(str(root / "complete.json")) != digest(root / "complete.json"):
            raise ValueError("exact qualified warm lineage required")
    launch = json.loads(a.old_launch.read_text())
    old_cases = {}
    baseline = []
    for job in launch["jobs"]:
        if job["dependencies"]:
            continue
        root = Path(job["root"])
        c = json.loads((root / "complete.json").read_text())
        for row in c["results"]:
            i = row["course_index"]
            if i in old_cases or i not in indices:
                raise ValueError("unique whole24 old actual cases required")
            baseline.append(dict(course_index=i, **matched_old_six_selection(row)))
            old_cases[i] = root / f"course{i:03d}"
    if set(old_cases) != set(indices):
        raise ValueError("matched old six baseline must cover every fixed24 case")
    assets = [a.world, a.motion_csv] + [
        a.model_root / n
        for n in (
            "planner_sonic.onnx",
            "low_latency/model_encoder.onnx",
            "low_latency/model_decoder.onnx",
            "low_latency/observation_config.yaml",
        )
    ]
    for f in assets:
        if pins.get(str(f)) != digest(f):
            raise ValueError("same physical/source/NN identity required")
    priors = [Path(k) for k in pins if Path(k).name == "entry-affordance.json"]
    if len(priors) != 1:
        raise ValueError("one original affordance required")
    check_pins(pins)
    from scripts.rsi_probe_student_intervention import validate_warm_pair

    validate_warm_pair(a, pins, indices)
    runtime = {
        str(f): digest(f)
        for root in (Path.cwd() / "src", Path.cwd() / "scripts", a.core_source)
        for f in root.rglob("*.py")
    }
    runtime.update({str(f): digest(f) for f in assets})
    pins.update(runtime)
    return (
        spec,
        near,
        pins,
        runtime,
        courses,
        indices,
        old_cases,
        sorted(baseline, key=lambda r: r["course_index"]),
        priors[0],
    )


def source_preflight(a, prior_path):
    """Both feet/whole reference limits; private FK only, no NN/native steps."""
    import mujoco
    import numpy as np

    from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
    from rosclaw_soccer.providers.g1.sonic_reference_preparation import prepare_sonic_reference
    from rosclaw_soccer.providers.g1.sonic_wrist_quarantine import quarantine_wrist_reference
    from rosclaw_soccer.rsi.ball_entry_teacher import SkillEntryAffordance
    from rosclaw_soccer.rsi.bilateral_entry import contact_foot_context
    from rosclaw_soccer.rsi.source_foot_event import source_foot_event
    from scripts.rsi_native_context_recovery_common import digest

    world = mujoco.MjModel.from_binary_path(str(a.world))
    poses = np.loadtxt(a.motion_csv, delimiter=",", skiprows=1, dtype=np.float64)
    stored = json.loads(prior_path.read_text())
    prior = SkillEntryAffordance(
        tuple(stored["ball_offset_xy_m"]),
        stored["historical_frame"],
        stored["historical_forward_speed_mps"],
        stored["historical_artifact_hash"],
    )
    if prior.contract_hash != stored["contract_hash"]:
        raise ValueError("original affordance contract changed")
    with np.load(a.warm / "course021/actual-student30.npz", allow_pickle=False) as z:
        entry = z["target_body_canonical_qpos_after_substep"][-1, -1, :36].copy()
    limits = world.jnt_range[[world.joint(n).id for n in G1_DDS_JOINT_NAMES]]
    evidence = []
    for side in ("right", "left"):
        source, _, _ = contact_foot_context(poses, prior, side)
        held, _ = quarantine_wrist_reference(source, entry)
        event = source_foot_event(world, held)
        if event.interval != 526 or event.foot != side:
            raise ValueError("declared source forward-foot event/side changed")
        for start in (286, 466):
            prepared = prepare_sonic_reference(
                held[start:767],
                entry,
                source_frequency_hz=120,
                entry_intervals=50,
                source_evidence_hash=digest(a.motion_csv),
            )
            data = np.asarray(prepared.reference.poses)
            excess = float(
                max(0, np.maximum(limits[:, 0] - data[:, 7:], data[:, 7:] - limits[:, 1]).max())
            )
            if excess > 1e-7 or data[0].tobytes() != entry.tobytes():
                raise ValueError("all named29 source/bridge native limits and exact entry required")
            evidence.append(
                dict(
                    side=side,
                    source_window=[start, 767],
                    event=event.__dict__,
                    prepared_frames=len(data),
                    maximum_joint_limit_excess_rad=excess,
                    measured_entry_byte_exact=True,
                    reference_hash=prepared.reference.contract_hash,
                    dynamic_feasibility_proven=False,
                )
            )
    return world, poses, prior, evidence


def load_student(spec, near, world_path):
    import torch

    from rosclaw_soccer.providers.g1.sonic_runup import G1SonicRunupConfig, G1SonicRunupController
    from rosclaw_soccer.rsi.categorical_motion import create_categorical_motion
    from rosclaw_soccer.rsi.optimizer_lineage_audit import exact_tensor_tree
    from scripts.rsi_native_context_recovery_common import digest
    from scripts.rsi_online_kinematic_motion import load_model

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    roster = near["roster"]
    parent = load_model(spec, Path(roster[0]["path"])).eval().requires_grad_(False)
    before = {k: v.clone() for k, v in parent.state_dict().items()}
    student = create_categorical_motion(parent)
    cp = torch.load(roster[3]["path"], map_location="cpu", weights_only=True)
    if (
        cp["model_kind"] != "categorical_fsq32_motion64"
        or cp["anchor_hash"] != roster[0]["hash"]
        or cp["world_hash"] != digest(world_path)
    ):
        raise ValueError("same frozen categorical43 world/parent required")
    student.load_state_dict(cp["state_dict"], strict=True)
    for k, v in parent.state_dict().items():
        exact_tensor_tree(v, before[k], path=k)
    student.eval().requires_grad_(False)
    muscle = G1SonicRunupController(
        Path(spec["model_root"]),
        G1SonicRunupConfig(model_variant="low_latency"),
        inference_threads=1,
    )
    return student, muscle
