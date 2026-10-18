"""Fixed-control diagnostic replays, not new closed-loop policy trials."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import mujoco
import numpy as np

from rosclaw_soccer.physics.ball_contact_replay_diagnostic import ball_contact_samples
from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.rsi.first_touch_outcome import label_first_touch
from rosclaw_soccer.rsi.first_touch_qualification import require_first_touch_qualification
from rosclaw_soccer.sim.contracts import G1_HARD_TORQUE_LIMITS, hash_json
from scripts.rsi_atomic_artifacts import write_once


def digest(path):
    with Path(path).open("rb") as stream:
        return "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()


def fixed_replay_scope(cfg):
    profile = cfg.get("replay_profile", "legacy-four")
    counts = {
        "legacy-four": 4,
        "surface-six": 6,
        "surface-missing-three": 3,
        "demonstration-missing-five": 5,
        "surface-context-six": 6,
        "context-baseline-twelve": 12,
        "catalog2-zero-c7": 1,
        "receding-catalog2-ten": 10,
        "surface-receding-six": 6,
        "positive-lateral-sixteen": 16,
        "reference-student-thirteen": 13,
    }
    if profile not in counts:
        raise ValueError("explicit fixed diagnostic profile required")
    count = counts[profile]
    steps = 3000 * count
    cases = cfg.get("cases")
    if (
        type(cases) is not list
        or len(cases) != count
        or type(cfg.get("new_replay_budget")) is not int
        or cfg["new_replay_budget"] != count
        or type(cfg.get("new_substep_budget")) is not int
        or cfg["new_substep_budget"] != steps
        or len({r["case_id"] for r in cases}) != count
        or len({r["receipt_hash"] for r in cases}) != count
    ):
        raise ValueError("whole fixed diagnostic budget and distinct endpoints required")
    if profile == "surface-six" and [r["case_id"] for r in cases] != [
        "surface_raw_8",
        "surface_raw_56",
        "surface_context_0",
        "surface_context_1",
        "surface_raw_4",
        "surface_raw_32",
    ]:
        raise ValueError("all six original surface endpoints required in declaration order")
    if profile == "surface-missing-three" and [r["case_id"] for r in cases] != [
        "baseline_context_1",
        "baseline_raw_4",
        "baseline_raw_32",
    ]:
        raise ValueError("only the three missing historical baseline endpoints may be replayed")
    if profile == "demonstration-missing-five" and [r["case_id"] for r in cases] != [
        "none_raw_4",
        "none_raw_8",
        "none_raw_32",
        "none_raw_56",
        "none_context_1",
    ]:
        raise ValueError("only five missing None endpoints may be replayed; C0 already exists")
    if profile == "surface-context-six" and [r["case_id"] for r in cases] != [
        f"surface_context_{i}" for i in range(2, 8)
    ]:
        raise ValueError("all six remaining material-point contexts required in original order")
    if profile == "context-baseline-twelve" and [r["case_id"] for r in cases] != [
        f"{family}_context_{i}" for i in range(2, 8) for family in ("none", "option4")
    ]:
        raise ValueError("both original controls for all six remaining contexts required")
    if profile == "catalog2-zero-c7" and [r["case_id"] for r in cases] != [
        "catalog2_zero_context_7"
    ]:
        raise ValueError("only the one exact archived zero-control C7 may be replayed")
    if profile == "receding-catalog2-ten":
        from rosclaw_soccer.rsi.receding_teacher_scope import receding_jobs

        if [r["case_id"] for r in cases] != [j["job"].replace("-", "_") for j in receding_jobs()]:
            raise ValueError("all ten original receding endpoints required in declaration order")
    if profile == "surface-receding-six":
        from rosclaw_soccer.rsi.surface_receding_evidence import surface_receding_jobs

        if [r["case_id"] for r in cases] != [
            j["job"].replace("-", "_") for j in surface_receding_jobs()
        ]:
            raise ValueError("all six original material-point receding endpoints required")
    if profile == "positive-lateral-sixteen" and [r["case_id"] for r in cases] != [
        f"{role}_positive_local_c{i}" for i in range(8) for role in ("control", "teacher")
    ]:
        raise ValueError("all sixteen prospective paired endpoints required in declaration order")
    if profile == "reference-student-thirteen":
        from rosclaw_soccer.rsi.reference_student_exam_scope import exam_scope

        jobs = exam_scope(cfg["behavior_model_hash"])["jobs"]
        if [r["case_id"] for r in cases] != [j["job"].replace("-", "_") for j in jobs]:
            raise ValueError("all thirteen original own-student and retention endpoints required")
    return count, steps


def reference_exam_sources(cfg, sealed, pins):
    """Narrow model intent from a whole original exam, not a receipt-supplied tag."""
    from rosclaw_soccer.rsi.reference_student_exam_diagnostics import require_cohort

    path = Path(cfg["producer_config"])
    if digest(path) != cfg["producer_config_file_hash"]:
        raise ValueError("exact original own-body producer configuration required")
    producer = json.loads(path.read_text())
    pins[str(path)] = digest(path)
    root = Path(producer["output_root"])
    if (
        Path(cfg["producer_complete"]).resolve() != (root / "complete.json").resolve()
        or Path(cfg["producer_declaration"]).resolve() != (root / "declaration.json").resolve()
        or cfg["world_hash"] != producer["mesh_task_contact_world_hash"]
        or cfg["behavior_model_hash"] != producer["behavior_model_hash"]
    ):
        raise ValueError("original owned complete, declaration, world and parent required")
    declaration = sealed(cfg["producer_declaration"])
    complete = sealed(cfg["producer_complete"])
    audit = sealed(cfg["source_terminal_readback"])
    require_cohort(producer, declaration, complete, audit)
    reader = Path(cfg["audit_source"])
    if (
        audit["reader_source_hash"] != digest(reader)
        or audit["report_hash"] != cfg["source_terminal_readback_hash"]
        or reader.resolve()
        != (Path(producer["source_root"]) / "scripts/rsi_read_reference_student_exam.py").resolve()
        or any(
            Path(case["receipt_path"]).resolve() != (root / f"receipt-{i:04d}.json").resolve()
            or case["receipt_hash"] != receipt["receipt_hash"]
            for i, (case, receipt) in enumerate(zip(cfg["cases"], complete["results"], strict=True))
        )
    ):
        raise ValueError("whole original thirteen-case audit and owned ordered receipts required")
    pins[str(reader)] = digest(reader)
    for name, h in {
        **declaration["prepared_pins"],
        **declaration["input_file_hashes"],
        **audit["input_file_hashes"],
    }.items():
        if digest(name) != h:
            raise ValueError("original own-student input, witness or source bytes changed")
        pins[name] = h
    return complete["results"]


def verify_none_demonstration_sources(cfg, receipts, sealed, pins):
    """Authenticate original None endpoints, without promoting an oracle selector.

    The first four are from the complete original-public study; the last is
    index 2 of the complete context/retention screen. Never replace either
    provenance by a self-resealed list of selected passing receipts.
    """
    if len(receipts) != 5:
        raise ValueError("exact five original None receipts required")
    for prefix, indexes, subset, schema, bodies in (
        (
            "nominal",
            (0, 1, 2, 3),
            receipts[:4],
            "soccer.rsi.mesh_original_public_terminal_readback.v1",
            4,
        ),
        (
            "baseline",
            (2,),
            receipts[4:],
            "soccer.rsi.mesh_option4_teacher_screen_terminal_readback.v1",
            18,
        ),
    ):
        audit = sealed(cfg[f"{prefix}_terminal_readback"])
        complete = sealed(cfg[f"{prefix}_complete"])
        reader = Path(cfg[f"{prefix}_reader_source"])
        if (
            audit.get("schema") != schema
            or audit["report_hash"] != cfg[f"{prefix}_terminal_readback_hash"]
            or audit.get("complete_hash") != complete["report_hash"]
            or audit.get("reader_source_hash") != digest(reader)
            or audit.get("source_native_bodies") != bodies
            or len(complete.get("results", [])) != bodies
            or any(
                audit.get(k) is not True
                for k in (
                    "whole_input_and_witness_files_verified",
                    "shared_payload_logical_seals_rebuilt",
                    "prior_independent_replays_authenticated",
                )
            )
            or any(
                receipt != complete["results"][index]
                or not any(
                    row["receipt_hash"] == receipt["receipt_hash"]
                    and row["job"] == receipt["job"]
                    and row["outcome"] == receipt["outcome"]
                    for row in audit["rows"]
                )
                for receipt, index in zip(subset, indexes, strict=True)
            )
        ):
            raise ValueError(
                "all original None endpoints must belong to their whole terminal audit"
            )
        pins[str(reader)] = digest(reader)
    if [r["job"] for r in receipts] != [
        "original-public-4",
        "original-public-8",
        "original-public-32",
        "original-public-56",
        "control-context-1",
    ]:
        raise ValueError("original None course identity required")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    count, substeps = fixed_replay_scope(cfg)
    source = Path(__file__).resolve().parents[1]
    root = Path(cfg["output_root"])
    if (
        root.exists()
        or cfg["mujoco_version"] != mujoco.__version__
        or any(
            cfg.get(k) is not False
            for k in ("hardware_authorized", "promotion_authorized", "neural_training_authorized")
        )
        or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
        != cfg["source_head"]
        or subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True).strip()
        or mujoco.get_mjcb_control() is not None
    ):
        raise ValueError("new frozen four-case same-version SIM diagnostic required")
    pins = {str(args.config): digest(args.config)}
    for relative in (
        "scripts/rsi_replay_first_touch_contacts.py",
        "src/rosclaw_soccer/physics/ball_contact_replay_diagnostic.py",
        "src/rosclaw_soccer/rsi/first_touch_outcome.py",
        "src/rosclaw_soccer/rsi/first_touch_qualification.py",
    ):
        pins[str(source / relative)] = digest(source / relative)
    receipts = []

    def sealed(path, key="report_hash"):
        value = json.loads(Path(path).read_text())
        if value[key] != hash_json({k: v for k, v in value.items() if k != key}):
            raise ValueError("source seal differs")
        pins[str(path)] = digest(path)
        return value

    if cfg.get("replay_profile") == "reference-student-thirteen":
        for relative in (
            "src/rosclaw_soccer/rsi/reference_student_exam_diagnostics.py",
            "src/rosclaw_soccer/rsi/reference_student_exam_scope.py",
            "src/rosclaw_soccer/rsi/reference_student_execution.py",
            "scripts/rsi_reference_student_exam.py",
            "scripts/rsi_fit_pre_parent_reference.py",
        ):
            pins[str(source / relative)] = digest(source / relative)
    reference = (
        reference_exam_sources(cfg, sealed, pins)
        if cfg.get("replay_profile") == "reference-student-thirteen"
        else None
    )
    archived = None
    if cfg.get("replay_profile") == "catalog2-zero-c7":
        archived = sealed(cfg["archived_expert_readback"])
        reader = Path(cfg["archived_expert_reader_source"])
        if (
            archived["report_hash"] != cfg["archived_expert_readback_hash"]
            or archived.get("schema") != "soccer.rsi.archived_receding_expert_readback.v1"
            or archived.get("recovered_catalog_index") != 2
            or archived.get("actual_all3000_cerebellar_outputs_exactly_zero") is not True
            or archived.get("exact_initial_canonical_qpos_qvel_equal") is not True
            or archived.get("single_parameter_speed_ablation") is not False
            or archived.get("teacher_qualified") is not False
            or archived.get("training_authorized") is not False
            or archived["input_file_hashes"].get(str(reader)) != digest(reader)
        ):
            raise ValueError("exact quarantined archived C7 readback and source required")
        for name, h in archived["input_file_hashes"].items():
            if digest(name) != h:
                raise ValueError("original archived expert evidence bytes changed")
            pins[name] = h
    for case in cfg["cases"]:
        receipt = sealed(case["receipt_path"], "receipt_hash")
        if (
            pins[case["receipt_path"]] != case["receipt_file_hash"]
            or receipt["receipt_hash"] != case["receipt_hash"]
            or receipt["witnesses"]["world"]["sha256"] != cfg["world_hash"]
        ):
            raise ValueError("exact original receipt and compiled world required")
        for witness in receipt["witnesses"].values():
            if digest(witness["path"]) != witness["sha256"]:
                raise ValueError("whole original witness bytes changed")
            pins[witness["path"]] = witness["sha256"]
        with np.load(receipt["witnesses"]["trace"]["path"], allow_pickle=False) as archive:
            trace = dict(archive)
        qualification_kwargs = {}
        if archived is not None:
            from scripts import rsi_recover_archived_receding_expert as helper

            if (
                digest(helper.__file__) != digest(cfg["archived_expert_reader_source"])
                or receipt["receipt_hash"] != archived["recovered_receipt_hash"]
                or receipt["outcome"] != archived["recovered_outcome"]
            ):
                raise ValueError(
                    "only exact previously authenticated zero expert may use this profile"
                )
            pins[helper.__file__] = digest(helper.__file__)
            helper.require_actual_zero_control(receipt, trace)
            qualification_kwargs = dict(
                cerebellar_model_hash=receipt["cerebellar_model_hash"],
                cerebellar_sampling_draws=0,
            )
        if reference is not None:
            if receipt != reference[len(receipts)]:
                raise ValueError("only the exact whole-audited reference receipt may be replayed")
            qualification_kwargs = dict(
                reference_student_model_hash=receipt.get("reference_student_model_hash")
            )
        outcome = label_first_touch(trace)
        review = sealed(receipt["witnesses"]["independent_review"]["path"])
        foundation = sealed(receipt["witnesses"]["foundation_review"]["path"])
        if (
            outcome.to_dict() != receipt["outcome"]
            or require_first_touch_qualification(
                review, foundation, outcome, **qualification_kwargs
            )
            != receipt["qualification_alignment"]
        ):
            raise ValueError("original primitive and replay qualifications differ")
        receipts.append(receipt)
    if cfg.get("replay_profile") in ("surface-missing-three", "context-baseline-twelve"):
        baseline_audit = sealed(cfg["baseline_terminal_readback"])
        baseline_complete = sealed(cfg["baseline_complete"])
        baseline_reader = Path(cfg["baseline_reader_source"])
        if (
            baseline_audit["report_hash"] != cfg["baseline_terminal_readback_hash"]
            or baseline_audit.get("complete_hash") != baseline_complete["report_hash"]
            or baseline_audit.get("reader_source_hash") != digest(baseline_reader)
            or baseline_audit.get("source_native_bodies") != 18
            or any(
                baseline_audit.get(k) is not True
                for k in (
                    "whole_input_and_witness_files_verified",
                    "shared_payload_logical_seals_rebuilt",
                    "prior_independent_replays_authenticated",
                )
            )
            or len(baseline_complete.get("results", [])) != 18
            or any(
                receipt != baseline_complete["results"][index]
                or not any(
                    row["receipt_hash"] == receipt["receipt_hash"]
                    and row["job"] == receipt["job"]
                    and row["outcome"] == receipt["outcome"]
                    for row in baseline_audit["rows"]
                )
                for receipt, index in zip(
                    receipts,
                    range(4, 16)
                    if cfg["replay_profile"] == "context-baseline-twelve"
                    else (3, 16, 17),
                    strict=True,
                )
            )
        ):
            raise ValueError("all exact original endpoints must belong to complete baseline audit")
        pins[str(baseline_reader)] = digest(baseline_reader)
    if cfg.get("replay_profile") == "demonstration-missing-five":
        verify_none_demonstration_sources(cfg, receipts, sealed, pins)
    if cfg.get("replay_profile") in (
        "receding-catalog2-ten",
        "surface-receding-six",
        "positive-lateral-sixteen",
    ):
        from scripts.rsi_stage_receding_contact_replay import require_receding_endpoints

        producer = json.loads(Path(cfg["producer_config"]).read_text())
        pins[cfg["producer_config"]] = digest(cfg["producer_config"])
        complete = sealed(cfg["producer_complete"])
        declaration = sealed(cfg["producer_declaration"])
        audit = sealed(cfg["source_terminal_readback"])
        if cfg["replay_profile"] == "positive-lateral-sixteen":
            from rosclaw_soccer.rsi.positive_lateral_diagnostics import require_cohort
            from scripts.rsi_positive_lateral_physical_pilot import whole_predecessor

            require_cohort(producer, declaration, complete, audit)
            whole_predecessor(producer, Path(producer["source_root"]), sealed, pins)
            for relative in (
                "src/rosclaw_soccer/rsi/positive_lateral_diagnostics.py",
                "src/rosclaw_soccer/rsi/positive_lateral_episode_inputs.py",
                "src/rosclaw_soccer/rsi/positive_lateral_receding_scope.py",
                "scripts/rsi_positive_lateral_physical_pilot.py",
            ):
                pins[str(source / relative)] = digest(source / relative)
        elif cfg["replay_profile"] == "surface-receding-six":
            from rosclaw_soccer.rsi.surface_receding_diagnostics import require_endpoints
            from rosclaw_soccer.rsi.surface_receding_evidence import load_baselines

            require_endpoints(producer, declaration, complete, audit)
            load_baselines(producer, Path(producer["source_root"]), sealed, digest, pins)
        else:
            require_receding_endpoints(producer, declaration, complete, audit)
        reader = Path(cfg["audit_source"])
        if (
            audit["reader_source_hash"] != digest(reader)
            or audit["report_hash"] != cfg["source_terminal_readback_hash"]
            or receipts != complete["results"]
        ):
            raise ValueError("whole receding audit and original receipt membership required")
        pins[str(reader)] = digest(reader)
        for name, h in declaration["input_file_hashes"].items():
            if digest(name) != h:
                raise ValueError("whole receding audit input bytes changed")
            pins[name] = h
        for relative in (
            "scripts/rsi_stage_receding_contact_replay.py",
            "src/rosclaw_soccer/rsi/receding_teacher_scope.py",
        ):
            pins[str(source / relative)] = digest(source / relative)
        if cfg["replay_profile"] == "surface-receding-six":
            for relative in (
                "src/rosclaw_soccer/rsi/surface_receding_diagnostics.py",
                "src/rosclaw_soccer/rsi/surface_receding_evidence.py",
                "src/rosclaw_soccer/rsi/public_course_cluster_accounting.py",
            ):
                pins[str(source / relative)] = digest(source / relative)
    if args.preflight:
        print(
            "FIXED_REPLAYS",
            count,
            substeps,
            "SUBSTEPS_PREFLIGHT_ZERO_MODEL_ALLOCATIONS",
            flush=True,
        )
        return
    root.mkdir()
    declaration = dict(
        cfg,
        input_file_hashes=pins,
        source_control_clock="original recorded torque every 2ms",
        new_training_episodes=0,
        neural_inference_repeated=False,
        new_closed_loop_policy_trials=0,
    )
    declaration["report_hash"] = hash_json(declaration)
    write_once(root / "declaration.json", declaration)
    attempted, integrated = 0, 0
    current_case = None
    rows = []
    model = mujoco.MjModel.from_binary_path(receipts[0]["witnesses"]["world"]["path"])
    if (
        (model.nq, model.nv, model.nu) != (43, 41, 29)
        or model.opt.timestep != 0.002
        or model.opt.integrator != mujoco.mjtIntegrator.mjINT_EULER
        or model.nplugin
    ):
        raise ValueError("unchanged 2ms Euler canonical G1 model required")
    qi = [int(model.jnt_qposadr[model.joint(n).id]) for n in G1_DDS_JOINT_NAMES]
    vi = [int(model.jnt_dofadr[model.joint(n).id]) for n in G1_DDS_JOINT_NAMES]
    joints = [int(model.joint(n).id) for n in G1_DDS_JOINT_NAMES]
    ai = []
    for joint in joints:
        ids = np.flatnonzero(
            (model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT)
            & (model.actuator_trnid[:, 0] == joint)
        )
        if len(ids) != 1:
            raise ValueError("one original actuator per canonical joint required")
        ai.append(int(ids[0]))
    ball = int(model.body("ball").id)
    bg = int(model.geom("ball_geom").id)
    bj = int(model.body_jntadr[ball])
    bq, bv = int(model.jnt_qposadr[bj]), int(model.jnt_dofadr[bj])
    if (
        model.geom_size[bg, 0] != 0.11
        or model.body_mass[ball] != 0.43
        or np.any(model.dof_damping[bv : bv + 6] != 0)
    ):
        raise ValueError("unchanged ball dimensions/mass/damping required")
    body_names = (
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_ankle_pitch_link",
        "right_ankle_pitch_link",
        "left_knee_link",
        "right_knee_link",
    )
    bodies = [int(model.body(n).id) for n in body_names]
    pelvis = int(model.body("pelvis").id)

    def equal(a, b, name):
        if not np.allclose(a, b, rtol=0, atol=1e-8):
            raise ValueError("fixed-control physics replay differs: " + name)

    try:
        for case, receipt in zip(cfg["cases"], receipts, strict=True):
            current_case = case["case_id"]
            write_once(
                root / f"{current_case}-started.json",
                dict(
                    case_id=current_case,
                    receipt_hash=receipt["receipt_hash"],
                    replay_attempt_charged=True,
                ),
            )
            with np.load(receipt["witnesses"]["trace"]["path"], allow_pickle=False) as archive:
                trace = dict(archive)
            q, v, tau = trace["canonical_qpos"], trace["canonical_qvel"], trace["torque_nm"]
            if any(
                a.shape != shape or a.dtype != np.float64 or not np.isfinite(a).all()
                for a, shape in ((q, (300, 1, 43)), (v, (300, 1, 41)), (tau, (300, 1, 10, 29)))
            ):
                raise ValueError("complete finite original state/control arrays required")
            if np.any(np.abs(tau) > np.asarray(G1_HARD_TORQUE_LIMITS) + 1e-8):
                raise ValueError("recorded control exceeds original hard torque limits")
            data = mujoco.MjData(model)
            data.qpos[:7], data.qpos[qi], data.qpos[bq : bq + 7] = (
                q[0, 0, :7],
                q[0, 0, 7:36],
                q[0, 0, 36:43],
            )
            data.qvel[:6], data.qvel[vi], data.qvel[bv : bv + 6] = (
                v[0, 0, :6],
                v[0, 0, 6:35],
                v[0, 0, 35:41],
            )
            mujoco.mj_forward(model, data)
            events = []
            forces = np.zeros((3000, 3))
            robot_forces = np.zeros_like(forces)
            poses = np.zeros((3000, 43))
            velocities = np.zeros((3000, 41))
            velocities_after = np.zeros_like(velocities)
            maximum_force_balance_error = 0.0
            for frame in range(300):
                equal(
                    np.concatenate((data.qpos[:7], data.qpos[qi], data.qpos[bq : bq + 7])),
                    q[frame, 0],
                    "frame qpos",
                )
                equal(
                    np.concatenate((data.qvel[:6], data.qvel[vi], data.qvel[bv : bv + 6])),
                    v[frame, 0],
                    "frame qvel",
                )
                peaks = np.zeros(6)
                for substep in range(10):
                    step = frame * 10 + substep
                    poses[step] = np.concatenate(
                        (data.qpos[:7], data.qpos[qi], data.qpos[bq : bq + 7])
                    )
                    velocities[step] = np.concatenate(
                        (data.qvel[:6], data.qvel[vi], data.qvel[bv : bv + 6])
                    )
                    before = data.qvel.copy()
                    data.ctrl[ai] = tau[frame, 0, substep]
                    attempted += 1
                    mujoco.mj_step(model, data)
                    integrated += 1
                    velocities_after[step] = np.concatenate(
                        (data.qvel[:6], data.qvel[vi], data.qvel[bv : bv + 6])
                    )
                    equal(
                        data.actuator_force[ai],
                        trace["actual_actuator_force_nm"][frame, 0, substep],
                        "actual actuator force",
                    )
                    equal(
                        data.xpos[pelvis, 2],
                        trace["pelvis_z_per_substep_m"][frame, 0, substep],
                        "pelvis height",
                    )
                    contacts = ball_contact_samples(model, data, bg, before)
                    for contact in contacts:
                        force = np.asarray(contact["force_on_ball_world_n"])
                        forces[step] += force
                        body = contact["other_body"]
                        if body != 0:
                            robot_forces[step] += force
                        if body in bodies:
                            channel = bodies.index(body)
                            peaks[channel] = max(peaks[channel], contact["force_norm_n"])
                        if body != 0 and contact["force_norm_n"] > 1.0:
                            events.append(
                                dict(
                                    contact,
                                    substep_index=step,
                                    frame=frame,
                                    ball_center_velocity_before_world_m_s=before[
                                        bv : bv + 3
                                    ].tolist(),
                                    other_body_origin_world_m=data.xpos[body].tolist(),
                                )
                            )
                    error = float(np.max(np.abs(forces[step] - data.qfrc_constraint[bv : bv + 3])))
                    maximum_force_balance_error = max(maximum_force_balance_error, error)
                    if error > 1e-7:
                        raise ValueError(
                            "contact world-force convention differs from ball generalized force"
                        )
                equal(peaks, trace["force_n"][frame, 0], "all six contact force channels")
                equal(
                    data.xpos[ball],
                    trace["ball_position_after_step_m"][frame, 0],
                    "ball after frame",
                )
            arrays_path = root / f"{current_case}-substeps.npz"
            with arrays_path.open("xb") as stream:
                np.savez_compressed(
                    stream,
                    canonical_qpos_before_substep=poses,
                    canonical_qvel_before_substep=velocities,
                    canonical_qvel_after_substep=velocities_after,
                    all_contact_force_on_ball_world_n=forces,
                    robot_contact_force_on_ball_world_n=robot_forces,
                )
            foot_events = [e for e in events if e["other_body"] in bodies[:2]]
            first = None if not foot_events else foot_events[0]["substep_index"]
            window = None
            if first is not None:
                end = min(first + 10, 3000)
                window = dict(
                    first_substep=first,
                    last_substep_inclusive=end - 1,
                    robot_contact_impulse_world_n_s=(
                        robot_forces[first:end].sum(axis=0) * 0.002
                    ).tolist(),
                    ball_center_velocity_before_world_m_s=velocities[first, 35:38].tolist(),
                    ball_center_velocity_after_world_m_s=velocities_after[end - 1, 35:38].tolist(),
                )
            result = dict(
                case_id=current_case,
                source_receipt_hash=receipt["receipt_hash"],
                original_outcome=receipt["outcome"],
                new_replay_substeps=3000,
                first_foot_contact_event=None if not foot_events else foot_events[0],
                first_foot_20ms_window=window,
                robot_contact_events_over_1n=events,
                unmonitored_robot_contact_events_over_1n=sum(
                    e["other_body"] not in bodies for e in events
                ),
                maximum_world_contact_force_balance_error_n=maximum_force_balance_error,
                arrays=dict(path=str(arrays_path), sha256=digest(arrays_path)),
                new_closed_loop_policy_trial=False,
                hardware_authorized=False,
                promotion_authorized=False,
            )
            result["report_hash"] = hash_json(result)
            write_once(root / f"{current_case}-complete.json", result)
            rows.append(result)
            print("CONTACT_REPLAY_COMPLETE", current_case, flush=True)
        if any(digest(path) != h for path, h in pins.items()):
            raise ValueError("original source/input bytes changed")
        complete = dict(
            schema="soccer.rsi.fixed_control_contact_replay_complete.v1",
            declaration_hash=declaration["report_hash"],
            rows=rows,
            attempted_step_calls=attempted,
            integrated_substeps=integrated,
            original_closed_loop_policy_reexecuted=False,
            neural_inference_repeated=False,
            shared_payload_logical_seals_recomputed_here=False,
            contact_kinematic_velocity_clock=(
                "saved before mj_step; Jacobians at that step's collision stage"
            ),
            new_training_episodes=0,
            new_optimizer_updates=0,
            new_independent_courses=0,
            hardware_authorized=False,
            promotion_authorized=False,
        )
        complete["report_hash"] = hash_json(complete)
        write_once(root / "complete.json", complete)
        print(complete["report_hash"], flush=True)
    except BaseException as error:
        write_once(
            root / "fatal.json",
            dict(
                type=type(error).__name__,
                message=str(error),
                case_id=current_case,
                attempted_step_calls=attempted,
                integrated_substeps=integrated,
                automatic_retry_authorized=False,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
