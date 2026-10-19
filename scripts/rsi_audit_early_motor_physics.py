"""Six own-body IK/recurrent reference audits with new physical integration.

Separate original constructor/compiler, not the producer episode factory.
Foundation outputs rely on the bound independent six-case component proof.
Selected torque schedule is recorded; action search is NOT reconstructed here.
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_audit_early_neural_components import load_inputs
from scripts.rsi_stage_release_contact_replay import digest, require_terminal


def require_scope(cfg):
    if (
        cfg.get("profile") != "six-early-original-motor-physics-v1"
        or cfg.get("mode") != "SIM_ONLY"
        or any(
            type(cfg.get(k)) is not int or cfg[k] != n
            for k, n in (
                ("source_cases", 6),
                ("physical_step_budget", 18000),
                ("motor_decisions", 1800),
                ("recurrent_updates", 1620),
                ("new_optimizer_updates", 0),
                ("new_foundation_inferences", 0),
            )
        )
        or any(
            cfg.get(k) is not False
            for k in (
                "teacher_qualified",
                "hardware_authorized",
                "promotion_authorized",
                "automatic_retry",
                "independent_search_qualification_claimed",
            )
        )
    ):
        raise ValueError("exact whole-six own-body original motor diagnostic required")


def prepare(path):
    cfg = json.loads(path.read_text())
    require_scope(cfg)
    source, root = Path(cfg["source_root"]), Path(cfg["output_root"])
    if (
        Path(__file__).resolve() != source / "scripts/rsi_audit_early_motor_physics.py"
        or root.exists()
        or not time.time() < cfg["deadline_unix"] <= time.time() + 18000
        or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
        != cfg["source_head"]
        or subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True).strip()
    ):
        raise ValueError(
            "exact clean frozen motor auditor, bounded deadline and unused output required"
        )
    _, _, pins, complete, _ = load_inputs(cfg, path, source, root)
    require_terminal(cfg["neural_unit"])

    def sealed(name, expected=None):
        p = Path(name)
        h = digest(p)
        if (
            p.is_symlink()
            or (expected is not None and h != expected)
            or (str(p) in pins and pins[str(p)] != h)
        ):
            raise ValueError("whole ordinary original motor input bytes required")
        pins[str(p)] = h
        d = json.loads(p.read_text())
        if d["report_hash"] != hash_json({k: v for k, v in d.items() if k != "report_hash"}):
            raise ValueError("whole motor input logical seal required")
        return d

    nnroot = Path(cfg["neural_root"])
    nn = sealed(nnroot / "complete.json", cfg["neural_complete_file_hash"])
    nd = sealed(nnroot / "declaration.json")
    if (
        nn["schema"] != "soccer.rsi.six_early_neural_components_complete.v1"
        or nn["declaration_hash"] != nd["report_hash"]
        or nd["producer_complete_hash"] != complete["report_hash"]
        or len(nn["rows"]) != 6
        or nn["executed_encoder_decoder_pairs"] != 1800
        or nn["completed_planner_calls"] != 90
    ):
        raise ValueError(
            "whole six independently reconstructed Foundation/planner components required"
        )
    for name, h in nd["input_file_hashes"].items():
        if (name in pins and pins[name] != h) or digest(name) != h:
            raise ValueError("neural proof source or input changed")
        pins[name] = h
    producer_decl = sealed(Path(cfg["producer_root"]) / "declaration.json")
    stage = sealed(producer_decl["config"]["stage_readback"])
    native = sealed(stage["effective_cohort_spec"]["producer_complete"])
    original_decl = sealed(stage["effective_cohort_spec"]["producer_declaration"])
    cases = []
    for i, (row, parent, neural) in enumerate(
        zip(complete["rows"], native["results"], nn["rows"], strict=True)
    ):
        if (
            neural["index"] != i
            or neural["source_case_hash"] != row["report_hash"]
            or row["source_receipt_hash"] != parent["receipt_hash"]
        ):
            raise ValueError("all six original same ordered source identities required")
        for name in ("report", "commitment", "world", "proof_payload"):
            w = parent["witnesses"][name]
            if digest(w["path"]) != w["sha256"]:
                raise ValueError("original complete motor policy/world/commitment bytes required")
            pins[w["path"]] = w["sha256"]
        cases.append(parent)
    if (
        len(
            {
                (p["witnesses"]["proof_payload"]["path"], p["witnesses"]["proof_payload"]["sha256"])
                for p in cases
            }
        )
        != 1
    ):
        raise ValueError(
            "one fixed original complete mean payload across all six policies required"
        )
    return cfg, root, pins, complete, cases, original_decl["candidate"]["parameters"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    cfg, root, pins, complete, parents, parameters = prepare(args.config)
    if args.preflight:
        print("SIX_EARLY_ORIGINAL_MOTOR_READY_NO_POLICY_ALLOCATION_OR_PHYSICS", flush=True)
        return
    available = next(
        int(line.split()[1]) * 1024
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemAvailable:")
    )
    if available < 56 * 1024**3:
        raise ValueError(
            "56GiB available before full original independent reference construction required"
        )
    root.mkdir()
    d = dict(
        schema="soccer.rsi.six_early_original_motor_declaration.v1",
        config=cfg,
        input_file_hashes=pins,
        producer_complete_hash=complete["report_hash"],
        shared_original_ik_and_residual_algorithms=True,
        producer_episode_factory_used=False,
        complete_shared_payload_restoration_required=True,
        recorded_foundation_targets_consumed=True,
        recorded_torque_offsets_consumed=True,
        teacher_qualified=False,
        hardware_authorized=False,
        promotion_authorized=False,
    )
    d["report_hash"] = hash_json(d)
    write_once(root / "declaration.json", d)
    rows, current, replay, motor = [], None, None, None
    try:
        import mujoco

        from rosclaw_soccer.physics.recorded_early_episode import RecordedEarlyEpisode
        from rosclaw_soccer.providers.g1.sonic_runup import G1SonicRunupController
        from rosclaw_soccer.rsi.bounded_first_touch_control import BoundedFirstTouchControl
        from rosclaw_soccer.rsi.fixed_physical_payload_reader import FixedPhysicalPayloadReader
        from rosclaw_soccer.rsi.foot_heading_control import FootHeadingContactProbe
        from rosclaw_soccer.rsi.json_artifact_io import load_json_artifact
        from rosclaw_soccer.rsi.original_motor_component_audit import OriginalMotorComponentAudit
        from rosclaw_soccer.rsi.sampling_reference_audit import SamplingReferenceAuditCompiler

        print("ALLOCATING_COMPLETE_INDEPENDENT_SAMPLING_REFERENCE", flush=True)
        payload = parents[0]["witnesses"]["proof_payload"]
        reader = FixedPhysicalPayloadReader(
            Path(payload["path"]), expected_file_hash=payload["sha256"]
        )

        def policy_for(parent):
            return reader.load_sealed(Path(parent["witnesses"]["report"]["path"]))[
                "executed_motor_policy"
            ]

        compiler = SamplingReferenceAuditCompiler(policy_for(parents[0]))
        print("INDEPENDENT_REFERENCE_READY", flush=True)
        for i, (row, parent) in enumerate(zip(complete["rows"], parents, strict=True)):
            current, replay, motor = i, None, None
            if time.time() >= cfg["deadline_unix"]:
                raise RuntimeError("fixed motor audit deadline exhausted")
            write_once(
                root / f"case-{i}-started.json", dict(index=i, phase="INDEPENDENT_POLICY_BIND")
            )
            policy = policy_for(parent)
            c = load_json_artifact(Path(parent["witnesses"]["commitment"]["path"]))
            if policy["policy_hash"] != c["motor_policy_hash"]:
                raise ValueError("original committed full policy required")
            decoder = compiler.new_episode(policy)
            model = mujoco.MjModel.from_binary_path(parent["witnesses"]["world"]["path"])
            replay = RecordedEarlyEpisode(
                model,
                world_hash=parent["witnesses"]["world"]["sha256"],
                kp=np.asarray(c["physics"]["canonical_pd_kp"], dtype=np.float64),
                kd=np.asarray(c["physics"]["canonical_pd_kd"], dtype=np.float64),
                course=tuple(c["course"]),
                defaults=np.asarray(G1SonicRunupController.default_angles, dtype=np.float64),
            )
            motor = OriginalMotorComponentAudit(
                model,
                policy=policy,
                decoder=decoder,
                heading=FootHeadingContactProbe(row["job"]["heading_enabled"]),
                profile=BoundedFirstTouchControl(**parameters),
            )
            with np.load(row["arrays"]["path"], allow_pickle=False) as a:
                review = replay.verify(dict(a), motor_audit=motor)
            result = dict(index=i, source_case_hash=row["report_hash"], review=review)
            result["report_hash"] = hash_json(result)
            write_once(root / f"case-{i}-complete.json", result)
            rows.append(result)
            print("ORIGINAL_MOTOR_PHYSICS_CASE_COMPLETE", i, flush=True)
            del policy, decoder, model, motor, replay
            motor = replay = None
        compiler._stable()
        if any(digest(name) != h for name, h in pins.items()):
            raise ValueError("whole independent motor source/inputs changed")
        result = dict(
            schema="soccer.rsi.six_early_original_motor_complete.v1",
            declaration_hash=d["report_hash"],
            rows=rows,
            actual_physical_steps=sum(r["review"]["actual_physical_steps"] for r in rows),
            motor_decisions=sum(r["review"]["motor_review"]["motor_decisions"] for r in rows),
            recurrent_updates=sum(r["review"]["motor_review"]["recurrent_updates"] for r in rows),
            new_optimizer_updates=0,
            new_foundation_inferences=0,
            action_search_reconstructed=False,
            teacher_qualified=False,
            hardware_authorized=False,
            promotion_authorized=False,
        )
        if (
            result["actual_physical_steps"] != 18000
            or result["motor_decisions"] != 1800
            or result["recurrent_updates"] != 1620
        ):
            raise ValueError("whole six bounded motor decisions and physical steps required")
        result["report_hash"] = hash_json(result)
        write_once(root / "complete.json", result)
    except BaseException as error:
        if replay is not None:
            with (root / "failure-state.npz").open("xb") as stream:
                np.savez_compressed(
                    stream, qpos=replay.data.qpos, qvel=replay.data.qvel, ctrl=replay.data.ctrl
                )
        write_once(
            root / "failure.json",
            dict(
                current_case=current,
                completed_cases=len(rows),
                completed_physical_steps=sum(r["review"]["actual_physical_steps"] for r in rows),
                current_attempted_steps=0 if replay is None else replay.attempted_steps,
                current_integrated_steps=0 if replay is None else replay.integrated_steps,
                current_motor_completed_frames=0 if motor is None else motor.next_frame,
                error_type=type(error).__name__,
                error=str(error),
                automatic_retry=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
