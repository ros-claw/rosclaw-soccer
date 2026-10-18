"""Once-only diagnostic succession after all four qualified release ablations.

No candidate generation, NN, retry, budget extension or policy activation.
Calls two independently frozen diagnostic programs, preserving all failures.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_atomic_artifacts import write_once

JOBS = tuple(f"release-zero-{p}" for p in ("raw-8", "raw-56", "context-0", "context-1"))


def expected_jobs(config):
    flag = config.get("use_contact_exit_torque_mpc", False)
    directional = config.get("use_directional_contact_compliance", False)
    surface = config.get("use_mesh_surface_swing", False)
    if (
        any(type(x) is not bool for x in (flag, directional, surface))
        or sum((flag, directional, surface)) > 1
    ):
        raise ValueError("explicit boolean contact-exit diagnostic required")
    if surface:
        return tuple(
            f"surface-{p}" for p in ("raw-8", "raw-56", "context-0", "context-1", "raw-4", "raw-32")
        )
    if directional:
        return tuple(f"directional-exit-{p}" for p in ("raw-8", "raw-56", "context-0", "context-1"))
    return (
        tuple(f"contact-exit-{p}" for p in ("raw-8", "raw-56", "context-0", "context-1"))
        if flag
        else JOBS
    )


def digest(path):
    with Path(path).open("rb") as stream:
        return "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()


def require_terminal(unit):
    status = subprocess.check_output(
        [
            "systemctl",
            "--user",
            "show",
            unit,
            "-p",
            "MainPID",
            "-p",
            "LoadState",
            "-p",
            "ActiveState",
            "-p",
            "Result",
        ],
        text=True,
    )
    if set(status.splitlines()) != {
        "MainPID=0",
        "LoadState=loaded",
        "ActiveState=inactive",
        "Result=success",
    }:
        raise ValueError("both real loaded processes must terminate successfully")


def require_four_endpoints(config, declaration, complete, readback):
    jobs = expected_jobs(config)
    torque_mpc = config.get("use_contact_exit_torque_mpc", False)
    directional = config.get("use_directional_contact_compliance", False)
    surface = config.get("use_mesh_surface_swing", False)
    all_jobs = ("surface-off-raw-8", *jobs) if surface else jobs
    body_count = len(all_jobs)
    kind = (
        "mesh_surface_swing"
        if surface
        else "directional_contact_compliance"
        if directional
        else "contact_exit_torque_mpc"
        if torque_mpc
        else "swing_release_ablation"
    )
    flags = [
        "whole_input_and_witness_files_verified",
        "shared_payload_logical_seals_rebuilt",
        "prior_independent_replays_authenticated",
        "all_original_pre_activation_prefixes_bit_equal"
        if surface
        else "all_original_first_contact_prefixes_bit_equal",
    ]
    if surface:
        flags.append("disabled_complete_300_frame_noninterference_verified")
    if not (torque_mpc or directional or surface):
        flags.append("actual_postcontact_taskspace_increment_exactly_zero")
    if (
        config.get("mode") != "SIM_ONLY"
        or any(
            config.get(k) is not False
            for k in ("hardware_authorized", "promotion_authorized", "neural_training_authorized")
        )
        or config.get("max_attempts") != body_count
        or config.get("new_physical_budget") != body_count
        or config.get("suppress_postcontact_swing_release")
        is not (not (torque_mpc or directional or surface))
        or complete.get("schema") != f"soccer.rsi.{kind}_pilot_complete.v1"
        or readback.get("schema") != f"soccer.rsi.{kind}_terminal_readback.v1"
        or complete.get("declaration_hash") != declaration["report_hash"]
        or readback.get("complete_hash") != complete["report_hash"]
        or readback.get("source_native_bodies") != body_count
        or complete.get("native_substeps") != 3000 * body_count
        or complete.get("independent_cpu_substeps") != 3000 * body_count
        or complete.get("new_optimizer_updates") != 0
        or tuple(j["job"] for j in declaration["jobs"]) != all_jobs
        or tuple(r["job"] for r in complete["results"]) != all_jobs
        or tuple(r["job"] for r in readback["rows"]) != all_jobs
        or any(readback.get(k) is not True for k in flags)
        or any(
            readback.get(k) is not False
            for k in (
                "teacher_qualified",
                "hardware_authorized",
                "promotion_authorized",
            )
        )
        or [r["receipt_hash"] for r in complete["results"]]
        != [r["receipt_hash"] for r in readback["rows"]]
        or [r["outcome"] for r in complete["results"]] != [r["outcome"] for r in readback["rows"]]
    ):
        raise ValueError(
            "all four fixed outcomes, including failures, need their whole terminal audit"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    surface_profile = cfg.get("diagnostic_profile", "legacy-four") == "surface-six"
    budget = 18000 if surface_profile else 12000
    if cfg.get("diagnostic_profile", "legacy-four") not in ("legacy-four", "surface-six"):
        raise ValueError("explicit fixed diagnostic stage profile required")
    if time.time() >= cfg["deadline_unix"] or cfg["new_substep_budget"] != budget:
        raise ValueError("original bounded 12000-step diagnostic budget and deadline required")
    start = Path(cfg["started_path"])
    if any(
        Path(cfg[k]).exists()
        for k in (
            "started_path",
            "complete_path",
            "replay_config",
            "replay_output_root",
            "metrics_output",
        )
    ):
        raise ValueError("unused once-only stage artifacts required")
    collected_handoff = None
    if cfg.get("collected_unit_journal") is not None:
        from rosclaw_soccer.rsi.collected_sim_stage_evidence import verify_collected_sim_journal

        path = Path(cfg["collected_unit_journal"])
        if digest(path) != cfg["collected_unit_journal_hash"]:
            raise ValueError("exact collected-worker journal bytes required")
        status = subprocess.check_output(
            [
                "systemctl",
                "--user",
                "show",
                cfg["producer_unit"],
                "-p",
                "LoadState",
                "-p",
                "ActiveState",
                "-p",
                "MainPID",
            ],
            text=True,
        )
        collected_handoff = verify_collected_sim_journal(
            json.loads(path.read_text()),
            status=dict(line.split("=", 1) for line in status.splitlines()),
            unit=cfg["producer_unit"],
            expected_command=[
                sys.executable,
                "-u",
                "scripts/rsi_swing_release_ablation_pilot.py",
                "--config",
                cfg["producer_config"],
            ],
            terminal_phase="CONTACT_EXIT_TORQUE_MPC_COMPLETE_NOT_QUALIFIED_TEACHER",
        )
    else:
        require_terminal(cfg["producer_unit"])
    require_terminal(cfg["audit_unit"])
    pins = {str(args.config): digest(args.config)}
    for name in ("stage", "replay", "metrics"):
        source = Path(cfg[name + "_source"])
        if (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
            != cfg[name + "_head"]
            or subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=source, text=True
            ).strip()
        ):
            raise ValueError("all three distinct diagnostic source checkouts must remain frozen")
        for folder in (source / "src", source / "scripts"):
            pins.update({str(p): digest(p) for p in folder.rglob("*.py")})
    if (
        Path(__file__).resolve()
        != Path(cfg["stage_source"]) / "scripts/rsi_stage_release_contact_replay.py"
    ):
        raise ValueError("stage must execute its own declared source")

    def sealed(path, key="report_hash"):
        value = json.loads(Path(path).read_text())
        if value[key] != hash_json({k: v for k, v in value.items() if k != key}):
            raise ValueError("original document seal changed")
        pins[str(path)] = digest(path)
        return value

    producer = json.loads(Path(cfg["producer_config"]).read_text())
    pins[cfg["producer_config"]] = digest(cfg["producer_config"])
    root = Path(producer["output_root"])
    declaration = sealed(root / "declaration.json")
    complete = sealed(root / "complete.json")
    readback = sealed(Path(cfg["readback"]))
    if collected_handoff is not None:
        if readback.get("collected_sim_stage_handoff") != collected_handoff:
            raise ValueError("whole audit must authenticate the same collected-worker handoff")
        pins[cfg["collected_unit_journal"]] = cfg["collected_unit_journal_hash"]
    require_four_endpoints(producer, declaration, complete, readback)
    if producer.get("use_mesh_surface_swing", False) is not surface_profile:
        raise ValueError("surface diagnostic scope must match original producer study")
    diagnostic_kind = (
        "mesh_surface"
        if surface_profile
        else "directional_compliance"
        if producer.get("use_directional_contact_compliance", False)
        else "contact_exit_mpc"
        if producer.get("use_contact_exit_torque_mpc", False)
        else "release"
    )
    if readback["reader_source_hash"] != digest(cfg["audit_source"]):
        raise ValueError("exact independently frozen terminal reader required")
    pins[cfg["audit_source"]] = digest(cfg["audit_source"])
    cases = []
    for index, job in enumerate(expected_jobs(producer)):
        source_index = index + 1 if surface_profile else index
        path = root / f"receipt-{source_index:04d}.json"
        receipt = sealed(path, "receipt_hash")
        if receipt != complete["results"][source_index]:
            raise ValueError("whole completed study must bind the original receipt bytes")
        cases.append(
            dict(
                case_id=job.replace("-", "_"),
                receipt_path=str(path),
                receipt_file_hash=digest(path),
                receipt_hash=receipt["receipt_hash"],
            )
        )
    replay_cfg = dict(
        source_head=cfg["replay_head"],
        output_root=cfg["replay_output_root"],
        new_replay_budget=len(cases),
        new_substep_budget=budget,
        cases=cases,
        world_hash=producer["mesh_task_contact_world_hash"],
        mujoco_version="3.12.0",
        source_terminal_readback_hash=readback["report_hash"],
        hardware_authorized=False,
        promotion_authorized=False,
        neural_training_authorized=False,
    )
    if surface_profile:
        replay_cfg["replay_profile"] = "surface-six"
    write_once(
        start,
        dict(
            schema=f"soccer.rsi.{diagnostic_kind}_contact_diagnostic_started.v1",
            input_file_hashes=pins,
            upstream_readback_hash=readback["report_hash"],
            automatic_retry_authorized=False,
            new_physical_substep_budget=budget,
        ),
    )
    write_once(Path(cfg["replay_config"]), replay_cfg)

    def child(source_key, script, arguments):
        source = Path(cfg[source_key + "_source"])
        environment = dict(
            os.environ,
            PYTHONPATH=os.pathsep.join(
                (str(source / "src"), str(source), str(Path(producer["core_root"]) / "src"))
            ),
        )
        subprocess.run(
            [sys.executable, str(source / "scripts" / script), *arguments],
            cwd=source,
            env=environment,
            check=True,
            timeout=1200,
        )

    try:
        child("replay", "rsi_replay_first_touch_contacts.py", ["--config", cfg["replay_config"]])
        child(
            "metrics",
            "rsi_read_contact_exit_metrics.py",
            [
                "--replay-complete",
                str(Path(cfg["replay_output_root"]) / "complete.json"),
                "--replay-complete",
                cfg["prior_replay_complete"],
                "--output",
                cfg["metrics_output"],
            ],
        )
        replay = sealed(Path(cfg["replay_output_root"]) / "complete.json")
        metrics = sealed(Path(cfg["metrics_output"]))
        if any(digest(p) != h for p, h in pins.items()):
            raise ValueError("stage/source/input bytes changed")
        result = dict(
            schema=f"soccer.rsi.{diagnostic_kind}_contact_diagnostic_complete.v1",
            upstream_readback_hash=readback["report_hash"],
            replay_complete_hash=replay["report_hash"],
            metrics_hash=metrics["report_hash"],
            new_fixed_control_replay_substeps=budget,
            new_closed_loop_policy_trials=0,
            new_neural_inferences=0,
            new_optimizer_updates=0,
            hardware_authorized=False,
            promotion_authorized=False,
        )
        result["report_hash"] = hash_json(result)
        write_once(Path(cfg["complete_path"]), result)
        print(
            diagnostic_kind.upper(),
            "CONTACT_DIAGNOSTIC_COMPLETE",
            result["report_hash"],
            flush=True,
        )
    except BaseException as error:
        write_once(
            start.with_suffix(".failure.json"),
            dict(
                type=type(error).__name__,
                message=str(error),
                automatic_retry_authorized=False,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
