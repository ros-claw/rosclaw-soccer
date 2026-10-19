"""Whole authenticated pulse stage diagnostics; no physics, fitting or admission."""

import argparse
import json
import subprocess
from pathlib import Path

from rosclaw_soccer.rsi.first_touch_stage_diagnostics import assess_first_touch_stages
from rosclaw_soccer.rsi.original_ik_planner_diagnostics import ik_contact_readout
from rosclaw_soccer.rsi.planner_navigation_pulse_cohort import pulse_contact_readout
from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_load_original_ik_cohort import load_closed_original_ik
from scripts.rsi_load_pre_planner_cohort import load_closed_pulse
from scripts.rsi_stage_release_contact_replay import digest, require_terminal


def completed_inputs(cfg, *, sealed):
    """Bind unknown future hashes only after all fixed original units terminate.

    This creates no policy or gate approval. Whole source, journal, receipt,
    physics/audit and dense-result checks still run on the effective inputs.
    """
    bind = cfg.get("bind_completed_source_hashes", False)
    if type(bind) is not bool:
        raise ValueError("explicit completed-source binding flag required")
    cohort, contact = dict(cfg["cohort"]), dict(cfg["contact"])
    if bind:
        if "source_terminal_readback_hash" in cohort or "complete_hash" in contact:
            raise ValueError("completed-source binding cannot override predeclared hash pins")
        for unit in (cohort["producer_unit"], cohort["audit_unit"], contact["unit"]):
            require_terminal(unit)
        cohort["source_terminal_readback_hash"] = sealed(cohort["source_terminal_readback"])[
            "report_hash"
        ]
        contact["complete_hash"] = sealed(contact["complete"])["report_hash"]
    return cohort, contact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    stage_profile = cfg.get("stage_profile", "zero-reference-pulse")
    source = Path(__file__).resolve().parents[1]
    output = Path(cfg["output"])
    started = output.with_suffix(".started.json")
    if (
        cfg.get("mode") != "SIM_ONLY"
        or stage_profile not in ("zero-reference-pulse", "original-ik-pulse")
        or any(cfg.get(k) is not False for k in ("hardware_authorized", "promotion_authorized"))
        or output.exists()
        or started.exists()
        or Path(cfg["source_root"]).resolve() != source
        or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
        != cfg["source_head"]
        or subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True).strip()
    ):
        raise ValueError("unused frozen SIM-only stage diagnostic required")
    pins = {str(args.config): digest(args.config)}
    for folder in (source / "src", source / "scripts"):
        pins.update({str(p): digest(p) for p in folder.rglob("*.py")})
    write_once(
        started,
        dict(
            schema="soccer.rsi.first_touch_stage_readback_started.v1",
            input_file_hashes=dict(pins),
            automatic_retry_authorized=False,
            new_physical_steps=0,
            hardware_authorized=False,
        ),
    )

    def sealed(path, key="report_hash"):
        path = Path(path)
        value = json.loads(path.read_text())
        if value[key] != hash_json({k: v for k, v in value.items() if k != key}):
            raise ValueError("original evidence seal required")
        h = digest(path)
        if str(path) in pins and pins[str(path)] != h:
            raise ValueError("source/evidence changed since authentication")
        pins[str(path)] = h
        return value

    try:
        cohort_spec, contact_spec = completed_inputs(cfg, sealed=sealed)
        loader = (
            load_closed_original_ik if stage_profile == "original-ik-pulse" else load_closed_pulse
        )
        cohort = loader(cohort_spec, sealed=sealed, pins=pins)
        declaration, complete, audit = (cohort[k] for k in ("declaration", "complete", "audit"))
        spec = contact_spec
        require_terminal(spec["unit"])
        contact, replay, metrics = (
            sealed(spec[k]) for k in ("complete", "replay_complete", "metrics")
        )
        readout = (
            ik_contact_readout if stage_profile == "original-ik-pulse" else pulse_contact_readout
        )
        measured = readout(declaration, complete, replay, metrics)
        contact_schema = (
            "soccer.rsi.original_ik_planner_contact_complete.v1"
            if stage_profile == "original-ik-pulse"
            else "soccer.rsi.pre_planner_pulse_contact_complete.v1"
        )
        if (
            contact.get("schema") != contact_schema
            or contact["report_hash"] != spec["complete_hash"]
            or contact["upstream_readback_hash"] != audit["report_hash"]
            or contact["replay_complete_hash"] != replay["report_hash"]
            or contact["metrics_hash"] != metrics["report_hash"]
            or contact["measured_contact_readout"] != measured
            or any(
                type(contact.get(k)) is not int or contact[k] != n
                for k, n in (
                    ("new_fixed_control_replay_substeps", 18000),
                    ("new_closed_loop_policy_trials", 0),
                    ("new_neural_inferences", 0),
                    ("new_optimizer_updates", 0),
                    ("total_metrics_endpoints", 6),
                )
            )
            or any(
                contact.get(k) is not False
                for k in (
                    "teacher_qualified",
                    "dataset_admission_authorized",
                    "neural_training_authorized",
                    "hardware_authorized",
                    "promotion_authorized",
                )
            )
        ):
            raise ValueError("whole original dense and motion cohort evidence required")
        for name, h in contact["input_file_hashes"].items():
            if digest(name) != h or (name in pins and pins[name] != h):
                raise ValueError("whole original contact source and evidence bytes changed")
            pins[name] = h
        roles = (
            ("ik_pulse_off", "ik_pulse")
            if stage_profile == "original-ik-pulse"
            else ("pulse_off", "pulse")
        )
        groups = [str(j["group"]) for j in declaration["scope"]["jobs"] if j["role"] == roles[0]]
        outcomes = {
            role: {str(r["group"]): r["outcome"] for r in complete["results"] if r["role"] == role}
            for role in roles
        }
        stages = assess_first_touch_stages(
            outcomes[roles[0]], outcomes[roles[1]], declared_pair_ids=groups
        )
        original_pairs = (
            complete["assessment"]["stages"]["original_m0_pairs"]
            if stage_profile == "original-ik-pulse"
            else complete["assessment"]["paired_results"]
        )
        if any(
            stages["original_m0_pairs"][k] != original_pairs[k]
            for k in (
                "pairs",
                "baseline_passes",
                "evaluated_passes",
                "rescued_pairs",
                "regressed_pairs",
                "newly_unsafe_pairs",
                "net_success_gain",
            )
        ):
            raise ValueError("stage diagnostics cannot change original whole-cohort M0 counts")
        if any(digest(p) != h for p, h in pins.items()):
            raise ValueError("source or input bytes changed during stage readback")
        result = dict(
            schema="soccer.rsi.original_ik_first_touch_stage_pairs_readback.v1"
            if stage_profile == "original-ik-pulse"
            else "soccer.rsi.pulse_first_touch_stage_pairs_readback.v1",
            configuration=cfg,
            effective_cohort_spec=cohort_spec,
            effective_contact_spec=contact_spec,
            input_file_hashes=pins,
            native_complete_hash=complete["report_hash"],
            native_audit_hash=audit["report_hash"],
            contact_complete_hash=contact["report_hash"],
            stage_diagnostics=stages,
            pulse_dense_nonfoot_endpoints=measured["pulse_endpoints_with_nonfoot_over1n"],
            derived_courses=3,
            source_scene_clusters=3,
            source_native_bodies=6,
            all_six_receipts_and_failures_retained=True,
            new_physical_steps=0,
            new_neural_inferences=0,
            new_optimizer_updates=0,
            teacher_qualified=False,
            dataset_admission_authorized=False,
            receiving_skill_qualified=False,
            successor_readiness_qualified=False,
            hardware_authorized=False,
            promotion_authorized=False,
        )
        result["report_hash"] = hash_json(result)
        write_once(output, result)
        print("WHOLE_PULSE_STAGES_READBACK_NOT_NEW_PHYSICS", result["report_hash"], flush=True)
    except BaseException as error:
        write_once(
            started.with_suffix(".failure.json"),
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
