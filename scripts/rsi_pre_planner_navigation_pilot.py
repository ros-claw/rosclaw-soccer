"""Six fixed OFF/pulse native bodies; zero leg references, no fitting or search."""

import argparse
import gc
import time
from pathlib import Path

import numpy as np

from rosclaw_soccer.rsi.failure_curriculum_evidence import _sealed
from rosclaw_soccer.rsi.first_touch_campaign import FirstTouchCampaign
from rosclaw_soccer.rsi.first_touch_campaign_worker import qualified_episode
from rosclaw_soccer.rsi.planner_navigation_pulse import PlannerNavigationPulse
from rosclaw_soccer.rsi.planner_navigation_pulse_evidence import (
    require_navigation_pulse_request,
    require_pulse_physical_parity,
)
from rosclaw_soccer.rsi.planner_navigation_pulse_scope import assess_pulse_scope
from rosclaw_soccer.rsi.sonic_encoder_reference_audit import audit_encoder_reference_consumption
from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_compare_matched_references import load_cohort
from scripts.rsi_native_first_touch_bundle import allocate_native_first_touch_bundle
from scripts.rsi_reference_student_exam import load_inputs
from scripts.rsi_stage_release_contact_replay import digest


def prepare(config_path, *, closed=False):
    c = load_inputs(config_path, pulse_research=True)
    if (
        type(closed) is not bool
        or (not c.root.exists() if closed else c.root.exists())
        or Path(__file__).resolve() != c.source / "scripts/rsi_pre_planner_navigation_pilot.py"
    ):
        raise ValueError("unused original frozen six-body pulse runner required")
    recorded = load_cohort(c.cfg["matched_zero_cohort"], sealed=c.sealed, pins=c.pins)
    d = recorded["declaration"]
    if (
        recorded["config"]["exam_profile"] != "reference-positive-zero-nine-v1"
        or d["candidate"] != c.candidate
        or d["models"]["zero"] != c.models["zero"]
        or d["reused_teachers"] != {str(g): r for g, r in c.teachers.items()}
        or hash_json(d["scope"]["source_plans"]["positive19"])
        != hash_json(c.positive_plan.contract())
        or d["mesh_task_contact_world_hash"] != c.cfg["mesh_task_contact_world_hash"]
    ):
        raise ValueError(
            "same recorded NAV controls, parent, normalization, world and source plan required"
        )
    c.recorded_zeros = {
        r["group"]: r for r in recorded["complete"]["results"] if r["role"] == "zero"
    }
    c.recorded_zero_complete_hash = recorded["complete"]["report_hash"]
    from rosclaw_soccer.rsi.bounded_first_touch_control import BoundedFirstTouchControl

    for delta in {j["lateral_delta_m_s"] for j in c.scope["jobs"]}:
        require_navigation_pulse_request(
            PlannerNavigationPulse(delta),
            c.models["zero"],
            profile=BoundedFirstTouchControl(**c.candidate["parameters"]),
            record_planner_events=True,
        )
    if "negative_predecessor" in c.cfg:
        from scripts.rsi_load_pre_planner_cohort import require_negative_predecessor

        c.negative_predecessor = require_negative_predecessor(
            c.cfg["negative_predecessor"], sealed=c.sealed, pins=c.pins
        )
    return c


def admit_resources(c):
    import subprocess

    available = next(
        int(s.split()[1]) * 1024
        for s in Path("/proc/meminfo").read_text().splitlines()
        if s.startswith("MemAvailable:")
    )
    units = subprocess.check_output(
        [
            "systemctl",
            "--user",
            "list-units",
            "--type=service",
            "--state=active,activating",
            "--plain",
            "--no-legend",
        ],
        text=True,
    )
    own = sum(
        line.split()[0].startswith("rosclaw-") and "-native-" in line.split()[0]
        for line in units.splitlines()
        if line.split()
    )
    if available < 56 * 1024**3 or own > 2 or time.time() >= c.cfg["deadline_unix"]:
        raise RuntimeError(
            "56GiB before allocation, at most two own native workers and fixed deadline required"
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--preflight", action="store_true")
    args = p.parse_args()
    c = prepare(args.config)
    if args.preflight:
        print("SIX_FIXED_PRE_PLANNER_PROBES_READY_ZERO_FULL_MODELS_OR_PHYSICS", flush=True)
        return
    admit_resources(c)
    c.root.mkdir()
    declaration = dict(
        c.cfg,
        schema="soccer.rsi.pre_planner_pulse_six_declaration.v1",
        scope=c.scope,
        zero_reference_model=c.models["zero"],
        candidate=c.candidate,
        reused_zero_controls={str(g): r for g, r in c.recorded_zeros.items()},
        recorded_zero_complete_hash=c.recorded_zero_complete_hash,
        original_fit_stage_hash=c.fit_stage["report_hash"],
        input_file_hashes=c.pins,
        prepared_pins=dict(c.p.PINS),
        source_heads=dict(c.p.HEADS),
    )
    declaration["report_hash"] = hash_json(declaration)
    write_once(c.root / "declaration.json", declaration)
    campaign = FirstTouchCampaign(c.root, declaration)

    def phase(name):
        campaign.state["phase"] = name
        campaign.record(name)
        print(name, flush=True)

    try:
        bundle = allocate_native_first_touch_bundle(
            c.assets, c.nominal["results"][0], c.root, record_phase=phase
        )
        results = []
        for job in c.scope["jobs"]:
            if time.time() >= c.cfg["deadline_unix"]:
                raise RuntimeError("fixed pulse study deadline reached")
            pulse = PlannerNavigationPulse(job["lateral_delta_m_s"])
            intent = dict(
                job,
                zero_reference_model_hash=c.models["zero"]["model_hash"],
                pulse_contract_hash=pulse.contract()["contract_hash"],
                declaration_hash=declaration["report_hash"],
            )
            intent["report_hash"] = hash_json(intent)
            ip = c.root / f"intent-{job['job']}.json"
            write_once(ip, intent)
            campaign.state["next_intent_hash"] = intent["report_hash"]
            campaign.record("INTENT_COMMITTED_BEFORE_BODY")
            attempt = campaign.begin(job["job"])
            folder = c.root / f"attempt-{attempt:04d}"
            receipt = qualified_episode(
                folder,
                job["group"],
                c.candidate,
                plan=c.positive_plan,
                manifest=c.positive_manifest,
                assets=c.assets,
                source=c.source,
                core=c.core,
                factory=bundle.factory,
                compiler=bundle.compiler,
                auditor=bundle.auditor,
                entry_source_path=Path(__file__),
                physical_payload_reader=bundle.reader,
                mesh_task_contact_world_hash=c.cfg["mesh_task_contact_world_hash"],
                mesh_world_review=c.review,
                planning_context=c.contexts[job["group"]],
                planning_stage="SELECTED_OPTION_EVALUATION",
                generated_seed_episode_binding=True,
                reference_student_model=c.models["zero"],
                planner_navigation_pulse=pulse,
            )
            with np.load(folder / "physical_trace.npz", allow_pickle=False) as a:
                trace = dict(a)
            witness = c.recorded_zeros[job["group"]]["witnesses"]["trace"]
            if (
                c.pins.get(witness["path"]) != witness["sha256"]
                or digest(witness["path"]) != witness["sha256"]
            ):
                raise ValueError("authenticated original NAV witness required")
            with np.load(witness["path"], allow_pickle=False) as a:
                baseline = dict(a)
            frames = 300 if job["role"] == "pulse_off" else 100
            prefix = require_pulse_physical_parity(trace, baseline, frames=frames)
            prefix.update(
                original_nav_receipt_hash=c.recorded_zeros[job["group"]]["receipt_hash"],
                original_nav_trace_hash=witness["sha256"],
                hardware_authorized=False,
                promotion_authorized=False,
            )
            prefix["report_hash"] = hash_json(prefix)
            pp = c.root / f"prefix-{attempt}.json"
            write_once(pp, prefix)
            report = _sealed(folder / "report.json", physical_payload_reader=bundle.reader)
            encoder = audit_encoder_reference_consumption(
                bundle.encoder_session, trace, report["default_locomotion_planner_events"], 2
            )
            ep = c.root / f"encoder-{attempt}.json"
            write_once(ep, encoder)
            receipt.update(
                job,
                intent_witness=dict(path=str(ip), sha256=digest(ip)),
                prefix_witness=dict(path=str(pp), sha256=digest(pp)),
                encoder_witness=dict(path=str(ep), sha256=digest(ep)),
            )
            receipt["receipt_hash"] = hash_json(receipt)
            write_once(c.root / f"receipt-{attempt:04d}.json", receipt)
            campaign.complete(job["job"], receipt)
            results.append(receipt)
            print("PRE_PLANNER_PULSE_BODY_COMMITTED", job["job"], receipt["outcome"], flush=True)
            del trace, baseline, report
            gc.collect()
        assessment = assess_pulse_scope(c.scope, results)
        c.p.stable()
        if any(digest(name) != h for name, h in c.pins.items()):
            raise ValueError("frozen source or original witness changed during pulse study")
        complete = dict(
            schema="soccer.rsi.pre_planner_pulse_six_complete.v1",
            declaration_hash=declaration["report_hash"],
            results=results,
            assessment=assessment,
            new_native_bodies=6,
            new_independent_cpu_bodies=6,
            native_substeps=18000,
            independent_cpu_substeps=18000,
            new_optimizer_updates=0,
            independent_terminal_readback_required=True,
            teacher_qualified=False,
            hardware_authorized=False,
            promotion_authorized=False,
        )
        complete["report_hash"] = hash_json(complete)
        write_once(c.root / "complete.json", complete)
        phase("PILOT_COMPLETE_EXPANDED_VALIDATION_REQUIRED")
    except BaseException as error:
        campaign.state["fatal_error"] = dict(type=type(error).__name__, message=str(error))
        phase("FATAL_CONTRACT_VIOLATION")
        raise


if __name__ == "__main__":
    main()
