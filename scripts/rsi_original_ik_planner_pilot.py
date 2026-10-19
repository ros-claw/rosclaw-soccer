"""Fixed six original-IK OFF/pulse bodies; no learned reference or fitting."""

import argparse
import gc
import time
from pathlib import Path

import numpy as np

from rosclaw_soccer.rsi.bounded_first_touch_control import BoundedFirstTouchControl
from rosclaw_soccer.rsi.failure_curriculum_evidence import _sealed
from rosclaw_soccer.rsi.first_touch_campaign import FirstTouchCampaign
from rosclaw_soccer.rsi.first_touch_campaign_worker import qualified_episode
from rosclaw_soccer.rsi.original_ik_planner_evidence import require_original_ik_parity
from rosclaw_soccer.rsi.original_ik_planner_pulse import (
    OriginalIkPlannerPulse,
    require_original_ik_planner_request,
)
from rosclaw_soccer.rsi.original_ik_planner_scope import assess_ik_scope
from rosclaw_soccer.rsi.planner_navigation_pulse_scope import POSITIVE_PROFILE
from rosclaw_soccer.rsi.planner_navigation_pulse_scope import PROFILE as NEGATIVE_PROFILE
from rosclaw_soccer.rsi.sonic_encoder_reference_audit import audit_encoder_reference_consumption
from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_load_pre_planner_cohort import load_closed_pulse
from scripts.rsi_native_first_touch_bundle import allocate_native_first_touch_bundle
from scripts.rsi_pre_planner_navigation_pilot import admit_resources
from scripts.rsi_reference_student_exam import load_inputs
from scripts.rsi_stage_release_contact_replay import digest, require_terminal


def prepare(config_path, *, closed=False):
    c = load_inputs(config_path, original_ik_research=True)
    if (
        type(closed) is not bool
        or (not c.root.exists() if closed else c.root.exists())
        or Path(__file__).resolve() != c.source / "scripts/rsi_original_ik_planner_pilot.py"
    ):
        raise ValueError("original unused frozen original-IK producer required")
    c.signed_predecessors = {}
    for name, profile in (("negative", NEGATIVE_PROFILE), ("positive", POSITIVE_PROFILE)):
        spec = c.cfg["signed_predecessors"][name]
        cohort = load_closed_pulse(spec["cohort"], sealed=c.sealed, pins=c.pins)
        require_terminal(spec["stage_unit"])
        stage = c.sealed(spec["stage_readback"])
        if (
            cohort["producer"]["exam_profile"] != profile
            or stage["report_hash"] != spec["stage_readback_hash"]
            or stage.get("schema") != "soccer.rsi.pulse_first_touch_stage_pairs_readback.v1"
            or stage["native_complete_hash"] != cohort["complete"]["report_hash"]
            or stage["native_audit_hash"] != cohort["audit"]["report_hash"]
            or stage.get("all_six_receipts_and_failures_retained") is not True
            or type(stage.get("source_native_bodies")) is not int
            or stage["source_native_bodies"] != 6
            or any(
                type(stage.get(k)) is not int or stage[k] != 0
                for k in ("new_physical_steps", "new_neural_inferences", "new_optimizer_updates")
            )
            or any(
                stage.get(k) is not False
                for k in (
                    "teacher_qualified",
                    "dataset_admission_authorized",
                    "hardware_authorized",
                    "promotion_authorized",
                    "receiving_skill_qualified",
                    "successor_readiness_qualified",
                )
            )
        ):
            raise ValueError("both whole closed signed studies and dense/stage reviews required")
        for path, h in stage["input_file_hashes"].items():
            if digest(path) != h or (path in c.pins and c.pins[path] != h):
                raise ValueError("signed predecessor source or evidence changed")
            c.pins[path] = h
        c.signed_predecessors[name] = dict(
            complete_hash=cohort["complete"]["report_hash"],
            audit_hash=cohort["audit"]["report_hash"],
            stage_hash=stage["report_hash"],
        )
    for delta in (0.0, 0.08):
        require_original_ik_planner_request(
            OriginalIkPlannerPulse(delta),
            None,
            profile=BoundedFirstTouchControl(**c.candidate["parameters"]),
            record_planner_events=True,
        )
    return c


def intent_for(job, declaration_hash):
    result = dict(
        job,
        leg_reference_kind="ORIGINAL_BOUNDED_IK",
        pulse_contract_hash=OriginalIkPlannerPulse(job["lateral_delta_m_s"]).contract()[
            "contract_hash"
        ],
        declaration_hash=declaration_hash,
    )
    return dict(result, report_hash=hash_json(result))


def prefix_for(job, trace, teacher, pins):
    witness = teacher["witnesses"]["trace"]
    if (
        pins.get(witness["path"]) != witness["sha256"]
        or digest(witness["path"]) != witness["sha256"]
    ):
        raise ValueError("source-authenticated original IK teacher trace required")
    with np.load(witness["path"], allow_pickle=False) as a:
        baseline = dict(a)
    result = require_original_ik_parity(
        trace, baseline, frames=300 if job["role"] == "ik_pulse_off" else 100
    )
    result.update(
        original_ik_teacher_receipt_hash=teacher["receipt_hash"],
        original_ik_teacher_trace_hash=witness["sha256"],
        hardware_authorized=False,
        promotion_authorized=False,
    )
    return dict(result, report_hash=hash_json(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    c = prepare(args.config)
    if args.preflight:
        print("ORIGINAL_IK_SIX_READY_NO_FULL_MODELS_OR_PHYSICS", flush=True)
        return
    admit_resources(c)
    c.root.mkdir()
    declaration = dict(
        c.cfg,
        schema="soccer.rsi.original_ik_planner_six_declaration.v1",
        scope=c.scope,
        candidate=c.candidate,
        reused_teachers={str(g): r for g, r in c.teachers.items()},
        signed_predecessor_reviews=c.signed_predecessors,
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
                raise RuntimeError("fixed original-IK study deadline reached")
            intent = intent_for(job, declaration["report_hash"])
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
                original_ik_planner_pulse=OriginalIkPlannerPulse(job["lateral_delta_m_s"]),
            )
            with np.load(folder / "physical_trace.npz", allow_pickle=False) as a:
                trace = dict(a)
            pp = c.root / f"prefix-{attempt}.json"
            write_once(pp, prefix_for(job, trace, c.teachers[job["group"]], c.pins))
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
            print("ORIGINAL_IK_BODY_COMMITTED", job["job"], receipt["outcome"], flush=True)
            del trace, report
            gc.collect()
        assessment = assess_ik_scope(c.scope, results)
        c.p.stable()
        if any(digest(n) != h for n, h in c.pins.items()):
            raise ValueError("frozen sources or original evidence changed during IK study")
        complete = dict(
            schema="soccer.rsi.original_ik_planner_six_complete.v1",
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
