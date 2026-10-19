"""Fixed six heading OFF/ON bodies on the same positive original IK control."""

import argparse
import gc
import time
from pathlib import Path

import numpy as np

from rosclaw_soccer.rsi.bounded_first_touch_control import BoundedFirstTouchControl
from rosclaw_soccer.rsi.failure_curriculum_evidence import _sealed
from rosclaw_soccer.rsi.first_touch_campaign import FirstTouchCampaign
from rosclaw_soccer.rsi.first_touch_campaign_worker import qualified_episode
from rosclaw_soccer.rsi.foot_heading_control import FootHeadingContactProbe, require_heading_request
from rosclaw_soccer.rsi.foot_heading_parity import require_heading_parity
from rosclaw_soccer.rsi.foot_heading_scope import assess_heading_scope
from rosclaw_soccer.rsi.original_ik_planner_pulse import OriginalIkPlannerPulse
from rosclaw_soccer.rsi.sonic_encoder_reference_audit import audit_encoder_reference_consumption
from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_load_heading_predecessor import load_heading_predecessor
from scripts.rsi_native_first_touch_bundle import allocate_native_first_touch_bundle
from scripts.rsi_pre_planner_navigation_pilot import admit_resources
from scripts.rsi_reference_student_exam import load_inputs
from scripts.rsi_stage_release_contact_replay import digest


def prepare(config_path, *, closed=False):
    c = load_inputs(config_path, heading_research=True)
    if (
        type(closed) is not bool
        or (not c.root.exists() if closed else c.root.exists())
        or Path(__file__).resolve() != c.source / "scripts/rsi_foot_heading_pilot.py"
    ):
        raise ValueError("original unused frozen heading producer required")
    previous = load_heading_predecessor(
        c.cfg["original_ik_predecessor"], sealed=c.sealed, pins=c.pins
    )
    cohort = previous["cohort"]
    if (
        cohort["producer"]["mesh_task_contact_world_hash"] != c.cfg["mesh_task_contact_world_hash"]
        or cohort["producer"]["behavior_model_hash"] != c.assets.MEAN
        or cohort["declaration"]["candidate"] != c.candidate
        or cohort["producer"]["core_root"] != str(c.core)
        or hash_json(cohort["declaration"]["scope"]["source_plan"])
        != hash_json(c.scope["source_plan"])
        or cohort["declaration"]["scope"]["source_seeds"] != c.scope["source_seeds"]
    ):
        raise ValueError("same C2 parent, B2, world, Core and source19 planning contexts required")
    c.heading_baselines = {
        r["group"]: r for r in cohort["complete"]["results"] if r["role"] == "ik_pulse"
    }
    if set(c.heading_baselines) != set(c.cfg["heading_group_order"]) or any(
        r["lateral_delta_m_s"] != 0.08 for r in c.heading_baselines.values()
    ):
        raise ValueError("all three current positive original IK baselines required")
    c.predecessor_summary = dict(
        native_complete_hash=cohort["complete"]["report_hash"],
        audit_hash=cohort["audit"]["report_hash"],
        contact_hash=previous["contact"]["report_hash"],
        stage_hash=previous["stage"]["report_hash"],
        all_six_bodies_and_failures_retained=True,
        teacher_qualified=False,
    )
    for enabled in (False, True):
        require_heading_request(
            FootHeadingContactProbe(enabled),
            pulse=OriginalIkPlannerPulse(0.08),
            reference_model=None,
            profile=BoundedFirstTouchControl(**c.candidate["parameters"]),
            record_planner_events=True,
        )
    return c


def intent_for(job, declaration_hash):
    result = dict(
        job,
        leg_reference_kind="ORIGINAL_BOUNDED_IK_WITH_HEADING_PROBE",
        heading_contract_hash=FootHeadingContactProbe(job["heading_enabled"]).contract()[
            "contract_hash"
        ],
        pulse_contract_hash=OriginalIkPlannerPulse(job["lateral_delta_m_s"]).contract()[
            "contract_hash"
        ],
        declaration_hash=declaration_hash,
    )
    return dict(result, report_hash=hash_json(result))


def prefix_for(job, trace, teacher, pins):
    if (
        job.get("role") not in ("heading_off", "heading_on")
        or type(job.get("heading_enabled")) is not bool
        or job["heading_enabled"] != (job["role"] == "heading_on")
        or teacher.get("role") != "ik_pulse"
        or teacher.get("group") != job["group"]
        or teacher.get("lateral_delta_m_s") != 0.08
    ):
        raise ValueError("matched whole C2 positive IK baseline and exact heading arm required")
    witness = teacher["witnesses"]["trace"]
    if (
        pins.get(witness["path"]) != witness["sha256"]
        or digest(witness["path"]) != witness["sha256"]
    ):
        raise ValueError("source-authenticated C2 positive original IK trace required")
    with np.load(witness["path"], allow_pickle=False) as a:
        baseline = dict(a)
    result = require_heading_parity(
        trace, baseline, frames=300 if job["role"] == "heading_off" else 30
    )
    result.update(
        c2_positive_receipt_hash=teacher["receipt_hash"],
        c2_positive_trace_hash=witness["sha256"],
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
        print("FOOT_HEADING_SIX_READY_NO_FULL_MODELS_OR_PHYSICS", flush=True)
        return
    admit_resources(c)
    c.root.mkdir()
    declaration = dict(
        c.cfg,
        schema="soccer.rsi.foot_heading_six_declaration.v1",
        scope=c.scope,
        candidate=c.candidate,
        reused_positive_ik_baselines={str(g): r for g, r in c.heading_baselines.items()},
        original_ik_predecessor_review=c.predecessor_summary,
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
                raise RuntimeError("fixed heading study deadline reached")
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
                foot_heading_control=FootHeadingContactProbe(job["heading_enabled"]),
            )
            with np.load(folder / "physical_trace.npz", allow_pickle=False) as a:
                trace = dict(a)
            pp = c.root / f"prefix-{attempt}.json"
            write_once(pp, prefix_for(job, trace, c.heading_baselines[job["group"]], c.pins))
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
            print("FOOT_HEADING_BODY_COMMITTED", job["job"], receipt["outcome"], flush=True)
            del trace, report
            gc.collect()
        assessment = assess_heading_scope(c.scope, results)
        c.p.stable()
        if any(digest(n) != h for n, h in c.pins.items()):
            raise ValueError("frozen sources or C2 evidence changed during heading study")
        complete = dict(
            schema="soccer.rsi.foot_heading_six_complete.v1",
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
