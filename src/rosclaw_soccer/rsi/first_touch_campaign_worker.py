"""Persistent first-touch SIM pilot: qualified episodes, feedback, next round."""

import argparse
import ast
import fcntl
import gc
import importlib.util
import json
import os
import shutil
import signal
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.bounded_first_touch_control import BOUNDS, BoundedFirstTouchControl
from rosclaw_soccer.rsi.bounded_first_touch_search import initial_population
from rosclaw_soccer.rsi.first_touch_campaign import (
    FirstTouchCampaign,
    load_verified_predecessor,
    summarize_first_touch_candidate,
)
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def atomic_status(root: Path, name: str, value: dict[str, Any]) -> None:
    target = root / name
    temporary = root / f".{name}-{os.getpid()}-{threading.get_ident()}"
    with temporary.open("w") as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, target)


def disk_admission(*, free_bytes: int, reserved_gib: int) -> bool:
    """Zero explicitly disables capacity gating, not evidence write checks."""
    if type(reserved_gib) is not int or reserved_gib < 0:
        raise ValueError("explicit nonnegative disk reservation required")
    if type(free_bytes) is not int or free_bytes < 0:
        raise ValueError("nonnegative observed disk capacity required")
    return reserved_gib == 0 or free_bytes >= reserved_gib * 1024**3


def require_same_action_functions(previous: Path, current: Path) -> None:
    names = {"qualified_episode", "propose_next"}

    def extract(path: Path) -> dict[str, str]:
        tree = ast.parse(path.read_text())
        result = {
            n.name: ast.dump(n, include_attributes=False)
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name in names
        }
        if set(result) != names:
            raise ValueError("complete original action and search functions required")
        return result

    if extract(previous) != extract(current):
        raise ValueError("operational succession cannot change action or search arithmetic")


def prepare(config_path: Path) -> tuple[Any, Any, dict[str, Any], dict[str, Any]]:
    config = json.loads(config_path.read_text())
    if (
        config.get("mode") != "SIM_ONLY"
        or config.get("hardware_authorized") is not False
        or config.get("promotion_authorized") is not False
        or config.get("private_fresh_accessed") is not False
    ):
        raise ValueError("SIM-only public research configuration required")
    disk_admission(free_bytes=0, reserved_gib=config.get("reserved_disk_gib", 101))
    helper = Path(config["qualified_preparation_path"])
    if hash_bytes(helper.read_bytes()) != config["qualified_preparation_hash"]:
        raise ValueError("exact independently qualified preparation required")
    spec = importlib.util.spec_from_file_location("qualified_campaign_preparation", helper)
    if spec is None or spec.loader is None:
        raise ValueError("ordinary pinned preparation module required")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    p, assets, manifest = module.preparation()
    source, core = Path(config["source_root"]), Path(config["core_root"])
    if Path(__file__).resolve() != source / "src/rosclaw_soccer/rsi/first_touch_campaign_worker.py":
        raise ValueError("worker must execute from declared immutable source")
    p.HEADS[str(source)] = config["source_head"]
    for folder in (source / "src", source / "scripts"):
        p.add_pins({str(path): p.digest(path) for path in folder.rglob("*.py")})
    predecessor = config.get("predecessor")
    if predecessor is not None:
        old_config_path = Path(predecessor["configuration_path"])
        if p.digest(old_config_path) != predecessor["configuration_hash"]:
            raise ValueError("original predecessor configuration required")
        old_config = json.loads(old_config_path.read_text())
        allowed_changes = {
            "source_root",
            "source_head",
            "output_root",
            "reserved_disk_gib",
            "predecessor",
        }
        if {k: v for k, v in old_config.items() if k not in allowed_changes} != {
            k: v for k, v in config.items() if k not in allowed_changes
        }:
            raise ValueError("operational succession cannot change scientific contract or budgets")
        old_source = Path(old_config["source_root"])
        if (
            Path(old_config["output_root"]).resolve()
            != Path(predecessor["evidence_root"]).resolve()
        ):
            raise ValueError("predecessor evidence must belong to the original declaration")
        p.HEADS[str(old_source)] = old_config["source_head"]
        allowed_source_changes = {
            "src/rosclaw_soccer/rsi/first_touch_campaign.py",
            "src/rosclaw_soccer/rsi/first_touch_campaign_worker.py",
            "scripts/rsi_run_first_touch_campaign.py",
        }

        def scientific_sources(base: Path) -> dict[str, str]:
            return {
                str(path.relative_to(base)): p.digest(path)
                for folder in (base / "src", base / "scripts")
                for path in folder.rglob("*.py")
                if str(path.relative_to(base)) not in allowed_source_changes
            }

        if scientific_sources(source) != scientific_sources(old_source):
            raise ValueError(
                "scientific control, scoring or learning sources changed during succession"
            )
        relative_worker = "src/rosclaw_soccer/rsi/first_touch_campaign_worker.py"
        require_same_action_functions(old_source / relative_worker, source / relative_worker)
        load_verified_predecessor(predecessor)
        for path in (old_config_path, Path(predecessor["ledger_path"])):
            p.add_pins({str(path): p.digest(path)})
    p.add_pins({str(config_path): p.digest(config_path)})
    result_path = Path(config["interface_result"])
    result = p.sealed(result_path)
    if (
        result["report_hash"] != config["interface_result_hash"]
        or result["actual_native_executions"] != 9
        or result["actual_independent_replay_substeps"] != 27000
        or result["actual_foundation_calls"] != 2700
        or result["private_fresh_accessed"] is not False
        or len(result["records"]) != 9
    ):
        raise ValueError("all nine independent interface qualifications required")
    for row in result["records"]:
        folder = result_path.parent / f"case-{row['group']}-{row['arm']}"
        replay = p.sealed(folder / "independent_review.json")
        foundation = p.sealed(folder / "foundation_review.json")
        if (
            replay["report_hash"] != row["independent_review_hash"]
            or replay["reviewed_report_hash"] != row["physical_report_hash"]
            or replay["physical_substeps"] != 3000
            or foundation["report_hash"] != row["foundation_review_hash"]
            or foundation["recomputed_executed_calls"] != 300
            or p.digest(folder / "physical_trace.npz") != row["physical_trace_hash"]
        ):
            raise ValueError("interface completion must bind actual independent witnesses")
        p.add_pins({str(path): p.digest(path) for path in folder.iterdir() if path.is_file()})
    p.add_pins({str(result_path): p.digest(result_path)})
    sys.path[:0] = [str(source / "src"), str(core / "src"), str(source)]
    p.stable()
    return p, assets, manifest, config


def propose_next(
    population: list[dict[str, Any]], summaries: list[dict[str, Any]], *, seed: int
) -> list[dict[str, Any]]:
    """Failure-aware proposal optimization, NEVER teacher or policy admission.

    Unlike safe teacher selection, SIM search can learn from partial failures.
    Minimize safety and retention regressions first. Failed parents remain
    explicitly unqualified; they only guide the next bounded parameter probe.
    """
    ranked = sorted(
        range(len(population)),
        key=lambda i: (
            -summaries[i]["safety_regression"],
            -summaries[i]["regression"],
            summaries[i]["high_quality"],
            summaries[i]["safe_states"],
            summaries[i]["mean_reward"],
            -i,
        ),
        reverse=True,
    )
    rng = np.random.default_rng(seed)
    names = list(BOUNDS)
    lower = np.asarray([BOUNDS[n][0] for n in names])
    span = np.asarray([BOUNDS[n][1] - BOUNDS[n][0] for n in names])
    values = np.asarray([[population[i]["parameters"][n] for n in names] for i in ranked[:2]])
    normalized = (values - lower) / span
    mean, std = normalized.mean(axis=0), np.maximum(normalized.std(axis=0), 0.08)
    result = [population[ranked[0]]]
    for i in range(2):
        for _ in range(10000):
            draw = rng.normal(mean, std)
            if np.all((draw >= 0) & (draw <= 1)):
                parameters = {n: float(v) for n, v in zip(names, lower + span * draw, strict=True)}
                BoundedFirstTouchControl(**parameters)
                result.append(dict(parameters=parameters, swing=population[ranked[i]]["swing"]))
                break
        else:
            raise ValueError("bounded search rejection budget exhausted")
    restart = initial_population(seed=seed + 1, count=2)[1]
    result.append(dict(parameters=asdict(restart), swing="bounded_contact_diagnostic"))
    return result


def run(config_path: Path, *, preflight: bool = False) -> None:
    p, assets, manifest, config = prepare(config_path)
    if preflight:
        print("CAMPAIGN_PREFLIGHT_NO_NEURAL_OR_PHYSICS_ALLOCATION", flush=True)
        return
    source, core, root = (Path(config[k]) for k in ("source_root", "core_root", "output_root"))
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / "owner.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    declaration = dict(config, input_file_hashes=dict(p.PINS), source_heads=p.HEADS)
    campaign = FirstTouchCampaign(root, declaration)
    campaign.adopt_predecessor()
    if campaign.state["phase"] == "FATAL_CONTRACT_VIOLATION":
        raise ValueError("fatal campaign requires incident review and an explicit successor")
    if campaign.state["phase"] in {
        "PLATEAU_RESEARCH_REQUIRED",
        "PILOT_COMPLETE_EXPANDED_VALIDATION_REQUIRED",
        "ATTEMPT_BUDGET_EXHAUSTED",
        "WALLTIME_BUDGET_EXHAUSTED",
        "GENERATION_BUDGET_EXHAUSTED",
    }:
        print(
            json.dumps(
                dict(event="TERMINAL_CAMPAIGN_NOT_RESTARTED", phase=campaign.state["phase"])
            ),
            flush=True,
        )
        return
    campaign.recover_interrupted_attempt()
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())

    def heartbeat() -> None:
        while not stop.is_set():
            atomic_status(
                root,
                "heartbeat.json",
                dict(
                    pid=os.getpid(),
                    time_unix=time.time(),
                    phase=campaign.state["phase"],
                    attempts=campaign.state["attempts"],
                    completed=len(campaign.state["completed"]),
                    active=campaign.state["active"],
                    neural_optimizer_updates=0,
                    hardware_authorized=False,
                    promotion_authorized=False,
                ),
            )
            stop.wait(15)

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()

    def resources() -> bool:
        while not stop.is_set():
            if time.time() > config["deadline_unix"]:
                campaign.state["phase"] = "WALLTIME_BUDGET_EXHAUSTED"
                campaign.record("WALLTIME_LIMIT")
                return False
            ram = next(
                int(line.split()[1]) * 1024
                for line in Path("/proc/meminfo").read_text().splitlines()
                if line.startswith("MemAvailable:")
            )
            phase = (
                "PAUSED"
                if (root / "PAUSE").exists()
                else (
                    "WAITING_RESOURCES"
                    if ram < 48 * 1024**3
                    or not disk_admission(
                        free_bytes=0
                        if config.get("reserved_disk_gib", 101) == 0
                        else shutil.disk_usage(root).free,
                        reserved_gib=config.get("reserved_disk_gib", 101),
                    )
                    else "READY_FOR_NEXT_JOB"
                )
            )
            if phase == "READY_FOR_NEXT_JOB":
                return True
            if campaign.state["phase"] != phase:
                campaign.state["phase"] = phase
                campaign.record(phase)
            stop.wait(10)
        return False

    try:
        if not resources():
            return
        campaign.state["phase"] = "INITIALIZING_FULL_FROZEN_MODEL"
        campaign.record("INITIALIZATION_STARTED")
        import torch

        from rosclaw_soccer.rsi.failure_curriculum_evidence import _sealed
        from rosclaw_soccer.rsi.foundation_torch_capture_audit import TorchCallAuditor
        from rosclaw_soccer.rsi.json_artifact_io import load_json_artifact
        from rosclaw_soccer.rsi.public_first_touch_training_plan import PublicFirstTouchTrainingPlan
        from rosclaw_soccer.rsi.recurrent_sampling_episode_factory import (
            RecurrentSamplingEpisodeFactory,
        )
        from rosclaw_soccer.rsi.sampling_reference_audit import SamplingReferenceAuditCompiler

        torch.set_num_threads(1)
        plan = PublicFirstTouchTrainingPlan(tuple(range(20261060, 20261070)), assets.MEAN, 18)
        groups = [d.group for d in plan.draws() if d.sample_index == 0 and d.seed in plan.seeds[:4]]
        if groups != config["groups"] or len(groups) != 16:
            raise ValueError("fixed four-stratum16 public pilot required")
        mean = load_json_artifact(assets.EXPORT / "model-0.json.gz")
        if mean["model_hash"] != assets.MEAN:
            raise ValueError("complete frozen parent identity required")
        factory = RecurrentSamplingEpisodeFactory(mean)
        del mean
        gc.collect()
        old = _sealed(
            Path(manifest["records"][0]["witnesses"]["trace"]["path"]).parent / "report.json"
        )
        compiler = SamplingReferenceAuditCompiler(old["executed_motor_policy"])
        del old
        gc.collect()
        auditor = TorchCallAuditor(assets.SONIC)

        def episode(
            job: str, group: int, candidate: dict[str, Any] | None
        ) -> dict[str, Any] | None:
            if job in campaign.state["completed"]:
                completed: dict[str, Any] = campaign.state["completed"][job]
                return completed
            identity = hash_json(
                dict(group=group, course=plan.draws()[group].course, candidate=candidate)
            )
            for original_job, receipt in campaign.state["completed"].items():
                if receipt.get("evaluation_identity") == identity:
                    return campaign.reuse(job, original_job, identity)
            if campaign.state["attempts"] >= config["max_attempts"]:
                campaign.state["phase"] = "ATTEMPT_BUDGET_EXHAUSTED"
                campaign.record("ATTEMPT_LIMIT")
                return None
            if not resources():
                return None
            p.stable()
            attempt = campaign.begin(job)
            folder = root / f"attempt-{attempt:04d}-{job}"
            receipt = qualified_episode(
                folder,
                group,
                candidate,
                plan=plan,
                manifest=manifest,
                assets=assets,
                source=source,
                core=core,
                factory=factory,
                compiler=compiler,
                auditor=auditor,
            )
            receipt["evaluation_identity"] = identity
            p.stable()
            campaign.complete(job, receipt)
            print(
                json.dumps(
                    dict(
                        event="EPISODE_VERIFIED",
                        job=job,
                        attempt=attempt,
                        outcome=receipt["outcome"],
                    ),
                    allow_nan=False,
                ),
                flush=True,
            )
            gc.collect()
            return receipt

        if episode("legacy-control", 0, None) is None:
            return
        if "populations" not in campaign.state:
            samples = initial_population(seed=config["search_seed"], count=3)
            default = asdict(BoundedFirstTouchControl())
            campaign.state["populations"] = [
                [
                    dict(parameters=default, swing="legacy"),
                    dict(parameters=default, swing="bounded_contact_diagnostic"),
                    *[
                        dict(parameters=asdict(p), swing="bounded_contact_diagnostic")
                        for p in samples[1:]
                    ],
                ]
            ]
            campaign.record("FIRST_POPULATION_PREDECLARED")
        for generation in range(config["max_generations"]):
            if generation < len(campaign.state["generations"]):
                continue
            if len(campaign.state["populations"]) == generation:
                previous = campaign.state["generations"][-1]
                campaign.state["populations"].append(
                    propose_next(
                        campaign.state["populations"][-1],
                        previous["candidate_summaries"],
                        seed=config["search_seed"] + 100 * generation,
                    )
                )
                campaign.record("FEEDBACK_CONDITIONED_NEXT_POPULATION_PREDECLARED")
            population = campaign.state["populations"][generation]
            summaries = []
            for i, candidate in enumerate(population):
                rows = []
                for group in groups:
                    receipt = episode(f"g{generation}-c{i}-s{group}", group, candidate)
                    if receipt is None:
                        return
                    rows.append(receipt["outcome"])
                summaries.append(
                    summarize_first_touch_candidate(
                        rows,
                        [manifest["records"][group]["label"] for group in groups],
                    )
                )
            gain = any(s["paired_development_gain"] for s in summaries)
            phase = campaign.finish_generation(
                dict(
                    generation=generation,
                    candidate_summaries=summaries,
                    paired_development_gain=gain,
                    neural_optimizer_updates=0,
                    search_parameter_update=generation > 0,
                    private_fresh_accessed=False,
                    teacher_qualified=False,
                    independent_generalization_proven=False,
                )
            )
            print(
                json.dumps(
                    dict(
                        event="GENERATION_REVIEWED",
                        phase=phase,
                        generation=generation,
                        summaries=summaries,
                    ),
                    allow_nan=False,
                ),
                flush=True,
            )
            if phase == "PLATEAU_RESEARCH_REQUIRED":
                return
        campaign.state["phase"] = "PILOT_COMPLETE_EXPANDED_VALIDATION_REQUIRED"
        campaign.record("BOUNDED_PILOT_COMPLETE_NOT_TEACHER_ADMISSION")
    except BaseException as error:
        campaign.state["phase"] = "FATAL_CONTRACT_VIOLATION"
        campaign.state["fatal_error"] = dict(type=type(error).__name__, message=str(error))
        campaign.record("FATAL_DO_NOT_AUTO_UNFREEZE")
        raise
    finally:
        if stop.is_set() and campaign.state["phase"] not in {
            "FATAL_CONTRACT_VIOLATION",
            "PLATEAU_RESEARCH_REQUIRED",
            "PILOT_COMPLETE_EXPANDED_VALIDATION_REQUIRED",
            "ATTEMPT_BUDGET_EXHAUSTED",
            "WALLTIME_BUDGET_EXHAUSTED",
        }:
            campaign.state["phase"] = "STOPPED_OPERATOR_REQUEST"
            campaign.record("GRACEFUL_STOP_CHECKPOINTED")
        stop.set()
        thread.join(timeout=20)
        atomic_status(root, "status.json", dict(campaign.state, ledger_tip=campaign.tip))
        campaign.db.close()
        lock.close()


def qualified_episode(
    folder: Path,
    group: int,
    candidate: dict[str, Any] | None,
    *,
    plan: Any,
    manifest: dict[str, Any],
    assets: Any,
    source: Path,
    core: Path,
    factory: Any,
    compiler: Any,
    auditor: Any,
) -> dict[str, Any]:
    """Actual native dynamics + independent CPU/Foundation; no synthetic labels."""
    from rosclaw_soccer.providers.g1.sonic_runup import G1SonicRunupController
    from rosclaw_soccer.rsi.contact_motor_primitive import CAP_RAD
    from rosclaw_soccer.rsi.cpu_motor_transfer_evidence import audit_cpu_transfer
    from rosclaw_soccer.rsi.failure_curriculum_evidence import _sealed
    from rosclaw_soccer.rsi.first_touch_outcome import label_first_touch
    from rosclaw_soccer.rsi.foundation_observation_capture import (
        validate_capture,
        validate_measured_history,
    )
    from rosclaw_soccer.rsi.json_artifact_io import load_json_artifact
    from rosclaw_soccer.rsi.native_first_touch_episode import run_native_first_touch_episode
    from rosclaw_soccer.rsi.native_first_touch_initial_state import NativeFirstTouchInitialState
    from scripts.rsi_atomic_artifacts import write_once

    draw = plan.draws()[group]
    reference = manifest["records"][group]
    if list(draw.course) != reference["draw"]["course"]:
        raise ValueError("same declared public state required")
    profile = None if candidate is None else BoundedFirstTouchControl(**candidate["parameters"])
    swing = "legacy" if candidate is None else candidate["swing"]
    args = argparse.Namespace(
        scene=assets.SCENE,
        model_root=assets.SONIC,
        late_swing_policy=assets.LATE,
        output_root=folder,
        neural_model=None,
        motor_policy=None,
        step_model=Path(reference["witnesses"]["sampling_view"]["path"]),
        body_response_bundle=None,
        cached_proposal_envelope=False,
        foundation_only=False,
        compressed_report=True,
        record_foundation_observation=True,
        shared_evidence=True,
        proposal_decoder="reference",
        root_velocity_reference="body-com",
        observation_snapshot="cached",
        seed=draw.seed,
        lane=draw.lane,
    )
    run_native_first_touch_episode(
        args,
        course=draw.course,
        partition="DECLARED_DEVELOPMENT_DIAGNOSTIC",
        entry_source_path=source / "scripts/rsi_run_first_touch_campaign.py",
        consumed_bank_hash=None,
        recurrent_sampling_factory=factory,
        control_profile=profile,
        approach_mode="legacy" if profile is None else "bounded_approach_diagnostic",
        swing_mode=swing,
    )
    report = _sealed(folder / "report.json")
    if report["compiled_model_hash"] != assets.WORLD or report["course"] != list(draw.course):
        raise ValueError("fixed world and course required")
    review = audit_cpu_transfer(
        folder,
        source / "src/rosclaw_soccer/rsi/native_first_touch_episode.py",
        sampling_reference_compiler=compiler,
        sampling_factory_source_roots=(source, core),
        allocate_sampling_reference_once=True,
        initial_state_reference=NativeFirstTouchInitialState(
            draw.course, G1SonicRunupController.default_angles
        ),
    )
    with np.load(folder / "physical_trace.npz", allow_pickle=False) as archive:
        trace = {k: archive[k].copy() for k in archive.files}
    with np.load(reference["witnesses"]["trace"]["path"], allow_pickle=False) as old:
        if any(
            not np.array_equal(trace[k][0], old[k][0]) for k in ("canonical_qpos", "canonical_qvel")
        ):
            raise ValueError("paired initial physical state changed")
        if candidate is None and (
            set(trace) != set(old.files)
            or any(
                trace[k].dtype != old[k].dtype or not np.array_equal(trace[k], old[k])
                for k in trace
            )
        ):
            raise ValueError("all28 legacy control arrays must exactly reproduce")
    if profile is not None:
        first = review["first_contact_frame"]
        for frame in range(300):
            gap = (
                trace["ball_position_before_step_m"][frame, 0]
                - trace["root_pose_xyzw_m"][frame, 0, :3]
            )
            before = first is None or frame <= first
            # Independent scalar reconstruction, not another call to navigation().
            if before:
                forward = float(trace["ball_linear_velocity_before_step_m_s"][frame, 0, 0]) + (
                    profile.closing_gain_s_inv * (float(gap[0]) - profile.stance_gap_m)
                )
                fade = min(
                    1.0,
                    max(
                        0.0,
                        (float(gap[0]) - profile.stance_gap_m)
                        / (profile.lateral_finish_gap_m - profile.stance_gap_m),
                    ),
                )
                lateral = (
                    profile.lateral_gain_s_inv * (float(gap[1]) - profile.lateral_offset_m) * fade
                )
            else:
                forward, lateral = profile.forward_cap_m_s, 0.0
            if not np.array_equal(
                np.asarray([forward, lateral, 0.0]),
                trace["approach_requested_navigation_command"][frame, 0],
            ):
                raise ValueError("requested navigation law mismatch")
            command = np.asarray(
                [np.clip(forward, 0.0, profile.forward_cap_m_s), np.clip(lateral, -0.2, 0.2), 0.0]
            )
            if not np.array_equal(command, trace["navigation_command"][frame, 0]):
                raise ValueError("projected navigation law mismatch")
        if not np.allclose(
            trace["foundation_navigation_joint_target_rad"]
            + trace["taskspace_joint_increment_rad"],
            trace["pre_motor_joint_target_rad"],
            rtol=0.0,
            atol=1e-15,
        ):
            raise ValueError("taskspace target composition mismatch")
        raw_motor = CAP_RAD * np.tanh(trace["recurrent_sampling_latent_action"])
        raw_motor[:30] = 0.0
        if not np.array_equal(raw_motor, trace["motor_pre_projection_requested_rad"]):
            raise ValueError("raw residual differs from actual original sampled latent action")
    commitment = load_json_artifact(folder / "commitment.json")
    validate_capture(commitment["foundation_observation_capture"], trace, frames=300, lanes=1)
    validate_measured_history(trace)
    foundation = auditor.recompute(trace)
    outcome = label_first_touch(trace)
    if (
        review["physical_substeps"] != 3000
        or foundation["recomputed_executed_calls"] != 300
        or review["actual_recurrent_sampling_draws_reconstructed"] != 270
        or review["high_quality"] is not outcome.task_pass
    ):
        raise ValueError("whole independent physical and Foundation qualification required")
    write_once(folder / "independent_review.json", review)
    foundation_document = dict(
        foundation,
        physical_report_hash=report["report_hash"],
        physical_trace_hash=report["physical_trace_hash"],
    )
    foundation_document["report_hash"] = hash_json(foundation_document)
    write_once(folder / "foundation_review.json", foundation_document)
    paths = dict(
        trace=folder / "physical_trace.npz",
        report=folder / "report.json.gz",
        independent_review=folder / "independent_review.json",
        foundation_review=folder / "foundation_review.json",
        commitment=folder / "commitment.json",
        world=folder / "compiled_model.mjb",
    )
    envelope = load_json_artifact(folder / "report.json.gz")
    payload_hash = envelope["payload_hash"]
    paths["proof_payload"] = folder.parent / ".shared-models" / f"{payload_hash[7:]}.json.gz"
    return dict(
        physical_substeps=3000,
        foundation_calls=300,
        sampling_draws=270,
        outcome=outcome.to_dict(),
        group=group,
        candidate=candidate,
        physical_report_hash=report["report_hash"],
        independent_review_hash=review["report_hash"],
        foundation_review_hash=foundation_document["report_hash"],
        witnesses={
            name: dict(path=str(path), sha256=hash_bytes(path.read_bytes()))
            for name, path in paths.items()
        },
        teacher_qualified=False,
        private_fresh_accessed=False,
        neural_optimizer_updates=0,
        promotion_authorized=False,
        hardware_authorized=False,
    )
