"""Fixed SIM paired controls: follow-through or precontact closing gain."""

import fcntl
import gc
import os
import signal
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.bounded_first_touch_control import BoundedFirstTouchControl
from rosclaw_soccer.rsi.first_touch_campaign import TERMINAL_PHASES, FirstTouchCampaign
from rosclaw_soccer.rsi.first_touch_campaign_worker import atomic_status, prepare, qualified_episode
from rosclaw_soccer.rsi.postcontact_navigation_probe import PostcontactNavigationProbe

CLOSING_PROFILE = "closing_gain_single_factor_eleven"
CLOSING_GROUPS = [0, 4, 8, 20, 24, 28, 36, 40, 44, 56, 60]


def require_declaration(config: dict[str, Any]) -> None:
    profiles = {
        "failure_directed_four": ([4, 12, 32, 52], 10),
        "balanced_public_sixteen_retention": (list(range(0, 64, 4)), 34),
        CLOSING_PROFILE: (CLOSING_GROUPS, 24),
    }
    profile = config.get("postcontact_study_profile", "failure_directed_four")
    if type(profile) is not str or profile not in profiles:
        raise ValueError("explicit fixed postcontact study profile required")
    groups, attempts = profiles[profile]
    if (
        config.get("postcontact_study_groups") != groups
        or config.get("max_attempts") != attempts
        or config.get("predecessor") is not None
        or config.get("neural_training_authorized") is not False
        or config.get("teacher_auto_admission") is not False
    ):
        raise ValueError("complete fixed paired diagnostic declaration required")
    if profile == CLOSING_PROFILE:
        if (
            "postcontact_probe" in config
            or type(config.get("closing_gain_probe_s_inv")) is not float
            or config["closing_gain_probe_s_inv"] != 3.0
        ):
            raise ValueError("single closing-gain factor without postcontact intervention required")
        control = BoundedFirstTouchControl(closing_gain_s_inv=config["closing_gain_probe_s_inv"])
        for gap in (0.2, 0.5, 1.0, 2.8):
            control.navigation(gap_x=gap, gap_y=0.1, ball_vx=0.6, before_contact=True)
        return
    if "closing_gain_probe_s_inv" in config or config.get("postcontact_probe") != asdict(
        PostcontactNavigationProbe()
    ):
        raise ValueError("fixed postcontact intervention without closing-gain changes required")
    # Exercise real forward-reference computation before neural allocation.
    probe = PostcontactNavigationProbe(**config["postcontact_probe"])
    for previous in (0.0, 0.7, 1.4):
        for gap in (-0.2, 0.1, 0.55, 1.0):
            raw, actual = probe.navigation(
                previous_forward=previous, gap_x=gap, ball_vx=0.0, forward_cap=1.4
            )
            if not np.isfinite(raw) or not 0.0 <= actual <= 1.4:
                raise ValueError("postcontact reference interface preflight failed")


def paired_candidate(config: dict[str, Any], arm: str) -> dict[str, Any]:
    require_declaration(config)
    if arm not in ("baseline", "probe"):
        raise ValueError("fixed paired arm required")
    parameters = asdict(BoundedFirstTouchControl())
    candidate: dict[str, Any] = dict(parameters=parameters, swing="bounded_contact_diagnostic")
    if arm == "probe":
        if config.get("postcontact_study_profile") == CLOSING_PROFILE:
            parameters["closing_gain_s_inv"] = config["closing_gain_probe_s_inv"]
        else:
            candidate["postcontact"] = config["postcontact_probe"]
    return candidate


def verify_initial_pair(baseline: dict[str, Any], probe: dict[str, Any]) -> None:
    """Gain intervention starts at frame0; compare initial physical state only.

    Navigation/NN inputs can immediately differ. This is not precontact parity
    and cannot establish identical policy trajectories after the intervention.
    """
    with (
        np.load(baseline["witnesses"]["trace"]["path"], allow_pickle=False) as a,
        np.load(probe["witnesses"]["trace"]["path"], allow_pickle=False) as b,
    ):
        for key in (
            "root_pose_xyzw_m",
            "root_velocity_world",
            "ball_position_before_step_m",
            "ball_linear_velocity_before_step_m_s",
            "foot_geometry_position_before_step_m",
            "joint_position_rad",
            "joint_velocity_rad_s",
            "canonical_qpos",
            "canonical_qvel",
        ):
            if (
                a[key].shape != b[key].shape
                or a[key].shape[0] != 300
                or a[key].dtype != b[key].dtype
                or not np.isfinite(a[key][0]).all()
                or not np.array_equal(a[key][0], b[key][0])
            ):
                raise ValueError(f"initial paired physical state changed: {key}")


def verify_precontact_pair(baseline: dict[str, Any], probe: dict[str, Any]) -> None:
    first = baseline["outcome"]["first_foot_contact_frame"]
    # The intervention begins after ANY contact, not just qualifying foot contact.
    with (
        np.load(baseline["witnesses"]["trace"]["path"], allow_pickle=False) as a,
        np.load(probe["witnesses"]["trace"]["path"], allow_pickle=False) as b,
    ):
        if set(a.files) != set(b.files):
            raise ValueError("paired trace fields changed")
        contacts = np.flatnonzero(np.any(a["force_n"][:, 0] > 1.0, axis=1))
        stop = int(contacts[0]) + 1 if len(contacts) else 300
        if first is not None and first < stop - 1:
            raise ValueError("invalid first-contact ordering")
        for key in a.files:
            if a[key].shape != b[key].shape or a[key].dtype != b[key].dtype:
                raise ValueError("paired trace layout changed")
            if not np.array_equal(a[key][:stop], b[key][:stop]):
                raise ValueError(f"pre-intervention paired observations changed: {key}")


def run(config_path: Path, *, preflight: bool = False) -> None:
    p, assets, manifest, config = prepare(config_path)
    require_declaration(config)
    if config.get("postcontact_study_profile") == "balanced_public_sixteen_retention":
        prior_path = Path(config["prior_study_report_path"])
        prior_digest = p.digest(prior_path)
        prior = p.sealed(prior_path)
        if (
            prior.get("report_hash") != config["prior_study_report_hash"]
            or prior.get("schema") != "soccer.rsi.qualified_postcontact_paired_study.v1"
            or prior.get("completed_pairs") != 4
            or prior.get("source_physical_executions") != 8
            or prior.get("source_independent_physical_substeps") != 24000
            or prior.get("source_foundation_recomputed_calls") != 2400
            or prior.get("source_sampling_draws_reconstructed") != 2160
            or prior.get("probe_high_quality", 0) <= prior.get("baseline_high_quality", 4)
            or prior.get("teacher_qualified") is not False
            or prior.get("promotion_authorized") is not False
            or prior.get("hardware_authorized") is not False
            or p.digest(prior_path) != prior_digest
        ):
            raise ValueError("completed qualified SIM pilot with a local gain required")
        p.add_pins({str(prior_path): prior_digest})
    if preflight:
        print("POSTCONTACT_PAIRED_PREFLIGHT_NO_NEURAL_OR_PHYSICS_ALLOCATION", flush=True)
        return
    root, source, core = (Path(config[k]) for k in ("output_root", "source_root", "core_root"))
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / "owner.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    campaign = FirstTouchCampaign(
        root, dict(config, input_file_hashes=dict(p.PINS), source_heads=p.HEADS)
    )
    if campaign.state["phase"] in TERMINAL_PHASES:
        campaign.db.close()
        lock.close()
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
                    active=campaign.state["active"],
                    completed=len(campaign.state["completed"]),
                    neural_optimizer_updates=0,
                    promotion_authorized=False,
                    hardware_authorized=False,
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
            if ram >= 48 * 1024**3:
                return True
            if campaign.state["phase"] != "WAITING_RESOURCES":
                campaign.state["phase"] = "WAITING_RESOURCES"
                campaign.record("WAITING_RESOURCES")
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
        model = load_json_artifact(assets.EXPORT / "model-0.json.gz")
        if model["model_hash"] != assets.MEAN:
            raise ValueError("complete frozen parent identity required")
        factory = RecurrentSamplingEpisodeFactory(model)
        del model
        old = _sealed(
            Path(manifest["records"][0]["witnesses"]["trace"]["path"]).parent / "report.json"
        )
        compiler = SamplingReferenceAuditCompiler(old["executed_motor_policy"])
        del old
        gc.collect()
        auditor = TorchCallAuditor(assets.SONIC)
        closing = config.get("postcontact_study_profile") == CLOSING_PROFILE
        prefix = "closing" if closing else "postcontact"
        for group in config["postcontact_study_groups"]:
            for arm in ("baseline", "probe"):
                job = f"{prefix}-{group}-{arm}"
                if job in campaign.state["completed"]:
                    continue
                if stop.is_set():
                    return
                if campaign.state["attempts"] >= config["max_attempts"]:
                    campaign.state["phase"] = "ATTEMPT_BUDGET_EXHAUSTED"
                    campaign.record("ATTEMPT_LIMIT")
                    return
                if not resources():
                    return
                p.stable()
                candidate = paired_candidate(config, arm)
                attempt = campaign.begin(job)
                receipt = qualified_episode(
                    root / f"attempt-{attempt:04d}-{job}",
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
                    entry_source_path=source / "scripts/rsi_run_postcontact_paired_study.py",
                )
                p.stable()
                campaign.complete(job, receipt)
                print(f"{prefix.upper()}_EPISODE_VERIFIED {job} {receipt['outcome']}", flush=True)
                gc.collect()
            pair_check = verify_initial_pair if closing else verify_precontact_pair
            pair_check(
                campaign.state["completed"][f"{prefix}-{group}-baseline"],
                campaign.state["completed"][f"{prefix}-{group}-probe"],
            )
            campaign.record(
                "INITIAL_PAIR_BIT_PARITY_VERIFIED"
                if closing
                else "PRECONTACT_PAIR_BIT_PARITY_VERIFIED"
            )
        campaign.state["phase"] = "PILOT_COMPLETE_EXPANDED_VALIDATION_REQUIRED"
        campaign.record("PAIRED_DIAGNOSTIC_COMPLETE_NOT_TEACHER_ADMISSION")
    except BaseException as error:
        campaign.state["phase"] = "FATAL_CONTRACT_VIOLATION"
        campaign.state["fatal_error"] = dict(type=type(error).__name__, message=str(error))
        campaign.record("FATAL_DO_NOT_AUTO_UNFREEZE")
        raise
    finally:
        if stop.is_set() and campaign.state["phase"] not in TERMINAL_PHASES:
            campaign.state["phase"] = "STOPPED_OPERATOR_REQUEST"
            campaign.record("GRACEFUL_STOP_CHECKPOINTED")
        stop.set()
        thread.join(timeout=20)
        atomic_status(root, "status.json", dict(campaign.state, ledger_tip=campaign.tip))
        atomic_status(
            root,
            "heartbeat.json",
            dict(
                pid=os.getpid(),
                time_unix=time.time(),
                phase=campaign.state["phase"],
                worker_exiting=True,
                completed=len(campaign.state["completed"]),
                neural_optimizer_updates=0,
                promotion_authorized=False,
                hardware_authorized=False,
            ),
        )
        campaign.db.close()
        lock.close()
