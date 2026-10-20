"""Opt-in replay of a completed immutable left candidate, not an open shard."""

import json
import os
from pathlib import Path

import numpy as np
import pytest


def test_external_completed_left_candidate():
    location = os.environ.get("ROSCLAW_BILATERAL_REVIEW_FIXTURE")
    if location is None:
        pytest.skip("explicit immutable completed left candidate required")
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    folder = Path(location)
    path = folder / "candidate6.json"
    assert digest(path) == os.environ["ROSCLAW_BILATERAL_CANDIDATE_SHA"]
    assert digest(folder.parent / "declaration.json") == os.environ["ROSCLAW_BILATERAL_DECL_SHA"]
    candidate = json.loads(path.read_text())
    assert candidate["index"] == 6 and candidate["contact_reference_foot"] == "left"
    declaration = json.loads((folder.parent / "declaration.json").read_text())
    check_pins(declaration["input_file_hashes"])
    assert declaration["schema"] == "soccer.rsi.bilateral_entry16.v1"
    inputs = [Path(p) for p in declaration["input_file_hashes"]]
    worlds = [p for p in inputs if p.suffix == ".mjb"]
    csvs = [p for p in inputs if p.name == "BGI_Shooting_00001.csv"]
    priors = [p for p in inputs if p.name == "entry-affordance.json"]
    assert len(worlds) == len(csvs) == len(priors) == 1
    artifact = folder / candidate["artifact"]
    assert digest(artifact) == candidate["artifact_hash"]
    with np.load(artifact, allow_pickle=False) as data:
        raw = {k: data[k].copy() for k in data.files}
    with np.load(folder / "own-prefix20.npz", allow_pickle=False) as data:
        prefix = {k: data[k].copy() for k in data.files}
    import mujoco

    from rosclaw_soccer.physics.g1_target_episode import G1TargetEpisode
    from rosclaw_soccer.providers.g1.sonic_runup import (
        G1SonicRunupController,
        _sonic_control_parameters,
    )
    from rosclaw_soccer.rsi.ball_entry_teacher import SkillEntryAffordance
    from rosclaw_soccer.rsi.bilateral_entry import contact_foot_context, observed_contact_foot
    from rosclaw_soccer.rsi.full_entry_forecast import feedback_entry_laws, score_full_entry
    from rosclaw_soccer.rsi.full_task_branch import integration_hash
    from rosclaw_soccer.rsi.recorded_entry_review import replay_prefix, replay_suffix
    from rosclaw_soccer.sim.contracts import hash_json

    stored = json.loads(priors[0].read_text())
    affordance = SkillEntryAffordance(
        tuple(stored["ball_offset_xy_m"]),
        stored["historical_frame"],
        stored["historical_forward_speed_mps"],
        stored["historical_artifact_hash"],
    )
    assert affordance.contract_hash == stored["contract_hash"]
    poses = np.loadtxt(csvs[0], delimiter=",", skiprows=1, dtype=np.float64)
    source, entry, context = contact_foot_context(poses, affordance, "left")
    assert context == candidate["contact_foot_context"]
    model = mujoco.MjModel.from_binary_path(str(worlds[0]))
    kp, kd, _ = _sonic_control_parameters(1.0, (1.0,) * 29)
    variant = int(folder.name.removeprefix("variant"))
    course = next(
        item["course"]
        for item in declaration["courses"]
        if item["parent_index"] == declaration["cluster"] and item["variant_index"] == variant
    )
    body = G1TargetEpisode(
        model,
        world_hash=digest(worlds[0]),
        kp=kp,
        kd=kd,
        course=tuple(course),
        defaults=G1SonicRunupController.default_angles.copy(),
        record_all_ball_contacts=True,
    )
    replay_prefix(body, prefix, affordance)
    assert integration_hash(model, body.data) == candidate["donor_body_hash"]
    replay_suffix(
        body,
        raw,
        law=feedback_entry_laws()[candidate["law_index"]],
        offset_root_m=tuple(candidate["offset_root_m"]),
        affordance=entry,
        source_poses=source,
        source_evidence_hash=hash_json(declaration),
        report=candidate["report"],
    )
    assert body.integrated_steps == body.attempted_steps == 3000
    assert score_full_entry(model, raw) == candidate["score"]
    assert (
        observed_contact_foot(model, raw, global_start_frame=20)
        == candidate["observed_contact_foot"]
    )
    check_pins(declaration["input_file_hashes"])
    print("COMPLETED_LEFT_CANDIDATE_NATIVE_REPLAY", body.integrated_steps, flush=True)
