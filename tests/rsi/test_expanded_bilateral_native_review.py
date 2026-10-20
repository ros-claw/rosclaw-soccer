"""Explicit completed-case fixtures; not a claim that their parent shards closed."""

import json
import os
from pathlib import Path

import pytest


@pytest.mark.parametrize("kind,expected_steps", [("DUAL", 31000), ("SHARED", 28200)])
def test_external_completed_expanded_case(kind, expected_steps):
    key = "ROSCLAW_EXPANDED40_" + kind
    location = os.environ.get(key + "_FIXTURE")
    if location is None:
        pytest.skip("explicit completed native expanded fixture required")
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    folder = Path(location)
    assert digest(folder / "complete.json") == os.environ[key + "_CASE_SHA"]
    assert digest(folder.parent / "declaration.json") == os.environ[key + "_DECL_SHA"]
    row = json.loads((folder / "complete.json").read_text())
    declaration = json.loads((folder.parent / "declaration.json").read_text())
    assert declaration["schema"] == "soccer.rsi.bilateral_expanded40.v1"
    check_pins(declaration["input_file_hashes"])
    inputs = [Path(p) for p in declaration["input_file_hashes"]]
    worlds = [p for p in inputs if p.suffix == ".mjb"]
    csvs = [p for p in inputs if p.name == "BGI_Shooting_00001.csv"]
    priors = [p for p in inputs if p.name == "entry-affordance.json"]
    assert len(worlds) == len(csvs) == len(priors) == 1
    import mujoco
    import numpy as np

    from rosclaw_soccer.rsi.ball_entry_teacher import SkillEntryAffordance
    from rosclaw_soccer.sim.contracts import hash_json
    from scripts.rsi_expanded_bilateral_review_common import replay_case

    stored = json.loads(priors[0].read_text())
    prior = SkillEntryAffordance(
        tuple(stored["ball_offset_xy_m"]),
        stored["historical_frame"],
        stored["historical_forward_speed_mps"],
        stored["historical_artifact_hash"],
    )
    assert prior.contract_hash == stored["contract_hash"]
    model = mujoco.MjModel.from_binary_path(str(worlds[0]))
    source = np.loadtxt(csvs[0], delimiter=",", skiprows=1, dtype=np.float64)
    progress = {}
    result = replay_case(
        model,
        folder,
        row,
        poses=source,
        affordance=prior,
        evidence_hash=hash_json(declaration),
        world_hash=digest(worlds[0]),
        progress=progress,
    )
    assert result["additional_native_steps"] == progress["completed_steps"] == expected_steps
    assert progress["active_body"] is None and result["neural_math_replayed"] is False
    assert result["shared_endpoint"] == (kind == "SHARED")
    check_pins(declaration["input_file_hashes"])
    print("COMPLETED_EXPANDED_NATIVE_CASE", kind, expected_steps, flush=True)
