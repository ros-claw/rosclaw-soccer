"""Synthetic admission metadata only, never physical Fresh evidence."""

import json
from types import SimpleNamespace

import pytest

from scripts.rsi_native_context_recovery_common import digest
from scripts.rsi_pass_fresh_common import (
    SEAL_STATUS,
    admit_pass_seal,
    confined_artifact,
    require_frozen_pass_protocol,
    verify_pass_scene,
)


def fixture(tmp_path):
    root = tmp_path / "seal"
    root.mkdir(mode=0o700)
    private = root / "private-courses.json"
    private.write_text(json.dumps(dict(seed=42, courses=[{}] * 96)))
    initial = root / "private-initial-states.json"
    initial.write_text(json.dumps(["sha256:" + f"{i:064x}" for i in range(48)]))
    reports = [
        dict(
            seed=s,
            eligible_for_diagnostic_Fresh=s != 105023,
            retention=dict(acquired_frontier_retained=s != 105023),
            pass_summary=dict(candidate_controlled=41, baseline_controlled=23),
        )
        for s in (105023, 105024, 105025)
    ]
    d = dict(
        protocol="soccer.pass_near_2D48_sealed.v1",
        input_file_hashes={},
        models=[dict(seed=s) for s in (65023, 105024, 105025)],
        reports=reports,
        independent_initial_states=48,
        goal_tasks=96,
        shards=4,
        physical_steps=0,
        fit_rows=0,
        weights_frozen_before_sampling=True,
        all_states_preflighted_before_workers=True,
        sampling_adapts_to_results=False,
        private_case_rows_printed=False,
        hardware_authorized=False,
        promotion_authorized=False,
        goal_complete=False,
        courses_hash=digest(private),
        initial_states_hash=digest(initial),
    )
    dp = root / "declaration.json"
    dp.write_text(json.dumps(d))
    cp = root / "complete.json"
    cp.write_text(json.dumps(dict(d, status=SEAL_STATUS, declaration_hash=digest(dp))))
    return root, digest(cp)


def test_complete_seal_admission_commits_private_files_without_revealing_rows(tmp_path, capsys):
    root, h = fixture(tmp_path)
    seal, pins, private, states = admit_pass_seal(root, h)
    assert len(private["courses"]) == 96 and len(states) == 48
    assert seal["fit_rows"] == 0 and str(root / "private-courses.json") in pins
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("fault", ["sample", "state", "declare", "failed", "hash"])
def test_private_sample_or_frozen_declaration_change_cannot_be_silently_replaced(tmp_path, fault):
    root, h = fixture(tmp_path)
    if fault == "sample":
        (root / "private-courses.json").write_text("{}")
    elif fault == "state":
        (root / "private-initial-states.json").write_text("[]")
    elif fault == "declare":
        (root / "declaration.json").write_text("{}")
    elif fault == "failed":
        (root / "failed.json").write_text("{}")
    elif fault == "hash":
        h = "sha256:" + "0" * 64
    with pytest.raises(ValueError):
        admit_pass_seal(root, h)


def test_frozen_protocol_code_cannot_change_sampling_or_screen_after_seal():
    paths = {
        "/old/src/rosclaw_soccer/rsi/pass_fresh_curriculum.py": "sha256:a",
        "/new/src/rosclaw_soccer/rsi/pass_fresh_curriculum.py": "sha256:a",
    }
    require_frozen_pass_protocol(paths)
    paths["/new/src/rosclaw_soccer/rsi/pass_fresh_curriculum.py"] = "sha256:b"
    with pytest.raises(ValueError):
        require_frozen_pass_protocol(paths)
    with pytest.raises(ValueError):
        require_frozen_pass_protocol({})


def test_raw_artifacts_cannot_escape_policy_directory_or_change_content(tmp_path):
    root = tmp_path / "policy"
    root.mkdir()
    file = root / "raw.npz"
    file.write_bytes(b"fixture")
    assert confined_artifact(root, "raw.npz", digest(file)) == file
    other = tmp_path / "outside.npz"
    other.write_bytes(b"fixture")
    with pytest.raises(ValueError):
        confined_artifact(root, "../outside.npz", digest(other))
    link = root / "link.npz"
    link.symlink_to(other)
    with pytest.raises(ValueError):
        confined_artifact(root, "link.npz", digest(other))
    with pytest.raises(ValueError):
        confined_artifact(root, "raw.npz", "sha256:other")


def test_compiled_tuple_inertia_matches_json_but_numeric_scene_changes_do_not():
    from rosclaw_soccer.sim.contracts import hash_json

    setup = SimpleNamespace(
        xml="<model/>",
        ball_dimensions=dict(
            radius_m=0.11, body_mass_kg=0.43, principal_inertia_kg_m2=(0.001, 0.001, 0.001)
        ),
    )
    seal = dict(
        scene_hash=hash_json(setup.xml), compiled_ball=json.loads(json.dumps(setup.ball_dimensions))
    )
    verify_pass_scene(setup, seal)
    setup.ball_dimensions["body_mass_kg"] += 0.001
    with pytest.raises(ValueError):
        verify_pass_scene(setup, seal)
