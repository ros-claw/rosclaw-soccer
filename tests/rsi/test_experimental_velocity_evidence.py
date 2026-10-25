import json
from pathlib import Path

import pytest

from rosclaw_soccer.rsi.experimental_velocity_evidence import admit_heading_evidence, digest


def evidence(tmp_path):
    codec = (
        Path(__file__).parents[2]
        / "src/rosclaw_soccer/providers/g1/experimental_velocity_envelope.py"
    )
    root = tmp_path / "points"
    root.mkdir()
    artifact = root / "fixture.npz"
    artifact.write_bytes(b"contract fixture only, no physical qualification")
    rows = [
        dict(
            course=course,
            kind=kind,
            repeat=repeat,
            point_feasible=True,
            no_ball_contact=True,
            healthy_full_horizon=True,
            actual_effort_bound_pass=True,
            native_steps=6000,
            fresh_reexecution_exact={"qpos": True} if repeat else None,
            artifact=dict(path=artifact.name, hash=digest(artifact)),
        )
        for course in (
            "standing",
            "forward-1",
            "forward-1p5",
            "forward-2",
            "lateral-positive-0p8",
            "lateral-negative-0p8",
        )
        for kind in (
            ("legacy", "experimental") if course in ("standing", "forward-1") else ("experimental",)
        )
        for repeat in range(2)
    ]
    report = dict(
        status="EXPERIMENTAL_VELOCITY_POINTS_MEASURED_NOT_A_DOMAIN_QUALIFICATION",
        records=rows,
        input_file_hashes={str(codec): digest(codec)},
    )
    rp = root / "complete.json"
    rp.write_text(json.dumps(report))
    (root / "declaration.json").write_text(
        json.dumps(
            dict(
                protocol="native_one_body_experimental_velocity_heading_hold.v2",
            )
        )
    )
    ref = tmp_path / "reference.py"
    ref.write_text("# fixture source")
    cfg = dict(
        schema="soccer.native_single_finisher_experimental_approach.v1",
        point_root=str(root),
        point_hash=digest(rp),
        reference_source=str(ref),
        reference_hash=digest(ref),
    )
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(cfg))
    return path, rp, report, cfg, artifact


def test_explicit_contract_and_asset_binding_without_claim_of_domain_qualification(tmp_path):
    path, _, _, _, _ = evidence(tmp_path)
    assets, pins = admit_heading_evidence(path, digest(path))
    assert assets["experimental_fast"] is True and str(path) in pins


@pytest.mark.parametrize("defect", ["duplicate", "unhealthy", "modified_raw", "bad_hash"])
def test_malformed_or_changed_point_evidence_cannot_allocate_policy(tmp_path, defect):
    path, rp, report, cfg, artifact = evidence(tmp_path)
    if defect == "duplicate":
        report["records"][1] = report["records"][0]
    elif defect == "unhealthy":
        report["records"][0]["healthy_full_horizon"] = False
    elif defect == "modified_raw":
        artifact.write_bytes(b"changed")
    else:
        cfg["reference_hash"] = "sha256:" + "0" * 64
    rp.write_text(json.dumps(report))
    cfg["point_hash"] = digest(rp)
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError):
        admit_heading_evidence(path, digest(path))
