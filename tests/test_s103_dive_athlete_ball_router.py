from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.dynamic_corner_save import (
    validate_dynamic_corner_evidence,
)

_EVIDENCE = Path(
    "/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/"
    "s103-dive-athlete-multi-corner-v5/evidence.json"
)


def test_current_s103_context_router_is_content_bound_when_available() -> None:
    if not _EVIDENCE.is_file():
        pytest.skip("S103 external CPU MuJoCo evidence is unavailable")

    report = validate_dynamic_corner_evidence(_EVIDENCE)

    assert report["passed"] is True
    assert report["portfolio_gates"]["dive_athlete_bound_all_lanes"] is True
    assert report["dive_athlete"]["authority_by_lane"] == {
        "left-inner": 0.05,
        "left-outer": 0.75,
        "right-inner": 0.5,
        "right-outer": 0.75,
    }
    assert all(case["result"]["goalkeeper_save_observed"] for case in report["cases"].values())


def test_s103_validator_rejects_a_forged_lane_authority_when_available(
    tmp_path: Path,
) -> None:
    if not _EVIDENCE.is_file():
        pytest.skip("S103 external CPU MuJoCo evidence is unavailable")
    copied = tmp_path / "evidence"
    shutil.copytree(_EVIDENCE.parent, copied)
    evidence_path = copied / "evidence.json"
    request_path = copied / "request.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    request = json.loads(request_path.read_text(encoding="utf-8"))
    evidence["dive_athlete"]["authority_by_lane"]["left-inner"] = 0.75
    request["dive_athlete"]["authority_by_lane"]["left-inner"] = 0.75
    request_path.write_text(json.dumps(request), encoding="utf-8")
    evidence["request_hash"] = hash_bytes(request_path.read_bytes())
    evidence.pop("report_hash")
    evidence["report_hash"] = hash_json(evidence)
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")

    with pytest.raises(ValueError, match="lane authority receipt"):
        validate_dynamic_corner_evidence(evidence_path)
