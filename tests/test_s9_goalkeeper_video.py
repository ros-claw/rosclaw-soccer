from __future__ import annotations

import json

import pytest

from rosclaw_soccer.evidence.goalkeeper_v2 import goalkeeper_v2_implementation_hash
from rosclaw_soccer.media.goalkeeper_v2_video import (
    render_goalkeeper_v2_development_video,
)


def test_goalkeeper_video_rejects_promoted_or_mismatched_candidate(tmp_path) -> None:
    evidence = tmp_path / "evidence.json"
    evidence.write_text(
        json.dumps(
            {
                "evidence_hash": "sha256:" + "0" * 64,
                "implementation_hash": goalkeeper_v2_implementation_hash(),
                "promotion_decision": {
                    "verdict": "PROMOTED",
                    "candidate_policy_hash": "sha256:" + "1" * 64,
                },
            }
        ),
        encoding="utf-8",
    )
    actor = tmp_path / "actor.json"
    actor.write_text(json.dumps({"policy_hash": "sha256:" + "1" * 64}), encoding="utf-8")

    with pytest.raises(ValueError, match="rejected candidate"):
        render_goalkeeper_v2_development_video(
            candidate_evidence_path=evidence,
            actor_artifact_path=actor,
            asset_root=tmp_path,
            output_path=tmp_path / "out.mp4",
            source_checkout=tmp_path / "source",
        )
