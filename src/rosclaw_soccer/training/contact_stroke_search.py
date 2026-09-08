"""Bounded bilateral action-duration curriculum with physical replay evidence."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from rosclaw_soccer.growth.owned_ball_contact import OwnedBallContactPolicy
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.active_team_probe import run_probe, validate_probe
from rosclaw_soccer.training.owned_contact_learning import reward


def _trial(job: tuple[str, str, bool, float, float]) -> str:
    asset, output, blue, duration, offset = job
    run_probe(
        asset_root=Path(asset),
        output=Path(output),
        active=True,
        duration=20.0,
        four_vs_four=True,
        blue_kickoff=blue,
        contact_policy=OwnedBallContactPolicy(),
        kickoff_offset_m=offset,
        pass_stroke_duration_sec=duration,
    )
    return str(Path(output) / "probe.json")


def search(*, asset_root: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    durations = (0.0, 0.18, 0.30, 0.50)
    request = {
        "method": "bounded_offline_action_duration_search",
        "duration_candidates_sec": durations,
        "training_offsets_m": [0.0],
        "development_offsets_m": [0.16, -0.16],
        "neural_network_training": False,
        "workers": 2,
        "activation_ceiling": "SIM_ONLY",
        "implementation_hash": hash_bytes(Path(__file__).read_bytes()),
    }
    (output / "request.json").write_text(json.dumps(request, indent=2) + "\n")
    with ProcessPoolExecutor(max_workers=2) as pool:
        paths = list(
            pool.map(
                _trial,
                [
                    (
                        str(asset_root),
                        str(output / f"train-{i}-{'blue' if blue else 'red'}"),
                        blue,
                        duration,
                        0.0,
                    )
                    for i, duration in enumerate(durations)
                    for blue in (False, True)
                ],
            )
        )
    scores = [reward(validate_probe(Path(path))) for path in paths]
    paired = [min(scores[i : i + 2]) for i in range(0, len(scores), 2)]
    # Baseline is eligible. No artificial winner when all challengers fail.
    selected = max(range(len(durations)), key=lambda i: (paired[i], -i))
    selection = {
        "scores": paired,
        "selected_index": selected,
        "selected_duration_sec": durations[selected],
        "selected_before_development": True,
    }
    (output / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    with ProcessPoolExecutor(max_workers=2) as pool:
        exams = list(
            pool.map(
                _trial,
                [
                    (
                        str(asset_root),
                        str(output / f"dev-{name}-{'blue' if blue else 'red'}"),
                        blue,
                        duration,
                        offset,
                    )
                    for name, duration in (("baseline", 0.0), ("selected", durations[selected]))
                    for blue, offset in ((False, 0.16), (True, -0.16))
                ],
            )
        )
    exam_scores = [reward(validate_probe(Path(path))) for path in exams]
    passed = (
        selected != 0
        and paired[selected][0] == 1
        and all(s[0] == 1 and s[1] > 0 for s in exam_scores[2:])
        and all(c[1] > b[1] for b, c in zip(exam_scores[:2], exam_scores[2:], strict=True))
    )
    result = {
        "request": request,
        "selection": selection,
        "training_scores": scores,
        "development_scores": exam_scores,
        "sources": {path: hash_bytes(Path(path).read_bytes()) for path in paths + exams},
        "status": "PASS_BILATERAL_DEVELOPMENT" if passed else "REJECTED",
        "promotion_eligible": False,
        "activation_ceiling": "SIM_ONLY",
    }
    result["report_hash"] = hash_json(result)
    (output / "search.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(search(asset_root=args.asset_root, output=args.output)))
