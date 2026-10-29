"""Bounded, paired, SIM-only physical teacher feasibility search."""

import argparse
import json
import subprocess
from pathlib import Path

from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_native_context_recovery_common import check_pins, digest
from scripts.rsi_native_fresh_common import clean, source_pins


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--external", type=Path, required=True)
    p.add_argument("--baselines", type=Path, nargs=2, required=True)
    p.add_argument("--baseline-hashes", nargs=2, required=True)
    p.add_argument("--zero-receipts", type=Path, nargs=2, required=True)
    p.add_argument("--zero-hashes", nargs=2, required=True)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--protocol-hash", required=True)
    p.add_argument("--detail", type=Path, required=True)
    p.add_argument("--detail-hash", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    clean(a.expected_head)
    if a.output.exists():
        raise ValueError("unused output required")
    pins = {}
    for path, h in ((a.protocol, a.protocol_hash), (a.detail, a.detail_hash)):
        if path.is_symlink() or digest(path) != h:
            raise ValueError("exact predeclared protocol required")
        pins[str(path.resolve())] = h
    sources = []
    for i, (path, h, zero_path, zero_h) in enumerate(
        zip(a.baselines, a.baseline_hashes, a.zero_receipts, a.zero_hashes, strict=True)
    ):
        if (
            any(x.is_symlink() for x in (path, zero_path))
            or digest(path) != h
            or digest(zero_path) != zero_h
        ):
            raise ValueError("exact paired baseline and zero-audit receipts required")
        baseline, zero = json.loads(path.read_text()), json.loads(zero_path.read_text())
        if (
            zero["status"] != "EXACT_ZERO_BASELINE"
            or zero["baseline_hash"] != h
            or zero["ground_zero_spin"] != bool(i)
            or zero["controls"] != 300
            or zero["runtime_failure"] is not None
            or not all(x["exact"] for x in zero["comparisons"].values())
            or baseline["reference_source"] != "NPZmotion"
        ):
            raise ValueError("both exact source-bound zero branches required")
        for name, commitment in baseline["input_file_hashes"].items():
            if name in pins and pins[name] != commitment:
                raise ValueError("paired lineage commitments disagree")
            pins[name] = commitment
        for file, commitment in ((path, h), (zero_path, zero_h)):
            pins[str(file.resolve())] = commitment
        artifact = path.parent / baseline["artifact"]["path"]
        pins[str(artifact.resolve())] = baseline["artifact"]["hash"]
        private = zero_path.parent / "private-forecast.npz"
        pins[str(private.resolve())] = zero["artifact_hash"]
        sources.append((baseline, artifact))
    external = a.external.resolve()
    head = subprocess.check_output(
        ["git", "-C", str(external), "rev-parse", "HEAD"], text=True
    ).strip()
    if subprocess.check_output(["git", "-C", str(external), "status", "--porcelain"]) or any(
        row[0]["external_head"] != head for row in sources
    ):
        raise ValueError("unchanged qualified external checkout required")
    scene = [Path(x) for x in pins if x.endswith("/cached-standard/declared-scene.xml")]
    reference = Path(sources[0][0]["motion_source"])
    if (
        len(scene) != 1
        or str(reference) not in pins
        or sources[1][0]["motion_source"] != str(reference)
    ):
        raise ValueError("unique paired source scene and reference required")
    source_pins(pins)
    check_pins(pins)
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            external_head=head,
            input_file_hashes=pins,
            protocol_hash=a.protocol_hash,
            detail_hash=a.detail_hash,
            generations=2,
            candidates_per_generation=16,
            cases=2,
            native_step_budget=192000,
            additional_zero_native_step_budget=6000,
            policy_history_uses_unmodified_nn_actions=True,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    import numpy as np

    from rosclaw_soccer.rsi.cftrack_motor_residual_plan import CFtrackMotorResidualPlan
    from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
    from rosclaw_soccer.rsi.omnicontact_policy_copy import cftrack_state_hash
    from rosclaw_soccer.rsi.shared_cftrack_forecast import SharedCFtrackForecaster
    from rosclaw_soccer.rsi.whole_leg_cem import (
        case_score,
        initial_population,
        pair_rank,
        population,
        update_elites,
    )
    from scripts.rsi_cftrack_native_context import build_reference_cftrack

    players, model_hashes, zero_scores = [], [], []
    for i in range(2):
        player, model_hash = build_reference_cftrack(
            scene[0].read_text(),
            external,
            reference,
            ground_zero_spin=bool(i),
            reuse_qualified_imports=bool(i),
        )
        players.append(player)
        model_hashes.append(model_hash)
        zero_plan = CFtrackMotorResidualPlan(
            origin_counter=player.policy.counter_step,
            start_control=21,
            peak_control=24,
            delta_hg=(0.0,) * 29,
            donor_policy_hash=cftrack_state_hash(player.policy, reference_source="NPZmotion"),
        )
        zero_forecast = SharedCFtrackForecaster(
            player,
            expected_policy_type=type(player.policy),
            expected_model_hash=model_hash,
            native_step_budget=3000,
            reference_source="NPZmotion",
        ).forecast(controls=300, motor_plan=zero_plan, capture_inference=True)
        keys = (
            "qpos",
            "qvel",
            "torque",
            "ball_forces",
            "joint_targets",
            "nn_decoded_targets",
            "neural_obs",
            "neural_actions",
        )
        with np.load(sources[i][1], allow_pickle=False) as old:
            comparisons = {
                k: bool(np.array_equal(zero_forecast[k], old[k][: len(zero_forecast[k])]))
                for k in keys
            }
        write_once(
            a.output / f"local-zero-{i}.json",
            dict(
                controls=zero_forecast["controls"],
                runtime_failure=zero_forecast["runtime_failure"],
                comparisons=comparisons,
            ),
        )
        if (
            zero_forecast["controls"] != 300
            or zero_forecast["runtime_failure"]
            or not all(comparisons.values())
        ):
            raise ValueError("current-source local zero mismatch; no candidate search")
        layout = NativeVelocityLayout(
            player.world.model, robot_prefix="a/", ball_body_name="soccer_ball"
        )
        zero_scores.append(case_score(zero_forecast, layout.bq))
    rng, mean, sigma = initial_population()
    all_rows, ranked_candidates = [], []
    for generation in range(2):
        raw, candidates = population(rng, mean, sigma)
        np.savez_compressed(
            a.output / f"generation-{generation}-proposals.npz",
            raw=raw,
            candidates=candidates,
            mean=mean,
            sigma=sigma,
        )
        forecasts = [
            SharedCFtrackForecaster(
                player,
                expected_policy_type=type(player.policy),
                expected_model_hash=model_hash,
                native_step_budget=48000,
                reference_source="NPZmotion",
            )
            for player, model_hash in zip(players, model_hashes, strict=True)
        ]
        ranks = []
        for index, candidate in enumerate(candidates):
            outcomes = []
            for case, (player, forecaster) in enumerate(zip(players, forecasts, strict=True)):
                delta = tuple(float(x) for x in np.r_[candidate, np.zeros(17)])
                plan = CFtrackMotorResidualPlan(
                    origin_counter=player.policy.counter_step,
                    start_control=21,
                    peak_control=24,
                    delta_hg=delta,
                    donor_policy_hash=cftrack_state_hash(
                        player.policy, reference_source="NPZmotion"
                    ),
                )
                result = forecaster.forecast(controls=300, motor_plan=plan, capture_inference=True)
                path = a.output / f"g{generation}-c{index:02d}-case{case}.npz"
                np.savez_compressed(
                    path, **{k: v for k, v in result.items() if isinstance(v, np.ndarray)}
                )
                layout = NativeVelocityLayout(
                    player.world.model, robot_prefix="a/", ball_body_name="soccer_ball"
                )
                score = case_score(result, layout.bq)
                metadata = dict(
                    generation=generation,
                    candidate=index,
                    case=case,
                    score=score,
                    controls=result["controls"],
                    body_failed=result["body_failed"],
                    runtime_failure=result["runtime_failure"],
                    artifact_hash=digest(path),
                    counters=forecaster.counters(),
                )
                write_once(path.with_suffix(".json"), metadata)
                outcomes.append(score)
                all_rows.append(metadata)
            ranks.append(pair_rank(*outcomes))
            ranked_candidates.append((ranks[-1], generation, index, candidate.tolist()))
        mean, sigma, elites = update_elites(candidates, ranks)
        write_once(
            a.output / f"generation-{generation}-selection.json",
            dict(
                ranks=ranks,
                elites=elites,
                next_mean=mean.tolist(),
                next_sigma=sigma.tolist(),
            ),
        )
    check_pins(pins)
    clean(a.expected_head)
    anchor_rank = pair_rank(*zero_scores)
    best = max(ranked_candidates, key=lambda x: (x[0], -x[1], -x[2]))
    selected = (
        dict(kind="candidate", generation=best[1], index=best[2], delta_hg12=best[3], rank=best[0])
        if best[0] > anchor_rank
        else dict(kind="zero_anchor", delta_hg12=[0.0] * 12, rank=anchor_rank)
    )
    write_once(
        a.output / "complete.json",
        dict(
            status="SEARCH_RECORDED_NOT_PROMOTED",
            source_head=a.expected_head,
            results=all_rows,
            candidates=32,
            physical_world_rollouts=64,
            zero_scores=zero_scores,
            selected=selected,
            optimizer_updates=0,
            hardware_authorized=False,
            promotion_authorized=False,
            team_game_verified=False,
        ),
    )


if __name__ == "__main__":
    main()
