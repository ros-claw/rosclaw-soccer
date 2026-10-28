"""Export own-state sampled receiver episodes after full physical review."""

import argparse
import json
from pathlib import Path

import numpy as np

from rosclaw_soccer.providers.g1.velocity_policy import VelocityPolicyParameters
from rosclaw_soccer.rsi.passer_curriculum_outcome import strict_pass_receive
from rosclaw_soccer.rsi.receiver_onpolicy_episode import (
    build_receiver_episode,
    receiver_terminal_reward,
)
from rosclaw_soccer.rsi.receiver_strict_terminal_reward import (
    admit_strict_training,
    strict_terminal_reward,
)
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean
from scripts.rsi_review_relay_balance_counterfactual import bind, measured, report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pair", nargs=2, action="append", required=True, metavar=("SOURCE", "AUDIT"))
    p.add_argument("--review", type=Path, required=True)
    for name in ("review-hash", "review-head", "producer-head", "expected-head"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--receiver-center-strict", action="store_true")
    a = p.parse_args()
    clean(a.expected_head)
    if (
        len(a.pair) != 3
        or a.output.exists()
        or a.output.resolve().is_relative_to(Path.cwd())
        or a.review.is_symlink()
        or digest(a.review) != a.review_hash
    ):
        raise ValueError("complete pinned paired review and unused external output required")
    review = json.loads(a.review.read_text())
    if (
        review.get("status") != "RELAY_RECEIVER_EXPLORATION_PAIRED_NOT_TRAINED"
        or review.get("source_head") != a.review_head
        or review.get("conditions") != 24
        or len(review.get("records", [])) != 72
        or len(review.get("source_pins", [])) != 3
        or review.get("zero_parent_byte_identical") is not True
        or review.get("own_physics_neural_and_density_audited") is not True
        or any(
            review.get(k) is not False
            for k in ("optimizer_authorized", "promotion_authorized", "hardware_authorized")
        )
    ):
        raise ValueError("complete unpromoted paired physical and density review required")
    pins = {r["block"]: r for r in review["source_pins"]}
    expected = {(r["context"], r["side"], r["seed"]): r for r in review["records"]}
    if len(pins) != 3 or len(expected) != 72:
        raise ValueError("complete unique seed and original condition coverage required")
    a.output.mkdir(parents=True)
    blocks, seen, rows, frozen = set(), set(), [], []
    model_identity = None
    for root_name, audit_name in a.pair:
        root, audit = Path(root_name), Path(audit_name)
        decpath = root / "declaration.json"
        dec = json.loads(decpath.read_text())
        block = dec["development_context_block"]
        if block in blocks or block not in pins:
            raise ValueError("unique pinned source blocks required")
        blocks.add(block)
        source, _, sh = report(
            root,
            audit,
            pins[block]["own_audit_hash"],
            "NEURAL_RELAY_RECEIVER_CENTER_EXPLORATION32_PENDING_OWN_REPLAY"
            if a.receiver_center_strict
            else "NEURAL_RELAY_RECEIVER_EXPLORATION32_PENDING_OWN_REPLAY",
            worlds=32,
        )
        if (
            sh != pins[block]["source_hash"]
            or source["source_head"] != a.producer_head
            or digest(decpath) != source["declaration_hash"]
        ):
            raise ValueError("exact reviewed sampled source required")
        paths = [
            Path(v)
            for v in source["input_file_hashes"]
            if v.endswith("/velocity/v0/exported/policy.onnx")
        ]
        configs = [
            Path(v)
            for v in source["input_file_hashes"]
            if v.endswith("/velocity/v0/params/deploy.yaml")
        ]
        if len(paths) != 1 or len(configs) != 1:
            raise ValueError("unique original velocity motor and codec required")
        identity = (
            str(paths[0]),
            source["input_file_hashes"][str(paths[0])],
            str(configs[0]),
            source["input_file_hashes"][str(configs[0])],
        )
        if model_identity is not None and identity != model_identity:
            raise ValueError("all episodes must share exact original velocity reference")
        model_identity = identity
        params = VelocityPolicyParameters.load(configs[0], expected_hash=identity[3])
        scale = np.asarray(params.action_scale, np.float32)
        if scale.shape != (29,) or not np.isfinite(scale).all() or not (scale > 0).all():
            raise ValueError("exact positive 29-action scale required")
        for trial in source["records"]:
            seed = trial["course"]["arm"]["exploration_seed"]
            if seed is None:
                continue
            key = (trial["course"]["context"], trial["course"]["lateral_m"], seed)
            if key in seen or key not in expected or seed not in (92117, 92143, 92171):
                raise ValueError("unique reviewed original condition and fixed seed required")
            seen.add(key)
            actual = measured(root, trial)
            if actual != expected[key]["candidate"]:
                raise ValueError("physical candidate differs from paired review")
            velocity, physics = bind(root, trial, "receiver_velocity"), bind(root, trial, "physics")
            if velocity is None or physics is None:
                raise ValueError("complete actual own neural and physical trajectories required")
            with (
                np.load(velocity, allow_pickle=False) as v,
                np.load(physics, allow_pickle=False) as raw,
            ):
                clean_outcome = strict_pass_receive(
                    raw["ball_forces"],
                    raw["body_failed"],
                    handoff=trial["result"]["handoff_executed"],
                )
                if a.receiver_center_strict:
                    if clean_outcome != expected[key].get("physical_clean_outcome"):
                        raise ValueError("strict physical outcome must match independent review")
                    score = strict_terminal_reward(clean_outcome)
                else:
                    score = receiver_terminal_reward(
                        handoff=actual["handoff"],
                        precise=actual["return_0p1m_and_healthy"],
                        kick=actual["outgoing_foot_strike"],
                        body_failure=actual["body_failure"],
                    )
                episode = build_receiver_episode(dict(v), reward=score)
                n = len(episode["inputs"])
                if n != expected[key]["stochastic_controls"] or raw["ball_forces"].shape[0] < n:
                    raise ValueError("exact own sampled controls and physical horizon required")
                nonfoot = bool((raw["ball_forces"][:n, :, 1, 2] > 1).any())
            target = a.output / f"episode-{len(rows):02d}.npz"
            np.savez_compressed(target, **episode, action_scale=scale)
            rows.append(
                dict(
                    context=key[0],
                    side=key[1],
                    seed=key[2],
                    source_block=block,
                    source_trial=trial["index"],
                    source_velocity_hash=digest(velocity),
                    source_physics_hash=digest(physics),
                    path=target.name,
                    hash=digest(target),
                    controls=n,
                    terminal_reward=score,
                    actual=actual,
                    receiver_nonfoot_contact_during_sampling=nonfoot,
                    physical_clean_outcome=clean_outcome if a.receiver_center_strict else None,
                )
            )
        frozen.extend(((root / "complete.json", sh), (audit, pins[block]["own_audit_hash"])))
    if (
        blocks != set(pins)
        or seen != set(expected)
        or len(rows) != 72
        or digest(a.review) != a.review_hash
        or any(digest(path) != h for path, h in frozen)
    ):
        raise ValueError("all reviewed sampled episodes and unchanged sources required")
    clean(a.expected_head)
    admission = admit_strict_training(rows) if a.receiver_center_strict else None
    write_once(
        a.output / "complete.json",
        dict(
            status="RECEIVER_ONPOLICY_PHYSICAL_EPISODES_NOT_TRAINED",
            source_head=a.expected_head,
            review_hash=a.review_hash,
            conditions=24,
            episodes=72,
            sampled_controls=sum(r["controls"] for r in rows),
            model_identity=model_identity,
            reward="2*strict_both_clean - A_nonfoot - B_nonfoot - either_body_failure"
            if a.receiver_center_strict
            else "2*precise_healthy + handoff + .25*return_foot - 2*either_body_failure",
            reward_profile="center-strict-full-history-v1"
            if a.receiver_center_strict
            else "legacy-mixed",
            research_data_sufficiency=admission,
            optimizer_authorized=False,
            terminal_only=True,
            actor_updates=0,
            critic_updates=0,
            online_rl=False,
            records=rows,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
