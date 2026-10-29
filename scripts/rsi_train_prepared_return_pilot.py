"""One bounded failure-driven SIM actor-critic update from audited own worlds."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from rosclaw_soccer.providers.g1.velocity_policy import VelocityPolicyParameters
from rosclaw_soccer.rsi.precontact_receiving_motor import PrecontactReceivingMotor
from rosclaw_soccer.rsi.receiver_actor_critic import ReceiverCritic, episode_balanced_value_loss
from rosclaw_soccer.rsi.receiver_bounded_update import bounded_actor_update
from rosclaw_soccer.rsi.return_player_outcome import (
    discounted_sample_returns,
    return_player_outcome,
)
from rosclaw_soccer.rsi.return_player_sampling_audit import audit_return_player_sampling
from rosclaw_soccer.rsi.velocity_prior_actor import VelocityPriorActor
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean
from scripts.rsi_review_relay_balance_counterfactual import bind, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--group", action="append", nargs=3, required=True, metavar=("SOURCE", "AUDIT", "DATA")
    )
    parser.add_argument("--audit-hash", action="append", required=True)
    parser.add_argument("--data-hash", action="append", required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    clean(args.expected_head)
    if len(args.group) != 2 or len(args.audit_hash) != 2 or len(args.data_hash) != 2:
        raise ValueError("two predeclared source, replay and dataset groups required")
    if args.output.exists() or args.output.resolve().is_relative_to(Path.cwd()):
        raise ValueError("unused external output required")
    torch.manual_seed(92701)
    torch.set_num_threads(1)
    episodes, inputs, references, pinned = [], [], [], []
    seen_seeds, identity = set(), None
    for group, ah, dh in zip(args.group, args.audit_hash, args.data_hash, strict=True):
        root, audit, data_root = map(Path, group)
        data_path = data_root / "complete.json"
        if data_path.is_symlink() or digest(data_path) != dh:
            raise ValueError("exact original exported dataset required")
        data = json.loads(data_path.read_text())
        source, _, source_hash = report(
            root, audit, ah, "NEURAL_RELAY_PREPARED_RETURN_SAMPLING16_PENDING_OWN_REPLAY", worlds=16
        )
        if (
            data["source_hash"] != source_hash
            or data["replay_hash"] != ah
            or data["status"] != "RETURN_PLAYER_OWN_SAMPLED_DATA_RESEARCH_ONLY"
            or data["reward_profile"] != "post-sampling-contact-proxy-full-horizon-v1"
            or any(
                data.get(k) is not False
                for k in ("optimizer_authorized", "promotion_authorized", "hardware_authorized")
            )
        ):
            raise ValueError("exact unpromoted full-horizon data and own replay required")
        pins = source["input_file_hashes"]
        models = [x for x in pins if x.endswith("/velocity/v0/exported/policy.onnx")]
        configs = [x for x in pins if x.endswith("/velocity/v0/params/deploy.yaml")]
        if len(models) != 1 or len(configs) != 1:
            raise ValueError("one pinned motor and config required")
        current = (models[0], pins[models[0]], configs[0], pins[configs[0]])
        if identity is not None and identity != current:
            raise ValueError("same behavior anchor across episodes required")
        identity = current
        params = VelocityPolicyParameters.load(Path(current[2]), expected_hash=current[3])
        scale = np.asarray(params.action_scale, np.float32)
        records = {r["index"]: r for r in data["records"]}
        if len(records) != 8:
            raise ValueError("all sampled arms including no-trigger cases required")
        for row in source["records"]:
            seed = row["course"]["arm"].get("passer_sampling_seed")
            if seed is None:
                continue
            if seed not in (94111, 94137) or row["course"]["context"] not in (15, 16, 17, 18):
                raise ValueError("predeclared block2 training seeds and contexts required")
            seen_seeds.add(seed)
            record = records[row["index"]]
            vp, pp = bind(root, row, "passer_velocity"), bind(root, row, "physics")
            if (
                vp is None
                or pp is None
                or digest(vp) != record["source_velocity_hash"]
                or digest(pp) != record["source_physics_hash"]
            ):
                raise ValueError("exact original own physical trajectories required")
            with np.load(vp, allow_pickle=False) as v, np.load(pp, allow_pickle=False) as raw:
                audit_return_player_sampling(v, scale)
                ix = np.flatnonzero(v["sampling_active"])
                if len(ix) != record["sampled_controls"]:
                    raise ValueError("sample count mismatch")
                if not len(ix):
                    if record["path"] is not None:
                        raise ValueError("no optimizer rows for untriggered cases")
                    continue
                ep_path = data_root / record["path"]
                if (
                    ep_path.is_symlink()
                    or ep_path.resolve().parent != data_root.resolve()
                    or digest(ep_path) != record["hash"]
                ):
                    raise ValueError("exact confined exported episode required")
                outcome = return_player_outcome(
                    raw["ball_forces"], raw["body_failed"], sample_start=int(ix[0])
                )
                returns = discounted_sample_returns(outcome, sample_end=int(ix[-1]))
                if outcome != record["outcome"]:
                    raise ValueError("complete actual post-sampling outcome required")
                with np.load(ep_path, allow_pickle=False) as ep:
                    expected = {
                        "inputs": v["sampling_inputs"][ix],
                        "reference": v["sampling_reference"][ix],
                        "old_mean": v["sampling_mean"][ix],
                        "sampled_action": v["sampling_action"][ix],
                        "old_logp": v["sampling_logp"][ix],
                        "returns": returns,
                        "action_scale": scale,
                    }
                    if any(not np.array_equal(ep[k], value) for k, value in expected.items()):
                        raise ValueError("exported observations, actions or returns differ")
                    def t(key):
                        return torch.from_numpy(ep[key].copy())
                    e = dict(
                        observations=t("inputs"),
                        actions=t("sampled_action"),
                        old_means=t("old_mean"),
                        reference=t("reference"),
                        old_logp=t("old_logp"),
                        returns=t("returns"),
                        action_scale=t("action_scale"),
                        sampling_law="box-truncated-ar1",
                        all_actor_actions_actually_executed=True,
                        density_audited=True,
                        full_task_outcome_observed=True,
                        assisted=False,
                        external_teacher=False,
                        authority_rejected=False,
                    )
                    episodes.append(e)
                    inputs.append(e["observations"])
                    references.append(e["reference"])
                pinned.append((ep_path, record["hash"]))
        pinned.extend(((root / "complete.json", source_hash), (audit, ah), (data_path, dh)))
    if seen_seeds != {94111, 94137} or len(episodes) != 12 or identity is None:
        raise ValueError("complete predeclared sampled episode set required")
    model, mh, config, ch = identity
    prior = VelocityPriorActor.from_onnx(Path(model), expected_hash=mh, reference_features=14)
    actor = PrecontactReceivingMotor(
        prior, torch.tensor(params.action_scale, dtype=torch.float32)
    ).eval()
    critic = ReceiverCritic()
    optimizer = torch.optim.Adam(critic.parameters(), lr=0.001)
    losses = []
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = episode_balanced_value_loss(critic, episodes)
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in critic.parameters()):
            raise ValueError("finite critic gradients required")
        optimizer.step()
        losses.append(float(loss.detach()))
    critic.eval().requires_grad_(False)
    obs = torch.cat(inputs)
    with torch.no_grad():
        advantage = torch.cat([e["returns"] for e in episodes]) - critic(obs)
        advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-6)
    start = 0
    for ep in episodes:
        end = start + len(ep["observations"])
        ep["advantages"] = advantage[start:end]
        start = end
    update = bounded_actor_update(
        actor,
        critic,
        episodes,
        torch.full((29,), math.log(0.02)),
        obs,
        torch.cat(references),
        step_size=1e-4,
    )
    if any(digest(p) != h for p, h in (*pinned, (Path(model), mh), (Path(config), ch))):
        raise ValueError("immutable training evidence required")
    clean(args.expected_head)
    args.output.mkdir(parents=True)
    checkpoint = args.output / "research-candidate.pt"
    torch.save(
        dict(
            actor=actor.plastic.state_dict(),
            critic=critic.state_dict(),
            model_hash=mh,
            config_hash=ch,
            data_hashes=args.data_hash,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
        checkpoint,
    )
    write_once(
        args.output / "complete.json",
        dict(
            status="PREPARED_RETURN_ACTOR_RESEARCH_CANDIDATE",
            source_head=args.expected_head,
            model_identity=identity,
            data_hashes=args.data_hash,
            audit_hashes=args.audit_hash,
            sampled_episodes=len(episodes),
            sampled_controls=len(obs),
            critic_updates=3,
            critic_losses=losses,
            actor_result=update,
            checkpoint_hash=digest(checkpoint),
            physical_retention_proven=False,
            possession_proven=False,
            promotion_authorized=False,
            hardware_authorized=False,
            independent_holdout=False,
        ),
    )


if __name__ == "__main__":
    main()
