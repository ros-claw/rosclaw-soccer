"""Independently read the offline receiver candidate and recompute its trust gates."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from rosclaw_soccer.providers.g1.velocity_policy import VelocityPolicyParameters
from rosclaw_soccer.rsi.anchored_receiving_motor import AnchoredReceivingMotor
from rosclaw_soccer.rsi.motor_sampling_law import history_motor_kl
from rosclaw_soccer.rsi.receiver_actor_critic import ReceiverCritic
from rosclaw_soccer.rsi.velocity_prior_actor import VelocityPriorActor
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "admission", "training", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in (
        "dataset-hash",
        "admission-hash",
        "training-hash",
        "producer-head",
        "expected-head",
    ):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    clean(a.expected_head)
    dp, ap, tp = (x / "complete.json" for x in (a.dataset, a.admission, a.training))
    pins = ((dp, a.dataset_hash), (ap, a.admission_hash), (tp, a.training_hash))
    if (
        a.output.exists()
        or a.output.resolve().is_relative_to(Path.cwd())
        or any(x.is_symlink() or digest(x) != h for x, h in pins)
    ):
        raise ValueError("pinned unchanged data, candidate and unused output required")
    data, admission, training = (json.loads(x.read_text()) for x, _ in pins)
    checkpoint, anchors = a.training / "research-candidate.pt", a.admission / "healthy-anchors.npz"
    if (
        training.get("status") != "RECEIVER_OFFLINE_RESEARCH_CANDIDATE"
        or training.get("source_head") != a.producer_head
        or training.get("dataset_hash") != a.dataset_hash
        or training.get("admission_hash") != a.admission_hash
        or training.get("actor_updates") != 1
        or training.get("critic_updates") != 3
        or training.get("actual_new_physics") is not False
        or admission.get("dataset_hash") != a.dataset_hash
        or any(
            x.get("hardware_authorized") is not False or x.get("promotion_authorized") is not False
            for x in (training, admission)
        )
        or any(x.is_symlink() for x in (checkpoint, anchors))
        or digest(checkpoint) != training.get("checkpoint_hash")
        or digest(anchors) != admission.get("healthy_anchor_hash")
    ):
        raise ValueError("unpromoted offline candidate and exact evidence required")
    torch.set_num_threads(2)
    model, model_hash, config, config_hash = data["model_identity"]
    params = VelocityPolicyParameters.load(Path(config), expected_hash=config_hash)
    actor = (
        AnchoredReceivingMotor(
            VelocityPriorActor.from_onnx(
                Path(model), expected_hash=model_hash, reference_features=14
            ),
            torch.tensor(params.action_scale, dtype=torch.float32),
        )
        .eval()
        .requires_grad_(False)
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (
        state.get("dataset_hash") != a.dataset_hash
        or state.get("admission_hash") != a.admission_hash
        or state.get("seed") != 92301
        or state.get("physical_retention_proven") is not False
        or state.get("promotion_authorized") is not False
        or state.get("hardware_authorized") is not False
    ):
        raise ValueError("exact nonpromoted research checkpoint metadata required")
    initial = {k: v.clone() for k, v in actor.plastic.state_dict().items()}
    actor.plastic.load_state_dict(state["actor"], strict=True)
    critic = ReceiverCritic()
    critic.load_state_dict(state["critic"], strict=True)
    if any(
        not torch.isfinite(v).all()
        for v in (*actor.plastic.state_dict().values(), *critic.state_dict().values())
    ):
        raise ValueError("finite actor and critic tensors required")
    changed = sorted(
        k for k, value in initial.items() if not torch.equal(value, actor.plastic.state_dict()[k])
    )
    if not changed:
        raise ValueError("candidate must contain an actual plastic model change")
    with np.load(anchors, allow_pickle=False) as f, torch.no_grad():
        x, ref = torch.from_numpy(f["inputs"].copy()), torch.from_numpy(f["reference"].copy())
        if len(x) != admission["healthy_anchor_controls"]:
            raise ValueError("complete healthy anchor history required")
        anchor_count = len(x)
        _, evidence = actor.forward_with_evidence(x, ref)
        max_delta = float(evidence["realized_delta_radians"].abs().max())
    kls, count = [], 0
    with torch.no_grad():
        for row in data["records"]:
            path = a.dataset / row["path"]
            if (
                path.is_symlink()
                or path.resolve().parent != a.dataset.resolve()
                or digest(path) != row["hash"]
            ):
                raise ValueError("exact complete sampled history required")
            with np.load(path, allow_pickle=False) as e:
                x, ref, old, actions, scale = (
                    torch.from_numpy(e[k].copy())
                    for k in ("inputs", "reference", "old_mean", "sampled_action", "action_scale")
                )
                kls.append(
                    float(
                        history_motor_kl(
                            "box-truncated-ar1",
                            old,
                            actor(x, ref),
                            actions,
                            ref,
                            scale,
                            torch.full((29,), math.log(0.02)),
                            rho=0.9,
                        )
                    )
                )
                count += len(x)
    mean_kl = sum(kls) / len(kls)
    attempt = training["attempts"][-1]
    if (
        len(kls) != 72
        or count != 18482
        or not math.isfinite(max_delta)
        or not math.isfinite(mean_kl)
        or max_delta > 0.005
        or not 0 <= mean_kl <= 0.02
        or not math.isclose(max_delta, attempt["anchor_delta_rad"], rel_tol=1e-6, abs_tol=1e-12)
        or not math.isclose(mean_kl, attempt["mean_conditional_kl"], rel_tol=1e-6, abs_tol=1e-12)
    ):
        raise ValueError("independent numerical trust gate differs or failed")
    if any(
        digest(x) != h
        for x, h in (
            *pins,
            (checkpoint, training["checkpoint_hash"]),
            (anchors, admission["healthy_anchor_hash"]),
            (Path(model), model_hash),
            (Path(config), config_hash),
        )
    ):
        raise ValueError("candidate evidence changed during readback")
    clean(a.expected_head)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "complete.json",
        dict(
            status="RECEIVER_OFFLINE_CANDIDATE_READBACK_MATCH",
            source_head=a.expected_head,
            producer_head=a.producer_head,
            training_hash=a.training_hash,
            checkpoint_hash=training["checkpoint_hash"],
            episodes=len(kls),
            sampled_controls=count,
            healthy_anchor_controls=anchor_count,
            max_anchor_delta_rad=max_delta,
            mean_conditional_kl=mean_kl,
            changed_tensors=changed,
            independent_new_physics=False,
            physical_retention_proven=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
