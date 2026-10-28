"""Independently reload a dense-trained SIM receiver and recompute trust gates."""

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
    for name in ("dataset", "admission", "training", "dense-audit", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in (
        "dataset-hash",
        "admission-hash",
        "training-hash",
        "dense-audit-hash",
        "producer-head",
        "expected-head",
    ):
        p.add_argument("--" + name, required=True)
    p.add_argument("--seed", type=int, choices=(92401, 92402, 92403), required=True)
    a = p.parse_args()
    clean(a.expected_head)
    dp, ap, tp = (x / "complete.json" for x in (a.dataset, a.admission, a.training))
    pins = (
        (dp, a.dataset_hash),
        (ap, a.admission_hash),
        (tp, a.training_hash),
        (a.dense_audit, a.dense_audit_hash),
    )
    if (
        a.output.exists()
        or a.output.resolve().is_relative_to(Path.cwd())
        or any(p.is_symlink() or digest(p) != h for p, h in pins)
    ):
        raise ValueError("exact immutable research input and unused external output required")
    data, admission, training, audit = [json.loads(p.read_text()) for p, _ in pins]
    checkpoint, anchors = a.training / "research-candidate.pt", a.admission / "healthy-anchors.npz"
    if (
        training.get("status") != "DENSE_RECEIVER_OFFLINE_CANDIDATE"
        or training.get("source_head") != a.producer_head
        or training.get("dataset_hash") != a.dataset_hash
        or training.get("dense_audit_hash") != a.dense_audit_hash
        or training.get("seed") != a.seed
        or training.get("actor_updates") != 1
        or training.get("critic_updates") != 3
        or training.get("strict_positive_conditions") != 0
        or audit.get("status") != "RECEIVER_DENSE_FEEDBACK_INDEPENDENTLY_RECOMPUTED_NOT_TRAINED"
        or audit.get("original_hash") != a.dataset_hash
        or audit.get("dense_hash") != training.get("dense_hash")
        or data.get("episodes") != 72
        or admission.get("healthy_anchor_controls") != 5201
        or any(
            x.get(k) is not False
            for x in (training, admission, audit)
            for k in ("hardware_authorized", "promotion_authorized")
        )
        or checkpoint.is_symlink()
        or anchors.is_symlink()
        or digest(checkpoint) != training.get("checkpoint_hash")
        or digest(anchors) != admission.get("healthy_anchor_hash")
    ):
        raise ValueError("unpromoted dense candidate and independently audited provenance required")
    torch.set_num_threads(2)
    model, mh, config, ch = data["model_identity"]
    params = VelocityPolicyParameters.load(Path(config), expected_hash=ch)
    actor = (
        AnchoredReceivingMotor(
            VelocityPriorActor.from_onnx(Path(model), expected_hash=mh, reference_features=14),
            torch.tensor(params.action_scale, dtype=torch.float32),
        )
        .eval()
        .requires_grad_(False)
    )
    initial = {k: v.clone() for k, v in actor.plastic.state_dict().items()}
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (
        state.get("dataset_hash") != a.dataset_hash
        or state.get("dense_hash") != training["dense_hash"]
        or state.get("dense_audit_hash") != a.dense_audit_hash
        or state.get("seed") != a.seed
        or any(
            state.get(k) is not False
            for k in ("physical_retention_proven", "promotion_authorized", "hardware_authorized")
        )
    ):
        raise ValueError("exact candidate metadata required")
    actor.plastic.load_state_dict(state["actor"], strict=True)
    critic = ReceiverCritic()
    critic.load_state_dict(state["critic"], strict=True)
    if any(
        not torch.isfinite(v).all()
        for v in (*actor.plastic.state_dict().values(), *critic.state_dict().values())
    ):
        raise ValueError("finite learned weights required")
    changed = sorted(
        k for k, v in initial.items() if not torch.equal(v, actor.plastic.state_dict()[k])
    )
    if not changed:
        raise ValueError("candidate has no actual parameter update")
    with np.load(anchors, allow_pickle=False) as f, torch.no_grad():
        x, ref = torch.from_numpy(f["inputs"].copy()), torch.from_numpy(f["reference"].copy())
        if len(x) != 5201:
            raise ValueError("complete anchor history required")
        _, evidence = actor.forward_with_evidence(x, ref)
        delta = float(evidence["realized_delta_radians"].abs().max())
    kls, count, episode_pins = [], 0, []
    with torch.no_grad():
        for row in data["records"]:
            path = a.dataset / row["path"]
            if (
                path.is_symlink()
                or path.resolve().parent != a.dataset.resolve()
                or digest(path) != row["hash"]
            ):
                raise ValueError("exact sampled episode required")
            episode_pins.append((path, row["hash"]))
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
    kl, attempt = sum(kls) / len(kls), training["attempts"][-1]
    if (
        len(kls) != 72
        or count != 18903
        or not math.isfinite(delta)
        or not math.isfinite(kl)
        or delta > 0.005
        or not 0 <= kl <= 0.02
        or not math.isclose(delta, attempt["anchor_delta_rad"], rel_tol=1e-6, abs_tol=1e-12)
        or not math.isclose(kl, attempt["mean_conditional_kl"], rel_tol=1e-6, abs_tol=1e-12)
    ):
        raise ValueError("independent KL/anchor gate differs or failed")
    if any(
        digest(p) != h
        for p, h in (
            *pins,
            *episode_pins,
            (checkpoint, training["checkpoint_hash"]),
            (anchors, admission["healthy_anchor_hash"]),
            (Path(model), mh),
            (Path(config), ch),
        )
    ):
        raise ValueError("candidate input changed during readback")
    clean(a.expected_head)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "complete.json",
        dict(
            status="DENSE_RECEIVER_CANDIDATE_READBACK_MATCH",
            source_head=a.expected_head,
            producer_head=a.producer_head,
            training_hash=a.training_hash,
            checkpoint_hash=training["checkpoint_hash"],
            dataset_hash=a.dataset_hash,
            dense_audit_hash=a.dense_audit_hash,
            seed=a.seed,
            episodes=len(kls),
            sampled_controls=count,
            healthy_anchor_controls=5201,
            max_anchor_delta_rad=delta,
            mean_conditional_kl=kl,
            changed_tensors=changed,
            independent_new_physics=False,
            physical_retention_proven=False,
            promotion_authorized=False,
            hardware_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
