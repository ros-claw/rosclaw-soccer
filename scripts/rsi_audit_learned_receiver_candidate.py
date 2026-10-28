"""Numerically read back one pilot continuation; this is not physical validation."""

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
    for name in ("dataset", "dataset-audit", "training", "parent", "admission", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in (
        "dataset-hash",
        "dataset-audit-hash",
        "training-hash",
        "parent-hash",
        "admission-hash",
        "expected-head",
    ):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    clean(a.expected_head)
    pins = (
        (a.dataset / "complete.json", a.dataset_hash),
        (a.dataset_audit, a.dataset_audit_hash),
        (a.training / "complete.json", a.training_hash),
        (a.parent, a.parent_hash),
        (a.admission, a.admission_hash),
    )
    if (
        a.output.exists()
        or a.output.resolve().is_relative_to(Path.cwd())
        or any(path.is_symlink() or digest(path) != h for path, h in pins)
    ):
        raise ValueError("exact pinned research sources and unused external output required")
    data, audit, training, admission = [
        json.loads(path.read_text()) for path in (pins[0][0], pins[1][0], pins[2][0], pins[4][0])
    ]
    cp, anchors = a.training / "research-candidate.pt", a.admission.parent / "healthy-anchors.npz"
    if (
        data.get("status") != "LEARNED_RECEIVER_PILOT_PHYSICAL_DATA_NOT_TRAINED"
        or audit.get("status") != "LEARNED_RECEIVER_PILOT_DATA_READBACK_AUDITED"
        or audit.get("dataset_hash") != a.dataset_hash
        or training.get("status") != "LEARNED_RECEIVER_PILOT_CONTINUATION_CANDIDATE"
        or training.get("dataset_hash") != a.dataset_hash
        or training.get("dataset_audit_hash") != a.dataset_audit_hash
        or training.get("parent_hash") != a.parent_hash
        or data.get("receiver_checkpoint_hash") != a.parent_hash
        or training.get("actor_updates") != 1
        or training.get("critic_updates") != 3
        or training.get("episodes") != 24
        or data.get("episodes") != 24
        or admission.get("healthy_anchor_controls") != 5201
        or cp.is_symlink()
        or digest(cp) != training.get("checkpoint_hash")
        or anchors.is_symlink()
        or digest(anchors) != admission.get("healthy_anchor_hash")
        or any(
            x.get(k) is not False
            for x in (data, audit, training, admission)
            for k in ("promotion_authorized", "hardware_authorized")
        )
    ):
        raise ValueError("exact bounded SIM continuation candidate and audited histories required")
    state, parent = (
        torch.load(path, map_location="cpu", weights_only=True) for path in (cp, a.parent)
    )
    if (
        state.get("parent_hash") != a.parent_hash
        or state.get("dataset_hash") != a.dataset_hash
        or state.get("dataset_audit_hash") != a.dataset_audit_hash
        or any(
            state.get(k) is not False
            for k in ("physical_retention_proven", "promotion_authorized", "hardware_authorized")
        )
    ):
        raise ValueError("unpromoted exact actor/critic continuation required")
    torch.set_num_threads(2)
    model, mh, config, ch = data["model_identity"]
    params = VelocityPolicyParameters.load(Path(config), expected_hash=ch)
    actor = AnchoredReceivingMotor(
        VelocityPriorActor.from_onnx(Path(model), expected_hash=mh, reference_features=14),
        torch.tensor(params.action_scale, dtype=torch.float32),
    ).eval()
    actor.plastic.load_state_dict(state["actor"], strict=True)
    critic = ReceiverCritic().eval()
    critic.load_state_dict(state["critic"], strict=True)
    if any(not torch.isfinite(p).all() for p in (*actor.parameters(), *critic.parameters())):
        raise ValueError("finite actor and critic checkpoint required")
    before = parent["actor"]
    if set(before) != set(state["actor"]):
        raise ValueError("exact parent parameter structure required")
    changed = sum(not torch.equal(before[k], state["actor"][k]) for k in before)
    if changed == 0:
        raise ValueError("accepted update must actually change parent actor")
    parent_actor = AnchoredReceivingMotor(
        VelocityPriorActor.from_onnx(Path(model), expected_hash=mh, reference_features=14),
        torch.tensor(params.action_scale, dtype=torch.float32),
    ).eval()
    parent_actor.plastic.load_state_dict(before, strict=True)
    with np.load(anchors, allow_pickle=False) as f, torch.no_grad():
        x, ref = torch.from_numpy(f["inputs"].copy()), torch.from_numpy(f["reference"].copy())
        _, evidence = actor.forward_with_evidence(x, ref)
        delta = float(evidence["realized_delta_radians"].abs().max())
    kls, behavior_error, controls, frozen = [], 0.0, 0, []
    for row in data["records"]:
        path = a.dataset / row["path"]
        if (
            path.is_symlink()
            or path.resolve().parent != a.dataset.resolve()
            or digest(path) != row["hash"]
        ):
            raise ValueError("exact confined sampled episode required")
        frozen.append((path, row["hash"]))
        with np.load(path, allow_pickle=False) as ep, torch.no_grad():
            x, ref, old, action, scale = (
                torch.from_numpy(ep[k].copy())
                for k in ("inputs", "reference", "old_mean", "sampled_action", "action_scale")
            )
            behavior_error = max(behavior_error, float((parent_actor(x, ref) - old).abs().max()))
            kls.append(
                float(
                    history_motor_kl(
                        "box-truncated-ar1",
                        old,
                        actor(x, ref),
                        action,
                        ref,
                        scale,
                        torch.full((29,), math.log(0.02)),
                        rho=0.9,
                    )
                )
            )
            controls += len(x)
    kl, attempt = sum(kls) / len(kls), training["attempts"][-1]
    if (
        len(kls) != 24
        or controls != data["sampled_controls"]
        or behavior_error > 1e-5
        or not math.isfinite(delta)
        or not math.isfinite(kl)
        or delta > 0.005
        or not 0 <= kl <= 0.02
        or not math.isclose(delta, attempt["anchor_delta_rad"], rel_tol=1e-6, abs_tol=1e-12)
        or not math.isclose(kl, attempt["mean_conditional_kl"], rel_tol=1e-6, abs_tol=1e-12)
        or any(
            digest(path) != h
            for path, h in (
                *pins,
                *frozen,
                (cp, training["checkpoint_hash"]),
                (anchors, admission["healthy_anchor_hash"]),
                (Path(model), mh),
                (Path(config), ch),
            )
        )
    ):
        raise ValueError(
            "actual behavior, unchanged inputs and independently recomputed gates required"
        )
    clean(a.expected_head)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "complete.json",
        dict(
            status="LEARNED_RECEIVER_CONTINUATION_NUMERIC_READBACK_MATCH",
            source_head=a.expected_head,
            checkpoint_hash=training["checkpoint_hash"],
            training_hash=a.training_hash,
            dataset_hash=a.dataset_hash,
            dataset_audit_hash=a.dataset_audit_hash,
            parent_hash=a.parent_hash,
            episodes=24,
            sampled_controls=controls,
            changed_tensors=changed,
            max_behavior_error=behavior_error,
            max_anchor_delta_rad=delta,
            mean_conditional_kl=kl,
            independent_new_physics=False,
            physical_retention_proven=False,
            fresh_holdout=False,
            promotion_authorized=False,
            hardware_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
