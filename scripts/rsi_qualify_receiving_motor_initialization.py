"""Qualify an UNTRAINED whole receiving MLP on authenticated actual states.

No physics or optimizer step. Saved ONNX actions are OFFLINE numerical gold,
never executed or used as inputs to a simulator. Contact cues are past-only
foot/other-body categories, not future success or excluded-contact labels.
"""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in (
        "source-root",
        "own-root",
        "review-root",
        "scene",
        "velocity-model",
        "velocity-config",
        "output",
    ):
        p.add_argument("--" + k, type=Path, required=True)
    for k in ("review-hash", "expected-head"):
        p.add_argument("--" + k, required=True)
    a = p.parse_args()
    import mujoco
    import numpy as np
    import torch

    from rosclaw_soccer.providers.g1.velocity_policy import VelocityPolicyParameters
    from rosclaw_soccer.rsi.anchored_receiving_motor import AnchoredReceivingMotor
    from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
    from rosclaw_soccer.rsi.receiver_body_context import receiver_context14
    from rosclaw_soccer.rsi.refinement_onpolicy_checkpoint import file_hash
    from rosclaw_soccer.rsi.velocity_prior_actor import VelocityPriorActor
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    if a.output.exists():
        raise ValueError("unused once-only initialization output required")
    for root in (a.source_root, a.own_root, a.review_root):
        if root.is_symlink() or (root / "failed.json").exists():
            raise ValueError("ordinary closed body evidence required")
    rp = a.review_root / "complete.json"
    if file_hash(rp) != a.review_hash:
        raise ValueError("exact closed all96 review required")
    review = json.loads(rp.read_text())
    source, own = [
        json.loads((root / "complete.json").read_text()) for root in (a.source_root, a.own_root)
    ]
    if (
        review["status"] != "RECEIVER_ROLE_PAIRED96_REVIEWED_NOT_LEARNING_OR_PROMOTION"
        or review["source_hash"] != file_hash(a.source_root / "complete.json")
        or review["own_hash"] != file_hash(a.own_root / "complete.json")
        or own["source_hash"] != review["source_hash"]
        or own["independent_whole_world_replay"] is not True
        or own["sampling_law_audit_passed"] is not True
        or own["records"]
        != [{k: v for k, v in r.items() if k != "artifacts"} for r in source["records"]]
    ):
        raise ValueError("all recorded receiving histories must be independently authenticated")
    pins = dict(review["input_file_hashes"])
    pins[str(rp)] = a.review_hash
    for path in (a.scene, a.velocity_model, a.velocity_config):
        if source["input_file_hashes"].get(str(path)) != file_hash(path):
            raise ValueError("original pinned body/layout/codec required")
    source_pins(pins)
    check_pins(pins)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    params = VelocityPolicyParameters.load(
        a.velocity_config, expected_hash=pins[str(a.velocity_config)]
    )
    initial = VelocityPriorActor.from_onnx(
        a.velocity_model, expected_hash=pins[str(a.velocity_model)], reference_features=14
    )
    actor = AnchoredReceivingMotor(initial, torch.tensor(params.action_scale, dtype=torch.float32))
    model = mujoco.MjModel.from_xml_path(str(a.scene))
    layout = NativeVelocityLayout(
        model, robot_prefix="b/", ball_body_name="soccer_ball", player_count=2
    )
    records = source["records"]
    if len(records) != 96:
        raise ValueError("all96 actual source histories required")
    # Static protocol heading reconstructed from DECLARED symmetric offsets,
    # not labels, trajectories after contact, a trial id, or future ball data.
    forward = (
        np.array(records[64]["course"]["ball_planar_offset_world"])
        - np.array(records[32]["course"]["ball_planar_offset_world"])
    ) / 0.01
    if abs(np.linalg.norm(forward) - 1) > 1e-10:
        raise ValueError("unchanged declared +/-0.005m calibration required")
    lateral = np.array([-forward[1], forward[0]])
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            review_hash=a.review_hash,
            protocol_heading=forward.tolist(),
            physical_steps=0,
            optimizer_updates=0,
            saved_actions_offline_gold_only=True,
            initialization_not_a_trained_candidate=True,
        ),
    )
    summaries = []
    last = None
    try:
        for row in records:
            data = {}
            for key in ("initial", "physical", "B"):
                art = row["artifacts"][key]
                path = a.source_root / art["path"]
                if pins.get(str(path)) != art["hash"]:
                    raise ValueError("all consumed primitives must be authenticated in review")
                with np.load(path, allow_pickle=False) as z:
                    data[key] = {k: z[k].copy() for k in z.files}
            n = row["controls"]
            raw = data["physical"]
            b = data["B"]
            ini = data["initial"]
            q = np.concatenate((ini["qpos"][None], raw["qpos"][:-1, -1]), axis=0)
            v = np.concatenate((ini["qvel"][None], raw["qvel"][:-1, -1]), axis=0)
            if (
                q.shape != (n, model.nq)
                or v.shape != (n, model.nv)
                or b["obs"].shape != (n, 98)
                or b["action"].shape != (n, 29)
            ):
                raise ValueError(
                    "all actual PRE-action states and original B98 observations required"
                )
            feet = np.r_[
                False,
                np.maximum.accumulate(np.any(raw["ball_forces"][:-1, :, 1, :2] > 1, axis=(1, 2))),
            ]
            other = np.r_[
                False, np.maximum.accumulate(np.any(raw["ball_forces"][:-1, :, 1, 2] > 1, axis=1))
            ]
            goal = np.array(row["course"]["target"], dtype=np.float64)
            anchor = goal.copy()
            anchor[:2] += 0.25 * forward + 0.12 * lateral
            cues = []
            for t in range(n):
                qc = np.concatenate(
                    (
                        q[t, layout.rq : layout.rq + 7],
                        q[t, layout.qi],
                        q[t, layout.bq : layout.bq + 7],
                    )
                )
                vc = np.concatenate(
                    (
                        v[t, layout.rv : layout.rv + 6],
                        v[t, layout.vi],
                        v[t, layout.bv : layout.bv + 6],
                    )
                )
                cues.append(
                    receiver_context14(
                        qc,
                        vc,
                        anchor,
                        foot_contact_seen=bool(feet[t]),
                        other_body_contact_seen=bool(other[t]),
                    )
                )
            obs = torch.from_numpy(np.concatenate((b["obs"], np.stack(cues)), axis=1))
            gold = torch.from_numpy(b["action"])
            with torch.no_grad():
                out, ev = actor.forward_with_evidence(obs, gold)
                if not torch.equal(out, gold) or ev["realized_delta_radians"].count_nonzero():
                    raise ValueError("untrained anchor must have exact zero identity")
                # Torch conversion error is separate from exact anchored identity.
                normalized = torch.cat((obs[:, :98], obs[:, 98:] / actor.cue_scale), dim=1)
                err = float((actor.baseline(normalized) - gold).abs().max())
                if err > 1e-5:
                    raise ValueError("converted original MLP differs on actual receiving inputs")
            summaries.append(
                dict(
                    index=row["index"],
                    rows=n,
                    exact_anchored_identity=True,
                    torch_conversion_max_error=err,
                )
            )
            last = (obs, gold)
        assert last is not None
        actor.train()
        out = actor(*last)
        out.square().mean().backward()
        gradients = [
            float(p.grad.norm()) if p.grad is not None else None for p in actor.plastic_parameters()
        ]
        if len(gradients) != 8 or any(g is None or g <= 0 for g in gradients):
            raise ValueError("all eight whole MLP tensors must receive gradients")
        if any(p.grad is not None for p in actor.baseline.parameters()):
            raise ValueError("frozen clone received gradients")
        actor.zero_grad(set_to_none=True)
        actor.eval()
        cp = a.output / "untrained-initialization.pt"
        torch.save(
            dict(
                schema="soccer.receiver_whole_mlp_initialization.v1",
                actor=dict(actor.state_dict()),
                velocity_model_hash=pins[str(a.velocity_model)],
                velocity_config_hash=pins[str(a.velocity_config)],
                context_features=14,
                optimizer_updates=0,
                trained=False,
            ),
            cp,
        )
        restored = torch.load(cp, map_location="cpu", weights_only=True)
        replica = AnchoredReceivingMotor(
            initial, torch.tensor(params.action_scale, dtype=torch.float32)
        )
        replica.load_state_dict(restored["actor"], strict=True)
        with torch.no_grad():
            if not torch.equal(replica(*last), last[1]):
                raise ValueError("safe initialization reload changed zero identity")
        check_pins(pins)
        clean(a.expected_head)
        write_once(
            a.output / "complete.json",
            dict(
                status="WHOLE_RECEIVING_MLP_RECORDED_INITIALIZATION_QUALIFIED_NOT_TRAINED",
                source_head=a.expected_head,
                input_file_hashes=pins,
                review_hash=a.review_hash,
                rows=summaries,
                actual_recorded_worlds=96,
                actual_recorded_observations=sum(r["rows"] for r in summaries),
                checkpoint_hash=file_hash(cp),
                gradient_norms_synthetic_probe=gradients,
                new_physics_steps=0,
                optimizer_updates=0,
                physical_retention_proven=False,
                neural_learning_proven=False,
                promotion_authorized=False,
                hardware_authorized=False,
                goal_complete=False,
            ),
        )
    except BaseException as error:
        write_once(
            a.output / "failed.json",
            dict(
                error=repr(error),
                completed_recorded_worlds=len(summaries),
                new_physics_steps=0,
                optimizer_updates=0,
                goal_complete=False,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
