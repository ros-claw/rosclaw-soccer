"""Independently recompute every neural proposal in a gated-contact exam.

Requires the separate native primitive audit first. This reader performs no
physics but DOES execute and count CPU neural inference; it is not a zero-work
audit or a replay of more independent physical episodes.
"""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--hash", required=True)
    p.add_argument("--primitive-root", type=Path, required=True)
    p.add_argument("--primitive-hash", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    rp, dp, ap = (
        a.root / "complete.json",
        a.root / "declaration.json",
        a.primitive_root / "complete.json",
    )
    if (
        a.output.exists()
        or digest(rp) != a.hash
        or digest(ap) != a.primitive_hash
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
    ):
        raise ValueError(
            "exact closed exam/primitiveaudit and unused frozen-source output required"
        )
    r, d, primitive = [json.loads(f.read_text()) for f in (rp, dp, ap)]
    if (
        r["status"] != "PAID_LIVE_DISTILLED_CPU_FULL_WORLD_EXAM_CLOSED"
        or primitive["status"] != "PAID_LIVE_DISTILLED_ALL_PRIMITIVE_AUDIT_PASS"
        or primitive["input_file_hashes"].get(str(rp)) != a.hash
        or r.get("contact_residual_trained") is not True
        or d.get("contact_residual_trained") is not True
        or r.get("contact_residual_zero") is not False
        or d["entry_controller"] != "qualified-prior"
        or not r["all_whole_neural_native_repeats_exact"]
    ):
        raise ValueError("primitive-audited learned-residual full-world exam required")
    pins = dict(primitive["input_file_hashes"])
    pins.update({str(rp): a.hash, str(dp): digest(dp), str(ap): a.primitive_hash})
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            own_native_steps=0,
            CPU_neural_recompute=True,
            hardware_authorized=False,
        ),
    )
    counts = dict(
        neural_rows=0,
        raw_episodes=0,
        residual_rows=0,
        base_only_rows=0,
        additional_gate_head_rows=0,
    )
    gate_episodes = []
    try:
        import numpy as np
        import torch

        from rosclaw_soccer.rsi.paid_contact_residual_actor import load_contact_residual_actor

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        policy = next(Path(f) for f in pins if f.endswith("/ckp/policy_30000.onnx"))
        scenes = [h for f, h in pins.items() if f.endswith("/cached-standard/declared-scene.xml")]
        if len(set(scenes)) != 1:
            raise ValueError("exact compiled standardbody required")
        seen = set()

        def raw(row, actor, *, residual=False):
            path = a.root / row["raw_path"]
            if (
                path.is_symlink()
                or not path.resolve().is_relative_to(a.root.resolve())
                or digest(path) != row["raw_hash"]
                or str(path) in seen
            ):
                raise ValueError("unique confined sealed episode required")
            seen.add(str(path))
            pins[str(path)] = row["raw_hash"]
            state = (torch.zeros(2, 1, 128), torch.zeros(2, 1, 128))
            gate_probabilities = []
            with np.load(path, allow_pickle=False) as archive:
                z = {
                    k: archive[k].copy()
                    for k in (
                        "NN_obs",
                        "NN_action",
                        "NN_nominal_action",
                        "NN_h_before",
                        "NN_c_before",
                        "NN_h_after",
                        "NN_c_after",
                    )
                }
            with torch.no_grad():
                for j, observation in enumerate(z["NN_obs"]):
                    if any(
                        not np.array_equal(x.numpy(), z[k][j])
                        for x, k in zip(state, ("NN_h_before", "NN_c_before"), strict=True)
                    ):
                        raise ValueError("causal own initial/previous neural memory differs")
                    mean, state = actor(torch.from_numpy(observation).reshape(1, 1, 160), state)
                    action = mean[0, 0].numpy()
                    if (
                        not np.array_equal(action, z["NN_action"][j])
                        or not np.array_equal(action, z["NN_nominal_action"][j])
                        or any(
                            not np.array_equal(x.numpy(), z[k][j])
                            for x, k in zip(state, ("NN_h_after", "NN_c_after"), strict=True)
                        )
                    ):
                        raise ValueError(
                            "declared own neural actor did NOT produce recorded actions/memory"
                        )
                    counts["neural_rows"] += 1
                    counts["residual_rows" if residual else "base_only_rows"] += 1
                    if residual:
                        # Diagnostic head-only pass from the VERIFIED own h/c,
                        # not another full actor rollout or new physical data.
                        feature = torch.cat(
                            [
                                (torch.from_numpy(observation).reshape(1, 160) - actor.base.mean)
                                / actor.base.denominator,
                                state[0][-1],
                            ],
                            dim=1,
                        )
                        probability = float(torch.sigmoid(actor.residual.gate(feature))[0, 0])
                        gate_probabilities.append(probability)
                        counts["additional_gate_head_rows"] += 1
            counts["raw_episodes"] += 1
            if residual:
                gate_episodes.append(
                    dict(
                        raw_path=row["raw_path"],
                        active_controls=[i for i, p in enumerate(gate_probabilities) if p >= 0.5],
                        minimum_probability=min(gate_probabilities),
                        maximum_probability=max(gate_probabilities),
                        controls=len(gate_probabilities),
                    )
                )

        for candidate in r["candidates"]:
            index = candidate["candidate"]
            root = Path(d["candidates"][index])
            cp = root / "contact-residual-epoch256.pt"
            tr = root / "complete.json"
            if (
                pins.get(str(cp)) != candidate["checkpoint_hash"]
                or pins.get(str(tr)) != candidate["training_receipt_hash"]
            ):
                raise ValueError("actual recorded contact policy commitments required")
            actor, _ = load_contact_residual_actor(
                policy,
                checkpoint=cp,
                checkpoint_hash=candidate["checkpoint_hash"],
                training_receipt=tr,
                training_receipt_hash=candidate["training_receipt_hash"],
                scene_hash=scenes[0],
            )
            if not seen:
                for rows in r["baseline_isolated"].values():
                    for row in rows:
                        raw(row, actor.base)
                for world in r["baseline_continuous"]:
                    raw(world["first"], actor.base)
                    if world["second"] is not None:
                        raw(world["second"], actor.base)
            for section in candidate["isolated"].values():
                for row in section["results"]:
                    raw(row, actor.base)
                    raw(
                        dict(
                            row,
                            raw_path=row["raw_path"].replace("-repeat0.npz", "-repeat1.npz"),
                            raw_hash=row["independent_repeat_raw_hash"],
                        ),
                        actor.base,
                    )
            for world in candidate["continuous"]["results"]:
                for artifacts in (world["artifacts"], world["repeat_artifacts"]):
                    data = {Path(x["raw_path"]).name: x for x in artifacts}
                    raw(data["first.npz"], actor.base)
                    if "second.npz" in data:
                        raw(data["second.npz"], actor, residual=True)
        if (
            counts["neural_rows"] != r["actual_neural_rows"]
            or counts["raw_episodes"] != primitive["episode_raws"]
        ):
            raise ValueError("all actual producer neural rows/episodes must be recomputed")
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="PAID_GATED_CONTACT_ALL_NEURAL_PROPOSALS_EXACT",
                input_file_hashes=pins,
                counts=counts,
                learned_gate_episodes=gate_episodes,
                gate_head_passes_are_additional_diagnostic_compute=True,
                own_native_steps=0,
                own_neural_rows=counts["neural_rows"],
                producer_native_steps_not_recounted=r["actual_native_steps"],
                not_new_independent_physics=True,
                goal_complete=False,
                hardware_authorized=False,
                promotion_authorized=False,
                commercial_publicity_authorized=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failed.json", dict(error=repr(error), counts=counts, own_native_steps=0)
        )
        raise


if __name__ == "__main__":
    main()
