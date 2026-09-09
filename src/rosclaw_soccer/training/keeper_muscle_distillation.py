"""Distill successful physical keeper demonstrations; fit is NOT promotion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.providers.g1.keeper_muscle_actor import (
    CONTRACT_HASH,
    KeeperMuscleActor,
    muscle_observation,
)
from rosclaw_soccer.providers.g1.mujoco_primitives import mirror_g1_joint_positions
from rosclaw_soccer.sim.contracts import hash_bytes
from rosclaw_soccer.skills.team.independent_team_world import _gravity_orientation


def train(
    source: Path,
    output: Path,
    *,
    epochs: int = 1500,
    seed: int = 230,
    task_reference_only: bool = False,
) -> dict[str, Any]:
    import torch

    if not 1 <= epochs <= 10000:
        raise ValueError("invalid distillation epoch budget")
    if type(task_reference_only) is not bool:
        raise ValueError("reference ablation must be explicit boolean")
    paths = sorted(source.glob("*-trajectory.npz"))
    if len(paths) != 4:
        raise ValueError("distillation requires the four declared S88 teacher lanes")
    evidence_bytes = (source / "evidence.json").read_bytes()
    evidence = json.loads(evidence_bytes)
    if evidence.get("passed") is not True or evidence.get("activation_ceiling") != "SIM_ONLY":
        raise ValueError("teacher portfolio lacks successful SIM_ONLY physics evidence")
    output.mkdir(parents=True, exist_ok=False)
    xs, ys, groups = [], [], []
    hashes = {}
    for lane, path in enumerate(paths):
        hashes[path.name] = str(hash_bytes(path.read_bytes()))
        case = evidence.get("cases", {}).get(path.name.removesuffix("-trajectory.npz"), {})
        if (
            case.get("passed") is not True
            or case.get("trajectory_hash") != hashes[path.name]
            or not case.get("first", {}).get("gates")
            or not all(v is True for v in case["first"]["gates"].values())
        ):
            raise ValueError("teacher trajectory is not bound to successful physical gates")
        with np.load(path, allow_pickle=False) as a:
            contacts = np.flatnonzero(
                a["goalkeeper_left_glove_contact"] | a["goalkeeper_right_glove_contact"]
            )
            if not len(contacts):
                raise ValueError("teacher lane has no glove contact")
            # Future contact is permitted for selecting training labels only;
            # neither its time nor its position is an actor input.
            for i in range(int(contacts[0])):
                intercept = a["goalkeeper_estimated_intercept"][i]
                if not (
                    0 < intercept[0] <= 0.65
                    and abs(intercept[1]) <= 0.65
                    and 1.0 <= intercept[2] <= 1.9
                ):
                    continue
                pose = a["goalkeeper_pelvis_pose"][i]
                gravity = _gravity_orientation(pose[3:7])
                q, dq = a["goalkeeper_joint_position"][i], a["goalkeeper_joint_velocity"][i]
                target = a["goalkeeper_policy_action"][i]
                for mirrored in (False, True):
                    ci, cg = intercept.copy(), gravity.copy()
                    cq, cdq, cy = q, dq, target
                    if mirrored:
                        ci[1] *= -1
                        cg[1] *= -1
                        cq, cdq, cy = (mirror_g1_joint_positions(v) for v in (q, dq, target))
                    xs.append(muscle_observation(ci, float(pose[2]), cg, cq, cdq))
                    ys.append(np.clip(cy[15:] / 3, -0.999, 0.999))
                    groups.append(lane)
    x, y = np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)
    if len(x) < 100:
        raise ValueError("not enough finite demonstration frames")
    torch.manual_seed(seed)
    first_layer = torch.nn.Linear(65, 64)
    model = torch.nn.Sequential(
        first_layer,
        torch.nn.Tanh(),
        torch.nn.Linear(64, 64),
        torch.nn.Tanh(),
        torch.nn.Linear(64, 14),
        torch.nn.Tanh(),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    if task_reference_only:
        with torch.no_grad():
            first_layer.weight[:, 4:] = 0
    tx, ty = torch.from_numpy(x), torch.from_numpy(y)
    for _ in range(epochs):
        # Small proprioceptive augmentation; do not jitter the causal target.
        noise = torch.randn_like(tx) * 0.003
        noise[:, :4] = 0
        loss = torch.mean((model(tx + noise) - ty) ** 2)
        optimizer.zero_grad()
        loss.backward()  # type: ignore[no-untyped-call]
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if task_reference_only:
            # This actor is a task-conditioned reference prior, not a body
            # stabilizer. PD/Loco still use measured body state downstream.
            with torch.no_grad():
                first_layer.weight[:, 4:] = 0
    model.eval()
    with torch.inference_mode():
        fitted = model(tx).numpy() * 3
    layers = [
        {
            "weight": layer.weight.detach().numpy().tolist(),
            "bias": layer.bias.detach().numpy().tolist(),
        }
        for layer in model
        if isinstance(layer, torch.nn.Linear)
    ]
    artifact = dict(
        schema="keeper-muscle-bc.v1",
        activation_ceiling="SIM_ONLY",
        promotion_authorized=False,
        observation_contract_hash=CONTRACT_HASH,
        joint_names=list(G1_DDS_JOINT_NAMES[15:]),
        layers=layers,
        training_sources=hashes,
        teacher_evidence_hash=str(hash_bytes(evidence_bytes)),
        seed=seed,
        epochs=epochs,
        method="supervised_behavioral_cloning_with_sagittal_augmentation",
        task_reference_only=task_reference_only,
        parent="S88_CPU_MUJOCO_EXECUTED_TEACHER_TARGETS",
        heldout_physics_passed=False,
    )
    path = output / "keeper-muscle-actor.json"
    path.write_text(json.dumps(artifact, separators=(",", ":")) + "\n")
    loaded = KeeperMuscleActor(path)
    predictions = np.stack([loaded.target(v) for v in x])
    parity = float(np.max(np.abs(predictions - fitted)))
    if parity > 1e-5:
        raise RuntimeError("numeric actor differs from trained Torch model")
    report = dict(
        activation_ceiling="SIM_ONLY",
        candidate_promoted=False,
        frames=len(x),
        physical_frames=len(x) // 2,
        source_lanes=4,
        training_rmse_rad=float(np.sqrt(np.mean((predictions - y * 3) ** 2))),
        numeric_parity_max_rad=parity,
        policy_hash=loaded.policy_hash,
        # These are fit metrics, NOT a held-out success rate.
        heldout_physics_passed=False,
        epochs=epochs,
        task_reference_only=task_reference_only,
        source_hashes=hashes,
        teacher_evidence_hash=str(hash_bytes(evidence_bytes)),
        source_code_hash=str(hash_bytes(Path(__file__).read_bytes())),
    )
    np.savez_compressed(output / "training-data.npz", observation=x, target=y * 3, lane=groups)
    (output / "training-report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--task-reference-only", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            train(
                args.source,
                args.output,
                epochs=args.epochs,
                task_reference_only=args.task_reference_only,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
