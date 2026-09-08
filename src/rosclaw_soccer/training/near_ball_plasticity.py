"""Use Core's single-focal training lease for private Soccer PPO updates.

This is a learning-data/weight boundary, NOT a rosclawd motion lease or Permit.
It never authorizes robot execution, activates a candidate, or clears a freeze.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from rosclaw.continual.plasticity_lease import (
    AgentPolicyBinding,
    AgentUpdateMode,
    PlasticityLease,
    audit_plasticity_lease,
)

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def private_weight_hashes(
    weights: Mapping[str, np.ndarray],
    ids: tuple[str, ...],
    body_hash: str,
) -> dict[str, str]:
    return {
        agent: str(
            hash_json(
                {
                    "agent": agent,
                    "body": body_hash,
                    "weights": {k: v[i].tolist() for k, v in weights.items()},
                }
            )
        )
        for i, agent in enumerate(ids)
    }


def begin_update(
    *,
    before: dict[str, str],
    focal: str,
    generation: int,
    dataset_hash: str,
    context_hash: str,
    maximum_steps: int,
) -> PlasticityLease:
    return PlasticityLease(
        lease_id=f"soccer.residual.{focal}.g{generation}",
        bindings=tuple(
            AgentPolicyBinding(
                agent, digest, AgentUpdateMode.PLASTIC if agent == focal else AgentUpdateMode.FROZEN
            )
            for agent, digest in sorted(before.items())
        ),
        dataset_manifest_hash=dataset_hash,
        scenario_contract_hash=context_hash,
        maximum_optimizer_steps=maximum_steps,
    )


def finish_update(
    *,
    lease: PlasticityLease,
    before: dict[str, str],
    after: dict[str, str],
    steps: int,
) -> dict[str, Any]:
    if type(steps) is not int or steps < 0:
        raise ValueError("optimizer steps must be a nonnegative integer")
    # Bind the actual starting weights, not merely an equal roster of labels.
    if before != {binding.agent_id: binding.policy_hash for binding in lease.bindings}:
        raise ValueError("optimizer starting weights differ from its training lease")
    result = audit_plasticity_lease(
        lease=lease, optimizer_steps=steps, before_policy_hashes=before, after_policy_hashes=after
    )
    if not result.passed:
        raise ValueError(
            "private optimizer violated Core plasticity lease: " + ";".join(result.reasons)
        )
    return {"lease": lease.to_dict(), "audit": result.to_dict()}


def verify_update_record(
    record: dict[str, Any],
    *,
    before: dict[str, str],
    after: dict[str, str],
    focal: str,
    generation: int,
    dataset_hash: str,
    context_hash: str,
    maximum_steps: int = 4,
) -> None:
    lease = begin_update(
        before=before,
        focal=focal,
        generation=generation,
        dataset_hash=dataset_hash,
        context_hash=context_hash,
        maximum_steps=maximum_steps,
    )
    expected = finish_update(
        lease=lease, before=before, after=after, steps=record["audit"]["optimizer_steps"]
    )
    if record != expected:
        raise ValueError("recorded Core plasticity proof differs from actual private weights")


def replay_recorded_update(root: Path, output: Path) -> dict[str, Any]:
    from rosclaw_soccer.growth.near_ball_residual import NearBallResidualPolicy
    from rosclaw_soccer.training.active_team_probe import validate_probe
    from rosclaw_soccer.training.near_ball_residual_ppo import update_private_actors

    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((root / "training.json").read_text())
    digest = manifest.pop("manifest_hash")
    if hash_json(manifest) != digest:
        raise ValueError("recorded training manifest differs")
    traces, proofs = [], []
    for side in ("red", "blue"):
        source = root / f"train-000-{side}" / "probe.json"
        report = validate_probe(source)
        proofs.append(report["report_hash"])
        with np.load(source.parent / "primary.npz", allow_pickle=False) as archive:
            traces.append({k: archive[k] for k in archive.files})
    if proofs != manifest["iterations"][0]["rollout_report_hashes"]:
        raise ValueError("first-generation rollout binding differs")
    initial_generation = manifest.get("initial_generation", 0)
    if type(initial_generation) is not int or not 0 <= initial_generation <= 1000000:
        raise ValueError("initial generation is invalid")
    parent = NearBallResidualPolicy.load(root / f"generation-{initial_generation:03d}.npz")
    expected = NearBallResidualPolicy.load(root / f"generation-{initial_generation + 1:03d}.npz")
    credit = manifest.get("credit", {"gamma": 0.99, "trace_decay": 0.95})
    child, rows = update_private_actors(
        parent, traces, gamma=credit["gamma"], trace_decay=credit["trace_decay"]
    )
    if child.policy_hash != expected.policy_hash:
        raise ValueError("Core lease integration changed the recorded optimizer result")
    result = {
        "source_manifest_hash": digest,
        "dataset_reports": proofs,
        "exact_checkpoint_reproduced": True,
        "policy_hash": child.policy_hash,
        "players": rows,
        "activation_ceiling": "SIM_ONLY",
        "promotion_eligible": False,
        "implementation_hash": hash_bytes(Path(__file__).read_bytes()),
    }
    result["report_hash"] = hash_json(result)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(replay_recorded_update(args.root, args.output)))
