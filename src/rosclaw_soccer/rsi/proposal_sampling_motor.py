"""Source-bound SIM-only AR(1) exploration of an actual proposal mean.

This creates no simulator or runtime authority. The original complete proposal
constructor remains the independent reference; no old mean is relabelled.
"""

import copy
from pathlib import Path
from types import MappingProxyType
from typing import Any

import rosclaw.growth.correlated_exploration as exploration

from rosclaw_soccer.rsi.kernel_guarded_step_network import FLAGS
from rosclaw_soccer.rsi.proposal_memory_motor import CompiledProposalMemoryMotor, validate_model
from rosclaw_soccer.rsi.proposal_memory_motor import make_preview as mean_preview
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

SCHEMA = "soccer.rsi.proposal_memory_sampling.v1"
EXPLORATION_PROFILES = MappingProxyType({"nominal": 0.1, "wider_bounded": 0.15})


def make_sampling_view(
    model: dict[str, Any], *, seed: int, exploration_profile: str = "nominal"
) -> dict[str, Any]:
    validate_model(model)
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("bounded integer sampling seed required")
    if type(exploration_profile) is not str or exploration_profile not in EXPLORATION_PROFILES:
        raise ValueError("explicit bounded exploration profile required")
    result = dict(
        schema=SCHEMA,
        mean_model=copy.deepcopy(model),
        seed=seed,
        std_raw=EXPLORATION_PROFILES[exploration_profile],
        exploration_profile=exploration_profile,
        rho=0.9,
        activation_ceiling="SIM_ONLY",
        training_only=True,
        source_hash=hash_bytes(Path(__file__).read_bytes()),
        exploration_source_hash=hash_bytes(Path(exploration.__file__).read_bytes()),
        **dict.fromkeys(FLAGS, False),
    )
    result["model_hash"] = hash_json(result)
    return result


def make_preview(view: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(view, dict) or view.get("schema") != SCHEMA:
        raise ValueError("complete sealed SIM-only proposal sampling view required")
    model = view.get("mean_model")
    if not isinstance(model, dict):
        raise ValueError("complete proposal mean required")
    seed = view.get("seed")
    if type(seed) is not int:
        raise ValueError("bounded integer sampling seed required")
    profile = view.get("exploration_profile")
    if not isinstance(profile, str):
        raise ValueError("explicit bounded exploration profile required")
    if hash_json(make_sampling_view(model, seed=seed, exploration_profile=profile)) != hash_json(
        view
    ):
        raise ValueError("complete sealed SIM-only proposal sampling view required")
    policy: dict[str, Any] = mean_preview(model)
    policy["step_motor_proof"]["model"] = copy.deepcopy(view)
    policy["step_motor_proof"]["execution_source_hash"] = view["source_hash"]
    policy["step_motor_proof"]["schema"] = "soccer.rsi.proposal_sampling_preview.v1"
    policy["proposal_sampling_motor_proof"] = policy.pop("proposal_memory_motor_proof")
    policy["proposal_sampling_motor_proof"].update(
        actual_behavior_model_hash=model["model_hash"],
        sampling_model_hash=view["model_hash"],
        training_only=True,
    )
    policy.pop("policy_hash")
    policy["policy_hash"] = hash_json(policy)
    return policy


class CompiledProposalSamplingMotor(CompiledProposalMemoryMotor):
    def __init__(self, policy: dict[str, Any]) -> None:
        view = copy.deepcopy(policy.get("step_motor_proof", {}).get("model", {}))
        if hash_json(make_preview(view)) != hash_json(policy):
            raise ValueError("complete proposal sampling execution commitment changed")
        super().__init__(mean_preview(view["mean_model"]))
        self._sampling = {k: view[k] for k in ("seed", "std_raw", "rho")}
        self._noise = exploration.stationary_noise(
            seed=view["seed"], rho=view["rho"], count=270, dimension=12, first_frame=30
        )
        self._noise.flags.writeable = False
        self._policy_hash = policy["policy_hash"]
