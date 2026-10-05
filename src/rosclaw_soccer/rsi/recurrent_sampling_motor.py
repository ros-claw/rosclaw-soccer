"""SIM-only exploration of the actual causal student, not its frozen parent.

Sampling uses the existing stationary AR(1) law and unchanged bounded motor
projection. Every recurrent state advances once at the execution boundary;
the logged likelihood describes latent exploration, not projected torques.
"""

import copy
from pathlib import Path
from typing import Any

import numpy as np
import rosclaw.growth.correlated_exploration as exploration

from rosclaw_soccer.rsi.recurrent_success_motor import (
    FLAGS,
    CompiledRecurrentSuccessMotor,
)
from rosclaw_soccer.rsi.recurrent_success_motor import (
    SCHEMA as IMITATION_SCHEMA,
)
from rosclaw_soccer.rsi.recurrent_success_motor import (
    make_preview as imitation_preview,
)
from rosclaw_soccer.rsi.recurrent_success_motor import (
    validate_model as validate_imitation,
)
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

SCHEMA = "soccer.rsi.recurrent_motor_sampling.v1"
TRACE_FIELDS = (
    "recurrent_sampling_raw_mean",
    "recurrent_sampling_latent_action",
    "recurrent_sampling_conditional_log_probability",
    "recurrent_sampling_performed",
)


def mean_preview(model: dict[str, Any]) -> dict[str, Any]:
    if model.get("schema") == IMITATION_SCHEMA:
        validate_imitation(model)
        return imitation_preview(model)
    from rosclaw_soccer.rsi.recurrent_clipped_motor import SCHEMA as CLIPPED_SCHEMA
    from rosclaw_soccer.rsi.recurrent_clipped_motor import make_preview as clipped_preview

    if model.get("schema") == CLIPPED_SCHEMA:
        return clipped_preview(model)
    raise ValueError("allowlisted actual causal mean model required")


def make_sampling_view(model: dict[str, Any], *, seed: int) -> dict[str, Any]:
    mean_preview(model)
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("bounded integer recurrent sampling seed required")
    value = dict(
        schema=SCHEMA,
        mean_model=copy.deepcopy(model),
        seed=seed,
        std_raw=0.1,
        rho=0.9,
        activation_ceiling="SIM_ONLY",
        training_only=True,
        source_hash=hash_bytes(Path(__file__).read_bytes()),
        exploration_source_hash=hash_bytes(Path(exploration.__file__).read_bytes()),
        **dict.fromkeys(FLAGS, False),
    )
    value["model_hash"] = hash_json(value)
    return value


def make_preview(view: dict[str, Any]) -> dict[str, Any]:
    if (
        type(view) is not dict
        or view.get("schema") != SCHEMA
        or type(view.get("mean_model")) is not dict
        or type(view.get("seed")) is not int
        or hash_json(view) != hash_json(make_sampling_view(view["mean_model"], seed=view["seed"]))
    ):
        raise ValueError("complete source-bound recurrent sampling view required")
    policy = mean_preview(view["mean_model"])
    policy["step_motor_proof"]["model"] = copy.deepcopy(view)
    policy["step_motor_proof"]["schema"] = "soccer.rsi.recurrent_sampling_preview.v1"
    policy["step_motor_proof"]["execution_source_hash"] = view["source_hash"]
    policy["step_motor_proof"]["qualification"] = "UNQUALIFIED_SIM_TRAINING"
    proof_key = (
        "recurrent_success_motor_proof"
        if view["mean_model"]["schema"] == IMITATION_SCHEMA
        else "recurrent_clipped_motor_proof"
    )
    policy["recurrent_sampling_motor_proof"] = policy.pop(proof_key)
    policy["recurrent_sampling_motor_proof"].update(
        actual_behavior_model_hash=view["mean_model"]["model_hash"],
        sampling_model_hash=view["model_hash"],
        training_only=True,
        latent_likelihood_is_not_projected_action_likelihood=True,
        actual_draws_and_causal_states_must_be_independently_replayed=True,
    )
    policy.pop("policy_hash")
    policy["policy_hash"] = hash_json(policy)
    return policy


class CompiledRecurrentSamplingMotor(CompiledRecurrentSuccessMotor):
    def __init__(self, policy: dict[str, Any]) -> None:
        view = copy.deepcopy(policy.get("step_motor_proof", {}).get("model", {}))
        if make_preview(view) != policy:
            raise ValueError("complete recurrent sampling execution commitment required")
        if view["mean_model"]["schema"] == IMITATION_SCHEMA:
            super().__init__(mean_preview(view["mean_model"]))
        else:
            from rosclaw_soccer.rsi.recurrent_clipped_motor import CompiledRecurrentClippedMotor

            # A fresh allowlisted reference owns every copied runtime object;
            # no user-provided factory, driver, or recursive past model graph.
            compiled = CompiledRecurrentClippedMotor(mean_preview(view["mean_model"]))
            self.__dict__.update(compiled.__dict__)
        self._sampling = {k: view[k] for k in ("seed", "std_raw", "rho")}
        self._noise = exploration.stationary_noise(
            seed=view["seed"], rho=view["rho"], count=270, dimension=12, first_frame=30
        )
        self._noise.flags.writeable = False
        self._policy_hash = policy["policy_hash"]
        self._last_mean = np.zeros(12)
        self._last_draw = np.zeros(12)
        self._last_log_probability = 0.0
        self._last_sampled = False

    @property
    def sampled_transition(self) -> dict[str, np.ndarray[Any, Any]]:
        return dict(
            recurrent_sampling_raw_mean=self._last_mean.copy(),
            recurrent_sampling_latent_action=self._last_draw.copy(),
            recurrent_sampling_conditional_log_probability=np.array(
                [self._last_log_probability], dtype=np.float64
            ),
            recurrent_sampling_performed=np.array([self._last_sampled], dtype=np.bool_),
        )

    def latent_sample(self, observation: Any, frame: int, phase: int) -> tuple[Any, float]:
        if (
            self._sampling is None
            or type(frame) is not int
            or not 30 <= frame < 300
            or frame != self._active_frame
        ):
            raise ValueError("actual sequential recurrent sampling boundary required")
        # Exactly one call: an extra mean query would advance the GRU twice.
        mean = self.raw_mean(observation, phase)
        index = frame - 30
        std, rho = self._sampling["std_raw"], self._sampling["rho"]
        noise = self._noise[index]
        offset = np.zeros(12) if index == 0 else rho * self._noise[index - 1]
        scale = exploration.conditional_scale(std, rho, first=index == 0)
        innovation = std * (noise - offset)
        logp = float(
            np.sum(-0.5 * (innovation / scale) ** 2 - np.log(scale) - 0.5 * np.log(2 * np.pi))
        )
        draw = np.asarray(mean + std * noise, dtype=np.float64)
        if not np.isfinite(draw).all() or not np.isfinite(logp):
            raise ValueError("nonfinite actual recurrent draw")
        self._last_mean, self._last_draw = mean.copy(), draw.copy()
        self._last_log_probability, self._last_sampled = logp, True
        return draw.copy(), logp
