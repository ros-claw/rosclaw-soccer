"""Opt-in per-episode numeric reuse, never a simulator or activation API."""

import re
from pathlib import Path
from typing import Any

from rosclaw_soccer.rsi import owned_smooth_preview as preview_module
from rosclaw_soccer.rsi import owned_smooth_sampling_factory as owned_factory_module
from rosclaw_soccer.rsi import smooth_sampling_decoder_factory as factory_module
from rosclaw_soccer.rsi.owned_smooth_sampling_factory import OwnedSmoothSamplingFactory
from rosclaw_soccer.rsi.smooth_memory_motor import (
    SAMPLING_SCHEMA,
    SCHEMA,
    CompiledSmoothMemoryMotor,
)
from rosclaw_soccer.rsi.smooth_sampling_decoder_factory import SmoothSamplingDecoderFactory
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def sampling_compilation_contract(
    policy: dict[str, Any], *, owned_preview: bool = False
) -> dict[str, Any]:
    proof = policy.get("step_motor_proof") if type(policy) is dict else None
    model = proof.get("model") if type(proof) is dict else None
    mean = model.get("mean_model") if type(model) is dict else None
    if (
        type(model) is not dict
        or type(mean) is not dict
        or model.get("schema") != SAMPLING_SCHEMA
        or mean.get("schema") != SCHEMA
        or any(
            type(v.get("model_hash")) is not str
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", v["model_hash"])
            for v in (model, mean)
        )
    ):
        raise ValueError("explicit sealed same-mean sampling family required")
    contract = dict(
        schema="soccer.rsi.offline_shared_smooth_compilation.v1",
        implementation="independent_episode_from_verified_same_mean",
        factory_source_hash=hash_bytes(Path(factory_module.__file__).read_bytes()),
        selection_source_hash=hash_bytes(Path(__file__).read_bytes()),
        mean_model_hash=mean["model_hash"],
        sampling_model_hash=model["model_hash"],
        full_original_preview_validation=True,
        independent_contact_history=True,
        independent_noise_state=True,
        actor_weights_changed=False,
        physical_dynamics_changed=False,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    if owned_preview:
        contract.update(
            schema="soccer.rsi.offline_owned_smooth_compilation.v1",
            implementation="independent_episode_from_owned_canonical_mean",
            factory_source_hash=hash_bytes(Path(owned_factory_module.__file__).read_bytes()),
            original_numeric_factory_source_hash=hash_bytes(
                Path(factory_module.__file__).read_bytes()
            ),
            preview_source_hash=hash_bytes(Path(preview_module.__file__).read_bytes()),
            full_original_preview_validation=False,
            original_mean_semantics_validated_once=True,
            canonical_mean_bytes_checked_each_bind=True,
            complete_original_preview_integrity_checked_each_bind=True,
            dependency_source_pinned=True,
        )
    return contract


def validate_sampling_compilation_contract(value: Any, policy: dict[str, Any]) -> None:
    if type(value) is not dict or hash_json(value) != hash_json(
        sampling_compilation_contract(
            policy,
            owned_preview=value.get("schema") == "soccer.rsi.offline_owned_smooth_compilation.v1",
        )
    ):
        raise ValueError("complete shared smooth compilation provenance required")


def select_smooth_decoder(
    policy: dict[str, Any], *, sampling_factory: Any = None
) -> tuple[CompiledSmoothMemoryMotor, dict[str, Any] | None]:
    """Keep the original default and two explicit fully source-bound compilers.

    The original factory repeats complete original validation each bind. The
    owned compiler validates mean semantics once, then exact canonical mean,
    original preview integrity and source on every bind. Its different contract
    says so explicitly. Neither may replace weights or share episode histories.
    """
    if sampling_factory is None:
        return CompiledSmoothMemoryMotor(policy), None
    if type(sampling_factory) not in (SmoothSamplingDecoderFactory, OwnedSmoothSamplingFactory):
        raise ValueError("only the verified same-mean sampling factory is accepted")
    decoder = sampling_factory.bind(policy)
    return decoder, sampling_compilation_contract(
        policy, owned_preview=type(sampling_factory) is OwnedSmoothSamplingFactory
    )
