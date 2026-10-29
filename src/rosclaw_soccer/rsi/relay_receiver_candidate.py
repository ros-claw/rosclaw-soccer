"""Load one independently audited whole receiver only for SIM evaluation."""

import json
from pathlib import Path

import torch

from rosclaw_soccer.providers.g1.velocity_policy import file_hash
from rosclaw_soccer.rsi.relay_whole_receiver_identity import zero_receiver_actor


def load_research_receiver(
    setup,
    checkpoint: Path,
    checkpoint_hash: str,
    audit: Path,
    audit_hash: str,
    dataset: Path,
    dataset_hash: str,
    *,
    dense_mode: bool = False,
    motion_mode: bool = False,
    continuation_mode: bool = False,
    functional_mode: bool = False,
    functional_timed_mode: bool = False,
):
    if (
        type(dense_mode) is not bool
        or type(motion_mode) is not bool
        or type(continuation_mode) is not bool
        or type(functional_mode) is not bool
        or type(functional_timed_mode) is not bool
        or (functional_timed_mode and not functional_mode)
        or (functional_mode and (not dense_mode or motion_mode or continuation_mode))
        or (motion_mode and not dense_mode)
        or (continuation_mode and (not dense_mode or motion_mode))
    ):
        raise ValueError("explicit simulation research protocol required")
    for path, expected in (
        (checkpoint, checkpoint_hash),
        (audit, audit_hash),
        (dataset, dataset_hash),
    ):
        if path.is_symlink() or file_hash(path) != expected:
            raise ValueError("exact independently audited candidate inputs required")
    review = json.loads(audit.read_text())
    source = json.loads(dataset.read_text())
    if continuation_mode:
        if (
            review.get("status") != "LEARNED_RECEIVER_CONTINUATION_NUMERIC_READBACK_MATCH"
            or review.get("checkpoint_hash") != checkpoint_hash
            or review.get("dataset_hash") != dataset_hash
            or review.get("episodes") != 24
            or review.get("sampled_controls") != source.get("sampled_controls")
            or review.get("physical_retention_proven") is not False
            or review.get("independent_new_physics") is not False
            or review.get("promotion_authorized") is not False
            or review.get("hardware_authorized") is not False
            or source.get("status") != "LEARNED_RECEIVER_PILOT_PHYSICAL_DATA_NOT_TRAINED"
            or source.get("model_identity")
            != [
                str(setup.velocity_path),
                setup.pins[str(setup.velocity_path)],
                str(setup.velocity_config),
                setup.pins[str(setup.velocity_config)],
            ]
        ):
            raise ValueError("exact numerically audited SIM continuation required")
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if (
            state.get("dataset_hash") != dataset_hash
            or state.get("dataset_audit_hash") != review.get("dataset_audit_hash")
            or state.get("parent_hash") != review.get("parent_hash")
            or any(
                state.get(k) is not False
                for k in (
                    "physical_retention_proven",
                    "promotion_authorized",
                    "hardware_authorized",
                )
            )
        ):
            raise ValueError("exact unpromoted SIM continuation checkpoint required")
        actor = zero_receiver_actor(setup)
        actor.plastic.load_state_dict(state["actor"], strict=True)
        if any(not torch.isfinite(p).all() for p in actor.parameters()):
            raise ValueError("finite candidate required")
        if any(
            file_hash(path) != expected
            for path, expected in (
                (checkpoint, checkpoint_hash),
                (audit, audit_hash),
                (dataset, dataset_hash),
            )
        ):
            raise ValueError("candidate evidence changed during loading")
        return actor
    if (
        review.get("status")
        != (
            "FUNCTIONAL_TIMED_RECEIVER_CANDIDATE_READBACK_MATCH"
            if functional_timed_mode
            else "FUNCTIONAL_RECEIVER_CANDIDATE_READBACK_MATCH"
            if functional_mode
            else "MOTION_RECEIVER_CANDIDATE_READBACK_MATCH"
            if motion_mode
            else "DENSE_RECEIVER_CANDIDATE_READBACK_MATCH"
            if dense_mode
            else "RECEIVER_OFFLINE_CANDIDATE_READBACK_MATCH"
        )
        or review.get("checkpoint_hash") != checkpoint_hash
        or review.get("episodes") != 72
        or review.get("sampled_controls") != (18903 if dense_mode else 18482)
        or (
            dense_mode
            and (
                review.get("dataset_hash") != dataset_hash
                or source.get("reward_profile") != "center-strict-full-history-v1"
            )
        )
        or review.get("physical_retention_proven") is not False
        or review.get("promotion_authorized") is not False
        or review.get("hardware_authorized") is not False
        or source.get("model_identity")
        != [
            str(setup.velocity_path),
            setup.pins[str(setup.velocity_path)],
            str(setup.velocity_config),
            setup.pins[str(setup.velocity_config)],
        ]
    ):
        raise ValueError("exact audited candidate and velocity reference required")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (
        state.get("dataset_hash") != dataset_hash
        or state.get("promotion_authorized") is not False
        or state.get("hardware_authorized") is not False
        or state.get("physical_retention_proven") is not False
        or (dense_mode and state.get("dense_audit_hash") != review.get("dense_audit_hash"))
        or (
            functional_mode
            and (
                state.get("feedback_profile")
                != (
                    "functional-receive-timed-proxy-v2"
                    if functional_timed_mode
                    else "functional-receive-proxy-v1"
                )
                or review.get("feedback_profile")
                != (
                    "functional-receive-timed-proxy-v2"
                    if functional_timed_mode
                    else "functional-receive-proxy-v1"
                )
                or (
                    functional_timed_mode
                    and (
                        not state.get("failure_timing_hash")
                        or state.get("failure_timing_hash") != review.get("failure_timing_hash")
                    )
                )
            )
        )
        or (
            motion_mode
            and (
                state.get("feedback_profile") != "receiver-motion-kinematic-v2"
                or review.get("feedback_profile") != "receiver-motion-kinematic-v2"
            )
        )
    ):
        raise ValueError("unpromoted SIM research checkpoint required")
    actor = zero_receiver_actor(setup)
    actor.plastic.load_state_dict(state["actor"], strict=True)
    if any(not torch.isfinite(p).all() for p in actor.parameters()):
        raise ValueError("finite candidate required")
    for path, expected in (
        (checkpoint, checkpoint_hash),
        (audit, audit_hash),
        (dataset, dataset_hash),
    ):
        if file_hash(path) != expected:
            raise ValueError("candidate evidence changed during loading")
    return actor
