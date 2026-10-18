"""Artifact/request admission only. Synthetic reviews are not physical evidence."""

import argparse
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from rosclaw_soccer.rsi import native_first_touch_episode as kernel
from rosclaw_soccer.rsi.cerebellar_torque_policy import initialize
from rosclaw_soccer.rsi.recurrent_sampling_episode_factory import RecurrentSamplingEpisodeFactory
from rosclaw_soccer.rsi.reviewed_contact_world import (
    ReviewedContactWorld,
    verify_contact_world_request,
)
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

WORLD = "sha256:" + "c" * 64


def artifact(tmp_path, *, public_four=False, change=None, bad_seal=False):
    value = dict(
        schema=(
            "soccer.rsi.mesh_original_public_terminal_readback.v1"
            if public_four
            else "soccer.rsi.mesh_contact_terminal_readback.v1"
        ),
        rows=[
            dict(world_hash=WORLD, outcome=dict(task_pass=False))
            for _ in range(4 if public_four else 3)
        ],
        whole_input_and_witness_files_verified=True,
        shared_payload_logical_seals_rebuilt=True,
        prior_independent_replays_authenticated=True,
        hardware_authorized=False,
        promotion_authorized=False,
    )
    if change:
        change(value)
    value["report_hash"] = hash_json(value)
    if bad_seal:
        value["report_hash"] = "sha256:" + "a" * 64
    path = tmp_path / "review.json"
    path.write_text(json.dumps(value))
    return ReviewedContactWorld(path, hash_bytes(path.read_bytes())), value


@pytest.mark.parametrize("public_four", [False, True])
def test_pinned_numerical_review_does_not_inherit_failed_task_or_teacher_labels(
    tmp_path, public_four
):
    review, value = artifact(tmp_path, public_four=public_four)
    before = copy.deepcopy(value)
    binding = verify_contact_world_request(review, world_hash=WORLD, neural_requested=True)
    assert binding["numerical_review_hash"] == value["report_hash"]
    assert not binding["task_success_inherited"] and not binding["teacher_qualification_inherited"]
    assert not binding["physical_witnesses_reopened_here"]
    assert json.loads(review.path.read_text()) == before


@pytest.mark.parametrize(
    "change",
    [
        lambda v: v.update(whole_input_and_witness_files_verified=False),
        lambda v: v.update(shared_payload_logical_seals_rebuilt=1),
        lambda v: v.update(prior_independent_replays_authenticated=False),
        lambda v: v.update(hardware_authorized=True),
        lambda v: v.update(promotion_authorized=True),
        lambda v: v.update(schema=[]),
        lambda v: v.update(rows=[]),
        lambda v: v["rows"][0].update(world_hash="sha256:" + "d" * 64),
    ],
)
def test_resealed_but_incomplete_or_wrong_review_rejected(tmp_path, change):
    review, _ = artifact(tmp_path, change=change)
    with pytest.raises(ValueError):
        review.verify(WORLD)


def test_whole_review_bytes_and_logical_seal_both_required(tmp_path):
    review, _ = artifact(tmp_path)
    review.path.write_text(review.path.read_text() + " ")
    with pytest.raises(ValueError, match="bytes changed"):
        review.verify(WORLD)
    review, _ = artifact(tmp_path, bad_seal=True)
    with pytest.raises(ValueError, match="whole sealed"):
        review.verify(WORLD)


def test_review_for_another_model_world_and_symlink_are_rejected(tmp_path):
    review, _ = artifact(tmp_path)
    with pytest.raises(ValueError, match="exact new contact world"):
        review.verify("sha256:" + "b" * 64)
    alias = tmp_path / "alias.json"
    alias.symlink_to(review.path)
    with pytest.raises(ValueError, match="ordinary"):
        ReviewedContactWorld(alias, review.expected_file_hash).verify(WORLD)


def test_legacy_and_geometry_only_defaults_unchanged_but_neural_requires_review(tmp_path):
    assert verify_contact_world_request(None, world_hash=None, neural_requested=True) is None
    assert verify_contact_world_request(None, world_hash=WORLD, neural_requested=False) is None
    with pytest.raises(ValueError, match="requires pinned"):
        verify_contact_world_request(None, world_hash=WORLD, neural_requested=True)
    with pytest.raises(ValueError, match="typed"):
        verify_contact_world_request({}, world_hash=WORLD, neural_requested=True)
    review, _ = artifact(tmp_path)
    with pytest.raises(ValueError, match="legacy"):
        verify_contact_world_request(review, world_hash=None, neural_requested=False)
    with pytest.raises(ValueError, match="flag"):
        verify_contact_world_request(None, world_hash=None, neural_requested=1)


@pytest.mark.parametrize("same_world", [False, True])
def test_reviewed_neural_request_keeps_exact_world_binding_before_allocation(
    tmp_path, monkeypatch, same_world
):
    review, _ = artifact(tmp_path)
    model = initialize(
        seed=42, world_hash=WORLD if same_world else "sha256:" + "d" * 64, time_aware=True
    )
    factory = object.__new__(RecurrentSamplingEpisodeFactory)
    monkeypatch.setattr(RecurrentSamplingEpisodeFactory, "_stable", lambda self: None)

    def no_allocation(path):
        raise RuntimeError("mock native allocation boundary")

    monkeypatch.setitem(
        sys.modules, "mujoco", SimpleNamespace(MjSpec=SimpleNamespace(from_file=no_allocation))
    )
    with pytest.raises(
        RuntimeError if same_world else ValueError,
        match="mock native allocation" if same_world else "cannot inherit or retag",
    ):
        kernel.run_native_first_touch_episode(
            argparse.Namespace(
                record_foundation_observation=True,
                foundation_only=False,
                scene=tmp_path / "absent.xml",
            ),
            course=(2.5, 0.0, -0.5),
            partition="DECLARED_DEVELOPMENT_DIAGNOSTIC",
            entry_source_path=Path(kernel.__file__),
            consumed_bank_hash=None,
            recurrent_sampling_factory=factory,
            cerebellar_model=model,
            mesh_task_contact_world_hash=WORLD,
            mesh_world_review=review,
        )
