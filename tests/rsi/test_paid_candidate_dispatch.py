import json

import pytest

from rosclaw_soccer.rsi.paid_candidate_dispatch import (
    contact_exam_candidate,
    neural_train_candidate,
)
from rosclaw_soccer.rsi.paid_canonical_behavior import digest
from rosclaw_soccer.rsi.paid_neighborhood_study import neighborhood_config
from rosclaw_soccer.rsi.paid_replay_neural_student import AUDITED_STATUS


@pytest.mark.parametrize("kind", ["frozen_growth_pair_unseen_local_probe32", "anything_else"])
def test_new_neural_never_bypasses_unseen_fixed_pair(kind, tmp_path):
    with pytest.raises(ValueError, match="consumed TRAIN32"):
        neural_train_candidate(
            tmp_path, "hash", {}, dict(plan_kind=kind, config=neighborhood_config())
        )


def test_changed_old_fixture_rejected(tmp_path):
    cfg = neighborhood_config()
    cfg["repeats"] = 1
    with pytest.raises(ValueError, match="consumed TRAIN32"):
        neural_train_candidate(tmp_path, "hash", {}, dict(config=cfg))


def test_unknown_or_forged_model_never_inferred_as_neural(tmp_path):
    p = tmp_path / "complete.json"
    p.write_text(json.dumps(dict(status="fit-count-is-not-admission")))
    with pytest.raises((ValueError, KeyError)):
        contact_exam_candidate(tmp_path, digest(p), {})


def test_exact_digest_checked_before_status_dispatch(tmp_path):
    p = tmp_path / "complete.json"
    p.write_text(json.dumps(dict(status=AUDITED_STATUS)))
    with pytest.raises(ValueError, match="exact closed"):
        contact_exam_candidate(tmp_path, "wrong", {})


def test_typed_neural_metadata_route_is_train_only_not_kernel(tmp_path):
    from rosclaw_soccer.rsi.paid_replay_neural import FIT_STATUS, replay_config

    fit = tmp_path / "fit"
    fit.mkdir()
    cp = fit / "weights.pt"
    cp.write_bytes(b"not_loaded_by_metadata_router")
    shared = dict(
        config=replay_config(0),
        certificate={},
        counts={},
        body_scene_hash="body",
        parent_hash="parent",
        checkpoint_path=str(cp),
        checkpoint_hash=digest(cp),
    )
    fp = fit / "complete.json"
    fp.write_text(json.dumps(dict(status=FIT_STATUS, **shared)))
    r = dict(
        status=AUDITED_STATUS,
        **shared,
        CPU_fit_reconstruction=True,
        own_native_steps=0,
        optimizer_updates=0,
        runtime_admitted=False,
        hardware_authorized=False,
        promotion_authorized=False,
        fit_root=str(fit),
        fit_hash=digest(fp),
        input_file_hashes={},
    )
    root = tmp_path / "audit"
    root.mkdir()
    rp = root / "complete.json"
    rp.write_text(json.dumps(r))
    plan = dict(
        config=neighborhood_config(),
        body_scene_hash="body",
        specifications=dict(models=dict(learned_memory=dict(hash="parent"))),
    )
    spec, cfg, _ = contact_exam_candidate(root, digest(rp), plan)
    assert spec == dict(root=str(root), hash=digest(rp))
    assert cfg["candidate_kind"] == "soft_replay_neural"
    assert cfg["consumed_for_training"] and not cfg["heldout_generalization_claim"]
    assert cfg["maximum_native_steps_per_arm"] == 64 * 3890
    assert cfg["maximum_native_steps"] == 3 * 64 * 3890
    # This reader binds metadata, it intentionally does NOT load the fake CP.
    # The separate full neural loader must reject these bytes before physics.
