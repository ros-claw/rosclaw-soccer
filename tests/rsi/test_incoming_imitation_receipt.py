import pytest

from scripts.rsi_probe_motiondecode_incoming import (
    imitation_reference_bound,
    require_unpromoted_imitation,
)


def test_imitation_receipt_requires_exact_training_fresh_exam_field():
    receipt = dict(
        status="INCOMING_PPO_PILOT_REQUIRES_INDEPENDENT_EXAM",
        imitation_hash="sha256:example",
        hardware_authorized=False,
        promotion_authorized=False,
        independent_fresh_exam=False,
    )
    require_unpromoted_imitation(receipt)
    for field in ("hardware_authorized", "promotion_authorized", "independent_fresh_exam"):
        with pytest.raises(ValueError):
            require_unpromoted_imitation({**receipt, field: True})
        missing = {k: v for k, v in receipt.items() if k != field}
        with pytest.raises(ValueError):
            require_unpromoted_imitation(missing)


def test_imitation_reference_requires_exact_declared_training_or_dev_binding():
    declaration = {"motion_reference_hash": "train", "input_file_hashes": {"/dev": "dev"}}
    assert imitation_reference_bound(declaration, "/train", "train")
    assert imitation_reference_bound(declaration, "/dev", "dev")
    assert not imitation_reference_bound(declaration, "/other", "dev")
    assert not imitation_reference_bound(declaration, "/dev", "wrong")
