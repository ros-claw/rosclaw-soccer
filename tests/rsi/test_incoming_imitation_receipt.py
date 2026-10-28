import pytest

from scripts.rsi_probe_motiondecode_incoming import require_unpromoted_imitation


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
