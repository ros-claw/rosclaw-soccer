import json

import pytest

from rosclaw_soccer.rsi.contact_consolidation import (
    CORRECTION_STATUS,
    STATUS,
    consolidation_config,
)
from rosclaw_soccer.rsi.dual_motor_checkpoint import digest
from rosclaw_soccer.rsi.dual_onpolicy_checkpoint import load_dual_candidate_for_sim_audit


@pytest.mark.parametrize(
    "stage,status,wrong_kind",
    [
        ("initial", STATUS, "contact-consolidated-v2"),
        ("own-state-correction", CORRECTION_STATUS, "contact-consolidated"),
    ],
)
def test_checkpoint_types_cannot_reinterpret_phases_before_model_or_physics_load(
    tmp_path, stage, status, wrong_kind
):
    declaration = tmp_path / "declaration.json"
    declaration.write_text("{}")
    receipt = tmp_path / "complete.json"
    receipt.write_text(
        json.dumps(
            dict(
                status=status,
                config=consolidation_config(65023, "cpu", stage=stage),
                declaration_hash=digest(declaration),
            )
        )
    )
    with pytest.raises(ValueError, match="phase accounting"):
        load_dual_candidate_for_sim_audit(
            None,
            parent_lineage_hash="not-loaded",
            root=tmp_path,
            expected_receipt_hash=digest(receipt),
            checkpoint_kind=wrong_kind,
        )
