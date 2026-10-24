import json

import pytest

from rosclaw_soccer.rsi import committed_motor_checkpoint as admission
from rosclaw_soccer.rsi.dual_motor_checkpoint import digest
from rosclaw_soccer.rsi.neural_option_contract import IDENTITIES, KIND, STATUS, option_config


@pytest.mark.parametrize(
    "fault",
    [
        "optimizer",
        "promotion",
        "descriptor",
        "escape",
        "declaration",
        "kind",
        "termination",
        "symlink",
    ],
)
def test_reference_bundle_rejects_fake_authority_or_identity_before_model_allocation(
    tmp_path, monkeypatch, fault
):
    def no_model(*args, **kwargs):
        raise AssertionError("unqualified identity reached model allocation")

    monkeypatch.setattr(admission, "option_ancestry", no_model)
    ids = {k: "sha256:" + str(i).zfill(64) for i, k in enumerate(IDENTITIES)}
    ids["parent_lineage_hash"] = "parent"
    cp = tmp_path / "options.json"
    descriptor = dict(schema=KIND, **ids)
    if fault == "descriptor":
        descriptor.pop("own97_hash")
    cp.write_text(json.dumps(descriptor))
    declared = dict(
        **ids,
        source_head="head",
        input_file_hashes={},
        config=option_config(),
        checkpoint_path=str(cp),
        checkpoint_hash=digest(cp),
        hardware_authorized=False,
        promotion_authorized=False,
        goal_complete=False,
    )
    if fault == "escape":
        declared["checkpoint_path"] = str(tmp_path.parent / "outside.json")
    if fault == "termination":
        declared["config"]["learned_termination"] = True
    if fault == "symlink":
        target = tmp_path / "target.json"
        cp.rename(target)
        cp.symlink_to(target)
    dp = tmp_path / "declaration.json"
    dp.write_text(json.dumps(declared))
    report = dict(
        **declared,
        status=STATUS,
        checkpoint_kind=KIND,
        declaration_hash=digest(dp),
        checkpoint_format="json.reference_bundle.v1",
        new_optimizer_updates=0,
        physical_steps_executed=0,
        old_retention_proven=False,
        online_RL_claim=False,
    )
    if fault == "optimizer":
        report["new_optimizer_updates"] = 1
    elif fault == "promotion":
        report["promotion_authorized"] = True
    elif fault == "declaration":
        report["declaration_hash"] = "changed"
    elif fault == "kind":
        report["checkpoint_kind"] = "consolidated-adapter-v1"
    rp = tmp_path / "complete.json"
    rp.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        admission.load_committed_motor(
            None, parent_lineage_hash="parent", root=tmp_path, expected_receipt_hash=digest(rp)
        )
