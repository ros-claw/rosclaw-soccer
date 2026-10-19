"""Synthetic whole-nine extraction protocol; no physical evidence fabrication."""

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import rosclaw.growth.canonical_json_snapshot as snapshot

from rosclaw_soccer.rsi import precontact_heading_pairs as helper
from rosclaw_soccer.sim.contracts import hash_json
from scripts import rsi_read_precontact_heading_pairs as mod
from scripts.rsi_stage_release_contact_replay import digest
from tests.rsi.test_postcontact_full_bank_readback import complete
from tests.rsi.test_precontact_heading_pairs import trace


@pytest.mark.parametrize("mode", ["complete", "fatal", "preflight", "duplicate"])
def test_whole9_preimport_proof_gated_extraction(monkeypatch, tmp_path, mode):
    def seal(path, value):
        value = dict(value)
        value["report_hash"] = hash_json({k: v for k, v in value.items() if k != "report_hash"})
        path.write_text(json.dumps(value))
        return value

    bank, phys, output = tmp_path / "bank", tmp_path / "physics", tmp_path / "pairs"
    bank.mkdir()
    phys.mkdir()
    contexts, proofs = [], []
    pairs = {ci: pair for pair, ids in enumerate(helper.PAIRS) for ci in ids}
    for ci in range(18):
        pair = pairs[ci]
        on = helper.PAIRS[pair][1] == ci
        t = trace(on)
        course = [2.4 + 0.1 * pair, 0.1, 0.3]
        t["canonical_qpos"][:, 0, 36:39] = (2.4 if mode == "duplicate" else course[0], 0.1, 0.13)
        arrays = bank / f"context-{ci}-trace.npz"
        np.savez_compressed(arrays, **t)
        rows = []
        for ai, name in enumerate([*helper.ACTION_NAMES, "baseline"]):
            row = seal(
                bank / f"context-{ci}-action-{ai}-complete.json",
                dict(
                    context_index=ci,
                    action_index=ai,
                    action_name=name,
                    original_policy_hash=f"policy-{pair}",
                    course=course,
                    outcome=helper.label_first_touch(t).to_dict(),
                    arrays=dict(path=str(arrays), sha256=digest(arrays)),
                ),
            )
            rows.append(row)
            proofs.append(
                seal(
                    phys / f"context-{ci}-action-{ai}-complete.json",
                    dict(
                        context_index=ci,
                        action_index=ai,
                        source_bank_case_hash=row["report_hash"],
                        physical_execution_reused=ci == 1,
                        physical_review=dict(
                            actual_physical_steps=3000, maximum_absolute_errors=dict(q=0.0)
                        ),
                    ),
                )
            )
        contexts.append(
            seal(
                bank / f"context-{ci}-complete.json",
                dict(
                    context_index=ci,
                    rows=rows,
                    source_index=ci // 2 if ci < 12 else ci - 12,
                    partition="CONSUMED_LOCAL_DEVELOPMENT"
                    if ci < 12
                    else "CONSUMED_ORIGINAL_RETENTION",
                ),
            )
        )
    c, d = complete()
    d["config"]["output_root"] = str(bank)
    d = seal(bank / "declaration.json", d)
    c.update(contexts=contexts, declaration_hash=d["report_hash"])
    c = seal(bank / "complete.json", c)
    rb = seal(
        tmp_path / "raw.json",
        dict(
            schema="soccer.rsi.postcontact_full_bank_readback.v1",
            complete_hash=c["report_hash"],
            input_file_hashes={},
            contexts=[dict(source_context_hash=r["report_hash"]) for r in contexts],
        ),
    )
    pd = seal(
        phys / "declaration.json",
        dict(
            raw_readback_hash=rb["report_hash"],
            producer_complete_hash=c["report_hash"],
            input_file_hashes={},
        ),
    )
    seal(
        phys / "complete.json",
        dict(
            schema="soccer.rsi.recorded_postcontact_full_bank_complete.v1",
            declaration_hash=pd["report_hash"],
            rows=proofs,
            comparison_bodies_verified=252,
            new_physical_bodies=238,
            new_physical_steps=714000,
            reused_physical_bodies=14,
            reused_physical_steps=42000,
            new_neural_inferences=0,
            new_optimizer_updates=0,
            teacher_qualified=False,
            training_authorized=False,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    argv = [
        "pairs",
        "--bank-root",
        str(bank),
        "--physics-root",
        str(phys),
        "--physics-unit",
        "physics.service",
        "--readback",
        str(tmp_path / "raw.json"),
        "--readback-file-hash",
        digest(tmp_path / "raw.json"),
        "--source-head",
        "fake",
        "--core-source",
        str(Path(snapshot.__file__).parents[2]),
        "--output",
        str(output),
    ]
    if mode == "preflight":
        argv.append("--preflight")
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(
        mod.subprocess, "check_output", lambda argv, **kw: "fake" if "rev-parse" in argv else ""
    )
    from scripts import rsi_stage_release_contact_replay as stage

    units = []
    monkeypatch.setattr(stage, "require_terminal", units.append)
    original = helper.paired_context
    calls = []

    def extract(*args, **kwargs):
        if mode == "fatal" and kwargs["index"] == 3:
            raise ValueError("synthetic failed pair")
        calls.append(kwargs["index"])
        return original(*args, **kwargs)

    monkeypatch.setattr(helper, "paired_context", extract)
    if mode in ("fatal", "duplicate"):
        with pytest.raises(ValueError, match="failed pair|distinct original"):
            mod.main()
        failure = json.loads((output / "failure.json").read_text())
        assert failure["completed_pairs"] == (3 if mode == "fatal" else 9)
        assert failure["new_physical_steps"] == 0 and not failure["automatic_retry"]
        assert not (output / "complete.json").exists()
    elif mode == "preflight":
        mod.main()
        assert not output.exists() and calls == []
    else:
        mod.main()
        result = json.loads((output / "complete.json").read_text())
        assert len(result["rows"]) == result["distinct_predecision_states"] == 9
        assert result["unique_heading_action_labels"] == 234
        assert result["original_comparison_bodies_covered"] == 252
        assert result["new_neural_inferences"] == result["new_physical_steps"] == 0
        assert not result["training_authorized"] and not result["teacher_qualified"]
        declaration = json.loads((output / "declaration.json").read_text())
        assert (
            declaration["source_pinned_before_import"]
            and len(declaration["input_file_hashes"]) > 100
        )
        assert sum(r["partition"] == "CONSUMED_LOCAL_DEVELOPMENT" for r in result["rows"]) == 6
        assert calls == list(range(9))
        changed = copy.deepcopy(result["rows"][0])
        changed["outcomes_by_heading_then_action"][0][0]["task_pass"] = False
        assert changed != result["rows"][0]
    assert units == ["physics.service"]
