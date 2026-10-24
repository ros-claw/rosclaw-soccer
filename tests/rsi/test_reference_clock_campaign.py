import json
import sys
from pathlib import Path

import pytest

from scripts import rsi_native_fresh_common as common
from scripts import rsi_run_reference_clock_campaign as campaign
from scripts.rsi_native_context_recovery_common import digest


def setup_campaign(tmp_path, monkeypatch, *, identity=True, exact=True):
    plan = tmp_path / "plan.json"
    output = tmp_path / "evidence"
    value = dict(
        protocol="robonaldo.reference_clock_authority.v1",
        source_commit="fake-head",
        reference_rates=[1.0, 1.06, 1.12, 1.18],
        hardware_authorized=False,
        core_root="unused",
        core_commit="fake-core",
    )
    for key in (
        "source_root",
        "source_hash",
        "own_root",
        "own_hash",
        "identity_source_root",
        "identity_source_hash",
        "identity_own_root",
        "identity_own_hash",
    ):
        value[key] = "unused"
    plan.write_text(json.dumps(value))
    monkeypatch.setattr(common, "clean", lambda head: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "campaign",
            "--plan",
            str(plan),
            "--plan-hash",
            digest(plan),
            "--output",
            str(output),
            "--expected-head",
            "fake-head",
        ],
    )
    calls = []

    def fake_worker(args, *, check):
        assert check
        calls.append(args)
        dest = Path(args[args.index("--output") + 1])
        dest.mkdir()
        if args[2] == "scripts.rsi_probe_live_neural_transition":
            rate = float(
                next(arg.split("=", 1)[1] for arg in args if arg.startswith("--reference-rate="))
            )
            (dest / "declaration.json").write_text(
                json.dumps(dict(clock_plan_hash=digest(plan), reference_rate=rate))
            )
            report = dict(
                source_head="fake-head",
                records=[
                    dict(zero_intervention_full_identity={"qpos": identity}) for _ in range(4)
                ],
            )
        else:
            report = dict(source_hash=args[args.index("--source-hash") + 1], all_exact=exact)
        (dest / "complete.json").write_text(json.dumps(report))

    monkeypatch.setattr(campaign.subprocess, "run", fake_worker)
    return output, calls


def test_actual_successor_calls_and_completed_resume_do_not_rerun(tmp_path, monkeypatch):
    output, calls = setup_campaign(tmp_path, monkeypatch)
    campaign.main()
    assert len(calls) == 8
    assert len(json.loads((output / "complete.json").read_text())["results"]) == 4
    campaign.main()
    assert len(calls) == 8


@pytest.mark.parametrize("identity,exact,expected_calls", [(False, True, 1), (True, False, 2)])
def test_identity_or_own_failure_blocks_accelerated_successors(
    tmp_path, monkeypatch, identity, exact, expected_calls
):
    output, calls = setup_campaign(tmp_path, monkeypatch, identity=identity, exact=exact)
    with pytest.raises(ValueError):
        campaign.main()
    assert len(calls) == expected_calls
    assert not (output / "complete.json").exists()
    assert (output / "rate-0" / "complete.json").exists()
