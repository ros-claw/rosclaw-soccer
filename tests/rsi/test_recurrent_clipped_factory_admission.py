"""Fail closed before asset/model allocation; not native physics evidence."""

import json

import pytest

from rosclaw_soccer.rsi.cpu_motor_transfer_evidence import audit_cpu_transfer
from rosclaw_soccer.rsi.recurrent_clipped_episode_factory import RecurrentClippedEpisodeFactory
from rosclaw_soccer.sim.contracts import hash_json


@pytest.mark.parametrize(
    "options,kind,mixed",
    [
        (["--step-model", "absent.json"], "foreign", None),
        (["--step-model", "absent.json"], "subclass", None),
        ([], "exact", None),
        (["--step-model", "absent.json", "--proposal-decoder", "owned_snapshot"], "exact", None),
        (["--step-model", "absent.json", "--cached-proposal-envelope"], "exact", None),
        (["--step-model", "absent.json", "--body-response-bundle", "absent.json"], "exact", None),
        (["--step-model", "absent.json", "--foundation-only"], "exact", None),
        *[
            (["--step-model", "absent.json"], "exact", name)
            for name in (
                "sampling_factory",
                "extended_factory",
                "proposal_factory",
                "imitation_factory",
                "recurrent_factory",
                "recurrent_sampling_factory",
            )
        ],
    ],
)
def test_foreign_subclass_mixed_or_unsupported_path_rejected(tmp_path, options, kind, mixed):
    from scripts.rsi_mujoco_motor_transfer import main

    class Derived(RecurrentClippedEpisodeFactory):
        pass

    factory = (
        object()
        if kind == "foreign"
        else object.__new__(Derived if kind == "subclass" else RecurrentClippedEpisodeFactory)
    )
    output = tmp_path / "must-not-exist"
    args = [
        "--scene",
        str(tmp_path / "absent.xml"),
        "--model-root",
        str(tmp_path / "absent"),
        "--late-swing-policy",
        str(tmp_path / "absent.json"),
        "--output-root",
        str(output),
        "--seed",
        "0",
        "--lane",
        "0",
        *options,
    ]
    with pytest.raises(SystemExit) as error:
        main(
            args,
            recurrent_clipped_factory=factory,
            **({mixed: object()} if mixed else {}),
        )
    assert error.value.code == 2
    assert not output.exists()


@pytest.mark.parametrize("declaration", [None, [], {}, {"hardware_authorized": True}])
def test_foreign_factory_provenance_rejected_before_physics(tmp_path, declaration):
    report = {"executed_motor_policy": {}, "recurrent_clipped_factory": declaration}
    report["report_hash"] = hash_json(report)
    (tmp_path / "report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="clipped|contract"):
        audit_cpu_transfer(tmp_path, tmp_path / "must-not-be-read.py")
