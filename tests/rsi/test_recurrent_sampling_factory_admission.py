"""Native admission and persisted declarations, not physical qualification."""

import copy

import pytest

from rosclaw_soccer.rsi.recurrent_sampling_episode_factory import (
    RecurrentSamplingEpisodeFactory,
    validate_compilation_contract,
)
from rosclaw_soccer.sim.contracts import hash_json
from tests.rsi.test_current_memory_motor import current  # noqa: F401
from tests.rsi.test_imitation_proposal_motor import imitation_parent  # noqa: F401
from tests.rsi.test_kernel_guarded_step_motor import candidate  # noqa: F401
from tests.rsi.test_recurrent_clipped_motor import learned  # noqa: F401
from tests.rsi.test_smooth_memory_motor import smooth_parent  # noqa: F401
from tests.rsi.test_step_motor_network import model  # noqa: F401


def test_declaration_binds_full_mean_sources_and_non_authorizing_types(learned):  # noqa: F811
    factory = RecurrentSamplingEpisodeFactory(learned[-1])
    policy = factory.preview(factory.sampling_view(seed=773))
    contract = factory.contract()
    validate_compilation_contract(contract, policy)
    for field, value in (
        ("hardware_authorized", 0),
        ("promotion_authorized", True),
        ("native_transport_qualification_performed", True),
        ("complete_canonical_mean_hash", "sha256:" + "0" * 64),
        ("source_pins", {}),
        ("unexpected", False),
    ):
        changed = copy.deepcopy(contract)
        changed[field] = value
        with pytest.raises(ValueError, match="contract"):
            validate_compilation_contract(changed, policy)
    changed = copy.deepcopy(policy)
    changed["step_motor_proof"]["model"]["mean_model"]["critic_parameters"]["bias_1"][0] += 1
    changed["policy_hash"] = hash_json({k: v for k, v in changed.items() if k != "policy_hash"})
    with pytest.raises(ValueError, match="contract"):
        validate_compilation_contract(contract, changed)
    changed["policy_hash"] = "changed"
    with pytest.raises(ValueError, match="preview"):
        validate_compilation_contract(contract, changed)
    for invalid in (None, [], {"step_motor_proof": {"model": None}}):
        with pytest.raises(ValueError, match="identity"):
            validate_compilation_contract(contract, invalid)


@pytest.mark.parametrize(
    "options,foreign,other",
    [
        (["--step-model", "missing.json"], True, None),
        ([], False, None),
        (["--step-model", "missing.json", "--proposal-decoder", "owned_snapshot"], False, None),
        (["--step-model", "missing.json", "--cached-proposal-envelope"], False, None),
        (["--step-model", "missing.json", "--body-response-bundle", "missing.json"], False, None),
        (["--step-model", "missing.json", "--foundation-only"], False, None),
        (["--step-model", "missing.json"], False, "sampling_factory"),
        (["--step-model", "missing.json"], False, "proposal_factory"),
        (["--step-model", "missing.json"], False, "extended_factory"),
        (["--step-model", "missing.json"], False, "imitation_factory"),
        (["--step-model", "missing.json"], False, "recurrent_factory"),
    ],
)
def test_foreign_or_mixed_native_factories_rejected(tmp_path, options, foreign, other):
    from scripts.rsi_mujoco_motor_transfer import main

    factory = object() if foreign else object.__new__(RecurrentSamplingEpisodeFactory)
    output = tmp_path / "must-not-exist"
    args = [
        "--scene",
        str(tmp_path / "absent.xml"),
        "--model-root",
        str(tmp_path / "absent"),
        "--late-swing-policy",
        str(tmp_path / "absent.json"),
        "--seed",
        "0",
        "--lane",
        "0",
        "--output-root",
        str(output),
        *options,
    ]
    with pytest.raises(SystemExit) as error:
        main(args, recurrent_sampling_factory=factory, **({other: object()} if other else {}))
    assert error.value.code == 2
    assert not output.exists()
