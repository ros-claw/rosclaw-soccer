import copy

import pytest

from rosclaw_soccer.rsi.neural_option_contract import OWN97_STATUS, option_config, option_cpu_scope


def proof():
    return dict(
        status=OWN97_STATUS,
        physical_steps_executed=0,
        optimizer_updates=0,
        own_actor_decisions=36276,
        combined_physical_performance_proven=False,
        native_checkpoint_admitted=False,
        Fresh_verified=False,
        team_game_verified=False,
        hardware_authorized=False,
        promotion_authorized=False,
        goal_complete=False,
        selected_neural_skills_have_exact_actions_on_recorded_streams=True,
        records=[
            dict(
                index=i,
                rows=372 if i == 0 else 374,
                scope="public_pass" if i < 80 else "protected_motion",
                option=0,
                all_source_actions_bitwise_equal=True,
                all_own_hc_bitwise_equal=True,
                option_committed_through_recovery=True,
            )
            for i in range(97)
        ],
    )


def test_own97_proof_is_complete_fixed_and_never_physics_or_training():
    assert option_cpu_scope(proof()) == 36276
    cfg = option_config()
    assert cfg["new_optimizer_updates"] == 0 and cfg["learned_termination"] is False
    cfg["source_head_seeds"].clear()
    assert len(option_config()["source_head_seeds"]) == 3


@pytest.mark.parametrize(
    "fault",
    [
        "short",
        "count",
        "physics",
        "updates",
        "scope",
        "old_skill",
        "choice",
        "claim",
        "borrowed",
        "uncommitted",
        "row_age",
    ],
)
def test_partial_incoherent_or_promoted_proof_cannot_admit_options(fault):
    r = copy.deepcopy(proof())
    if fault == "short":
        r["records"].pop()
    elif fault == "count":
        r["own_actor_decisions"] -= 1
    elif fault == "physics":
        r["physical_steps_executed"] = 1
    elif fault == "updates":
        r["optimizer_updates"] = 1
    elif fault == "scope":
        r["records"][0]["scope"] = "protected_motion"
    elif fault == "old_skill":
        r["records"][-1]["option"] = 1
    elif fault == "choice":
        r["records"][0]["option"] = True
    elif fault == "claim":
        r["combined_physical_performance_proven"] = True
    elif fault == "borrowed":
        r["records"][0]["all_own_hc_bitwise_equal"] = False
    elif fault == "uncommitted":
        r["records"][0]["option_committed_through_recovery"] = False
    else:
        r["records"][0]["rows"] = 0
    with pytest.raises(ValueError):
        option_cpu_scope(r)
