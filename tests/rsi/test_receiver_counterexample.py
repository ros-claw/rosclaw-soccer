import copy

import pytest
from test_parallel_refinement_batch import fixture as factual_fixture

from rosclaw_soccer.rsi.receiver_counterexample import config, verify


def fixture():
    factual, _ = factual_fixture()
    from rosclaw_soccer.rsi.parallel_refinement_batch import pair_plan, parallel_config

    factual["config"] = parallel_config(26, 32, full_recovery=True)
    courses = [factual["records"][i]["course"] | dict(state_id=26) for i in (0, 32, 64)]
    for pair in range(48):
        for plan in pair_plan(factual["config"], courses, pair):
            factual["records"][plan["index"]].update(plan)
    cfg = config()
    rows = []
    for i in range(4):
        j = cfg["source_case_indices"][i // 2]
        row = copy.deepcopy(factual["records"][j])
        row.update(
            index=i,
            source_case_index=j,
            receiver_intent_mode="anchor-plane-v2",
            receiver_role_ablation=True,
            controller_kind="receiver-role-stochastic-diagnostic-native-v3",
            stochastic_controls=389,
            deterministic_recovery_excluded_from_actor_rows=False,
        )
        rows.append(row)
    return dict(
        config=cfg, records=rows, counts=dict(native_steps=15560, actor_calls=3112)
    ), factual


def test_exact_four_new_worlds_reuse_factual_and_are_not_PPO_or_Fresh():
    r, f = fixture()
    verify(r, f)
    assert r["config"]["new_worlds"] == 4
    assert r["config"]["selected_consumed_TRAIN_not_Fresh"]
    assert r["config"]["formal_controls"] == 300


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_case_index", 24),
        ("sampling_seed", 1),
        ("receiver_role_ablation", False),
        ("receiver_intent_mode", "pursuit-v1"),
        ("controller_kind", "pure-refinement-full-recovery-stochastic-native-v2"),
        ("deterministic_recovery_excluded_from_actor_rows", True),
        ("stochastic_controls", 289),
        ("PPO_actor_rows", 389),
        ("neural_actor_optimizer_updates", 1),
        ("native_steps", 4000),
        ("stand_controls", 99),
        ("requires_whole_native_and_law_audit", False),
        ("feedback_variant", 1),
    ],
)
def test_unclosed_or_mislabelled_interventions_rejected(field, value):
    r, f = fixture()
    r["records"][0][field] = value
    with pytest.raises(ValueError):
        verify(r, f)


def test_complete_failure_remains_observed_not_padded():
    r, f = fixture()
    for row in r["records"][:2]:
        row.update(
            termination="BODY_FAILURE",
            controls=90,
            stochastic_controls=90,
            total_observed_controls=90,
            stand_controls=0,
            observed_recovery_controls=0,
            native_steps=900,
            actor_calls=180,
        )
    r["counts"] = dict(native_steps=9580, actor_calls=1916)
    verify(r, f)


def test_diagnostic_role_rows_cannot_be_admitted_by_legacy_or_full_PPO():
    from rosclaw_soccer.rsi.refinement_actor_critic import _validate_episode

    r, _ = fixture()
    with pytest.raises(ValueError, match="pure learner"):
        _validate_episode(r["records"][0])


@pytest.mark.parametrize(
    "module_name", ["rsi_pass_transfer_trial", "rsi_pass_transfer_replay_trial"]
)
def test_counterexample_mode_requires_explicit_actual_full_sampling_before_allocation(module_name):
    import importlib
    from types import SimpleNamespace

    m = importlib.import_module("scripts." + module_name)
    setup = SimpleNamespace(prepare=lambda *a, **k: pytest.fail("no allocation"))
    args = dict(
        output=None,
        index=0,
        counts={},
        receiver_role_ablation=True,
        receiver_intent_mode="anchor-plane-v2",
        sampling_seed=1,
        sampling_profile="contact-v1",
        sampling_include_recovery=False,
    )
    with pytest.raises(ValueError):
        if module_name.endswith("replay_trial"):
            m.replay_trial(setup, None, dict(assisted=False), {}, raw_root=None, **args)
        else:
            m.collect_trial(setup, None, dict(assisted=False), **args)
