import copy
import subprocess
import sys

import pytest
from test_stochastic_refinement_pilot import fixture as old_fixture

from rosclaw_soccer.rsi.parallel_refinement_batch import (
    pair_plan,
    parallel_config,
    verify_parallel_batch,
)


def fixture(seeds=32):
    old, courses = old_fixture()
    cfg = parallel_config(25, seeds)
    records = []
    for pair in range(cfg["worlds"] // 2):
        for p in pair_plan(cfg, courses, pair):
            records.append(copy.deepcopy(old["records"][0]) | p)
    return dict(
        config=cfg,
        records=records,
        counts=dict(native_steps=3890 * cfg["worlds"], actor_calls=778 * cfg["worlds"]),
    ), courses


@pytest.mark.parametrize("seeds", [4, 32])
def test_all_worlds_exact_plan_no_old_arm_or_threshold_change(seeds):
    r, courses = fixture(seeds)
    verify_parallel_batch(r, courses)
    assert len(r["records"]) == 3 * seeds
    assert r["config"]["maximum_native_steps"] == 12000 * seeds
    assert r["config"]["optimizer_updates"] == 0


def test_calibration_and_training_noise_are_disjoint():
    assert not set(parallel_config(25, 4)["seeds"]) & set(parallel_config(25, 32)["seeds"])


def test_full_coverage_new_version_seeds_and_all_recovery_rows():
    from rosclaw_soccer.rsi.refinement_actor_clock import FULL

    r, courses = fixture()
    cfg = parallel_config(25, 32, full_recovery=True)
    r["config"] = cfg
    for pair in range(48):
        for plan in pair_plan(cfg, courses, pair):
            row = r["records"][plan["index"]]
            row.update(
                plan,
                controller_kind=FULL,
                stochastic_controls=389,
                deterministic_recovery_excluded_from_actor_rows=False,
            )
    verify_parallel_batch(r, courses)
    used = set()
    for full in (False, True):
        for seeds in (4, 32):
            new = set(parallel_config(25, seeds, full_recovery=full)["seeds"])
            assert not used & new
            used |= new
    assert cfg["formal_motion_deadline_controls"] == 300
    r["records"][0]["stochastic_controls"] = 289
    with pytest.raises(ValueError):
        verify_parallel_batch(r, courses)


@pytest.mark.parametrize("projected", [False, True])
def test_parallel_role_intervention_uses_same_noise_new_environment_not_old_PPO_protocol(projected):
    from rosclaw_soccer.rsi.refinement_actor_clock import DIAGNOSTIC, PROJECTED_DIAGNOSTIC

    r, _ = fixture()
    cfg = parallel_config(
        26, 32, full_recovery=True, receiver_role_ablation=True, projected_capture=projected
    )
    r["config"] = cfg
    courses = [r["records"][i]["course"] | dict(state_id=26) for i in (0, 32, 64)]
    for pair in range(48):
        for plan in pair_plan(cfg, courses, pair):
            r["records"][plan["index"]].update(
                plan,
                controller_kind=PROJECTED_DIAGNOSTIC if projected else DIAGNOSTIC,
                stochastic_controls=389,
                deterministic_recovery_excluded_from_actor_rows=False,
                receiver_intent_mode=cfg["receiver_intent_mode"],
                receiver_role_ablation=True,
            )
    verify_parallel_batch(r, courses)
    assert cfg["seeds"] == parallel_config(26, 32, full_recovery=True)["seeds"]
    assert cfg["noise_is_new"] is False and cfg["factual_worlds_reexecuted"] == 0
    old = copy.deepcopy(r)
    old["config"] = parallel_config(26, 32, full_recovery=True)
    with pytest.raises(ValueError):
        verify_parallel_batch(old, courses)
    r["records"][0]["receiver_intent_mode"] = "pursuit-v1"
    with pytest.raises(ValueError):
        verify_parallel_batch(r, courses)


@pytest.mark.parametrize("state,full", [(25, True), (26, False)])
def test_role_variant_requires_explicit_preregistered_model_and_full_clock(state, full):
    with pytest.raises(ValueError):
        parallel_config(state, 32, full_recovery=full, receiver_role_ablation=True)


@pytest.mark.parametrize("value", [True, 1, None])
def test_projected_mode_cannot_be_silently_enabled_on_old_environment(value):
    with pytest.raises(ValueError):
        parallel_config(26, 32, full_recovery=True, projected_capture=value)


@pytest.mark.parametrize("fault", ["missing", "seed", "count", "PPO", "law", "arm", "tail"])
def test_invalid_partial_or_authority_data_cannot_train(fault):
    r, courses = fixture()
    if fault == "missing":
        r["records"].pop()
    elif fault == "seed":
        r["records"][1]["sampling_seed"] += 1
    elif fault == "count":
        r["counts"]["native_steps"] += 1
    elif fault == "PPO":
        r["records"][1]["PPO_actor_rows"] = 289
    elif fault == "law":
        r["records"][1]["sampling_law"] = "gaussian-ar1"
    elif fault == "arm":
        r["records"][1]["external_teacher"] = True
    else:
        r["records"][1]["stand_controls"] = 99
    with pytest.raises(ValueError):
        verify_parallel_batch(r, courses)


def test_no_import_allocation_and_real_spawn_worker_pickle(tmp_path):
    code = (
        "import concurrent.futures,multiprocessing,runpy,pickle,sys; "
        "runpy.run_module('scripts.rsi_run_parallel_refinement_batch',"
        "run_name='__mp_main__',alter_sys=True); "
        "import scripts.rsi_parallel_refinement_worker as w; "
        "assert sys.modules[w.__name__] is w; "
        "assert pickle.loads(pickle.dumps(w.run_pair)) is w.run_pair; "
        "assert not w.bootstrap_probe()['model_allocated']; "
        "pool=concurrent.futures.ProcessPoolExecutor(max_workers=1,"
        "mp_context=multiprocessing.get_context('spawn')); "
        "r=pool.submit(w.bootstrap_probe).result(timeout=30); pool.shutdown(); "
        "assert r['module_registered'] and r['worker_pickle_roundtrip']; "
        "assert not r['model_allocated']"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=45
    )
    assert result.returncode == 0, result.stderr
    assert not list(tmp_path.iterdir())


def test_large_batch_without_calibration_rejected_before_admission(tmp_path):
    output = tmp_path / "unused"
    command = [
        sys.executable,
        "-m",
        "scripts.rsi_run_parallel_refinement_batch",
        "--experience-root",
        "x",
        "--own-root",
        "x",
        "--training-root",
        "x",
        "--output",
        str(output),
        "--own-hash",
        "h",
        "--training-hash",
        "h",
        "--expected-head",
        "h",
        "--state-id",
        "25",
        "--seeds-per-initial",
        "32",
        "--workers",
        "4",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 2 and not output.exists()
