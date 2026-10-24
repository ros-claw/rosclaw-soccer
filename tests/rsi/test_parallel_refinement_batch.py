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
