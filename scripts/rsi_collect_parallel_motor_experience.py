"""Four isolated native CLI collectors; each world keeps its own neural/physics state.

No GPU fallback, shared simulation object, multiprocessing manager or synthetic
trajectory merge. Every complete world is subsequently replayed independently.
The exact child entry also passes __mp_main__/pickle/no-physics bootstrap checks
before process allocation, even though these workers use plain CLI subprocesses.
"""

import argparse
import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path

BOOTSTRAP = """import json, pickle, runpy, sys, types
from pathlib import Path
output = Path(sys.argv[1])
before = set(output.rglob('*.npz'))
namespace = runpy.run_module('scripts.rsi_collect_dual_ar_experience', run_name='__mp_main__')
module = types.ModuleType('__mp_main__')
module.__dict__.update(namespace)
sys.modules['__mp_main__'] = module
assert pickle.loads(pickle.dumps(module.main)) is module.main
assert 'mujoco' not in sys.modules and 'torch' not in sys.modules
assert before == set(output.rglob('*.npz'))
print(json.dumps(dict(module_registered=True, entry_pickle_roundtrip=True,
                     no_physics_created=True)))
"""


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("exam-root", "audit-root", "output", "linter"):
        p.add_argument("--" + n, type=Path, required=True)
    for n in ("audit-hash", "expected-head"):
        p.add_argument("--" + n, required=True)
    p.add_argument("--generation-index", type=int, choices=range(100), required=True)
    p.add_argument("--curriculum", choices=("run2", "retention-run64"), default="retention-run64")
    p.add_argument("--sampling-law", choices=("box-truncated-ar1",), default="box-truncated-ar1")
    p.add_argument("--exploration-profile", choices=("micro-v1", "contact-v1"), default="micro-v1")
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    def clean():
        if (
            subprocess.check_output(["git", "status", "--porcelain"])
            or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            != a.expected_head
        ):
            raise ValueError("exact frozen clean parallel collector required")

    clean()
    from rosclaw_soccer.rsi.motor_exploration_profile import exploration_sigma

    sigma = exploration_sigma(a.exploration_profile, a.sampling_law)
    if a.output.exists():
        raise ValueError("unused complete parallel course output required")
    ap = a.audit_root / "complete.json"
    rp = a.exam_root / "complete.json"
    if digest(ap) != a.audit_hash:
        raise ValueError("exact full source candidate exam required")
    ar = json.loads(ap.read_text())
    if (
        ar["status"] != "DUAL_RECURRENT_ALL44_WHOLE_NATIVE_OWN_HISTORY_AUDIT_PASS"
        or ar["source_hash"] != digest(rp)
        or ar["worlds"] != 44
        or ar["hardware_authorized"] is not False
        or ar["promotion_authorized"] is not False
    ):
        raise ValueError("actual independently examined motor required")
    pins = dict(ar["input_file_hashes"])
    for x in (ap, rp):
        pins[str(x)] = digest(x)
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(x): digest(x) for x in folder.rglob("*.py")})
    check_pins(pins)
    from rosclaw_soccer.rsi.motor_exploration_curriculum import motor_curriculum

    c = motor_curriculum(a.curriculum)
    pairs = c.shard_pairs(a.generation_index, 0, 1)
    a.output.mkdir()
    launch = a.output / "launch.json"
    write_once(
        launch,
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            workers=4,
            execution="independent_native_cli_processes",
            pairs=pairs,
            sampling_law="box-truncated-ar1",
            exploration_profile=a.exploration_profile,
            sigma_policy_units=sigma,
            maximum_native_steps=4000 * c.worlds,
            maximum_neural_decisions=400 * c.worlds,
            automatic_retry=False,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    processes = []
    start = time.monotonic()
    try:
        subprocess.run(
            [
                str(a.linter),
                "check",
                "--isolated",
                "--select",
                "F821,F822",
                "--stdin-filename",
                "native_collector_bootstrap.py",
                "-",
            ],
            input=BOOTSTRAP,
            text=True,
            capture_output=True,
            check=True,
            timeout=60,
        )
        boot = subprocess.run(
            [sys.executable, "-c", BOOTSTRAP, str(a.output)],
            env=env,
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )
        proof = json.loads(boot.stdout)
        if proof != dict(
            module_registered=True, entry_pickle_roundtrip=True, no_physics_created=True
        ):
            raise ValueError("exact child module/entry/no-physics bootstrap required")
        bp = a.output / "bootstrap.json"
        write_once(bp, proof)
        for slot in range(4):
            argv = [
                sys.executable,
                "-m",
                "scripts.rsi_collect_dual_ar_experience",
                "--exam-root",
                str(a.exam_root),
                "--audit-root",
                str(a.audit_root),
                "--audit-hash",
                a.audit_hash,
                "--expected-head",
                a.expected_head,
                "--generation-index",
                str(a.generation_index),
                "--curriculum",
                c.name,
                "--sampling-law",
                "box-truncated-ar1",
                "--exploration-profile",
                a.exploration_profile,
                "--shard-count",
                "4",
                "--shard-slot",
                str(slot),
                "--output",
                str(a.output / f"worker{slot}"),
            ]
            processes.append(subprocess.Popen(argv, env=env))
        codes = [p.wait() for p in processes]
        if any(x != 0 for x in codes):
            raise ValueError("native worker failed; preserve every child, no retry")
        records = []
        reports = []
        declarations = []
        workers = []
        final_pins = dict(pins)
        for x in (launch, bp):
            final_pins[str(x)] = digest(x)
        for slot, process in enumerate(processes):
            root = a.output / f"worker{slot}"
            rp, dp = root / "complete.json", root / "declaration.json"
            r, d = [json.loads(x.read_text()) for x in (rp, dp)]
            expected = c.shard_pairs(a.generation_index, slot, 4)
            if (
                r["status"] != "DUAL_AR1_ACTUAL_CALIBRATION_CLOSED_PENDING_NATIVE_AND_LAW_AUDIT"
                or r["source_head"] != a.expected_head
                or r["partial_curriculum_worker"] is not True
                or r["declaration_hash"] != digest(dp)
                or d["shard_count"] != 4
                or d["shard_slot"] != slot
                or d.get("exploration_profile") != a.exploration_profile
                or d["sigma_policy_units"] != sigma
                or [(x["context"], x["seed"]) for x in r["records"]] != expected
                or r["actual_native_steps"] > 4000 * len(expected)
                or r["actual_neural_decisions"] > 400 * len(expected)
                or any(x["rejection"] is not None for x in r["records"])
            ):
                raise ValueError("exact complete disjoint bounded native worker proof required")
            for f, h in r["input_file_hashes"].items():
                if f in final_pins and final_pins[f] != h:
                    raise ValueError("conflicting worker source lineage")
                final_pins[f] = h
            for x in (rp, dp):
                final_pins[str(x)] = digest(x)
            for original in r["records"]:
                row = copy.deepcopy(original)
                for art in row["artifacts"].values():
                    path = root / art["path"]
                    if (
                        path.is_symlink()
                        or not path.resolve().is_relative_to(root.resolve())
                        or digest(path) != art["hash"]
                    ):
                        raise ValueError("exact confined physical worker artifact required")
                    final_pins[str(path)] = art["hash"]
                    art["path"] = str(path.relative_to(a.output))
                records.append(row)
            reports.append(r)
            declarations.append(d)
            workers.append(
                dict(slot=slot, pid=process.pid, worlds=len(expected), receipt_hash=digest(rp))
            )
        records.sort(key=lambda x: (c.context_ids.index(x["context"]), x["seed"]))
        if [(x["context"], x["seed"]) for x in records] != pairs:
            raise ValueError("assembled physical course has duplicates or missing worlds")
        d = copy.deepcopy(declarations[0])
        d.update(
            shard_count=1,
            shard_slot=0,
            execution_worlds=c.worlds,
            input_file_hashes=final_pins,
            preregistered_launch_hash=digest(launch),
            native_worker_count=4,
            execution="four_independent_native_cli_processes",
        )
        write_once(a.output / "declaration.json", d)
        check_pins(final_pins)
        clean()
        write_once(
            a.output / "complete.json",
            dict(
                status="DUAL_AR1_ACTUAL_CALIBRATION_CLOSED_PENDING_NATIVE_AND_LAW_AUDIT",
                source_head=a.expected_head,
                input_file_hashes=final_pins,
                declaration_hash=digest(a.output / "declaration.json"),
                training_hash=reports[0]["training_hash"],
                records=records,
                worker_proofs=workers,
                actual_native_steps=sum(x["actual_native_steps"] for x in reports),
                actual_neural_decisions=sum(x["actual_neural_decisions"] for x in reports),
                stochastic_executed=sum(x["stochastic_executed"] for x in reports),
                collection_wall_seconds=time.monotonic() - start,
                actor_updates=0,
                critic_updates=0,
                actor_training_admitted=False,
                hardware_authorized=False,
                promotion_authorized=False,
                goal_complete=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failed.json",
            dict(
                error=repr(error),
                child_pids=[x.pid for x in processes],
                child_returncodes=[x.poll() for x in processes],
                automatic_retry=False,
                no_missing_worker_is_assumed_zero_work=True,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
