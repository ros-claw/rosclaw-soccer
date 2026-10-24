"""Once-only readback of all three matched BC400/task-DAgger native pilots.

No simulator, policy, optimizer, video, promotion, or blind retry is launched.
The scheduler may chain this AFTER every own-native stage closes successfully.
"""

import argparse
import json
from pathlib import Path

from rosclaw_soccer.rsi.receiving_command_comparison import SEEDS, admit_pair, compare_records
from rosclaw_soccer.rsi.refinement_onpolicy_checkpoint import file_hash
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_native_context_recovery_common import check_pins
from scripts.rsi_native_fresh_common import clean, source_pins


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("parent-training-root", "child-training-root", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("parent-training-hash", "child-training-hash", "expected-head"):
        p.add_argument("--" + key, required=True)
    p.add_argument(
        "--cohort",
        nargs=6,
        action="append",
        required=True,
        metavar=(
            "SEED",
            "PARENT_SOURCE",
            "PARENT_OWN",
            "PARENT_OWN_HASH",
            "CHILD_SOURCE",
            "CHILD_OWN",
        ),
    )
    a = p.parse_args()
    clean(a.expected_head)
    if a.output.exists() or [int(c[0]) for c in a.cohort] != list(SEEDS):
        raise ValueError("unused output and all three preregistered seed cohorts required")
    pins = {}

    def read(root, expected=None):
        path = root / "complete.json"
        if (
            root.is_symlink()
            or path.is_symlink()
            or not path.is_file()
            or (root / "failed.json").exists()
        ):
            raise ValueError("ordinary closed successful evidence required")
        digest = file_hash(path)
        if expected is not None and digest != expected:
            raise ValueError("predeclared exact receipt hash required")
        d = json.loads(path.read_text())
        for key, value in d["input_file_hashes"].items():
            if key in pins and pins[key] != value:
                raise ValueError("inherited evidence pin conflict")
            pins[key] = value
        pins[str(path)] = digest
        return d, digest

    parent, _ = read(a.parent_training_root, a.parent_training_hash)
    child, _ = read(a.child_training_root, a.child_training_hash)
    if (
        parent["status"] != "RECEIVING_COMMAND_BC_THREE_SEEDS_TRAINED_PENDING_OWN_NATIVE"
        or child["status"] != "RECEIVING_TASK_COMMAND_DAGGER_THREE_SEEDS_PENDING_NATIVE"
        or child["parent_bc_hash"] != a.parent_training_hash
        or parent["optimizer_updates"] != 1200
        or child["optimizer_updates"] != 1200
        or any(d["new_physical_steps"] != 0 for d in (parent, child))
        or any([r["seed"] for r in d["candidates"]] != list(SEEDS) for d in (parent, child))
    ):
        raise ValueError("exact all-three actual BC-to-DAgger training lineage required")
    for pc, cc in zip(parent["candidates"], child["candidates"], strict=True):
        if (
            pc["optimizer_updates"] != 400
            or cc["optimizer_updates"] != 400
            or len(pc["changed_parameter_tensors"]) != 6
            or len(cc["changed_parameter_tensors"]) != 7
            or cc["parent_checkpoint_hash"] != pc["checkpoint_hash"]
            or cc["exact_parent_function_before_updates"] is not True
            or cc["cumulative_prior_steps"] != 800
            or cc["optimizer_parameter_steps"] != [800.0] * 6 + [400.0]
        ):
            raise ValueError("actual parent-preserving seven-tensor continuation required")
        for root, candidate in ((a.parent_training_root, pc), (a.child_training_root, cc)):
            cp = Path(candidate["checkpoint_path"])
            if (
                cp.is_symlink()
                or cp.resolve().parent != root.resolve()
                or file_hash(cp) != candidate["checkpoint_hash"]
            ):
                raise ValueError("exact confined trained checkpoint required")
            pins[str(cp)] = candidate["checkpoint_hash"]

    def primitives(root, report):
        for row in report["records"]:
            trial = root / f"trial{row['index']}"
            if trial.is_symlink():
                raise ValueError("ordinary source trial directory required")
            for artifact in row["artifacts"].values():
                path = root / artifact["path"]
                if (
                    path.is_symlink()
                    or not path.is_file()
                    or path.resolve().parent != root.resolve() / trial.name
                    or file_hash(path) != artifact["hash"]
                ):
                    raise ValueError("exact confined actual native primitives required")
                pins[str(path)] = artifact["hash"]

    effects, cohorts = [], []
    charged_steps = charged_calls = 0
    for declaration in a.cohort:
        seed = int(declaration[0])
        psroot, poroot = map(Path, declaration[1:3])
        csroot, coroot = map(Path, declaration[4:6])
        ps, psh = read(psroot)
        po, poh = read(poroot, declaration[3])
        cs, csh = read(csroot)
        co, coh = read(coroot)
        pr = admit_pair(
            ps, po, source_hash=psh, training_hash=a.parent_training_hash, seed=seed, task=False
        )
        cr = admit_pair(
            cs, co, source_hash=csh, training_hash=a.child_training_hash, seed=seed, task=True
        )
        if any(ps[key] != cs[key] for key in ("training_hash", "initialization_hash")):
            raise ValueError("fixed A theta12 and fixed zero-correction lower B motor required")
        primitives(psroot, ps)
        primitives(csroot, cs)
        effect = dict(seed=seed, **compare_records(pr, cr))
        effects.append(effect)
        cohorts.append(
            dict(
                seed=seed,
                parent_source_hash=psh,
                parent_own_hash=poh,
                child_source_hash=csh,
                child_own_hash=coh,
            )
        )
        for report in (ps, po, cs, co):
            charged_steps += report["counts"]["native_steps"]
            charged_calls += report["counts"]["actor_calls"]
    source_pins(pins)
    check_pins(pins)
    clean(a.expected_head)
    report = dict(
        schema="soccer.receiving_task_command.paired_local_review.v1",
        status="THREE_SEED_TASK_DAGGER_NATIVE_REVIEWED_NOT_PROMOTED",
        source_head=a.expected_head,
        parent_training_hash=a.parent_training_hash,
        child_training_hash=a.child_training_hash,
        cohorts=cohorts,
        effects=effects,
        all_three_local_precision_gain_without_loss=all(
            e["local_strict_gain_without_any_measured_loss"] for e in effects
        ),
        measured_prior_and_current_native_steps=charged_steps,
        measured_prior_and_current_neural_calls=charged_calls,
        physical_steps_executed_by_review=0,
        optimizer_updates=0,
        distinct_physical_initial_conditions=3,
        all_data_consumed_TRAIN=True,
        Fresh_verified=False,
        online_RL_proven=False,
        team_game_verified=False,
        promotion_authorized=False,
        hardware_authorized=False,
        goal_complete=False,
        input_file_hashes=pins,
    )
    a.output.mkdir()
    write_once(a.output / "complete.json", report)
    print(json.dumps(dict(status=report["status"], effects=effects), allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
