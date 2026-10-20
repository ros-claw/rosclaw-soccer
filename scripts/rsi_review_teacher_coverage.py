"""Authenticate fixed24 teacher coverage and full NN/native reproductions.

Read-only, zero new physical steps/learning. Nine-option teacher results remain
privileged posthoc diagnostics, not general reachability or deployed skill.
"""

import argparse
import json
import re
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("launch", "worker-source", "near-review", "inputs", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in (
        "expected-launch-hash",
        "expected-head",
        "expected-near-review-hash",
        "expected-inputs-hash",
    ):
        p.add_argument("--" + key, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_collect_shadow_motor import wait_for_terminal_collection
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    if (
        a.output.exists()
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
        or subprocess.check_output(["git", "status", "--porcelain"])
        or digest(a.launch) != a.expected_launch_hash
        or digest(a.near_review / "complete.json") != a.expected_near_review_hash
        or digest(a.inputs) != a.expected_inputs_hash
    ):
        raise ValueError("immutable reviewer and exact launch/input/near evidence required")
    launch = json.loads(a.launch.read_text())
    if (
        len(launch["jobs"]) != 17
        or len({j["unit"] for j in launch["jobs"]}) != 17
        or subprocess.check_output(
            ["git", "-C", str(a.worker_source), "rev-parse", "HEAD"], text=True
        ).strip()
        != launch["source_head"]
    ):
        raise ValueError("exact zero plus eight producers/eight reproductions required")
    for directory in ("src", "scripts"):
        for f in (a.worker_source / directory).rglob("*.py"):
            if digest(f) != digest(Path.cwd() / f.relative_to(a.worker_source)):
                raise ValueError("review cannot change teacher control math")
    near = json.loads((a.near_review / "complete.json").read_text())
    pins = dict(near["input_file_hashes"])
    for directory in ("src", "scripts"):
        pins.update({str(f.resolve()): digest(f) for f in Path(directory).rglob("*.py")})
    pins.update(
        {
            str(a.launch): a.expected_launch_hash,
            str(a.near_review / "complete.json"): a.expected_near_review_hash,
            str(a.inputs): a.expected_inputs_hash,
        }
    )
    a.output.mkdir(parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            launch_hash=a.expected_launch_hash,
            new_native_steps=0,
            optimizer_steps=0,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )
    try:
        imports = {}
        charged = 0
        for job in launch["jobs"]:
            wait_for_terminal_collection(job["unit"] + ".service", wait=True)
            root = Path(job["output"])
            d = json.loads((root / "declaration.json").read_text())
            c = json.loads((root / "complete.json").read_text())
            if (
                d["source_head"] != launch["source_head"]
                or c["source_head"] != launch["source_head"]
                or d["indices"] != job["indices"]
                or c["optimizer_steps"] != 0
                or c["hardware_authorized"] is not False
                or c["promotion_authorized"] is not False
            ):
                raise ValueError("fixed source/course/accounting evidence required")
            for k, h in d["input_file_hashes"].items():
                if k in pins and pins[k] != h:
                    raise ValueError("conflicting past/raw/source identity")
                pins[k] = h
            for f in root.rglob("*"):
                if f.is_file():
                    pins[str(f)] = digest(f)
            charged += c["new_native_steps"]
            imports[job["unit"]] = (root, d, c)
        if charged > 1550000:
            raise ValueError("fixed physical budget exceeded")
        check_pins(pins)
        result = review(a, launch, imports, near, pins)
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="PRIVILEGED_TEACHER_FIXED24_REPRODUCTION_AUTHENTICATED",
                source_head=a.expected_head,
                input_file_hashes=pins,
                summary=result,
                reviewed_native_steps=charged,
                new_native_steps=0,
                optimizer_steps=0,
                probe_neural_reexecutions_in_separate_processes_same_implementation=True,
                zero_reexecutions_in_same_process=True,
                reviewer_itself_does_not_reexecute_neural_or_physics=True,
                original_far_or_private_fresh=False,
                hardware_authorized=False,
                promotion_authorized=False,
                goal_complete=False,
            ),
        )
        print("PRIVILEGED_TEACHER_COVERAGE_REVIEWED", result, flush=True)
    except Exception as error:
        write_once(
            a.output / "failure.json",
            dict(error=repr(error), automatic_retry_authorized=False, new_native_steps=0),
        )
        raise


def review(a, launch, imports, near, pins):
    import mujoco
    import numpy as np

    from rosclaw_soccer.rsi.bilateral_entry import choose_bilateral_candidate
    from rosclaw_soccer.rsi.entry_prefix_diagnostics import diagnose_entry_prefix
    from rosclaw_soccer.rsi.full_entry_forecast import score_full_entry
    from rosclaw_soccer.rsi.recorded_entry_review import validate_recording
    from rosclaw_soccer.rsi.teacher_coverage_cohort import teacher_coverage_indices
    from scripts.rsi_motor_student_rollout import PHYSICAL_FIELDS, exact_array
    from scripts.rsi_probe_teacher_coverage import compare_case

    world = mujoco.MjModel.from_binary_path(json.loads(a.inputs.read_text())["world"])
    zero = [j for j in launch["jobs"] if j["role"] == "zero"]
    if (
        len(zero) != 1
        or imports[zero[0]["unit"]][2]["status"] != "NOMINAL_ZERO_TEACHER_NEURAL_NATIVE_REPRODUCED"
        or imports[zero[0]["unit"]][2]["all18_array_fields_byte_exact_to_original"] is not True
    ):
        raise ValueError("actual nominal zero parity prerequisite required")
    zero_root, zero_d, zero_c = imports[zero[0]["unit"]]
    original = [
        Path(k).parent
        for k in zero_d["input_file_hashes"]
        if k.endswith("/rsi-bilateral-expanded40-c0-20261020/complete.json")
    ]
    if len(original) != 1:
        raise ValueError("one original nominal teacher binding required")
    first = zero_root / "first/course00"
    second = zero_root / "second/course00"
    first_row = json.loads((first / "complete.json").read_text())
    second_row = json.loads((second / "complete.json").read_text())
    compare_case(first, original[0] / "course00", first_row, original=True)
    compare_case(second, first, second_row)
    if zero_c["new_native_steps"] != sum(
        r["forecast_native_steps"] + r["endpoint_native_steps"] for r in (first_row, second_row)
    ):
        raise ValueError("zero first/second native charge differs")
    # The prior raw review already authenticated these96; retain exact bindings.
    old_rows = {}
    for name in near["input_file_hashes"]:
        match = re.search(
            r"/rsi-categorical-near96-p\d+-20261020/course(\d{3})/complete.json$", name
        )
        if match:
            i = int(match[1])
            if i in old_rows:
                raise ValueError("one original NN comparison per context required")
            old_rows[i] = (Path(name).parent, json.loads(Path(name).read_text()))
    if set(old_rows) != set(range(96)):
        raise ValueError("complete consumed96 neuronal comparison required")
    result = []
    for part in range(8):
        producers = [j for j in launch["jobs"] if j["role"] == "producer" and j["part"] == part]
        repeaters = [j for j in launch["jobs"] if j["role"] == "reproducer" and j["part"] == part]
        if len(producers) != 1 or len(repeaters) != 1:
            raise ValueError("one exact producer/reproducer per part required")
        p, pd, pc = imports[producers[0]["unit"]]
        q, qd, qc = imports[repeaters[0]["unit"]]
        indices = list(teacher_coverage_indices()[part::8])
        if (
            pd["indices"] != indices
            or qd["indices"] != indices
            or pc["status"] != "PRIVILEGED_TEACHER_COVERAGE_PRODUCED"
            or qc["status"] != "PRIVILEGED_TEACHER_COVERAGE_REPRODUCED"
            or len(pc["results"]) != 3
            or len(qc["results"]) != 3
            or qc["same_implementation_neural_and_physics_reexecuted"] is not True
            or pd["teacher_evidence_hash"] != qd["teacher_evidence_hash"]
            or pc["new_native_steps"] != qc["new_native_steps"]
        ):
            raise ValueError("complete matched three-case NN/native repeats required")
        for i, row, repeated in zip(indices, pc["results"], qc["results"], strict=True):
            folder = p / f"course{i:02d}"
            compare_case(q / f"course{i:02d}", folder, repeated)
            if (
                json.loads((folder / "complete.json").read_text()) != row
                or row["course"]["course_index"] != i
                or len(row["candidates"]) != 9
                or row["forecast_native_steps"] != 25200
                or row["endpoint_native_steps"] != (3000 if row["shared_endpoint"] else 5800)
                or choose_bilateral_candidate(row["candidates"])
                != row["selected"]["selected_index"]
            ):
                raise ValueError("exact full nine-option teacher case required")
            with np.load(folder / "own-prefix20.npz", allow_pickle=False) as z:
                prefix = {k: z[k].copy() for k in z.files}
            validate_recording(prefix, frames=20, nbody=world.nbody)
            prefix_d = diagnose_entry_prefix(world, prefix)
            if prefix_d != row["prefix_diagnostics"]:
                raise ValueError("prefix attribution differs")
            old_folder, old = old_rows[i]
            if old["course"] != row["course"]:
                raise ValueError("teacher/learner launch identities differ")
            with np.load(old_folder / "anchor/native.npz", allow_pickle=False) as z:
                for key in PHYSICAL_FIELDS:
                    exact_array(prefix[key], z[key][:20])
                exact_array(prefix["neural_action_isaac"], z["consumed_action_isaac"][:20])

            def score(raw, prefix_d=prefix_d):
                s = score_full_entry(world, raw)
                s["original_suffix_qualified"] = s["qualified"]
                s["common_prefix_physical_safe"] = prefix_d["prefix_physical_safe"]
                if not prefix_d["prefix_physical_safe"]:
                    s["qualified"], s["cost"] = False, None
                    s["physical_prefix_and_suffix_safe"] = False
                return s

            for k, c in enumerate(row["candidates"]):
                if c["index"] != k or c["artifact"] != f"candidate{k}.npz":
                    raise ValueError("ordered complete future set required")
                with np.load(folder / c["artifact"], allow_pickle=False) as z:
                    raw = {name: z[name].copy() for name in z.files}
                validate_recording(raw, frames=280, nbody=world.nbody)
                if score(raw) != c["score"]:
                    raise ValueError("future raw outcome differs")
            if row["selected"]["artifact"] != "selected-actual.npz":
                raise ValueError("bounded selected endpoint path required")
            with np.load(folder / "selected-actual.npz", allow_pickle=False) as z:
                raw = {name: z[name].copy() for name in z.files}
            validate_recording(raw, frames=300, nbody=world.nbody)
            if score({k: v[20:] for k, v in raw.items()}) != row["selected"]["score"]:
                raise ValueError("actual teacher raw outcome differs")
            teacher = bool(
                row["selected"]["score"]["qualified"]
                and prefix_d["prefix_physical_safe"]
                and prefix_d["post_boundary_control_can_cause_first_touch"]
            )
            actor = {v["variant"]: v["causal_qualified"] for v in old["variants"]}
            result.append(
                dict(
                    course_index=i,
                    stratum=row["course"]["stratum"],
                    teacher_qualified=teacher,
                    selected_option=row["selected"]["selected_index"],
                    actor_outcomes=actor,
                    nine_future_passes=sum(c["score"]["qualified"] for c in row["candidates"]),
                )
            )
    if {r["course_index"] for r in result} != set(teacher_coverage_indices()) or len(result) != 24:
        raise ValueError("exact fixed24 coverage required")
    return dict(
        contexts=24,
        rows=sorted(result, key=lambda r: r["course_index"]),
        teacher_success=sum(r["teacher_qualified"] for r in result),
        actor_success={
            name: sum(r["actor_outcomes"][name] for r in result)
            for name in ("anchor", "seed17", "seed29", "seed43")
        },
        by_stratum={
            s: dict(
                cases=sum(r["stratum"] == s for r in result),
                teacher_success=sum(r["teacher_qualified"] for r in result if r["stratum"] == s),
                actor_success={
                    name: sum(r["actor_outcomes"][name] for r in result if r["stratum"] == s)
                    for name in ("anchor", "seed17", "seed29", "seed43")
                },
            )
            for s in sorted({r["stratum"] for r in result})
        },
        teacher_success_missed_by_all_actors=[
            r["course_index"]
            for r in result
            if r["teacher_qualified"] and not any(r["actor_outcomes"].values())
        ],
        privileged_option_oracle_not_global_reachability=True,
        not_actor_own_state_recovery_test=True,
        no_training_or_promotion_authorized=True,
    )


if __name__ == "__main__":
    main()
