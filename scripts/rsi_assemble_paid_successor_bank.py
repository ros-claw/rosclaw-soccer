"""Assemble audited whole successful trajectories for offline self-distillation.

This training oracle may select complete historical experiences. It is NOT a
deployable case-ID router or a claim that one policy succeeds on all courses.
Both skills must come from the SAME actual world, with its physical handoff.
"""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_search_paid_foot_contact import contact_rank

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--shard",
        nargs=4,
        action="append",
        required=True,
        metavar=("ROOT", "HASH", "AUDIT", "AUDIT_HASH"),
    )
    p.add_argument(
        "--history",
        nargs=4,
        action="append",
        default=[],
        metavar=("ROOT", "HASH", "AUDIT", "AUDIT_HASH"),
    )
    p.add_argument("--bootstrap-root", type=Path, required=True)
    p.add_argument("--base-root", type=Path, required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if (
        a.output.exists()
        or len(a.shard) != 8
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
    ):
        raise ValueError("unused output, eight complete shards and frozen assembly source required")
    pins = {}

    def pin(path, h=None):
        key = str(path)
        h = digest(path) if h is None else h
        if key in pins and pins[key] != h:
            raise ValueError("conflicting lineage identity: " + key)
        pins[key] = h

    def pair(item, status):
        root, h, aroot, ah = item
        root, aroot = Path(root), Path(aroot)
        rp, ap = root / "complete.json", aroot / "complete.json"
        if digest(rp) != h or digest(ap) != ah:
            raise ValueError("exact completed source/audit required")
        r, u = json.loads(rp.read_text()), json.loads(ap.read_text())
        if r["status"] != status or u["input_file_hashes"].get(str(rp)) != h:
            raise ValueError("audit must bind its actual producer")
        for path, ph in u["input_file_hashes"].items():
            pin(Path(path), ph)
        pin(rp, h)
        pin(ap, ah)
        dp = root / "declaration.json"
        pin(dp)
        return root, r, u, json.loads(dp.read_text())

    shards = {}
    for item in a.shard:
        root, r, u, d = pair(item, "PAID_LIVE_BODY_CONTACT_TEACHER_RESEARCH_NOT_PROMOTED")
        ids = d.get("search_source_indices", [])
        if (
            len(ids) != 1
            or ids[0] in shards
            or ids[0] not in range(8)
            or u["status"] != "PAID_LIVE_CONTACT_PRIMITIVE_CEM_AUDIT_PASS"
            or u["raw_episode_count"] != 112
            or d["final_case_indices"] != [2 * ids[0], 2 * ids[0] + 1]
            or r["first_kick_reference"] != "learned_safe_successor"
        ):
            raise ValueError("all eight unique audited learned-successor source shards required")
        shards[ids[0]] = (root, r, u, d)
    if set(shards) != set(range(8)):
        raise ValueError("incomplete source coverage")
    root0, r0, _, d0 = shards[0]
    cases = d0["TRAIN_cases"]
    for _root, r, _u, d in shards.values():
        if (
            d["TRAIN_cases"] != cases
            or d["bootstrap_first_references"] != d0["bootstrap_first_references"]
            or d["base_model_checkpoint_hash"] != d0["base_model_checkpoint_hash"]
            or d["bootstrap_training_receipt_hash"] != d0["bootstrap_training_receipt_hash"]
            or [x["raw_hash"] for x in r["baseline"]] != [x["raw_hash"] for x in r0["baseline"]]
        ):
            raise ValueError("same compiled body, learned prefix and exact zero bank required")
    bp = a.base_root / "complete.json"
    bcp = a.base_root / "live-distilled-epoch256.pt"
    if (
        digest(bp) != d0["bootstrap_training_receipt_hash"]
        or digest(bcp) != d0["base_model_checkpoint_hash"]
    ):
        raise ValueError("qualified learned initialization lineage differs")
    bootstrap_path = a.bootstrap_root / "complete.json"
    if pins.get(str(bootstrap_path)) != digest(bootstrap_path):
        raise ValueError("qualified initialization exam must bind all parent trajectories")
    bootstrap = json.loads(bootstrap_path.read_text())
    baseline_worlds = next(
        c["continuous"]["results"]
        for c in bootstrap["candidates"]
        if c["training_receipt_hash"] == digest(bp)
    )
    histories = []
    scene_hashes = {
        h for path, h in pins.items() if path.endswith("/cached-standard/declared-scene.xml")
    }
    if len(scene_hashes) != 1:
        raise ValueError("one compiled standard-ball scene identity required")
    for item in a.history:
        root, r, u, d = pair(item, "PAID_LIVE_DISTILLED_CPU_FULL_WORLD_EXAM_CLOSED")
        if (
            u["status"] != "PAID_LIVE_DISTILLED_ALL_PRIMITIVE_AUDIT_PASS"
            or d["old_cases"] != cases
            or r["team_game"]
            or {
                h
                for path, h in r["input_file_hashes"].items()
                if path.endswith("/cached-standard/declared-scene.xml")
            }
            != scene_hashes
        ):
            raise ValueError("history must use the exact same audited practice courses")
        for candidate in r["candidates"]:
            histories.append((root, candidate["candidate"], candidate["continuous"]["results"]))
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        for f in folder.rglob("*.py"):
            pin(f)
    check_pins(pins)

    def quality(row):
        return contact_rank([row, row])[1] > 0

    def row_at(root, row):
        path = root / row["raw_path"]
        if not path.resolve().is_relative_to(root.resolve()) or path.is_symlink():
            raise ValueError("ordinary confined source trajectory required")
        pin(path, row["raw_hash"])
        return dict(row, path=str(path))

    courses = []
    for i in range(16):
        root, r, u, d = shards[i // 2]
        proposed = r["final"][i % 2]
        base = r0["baseline"][i]
        first = second = None
        origin = None
        if quality(base):
            w = baseline_worlds[i]
            if not (w["first_quality"] and w["second_quality"] and w["world_stable"]):
                raise ValueError("baseline retention must come from an actual complete success")
            first, second = (
                row_at(a.bootstrap_root, w["first"]),
                row_at(a.bootstrap_root, w["second"]),
            )
            origin = "retain_qualified_learned_world"
        elif quality(proposed):
            fp = root / f"final{i}-repeat0/complete.json"
            if str(fp) not in pins:
                raise ValueError("whole teacher-world record must be sealed")
            record = json.loads(fp.read_text())
            first, second = row_at(root, record["first"]), row_at(root, proposed)
            origin = "new_composite_contact_teacher_world"
        else:
            for hroot, model_id, worlds in histories:
                w = worlds[i]
                if w["first_quality"] and w["second_quality"] and w["world_stable"]:
                    first, second = row_at(hroot, w["first"]), row_at(hroot, w["second"])
                    origin = "historical_complete_success_model" + str(model_id)
                    break
        if first is None or second is None or not quality(first) or not quality(second):
            raise ValueError(f"no complete successful experience for course{i}; never drop it")
        courses.append(
            dict(case=i, source=cases[i]["source"], first=first, second=second, origin=origin)
        )
    check_pins(pins)
    import numpy as np

    for c in courses:
        with (
            np.load(c["first"]["path"], allow_pickle=False) as f,
            np.load(c["second"]["path"], allow_pickle=False) as s,
        ):
            if (
                f["NN_obs"].shape[1] != 160
                or s["NN_obs"].shape[1] != 160
                or not np.array_equal(f["NN_action"][-1], s["NN_obs"][0, 125:154])
                or f["initial_time"] != 0
                or s["initial_time"] <= f["time_before"][-1]
            ):
                raise ValueError("entire physical-world time and executed-action handoff required")
            c["neural_observation_gap_s"] = float(
                s["time_before"][0] - f["time_before"][len(f["NN_action"]) - 1]
            )
    a.output.mkdir()
    bank = dict(
        schema="soccer.audited_successor_sequence_bank.v1",
        source_head=a.expected_head,
        input_file_hashes=pins,
        cases=cases,
        courses=courses,
        base_root=str(a.base_root),
        base_checkpoint_hash=d0["base_model_checkpoint_hash"],
        bootstrap_exam_root=str(a.bootstrap_root),
        complete_successful_course_coverage=16,
        first_neural_rows=sum(c["first"]["neural_rows"] for c in courses),
        second_neural_rows=sum(c["second"]["neural_rows"] for c in courses),
        oracle_training_data_only=True,
        single_policy_16of16_proven=False,
        runtime_case_lookup_forbidden=True,
        new_native_steps=0,
        new_neural_rows=0,
        training_updates=0,
        lineage_CEM_native_steps=sum(s[1]["actual_native_steps"] for s in shards.values()),
        lineage_CEM_neural_rows=sum(s[1]["actual_neural_rows"] for s in shards.values()),
        historical_physics_not_recounted=True,
        body_reset_between_skills=False,
        goal_complete=False,
        team_game=False,
        fresh_exam=False,
        hardware_authorized=False,
        promotion_authorized=False,
        commercial_publicity_authorized=False,
    )
    write_once(a.output / "complete.json", bank)
    print(
        json.dumps(
            {k: v for k, v in bank.items() if k not in ("input_file_hashes", "cases", "courses")}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
