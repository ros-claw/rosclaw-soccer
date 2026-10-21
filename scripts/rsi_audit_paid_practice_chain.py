"""Independent native continuity AND causal neural audit for practice chains."""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    p = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "parent-root", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--hash", required=True)
    p.add_argument("--parent-hash", required=True)
    p.add_argument("--expected-head", required=True)
    a = p.parse_args()
    rp, dp, pp = (
        a.root / "complete.json",
        a.root / "declaration.json",
        a.parent_root / "complete.json",
    )
    if (
        a.output.exists()
        or digest(rp) != a.hash
        or digest(pp) != a.parent_hash
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
    ):
        raise ValueError(
            "exact closed chain, qualified parent and clean frozen audit source required"
        )
    r, d, parent = [json.loads(f.read_text()) for f in (rp, dp, pp)]
    if (
        r["status"] != "PAID_CONTINUING_CHAIN_CPU_EXAM_CLOSED_PENDING_INDEPENDENT_AUDIT"
        or r["skills"] != d["skills"]
        or r["skills"] not in (2, 3, 4)
        or len(d["cases"]) != 16
        or d["body_resets_between_skills"] != 0
        or d["input_file_hashes"].get(str(pp)) != a.parent_hash
        or set(r["results"]) != {"baseline", "learned"}
    ):
        raise ValueError("complete declared body-preserving practice protocol required")
    parents = [c for c in parent["candidates"] if c["checkpoint_hash"] == d["checkpoint_hash"]]
    if len(parents) != 1 or not parents[0]["full_continuous_skill_gate"]:
        raise ValueError("exact actually qualified learned parent required")
    candidate = parents[0]
    from rosclaw_soccer.rsi.paid_chain_courses import chain_parent_profile

    parent_declaration = a.parent_root / "declaration.json"
    if d["input_file_hashes"].get(str(parent_declaration)) != digest(parent_declaration):
        raise ValueError("exact parent course declaration required")
    course_bank = d.get("course_bank", "train")
    cases, parent_prefixes = chain_parent_profile(
        json.loads(parent_declaration.read_text()), parent, candidate, course_bank
    )
    prefix_count = 2 if course_bank == "train" else 1
    if (
        cases != d["cases"]
        or r.get("course_bank", "train") != course_bank
        or d.get("qualified_prefix_skills", 2) != prefix_count
        or r.get("qualified_prefix_skills", 2) != prefix_count
        or d["first_two_arrays_must_match_parent"] is not (course_bank == "train")
        or r["all_first_two_arrays_exact_parent"] is not (course_bank == "train")
    ):
        raise ValueError("course bank or qualified prefix coverage changed")
    pins = dict(r["input_file_hashes"])
    pins.update({str(rp): a.hash, str(dp): digest(dp), str(pp): a.parent_hash})
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            input_file_hashes=pins,
            source_head=a.expected_head,
            own_native_steps=0,
            own_neural_recompute=True,
            hardware_authorized=False,
        ),
    )
    counts = dict(native=0, neural=0, raw_episodes=0, holds=0, boundaries=0)
    try:
        import numpy as np
        import torch

        from rosclaw_soccer.rsi.paid_contact_residual_actor import load_contact_residual_actor
        from rosclaw_soccer.rsi.paid_continuity_audit import (
            audit_practice_boundary,
            audit_terminal,
            identical,
            snapshot_fields,
        )
        from rosclaw_soccer.rsi.paid_research_setup import open_paid_auditor

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        auditor, motions = open_paid_auditor(pins, d["cases"])

        def unique(h):
            paths = [Path(p) for p, ph in pins.items() if ph == h]
            if len(paths) != 1:
                raise ValueError("one exact source-bound learned actor required")
            return paths[0]

        policy = next(Path(p) for p in pins if p.endswith("/ckp/policy_30000.onnx"))
        scene = next(
            h for p, h in pins.items() if p.endswith("/cached-standard/declared-scene.xml")
        )
        actor, _ = load_contact_residual_actor(
            policy,
            checkpoint=unique(d["checkpoint_hash"]),
            checkpoint_hash=d["checkpoint_hash"],
            training_receipt=unique(d["training_receipt_hash"]),
            training_receipt_hash=d["training_receipt_hash"],
            scene_hash=scene,
        )

        def read(row, root=a.root):
            path = root / row["raw_path"]
            if (
                path.is_symlink()
                or not path.resolve().is_relative_to(root.resolve())
                or digest(path) != row["raw_hash"]
            ):
                raise ValueError("confined exact chain trajectory required")
            pins[str(path)] = row["raw_hash"]
            with np.load(path, allow_pickle=False) as z:
                return {k: z[k].copy() for k in z.files}

        summaries = {}
        for arm in ("baseline", "learned"):
            worlds = r["results"][arm]["worlds"]
            if len(worlds) != 16 or [w["case"] for w in worlds] != list(range(16)):
                raise ValueError("no case may be dropped, replaced or reordered")
            qualities = []
            for i, (world, case) in enumerate(zip(worlds, d["cases"], strict=True)):
                for artifacts in (world["artifacts"], world["repeat_artifacts"]):
                    data = {Path(x["raw_path"]).stem: x for x in artifacts}
                    if len(data) != len(artifacts):
                        raise ValueError("unique chain artifact names required")
                    used = set()

                    def block(key, used=used, data=data):
                        used.add(key)
                        return read(data[key])

                    start_native = counts["native"]
                    previous = np.zeros(29, np.float32)
                    entry = None
                    prev_raw = terminal = None
                    first_goal = None
                    checked_quality = []
                    status = "ALL_SKILLS_EXECUTED"
                    motion = motions[case["source"]]

                    def boundary(
                        index, previous, terminal, world=world, data=data, block=block, case=case
                    ):
                        prefix = f"between{index - 1}-{index}"
                        record = world["boundaries"][index - 1]
                        if record["status"] == "FIXTURE_REJECTED":
                            if any(k.startswith(prefix) for k in data):
                                raise ValueError("rejected fixture cannot mutate state")
                            return dict(status="FIXTURE_REJECTED", native_steps=0, state=None)
                        checked = audit_practice_boundary(
                            auditor,
                            previous=previous,
                            terminal=terminal,
                            drop=block(prefix + "-drop"),
                            hold=block(prefix + "-hold") if prefix + "-hold" in data else None,
                            boundary=block(prefix + "-boundary")
                            if prefix + "-boundary" in data
                            else None,
                            source=case["source"],
                        )
                        if checked["status"] != record["status"]:
                            raise ValueError("declared physical boundary status differs")
                        counts["native"] += checked["native_steps"]
                        counts["holds"] += int(prefix + "-hold" in data)
                        counts["boundaries"] += int(checked["status"] == "READY")
                        return checked

                    for index, skill in enumerate(world["skills"]):
                        if index:
                            b = boundary(index, prev_raw, terminal)
                            if b["status"] != "READY":
                                raise ValueError("failed boundary may not execute another skill")
                            entry = b["state"]
                        row = dict(skill["result"], **data[f"skill{index}"])
                        z = block(f"skill{index}")
                        if skill["index"] != index or row["source"] != case["source"]:
                            raise ValueError("actual skill/source sequence differs")
                        if index == 0:
                            if z["initial_time"] != 0:
                                raise ValueError("world must start once at time0")
                        else:
                            initial = snapshot_fields(z, "initial_")
                            identical(initial, {k: entry[k] for k in initial})
                            if first_goal is None or not np.array_equal(
                                z["initial_mocap_pos"], first_goal
                            ):
                                raise ValueError("world goal changed during chain")
                        if index < prefix_count:
                            identical(
                                z,
                                read(
                                    parent_prefixes[arm][i][index],
                                    a.parent_root,
                                ),
                            )
                        checked = auditor.audit(
                            z, row, horizon=len(motion["q"]), previous_action=previous
                        )
                        n = len(z["NN_action"])
                        if (
                            z["NN_obs"][0, :58].any()
                            or z["NN_obs"][0, 61:64].any()
                            or not np.array_equal(z["NN_obs"][1:, :29], motion["q"][: n - 1])
                            or not np.array_equal(z["NN_obs"][1:, 29:58], motion["v"][: n - 1])
                            or not np.array_equal(z["NN_obs"][1:, 61:64], motion["w"][: n - 1])
                        ):
                            raise ValueError("reference clock/cache was not explicitly reset")
                        model = actor if arm == "learned" and index else actor.base
                        state = (torch.zeros(2, 1, 128), torch.zeros(2, 1, 128))
                        with torch.no_grad():
                            for j, obs in enumerate(z["NN_obs"]):
                                if any(
                                    not np.array_equal(t.numpy(), z[k][j])
                                    for t, k in zip(
                                        state, ("NN_h_before", "NN_c_before"), strict=True
                                    )
                                ):
                                    raise ValueError("private causal neural h/c differs")
                                mean, state = model(torch.from_numpy(obs).reshape(1, 1, 160), state)
                                if not np.array_equal(mean[0, 0].numpy(), z["NN_action"][j]) or any(
                                    not np.array_equal(t.numpy(), z[k][j])
                                    for t, k in zip(
                                        state, ("NN_h_after", "NN_c_after"), strict=True
                                    )
                                ):
                                    raise ValueError(
                                        "recorded motor output differs from qualified own network"
                                    )
                        counts["native"] += checked["native_steps"]
                        counts["neural"] += checked["neural_rows"]
                        counts["raw_episodes"] += 1
                        checked_quality.append(checked["quality"])
                        terminal = block(f"terminal{index}")
                        audit_terminal(auditor, z, terminal)
                        previous = z["NN_action"][-1]
                        prev_raw = z
                        if index == 0:
                            first_goal = z["initial_mocap_pos"].copy()
                        if checked["body_failed"]:
                            status = "SKILL_BODY_FAILURE"
                            if index != len(world["skills"]) - 1:
                                raise ValueError("body failure cannot silently restart")
                    if status != "SKILL_BODY_FAILURE" and len(world["skills"]) < r["skills"]:
                        b = boundary(len(world["skills"]), prev_raw, terminal)
                        if b["status"] == "READY":
                            raise ValueError("healthy requested skill was silently omitted")
                        status = b["status"]
                    if (
                        used != set(data)
                        or checked_quality != world["quality_by_skill"]
                        or status != world["status"]
                        or world["world_stable"] != (status == "ALL_SKILLS_EXECUTED")
                        or world["body_resets_between_skills"] != 0
                        or world["robot_initializations"] != 1
                        or counts["native"] - start_native != world["native_steps"]
                        or not np.isclose(
                            world["final_sim_time"],
                            world["native_steps"] * 0.002,
                            rtol=0,
                            atol=1e-10,
                        )
                    ):
                        raise ValueError(
                            "chain native work, quality, clock, state, status or coverage differs"
                        )
                for x, y in zip(world["artifacts"], world["repeat_artifacts"], strict=True):
                    identical(read(x), read(y))
                qualities.append(world["quality_by_skill"])
            summary = dict(
                stable=sum(w["world_stable"] for w in worlds),
                quality_by_skill=[
                    sum(len(q) > k and q[k] for q in qualities) for k in range(r["skills"])
                ],
            )
            if any(r["results"][arm][k] != v for k, v in summary.items()):
                raise ValueError("aggregate chain statistics differ")
            summaries[arm] = summary
        if (
            counts["native"] != r["actual_native_steps"]
            or counts["neural"] != r["actual_neural_rows"]
        ):
            raise ValueError("complete producer native/neural work totals differ")
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="PAID_CONTINUING_CHAIN_NATIVE_AND_NEURAL_AUDIT_PASS",
                input_file_hashes=pins,
                summaries=summaries,
                course_bank=course_bank,
                qualified_prefix_skills=prefix_count,
                counts=counts,
                producer_native_steps=counts["native"],
                own_native_steps=0,
                own_neural_rows=counts["neural"],
                not_additional_independent_physics=True,
                goal_complete=False,
                team_game=False,
                fresh_exam=False,
                hardware_authorized=False,
                promotion_authorized=False,
                commercial_publicity_authorized=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failed.json", dict(error=repr(error), counts=counts, own_native_steps=0)
        )
        raise


if __name__ == "__main__":
    main()
