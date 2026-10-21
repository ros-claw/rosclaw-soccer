"""Read-only independent primitive/continuity audit of full learned worlds.

All original and repeat raw blocks are inspected. Own physics/NN work is zero;
the producer's independent full-world repeats are not recounted as audit work.
"""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--hash", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    rp, dp = a.root / "complete.json", a.root / "declaration.json"
    if (
        a.output.exists()
        or digest(rp) != a.hash
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
    ):
        raise ValueError("exact closed exam/frozen source and unused output required")
    r, d = json.loads(rp.read_text()), json.loads(dp.read_text())
    indices = d.get("exam_candidate_indices", list(range(4)))
    sequences = d.get("sequence_protocol", False)
    specialist = d.get("successor_specialist", False)
    zero_residual = d.get("contact_residual_zero", False)
    if (
        r["status"] != "PAID_LIVE_DISTILLED_CPU_FULL_WORLD_EXAM_CLOSED"
        or [x["candidate"] for x in r["candidates"]] != indices
        or not indices
        or len(set(indices)) != len(indices)
        or any(i not in range(4) for i in indices)
        or r.get("sequence_protocol", False) != sequences
        or (specialist and not sequences)
        or (zero_residual and not specialist)
        or r.get("contact_residual_zero", False) != zero_residual
        or r.get("successor_specialist", False) != specialist
        or (
            sequences
            and d.get("entry_controller") != ("qualified-prior" if specialist else "candidate")
        )
    ):
        raise ValueError("complete declared CPU candidate set required")
    pins = {**r["input_file_hashes"], str(rp): a.hash, str(dp): digest(dp)}
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            input_file_hashes=pins, own_native_steps=0, own_neural_rows=0, hardware_authorized=False
        ),
    )
    import ast

    import mujoco
    import numpy as np
    import onnx

    from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
    from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
    from rosclaw_soccer.rsi.paid_primitive_audit import PaidPrimitiveAuditor
    from rosclaw_soccer.rsi.paid_teacher_policy import strict_parameters

    policy = next(Path(f) for f in pins if f.endswith("/ckp/policy_30000.onnx"))
    xml = next(Path(f) for f in pins if f.endswith("/cached-standard/declared-scene.xml"))
    graph = onnx.load(policy, load_external_data=False)
    names, params = strict_parameters({x.key: x.value for x in graph.metadata_props})
    order = np.array([names.index(n) for n in G1_DDS_JOINT_NAMES])
    effort = np.array(
        [
            139
            if ("hip_roll" in n or "knee" in n)
            else 88
            if ("hip_pitch" in n or "hip_yaw" in n or n == "waist_yaw_joint")
            else 50
            if ("ankle" in n or n in ("waist_roll_joint", "waist_pitch_joint"))
            else 5
            if ("wrist_pitch" in n or "wrist_yaw" in n)
            else 25
            for n in G1_DDS_JOINT_NAMES
        ]
    )
    model = mujoco.MjModel.from_xml_path(str(xml))
    layout = NativeVelocityLayout(model, robot_prefix="", ball_body_name="soccer_ball")
    goal_id = int(model.body_mocapid[model.body("soccer_goal_marker").id])
    auditor = PaidPrimitiveAuditor(
        layout=layout, parameters=params, order=order, effort=effort, goal_mocap_id=goal_id
    )
    const = policy.parent.parent / "exp/mujoco_soccer/constants.py"
    if str(const) not in pins:
        raise ValueError("exact source body ordering required")
    body_names = next(
        ast.literal_eval(n.value)
        for n in ast.parse(const.read_text()).body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "BODY_NAMES" for t in n.targets)
    )
    torso = body_names.index("torso_link")
    motions = {}
    for name in dict.fromkeys(c["source"] for c in d["old_cases"]):
        path = policy.parent.parent / "motions/soccer-standard" / name
        if str(path) not in pins:
            raise ValueError("motion reference must be pinned")
        with np.load(path, allow_pickle=False) as z:
            motions[name] = dict(
                q=z["joint_pos"].copy(),
                v=z["joint_vel"].copy(),
                w=z["body_ang_vel_w"][:, torso].copy(),
            )
    native = neural = episode_count = hold_count = boundary_count = 0

    def read(artifact):
        path = a.root / artifact["raw_path"]
        if (
            path.is_symlink()
            or not path.resolve().is_relative_to(a.root.resolve())
            or digest(path) != artifact["raw_hash"]
        ):
            raise ValueError("ordinary confined sealed raw required")
        pins[str(path)] = artifact["raw_hash"]
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k].copy() for k in z.files}

    def same(x, y):
        if set(x) != set(y) or any(not np.array_equal(x[k], y[k]) for k in x):
            raise ValueError("independent raw/native state continuity differs")

    def snap(raw, prefix):
        return {k.removeprefix(prefix): v for k, v in raw.items() if k.startswith(prefix)}

    def source_raw(root, row):
        path = root / row["raw_path"]
        if (
            path.is_symlink()
            or not path.resolve().is_relative_to(root.resolve())
            or digest(path) != row["raw_hash"]
        ):
            raise ValueError("actual qualified baseline trajectory hash differs")
        if str(path) in pins and pins[str(path)] != row["raw_hash"]:
            raise ValueError("conflicting qualified baseline identity")
        pins[str(path)] = row["raw_hash"]
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k].copy() for k in z.files}

    def bind_sequence_baseline():
        if not sequences:
            return
        training = Path(d["candidates"][0]) / "declaration.json"
        if pins.get(str(training)) != digest(training):
            raise ValueError("actual sequence training declaration must be sealed")
        bank_hash = json.loads(training.read_text())["bank_hash"]
        bank_paths = [Path(p) for p, h in pins.items() if h == bank_hash]
        if len(bank_paths) != 1 or digest(bank_paths[0]) != bank_hash:
            raise ValueError("one exact completed paired-world bank required")
        bank = json.loads(bank_paths[0].read_text())
        root = Path(bank["bootstrap_exam_root"])
        rp = root / "complete.json"
        if pins.get(str(rp)) != digest(rp):
            raise ValueError("qualified initialization exam must bind retention baselines")
        report = json.loads(rp.read_text())
        matches = [
            c for c in report["candidates"] if c["checkpoint_hash"] == bank["base_checkpoint_hash"]
        ]
        if len(matches) != 1:
            raise ValueError("one exact initial-skill and successor-body baseline required")
        baseline = matches[0]
        for key in ("old16", "dev16"):
            rows = baseline["isolated"][key]["results"]
            if len(rows) != 16:
                raise ValueError("complete stronger historical retention cohort required")
            for before, after in zip(rows, r["baseline_isolated"][key], strict=True):
                same(source_raw(root, before), read(after))
        for before, after in zip(
            baseline["continuous"]["results"], r["baseline_continuous"], strict=True
        ):
            for key in ("first", "second"):
                same(source_raw(root, before[key]), read(after[key]))

    def audit_row(row, case, previous, artifact=None, initial_memory=None):
        nonlocal native, neural, episode_count
        z = read(row if artifact is None else artifact)
        motion = motions[case["source"]]
        checked = auditor.audit(
            z,
            row,
            horizon=len(motion["q"]),
            previous_action=previous,
            initial_memory=initial_memory,
        )
        n = len(z["NN_action"])
        if (
            row["source"] != case["source"]
            or z["NN_obs"][0, :58].any()
            or z["NN_obs"][0, 61:64].any()
            or not np.array_equal(z["NN_obs"][1:, :29], motion["q"][: n - 1])
            or not np.array_equal(z["NN_obs"][1:, 29:58], motion["v"][: n - 1])
            or not np.array_equal(z["NN_obs"][1:, 61:64], motion["w"][: n - 1])
        ):
            raise ValueError("causal reference reset/cache or source identity differs")
        native += checked["native_steps"]
        neural += checked["neural_rows"]
        episode_count += 1
        return z, checked

    def terminal(raw, state):
        q, v = auditor.canonical(state["qpos"], state["qvel"])
        if (
            not np.array_equal(q, raw["qpos"][-1, -1])
            or not np.array_equal(v, raw["qvel"][-1, -1])
            or not np.array_equal(state["ctrl"][layout.ai], raw["torque"][-1, -1])
            or not np.isclose(state["time"], raw["time_before"][-1] + 0.02, rtol=0, atol=1e-11)
        ):
            raise ValueError("actual physical terminal state/time differs")

    def isolated(rows, bank, repeats, prior_rows=None):
        if len(rows) != 16 or len(bank) != 16:
            raise ValueError("complete16cases required")
        checks = []
        for i, (row, case) in enumerate(zip(rows, bank, strict=True)):
            z, checked = audit_row(row, case, np.zeros(29, np.float32))
            if prior_rows is not None:
                same(z, read(prior_rows[i]))
            checks.append(checked)
            if repeats:
                other = dict(
                    row,
                    raw_path=row["raw_path"].replace("-repeat0.npz", "-repeat1.npz"),
                    raw_hash=row["independent_repeat_raw_hash"],
                )
                second, _ = audit_row(other, case, np.zeros(29, np.float32))
                same(z, second)
        return checks

    def world(row, case, artifacts, memory_handoff="reset"):
        nonlocal native, hold_count, boundary_count
        start_native = native
        if (
            memory_handoff not in ("carry", "reset")
            or row.get("memory_handoff", "reset") != memory_handoff
            or (not sequences and memory_handoff != "reset")
        ):
            raise ValueError("only explicitly trained single-actor memory handoff may carry")
        data = {Path(x["raw_path"]).name: x for x in artifacts}
        if len(data) != len(artifacts):
            raise ValueError("unique named world artifacts required")
        first, fc = audit_row(row["first"], case, np.zeros(29, np.float32), data["first.npz"])
        entry = read(data["terminal-first.npz"])
        terminal(first, entry)
        second_quality = False
        stable = False
        if fc["body_failed"]:
            if row["status"] != "FIRST_BODY_FAILURE" or row["second"] is not None:
                raise ValueError("failed first kick cannot be silently reset")
        elif row["status"] == "FIXTURE_REJECTED":
            if row["second"] is not None or "drop.npz" in data:
                raise ValueError("rejected fixture may not execute another skill")
        else:
            drop = read(data["drop.npz"])
            before = snap(drop, "before_")
            same(before, entry)
            q, _ = auditor.canonical(before["qpos"], before["qvel"])
            performed = bool(drop["performed"])
            if performed != bool(np.linalg.norm(q[36:38] - q[:2]) >= 2):
                raise ValueError("ball return eligibility differs")
            if performed:
                after = snap(drop, "after_")
                modified = {k: v.copy() for k, v in after.items()}
                modified["qpos"][layout.bq : layout.bq + 7] = before["qpos"][
                    layout.bq : layout.bq + 7
                ]
                modified["qvel"][layout.bv : layout.bv + 6] = before["qvel"][
                    layout.bv : layout.bv + 6
                ]
                same(before, modified)
                direction = before["mocap_pos"][goal_id, :2] - q[:2]
                direction /= np.linalg.norm(direction)
                sign = 1 if "left" in case["source"] else -1
                point = (
                    q[:2] + 0.8 * direction + sign * 0.15 * np.array([-direction[1], direction[0]])
                )
                if (
                    not np.array_equal(
                        after["qpos"][layout.bq : layout.bq + 7], np.r_[point, 0.4, 1.0, 0, 0, 0]
                    )
                    or after["qvel"][layout.bv : layout.bv + 6].any()
                ):
                    raise ValueError("declared ball release geometry or velocity differs")
                entry = after
                hold = read(data["hold.npz"])
                hc = auditor.audit_hold(hold, entry, first["target"][-1])
                native += hc["native_steps"]
                hold_count += 1
                if hc["body_failed"]:
                    if row["status"] != "HOLD_BODY_FAILURE" or row["second"] is not None:
                        raise ValueError("hold failure cannot restart skill")
                elif not hc["settled"] and (
                    row["status"] != "BALL_NOT_SETTLED" or row["second"] is not None
                ):
                    raise ValueError("unsettled ball cannot be silently corrected")
            if (not performed or (not hc["body_failed"] and hc["settled"])) and row[
                "second"
            ] is None:
                raise ValueError("healthy complete world must record the actual second skill")
            if row["second"] is not None:
                boundary = read(data["boundary.npz"])
                before, after = snap(boundary, "before_"), snap(boundary, "after_")
                same(before, after)
                if performed:
                    terminal(hold, before)
                else:
                    same(entry, before)
                previous = first["NN_action"][-1]
                if not np.array_equal(previous, boundary["previous_actual_action"]):
                    raise ValueError("actual first kick action history was erased")
                memory = (
                    tuple(first[k][-1] for k in ("NN_h_after", "NN_c_after"))
                    if memory_handoff == "carry"
                    else None
                )
                second, sc = audit_row(
                    row["second"], case, previous, data["second.npz"], initial_memory=memory
                )
                same(snap(second, "initial_"), {k: after[k] for k in snap(second, "initial_")})
                terminal(second, read(data["terminal-second.npz"]))
                if not np.array_equal(first["initial_mocap_pos"], second["initial_mocap_pos"]):
                    raise ValueError("world target moved between skills")
                expected = "SECOND_BODY_FAILURE" if sc["body_failed"] else "TWO_SKILLS_EXECUTED"
                if expected != row["status"]:
                    raise ValueError("world physical terminal status differs")
                second_quality = sc["quality"]
                stable = not sc["body_failed"]
                boundary_count += 1
        if (
            fc["quality"] != row["first_quality"]
            or second_quality != row["second_quality"]
            or stable != row["world_stable"]
            or row["robot_resets"] != 1
            or native - start_native != row["native_steps"]
            or not np.isclose(
                row["final_sim_time"],
                first["initial_time"] + row["native_steps"] * 0.002,
                rtol=0,
                atol=1e-10,
            )
        ):
            raise ValueError("whole world physical quality or stability differs")
        return dict(quality=second_quality, stable=stable)

    try:
        bind_sequence_baseline()
        baseline = {
            key: isolated(r["baseline_isolated"][key], bank, False)
            for key, bank in (("old16", d["old_cases"]), ("dev16", d["DEV_cases"]))
        }
        continuous = [
            world(row, case, row["artifacts"])
            for row, case in zip(r["baseline_continuous"], d["old_cases"], strict=True)
        ]
        summaries = []
        for candidate in r["candidates"]:
            if candidate.get("contact_residual_zero", False) != zero_residual or (
                zero_residual and candidate["checkpoint_hash"] != d["baseline_checkpoint_hash"]
            ):
                raise ValueError("zero residual must declare its actual unchanged base motor")
            memory_handoff = "reset"
            if sequences:
                memory_handoff = d["candidate_memory_handoff"][candidate["candidate"]]
                expected = "carry" if candidate["candidate"] < 3 and not specialist else "reset"
                if candidate.get("memory_handoff") != memory_handoff or memory_handoff != expected:
                    raise ValueError(
                        "exact predeclared sequence seed/ablation memory mode required"
                    )
                training = Path(d["candidates"][candidate["candidate"]]) / "complete.json"
                if (
                    pins.get(str(training)) != candidate["training_receipt_hash"]
                    or digest(training) != candidate["training_receipt_hash"]
                    or json.loads(training.read_text())["config"]["memory_handoff"]
                    != memory_handoff
                    or json.loads(training.read_text())["config"]["protocol"]
                    != (
                        "protected-successor-specialist-v1"
                        if specialist
                        else "complete-successor-sequences-v1"
                    )
                ):
                    raise ValueError("memory handoff must bind actual completed training")
            contextual = d.get("entry_controller", "candidate") == "qualified-prior"
            if candidate.get("entry_controller", "candidate") != d.get(
                "entry_controller", "candidate"
            ):
                raise ValueError("declared motor-context routing differs")
            iso = {}
            for key, bank in (("old16", d["old_cases"]), ("dev16", d["DEV_cases"])):
                section = candidate["isolated"][key]
                checks = isolated(
                    section["results"],
                    bank,
                    True,
                    r["baseline_isolated"][key] if contextual else None,
                )
                iso[key] = dict(
                    quality=sum(x["quality"] for x in checks),
                    stable=sum(not x["body_failed"] for x in checks),
                    regressions=[
                        i
                        for i, (old, new) in enumerate(zip(baseline[key], checks, strict=True))
                        if old["quality"] and not new["quality"]
                    ],
                )
                if any(section[k] != v for k, v in iso[key].items()):
                    raise ValueError("isolated statistics differ")
            section = candidate["continuous"]
            checked = []
            for i, (row, case) in enumerate(zip(section["results"], d["old_cases"], strict=True)):
                if contextual:
                    actual = next(
                        x for x in row["artifacts"] if Path(x["raw_path"]).name == "first.npz"
                    )
                    expected = next(
                        x
                        for x in r["baseline_continuous"][i]["artifacts"]
                        if Path(x["raw_path"]).name == "first.npz"
                    )
                    same(read(actual), read(expected))
                if zero_residual and row["second"] is not None:
                    same(read(row["second"]), read(r["baseline_continuous"][i]["second"]))
                checked.append(world(row, case, row["artifacts"], memory_handoff))
                world(row, case, row["repeat_artifacts"], memory_handoff)
                for x, y in zip(row["artifacts"], row["repeat_artifacts"], strict=True):
                    same(read(x), read(y))
            summary = dict(
                second_quality=sum(x["quality"] for x in checked),
                stable=sum(x["stable"] for x in checked),
                quality_regressions=[
                    i
                    for i, (old, new) in enumerate(zip(continuous, checked, strict=True))
                    if old["quality"] and not new["quality"]
                ],
                stability_regressions=[
                    i
                    for i, (old, new) in enumerate(zip(continuous, checked, strict=True))
                    if old["stable"] and not new["stable"]
                ],
            )
            if any(section[k] != v for k, v in summary.items()):
                raise ValueError("continuous statistics differ")
            preserved = all(
                x["quality"] >= (16 if sequences and key == "old16" else 14)
                and x["stable"] == 16
                and not x["regressions"]
                for key, x in iso.items()
            )
            gain = (
                preserved
                and summary["second_quality"] >= 8
                and (
                    not sequences
                    or summary["second_quality"] > sum(w["quality"] for w in continuous)
                )
                and not summary["quality_regressions"]
                and not summary["stability_regressions"]
            )
            full = gain and summary["second_quality"] >= 12 and summary["stable"] == 16
            if (
                gain != candidate["research_contact_gain"]
                or full != candidate["full_continuous_skill_gate"]
            ):
                raise ValueError("research/fullskill gate differs")
            summaries.append(
                dict(
                    candidate=candidate["candidate"],
                    isolated=iso,
                    continuous=summary,
                    research_contact_gain=gain,
                    full_continuous_skill_gate=full,
                )
            )
        if native != r["actual_native_steps"] or neural != r["actual_neural_rows"]:
            raise ValueError("all actual neural/native work totals differ")
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="PAID_LIVE_DISTILLED_ALL_PRIMITIVE_AUDIT_PASS",
                input_file_hashes=pins,
                summaries=summaries,
                episode_raws=episode_count,
                hold_blocks=hold_count,
                no_reset_skill_boundaries=boundary_count,
                producer_native_steps=native,
                producer_neural_rows=neural,
                own_native_steps=0,
                own_neural_rows=0,
                goal_complete=False,
                team_game=False,
                hardware_authorized=False,
                promotion_authorized=False,
                commercial_publicity_authorized=False,
            ),
        )
    except Exception as error:
        write_once(a.output / "failed.json", dict(error=repr(error), hardware_authorized=False))
        raise


if __name__ == "__main__":
    main()
