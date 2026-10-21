"""Reconstruct all live-body CEM outcomes, selection and work from sealed raw.

Producer full-sequence physics repeats are checked, not recounted as new audit
physics. This auditor independently computes primitive PD/contact outcomes.
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
        raise ValueError("unused output and exact completed search/frozen audit code required")
    r, d = json.loads(rp.read_text()), json.loads(dp.read_text())
    if r["status"] != "PAID_LIVE_BODY_CONTACT_TEACHER_RESEARCH_NOT_PROMOTED":
        raise ValueError("completed post-kick physical teacher search required")
    pins = {**r["input_file_hashes"], str(rp): a.hash, str(dp): digest(dp)}
    for relative, h in r["artifact_hashes"].items():
        path = a.root / relative
        if path.is_symlink() or not path.resolve().is_relative_to(a.root.resolve()):
            raise ValueError("ordinary confined sealed artifact required")
        pins[str(path)] = h
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            input_file_hashes=pins,
            own_native_steps=0,
            own_neural_rows=0,
            nonselected_new_physics_replay=False,
            hardware_authorized=False,
        ),
    )
    import mujoco
    import numpy as np
    import onnx

    from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
    from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
    from rosclaw_soccer.rsi.paid_primitive_audit import PaidPrimitiveAuditor
    from rosclaw_soccer.rsi.paid_teacher_policy import strict_parameters
    from scripts.rsi_search_paid_foot_contact import contact_rank

    policy = next(Path(f) for f in pins if f.endswith("/ckp/policy_30000.onnx"))
    xml = next(Path(f) for f in pins if f.endswith("/cached-standard/declared-scene.xml"))
    graph = onnx.load(policy, load_external_data=False)
    joint_names, params = strict_parameters({x.key: x.value for x in graph.metadata_props})
    order = np.array([joint_names.index(n) for n in G1_DDS_JOINT_NAMES])
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
    cases = d["TRAIN_cases"]
    names = list(dict.fromkeys(c["source"] for c in cases))
    if len(cases) != 16 or len(names) != 8:
        raise ValueError("all sixteen two-condition live parents required")
    source_indices = d.get("search_source_indices", list(range(8)))
    final_indices = d.get("final_case_indices", list(range(16)))
    expected_final = [i for i, c in enumerate(cases) if names.index(c["source"]) in source_indices]
    if (
        not source_indices
        or len(set(source_indices)) != len(source_indices)
        or any(type(i) is not int or i not in range(8) for i in source_indices)
        or final_indices != expected_final
        or len(r["selected"]) != len(source_indices)
        or len(r["final"]) != len(final_indices)
        or len(r["baseline"]) != 16
    ):
        raise ValueError("complete predeclared source shard and paired final courses required")
    expected_raw_count = 32 + 72 * len(source_indices) + 4 * len(final_indices)
    horizons = {}
    for name in names:
        path = policy.parent.parent / "motions/soccer-standard" / name
        if str(path) not in pins:
            raise ValueError("motion reference horizon must be pinned")
        with np.load(path, allow_pickle=False) as z:
            horizons[name] = len(z["joint_pos"])
    centers = json.loads((a.root / "baseline/complete.json").read_text())["centers"]
    audited = {}
    native = neural = hold_steps = 0

    def read(path):
        if str(path) not in pins:
            raise ValueError("every consumed artifact must have been sealed by producer")
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k].copy() for k in z.files}

    def snap(raw, prefix):
        return {k.removeprefix(prefix): v for k, v in raw.items() if k.startswith(prefix)}

    def same(x, y):
        if set(x) != set(y) or any(not np.array_equal(x[k], y[k]) for k in x):
            raise ValueError("full raw integration-state chain differs")

    def terminal(z, state):
        q, v = auditor.canonical(state["qpos"], state["qvel"])
        if (
            not np.array_equal(q, z["qpos"][-1, -1])
            or not np.array_equal(v, z["qvel"][-1, -1])
            or not np.array_equal(state["ctrl"][layout.ai], z["torque"][-1, -1])
            or not np.isclose(state["time"], z["time_before"][-1] + 0.02, rtol=0, atol=1e-11)
        ):
            raise ValueError("parent physical terminal q/v/ctrl/time differs")

    def audit_raw(row, i, latent=None, *, parent=None):
        nonlocal native, neural
        path = a.root / row["raw_path"]
        if pins.get(str(path)) != row["raw_hash"]:
            raise ValueError("row raw identity differs from sealed artifact")
        z = read(path)
        if parent is not None:
            same(snap(z, "initial_"), {k: parent[k] for k in snap(z, "initial_")})
        result = auditor.audit(
            z,
            row,
            horizon=horizons[cases[i]["source"]],
            previous_action=np.zeros(29, np.float32)
            if parent is None
            else parent["previous_actual_action"],
            intent=None if latent is None else (centers[cases[i]["source"]], latent),
        )
        if row["source"] != cases[i]["source"]:
            raise ValueError("source/parent foot identity differs")
        if row["raw_path"] not in audited:
            native += result["native_steps"]
            neural += result["neural_rows"]
            audited[row["raw_path"]] = result
        return z, result

    def parent(folder, i):
        nonlocal hold_steps
        record = json.loads((folder / "complete.json").read_text())
        first, _ = audit_raw(record["first"], i)
        # Re-admit every first skill against the previously qualified neural actor exam.
        references = d.get("bootstrap_first_references", [])
        if references:
            if len(references) != 16:
                raise ValueError("all sixteen qualified learned-parent references required")
            old = Path(references[i]["path"])
            if pins.get(str(old)) != references[i]["hash"]:
                raise ValueError("learned first-skill reference is not bound to producer inputs")
        else:
            old = next(
                Path(f) for f in pins if f.endswith(f"/candidate0/old16/case{i}-repeat0.npz")
            )
        with np.load(old, allow_pickle=False) as z:
            same(first, {k: z[k].copy() for k in z.files})
        drop = read(folder / "drop.npz")
        before = snap(drop, "before_")
        terminal(first, before)
        q, _ = auditor.canonical(before["qpos"], before["qvel"])
        performed = bool(drop["performed"])
        if performed != bool(np.linalg.norm(q[36:38] - q[:2]) >= 2):
            raise ValueError("practice ball return violated its declared eligibility")
        entry = before
        if performed:
            after = snap(drop, "after_")
            modified = {k: v.copy() for k, v in after.items()}
            modified["qpos"][layout.bq : layout.bq + 7] = before["qpos"][layout.bq : layout.bq + 7]
            modified["qvel"][layout.bv : layout.bv + 6] = before["qvel"][layout.bv : layout.bv + 6]
            same(before, modified)
            direction = before["mocap_pos"][goal_id, :2] - q[:2]
            direction /= np.linalg.norm(direction)
            sign = 1 if "left" in cases[i]["source"] else -1
            point = q[:2] + 0.8 * direction + sign * 0.15 * np.array([-direction[1], direction[0]])
            if (
                not np.array_equal(
                    after["qpos"][layout.bq : layout.bq + 7], np.r_[point, 0.4, 1.0, 0, 0, 0]
                )
                or after["qvel"][layout.bv : layout.bv + 6].any()
            ):
                raise ValueError("ball gravity-drop geometry or release velocity differs")
            entry = after
        current = read(folder / "parent.npz")
        if not np.array_equal(current["previous_actual_action"], first["NN_action"][-1]):
            raise ValueError("actual first-kick last action was not carried")
        previous = first
        if performed:
            z = read(folder / "hold.npz")
            n = len(z["target"])
            qb, vb = auditor.canonical(z["native_qpos_before"], z["native_qvel_before"])
            eq, ev = auditor.canonical(entry["qpos"], entry["qvel"])
            if (
                not 1 <= n <= 50
                or z["neural_control"].any()
                or not z["post_reference_hold"].all()
                or not np.array_equal(qb[0], eq)
                or not np.array_equal(vb[0], ev)
                or not np.array_equal(qb[1:], z["qpos"][:-1, -1])
                or not np.array_equal(vb[1:], z["qvel"][:-1, -1])
                or not np.isclose(z["time_before"][0], entry["time"], rtol=0, atol=1e-11)
                or not np.allclose(np.diff(z["time_before"]), 0.02, rtol=0, atol=1e-11)
                or not np.array_equal(
                    z["target"], np.broadcast_to(first["target"][-1], z["target"].shape)
                )
            ):
                raise ValueError(
                    "explicit native hold must preserve target/body/time and not pad NN"
                )
            jq = np.concatenate((qb[:, None, 7:36], z["qpos"][:, :-1, 7:36]), axis=1)
            jv = np.concatenate((vb[:, None, 6:35], z["qvel"][:, :-1, 6:35]), axis=1)
            if not np.array_equal(
                np.clip(
                    auditor.kp * (z["target"][:, None] - jq) - auditor.kd * jv, -effort, effort
                ),
                z["torque"],
            ):
                raise ValueError("actual hold torque differs")
            if (
                (z["qpos"][:, :, 2] < 0.65)
                | (1 - 2 * (z["qpos"][:, :, 4] ** 2 + z["qpos"][:, :, 5] ** 2) < 0.9)
            ).any():
                raise ValueError("failed hold may not produce a healthy parent")
            if z["qpos"][-1, -1, 38] > 0.12 or abs(z["qvel"][-1, -1, 37]) > 0.1:
                raise ValueError("ball was not actually settled in declared window")
            previous = z
            hold_steps += n * 10
        terminal(previous, current)
        # Independent exact native checkpoint restoration, without stepping.
        scratch = mujoco.MjData(model)
        mujoco.mj_setState(
            model, scratch, current["full_integration_state"], mujoco.mjtState.mjSTATE_INTEGRATION
        )
        for k in entry:
            if not np.array_equal(np.array(getattr(scratch, k)), current[k]):
                raise ValueError("serialized native parent integration state differs")
        return current

    try:
        parents = [parent(a.root / f"parent{i}", i) for i in range(16)]
        for i, row in enumerate(r["baseline"]):
            raw, _ = audit_raw(row, i, np.zeros((2, 6)), parent=parents[i])
            references = d.get("bootstrap_second_references", [])
            if references:
                if (
                    len(references) != 16
                    or pins.get(references[i]["path"]) != references[i]["hash"]
                ):
                    raise ValueError(
                        "all sixteen learned full-world second-skill references required"
                    )
                with np.load(references[i]["path"], allow_pickle=False) as z:
                    same({k: raw[k] for k in z.files}, {k: z[k].copy() for k in z.files})
        for slot, si in enumerate(source_indices):
            name = names[si]
            ids = [i for i, c in enumerate(cases) if c["source"] == name]
            rng = np.random.default_rng(20261022 + si)
            best_z, mean, std = np.zeros((2, 6)), np.zeros((2, 6)), np.full((2, 6), 0.5)
            best_rows = [r["baseline"][i] for i in ids]
            best_rank = contact_rank(best_rows)
            for generation in range(3):
                gp = a.root / f"source{si}/generation{generation}/complete.json"
                record = json.loads(gp.read_text())
                zs = [best_z.copy()] + list(rng.normal(mean, std, size=(11, 2, 6)))
                if len(record["candidates"]) != 12:
                    raise ValueError("all declared CEM candidates required")
                ranks = []
                for candidate, z in zip(record["candidates"], zs, strict=True):
                    latent = np.tanh(z)
                    if not np.array_equal(candidate["z"], z) or not np.array_equal(
                        candidate["latent"], latent
                    ):
                        raise ValueError("source-seeded CEM proposal differs")
                    for i, row in zip(ids, candidate["results"], strict=True):
                        audit_raw(row, i, latent, parent=parents[i])
                    rank = contact_rank(candidate["results"])
                    if list(rank) != candidate["rank"]:
                        raise ValueError("candidate selection rank differs")
                    ranks.append(rank)
                    if rank > best_rank:
                        best_z, best_rows, best_rank = z.copy(), candidate["results"], rank
                elite = np.stack(
                    [zs[j] for j in sorted(range(12), key=lambda j: ranks[j], reverse=True)[:4]]
                )
                mean = 0.5 * mean + 0.5 * elite.mean(axis=0)
                std = np.clip(0.5 * std + 0.5 * elite.std(axis=0), 0.15, 1.2)
                if (
                    record["best_rank"] != list(best_rank)
                    or not np.array_equal(record["next_z_mean"], mean)
                    or not np.array_equal(record["next_z_std"], std)
                ):
                    raise ValueError("CEM distribution update differs")
            selected = r["selected"][slot]
            if (
                selected["source"] != name
                or not np.array_equal(selected["selected_z"], best_z)
                or selected["results"] != best_rows
                or selected["rank"] != list(best_rank)
            ):
                raise ValueError("selected teacher not bound to complete CEM history")
            if not np.array_equal(selected["selected_latent"], np.tanh(best_z)):
                raise ValueError("selected bounded intent differs")
            print(json.dumps(dict(source=name, rank=list(best_rank))), flush=True)
        for i, row in zip(final_indices, r["final"], strict=True):
            selected = next(x for x in r["selected"] if x["source"] == cases[i]["source"])
            latent = np.array(selected["selected_latent"])
            chosen = selected["results"][i % 2]
            selected_raw = read(a.root / chosen["raw_path"])
            previous = None
            for repeat in (0, 1):
                folder = a.root / f"final{i}-repeat{repeat}"
                final_parent = parent(folder, i)
                same(final_parent, parents[i])
                actual = dict(row, raw_path=str((folder / "second.npz").relative_to(a.root)))
                actual["raw_hash"] = (
                    row["raw_hash"] if repeat == 0 else row["independent_repeat_raw_hash"]
                )
                z, _ = audit_raw(actual, i, latent, parent=final_parent)
                same(z, selected_raw)
                if previous is not None:
                    same(z, previous)
                previous = z
        if (
            len(audited) != expected_raw_count
            or native + hold_steps != r["actual_native_steps"]
            or neural != r["actual_neural_rows"]
        ):
            raise ValueError("declared complete raw episodes and explicit hold work ledger differ")

        def quality(rows):
            return sum(contact_rank([x, x])[1] > 0 for x in rows)

        if (
            quality(r["baseline"]) != r["baseline_quality"]
            or quality(r["final"]) != r["final_quality"]
            or sum(not x["body_failed"] for x in r["final"]) != r["stable"]
            or r["cem_distribution_updates"] != 3 * len(source_indices)
        ):
            raise ValueError("final physical improvement statistics differ")
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="PAID_LIVE_CONTACT_PRIMITIVE_CEM_AUDIT_PASS",
                input_file_hashes=pins,
                raw_episode_count=expected_raw_count,
                search_source_indices=source_indices,
                final_case_indices=final_indices,
                producer_native_steps=native + hold_steps,
                producer_neural_rows=neural,
                baseline_quality=r["baseline_quality"],
                final_quality=r["final_quality"],
                all_source_seeded_CEM_updates_reconstructed=True,
                all_selected_full_repeats_and_branches_exact=True,
                all_generation_artifacts_sealed=True,
                own_native_steps=0,
                own_neural_rows=0,
                nonselected_new_physics_replay=False,
                goal_complete=False,
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
