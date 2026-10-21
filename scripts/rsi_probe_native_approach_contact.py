"""Qualify new continuous approach/contact/recovery mechanics, NOT a learner.

Fixed original177 foundation, TRAIN balanced sources, no reward optimization,
no new model and no inherited contact24 qualification. Native strict replays.
"""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    p = argparse.ArgumentParser(description=__doc__)
    for key in ("proof-root", "reference-root", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("proof-hash", "expected-head"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--worlds", choices=(8, 64), type=int, default=8)
    p.add_argument("--leads", choices=(10, 20, 40, 80), type=int, nargs="+", default=[80])
    p.add_argument(
        "--faults", action="store_true", help="declared8world head/tail failure injection"
    )
    a = p.parse_args()
    if (
        a.output.exists()
        or len(set(a.leads)) != len(a.leads)
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
        or (a.faults and (a.worlds != 8 or a.leads != [80]))
    ):
        raise ValueError("new output, unique declared leads and clean frozen source required")
    receipt = a.proof_root / "complete.json"
    if digest(receipt) != a.proof_hash:
        raise ValueError("exact actual contact/recovery prerequisite required")
    proof = json.loads(receipt.read_text())
    if (
        proof["status"] != "NATIVE_CONTACT24_RECOVERY60_PROBE_NOT_LEARNER"
        or proof["strict_replays_exact"] is not True
        or proof["repeats_exact"] is not True
    ):
        raise ValueError("actual completed continuous native prerequisite required")
    pins = {**proof["input_file_hashes"], str(receipt): a.proof_hash}
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts", a.reference_root / "src"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            protocol="native-approach-measured-contact-recovery60-probe-v1",
            worlds=a.worlds,
            leads=a.leads,
            independent_full_repeats=2,
            maximum_head_controls=300,
            recovery_controls=60,
            measured_foot_contact_threshold_N=10,
            following_contact_controls=14,
            maximum_live_native_steps=len(a.leads) * 2 * a.worlds * 360 * 10,
            maximum_replay_native_steps=len(a.leads) * 2 * a.worlds * 360 * 10,
            maximum_neural_rows=len(a.leads) * 2 * a.worlds * 360,
            same_live_MjData_at_handoff=True,
            reference_padding=False,
            body_height_m=0.65,
            body_upright=0.9,
            guards_at_control_boundary=True,
            optimizer_updates=0,
            original177_foundation_only=True,
            foundation_inference_batch_rows=1,
            legacy24_reward_diagnostic_not_for_learning=True,
            fault_injection=(
                dict(
                    source0_reference_ankle_z_error_m=1.0,
                    tail_world1_HG_joint_ids=[0, 3, 4, 6, 9, 10],
                    tail_world1_joint_target_rad=[-1.2, 2.4, -0.8, -1.2, 2.4, -0.8],
                    original_PD_joint_limits_required=True,
                )
                if a.faults
                else None
            ),
            old24_protocol_unchanged_not_inherited_qualification=True,
            goal_complete=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )
    course = None
    closed_live = replay = neural = 0
    head_rows, tail_rows = [], []
    try:
        import mujoco
        import numpy as np
        import torch

        from rosclaw_soccer.providers.g1.velocity_policy import VelocityPolicyParameters, target29
        from rosclaw_soccer.rsi.contact_option_clock import enter_measured_recovery
        from rosclaw_soccer.rsi.mjlab_motion_curriculum import TRAIN_MOTION_NAMES
        from rosclaw_soccer.rsi.native_approach_contact import (
            NativeApproachContactCourse,
            NativeApproachRecoveryTail,
        )
        from rosclaw_soccer.rsi.native_contact_window import native_contact_window
        from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
        from rosclaw_soccer.rsi.velocity_prior_actor import VelocityPriorActor
        from scripts.rsi_native_course_inputs import load_body_references

        torch.set_num_threads(1)

        def pinned(suffix):
            matches = [Path(f) for f in pins if f.endswith(suffix)]
            if len(matches) != 1:
                raise ValueError("exact unique prerequisite required: " + suffix)
            return matches[0]

        def load(path):
            with np.load(path, allow_pickle=False) as z:
                return {k: z[k].copy() for k in z.files}

        model = mujoco.MjModel.from_binary_path(
            str(pinned("/rsi-body-bound-foot-only-qualification-20261021/compiled_model.mjb"))
        )
        layout = NativeVelocityLayout(model, robot_prefix="robot/", ball_body_name="ball/ball")
        config = pinned("/policy/velocity/v0/params/deploy.yaml")
        params = VelocityPolicyParameters.load(config, expected_hash=pins[str(config)])
        bank = load(pinned("/precontact-resets.npz"))
        plans = [
            load(
                pinned(
                    "/rsi-train-starts-body-bound-index-check-20261021/"
                    + n.removesuffix(".npz")
                    + "-plan.npz"
                )
            )
            for n in TRAIN_MOTION_NAMES
        ]
        refs = load_body_references(a.reference_root, model, layout, plans)
        fault_ids = np.array([0, 3, 4, 6, 9, 10])
        fault_targets = np.array([-1.2, 2.4, -0.8, -1.2, 2.4, -0.8], dtype=np.float32)
        if a.faults:
            from rosclaw_soccer.rsi.native_course_tracking import TRACKING_BODIES

            refs[0]["position"][:, TRACKING_BODIES.index("left_ankle_roll_link"), 2] += 1.0
            bounds = model.jnt_range[layout.joints][fault_ids]
            if (fault_targets < bounds[:, 0]).any() or (fault_targets > bounds[:, 1]).any():
                raise ValueError("declared fault targets outside actual joints; no repair")
        onnx = pinned("/policy/velocity/v0/exported/policy.onnx")
        actor = VelocityPriorActor.from_onnx(
            onnx, expected_hash=pins[str(onnx)], reference_features=79
        ).eval()
        state = torch.load(
            pinned("/ball-aware-update200.pt"), map_location="cpu", weights_only=True
        )
        actor.load_state_dict(
            {k: state["algorithm"]["actor_state_dict"]["mlp." + k] for k in actor.state_dict()},
            strict=True,
        )
        for parameter in actor.parameters():
            parameter.requires_grad_(False)
        original = {k: v.clone() for k, v in actor.state_dict().items()}

        def act(obs, ids):
            nonlocal neural
            action = np.zeros((a.worlds, 29), dtype=np.float32)
            if len(ids):
                with torch.no_grad():
                    # Match the original frame-zero native teacher's GEMV
                    # execution, not size-dependent batched GEMM rounding.
                    for i in ids:
                        action[i] = actor(torch.from_numpy(obs[i : i + 1])).numpy()[0]
                neural += len(ids)
            return action

        def snapshot(obj, ids):
            return {
                int(i): (
                    obj.batch.data[i],
                    obj.batch.data[i].qpos.copy(),
                    obj.batch.data[i].qvel.copy(),
                    obj.batch.data[i].qacc_warmstart.copy(),
                    obj.batch.data[i].ctrl.copy(),
                    obj.batch.data[i].time,
                    obj.phase[i].copy(),
                    obj.last_action[i].copy(),
                    obj.observation[i].copy(),
                )
                for i in ids
            }

        def assert_snapshot(obj, saved, *, observation=True):
            for i, (old, q, v, warm, ctrl, time, phase, last, obs) in saved.items():
                data = obj.batch.data[i]
                pairs = [
                    (data.qpos, q),
                    (data.qvel, v),
                    (data.qacc_warmstart, warm),
                    (data.ctrl, ctrl),
                    (data.time, time),
                    (obj.phase[i], phase),
                    (obj.last_action[i], last),
                ]
                if observation:
                    pairs.append((obj.observation[i], obs))
                if data is not old or any(not np.array_equal(x, y) for x, y in pairs):
                    raise ValueError("handoff/failure-freeze mutated actual native state/history")

        qi = [*range(layout.rq, layout.rq + 7), *layout.qi, *range(layout.bq, layout.bq + 7)]
        vi = [*range(layout.rv, layout.rv + 6), *layout.vi, *range(layout.bv, layout.bv + 6)]
        results, hashes = [], {}
        for lead in a.leads:
            first_raw = None
            for repeat in range(2):
                course = NativeApproachContactCourse(
                    model=model,
                    params=params,
                    bank=bank,
                    plans=plans,
                    body_references=refs,
                    worlds=a.worlds,
                    lead=lead,
                    threads=8,
                )
                course.reset()
                head_rows, tail_rows = [], []
                while not course.ended.all():
                    stopped = snapshot(course, np.flatnonzero(course.ended))
                    action = act(course.observation, np.flatnonzero(~course.ended))
                    head_rows.append(dict(**course.step(action), action=action.copy()))
                    assert_snapshot(course, stopped)
                saved = snapshot(course, np.arange(a.worlds))
                tail = NativeApproachRecoveryTail(course)
                assert_snapshot(tail, saved, observation=False)
                while not tail.ended.all():
                    stopped = snapshot(tail, np.flatnonzero(tail.ended))
                    action = act(tail.observation, tail.active_ids)
                    if a.faults and 1 in tail.active_ids:
                        action[1, fault_ids] = (
                            fault_targets - np.asarray(params.action_offset)[fault_ids]
                        ) / np.asarray(params.action_scale)[fault_ids]
                    tail_rows.append(tail.step(action))
                    assert_snapshot(tail, stopped)
                if a.faults and (
                    not course.head_failed[0]
                    or course.elapsed[0] != 1
                    or course.head_failed[1:].any()
                    or not tail.failed[1]
                    or not 0 < tail.elapsed[1] < 60
                    or tail.failed[2:].any()
                    or not np.all(tail.elapsed[2:] == 60)
                ):
                    raise ValueError(
                        "declared head/tail failures did not isolate actual expected worlds"
                    )
                raw = {"head_" + k: np.stack([row[k] for row in head_rows]) for k in head_rows[0]}
                if tail_rows:
                    raw.update(
                        {"tail_" + k: np.stack([row[k] for row in tail_rows]) for k in tail_rows[0]}
                    )
                raw["bank_index"] = course.index.copy()
                if first_raw is not None and (
                    set(raw) != set(first_raw)
                    or any(not np.array_equal(raw[k], first_raw[k]) for k in raw)
                ):
                    raise ValueError("full independent neural/physical repeat differs")
                first_raw = raw
                path = a.output / f"lead{lead}-repeat{repeat}.npz"
                np.savez_compressed(path, **raw)
                hashes[path.name] = digest(path)
                measured = []
                for w, index in enumerate(course.index):
                    rd = mujoco.MjData(model)
                    rd.qpos[:], rd.qvel[:] = bank["qpos"][index], bank["qvel"][index]
                    rd.time = float(bank["reference_frame"][index] * 0.02)
                    rd.ctrl[layout.ai] = target29(bank["previous_action"][index], params)
                    mujoco.mj_forward(model, rd)
                    rd.qacc_warmstart[:] = bank["qacc_warmstart"][index]
                    qq, vv, ff = [], [], []
                    for part in ("head_", "tail_"):
                        if part + "valid" not in raw:
                            continue
                        for control in range(len(raw[part + "valid"])):
                            if not raw[part + "valid"][control, w]:
                                continue
                            rd.ctrl[layout.ai] = raw[part + "target"][control, w]
                            for substep in range(10):
                                mujoco.mj_step(model, rd)
                                replay += 1
                                f = course._forces(rd)
                                q, v = (
                                    raw[part + "qpos"][control, w, substep],
                                    raw[part + "qvel"][control, w, substep],
                                )
                                if (
                                    not np.array_equal(rd.qpos, q)
                                    or not np.array_equal(rd.qvel, v)
                                    or not np.array_equal(
                                        f, raw[part + "forces"][control, w, substep]
                                    )
                                ):
                                    raise ValueError(
                                        "strict native recorded-control replay differs"
                                    )
                                qq.append(q[qi])
                                vv.append(v[vi])
                                ff.append(f)
                            mujoco.mj_forward(model, rd)
                    q, v, f = np.array(qq), np.array(vv), np.array(ff)
                    first = int(course.first_contact_control[w])
                    if not course.head_failed[w] and not enter_measured_recovery(
                        completed_controls=int(course.elapsed[w]),
                        first_contact_control=None if first < 0 else first,
                        reference_horizon=int(course.reference_horizon[w]),
                    ):
                        raise ValueError(
                            "raw execution did not reach its declared measured handoff"
                        )
                    measured.append(
                        dict(
                            world=w,
                            bank_index=int(index),
                            head_controls=int(course.elapsed[w]),
                            first_contact_control=first,
                            reference_horizon=int(course.reference_horizon[w]),
                            head_failed=bool(course.head_failed[w]),
                            tail_failed=bool(tail.failed[w]),
                            tail_controls=int(tail.elapsed[w]),
                            forbidden=bool((f[:, 2] > 1).any()),
                            stable=bool(
                                q[:, 2].min() >= 0.65
                                and (1 - 2 * (q[:, 4] ** 2 + q[:, 5] ** 2)).min() >= 0.9
                            ),
                            postcontact=native_contact_window(
                                force=f[:, :2],
                                canonical_qpos=q,
                                canonical_qvel=v,
                                goal=bank["goal_world"][index],
                                intended_foot=int(bank["expected_foot"][index]),
                            ),
                        )
                    )
                live = course.native_steps
                if live != (sum(x["head_controls"] + x["tail_controls"] for x in measured) * 10):
                    raise ValueError("physical work differs from actual active controls")
                closed_live += live
                course.close()
                course = None
                results.append(dict(lead=lead, repeat=repeat, worlds=measured, native_steps=live))
                print(
                    json.dumps(
                        dict(
                            lead=lead,
                            repeat=repeat,
                            native_steps=live,
                            stable=sum(x["stable"] for x in measured),
                            head_controls=[x["head_controls"] for x in measured],
                        )
                    ),
                    flush=True,
                )
        if replay != closed_live or neural * 10 != closed_live:
            raise ValueError("actual native/replay/inference counts differ")
        if any(not torch.equal(v, actor.state_dict()[k]) for k, v in original.items()):
            raise ValueError("immutable foundation changed")
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="NATIVE_APPROACH_MEASURED_CONTACT_RECOVERY_PROBE_NOT_LEARNER",
                input_file_hashes=pins,
                raw_hashes=hashes,
                results=results,
                live_native_steps=closed_live,
                replay_native_steps=replay,
                neural_rows=neural,
                continuous_handoff_exact=True,
                strict_replays_exact=True,
                full_repeats_exact=True,
                stopped_worlds_frozen=True,
                no_reference_end_padding=True,
                optimizer_updates=0,
                goal_complete=False,
                promotion_authorized=False,
                hardware_authorized=False,
            ),
        )
    except Exception as error:
        import numpy as np

        partial = {}
        for name, rows in (("head_", head_rows), ("tail_", tail_rows)):
            if rows:
                partial.update({name + k: np.stack([row[k] for row in rows]) for k in rows[0]})
        if partial:
            np.savez_compressed(a.output / "failed-partial.npz", **partial)
        write_once(
            a.output / "failed.json",
            dict(
                error=repr(error),
                live_native_steps=closed_live + (course.native_steps if course else 0),
                replay_native_steps=replay,
                neural_rows=neural,
                optimizer_updates=0,
                promotion_authorized=False,
                hardware_authorized=False,
            ),
        )
        raise
    finally:
        if course is not None:
            course.close()


if __name__ == "__main__":
    main()
