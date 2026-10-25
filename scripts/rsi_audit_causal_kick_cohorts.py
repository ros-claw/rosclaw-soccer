"""Independent raw-physics, own-observation/RNG and motor rollback audit of three seeds."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, nargs=3, required=True)
    p.add_argument("--producer-unit", nargs=3, required=True)
    p.add_argument("--expected-producer-head", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_close_walk_kick_teacher import wait_for_producers
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    if a.output.exists() or len({p.resolve() for p in a.root}) != 3:
        raise ValueError("unused audit output and three unique cohort roots required")
    wait_for_producers(a.producer_unit)
    reports, pins = [], {}
    for seed, root in zip((926211, 926212, 926213), a.root, strict=True):
        cp = root / "complete.json"
        if cp.is_symlink():
            raise ValueError("ordinary completed cohort required")
        report = json.loads(cp.read_text())
        if report[
            "status"
        ] != "FOUR_CAUSAL_PPO_CYCLES_PHYSICALLY_GATED_AND_FINAL40_MEASURED_NOT_PROMOTED" or (
            report["source_head"] != a.expected_producer_head
            or report["seed"] != seed
            or len(report["cycles"]) != 4
            or report["value_optimizer_updates"] != 16
            or report["promotion_authorized"] is not False
            or report["hardware_authorized"] is not False
        ):
            raise ValueError("all three fixed four-cycle unpromoted cohorts required")
        for name, h in {**report["input_file_hashes"], str(cp): digest(cp)}.items():
            if name in pins and pins[name] != h:
                raise ValueError("conflicting source identity")
            pins[name] = h
        for name, h in report["physical_record_hashes"].items():
            path = root / name
            if (
                path.is_symlink()
                or not path.resolve().is_relative_to(root.resolve())
                or digest(path) != h
            ):
                raise ValueError("exact confined physical record required")
            pins[str(path)] = h
            row = json.loads(path.read_text())
            artifact = path.parent / row["artifact"]["path"]
            if artifact.is_symlink() or artifact.parent.resolve() != path.parent.resolve():
                raise ValueError("confined physical arrays required")
            pins[str(artifact)] = row["artifact"]["hash"]
            pins[str(path.with_name(path.stem + "-attempt.json"))] = row["attempt_record_hash"]
        for path in root.glob("*.pt"):
            if path.is_symlink():
                raise ValueError("ordinary candidate/active checkpoints required")
            pins[str(path)] = digest(path)
        reports.append(report)
    source_pins(pins)
    check_pins(pins)
    import mujoco
    import numpy as np
    import torch
    from rosclaw.simforge.backends.unitree_mujoco_backend import _load_robonaldo
    from rosclaw.simforge.tasks.g1_goalforge.concepts import G1_HARD_TORQUE_LIMITS

    from rosclaw_soccer.rsi.correlated_motor_exploration import conditional_logp, sample_action
    from rosclaw_soccer.rsi.kick_entry_reconstruction import KickEntryReconstructor
    from rosclaw_soccer.rsi.kick_event_reward import kick_event_rewards
    from rosclaw_soccer.rsi.kick_prior_actor import KICK_PRIOR_HASH, KickPriorActor
    from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
    from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
    from rosclaw_soccer.rsi.walk_kick_teacher_audit import relabel_arrays
    from rosclaw_soccer.sim.contracts import hash_json

    def one(suffix):
        found = [Path(k) for k in pins if k.endswith(suffix)]
        if len(found) != 1:
            raise ValueError("unique bound body/policy required")
        return found[0]

    torch.set_num_threads(1)
    model = mujoco.MjModel.from_xml_string(
        shared_native_scene(
            one("/cached-standard/declared-scene.xml").read_text(), prefixes=("a/",)
        )
    )
    layout = NativeVelocityLayout(model, robot_prefix="a/", ball_body_name="soccer_ball")
    state, output, policy_type, mapping = _load_robonaldo(
        one("/policy/robonaldo/FreeKick.py").parents[2]
    )
    policy = policy_type(state(29), output(29))
    actor = KickPriorActor.from_onnx(
        one("/policy/robonaldo/model/policy-obs-aic.onnx"), expected_hash=KICK_PRIOR_HASH
    ).eval()
    prior = {k: v.clone() for k, v in actor.state_dict().items()}
    log_std = torch.full((29,), float(np.log(0.015)), dtype=torch.float32)
    originals = {}
    for index in range(40):
        candidates = [Path(k) for k in pins if k.endswith(f"/course-{index:02d}/onnx.json")]
        if not candidates:
            raise ValueError("verified original physical comparison missing")
        versions = [json.loads(path.read_text()) for path in candidates]
        signatures = [
            hash_json(
                dict(
                    course=r["course"],
                    parameters=r["parameters"],
                    healthy=r["healthy_full_horizon"],
                    outcome=r["outcome"],
                    experimental=r["experimental_fast_approach"],
                )
            )
            for r in versions
        ]
        if len(set(signatures)) != 1:
            raise ValueError("original physical comparison is ambiguous")
        originals[index] = versions[0]

    def equal(x, y):
        if isinstance(x, torch.Tensor):
            return isinstance(y, torch.Tensor) and torch.equal(x, y)
        if isinstance(x, dict):
            return isinstance(y, dict) and set(x) == set(y) and all(equal(x[k], y[k]) for k in x)
        if isinstance(x, (tuple, list)):
            return (
                isinstance(y, type(x))
                and len(x) == len(y)
                and all(equal(a, b) for a, b in zip(x, y, strict=True))
            )
        return x == y

    def checkpoint(root, item):
        path = root / item["path"]
        if path.parent.resolve() != root.resolve() or digest(path) != item["hash"]:
            raise ValueError("exact committed checkpoint required")
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["source_head"] != a.expected_producer_head or saved["body_hash"] != hash_json(
            layout.profile()
        ):
            raise ValueError("checkpoint source/body mismatch")
        if any(
            not torch.equal(v, saved["actor"][k])
            for k, v in prior.items()
            if not k.startswith("actor.6.")
        ):
            raise ValueError("original motor features changed")
        return saved

    class Session:
        def __init__(self, arrays, seed):
            self.arrays, self.i = arrays, 0
            self.generator = None if seed is None else torch.Generator().manual_seed(seed)
            self.last_action = self.last_mean = None

        def run(self, outputs, feed):
            obs = feed["obs"]
            with torch.no_grad():
                mean = actor(torch.from_numpy(obs.copy()))[0]
            if self.generator is None:
                self.i += 1
                return [mean.numpy()[None].copy()]
            if not np.array_equal(
                obs[0], self.arrays["own_observations"][self.i]
            ) or not np.array_equal(mean.numpy(), self.arrays["own_old_means"][self.i]):
                raise ValueError(
                    "actual own causal observation/mean differs from native state reconstruction"
                )
            action, _ = sample_action(
                mean,
                log_std,
                rho=0.9,
                previous_action=self.last_action,
                previous_mean=self.last_mean,
                generator=self.generator,
            )
            if not np.array_equal(action.numpy(), self.arrays["own_raw_actions"][self.i]):
                raise ValueError("actual own raw action failed exact seed/RNG replay")
            self.last_action, self.last_mean = action, mean
            self.i += 1
            return [action.numpy()[None].copy()]

    summaries = []
    for root, report in zip(a.root, reports, strict=True):
        rows, native, samples = {}, 0, 0
        for name in report["physical_record_hashes"]:
            path = root / name
            row = json.loads(path.read_text())
            with np.load(path.parent / row["artifact"]["path"]) as source:
                arrays = dict(source)
            attempt = json.loads(path.with_name(path.stem + "-attempt.json").read_text())
            if (
                attempt["course_index"] != row["course_index"]
                or attempt["policy_checkpoint"] != row["policy_checkpoint"]
                or ("exploration_seed" in row and attempt["seed"] != row["exploration_seed"])
            ):
                raise ValueError("charged attempt differs from actual execution")
            label = relabel_arrays(
                arrays, controls=row["controls"], ball_address=layout.bq, root_address=layout.rq
            )
            original = originals[row["course_index"]]
            if (
                row["course"] != original["course"]
                or row["parameters"] != original["parameters"]
                or (row["experimental_fast_approach"] != original["experimental_fast_approach"])
            ):
                raise ValueError("physical course or approach program changed")
            pose = arrays["qpos"][:, layout.rq : layout.rq + 7]
            healthy = row["controls"] == 360 and not bool(
                np.any((pose[:, 2] < 0.65) | (1 - 2 * (pose[:, 4] ** 2 + pose[:, 5] ** 2) < 0.9))
            )
            if (
                hash_json(label) != hash_json(row["outcome"])
                or healthy != row["healthy_full_horizon"]
                or (
                    np.any(
                        abs(arrays["torque"][:, layout.ai])
                        > 0.85 * np.array(G1_HARD_TORQUE_LIMITS) + 1e-12
                    )
                    or row["native_steps"] != row["controls"] * 10
                )
            ):
                raise ValueError("native task/body/effort/clock label mismatch")
            saved = checkpoint(root, row["policy_checkpoint"])
            actor.load_state_dict(saved["actor"], strict=True)
            if row["entry_control"] is not None:
                own = "own_raw_actions" in arrays
                session = Session(arrays, row["exploration_seed"] if own else None)
                policy.ort_session = session
                rebuilt = KickEntryReconstructor(model, layout, policy, mapping).reconstruct_trace(
                    row, arrays
                )
                if session.i != len(rebuilt["observations"]):
                    raise ValueError("all actual neural control frames must replay")
                if own:
                    if not torch.equal(
                        session.generator.get_state(),
                        torch.from_numpy(arrays["own_generator_state"]),
                    ):
                        raise ValueError("saved final own RNG state differs")
                    replay = conditional_logp(
                        torch.from_numpy(arrays["own_old_means"]),
                        torch.from_numpy(arrays["own_raw_actions"]),
                        log_std,
                        rho=0.9,
                    )
                    if float(abs(replay - torch.from_numpy(arrays["own_raw_logp"])).max()) > 0.002:
                        raise ValueError("own raw likelihood replay mismatch")
                    n = row["controls"]
                    rewards = kick_event_rewards(
                        ball_after=arrays["ball_position_after_step_m"],
                        pelvis_native=pose.reshape(n, 10, 7),
                        force_native=arrays["ball_forces"][:, 0].reshape(n, 10, 3),
                    )
                    if not np.array_equal(rewards["rewards"], arrays["causal_rewards"]):
                        raise ValueError("actual causal reward changed")
                    samples += session.i
            rows[name] = row
            native += row["native_steps"]
        if (
            len(rows) != report["worlds"]
            or native != report["native_steps"]
            or samples != report["own_motor_samples"]
        ):
            raise ValueError("cohort execution accounting differs")
        previous = checkpoint(root, dict(path="initial.pt", hash=pins[str(root / "initial.pt")]))
        commits = 0
        previous_item = dict(path="initial.pt", hash=pins[str(root / "initial.pt")])
        for cycle in report["cycles"]:
            candidate = checkpoint(root, cycle["candidate"])
            active = checkpoint(root, cycle["active_checkpoint"])
            accepted = cycle["proposal"]["actor_decision"]["accepted"]
            chosen = cycle["chosen_courses"]
            if len(set(chosen)) != 8 or any(
                sum(i // 10 == k for i in chosen) != 2 for k in range(4)
            ):
                raise ValueError("balanced current-policy eight-course curriculum required")
            for index in chosen:
                own = rows[f"cycle-{cycle['cycle']:02d}/own-{index:02d}.json"]
                if own["policy_checkpoint"] != previous_item:
                    raise ValueError(
                        "own learning batch was not generated by current active checkpoint"
                    )
            if accepted and (
                not equal(candidate["pending"]["actor"], previous["actor"])
                or not equal(candidate["pending"]["optimizer"], previous["actor_optimizer"])
            ):
                raise ValueError(
                    "physical rollback snapshot was not bound to previous current actor"
                )
            measured_losses = []
            for item in cycle["retention"]:
                index = item["course_index"]
                row = rows[f"cycle-{cycle['cycle']:02d}/retention-{index:02d}.json"]
                old = originals[index]
                regression = (
                    (old["healthy_full_horizon"] and not row["healthy_full_horizon"])
                    or (old["outcome"]["safe"] and not row["outcome"]["safe"])
                    or (old["outcome"]["task_pass"] and not row["outcome"]["task_pass"])
                )
                if (
                    regression != item["regression"]
                    or row["policy_checkpoint"] != cycle["candidate"]
                ):
                    raise ValueError("physical regression or candidate binding differs")
                measured_losses.append(bool(regression))
            committed = cycle["physically_committed"]
            if committed != bool(
                accepted and len(measured_losses) == 40 and not any(measured_losses)
            ):
                raise ValueError("physical commit disagrees with actual retained abilities")
            if committed and sorted(i["course_index"] for i in cycle["retention"]) != list(
                range(40)
            ):
                raise ValueError("forty UNIQUE retained contexts required")
            if (
                accepted
                and not committed
                and (not measured_losses or not measured_losses[-1] or any(measured_losses[:-1]))
            ):
                raise ValueError("rejected physical prefix must end at its first actual regression")
            reference = candidate if committed else previous
            if not equal(active["actor"], reference["actor"]) or not equal(
                active["actor_optimizer"], reference["actor_optimizer"]
            ):
                raise ValueError("physical motor/Adam commit or rollback is not exact")
            if committed and (
                len(cycle["retention"]) != 40 or any(x["regression"] for x in cycle["retention"])
            ):
                raise ValueError("committed candidate did not clear all forty retention contexts")
            commits += int(committed)
            if (
                active["physical_motor_commits"] != commits
                or active["value_steps"] != 4 * (cycle["cycle"] + 1)
                or active["pending"] is not None
            ):
                raise ValueError("active optimizer/value/transaction counters disagree")
            previous = active
            previous_item = cycle["active_checkpoint"]
        if commits != report["physical_motor_commits"]:
            raise ValueError("physical commit total differs")
        if report["final_checkpoint"] != previous_item:
            raise ValueError("final evaluation must use final active policy, not a best checkpoint")
        moment_steps = [int(s["step"]) for s in previous["actor_optimizer"]["state"].values()]
        if moment_steps != ([commits, commits] if commits else []):
            raise ValueError("active Adam moments do not match committed motor updates")
        if len(previous["value_optimizer"]["state"]) != 4 or any(
            int(s["step"]) != 16 for s in previous["value_optimizer"]["state"].values()
        ):
            raise ValueError("actual value optimizer step counts differ")
        final = [rows[f"final40/course-{i:02d}.json"] for i in range(40)]
        if any(r["policy_checkpoint"] != previous_item for r in final):
            raise ValueError("final40 mixed candidate or stale policies")
        passes = sum(r["healthy_full_horizon"] and r["outcome"]["task_pass"] for r in final)
        if passes != report["passes"]:
            raise ValueError("whole40 score differs")
        summaries.append(
            dict(
                seed=report["seed"],
                worlds=len(rows),
                native_steps=native,
                own_neural_samples=samples,
                physical_motor_commits=commits,
                numerical_actor_accepts=report["numerical_actor_accepts"],
                value_updates=16,
                final_passes=passes,
                final_checkpoint=report["final_checkpoint"],
            )
        )
    check_pins(pins)
    clean(a.expected_head)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "complete.json",
        dict(
            status="ALL_THREE_CAUSAL_COHORTS_NATIVE_NEURAL_RNG_AND_ROLLBACK_REPLAY_VERIFIED",
            source_head=a.expected_head,
            input_file_hashes=pins,
            summaries=summaries,
            new_physical_steps=0,
            optimizer_updates=0,
            promotion_authorized=False,
            hardware_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
