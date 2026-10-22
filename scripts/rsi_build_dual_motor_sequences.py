"""Admit audited whole histories and recompute own frozen reference for learning.

Old failures remain explicitly negative. New partial skill improvements are
not relabeled as successful football. No teacher h/c enters the dual learner.
"""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("rehearsal", "rehearsal-audit", "teacher", "teacher-audit", "qualification"):
        p.add_argument("--" + name + "-root", type=Path, required=True)
        p.add_argument("--" + name + "-hash", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    def clean():
        if (
            subprocess.check_output(["git", "status", "--porcelain"])
            or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            != a.expected_head
        ):
            raise ValueError("clean frozen actual-history sequence builder required")

    clean()
    if a.output.exists():
        raise ValueError("unused learning-sequence output required")
    reports = {}
    roots = {}
    pins = {}
    for name in ("rehearsal", "rehearsal_audit", "teacher", "teacher_audit", "qualification"):
        root = getattr(a, name + "_root")
        path = root / "complete.json"
        if (
            path.is_symlink()
            or (root / "failed.json").exists()
            or digest(path) != getattr(a, name + "_hash")
        ):
            raise ValueError("exact complete audited sequence sources required")
        report = json.loads(path.read_text())
        if (
            report["hardware_authorized"] is not False
            or report["promotion_authorized"] is not False
        ):
            raise ValueError("SIM_ONLY nonpromoted sources required")
        for f, h in report["input_file_hashes"].items():
            if f in pins and pins[f] != h:
                raise ValueError("conflicting source model or body lineage")
            pins[f] = h
        pins[str(path)] = digest(path)
        reports[name] = report
        roots[name] = root
    r, ra, t, ta, q = [
        reports[x]
        for x in ("rehearsal", "rehearsal_audit", "teacher", "teacher_audit", "qualification")
    ]
    if (
        ra["status"] != "ACTUAL_PARENT_TRAIN20_REHEARSAL_WHOLE_NATIVE_OWN_NN_AUDIT_PASS"
        or ra["source_hash"] != a.rehearsal_hash
        or ra["counts"]["native"] != r["actual_native_steps"]
        or ra["counts"]["neural"] != r["actual_neural_calls"]
        or ra["worlds"] != 20
        or not ra["independent_whole_world_replay"]
        or ta["status"] != "INHERITED_RUN_EXPERT_ALL_TRIAL_WHOLE_NATIVE_OWN_NN_IK_AUDIT_PASS"
        or ta["source_hash"] != a.teacher_hash
        or ta["counts"]["native"] != t["actual_native_steps"]
        or ta["counts"]["neural"] != t["actual_neural_calls"]
        or ta["worlds"] != 28
        or not ta["independent_whole_world_replay"]
        or q["status"] != "DUAL256_INITIAL_NATIVE_REFERENCE_STANDING_EQUIVALENCE_PASS_NOT_TRAINED"
        or q["actual_native_steps"] != 15560
    ):
        raise ValueError("full own-NN native source witnesses required")
    declarations = {}
    for name in ("rehearsal", "teacher", "qualification"):
        path = roots[name] / "declaration.json"
        if digest(path) != reports[name]["declaration_hash"]:
            raise ValueError("source declaration commitment changed")
        pins[str(path)] = digest(path)
        declarations[name] = json.loads(path.read_text())
    lineage = declarations["rehearsal"]["parent_lineage_hash"]
    if (
        lineage != declarations["teacher"]["motor_lineage_hash"]
        or lineage != declarations["qualification"]["parent_lineage_hash"]
    ):
        raise ValueError("same unchanged canonical parent required")
    paths = [Path(f) for f, h in pins.items() if h == lineage and f.endswith("/complete.json")]
    if len(paths) != 1:
        raise ValueError("unique canonical parent lineage required")
    training = json.loads(paths[0].read_text())
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir()
    queries = 0
    try:
        import numpy as np
        import torch

        from rosclaw_soccer.rsi.paid_progressive_student import load_paid_student
        from rosclaw_soccer.rsi.paid_research_setup import open_paid_bench
        from rosclaw_soccer.rsi.run_teacher_objective import run_teacher_rank

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        case = declarations["teacher"]["source_case"]
        _, policy, scene = open_paid_bench(pins, [case])
        parent, _ = load_paid_student(
            policy,
            checkpoint=Path(training["checkpoint_path"]),
            checkpoint_hash=training["checkpoint_hash"],
            training_receipt=paths[0],
            training_receipt_hash=lineage,
            scene_hash=scene,
        )
        episodes = [("parent", roots["rehearsal"], x) for x in r["records"]]
        for ci in range(2):
            row = next(x for x in t["records"] if x["tag"] == f"selected-c{ci}-repeat0")
            if not run_teacher_rank(row)[1] or row["result"]["forbidden"]:
                raise ValueError("novel target must be actually stable useful and no-nonfoot")
            episodes.append(("relative_improvement", roots["teacher"], row))
        records = []
        for index, (kind, root, row) in enumerate(episodes):
            blobs = {}
            for name, art in row["artifacts"].items():
                path = root / art["path"]
                if (
                    path.is_symlink()
                    or not path.resolve().is_relative_to(root.resolve())
                    or digest(path) != art["hash"]
                ):
                    raise ValueError("confined actual sequence primitives required")
                pins[str(path)] = art["hash"]
                with np.load(path, allow_pickle=False) as z:
                    blobs[name] = {k: z[k].copy() for k in z.files}
            streams = [blobs["kick"]] + ([blobs["stand"]] if "stand" in blobs else [])
            obs = np.concatenate([x["NN_obs"] for x in streams])
            actions = np.concatenate([x["NN_action"] for x in streams])
            hb = np.concatenate([x["NN_h_before"] for x in streams])
            cb = np.concatenate([x["NN_c_before"] for x in streams])
            if (
                obs.shape != (len(actions), 160)
                or actions.shape[1:] != (29,)
                or not 1 <= len(obs) <= 400
            ):
                raise ValueError("one actual unpadded reference plus observed standing required")
            state = (torch.zeros(2, 1, 128), torch.zeros(2, 1, 128))
            nominal = []
            prior = []
            for i, actual in enumerate(obs):
                if not np.array_equal(state[0].numpy(), hb[i]) or not np.array_equal(
                    state[1].numpy(), cb[i]
                ):
                    raise ValueError(
                        "independently replayed reference history differs from actual source"
                    )
                x = torch.from_numpy(actual.copy()).reshape(1, 1, 160)
                with torch.no_grad():
                    mean, state, hidden = parent.base.forward_with_features(x, state)
                    features = torch.cat(
                        ((x - parent.base.mean) / parent.base.denominator, hidden), dim=-1
                    )
                    delta = parent.residual(features.reshape(1, 288))
                queries += 1
                n = mean[0, 0].numpy().copy()
                d = delta[0].numpy().copy()
                if kind == "parent" and not np.array_equal(
                    n + d / parent.get_buffer("action_scale").numpy(), actions[i]
                ):
                    raise ValueError("rehearsal must label the actual frozen parent action")
                nominal.append(n)
                prior.append(d)
            nominal = np.stack(nominal)
            prior = np.stack(prior)
            scale = parent.get_buffer("action_scale").numpy().copy()
            if (
                not np.isfinite(actions).all()
                or np.abs((actions - nominal) * scale).max() > 0.20002
            ):
                raise ValueError("supervision outside unchanged deterministic motor authority")
            role = (
                "acquired_parent_skill"
                if kind == "parent" and row["healthy_quality"]
                else "negative_parent_task"
                if kind == "parent"
                else "novel_task_success"
                if row["healthy_quality"]
                else "novel_partial_skill_task_still_failed"
            )
            path = a.output / f"sequence{index}.npz"
            np.savez_compressed(
                path,
                observations=obs,
                targets=actions,
                reference_nominal=nominal,
                reference_delta=prior,
                action_scale=scale,
            )
            records.append(
                dict(
                    index=index,
                    role=role,
                    bc_weight=4.0
                    if role == "acquired_parent_skill"
                    else 0.0
                    if role == "negative_parent_task"
                    else 1.0,
                    task_success=row["healthy_quality"],
                    rows=len(obs),
                    source_kind=kind,
                    source_case=row.get("case", row.get("course")),
                    raw_path=path.name,
                    raw_hash=digest(path),
                )
            )
            print(
                json.dumps(
                    dict(sequence=index, role=role, rows=len(obs), reference_queries=queries)
                ),
                flush=True,
            )
        check_pins(pins)
        clean()
        write_once(
            a.output / "complete.json",
            dict(
                status="DUAL_MOTOR_ACTUAL_HISTORY_SEQUENCES_READY_NOT_TRAINED",
                source_head=a.expected_head,
                input_file_hashes=pins,
                parent_lineage_hash=lineage,
                qualification_hash=a.qualification_hash,
                teacher_hash=a.teacher_hash,
                rehearsal_hash=a.rehearsal_hash,
                records=records,
                own_reference_queries=queries,
                observation_width=160,
                learner_hidden_width=256,
                teacher_hidden_state_supplied_to_learner=False,
                whole_actual_sequences=True,
                DEV_fit_rows=0,
                Fresh_fit_rows=0,
                optimizer_updates=0,
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
                own_reference_queries=queries,
                physical_steps=0,
                automatic_retry=False,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
