"""Admit ONLY audited competent corrections on actual failed learner histories."""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("base", "recovery", "audit"):
        p.add_argument("--" + n + "-root", type=Path, required=True)
        p.add_argument("--" + n + "-hash", required=True)
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
            raise ValueError("clean frozen actual DAgger builder required")

    clean()
    if a.output.exists():
        raise ValueError("unused sequence output required")
    reports, pins = {}, {}
    for n in ("base", "recovery", "audit"):
        root = getattr(a, n + "_root")
        path = root / "complete.json"
        if (
            path.is_symlink()
            or digest(path) != getattr(a, n + "_hash")
            or (root / "failed.json").exists()
        ):
            raise ValueError("exact closed source required")
        report = json.loads(path.read_text())
        if (
            report["hardware_authorized"] is not False
            or report["promotion_authorized"] is not False
        ):
            raise ValueError("nonpromoted SIM sources only")
        for f, h in report["input_file_hashes"].items():
            if f in pins and pins[f] != h:
                raise ValueError("conflicting immutable source lineages")
            pins[f] = h
        pins[str(path)] = digest(path)
        reports[n] = report
    base, r, audit = [reports[n] for n in ("base", "recovery", "audit")]
    if (
        base["status"] != "DUAL_MOTOR_ACTUAL_HISTORY_SEQUENCES_READY_NOT_TRAINED"
        or len(base["records"]) != 22
        or audit["status"] != "DUAL_MOTOR_ASSISTED24_OWN_NATIVE_AUDIT_PASS"
        or audit["source_hash"] != a.recovery_hash
        or audit["worlds"] != 24
        or audit["teacher_mode"] != "measured_contact_center"
        or not audit["independent_whole_world_replay"]
        or audit["counts"]["native"] != r["actual_native_steps"]
        or audit["counts"]["neural"] != r["actual_actor_decisions"]
        or r["PPO_actor_rows"] != 0
        or r["learner_success_claim"] is not False
    ):
        raise ValueError("whole actual teacher correction witness required")
    selected = [x for x in r["records"] if x["tag"] in r["selected_tags"] and x["healthy_quality"]]
    if [x["source_context"] for x in selected] != [5, 21]:
        raise ValueError("this preregistered revision admits exactly corrected contexts5/21")
    for x in selected:
        witnesses = [
            v for v in audit["results"] if v["context"] == x["source_context"] and v["quality"]
        ]
        if not any(v["tag"] == x["tag"] for v in witnesses) or not all(
            any(v["tag"] == f"selected-context{x['source_context']}-repeat{i}" for v in witnesses)
            for i in (0, 1)
        ):
            raise ValueError("competent teacher and both exact repetitions must independently pass")
    lineage = base["parent_lineage_hash"]
    paths = [Path(f) for f, h in pins.items() if h == lineage and f.endswith("/complete.json")]
    if len(paths) != 1:
        raise ValueError("unique original frozen reference required")
    parent_record = json.loads(paths[0].read_text())
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(x): digest(x) for x in folder.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir()
    queries = 0
    try:
        import numpy as np
        import torch

        from rosclaw_soccer.rsi.paid_progressive_student import load_paid_student
        from rosclaw_soccer.rsi.paid_research_setup import open_paid_bench

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        declpath = a.recovery_root / "declaration.json"
        if digest(declpath) != r["declaration_hash"]:
            raise ValueError("exact source fixture declaration required")
        decl = json.loads(declpath.read_text())
        pins[str(declpath)] = digest(declpath)
        _, policy, scene = open_paid_bench(pins, [x["case"] for x in decl["contexts"]])
        parent, _ = load_paid_student(
            policy,
            checkpoint=Path(parent_record["checkpoint_path"]),
            checkpoint_hash=parent_record["checkpoint_hash"],
            training_receipt=paths[0],
            training_receipt_hash=lineage,
            scene_hash=scene,
        )
        records = []
        for row in base["records"]:
            src = a.base_root / row["raw_path"]
            if (
                src.is_symlink()
                or not src.resolve().is_relative_to(a.base_root.resolve())
                or digest(src) != row["raw_hash"]
            ):
                raise ValueError("exact retained original whole sequence required")
            pins[str(src)] = row["raw_hash"]
            path = a.output / row["raw_path"]
            path.write_bytes(src.read_bytes())
            if digest(path) != row["raw_hash"]:
                raise ValueError("retained sequence bytes changed")
            records.append(dict(row))
        for source in selected:
            streams = []
            for name in ("kick", "stand"):
                art = source["artifacts"][name]
                path = a.recovery_root / art["path"]
                if (
                    path.is_symlink()
                    or not path.resolve().is_relative_to(a.recovery_root.resolve())
                    or digest(path) != art["hash"]
                ):
                    raise ValueError("complete competent teacher primitives required")
                pins[str(path)] = art["hash"]
                with np.load(path, allow_pickle=False) as z:
                    streams.append(
                        {
                            k: z[k].copy()
                            for k in ("NN_obs", "NN_action", "NN_h_before", "NN_c_before")
                        }
                    )
            obs = np.concatenate([x["NN_obs"] for x in streams])
            targets = np.concatenate([x["NN_action"] for x in streams])
            hb = np.concatenate([x["NN_h_before"] for x in streams])
            cb = np.concatenate([x["NN_c_before"] for x in streams])
            if (
                obs.shape != (len(targets), 160)
                or hb.shape != (len(obs), 2, 1, 256)
                or cb.shape != hb.shape
                or not 1 <= len(obs) <= 400
            ):
                raise ValueError("whole real dual-state history required, not hidden-state labels")
            state = (torch.zeros(2, 1, 128), torch.zeros(2, 1, 128))
            nominal, prior = [], []
            for i, row in enumerate(obs):
                if not np.array_equal(state[0].numpy(), hb[i, :, :, :128]) or not np.array_equal(
                    state[1].numpy(), cb[i, :, :, :128]
                ):
                    raise ValueError("own reference recomputation differs from actual observer")
                x = torch.from_numpy(row).reshape(1, 1, 160)
                with torch.no_grad():
                    mean, state, hidden = parent.base.forward_with_features(x, state)
                    features = torch.cat(
                        ((x - parent.base.mean) / parent.base.denominator, hidden), dim=-1
                    )
                    delta = parent.residual(features.reshape(1, 288))
                queries += 1
                nominal.append(mean[0, 0].numpy().copy())
                prior.append(delta[0].numpy().copy())
            nominal, prior = np.stack(nominal), np.stack(prior)
            scale = parent.get_buffer("action_scale").numpy().copy()
            if (
                not np.isfinite(targets).all()
                or np.abs((targets - nominal) * scale).max() > 0.20002
            ):
                raise ValueError("unchanged motor authority required")
            index = len(records)
            path = a.output / f"sequence{index}.npz"
            np.savez_compressed(
                path,
                observations=obs,
                targets=targets,
                reference_nominal=nominal,
                reference_delta=prior,
                action_scale=scale,
            )
            records.append(
                dict(
                    index=index,
                    role="actual_state_competent_teacher",
                    bc_weight=1.0,
                    task_success=True,
                    rows=len(obs),
                    source_kind="actual_state_dagger",
                    source_context=source["source_context"],
                    source_tag=source["tag"],
                    raw_path=path.name,
                    raw_hash=digest(path),
                    learner_success_claim=False,
                    PPO_actor_rows=0,
                )
            )
            print(
                json.dumps(
                    dict(
                        sequence=index,
                        source_context=source["source_context"],
                        rows=len(obs),
                        own_reference_queries=queries,
                    )
                ),
                flush=True,
            )
        check_pins(pins)
        clean()
        write_once(
            a.output / "complete.json",
            dict(
                status="DUAL_MOTOR_ACTUAL_DAGGER_SEQUENCES_READY_NOT_TRAINED",
                source_head=a.expected_head,
                input_file_hashes=pins,
                base_data_hash=a.base_hash,
                recovery_hash=a.recovery_hash,
                recovery_audit_hash=a.audit_hash,
                parent_lineage_hash=lineage,
                teacher_hash=base["teacher_hash"],
                records=records,
                own_new_reference_queries=queries,
                original_sequences_retained=22,
                new_corrections=2,
                observation_width=160,
                learner_hidden_width=256,
                teacher_hidden_state_supplied_to_learner=False,
                whole_actual_sequences=True,
                DEV_fit_rows=0,
                Fresh_fit_rows=0,
                optimizer_updates=0,
                physical_steps=0,
                PPO_actor_rows=0,
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
                automatic_retry=False,
                physical_steps=0,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
