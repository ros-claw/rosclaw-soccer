"""Bounded local objective-gradient diagnosis, no optimizer/physics/promotion.

Compare ALL138 positive histories at before75024 and preregistered first-seed
after85023. Report teacher/retention group tradeoffs and every original
co-success teacher/baseline task pair. Local gradient opposition is not proof
of infeasibility, physical causality or a globally optimal training change.
"""

import argparse
import json
from pathlib import Path


def gradient_relation(a, b):
    import numpy as np

    if (
        not isinstance(a, np.ndarray)
        or not isinstance(b, np.ndarray)
        or a.shape != b.shape
        or a.ndim != 1
        or not len(a)
        or any(not np.isfinite(x).all() for x in (a, b))
    ):
        raise ValueError("two same-size finite gradient vectors required")
    a, b = a.astype(np.float64), b.astype(np.float64)
    x, y = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    dot = float(a @ b)
    return dict(
        left_norm=x,
        right_norm=y,
        dot=dot,
        cosine=None if x == 0 or y == 0 else float(np.clip(dot / (x * y), -1.0, 1.0)),
        strictly_opposed=dot < 0,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("training-root", "cpu-own-root", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("training-hash", "cpu-own-hash", "expected-head"):
        p.add_argument("--" + key, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, receipt, source_pins

    clean(a.expected_head)
    paths = [a.training_root / "complete.json", a.cpu_own_root / "complete.json"]
    if (
        a.output.exists()
        or [digest(x) for x in paths] != [a.training_hash, a.cpu_own_hash]
        or any(x.is_symlink() or (x.parent / "failed.json").exists() for x in paths)
    ):
        raise ValueError("unused output and exact complete trained/own CPU witnesses required")
    trained, own = [json.loads(x.read_text()) for x in paths]
    if (
        own["status"] != "DUAL_RECURRENT_TRAINED_CHECKPOINT_CPU_OWN_HISTORY_AUDIT_PASS_NOT_PHYSICS"
        or own["checkpoint_kind"] != "team-pass-corrected-v2"
        or own["source_hash"] != a.training_hash
        or own["checkpoint_hash"] != trained["checkpoint_hash"]
        or trained["config"]["seed"] != 85023
    ):
        raise ValueError("independently CPU-replayed preregistered first-seed model required")
    pins = dict(own["input_file_hashes"])
    pins.update({str(x): digest(x) for x in paths})
    bp, bank = receipt(pins, trained["data_hash"])
    wp, warm = receipt(pins, trained["warm_start_hash"])
    from rosclaw_soccer.rsi.single_mode_pass_curriculum import single_mode_pass_curriculum
    from rosclaw_soccer.rsi.team_pass_correction import correction_pools

    old, teachers, new = correction_pools(bank["records"])
    selection = single_mode_pass_curriculum(bank["records"])
    active = sorted(old + teachers + new)
    if len(set(active)) != len(active) or len(active) != 138 or warm["config"]["seed"] != 75024:
        raise ValueError("complete non-overlapping positive training pools required")
    pairs = []

    def task(r):
        return r["source_course"]["state_id"], r["source_course"]["task"]

    originals = {task(bank["records"][i]): i for i in new}
    for i in teachers:
        r = bank["records"][i]
        if r["role"] == "new_teacher" and task(r) in originals:
            pairs.append(
                dict(
                    state_id=task(r)[0],
                    task=task(r)[1],
                    teacher_index=i,
                    anchor_index=originals[task(r)],
                )
            )
    if not 1 <= len(pairs) <= 80:
        raise ValueError("bounded ALL original co-success pairs required")
    for row in bank["records"]:
        path = bp.parent / row["raw_path"]
        if path.is_symlink() or digest(path) != row["raw_hash"]:
            raise ValueError("all274 actual histories including negatives required unchanged")
        pins[str(path)] = row["raw_hash"]
    source_pins(pins)
    check_pins(pins)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            protocol="soccer.local_pass_objective_gradient_tradeoff.v2",
            source_head=a.expected_head,
            input_file_hashes=pins,
            training_hash=a.training_hash,
            cpu_own_hash=a.cpu_own_hash,
            models=[75024, 85023],
            positive_histories=138,
            co_success_pairs=pairs,
            single_mode_comparison=selection,
            maximum_gradient_queries=276,
            optimizer_updates=0,
            physical_steps=0,
            first_seed_not_best_seed_selection=True,
            local_diagnosis_not_physical_proof=True,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    queries, controls = 0, 0
    reports = []
    try:
        import numpy as np
        import torch

        from rosclaw_soccer.rsi.contact_consolidation import sequence_loss
        from rosclaw_soccer.rsi.dual_onpolicy_checkpoint import load_dual_candidate_for_sim_audit
        from scripts.rsi_stroke_search_common import open_motor

        contexts = (
            {int(k): v for k, v in bank["contexts"].items()}
            if isinstance(bank["contexts"], dict)
            else dict(enumerate(bank["contexts"]))
        )
        bench, after = open_motor(
            pins,
            a.training_hash,
            {i: contexts[i] for i in (1, 4, 20, 21)},
            checkpoint_kind="team-pass-corrected-v2",
        )
        before, _ = load_dual_candidate_for_sim_audit(
            after.parent,
            parent_lineage_hash=warm["parent_lineage_hash"],
            root=wp.parent,
            expected_receipt_hash=trained["warm_start_hash"],
            checkpoint_kind="team-pass-consolidated-v1",
        )
        torch.use_deterministic_algorithms(True)
        episodes = {}
        for i in active:
            row = bank["records"][i]
            with np.load(bp.parent / row["raw_path"], allow_pickle=False) as z:
                episodes[i] = {k: torch.from_numpy(z[k].copy()) for k in z.files}
        for seed, model in ((75024, before), (85023, after)):
            snapshot = {k: v.clone() for k, v in model.state_dict().items()}
            model.train()
            model.plastic.requires_grad_(True)
            parameters = list(model.plastic.parameters())
            grads, losses = {}, {}
            for i in active:
                loss = sequence_loss(model.plastic, episodes[i])
                g = torch.autograd.grad(loss, parameters, create_graph=False, retain_graph=False)
                vector = torch.cat([x.detach().flatten() for x in g]).double().numpy().copy()
                if not np.isfinite(vector).all() or not torch.isfinite(loss):
                    raise ValueError("finite actual whole-history gradients required")
                queries += 1
                controls += len(episodes[i]["observations"])
                grads[i] = vector
                losses[i] = float(loss.detach())
            means = {
                k: np.mean([grads[i] for i in ids], axis=0)
                for k, ids in (("teacher", teachers), ("old", old), ("new", new))
            }
            co = [
                r | gradient_relation(grads[r["teacher_index"]], grads[r["anchor_index"]])
                for r in pairs
            ]
            critical = []
            for i in old:
                row = bank["records"][i]
                critical.append(
                    dict(
                        index=i,
                        context=row["source_context"],
                        role=row["role"],
                        loss=losses[i],
                        **gradient_relation(means["teacher"], grads[i]),
                    )
                )
            if any(p.grad is not None for p in model.parameters()) or any(
                not torch.equal(v, snapshot[k]) for k, v in model.state_dict().items()
            ):
                raise ValueError("read-only gradient diagnosis may not update any tensor or .grad")
            report = dict(
                seed=seed,
                single_mode_comparison=dict(
                    selection_hash=selection["selection_hash"],
                    fit_histories=len(selection["fit_indices"]),
                    teacher_histories=len(selection["selected_teacher_indices"]),
                    teacher_vs_current17=gradient_relation(
                        np.mean([grads[i] for i in selection["selected_teacher_indices"]], axis=0),
                        np.mean([grads[i] for i in selection["current17_anchor_indices"]], axis=0),
                    ),
                    teacher_vs_baseline23=gradient_relation(
                        np.mean([grads[i] for i in selection["selected_teacher_indices"]], axis=0),
                        means["new"],
                    ),
                ),
                teacher_vs_all34_retention=gradient_relation(means["teacher"], means["old"]),
                teacher_vs_all23_pass_anchors=gradient_relation(means["teacher"], means["new"]),
                co_success_pairs=co,
                co_success_opposed=sum(x["strictly_opposed"] for x in co),
                critical_skill_gradients=critical,
                group_mean_loss={
                    k: float(np.mean([losses[i] for i in ids]))
                    for k, ids in (("teacher", teachers), ("old", old), ("new", new))
                },
            )
            reports.append(report)
            write_once(a.output / f"model{seed}.json", report)
            print(
                json.dumps(
                    {
                        k: v
                        for k, v in report.items()
                        if k not in ("co_success_pairs", "critical_skill_gradients")
                    }
                ),
                flush=True,
            )
            model.eval().requires_grad_(False)
        if queries != 276 or bench.native_steps:
            raise ValueError("whole bounded zero-physics diagnosis required")
        check_pins(pins)
        clean(a.expected_head)
        write_once(
            a.output / "complete.json",
            dict(
                status="PASS_CURRICULUM_GRADIENT_TRADEOFF_DIAGNOSED_NOT_PHYSICAL_PROOF",
                source_head=a.expected_head,
                input_file_hashes=pins,
                reports=reports,
                actual_gradient_queries=queries,
                actual_controls_differentiated=controls,
                optimizer_updates=0,
                physical_steps=0,
                local_gradient_not_global_infeasibility=True,
                goal_complete=False,
                hardware_authorized=False,
                promotion_authorized=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failed.json",
            dict(
                status="GRADIENT_DIAGNOSIS_FAILED_PRESERVE_NO_RETRY",
                error_type=type(error).__name__,
                error=str(error),
                actual_gradient_queries=queries,
                actual_controls_differentiated=controls,
                optimizer_updates=0,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
