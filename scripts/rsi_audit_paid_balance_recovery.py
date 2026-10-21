"""Audit every active-balance branch, CEM update and selected continuation.

No physics is executed. Qualified exact parent arrays reuse their prior proof;
any NEW third-skill own-network inference is independently recomputed/counted.
"""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    p = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "source-root", "source-audit-root", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("hash", "source-hash", "source-audit-hash", "expected-head"):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    rp = a.root / "complete.json"
    if (
        a.output.exists()
        or digest(rp) != a.hash
        or (a.root / "failed.json").exists()
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
    ):
        raise ValueError("closed exact source, unused output and clean frozen auditor required")
    r, d = [json.loads((a.root / n).read_text()) for n in ("complete.json", "declaration.json")]
    from rosclaw_soccer.rsi.paid_recovery_proof import recovery_proof

    sd, source_pins = recovery_proof(
        a.source_root,
        a.source_hash,
        a.source_audit_root,
        a.source_audit_hash,
        cases=d["source_cases"],
    )
    source = json.loads((a.source_root / "complete.json").read_text())
    if (
        r["status"] != "PAID_ACTIVE_BALANCE_TEACHER_PILOT_CLOSED_PENDING_AUDIT"
        or r["pilot_cases"] != [0, 15]
        or d["pilot_cases"] != [0, 15]
        or d["source_head"] != r["source_head"]
        or d["generations"] != 4
        or d["population"] != 24
        or d["elite"] != 6
        or d["latent_shape"] != [9, 4]
        or d["seed"] not in (20261112, 20261113, 20261114)
        or d["new_phase_controls"] != 100
        or d["max_delta_rad"] != 0.2
        or d["max_delta_per_control_rad"] != 0.02
        or r["CEM_distribution_updates"] != 4
        or r["neural_optimizer_updates"] != 0
        or not r["all_selected_recovery_and_third_repeats_exact"]
    ):
        raise ValueError("exact bounded two-state teacher protocol required")
    pins = dict(r["input_file_hashes"])
    if any(pins.get(path) != h for path, h in source_pins.items()):
        raise ValueError("all parent proofs must be bound by producer")
    for relative, h in r["artifact_hashes"].items():
        path = a.root / relative
        if (
            path.is_symlink()
            or not path.resolve().is_relative_to(a.root.resolve())
            or digest(path) != h
        ):
            raise ValueError("search artifact changed or escaped evidence root")
        pins[str(path)] = h
    actual = {str(f.relative_to(a.root)) for f in a.root.rglob("*") if f.is_file() and f != rp}
    if actual != set(r["artifact_hashes"]):
        raise ValueError("complete search artifact coverage required")
    pins[str(rp)] = a.hash
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(input_file_hashes=pins, own_native_steps=0, source_head=a.expected_head),
    )
    counts = dict(
        producer_native=0,
        producer_neural=0,
        own_neural=0,
        recovery_segments=0,
        exact_parent_skills=0,
    )
    try:
        import numpy as np
        import torch

        from rosclaw_soccer.rsi.paid_balance_audit import audit_balance_recovery
        from rosclaw_soccer.rsi.paid_contact_residual_actor import load_contact_residual_actor
        from rosclaw_soccer.rsi.paid_continuity_audit import (
            audit_terminal,
            identical,
            snapshot_fields,
        )
        from rosclaw_soccer.rsi.paid_recovery_proof import unique_proof_path
        from rosclaw_soccer.rsi.paid_research_setup import open_paid_auditor

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        auditor, motions = open_paid_auditor(pins, d["source_cases"])
        used = set()

        def read(row, root=a.root):
            path = root / row["raw_path"]
            if (
                path.is_symlink()
                or not path.resolve().is_relative_to(root.resolve())
                or digest(path) != row["raw_hash"]
            ):
                raise ValueError("exact confined trajectory required")
            if root == a.root:
                used.add(row["raw_path"])
            with np.load(path, allow_pickle=False) as z:
                return {k: z[k].copy() for k in z.files}

        def artifact(path):
            return dict(raw_path=path, raw_hash=r["artifact_hashes"][path])

        entries, anchors, actions = {}, {}, {}
        if [p["case"] for p in r["parents"]] != [0, 15, 0, 0, 15, 15]:
            raise ValueError("all two initial plus four fresh selected parents required")
        for parent in r["parents"]:
            i = parent["case"]
            old = source["results"]["learned"]["worlds"][i]
            refs = {Path(x["raw_path"]).stem: x for x in old["artifacts"]}
            parent_data = {Path(x["raw_path"]).stem: read(x) for x in parent["artifacts"]}
            drop = read(refs["between1-2-drop"], a.source_root)
            entry = snapshot_fields(drop, "after_" if bool(drop["performed"]) else "before_")
            for key, z in parent_data.items():
                if key == "recovery-entry":
                    identical(z, entry)
                else:
                    identical(z, read(refs[key], a.source_root))
            if set(parent_data) != set(refs) - {
                "skill2",
                "terminal2",
                "between1-2-hold",
                "between1-2-boundary",
            } | {"recovery-entry"}:
                raise ValueError("exact complete parent prefix artifact set required")
            for k in range(2):
                z = parent_data[f"skill{k}"]
                counts["producer_native"] += len(z["target"]) * 10
                counts["producer_neural"] += len(z["NN_action"])
                counts["exact_parent_skills"] += 1
            if "between0-1-hold" in parent_data:
                counts["producer_native"] += len(parent_data["between0-1-hold"]["target"]) * 10
            entries[i] = entry
            anchors[i] = parent_data["skill1"]["target"][-1]
            actions[i] = parent_data["skill1"]["NN_action"][-1]

        def segment(row, i, latent):
            z = read(row)
            checked = audit_balance_recovery(
                auditor,
                z,
                entry=entries[i],
                anchor=anchors[i],
                anchor_action=actions[i],
                latent=latent,
            )
            if any(row.get(k) != v for k, v in checked.items()):
                raise ValueError("recorded recovery score differs from primitive evidence")
            counts["producer_native"] += checked["native_steps"]
            counts["recovery_segments"] += 1
            return checked, z

        def rank(rows):
            return (
                sum(r["healthy"] for r in rows),
                sum(r["recovery_stability_observed"] for r in rows),
                min(r["controls"] for r in rows),
                -sum(r["cost"] for r in rows),
            )

        baseline = [
            segment(row, i, np.zeros((9, 4)))[0]
            for i, row in zip((0, 15), r["baseline"], strict=True)
        ]
        for i, row in zip((0, 15), r["baseline"], strict=True):
            source_world = source["results"]["learned"]["worlds"][i]
            hold = next(
                (
                    x
                    for x in source_world["artifacts"]
                    if Path(x["raw_path"]).stem == "between1-2-hold"
                ),
                None,
            )
            if hold is not None:
                old, actual = read(hold, a.source_root), read(row)
                if any(
                    not np.array_equal(actual[k][: len(value)], value)
                    for k, value in old.items()
                    if k != "post_reference_hold"
                ):
                    raise ValueError("zero active recovery changed the original held prefix")
        best_z = mean = np.zeros((9, 4))
        std = np.full((9, 4), 0.5)
        best_rank = rank(baseline)
        rng = np.random.default_rng(d["seed"])
        for generation in range(4):
            g = json.loads((a.root / f"generation{generation}/complete.json").read_text())
            zs = [best_z.copy()] + list(rng.normal(mean, std, size=(23, 9, 4)))
            if len(g["candidates"]) != 24:
                raise ValueError("all CEM candidate branches required")
            for z, c in zip(zs, g["candidates"], strict=True):
                latent = np.tanh(z)
                if not np.array_equal(z, c["z"]) or not np.array_equal(latent, c["latent"]):
                    raise ValueError("declared seeded CEM proposal differs")
                rows = [
                    segment(row, i, latent)[0] for i, row in zip((0, 15), c["results"], strict=True)
                ]
                score = rank(rows)
                if list(score) != c["rank"]:
                    raise ValueError("CEM rank differs")
                if score > best_rank:
                    best_rank, best_z = score, z.copy()
            elite = np.stack(
                [
                    zs[j]
                    for j in sorted(
                        range(24), key=lambda j: tuple(g["candidates"][j]["rank"]), reverse=True
                    )[:6]
                ]
            )
            mean = 0.5 * mean + 0.5 * elite.mean(axis=0)
            std = np.clip(0.5 * std + 0.5 * elite.std(axis=0), 0.15, 1.2)
            if (
                not np.array_equal(mean, g["next_mean"])
                or not np.array_equal(std, g["next_std"])
                or list(best_rank) != g["best_rank"]
            ):
                raise ValueError("CEM update or selected frontier differs")
        if (
            not np.array_equal(np.tanh(best_z), r["selected_latent"])
            or list(best_rank) != r["selected_rank"]
        ):
            raise ValueError("final selected bounded controller differs")
        policy = next(Path(p) for p in pins if p.endswith("/ckp/policy_30000.onnx"))
        scene = next(
            h for p, h in pins.items() if p.endswith("/cached-standard/declared-scene.xml")
        )
        actor, _ = load_contact_residual_actor(
            policy,
            checkpoint=unique_proof_path(pins, sd["checkpoint_hash"]),
            checkpoint_hash=sd["checkpoint_hash"],
            training_receipt=unique_proof_path(pins, sd["training_receipt_hash"]),
            training_receipt_hash=sd["training_receipt_hash"],
            scene_hash=scene,
        )
        for i, replicas, selected in zip((0, 15), r["final"], r["selected_results"], strict=True):
            repeats = []
            for result in replicas:
                checked, z = segment(result["recovery"], i, np.tanh(best_z))
                identical(z, read(selected))
                third = result["third"]
                q, v = z["qpos"][-1, -1], z["qvel"][-1, -1]
                expected = checked["healthy"] and q[38] <= 0.12 and abs(v[37]) <= 0.1
                if (third is not None) != expected:
                    raise ValueError("selected physical continuation branch differs")
                third_raw = None
                if third is not None:
                    path = str(Path(result["recovery"]["raw_path"]).parent / "third-boundary.npz")
                    boundary = read(artifact(path))
                    before = snapshot_fields(boundary, "before_")
                    audit_terminal(auditor, z, before)
                    if not np.array_equal(
                        boundary["previous_actual_action"], z["RECOVERY_actual_action"][-1]
                    ):
                        raise ValueError("actual previous recovery action not carried")
                    third_raw = read(third)
                    for k in before:
                        if "initial_" + k in third_raw and not np.array_equal(
                            third_raw["initial_" + k], before[k]
                        ):
                            raise ValueError("third skill resets actual integration state")
                    checked_third = auditor.audit(
                        third_raw,
                        third,
                        horizon=len(motions[third["source"]]["q"]),
                        previous_action=z["RECOVERY_actual_action"][-1],
                    )
                    counts["producer_native"] += checked_third["native_steps"]
                    counts["producer_neural"] += checked_third["neural_rows"]
                    state = (torch.zeros(2, 1, 128), torch.zeros(2, 1, 128))
                    with torch.no_grad():
                        for j, obs in enumerate(third_raw["NN_obs"]):
                            if any(
                                not np.array_equal(t.numpy(), third_raw[k][j])
                                for t, k in zip(state, ("NN_h_before", "NN_c_before"), strict=True)
                            ):
                                raise ValueError("third private causal memory differs")
                            action, state = actor(torch.from_numpy(obs).reshape(1, 1, 160), state)
                            if not np.array_equal(
                                action[0, 0].numpy(), third_raw["NN_action"][j]
                            ) or any(
                                not np.array_equal(t.numpy(), third_raw[k][j])
                                for t, k in zip(state, ("NN_h_after", "NN_c_after"), strict=True)
                            ):
                                raise ValueError("third own neural action/memory differs")
                            counts["own_neural"] += 1
                repeats.append((z, third_raw))
            identical(repeats[0][0], repeats[1][0])
            if repeats[0][1] is not None:
                identical(repeats[0][1], repeats[1][1])
        if (
            counts["producer_native"] != r["actual_native_steps"]
            or counts["producer_neural"] != r["actual_neural_rows"]
            or used != {p for p in r["artifact_hashes"] if p.endswith(".npz")}
        ):
            raise ValueError("complete work or physical artifact coverage differs")
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="PAID_ACTIVE_BALANCE_ALL_PRIMITIVE_AND_CONTINUATION_AUDIT_PASS",
                input_file_hashes=pins,
                counts=counts,
                own_native_steps=0,
                own_neural_rows=counts["own_neural"],
                goal_complete=False,
                hardware_authorized=False,
                promotion_authorized=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failed.json",
            dict(error=repr(error), counts=counts, own_native_steps=0, hardware_authorized=False),
        )
        raise


if __name__ == "__main__":
    main()
