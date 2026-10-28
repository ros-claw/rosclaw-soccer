"""Fit a small SIM event detector and evaluate held-out source contexts.

Uses exact simulator state (including teammate torso), not camera observations.
This is an offline diagnostic, not football policy training or promotion.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np
import torch

from rosclaw_soccer.rsi.causal_return_event_learning import (
    causal_features,
    classify_first,
    event_labels,
)
from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean
from scripts.rsi_review_relay_balance_counterfactual import bind, report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("trial", "scene", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("trial-hash", "scene-hash", "expected-head"):
        p.add_argument("--" + key, required=True)
    p.add_argument(
        "--original-source",
        nargs=3,
        action="append",
        default=[],
        metavar=("SOURCE_ROOT", "AUDIT_RECEIPT", "AUDIT_HASH"),
    )
    a = p.parse_args()
    clean(a.expected_head)
    receipt = a.trial / "complete.json"
    if (
        a.output.exists()
        or a.output.resolve().is_relative_to(Path.cwd())
        or receipt.is_symlink()
        or a.scene.is_symlink()
        or digest(receipt) != a.trial_hash
        or digest(a.scene) != a.scene_hash
    ):
        raise ValueError("unused external output and exact immutable input hashes required")
    source = json.loads(receipt.read_text())
    if (
        source.get("status") != "RELAY_NONDEFAULT_SMOOTH_RETURN_PHYSICAL_CANDIDATE_NOT_SUCCESS"
        or source.get("smooth_transition_duration") != 10
        or len(source.get("rows", [])) != 24
        or any(
            source.get(k) is not False
            for k in (
                "independent_fresh",
                "hardware_authorized",
                "promotion_authorized",
                "goal_complete",
            )
        )
    ):
        raise ValueError("unpromoted complete 10-control development trial required")
    model = mujoco.MjModel.from_xml_path(str(a.scene))
    passer = NativeVelocityLayout(
        model, robot_prefix="a/", ball_body_name="soccer_ball", player_count=2
    )
    returner = NativeVelocityLayout(
        model, robot_prefix="b/", ball_body_name="soccer_ball", player_count=2
    )
    cases = []
    trajectory_pins = []
    original_pins = []
    for row in source["rows"]:
        if not row["receiving_ready"]:
            continue
        path = a.trial / row["physical_trajectory"]
        if (
            path.name != row["physical_trajectory"]
            or path.is_symlink()
            or digest(path) != row["physical_trajectory_hash"]
        ):
            raise ValueError("bound physical trajectory required")
        with np.load(path, allow_pickle=False) as data:
            n = row["receiver_velocity_controls"]
            x = causal_features(
                data["qpos"],
                data["qvel"],
                start=n,
                ball_q=passer.bq,
                ball_v=passer.bv,
                passer_q=passer.rq,
                passer_v=passer.rv,
                returner_q=returner.rq,
                returner_v=returner.rv,
            )
            y, onset, censored = event_labels(data["ball_forces"], start=n)
        if digest(path) != row["physical_trajectory_hash"] or len(x) != len(y):
            raise ValueError("unchanged and aligned case required")
        trajectory_pins.append((path, row["physical_trajectory_hash"]))
        if not len(x):
            raise ValueError("empty return trajectory")
        cases.append(
            dict(
                collection="smooth10",
                index=row["index"],
                context=row["context"],
                x=x,
                y=y,
                onset=onset,
                censored=censored,
            )
        )
    for group_index, (root_name, audit_name, audit_hash) in enumerate(a.original_source):
        root, audit = Path(root_name), Path(audit_name)
        original, _, original_hash = report(
            root,
            audit,
            audit_hash,
            "NEURAL_RELAY_RECEIVER_LATERAL_AUTHORITY32_PENDING_OWN_REPLAY",
            worlds=32,
        )
        if digest(root / "scene.xml") != a.scene_hash:
            raise ValueError("all original trajectories must use the exact compiled scene")
        original_pins.extend(
            (
                (root / "complete.json", original_hash),
                (audit, audit_hash),
                (root / "scene.xml", a.scene_hash),
            )
        )
        for row in original["records"]:
            if row["handoff"] is None:
                continue
            path = bind(root, row, "physics")
            if path is None:
                raise ValueError("original physical trajectory required")
            expected = row["artifacts"]["physics"]["hash"]
            with np.load(path, allow_pickle=False) as data:
                n = row["handoff"]["control"]
                x = causal_features(
                    data["qpos"],
                    data["qvel"],
                    start=n,
                    ball_q=passer.bq,
                    ball_v=passer.bv,
                    passer_q=passer.rq,
                    passer_v=passer.rv,
                    returner_q=returner.rq,
                    returner_v=returner.rv,
                )
                y, onset, censored = event_labels(data["ball_forces"], start=n)
            if digest(path) != expected or len(x) != len(y):
                raise ValueError("unchanged original physical evidence required")
            trajectory_pins.append((path, expected))
            if len(x):
                cases.append(
                    dict(
                        collection=f"original-{group_index}",
                        index=row["index"],
                        context=row["course"]["context"],
                        x=x,
                        y=y,
                        onset=onset,
                        censored=censored,
                    )
                )
    groups = sorted({r["context"] for r in cases})
    if (
        len(cases) < 11
        or len(groups) < 3
        or len({(r["collection"], r["index"]) for r in cases}) != len(cases)
    ):
        raise ValueError("distinct observed cases over at least three contexts required")
    torch.set_num_threads(1)
    results = []
    models = []
    for holdout in groups:
        train = [r for r in cases if r["context"] != holdout and not r["censored"]]
        test = [r for r in cases if r["context"] == holdout]
        x = torch.from_numpy(np.concatenate([r["x"] for r in train]))
        y = torch.from_numpy(np.concatenate([r["y"] for r in train]))
        if not (torch.any(y == 0) and torch.any(y == 1)):
            raise ValueError("each training fold needs positive and negative states")
        mean = x.mean(0)
        scale = x.std(0).clamp(min=0.05)
        norm = (x - mean) / scale
        for seed in (11, 23, 37):
            torch.manual_seed(seed)
            net = torch.nn.Sequential(
                torch.nn.Linear(39, 32), torch.nn.Tanh(), torch.nn.Linear(32, 1)
            )
            optimizer = torch.optim.Adam(net.parameters(), lr=0.003)
            for _ in range(200):
                logits = net(norm).reshape(-1)
                loss0 = torch.nn.functional.softplus(logits[y == 0]).mean()
                loss1 = torch.nn.functional.softplus(-logits[y == 1]).mean()
                loss = (loss0 + loss1) / 2
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            per_case = []
            for r in test:
                with torch.no_grad():
                    logits = net((torch.from_numpy(r["x"]) - mean) / scale).reshape(-1).numpy()
                prediction = logits >= 0
                baseline = np.linalg.norm(r["x"][:, -3:], axis=1) >= 0.3
                per_case.append(
                    dict(
                        index=r["index"],
                        collection=r["collection"],
                        event_control=r["onset"],
                        censored=r["censored"],
                        model=classify_first(prediction, r["onset"], r["censored"]),
                        delta_baseline=classify_first(baseline, r["onset"], r["censored"]),
                        first_model=None
                        if not prediction.any()
                        else int(np.flatnonzero(prediction)[0]),
                        first_baseline=None
                        if not baseline.any()
                        else int(np.flatnonzero(baseline)[0]),
                    )
                )
            results.append(
                dict(
                    holdout_context=holdout,
                    seed=seed,
                    training_cases=len(train),
                    training_states=len(x),
                    test_cases=per_case,
                    final_loss=float(loss.detach()),
                )
            )
            models.append(
                (
                    holdout,
                    seed,
                    mean.numpy(),
                    scale.numpy(),
                    {k: v.detach().numpy() for k, v in net.state_dict().items()},
                )
            )
    if (
        digest(receipt) != a.trial_hash
        or digest(a.scene) != a.scene_hash
        or any(digest(path) != expected for path, expected in trajectory_pins)
        or any(digest(path) != expected for path, expected in original_pins)
    ):
        raise ValueError("training evidence changed")
    clean(a.expected_head)
    a.output.mkdir(parents=True)
    model_artifacts = []
    for group, seed, mean, scale, state in models:
        path = a.output / f"context-{group}-seed-{seed}.npz"
        np.savez_compressed(path, mean=mean, scale=scale, **state)
        model_artifacts.append(dict(context=group, seed=seed, name=path.name, hash=digest(path)))
    write_once(
        a.output / "complete.json",
        dict(
            status="OFFLINE_CONTEXT_HELDOUT_EVENT_DIAGNOSTIC_NOT_POLICY",
            source_head=a.expected_head,
            trial_hash=a.trial_hash,
            scene_hash=a.scene_hash,
            cases=len(cases),
            contexts=groups,
            original_source_audit_hashes=[x[2] for x in a.original_source],
            optimizer_updates=200 * len(results),
            seeds=[11, 23, 37],
            model_artifacts=model_artifacts,
            input="privileged_exact_sim_ball_and_both_torsos_three_control_history",
            results=results,
            model_outcome_counts=dict(
                Counter(r["model"] for fold in results for r in fold["test_cases"])
            ),
            baseline_outcome_counts=dict(
                Counter(r["delta_baseline"] for fold in results for r in fold["test_cases"])
            ),
            independent_fresh=False,
            physical_rollouts=0,
            promotion_authorized=False,
            hardware_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
