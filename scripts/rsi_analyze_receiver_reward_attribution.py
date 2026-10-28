"""Quantify ball-motion confounding in audited receiver SIM feedback."""

import argparse
import json
from pathlib import Path

import numpy as np

from rosclaw_soccer.rsi.receiving_progress_reward import decompose_proximity_change
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--dataset-hash", required=True)
    p.add_argument("--audit-hash", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    clean(a.expected_head)
    dp = a.dataset / "complete.json"
    if (
        a.output.exists()
        or a.output.resolve().is_relative_to(Path.cwd())
        or digest(dp) != a.dataset_hash
        or digest(a.audit) != a.audit_hash
    ):
        raise ValueError("pinned audited physical feedback and external output required")
    data, audit = json.loads(dp.read_text()), json.loads(a.audit.read_text())
    if (
        data.get("status") != "RECEIVER_DENSE_PHYSICAL_FEEDBACK_EXPORTED_NOT_TRAINED"
        or audit.get("status") != "RECEIVER_DENSE_FEEDBACK_INDEPENDENTLY_RECOMPUTED_NOT_TRAINED"
        or audit.get("dense_hash") != a.dataset_hash
        or len(data.get("records", [])) != 72
    ):
        raise ValueError("complete independently recomputed feedback required")
    rows, paths = [], []
    for row in data["records"]:
        path = a.dataset / row["path"]
        if (
            path.is_symlink()
            or path.resolve().parent != a.dataset.resolve()
            or digest(path) != row["hash"]
        ):
            raise ValueError("exact original feedback episode required")
        paths.append((path, row["hash"]))
        with np.load(path, allow_pickle=False) as f:
            parts = decompose_proximity_change(f["receiver_xy"], f["ball_xy"])
            receiver, ball = parts["receiver_motion"], parts["ball_motion"]
            if len(receiver) != row["controls"]:
                raise ValueError("aligned actual controls required")
            rows.append(
                dict(
                    context=row["context"],
                    side=row["side"],
                    seed=row["seed"],
                    controls=len(receiver),
                    receiver_net=float(receiver.sum()),
                    ball_net=float(ball.sum()),
                    receiver_positive=float(np.maximum(receiver, 0).sum()),
                    ball_positive=float(np.maximum(ball, 0).sum()),
                    receiver_absolute=float(np.abs(receiver).sum()),
                    ball_absolute=float(np.abs(ball).sum()),
                )
            )
    if sum(r["controls"] for r in rows) != 18903 or any(
        digest(p) != h for p, h in (*paths, (dp, a.dataset_hash), (a.audit, a.audit_hash))
    ):
        raise ValueError("complete unchanged physical histories required")
    summary = {
        name: float(sum(row[name] for row in rows))
        for name in (
            "receiver_net",
            "ball_net",
            "receiver_positive",
            "ball_positive",
            "receiver_absolute",
            "ball_absolute",
        )
    }
    clean(a.expected_head)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "complete.json",
        dict(
            status="RECEIVER_PROXIMITY_ATTRIBUTION_EXPLORATORY",
            source_head=a.expected_head,
            dataset_hash=a.dataset_hash,
            audit_hash=a.audit_hash,
            episodes=72,
            controls=18903,
            summary=summary,
            records=rows,
            kinematic_not_causal=True,
            ball_first_attribution_order=False,
            trained=False,
            promotion_authorized=False,
            hardware_authorized=False,
            goal_complete=False,
        ),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
