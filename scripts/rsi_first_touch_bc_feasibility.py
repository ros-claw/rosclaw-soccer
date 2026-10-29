"""Offline group-separated first-touch imitation probe; never deploys a policy."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from rosclaw_soccer.rsi.walk_kick_teacher_audit import relabel_arrays
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_mjlab_stage1_training_smoke import file_digest
from scripts.rsi_native_fresh_common import clean


def causal_examples(raw):
    """Use preceding native state, never post-action state, for controls t>=1."""
    target = np.asarray(raw["joint_targets"])
    q = np.asarray(raw["qpos"])[9:-1:10]
    v = np.asarray(raw["qvel"])[9:-1:10]
    if target.ndim != 2 or target.shape[1] != 29 or q.shape != (len(target) - 1, 43):
        raise ValueError("incomplete aligned native controls")
    if v.shape != (len(target) - 1, 41):
        raise ValueError("incomplete aligned native velocities")
    robot = q[:, 7:43]
    robot_v = v[:, 6:41]
    ball_rel = q[:, :3] - robot[:, :3]
    ball_relative_velocity = v[:, :3] - robot_v[:, :3]
    x = np.concatenate(
        (robot[:, 2:36], robot_v, ball_rel, ball_relative_velocity, target[:-1]), axis=1
    )
    y = target[1:]
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("nonfinite actual state or target")
    return x, y


def kick_phase_examples(raw):
    """Teacher kick controls with a causal elapsed-since-switch clock."""
    x, y = causal_examples(raw)
    active = np.asarray(raw["velocity_control_active"])
    if active.shape != (len(x) + 1,) or active.dtype != np.bool_ or not active[0]:
        raise ValueError("valid measured velocity-to-kick phase required")
    transitions = np.flatnonzero(~active)
    if not len(transitions) or np.any(active[transitions[0] :]):
        raise ValueError("one observed kick transition without reset required")
    entry = int(transitions[0])
    indices = np.arange(entry, len(active))
    time_since_entry = ((indices - entry) * 0.02).reshape(-1, 1)
    return np.concatenate((x[indices - 1], time_since_entry), axis=1), y[indices - 1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, action="append", required=True)
    p.add_argument("--source-hash", action="append", required=True)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--protocol-hash", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--kick-phase-only", action="store_true")
    p.add_argument("--bounded-codec", type=Path)
    p.add_argument("--bounded-codec-hash")
    p.add_argument("--source-declaration-hash")
    a = p.parse_args()
    clean(a.expected_head)
    if len(a.source) != 4 or len(a.source_hash) != 4 or a.output.exists():
        raise ValueError("four pinned source shards and unused output required")
    if a.output.resolve().is_relative_to(Path.cwd()) or file_digest(a.protocol) != a.protocol_hash:
        raise ValueError("external output and exact protocol required")
    if bool(a.bounded_codec) != bool(a.bounded_codec_hash) or bool(a.bounded_codec) != bool(
        a.source_declaration_hash
    ):
        raise ValueError("bounded codec requires source-bound configuration")
    bounds = None
    if a.bounded_codec:
        import yaml

        from rosclaw_soccer.rsi.bounded_kick_decoder import kick_target_bounds

        if not a.kick_phase_only:
            raise ValueError("bounded kick codec requires kick-phase training")
        declaration = a.source[0] / "declaration.json"
        if (
            file_digest(declaration) != a.source_declaration_hash
            or file_digest(a.bounded_codec) != a.bounded_codec_hash
        ):
            raise ValueError("source declaration and codec hash mismatch")
        if (
            json.loads(declaration.read_text())["input_file_hashes"].get(str(a.bounded_codec))
            != a.bounded_codec_hash
        ):
            raise ValueError("codec is not bound to source declaration")
        bounds = kick_target_bounds(yaml.safe_load(a.bounded_codec.read_text()))
    rows, datasets, pins = [], [], []
    for shard, (root, expected) in enumerate(zip(a.source, a.source_hash, strict=True)):
        receipt = root / "complete.json"
        if file_digest(receipt) != expected:
            raise ValueError("source receipt mismatch")
        report = json.loads(receipt.read_text())
        if report.get("source_head") != "079fb106adf506f2be5d3f11ca4f794a882e8859":
            raise ValueError("source revision mismatch")
        if len(report["rows"]) != 20:
            raise ValueError("complete paired source shard required")
        for meta in report["rows"]:
            i, arm, result = meta["course_index"], meta["arm"], meta["result"]
            if i not in range(shard, 40, 4) or arm not in ("parent", "retrieval"):
                raise ValueError("unexpected paired source course")
            path = root / f"course-{i:02d}-{arm}.npz"
            if path.is_symlink() or file_digest(path) != meta["raw_hash"]:
                raise ValueError("native trajectory hash mismatch")
            pins.append((path, meta["raw_hash"]))
            with np.load(path, allow_pickle=False) as archive:
                raw = {key: archive[key] for key in archive.files}
            label = relabel_arrays(
                raw,
                controls=result["controls"],
                ball_address=result["ball_qpos_address"],
                root_address=result["robot_qpos_address"],
            )
            for key in ("safe", "task_pass", "first_foot_contact_frame"):
                if label[key] != result["outcome"][key]:
                    raise ValueError("native label mismatch")
            admitted = bool(label["safe"] and label["task_pass"] and result["healthy_full_horizon"])
            row = dict(
                course=i,
                arm=arm,
                admitted=admitted,
                split="validation" if i % 5 == 0 else "train",
                controls=result["controls"],
                label=label,
            )
            rows.append(row)
            if admitted:
                x, y = kick_phase_examples(raw) if a.kick_phase_only else causal_examples(raw)
                if bounds is not None and (
                    np.any(y < bounds[0] - 1e-6) or np.any(y > bounds[1] + 1e-6)
                ):
                    raise ValueError("teacher target violates declared source codec")
                datasets.append((row, x, y))
    if len({(r["course"], r["arm"]) for r in rows}) != 80:
        raise ValueError("duplicate or missing course/arm")
    train = [entry for entry in datasets if entry[0]["split"] == "train"]
    valid = [entry for entry in datasets if entry[0]["split"] == "validation"]
    if not train or not valid:
        raise ValueError("both course-group splits need admitted episodes")
    x = np.concatenate([r[1] for r in train])
    y = np.concatenate([r[2] for r in train])
    mean = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0), 1e-3)
    torch.manual_seed(24029)
    torch.set_num_threads(1)
    net = torch.nn.Sequential(
        torch.nn.Linear(x.shape[1], 128),
        torch.nn.Tanh(),
        torch.nn.Linear(128, 128),
        torch.nn.Tanh(),
        torch.nn.Linear(128, 29),
    )
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    if bounds is not None:
        from rosclaw_soccer.rsi.bounded_kick_decoder import decode_bounded

        lower = torch.tensor(bounds[0], dtype=torch.float32)
        upper = torch.tensor(bounds[1], dtype=torch.float32)

    def predict(inputs):
        logits = net(inputs)
        return decode_bounded(logits, lower, upper) if bounds is not None else logits

    tx = torch.tensor((x - mean) / scale, dtype=torch.float32)
    ty = torch.tensor(y, dtype=torch.float32)
    generator = torch.Generator().manual_seed(24029)
    for _ in range(1000):
        ix = torch.randint(len(tx), (256,), generator=generator)
        loss = torch.nn.functional.mse_loss(predict(tx[ix]), ty[ix])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    metrics = []
    with torch.no_grad():
        for row, features, targets in datasets:
            pred = predict(torch.tensor((features - mean) / scale, dtype=torch.float32)).numpy()
            if bounds is not None and (
                np.any(pred < bounds[0] - 1e-6) or np.any(pred > bounds[1] + 1e-6)
            ):
                raise ValueError("bounded student escaped its source codec")
            metrics.append(
                dict(
                    course=row["course"],
                    arm=row["arm"],
                    split=row["split"],
                    frames=len(targets),
                    model_mse=float(np.mean((pred - targets) ** 2)),
                    persistence_mse=float(
                        np.mean(
                            (
                                (features[:, -30:-1] if a.kick_phase_only else features[:, -29:])
                                - targets
                            )
                            ** 2
                        )
                    ),
                )
            )
    if (
        file_digest(a.protocol) != a.protocol_hash
        or any(file_digest(path) != h for path, h in pins)
        or (
            a.bounded_codec is not None
            and (
                file_digest(a.bounded_codec) != a.bounded_codec_hash
                or file_digest(a.source[0] / "declaration.json") != a.source_declaration_hash
            )
        )
        or any(
            file_digest(root / "complete.json") != h
            for root, h in zip(a.source, a.source_hash, strict=True)
        )
    ):
        raise ValueError("input changed during training")
    clean(a.expected_head)
    a.output.mkdir(parents=True)
    model = a.output / "offline-model.pt"
    torch.save(
        dict(
            state_dict=net.state_dict(),
            mean=torch.from_numpy(mean),
            scale=torch.from_numpy(scale),
            feature_dim=x.shape[1],
            **(
                {
                    "target_lower": torch.from_numpy(bounds[0]),
                    "target_upper": torch.from_numpy(bounds[1]),
                    "codec_hash": a.bounded_codec_hash,
                }
                if bounds is not None
                else {}
            ),
        ),
        model,
    )
    write_once(
        a.output / "complete.json",
        dict(
            status="OFFLINE_ONLY_NOT_EVALUATED_IN_PHYSICS",
            source_head=a.expected_head,
            source_hashes=a.source_hash,
            protocol_hash=a.protocol_hash,
            rows=rows,
            metrics=metrics,
            train_frames=len(x),
            kick_phase_only=a.kick_phase_only,
            bounded_codec_hash=a.bounded_codec_hash,
            optimizer_steps=1000,
            seed=24029,
            model_hash=file_digest(model),
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
