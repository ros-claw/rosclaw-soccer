"""Cross-fit a SIM-only option predictor on authenticated paired physical traces."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_native_context_recovery_common import check_pins, digest
from scripts.rsi_native_fresh_common import clean, source_pins

OPTIONS = ("parent", "retrieval", "outcome-model", "experimental-1.5", "feedback-1.5")


def make_model():
    return torch.nn.Sequential(
        torch.nn.Linear(3, 32),
        torch.nn.Tanh(),
        torch.nn.Linear(32, 32),
        torch.nn.Tanh(),
        torch.nn.Linear(32, 10),
    )


def choose(logits):
    if logits.shape != (5, 2) or not np.isfinite(logits).all():
        raise ValueError("finite five-option success/unsafe scores required")
    return int(np.argmax(logits[:, 0] - logits[:, 1]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("ranker", "ranker-hash", "feedback", "feedback-hash"):
        p.add_argument("--" + key, action="append", required=True)
    for key in ("protocol", "output", "core-source"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("protocol-hash", "expected-head"):
        p.add_argument("--" + key, required=True)
    a = p.parse_args()
    clean(a.expected_head)
    if a.output.exists() or a.output.resolve().is_relative_to(Path.cwd()):
        raise ValueError("unused external output required")
    if any(len(x) != 4 for x in (a.ranker, a.ranker_hash, a.feedback, a.feedback_hash)):
        raise ValueError("four complete shards of each study required")
    if a.protocol.is_symlink() or digest(a.protocol) != a.protocol_hash:
        raise ValueError("exact declared protocol required")
    pins = {str(a.protocol): a.protocol_hash}
    ranks, feedback = [], []
    for roots, hashes, destination in (
        (a.ranker, a.ranker_hash, ranks),
        (a.feedback, a.feedback_hash, feedback),
    ):
        for root, h in zip(roots, hashes, strict=True):
            path = Path(root) / "complete.json"
            if path.is_symlink() or digest(path) != h:
                raise ValueError("exact source receipt required")
            pins[str(path)] = h
            destination.append((Path(root), json.loads(path.read_text())))
    source_pins(pins)
    core = a.core_source.resolve()
    if not (core / "rosclaw" / "__init__.py").is_file():
        raise ValueError("explicit ROSClaw source required")
    pins.update({str(path): digest(path) for path in (core / "rosclaw").rglob("*.py")})
    check_pins(pins)
    from rosclaw_soccer.rsi.walk_kick_teacher_audit import relabel_arrays
    from rosclaw_soccer.sim.contracts import hash_json

    data = []
    for i in range(40):
        shard = i % 4
        records = []
        for option in OPTIONS:
            root, report = ranks[shard] if option in OPTIONS[:3] else feedback[shard]
            if option in OPTIONS[:3]:
                matches = [
                    r for r in report["rows"] if r["course_index"] == i and r["arm"] == option
                ]
                name, hash_key = f"course-{i:02d}-{option}.npz", "raw_hash"
            else:
                matches = [
                    r
                    for r in report["rows"]
                    if r["course"] == i and r["arm"] == option and r["repeat"] == 0
                ]
                name, hash_key = f"course-{i:02d}-{option}-0.npz", "artifact_hash"
            if len(matches) != 1:
                raise ValueError("one qualified option per paired course required")
            row = matches[0]
            path = root / name
            if path.is_symlink() or digest(path) != row[hash_key]:
                raise ValueError("exact raw physical trajectory required")
            pins[str(path)] = row[hash_key]
            result = row["result"]
            with np.load(path, allow_pickle=False) as trace:
                label = relabel_arrays(
                    trace,
                    controls=result["controls"],
                    ball_address=result["ball_qpos_address"],
                    root_address=result["robot_qpos_address"],
                )
            if hash_json(label) != hash_json(result["outcome"]):
                raise ValueError("independently recomputed physical label mismatch")
            records.append((result["course"], label, row[hash_key]))
        if any(r[0] != records[0][0] for r in records):
            raise ValueError("options must share exact launch course")
        data.append(records)
    x = np.asarray([r[0][0] for r in data], dtype=np.float32)
    y = np.asarray(
        [[[float(v[1]["task_pass"]), float(not v[1]["safe"])] for v in row] for row in data],
        dtype=np.float32,
    )
    if x.shape != (40, 3) or y.shape != (40, 5, 2) or not np.isfinite(x).all():
        raise ValueError("finite paired physical dataset required")
    a.output.mkdir(parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            protocol_hash=a.protocol_hash,
            input_file_hashes=pins,
            options=OPTIONS,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    rows, folds = [], []
    try:
        for fold in range(5):
            train = np.array([i for i in range(40) if i % 5 != fold])
            test = np.array([i for i in range(40) if i % 5 == fold])
            mean = x[train].mean(axis=0)
            scale = x[train].std(axis=0)
            if np.any(scale <= 0):
                raise ValueError("nondegenerate training-only normalization required")
            torch.manual_seed(907)
            model = make_model()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
            tx = torch.tensor((x[train] - mean) / scale)
            ty = torch.tensor(y[train].reshape(-1, 10))
            for _ in range(400):
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(model(tx), ty)
                loss.backward()
                optimizer.step()
            path = a.output / f"fold-{fold}.pt"
            torch.save(
                dict(
                    state_dict=model.state_dict(),
                    mean=torch.from_numpy(mean),
                    scale=torch.from_numpy(scale),
                    train=train.tolist(),
                    test=test.tolist(),
                    steps=400,
                ),
                path,
            )
            with torch.no_grad():
                scores = (
                    model(torch.tensor((x[test] - mean) / scale))
                    .sigmoid()
                    .numpy()
                    .reshape(-1, 5, 2)
                )
            for idx, score in zip(test, scores, strict=True):
                selected = choose(score)
                rows.append(
                    dict(
                        course=int(idx),
                        fold=fold,
                        launch=x[idx].tolist(),
                        probabilities=score.tolist(),
                        selected=OPTIONS[selected],
                        actual_outcomes=[v[1] for v in data[idx]],
                        selected_outcome=data[idx][selected][1],
                        source_hashes=[v[2] for v in data[idx]],
                    )
                )
            folds.append(
                dict(
                    fold=fold,
                    train=train.tolist(),
                    test=test.tolist(),
                    weights_hash=digest(path),
                    steps=400,
                    final_loss=float(loss.detach()),
                )
            )
        clean(a.expected_head)
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="CROSSFIT_OPTION_PREDICTION_NOT_PHYSICAL_POLICY",
                source_head=a.expected_head,
                protocol_hash=a.protocol_hash,
                folds=folds,
                rows=sorted(rows, key=lambda r: r["course"]),
                promotion_authorized=False,
                hardware_authorized=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failed.json", dict(error_type=type(error).__name__, error=str(error))
        )
        raise


if __name__ == "__main__":
    main()
