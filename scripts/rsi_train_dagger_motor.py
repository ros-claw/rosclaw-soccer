"""One fixed DAgger update after independent full40 neural/native data review."""

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("bank", "review", "parent-train", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    for key in (
        "expected-head",
        "expected-bank-hash",
        "expected-parent-training-hash",
        "expected-review-head",
        "review-unit",
    ):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--wait-for-review", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    args = parser.parse_args()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if (
        head != args.expected_head
        or args.output.exists()
        or subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"])
    ):
        raise ValueError("clean frozen DAgger source and unused output required")
    from scripts.rsi_collect_shadow_motor import wait_for_terminal_collection
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    wait_for_terminal_collection(args.review_unit, wait=args.wait_for_review)
    if (
        digest(args.bank / "complete.json") != args.expected_bank_hash
        or digest(args.parent_train / "complete.json") != args.expected_parent_training_hash
    ):
        raise ValueError("original bank/parent checkpoint identity differs")
    rd = json.loads((args.review / "declaration.json").read_text())
    rc = json.loads((args.review / "complete.json").read_text())
    bd = json.loads((args.bank / "declaration.json").read_text())
    bc = json.loads((args.bank / "complete.json").read_text())
    pc = json.loads((args.parent_train / "complete.json").read_text())
    pd = json.loads((args.parent_train / "declaration.json").read_text())
    if (
        rd["source_head"] != args.expected_review_head
        or rd["mode"] != "verify"
        or rd["bank_hash"] != args.expected_bank_hash
        or rd["training_hash"] != args.expected_parent_training_hash
        or rd["beta"] != 0.9
        or rc["status"] != "ALL40_MIXED_NN_AND_NATIVE_REPLAY_VERIFIED"
        or rc["native_steps"] != 120000
        or rc["optimizer_steps"] != 0
        or len(rc["results"]) != 40
        or pc["optimizer_steps"] != 3000
        or pd["seed"] != 17
        or pc["bank_complete_hash"] != args.expected_bank_hash
        or bc["total_courses"] != 40
        or bc["train_courses"] != 32
        or bc["validation_courses"] != 8
        or digest(args.bank / bc["artifact"]) != bc["artifact_hash"]
        or digest(args.parent_train / pc["checkpoint"]) != pc["checkpoint_hash"]
    ):
        raise ValueError("complete verified same-state bank and fixed seed17 parent required")
    pins = dict(rd["input_file_hashes"])
    for root in (args.review, args.bank, args.parent_train):
        pins.update({str(p): digest(p) for p in root.rglob("*") if p.is_file()})
    check_pins(pins)
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(p): digest(p) for p in folder.rglob("*.py")})
    protocol = Path.cwd() / "docs/rsi/dagger4500-preregister-20261020.zh-CN.md"
    pins[str(protocol)] = digest(protocol)
    import numpy as np

    from rosclaw_soccer.rsi.dagger_motor_data import (
        same_state_actor_mask,
        same_state_teacher_sequence,
    )
    from scripts.rsi_atomic_artifacts import write_once

    with np.load(args.bank / bc["artifact"], allow_pickle=False) as z:
        old_x, old_y, train, old_actor = (
            z[k].copy() for k in ("features", "targets", "train", "actor_eligible")
        )
    new_actor = same_state_actor_mask(rc["results"], bd["rows"])
    if (
        train.dtype != np.bool_
        or train.shape != (40,)
        or int(train.sum()) != 32
        or not np.any(new_actor & train)
    ):
        raise ValueError("unchanged32/8 split and qualified new training episodes required")
    if not np.array_equal(train, np.asarray([r["split"] == "train" for r in bd["rows"]])):
        raise ValueError("whole-course train mask differs from original identity declaration")
    xs, ys = [], []
    for index, result in enumerate(rc["results"]):
        path = args.review / f"course{index:02d}" / "trace.npz"
        if digest(path) != result["artifact_hash"]:
            raise ValueError("reviewed same-state raw bytes differ")
        with np.load(path, allow_pickle=False) as z:
            x, y = same_state_teacher_sequence({k: z[k].copy() for k in z.files})
        if new_actor[index] and abs(y).max() >= 8:
            raise ValueError("qualified labels outside fixed actor range, no clipping")
        xs.append(x)
        ys.append(y)
    new_x, new_y = np.stack(xs), np.stack(ys)
    novel = int(
        np.count_nonzero(np.any(new_x[train & new_actor] != old_x[train & new_actor], axis=2))
    )
    if novel == 0:
        raise ValueError("no new actual-state supervision, refuse a nominal-only epoch repeat")
    import torch

    from rosclaw_soccer.rsi.neural_motor_student import create_motor_student

    torch.set_num_threads(1)
    torch.manual_seed(17)
    torch.use_deterministic_algorithms(True)
    if args.device.startswith("cuda"):
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
            raise ValueError("deterministic CUDA workspace required")
        torch.cuda.set_per_process_memory_fraction(0.10)
    device = torch.device(args.device)
    payload = torch.load(
        args.parent_train / pc["checkpoint"], map_location="cpu", weights_only=True
    )
    if (
        payload["optimizer_steps"] != 3000
        or payload["seed"] != 17
        or payload["bank_hash"] != args.expected_bank_hash
    ):
        raise ValueError("fixed parent tensor provenance differs")
    model = create_motor_student().to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    initial = {k: p.detach().cpu().clone() for k, p in model.named_parameters()}
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        args.output / "dagger-bank.npz",
        original_features=old_x,
        original_targets=old_y,
        same_state_features=new_x,
        same_state_targets=new_y,
        train=train,
        original_actor=old_actor,
        same_state_actor=new_actor,
    )
    write_once(
        args.output / "declaration.json",
        dict(
            schema="soccer.rsi.recurrent_motor_dagger.v1",
            source_head=head,
            input_file_hashes=pins,
            training_protocol="dagger4500",
            bank_complete_hash=args.expected_bank_hash,
            parent_training_hash=args.expected_parent_training_hash,
            reviewed_data_hash=digest(args.review / "complete.json"),
            seed=17,
            additional_optimizer_steps=1500,
            previous_optimizer_steps=3000,
            train_actor_original=int((train & old_actor).sum()),
            train_actor_same_state=int((train & new_actor).sum()),
            novel_train_state_frames=novel,
            observation_normalizer="unchanged_parent_training_only",
            original_and_same_state_actor_weight=[0.5, 0.5],
            learning_rate=0.0001,
            auxiliary_value_weight=0.1,
            dagger_bank_hash=digest(args.output / "dagger-bank.npz"),
            online_reinforcement_learning=False,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )

    def checkpoint(name, steps):
        saved = dict(
            state_dict={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            seed=17,
            optimizer_steps=steps,
            hidden_size=128,
            action_limit=8.0,
            source_head=head,
            bank_hash=args.expected_bank_hash,
            training_protocol="dagger4500",
            parent_training_hash=args.expected_parent_training_hash,
        )
        with tempfile.NamedTemporaryFile(dir=args.output, suffix=".tmp", delete=False) as stream:
            tmp = Path(stream.name)
            try:
                torch.save(saved, stream)
                stream.flush()
                os.fsync(stream.fileno())
                os.link(tmp, args.output / name)
            finally:
                tmp.unlink(missing_ok=True)

    checkpoint("initial.pt", 3000)

    def tensor(a):
        return torch.from_numpy(a).to(device)

    ox, oy, nx, ny = (tensor(a[train]) for a in (old_x, old_y, new_x, new_y))
    om, nm = (tensor(a[train]) for a in (old_actor, new_actor))
    discount = torch.pow(
        torch.tensor(0.99, device=device), torch.arange(299, -1, -1, device=device)
    )
    ov, nv = (
        tensor(a[train].astype(np.float32))[:, None] * discount[None]
        for a in (old_actor, new_actor)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    final = {}
    for step in range(1, 1501):
        optimizer.zero_grad(set_to_none=True)
        oa, value_old, _ = model(ox)
        na, value_new, _ = model(nx)
        anchor_loss = (oa[om] - oy[om]).square().mean()
        same_state_loss = (na[nm] - ny[nm]).square().mean()
        value_loss = 0.5 * ((value_old - ov).square().mean() + (value_new - nv).square().mean())
        loss = 0.5 * (anchor_loss + same_state_loss) + 0.1 * value_loss
        if not torch.isfinite(loss):
            raise ValueError("nonfinite DAgger loss")
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError("missing/nonfinite DAgger gradient")
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 100 == 0:
            with torch.no_grad():
                valid_old, _, _ = model(tensor(old_x[~train]))
                valid_new, _, _ = model(tensor(new_x[~train]))

                def validation(predicted, labels, mask):
                    m = tensor(mask[~train])
                    return (
                        float((predicted[m] - tensor(labels[~train])[m]).square().mean().cpu())
                        if m.any()
                        else None
                    )

                final = dict(
                    additional_step=step,
                    total_step=3000 + step,
                    anchor_actor_mse=float(anchor_loss.detach().cpu()),
                    same_state_actor_mse=float(same_state_loss.detach().cpu()),
                    auxiliary_value_mse=float(value_loss.detach().cpu()),
                    validation_anchor_mse=validation(valid_old, old_y, old_actor),
                    validation_same_state_mse=validation(valid_new, new_y, new_actor),
                    gradient_norm=float(gradient.detach().cpu()),
                )
            write_once(args.output / f"step{3000 + step:04d}.json", final)
            print("DAGGER_MOTOR_OPTIMIZER", json.dumps(final, sort_keys=True), flush=True)
        if step % 500 == 0:
            checkpoint(f"step{3000 + step:04d}.pt", 3000 + step)
    change = sum(
        float((p.detach().cpu() - initial[n]).square().sum()) for n, p in model.named_parameters()
    )
    if change <= 0:
        raise ValueError("no actual DAgger parameter change")
    check_pins(pins)
    write_once(
        args.output / "complete.json",
        dict(
            **final,
            training_protocol="dagger4500",
            optimizer_steps=4500,
            additional_optimizer_steps=1500,
            parameter_change_l2_squared=change,
            checkpoint="step4500.pt",
            checkpoint_hash=digest(args.output / "step4500.pt"),
            initial_checkpoint_hash=digest(args.output / "initial.pt"),
            source_head=head,
            bank_complete_hash=args.expected_bank_hash,
            parent_training_hash=args.expected_parent_training_hash,
            reviewed_data_hash=digest(args.review / "complete.json"),
            physical_rollouts=0,
            closed_loop_skill_verified=False,
            online_reinforcement_learning=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
