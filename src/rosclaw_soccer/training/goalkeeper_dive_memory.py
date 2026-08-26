"""Neural muscle memory distilled from the qualified balanced dive seed.

This is an imitation-learning bridge, not a promoted goalkeeper policy.  It
compresses the bilateral 29-DoF motion into a small phase-conditioned decoder
and then sends the decoded motion back through independent CPU MuJoCo.  The
derived checkpoint inherits the source dataset's non-commercial, train-only,
``SIM_ONLY`` boundary.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.goalkeeper_dive_option import (
    build_balanced_dive_imitation_seed,
    load_official_goalkeeper_dive_atlas,
    mirror_g1_joint_positions,
    qualify_balanced_dive_seed_cpu_mujoco,
)


@dataclass(frozen=True)
class GoalkeeperDiveMemoryConfig:
    """Bounded supervised-learning contract for one dive decoder."""

    hidden_size: int = 128
    epochs: int = 3_000
    learning_rate: float = 2.0e-3
    velocity_loss_weight: float = 0.10
    acceleration_loss_weight: float = 0.02
    weight_decay: float = 1.0e-5
    random_seed: int = 4107
    maximum_position_rmse_rad: float = 0.035
    maximum_absolute_error_rad: float = 0.18
    maximum_velocity_rmse_rad_s: float = 0.80
    maximum_symmetry_rmse_rad: float = 0.020
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    commercial_use_allowed: bool = False
    schema_version: str = "rosclaw_soccer.goalkeeper_dive_memory_config.v1"

    def __post_init__(self) -> None:
        if not 32 <= self.hidden_size <= 512:
            raise ValueError("goalkeeper dive memory hidden size is invalid")
        if not 10 <= self.epochs <= 100_000:
            raise ValueError("goalkeeper dive memory epoch count is invalid")
        positive = (
            self.learning_rate,
            self.velocity_loss_weight,
            self.acceleration_loss_weight,
            self.maximum_position_rmse_rad,
            self.maximum_absolute_error_rad,
            self.maximum_velocity_rmse_rad_s,
            self.maximum_symmetry_rmse_rad,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("goalkeeper dive memory settings must be finite and positive")
        if not math.isfinite(self.weight_decay) or not 0.0 <= self.weight_decay <= 0.01:
            raise ValueError("goalkeeper dive memory weight decay is invalid")
        if (
            self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized
            or self.commercial_use_allowed
        ):
            raise ValueError("goalkeeper dive memory must remain non-commercial SIM_ONLY")

    @property
    def config_hash(self) -> str:
        return str(hash_json(asdict(self)))


def _build_decoder(torch: Any, nn: Any, *, hidden_size: int) -> Any:
    return nn.Sequential(
        nn.Linear(7, hidden_size),
        nn.SiLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.SiLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.SiLU(),
        nn.Linear(hidden_size, 29),
    )


def _phase_features(torch: Any, phase: Any, direction: Any) -> Any:
    pi = float(math.pi)
    return torch.stack(
        (
            direction,
            phase,
            phase.square(),
            torch.sin(pi * phase),
            torch.cos(pi * phase),
            torch.sin(2.0 * pi * phase),
            torch.cos(2.0 * pi * phase),
        ),
        dim=-1,
    )


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)


def decode_goalkeeper_dive_memory(
    *,
    checkpoint_path: Path,
    direction: Literal["left", "right"],
    phases: NDArray[np.float64],
    device: str = "cpu",
) -> NDArray[np.float64]:
    """Decode joint targets without granting execution or promotion authority."""

    import torch
    from torch import nn

    phase = np.asarray(phases, dtype=np.float64)
    if phase.ndim != 1 or not np.all(np.isfinite(phase)) or np.any((phase < 0) | (phase > 1)):
        raise ValueError("goalkeeper dive phases must be finite values in [0, 1]")
    if direction not in {"left", "right"}:
        raise ValueError("goalkeeper dive direction is invalid")
    checkpoint = torch.load(
        checkpoint_path.expanduser().resolve(),
        map_location=device,
        weights_only=True,
    )
    if (
        checkpoint.get("activation_ceiling") != "SIM_ONLY"
        or checkpoint.get("hardware_authorized") is not False
        or checkpoint.get("commercial_use_allowed") is not False
        or int(checkpoint.get("input_size", -1)) != 7
        or int(checkpoint.get("output_size", -1)) != 29
    ):
        raise ValueError("goalkeeper dive memory checkpoint boundary is invalid")
    model = _build_decoder(torch, nn, hidden_size=int(checkpoint["hidden_size"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    scale = checkpoint["target_scale"].to(device=device, dtype=torch.float32)
    phase_tensor = torch.as_tensor(phase, dtype=torch.float32, device=device)
    direction_value = -1.0 if direction == "left" else 1.0
    direction_tensor = torch.full_like(phase_tensor, direction_value)
    with torch.inference_mode():
        decoded = model(_phase_features(torch, phase_tensor, direction_tensor)) * scale
    return np.asarray(decoded.detach().cpu().numpy(), dtype=np.float64)


def train_goalkeeper_dive_memory(
    *,
    asset_root: Path,
    source_checkout: Path,
    output_dir: Path,
    config: GoalkeeperDiveMemoryConfig | None = None,
    device: str = "cuda:0",
) -> dict[str, Any]:
    """Distill, serialize, reload, and independently physics-check the decoder."""

    import torch
    from torch import nn

    active = config or GoalkeeperDiveMemoryConfig()
    torch.manual_seed(active.random_seed)
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for goalkeeper dive memory training")
        torch.cuda.manual_seed_all(active.random_seed)
    atlas = load_official_goalkeeper_dive_atlas(checkout=source_checkout)
    seed = build_balanced_dive_imitation_seed(atlas)
    destination = output_dir.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    target_np = np.asarray(seed.joint_position_rad, dtype=np.float32)
    frame_count = target_np.shape[1]
    phases_np = np.linspace(0.0, 1.0, frame_count, dtype=np.float32)
    phases = torch.as_tensor(
        np.concatenate((phases_np, phases_np)), dtype=torch.float32, device=device
    )
    directions = torch.cat(
        (
            -torch.ones(frame_count, dtype=torch.float32, device=device),
            torch.ones(frame_count, dtype=torch.float32, device=device),
        )
    )
    target = torch.as_tensor(target_np.reshape(-1, 29), dtype=torch.float32, device=device)
    target_scale = torch.clamp(torch.amax(torch.abs(target), dim=0), min=0.05)
    target_normalized = target / target_scale
    features = _phase_features(torch, phases, directions)
    model = _build_decoder(torch, nn, hidden_size=active.hidden_size).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=active.learning_rate, weight_decay=active.weight_decay
    )
    history: list[dict[str, float]] = []
    for epoch in range(active.epochs):
        prediction = model(features)
        position_loss = torch.mean((prediction - target_normalized).square())
        prediction_by_side = prediction.reshape(2, frame_count, 29)
        target_by_side = target_normalized.reshape(2, frame_count, 29)
        velocity_loss = torch.mean(
            (torch.diff(prediction_by_side, dim=1) - torch.diff(target_by_side, dim=1)).square()
        )
        acceleration_loss = torch.mean(
            (
                torch.diff(prediction_by_side, n=2, dim=1) - torch.diff(target_by_side, n=2, dim=1)
            ).square()
        )
        loss = (
            position_loss
            + active.velocity_loss_weight * velocity_loss
            + active.acceleration_loss_weight * acceleration_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if epoch == 0 or epoch == active.epochs - 1 or (epoch + 1) % 250 == 0:
            history.append(
                {
                    "epoch": float(epoch + 1),
                    "total_loss": float(loss.detach().cpu()),
                    "position_loss": float(position_loss.detach().cpu()),
                    "velocity_loss": float(velocity_loss.detach().cpu()),
                }
            )

    checkpoint_path = destination / "goalkeeper-dive-muscle-memory.pt"
    checkpoint = {
        "schema_version": "rosclaw_soccer.goalkeeper_dive_memory_checkpoint.v1",
        "model_state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "target_scale": target_scale.detach().cpu(),
        "input_size": 7,
        "output_size": 29,
        "hidden_size": active.hidden_size,
        "config": asdict(active),
        "config_hash": active.config_hash,
        "dive_seed_hash": seed.seed_hash,
        "source_atlas_hash": atlas.atlas_hash,
        "activation_ceiling": "SIM_ONLY",
        "hardware_authorized": False,
        "commercial_use_allowed": False,
    }
    torch.save(checkpoint, checkpoint_path)

    decoded = np.stack(
        (
            decode_goalkeeper_dive_memory(
                checkpoint_path=checkpoint_path,
                direction="left",
                phases=np.asarray(phases_np, dtype=np.float64),
            ),
            decode_goalkeeper_dive_memory(
                checkpoint_path=checkpoint_path,
                direction="right",
                phases=np.asarray(phases_np, dtype=np.float64),
            ),
        )
    )
    error = decoded - np.asarray(seed.joint_position_rad)
    velocity_error = np.diff(error, axis=1) * seed.frame_rate_hz
    symmetry_error = decoded[1] - mirror_g1_joint_positions(decoded[0])
    metrics = {
        "position_rmse_rad": float(np.sqrt(np.mean(np.square(error)))),
        "maximum_absolute_error_rad": float(np.max(np.abs(error))),
        "velocity_rmse_rad_s": float(np.sqrt(np.mean(np.square(velocity_error)))),
        "symmetry_rmse_rad": float(np.sqrt(np.mean(np.square(symmetry_error)))),
    }
    fit_passed = bool(
        metrics["position_rmse_rad"] <= active.maximum_position_rmse_rad
        and metrics["maximum_absolute_error_rad"] <= active.maximum_absolute_error_rad
        and metrics["velocity_rmse_rad_s"] <= active.maximum_velocity_rmse_rad_s
        and metrics["symmetry_rmse_rad"] <= active.maximum_symmetry_rmse_rad
    )
    qualification = qualify_balanced_dive_seed_cpu_mujoco(
        asset_root=asset_root,
        source_checkout=source_checkout,
        output_path=destination / "decoded-cpu-mujoco-qualification.json",
        joint_position_rad=decoded,
        trajectory_kind="neural_muscle_memory_decoded",
    )
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.goalkeeper_dive_memory_training.v1",
        "config": asdict(active),
        "config_hash": active.config_hash,
        "dive_seed_hash": seed.seed_hash,
        "source_atlas_hash": atlas.atlas_hash,
        "checkpoint": checkpoint_path.name,
        "checkpoint_hash": hash_bytes(checkpoint_path.read_bytes()),
        "decoded_trajectory_hash": hash_bytes(decoded.tobytes()),
        "training_device": str(device),
        "sample_count": int(target.shape[0]),
        "history": history,
        "metrics": metrics,
        "fit_gate_passed": fit_passed,
        "cpu_mujoco_qualification_hash": qualification["report_hash"],
        "cpu_mujoco_gate_passed": bool(qualification["passed"]),
        "passed": bool(fit_passed and qualification["passed"]),
        "status": (
            "TRAINING_OPTION_CANDIDATE"
            if fit_passed and qualification["passed"]
            else "REJECTED_DIVE_MEMORY"
        ),
        "policy_integration_completed": False,
        "authority": "TRAINING_REFERENCE_ONLY",
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
        "commercial_use_allowed": False,
    }
    report["report_hash"] = hash_json(report)
    _atomic_json(destination / "training-report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--source-checkout", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=3_000)
    parser.add_argument("--learning-rate", type=float, default=2.0e-3)
    args = parser.parse_args()
    report = train_goalkeeper_dive_memory(
        asset_root=args.asset_root,
        source_checkout=args.source_checkout,
        output_dir=args.output_dir,
        config=GoalkeeperDiveMemoryConfig(
            hidden_size=args.hidden_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
        ),
        device=args.device,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "GoalkeeperDiveMemoryConfig",
    "decode_goalkeeper_dive_memory",
    "train_goalkeeper_dive_memory",
]
