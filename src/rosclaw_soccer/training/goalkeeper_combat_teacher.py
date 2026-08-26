"""Pinned research-only Humanoid-Goalkeeper policy used as a motion teacher.

The upstream checkpoint is never a ROSClaw champion and never receives
hardware authority.  It proposes coordinated 29-DoF goalkeeper targets inside
simulation; ROSClaw still owns blending, physical guards, examination, and
promotion.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from rosclaw_soccer.sim.contracts import hash_bytes

_SOURCE_COMMIT = "976a81ff19b7306bafbe923d2890066b68a85271"
_LICENSE_HASH = "sha256:6c8cd1cdbe7accec4f63b6c3afb45ce0ffae9ed6abc0ca55acf5900b37970a82"
_CHECKPOINT_HASH = "sha256:7ecdedff5de6e30a0a4d11742561a9be6c94d8faeefc4701f3e8788381b67b14"
_CHECKPOINT_RELATIVE = Path("legged_gym/resources/weight/goalkeeper.pt")

OFFICIAL_GOALKEEPER_DEFAULT_QPOS = (
    -0.1,
    0.2,
    0.0,
    0.3,
    -0.2,
    -0.2,
    -0.1,
    -0.2,
    0.0,
    0.3,
    -0.2,
    0.2,
    0.0,
    0.0,
    0.0,
    0.0,
    0.5,
    0.0,
    1.2,
    0.0,
    0.0,
    0.0,
    0.0,
    -0.5,
    0.0,
    1.2,
    0.0,
    0.0,
    0.0,
)

# The upstream environment records this randomized-reset reference separately
# from the zero-action posture above.  Reusing it for cross-simulator reset
# avoids asking the actor to absorb an unrelated locomotion stance transient.
OFFICIAL_GOALKEEPER_INITIAL_QPOS = (
    -0.34930936,
    -0.03763366,
    -0.22198406,
    0.93093884,
    -0.50943524,
    -0.08583859,
    0.13749947,
    -0.44516975,
    -0.06791031,
    0.11570476,
    -0.17351833,
    0.34241587,
    -0.00869134,
    0.00670955,
    0.01293622,
    0.00395479,
    0.49003497,
    -0.00168978,
    1.2062242,
    -0.01060604,
    0.00490874,
    -0.00869134,
    0.00319979,
    -0.4975251,
    -0.00450607,
    1.20307243,
    0.00536893,
    0.0053766,
    0.00324437,
)

OFFICIAL_GOALKEEPER_KP = (
    (150.0, 150.0, 150.0, 300.0, 40.0, 40.0) * 2
    + (150.0, 150.0, 150.0)
    + (150.0, 150.0, 150.0, 150.0, 20.0, 20.0, 20.0) * 2
)
OFFICIAL_GOALKEEPER_KD = (
    (2.0, 2.0, 2.0, 4.0, 2.0, 2.0) * 2
    + (2.0, 2.0, 2.0)
    + (2.0, 2.0, 2.0, 2.0, 0.5, 0.5, 0.5) * 2
)

# Preserve locomotion ownership in the legs while allowing the position-
# conditioned goalkeeper teacher to express useful reaches with both arms.
OFFICIAL_GOALKEEPER_BLEND_GROUP_SCALE = (
    (0.25,) * 12 + (0.60,) * 3 + (2.0,) * 14
)


def load_official_goalkeeper_teacher(
    *,
    checkout: Path,
    checkpoint: Path,
    device: Any,
) -> tuple[Any, dict[str, Any]]:
    """Load only the frozen actor after verifying source, license, and bytes."""

    root = checkout.expanduser().resolve()
    weight = checkpoint.expanduser().resolve()
    if not root.is_dir() or weight != root / _CHECKPOINT_RELATIVE or not weight.is_file():
        raise ValueError("Humanoid-Goalkeeper teacher path is not the pinned release layout")
    commit = subprocess.run(
        ("git", "-C", str(root), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != _SOURCE_COMMIT:
        raise ValueError("Humanoid-Goalkeeper teacher checkout is not pinned")
    license_path = root / "LICENSE"
    if hash_bytes(license_path.read_bytes()) != _LICENSE_HASH:
        raise ValueError("Humanoid-Goalkeeper teacher license changed")
    if hash_bytes(weight.read_bytes()) != _CHECKPOINT_HASH:
        raise ValueError("Humanoid-Goalkeeper teacher checkpoint changed")

    import torch
    from torch import nn

    class FrozenGoalkeeperActor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.history_encoder = nn.Sequential(
                nn.Linear(960, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 16),
            )
            self.ball_estimator = nn.Sequential(
                nn.Linear(960, 128),
                nn.ReLU(),
                nn.Linear(128, 32),
                nn.ReLU(),
                nn.Linear(32, 6),
            )
            self.region_estimator = nn.Sequential(
                nn.Linear(960, 128),
                nn.ReLU(),
                nn.Linear(128, 32),
                nn.ReLU(),
                nn.Linear(32, 6),
            )
            self.actor = nn.Sequential(
                nn.Linear(119, 512),
                nn.ELU(),
                nn.Linear(512, 256),
                nn.ELU(),
                nn.Linear(256, 256),
                nn.ELU(),
                nn.Linear(256, 29),
            )

        def forward(self, observation_history: Any) -> Any:
            if observation_history.ndim != 2 or observation_history.shape[1] != 960:
                raise ValueError("Humanoid-Goalkeeper teacher history must have shape (N, 960)")
            history = self.history_encoder(observation_history)
            ball = self.ball_estimator(observation_history)
            region_logits = self.region_estimator(observation_history)
            region = torch.argmax(region_logits, dim=-1, keepdim=True).to(
                observation_history.dtype
            )
            actor_input = torch.cat(
                (observation_history[:, -96:], history, ball, region), dim=-1
            )
            return self.actor(actor_input)

    payload = torch.load(weight, map_location=device, weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("model_state_dict"), dict):
        raise ValueError("Humanoid-Goalkeeper teacher checkpoint payload changed")
    source = payload["model_state_dict"]
    prefixes = ("history_encoder.", "ball_estimator.", "region_estimator.", "actor.")
    selected = {name: value for name, value in source.items() if name.startswith(prefixes)}
    model = FrozenGoalkeeperActor().to(device)
    model.load_state_dict(selected, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    report = {
        "schema_version": "rosclaw_soccer.goalkeeper_combat_teacher.v1",
        "source_repository": "https://github.com/InternRobotics/Humanoid-Goalkeeper",
        "source_commit": commit,
        "source_license": "CC-BY-NC-SA-4.0",
        "source_license_hash": _LICENSE_HASH,
        "checkpoint_hash": _CHECKPOINT_HASH,
        "observation_history_shape": [10, 96],
        "action_size": 29,
        "role": "FROZEN_RESEARCH_TEACHER_ONLY",
        "champion_eligible": False,
        "commercial_use_allowed": False,
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
    }
    return model, report


def rotate_inverse_torch(*, torch: Any, quaternion_wxyz: Any, vector: Any) -> Any:
    """Rotate world-frame vectors into a MuJoCo free-joint body frame."""

    scalar = quaternion_wxyz[:, :1]
    axis = quaternion_wxyz[:, 1:]
    return (
        vector * (2.0 * scalar * scalar - 1.0)
        - 2.0 * scalar * torch.linalg.cross(axis, vector, dim=1)
        + 2.0 * axis * torch.sum(axis * vector, dim=1, keepdim=True)
    )


__all__ = [
    "OFFICIAL_GOALKEEPER_BLEND_GROUP_SCALE",
    "OFFICIAL_GOALKEEPER_DEFAULT_QPOS",
    "OFFICIAL_GOALKEEPER_INITIAL_QPOS",
    "OFFICIAL_GOALKEEPER_KD",
    "OFFICIAL_GOALKEEPER_KP",
    "load_official_goalkeeper_teacher",
    "rotate_inverse_torch",
]
