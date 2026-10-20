"""Experimental SIM-only recurrent29-action student, not an option classifier.

No motion reference, future body, course ID, teacher choice or reward enters
the actor. Targets must come from separately verified physical demonstrations.
Creating/training this model confers no control or promotion authority.
"""

from typing import Any

import numpy as np

FEATURE_WIDTH = 108
ACTION_WIDTH = 29
FEATURE_LAYOUT = (
    ("remaining_task_fraction", 0, 1),
    ("pelvis_height_m", 1, 2),
    ("root_orientation_wxyz", 2, 6),
    ("native_root_freejoint_velocity", 6, 12),
    ("canonical_hg_joint_position_rad", 12, 41),
    ("canonical_hg_joint_velocity_rad_s", 41, 70),
    ("ball_relative_position_root_frame_m", 70, 73),
    ("ball_relative_linear_velocity_root_frame_mps", 73, 76),
    ("world_positive_x_shot_direction_root_frame", 76, 79),
    ("previous_executed_isaac_action", 79, 108),
)


def motor_student_features(
    qpos: np.ndarray, qvel: np.ndarray, previous_action_isaac: np.ndarray, *, frame: int
) -> np.ndarray:
    """Current measured canonical state only; returns one owned float32 vector."""
    for array, shape in ((qpos, (43,)), (qvel, (41,)), (previous_action_isaac, (29,))):
        if (
            type(array) is not np.ndarray
            or array.shape != shape
            or array.dtype.kind != "f"
            or not np.isfinite(array).all()
        ):
            raise ValueError(
                "finite measured canonical floating state and previous executed action required"
            )
    if type(frame) is not int or not 0 <= frame < 300 or abs(np.linalg.norm(qpos[3:7]) - 1) > 1e-4:
        raise ValueError("actual H300 frame and unit root quaternion required")
    quat = qpos[3:7].astype(np.float64, copy=True)
    if quat[0] < 0:
        quat *= -1
    w, x, y, z = quat
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )
    feature = np.concatenate(
        (
            [(300 - frame) / 300, qpos[2]],
            quat,
            qvel[:6],
            qpos[7:36],
            qvel[6:35],
            rotation.T @ (qpos[36:39] - qpos[:3]),
            rotation.T @ (qvel[35:38] - qvel[:3]),
            rotation.T @ np.array([1.0, 0.0, 0.0]),
            previous_action_isaac,
        )
    ).astype(np.float32)
    if feature.shape != (FEATURE_WIDTH,) or not np.isfinite(feature).all():
        raise ValueError("finite representable108 observation required")
    return feature


def recorded_student_sequence(raw: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Build observations with lagged EXECUTED actions, never current labels.

    This adapter checks shape/finite data, not provenance or physical success.
    A separate native review/curation must authorize imitation targets.
    """
    q, v, actions = (
        raw[key] for key in ("canonical_qpos", "canonical_qvel", "neural_action_isaac")
    )
    if (
        q.shape != (300, 43)
        or v.shape != (300, 41)
        or actions.shape != (300, 29)
        or actions.dtype != np.float32
        or not np.isfinite(actions).all()
    ):
        raise ValueError("complete canonical H300 and recorded float32 action targets required")
    previous = np.zeros(29, dtype=np.float32)
    features = []
    for frame in range(300):
        features.append(motor_student_features(q[frame], v[frame], previous, frame=frame))
        previous = actions[frame].copy()
    return np.stack(features), actions.copy()


def create_motor_student(*, hidden_size: int = 128, action_limit: float = 8.0) -> Any:
    """Torch optional factory; direct29 ISAAC actions -> original native PD map.

    The scalar critic is an auxiliary value head, not evidence of online RL.
    Callers save/load state_dict tensors only, using weights_only=True on load.
    """
    if (
        type(hidden_size) is not int
        or not 32 <= hidden_size <= 512
        or type(action_limit) not in (int, float)
        or not np.isfinite(action_limit)
        or not 0 < action_limit <= 8
    ):
        raise ValueError("bounded recurrent architecture and action limit required")
    import torch
    from torch import nn

    class MotorStudent(nn.Module):
        observation_mean: torch.Tensor
        observation_std: torch.Tensor

        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("observation_mean", torch.zeros(FEATURE_WIDTH))
            self.register_buffer("observation_std", torch.ones(FEATURE_WIDTH))
            self.encoder = nn.Sequential(
                nn.Linear(FEATURE_WIDTH, hidden_size), nn.ELU(), nn.LayerNorm(hidden_size)
            )
            self.memory = nn.GRU(hidden_size, hidden_size, batch_first=True)
            self.actor = nn.Linear(hidden_size, ACTION_WIDTH)
            self.critic = nn.Linear(hidden_size, 1)
            nn.init.normal_(self.actor.weight, std=0.001)
            nn.init.zeros_(self.actor.bias)
            self.action_limit = float(action_limit)

        def forward(self, observation: Any, hidden: Any = None) -> tuple[Any, Any, Any]:
            if (
                observation.ndim != 3
                or observation.shape[-1] != FEATURE_WIDTH
                or observation.shape[0] < 1
                or not 1 <= observation.shape[1] <= 300
                or observation.dtype != torch.float32
            ):
                raise ValueError("batch,time,108 measured observations required")
            if not torch.isfinite(observation).all():
                raise ValueError("nonfinite motor observation")
            if (
                not torch.isfinite(self.observation_mean).all()
                or not torch.isfinite(self.observation_std).all()
                or (self.observation_std < 0.01).any()
            ):
                raise ValueError("finite well-conditioned observation normalization required")
            if hidden is not None and (
                hidden.shape != (1, observation.shape[0], hidden_size)
                or not torch.isfinite(hidden).all()
            ):
                raise ValueError("finite owned recurrent state required")
            normalized = ((observation - self.observation_mean) / self.observation_std).clamp(
                -10, 10
            )
            encoded = self.encoder(normalized)
            memory, next_hidden = self.memory(encoded, hidden)
            action = self.action_limit * torch.tanh(self.actor(memory) / self.action_limit)
            value = self.critic(memory).squeeze(-1)
            if not (
                torch.isfinite(action).all()
                and torch.isfinite(value).all()
                and torch.isfinite(next_hidden).all()
            ):
                raise ValueError("nonfinite motor network output; no fallback action synthesized")
            return action, value, next_hidden

    return MotorStudent()
