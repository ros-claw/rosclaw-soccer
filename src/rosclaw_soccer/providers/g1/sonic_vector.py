"""Independent batched SONIC histories and measured-state reference tracking."""

from __future__ import annotations

from typing import Any

from rosclaw_soccer.providers.g1.sonic_runup import (
    ISAACLAB_TO_MUJOCO,
    MUJOCO_TO_ISAACLAB,
    G1SonicRunupController,
    _sonic_control_parameters,
)
from rosclaw_soccer.providers.g1.sonic_torch import FrozenSonicG1Torch


class BatchedSonicTracker:
    """Frozen full-body foundation; references are inputs, never world writes."""

    def __init__(
        self,
        model: FrozenSonicG1Torch,
        reference: Any,
        *,
        native_velocity: bool = False,
        low_latency_legacy_encoder_layout: bool = False,
        capture_neural_observation: bool = False,
    ) -> None:
        import torch

        if type(native_velocity) is not bool:
            raise ValueError("explicit reference derivative contract required")
        if type(low_latency_legacy_encoder_layout) is not bool or (
            low_latency_legacy_encoder_layout and getattr(model, "variant", None) != "low_latency"
        ):
            raise ValueError("legacy encoder layout applies only to low-latency SONIC")
        self.low_latency_legacy_encoder_layout = low_latency_legacy_encoder_layout
        if type(capture_neural_observation) is not bool:
            raise ValueError("explicit neural observation capture flag required")
        self._capture_neural_observation = capture_neural_observation
        self._neural_observation: dict[str, Any] | None = None
        self._torch, self.model = torch, model
        self.reference = torch.as_tensor(
            reference, device=model.device, dtype=torch.float32
        ).clone()
        if (
            self.reference.ndim != 3
            or self.reference.shape[2] != 36
            or not 1 <= self.reference.shape[0] <= 4096
            or not bool(torch.isfinite(self.reference).all())
        ):
            raise ValueError("finite batched G1 references required")
        norms = torch.linalg.vector_norm(self.reference[:, :, 3:7], dim=2)
        if not bool(torch.all(torch.abs(norms - 1) < 1e-4)):
            raise ValueError("reference orientations must be normalized")
        self.count = self.reference.shape[0]
        self.stride = model.qualification.reference_stride
        if self.reference.shape[1] < 9 * self.stride + 2:
            raise ValueError("reference lacks a complete future window")
        self.native_velocity = native_velocity
        self.default = torch.as_tensor(
            G1SonicRunupController.default_angles, device=model.device, dtype=torch.float32
        )
        kp, kd, scale = _sonic_control_parameters(1.0, (1.0,) * 29)
        self.kp, self.kd, self.scale = [
            torch.as_tensor(v, device=model.device, dtype=torch.float32) for v in (kp, kd, scale)
        ]
        self._to_isaac = torch.as_tensor(MUJOCO_TO_ISAACLAB, device=model.device, dtype=torch.long)
        self._to_mujoco = torch.as_tensor(ISAACLAB_TO_MUJOCO, device=model.device, dtype=torch.long)
        self._history: list[tuple[Any, ...]] = []
        self._next_frame = 0
        self._pending_observation = False
        self.action = torch.zeros((self.count, 29), device=model.device)

    def _state(self, qpos: Any, qvel: Any) -> tuple[Any, Any]:
        torch = self._torch
        q, v = [
            torch.as_tensor(x, device=self.model.device, dtype=torch.float32) for x in (qpos, qvel)
        ]
        if (
            q.ndim != 2
            or v.ndim != 2
            or q.shape[0] != self.count
            or v.shape[0] != self.count
            or q.shape[1] < 36
            or v.shape[1] < 35
            or not bool(torch.isfinite(q).all() and torch.isfinite(v).all())
        ):
            raise ValueError("finite canonical G1 body state required")
        if not bool(torch.all(torch.abs(torch.linalg.vector_norm(q[:, 3:7], dim=1) - 1) < 1e-4)):
            raise ValueError("measured orientation must be normalized")
        return q, v

    def _entry(self, q: Any, v: Any) -> tuple[Any, ...]:
        torch = self._torch
        w, x, y, z = q[:, 3:7].unbind(1)
        gravity = torch.stack(
            (2 * (w * y - x * z), -2 * (w * x + y * z), 2 * (x * x + y * y) - 1), dim=1
        )
        return (
            v[:, 3:6].clone(),
            (q[:, 7:36] - self.default)[:, self._to_isaac],
            v[:, 6:35][:, self._to_isaac].clone(),
            self.action.clone(),
            gravity,
        )

    def reset(self, qpos: Any, qvel: Any) -> None:
        q, v = self._state(qpos, qvel)
        self.action.zero_()
        entry = self._entry(q, v)
        self._history = [tuple(x.clone() for x in entry) for _ in range(10)]
        self._next_frame = 0
        self._pending_observation = False
        self._neural_observation = None

    def neural_observation(self) -> dict[str, Any]:
        """Owned actual neural inputs/outputs; no motion or learning authority.

        Snapshot belongs to the last completed update. Raw actions use ISAAC
        order; targets use canonical MuJoCo order. Capture changes no layout.
        """
        if self._neural_observation is None:
            raise RuntimeError("neural observation capture requires an enabled completed update")
        return {key: value.clone() for key, value in self._neural_observation.items()}

    def refresh_unexecuted_reference(
        self, frame: int, reference: Any, *, unchanged_lookahead_frames: int = 10
    ) -> None:
        """Adopt an explicit replan without changing executed/current lookahead poses."""

        torch = self._torch
        if (
            type(frame) is not int
            or frame != self._next_frame
            or self._pending_observation
            or type(unchanged_lookahead_frames) is not int
            or not 1 <= unchanged_lookahead_frames <= 50
        ):
            raise RuntimeError("reference refresh requires the next unexecuted frame")
        proposed = torch.as_tensor(reference, device=self.model.device, dtype=torch.float32)
        if (
            proposed.shape != self.reference.shape
            or not bool(torch.isfinite(proposed).all())
            or not bool(
                torch.all(
                    torch.abs(torch.linalg.vector_norm(proposed[:, :, 3:7], dim=2) - 1) < 1e-4
                )
            )
            or not bool(
                torch.allclose(
                    proposed[:, : frame + unchanged_lookahead_frames],
                    self.reference[:, : frame + unchanged_lookahead_frames],
                    atol=1e-6,
                    rtol=0,
                )
            )
        ):
            raise ValueError("replan altered frozen or current-lookahead reference")
        self.reference = proposed.clone()

    def encoder_features(
        self, frame: int, qpos: Any, qvel: Any, *, phase_offsets_frames: Any = None
    ) -> Any:
        torch = self._torch
        q, _ = self._state(qpos, qvel)
        if (
            type(frame) is not int
            or frame < 0
            or frame + 9 * self.stride >= self.reference.shape[1]
        ):
            raise ValueError("reference frame outside complete lookahead")
        indices = frame + torch.arange(10, device=self.model.device) * self.stride
        if phase_offsets_frames is None:
            future = self.reference[:, indices]
        else:
            offsets = torch.as_tensor(
                phase_offsets_frames, device=self.model.device, dtype=torch.float32
            )
            if (
                offsets.shape != (self.count,)
                or not bool(torch.isfinite(offsets).all())
                or bool(torch.any(torch.abs(offsets) > 8.0))
            ):
                raise ValueError("finite bounded per-lane SONIC phase offsets required")
            sample_at = indices[None, :].to(torch.float32) + offsets[:, None]
            if bool(torch.any(sample_at < 0)) or bool(
                torch.any(sample_at + (1 if self.native_velocity else 0) >= self.reference.shape[1])
            ):
                raise ValueError("phase-shifted reference outside complete lookahead")
            lower = torch.floor(sample_at).to(torch.long)
            upper = torch.clamp(lower + 1, max=self.reference.shape[1] - 1)
            alpha = (sample_at - lower).unsqueeze(2)
            first = torch.gather(self.reference, 1, lower.unsqueeze(2).expand(-1, -1, 36))
            second = torch.gather(self.reference, 1, upper.unsqueeze(2).expand(-1, -1, 36))
            future = first + alpha * (second - first)
            orientation = future[:, :, 3:7]
            # Equivalent quaternion signs can otherwise cancel at interpolation.
            signed_second = torch.where(
                (first[:, :, 3:7] * second[:, :, 3:7]).sum(dim=2, keepdim=True) < 0,
                -second[:, :, 3:7],
                second[:, :, 3:7],
            )
            orientation = first[:, :, 3:7] + alpha * (signed_second - first[:, :, 3:7])
            future[:, :, 3:7] = torch.nn.functional.normalize(orientation, dim=2)
        positions = future[:, :, 7:36][:, :, self._to_isaac]
        if self.native_velocity:
            if phase_offsets_frames is None:
                start = torch.clamp(indices, max=self.reference.shape[1] - 2)
                velocities = (
                    self.reference[:, start + 1, 7:36] - self.reference[:, start, 7:36]
                ) / 0.02
            else:
                next_at = sample_at + 1
                next_lower = torch.floor(next_at).to(torch.long)
                next_upper = torch.clamp(next_lower + 1, max=self.reference.shape[1] - 1)
                next_alpha = (next_at - next_lower).unsqueeze(2)
                next_first = torch.gather(
                    self.reference, 1, next_lower.unsqueeze(2).expand(-1, -1, 36)
                )
                next_second = torch.gather(
                    self.reference, 1, next_upper.unsqueeze(2).expand(-1, -1, 36)
                )
                next_future = next_first + next_alpha * (next_second - next_first)
                velocities = (next_future[:, :, 7:36] - future[:, :, 7:36]) / 0.02
            velocities = velocities[:, :, self._to_isaac]
        else:
            velocities = torch.gradient(positions, spacing=(0.02 * self.stride,), dim=(1,))[0]
        current = q[:, 3:7].clone()
        if self.model.qualification.heading_normalized:
            w, x, y, z = current.unbind(1)
            yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            current = torch.stack(
                (
                    torch.cos(yaw / 2),
                    torch.zeros_like(yaw),
                    torch.zeros_like(yaw),
                    torch.sin(yaw / 2),
                ),
                dim=1,
            )
        current[:, 1:] *= -1
        a, b, c, d = current[:, None, :].unbind(2)
        e, f, g, h = future[:, :, 3:7].unbind(2)
        w = a * e - b * f - c * g - d * h
        x = a * f + b * e + c * h - d * g
        y = a * g - b * h + c * e + d * f
        z = a * h + b * g - c * f + d * e
        rotation = torch.stack(
            (
                1 - 2 * (y * y + z * z),
                2 * (x * y - w * z),
                2 * (x * y + w * z),
                1 - 2 * (x * x + z * z),
                2 * (x * z - w * y),
                2 * (y * z + w * x),
            ),
            dim=2,
        )
        features = torch.cat(
            (
                positions.reshape(self.count, 290),
                velocities.reshape(self.count, 290),
                rotation.reshape(self.count, 60),
            ),
            dim=1,
        )
        if self.low_latency_legacy_encoder_layout:
            # The deployed legacy controller placed orientation at 584:644,
            # while this ONNX graph selects 590:650 for its G1 branch.
            # Reproduce exactly the effective 54 values plus six zeros.
            features = torch.cat(
                (
                    features[:, :580],
                    features[:, 586:640],
                    torch.zeros((self.count, 6), device=self.model.device),
                ),
                dim=1,
            )
        return features

    def update(self, frame: int, qpos: Any, qvel: Any, *, phase_offsets_frames: Any = None) -> Any:
        if len(self._history) != 10 or self._pending_observation or frame != self._next_frame:
            raise RuntimeError("SONIC reset/update/observe sequence differs")
        torch = self._torch
        features = self.encoder_features(
            frame, qpos, qvel, phase_offsets_frames=phase_offsets_frames
        )
        token = self.model.encode_g1(features)
        history = [
            torch.stack([entry[i] for entry in self._history], dim=1).reshape(self.count, -1)
            for i in range(5)
        ]
        decoder_input = torch.cat((token, *history), dim=1)
        self.action = self.model.decode(decoder_input)
        if not bool(torch.isfinite(self.action).all()):
            raise FloatingPointError("nonfinite full-body SONIC action")
        self._next_frame += 1
        self._pending_observation = True
        target = self.default + self.action[:, self._to_mujoco] * self.scale
        if self._capture_neural_observation:
            self._neural_observation = {
                "encoder_features": features.detach().clone(),
                "latent_token": token.detach().clone(),
                "decoder_input": decoder_input.detach().clone(),
                "raw_action_isaac": self.action.detach().clone(),
                "joint_target_mujoco_rad": target.detach().clone(),
            }
        return target

    def observe(self, qpos: Any, qvel: Any) -> None:
        if not self._pending_observation:
            raise RuntimeError("observation requires a preceding policy update")
        q, v = self._state(qpos, qvel)
        self._history.pop(0)
        self._history.append(self._entry(q, v))
        self._pending_observation = False
