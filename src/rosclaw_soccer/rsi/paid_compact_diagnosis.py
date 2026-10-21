"""Own-policy support traces: distinguish fitting error from closed-loop drift.

Teacher same-index comparisons are trajectory diagnostics, not expert labels on
the student's state. No trajectory from this reader is silently admitted to fit.
"""

from typing import Any

import numpy as np
import torch

from rosclaw_soccer.rsi.paid_compact_student import FrozenCompactResidual


def compact_own_trace(actor: Any, raw: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if type(actor.residual) is not FrozenCompactResidual or actor.training:
        raise ValueError("verified frozen compact research actor required")
    n = len(raw["NN_action"])
    shapes = dict(
        NN_obs=(n, 160),
        NN_action=(n, 29),
        NN_h_before=(n, 2, 1, 128),
        NN_c_before=(n, 2, 1, 128),
        NN_h_after=(n, 2, 1, 128),
        NN_c_after=(n, 2, 1, 128),
    )
    if not 1 <= n <= 300 or any(
        raw[k].shape != shape or raw[k].dtype != np.float32 or not np.isfinite(raw[k]).all()
        for k, shape in shapes.items()
    ):
        raise ValueError("complete bounded own neural trajectory required")
    state = (torch.zeros(2, 1, 128), torch.zeros(2, 1, 128))
    features, gains, deltas = [], [], []
    with torch.no_grad():
        for i, row in enumerate(raw["NN_obs"]):
            if any(
                not np.array_equal(t.numpy(), raw[k][i])
                for t, k in zip(state, ("NN_h_before", "NN_c_before"), strict=True)
            ):
                raise ValueError("own recurrent before-state differs")
            obs = torch.from_numpy(row.copy()).reshape(1, 1, 160)
            nominal, state, hidden = actor.base.forward_with_features(obs, state)
            if any(
                not np.array_equal(t.numpy(), raw[k][i])
                for t, k in zip(state, ("NN_h_after", "NN_c_after"), strict=True)
            ):
                raise ValueError("own recurrent after-state differs")
            x = torch.cat([(obs - actor.base.mean) / actor.base.denominator, hidden], -1)
            x = x.reshape(1, 288)
            prior = actor.residual.parent(x)
            gain = actor.residual.head.support(x)
            delta = actor.residual.head.combine(x, prior, gain)
            action = nominal[0, 0] + delta[0] / actor.action_scale
            if not np.array_equal(action.numpy(), raw["NN_action"][i]):
                raise ValueError("own compact NN action differs from executed trajectory")
            features.append(x[0].numpy().copy())
            gains.append(gain[0].numpy().copy())
            deltas.append(delta[0].numpy().copy())
    return dict(features=np.stack(features), gain=np.stack(gains)[:, 0], delta=np.stack(deltas))


def summarize_compact_trace(
    trace: dict[str, np.ndarray],
    *,
    teacher_features: np.ndarray | None = None,
    teacher_delta: np.ndarray | None = None,
    first_foot_substep: int | None = None,
) -> dict[str, Any]:
    x, g, d = (trace[k] for k in ("features", "gain", "delta"))
    n = len(x)
    if (
        not 1 <= n <= 300
        or x.shape != (n, 288)
        or g.shape != (n,)
        or d.shape != (n, 29)
        or any(v.dtype != np.float32 or not np.isfinite(v).all() for v in (x, g, d))
        or (g < 0).any()
        or (g > 1).any()
        or (np.abs(d) > 0.2).any()
        or first_foot_substep is not None
        and (type(first_foot_substep) is not int or first_foot_substep < 0)
    ):
        raise ValueError("finite own288/support/final-delta trace and actual contact required")

    def first(mask: np.ndarray) -> int | None:
        ids = np.flatnonzero(mask)
        return int(ids[0]) if len(ids) else None

    contact_control = first_foot_substep // 10 if first_foot_substep is not None else None
    report = dict(
        rows=n,
        support_positive_rows=int((g > 0).sum()),
        support_at_least_half_rows=int((g >= 0.5).sum()),
        first_support_below_half_control=first(g < 0.5),
        first_zero_support_control=first(g == 0),
        first_nonzero_final_residual_control=first(np.abs(d).max(axis=1) > 1e-6),
        support_quantiles=np.quantile(g, [0, 0.1, 0.5, 0.9, 1]).tolist(),
        first_contact_control=contact_control,
        support_at_first_foot_contact=float(g[contact_control])
        if contact_control is not None and contact_control < n
        else None,
        training_labels_created=False,
        physical_replay_performed=False,
    )
    if (teacher_features is None) != (teacher_delta is None):
        raise ValueError("paired teacher features and desired actions required")
    if teacher_features is not None and teacher_delta is not None:
        if (
            teacher_features.ndim != 2
            or teacher_features.shape[1] != 288
            or not 1 <= len(teacher_features) <= 300
            or teacher_delta.shape != (len(teacher_features), 29)
            or any(
                v.dtype != np.float32 or not np.isfinite(v).all()
                for v in (teacher_features, teacher_delta)
            )
        ):
            raise ValueError("bounded own-causal TRAIN teacher required")
        common = min(n, len(teacher_features))
        distance = np.linalg.norm(
            x[:common].astype(np.float64) - teacher_features[:common].astype(np.float64), axis=1
        )
        error = np.abs(d[:common].astype(np.float64) - teacher_delta[:common].astype(np.float64))
        report["same_index_teacher_comparison"] = dict(
            is_expert_label_on_student_state=False,
            own_horizon_rows=n,
            teacher_horizon_rows=len(teacher_features),
            compared_prefix_rows=common,
            unobserved_tail_extrapolated=False,
            first_feature_difference_control=first(
                np.any(x[:common] != teacher_features[:common], axis=1)
            ),
            own_feature_distance_quantiles=np.quantile(distance, [0, 0.5, 0.9, 1]).tolist(),
            maximum_joint_action_difference_rad=float(error.max()),
            maximum_joint_action_difference_control=int(error.max(axis=1).argmax()),
            mean_squared_action_difference_rad2=float(np.square(error).mean()),
        )
    return report
