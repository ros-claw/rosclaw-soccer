"""Offline SIM diagnostics, not actuator sensitivity or policy authorization."""

from __future__ import annotations

from typing import Any

import numpy as np

from rosclaw_soccer.rsi.sampling_credit_diagnostics import sampling_credit_inventory


def sampling_action_opportunity(
    records: list[dict[str, Any]],
    gates: Any,
    latent_actions: Any,
    *,
    expected_episodes: int,
) -> dict[str, Any]:
    """Inspect complete training gates and latent tanh range without fitting.

    Outcome strata are retrospective and never actor inputs. A latent tanh
    derivative is NOT the full joint-target/torque/dynamics Jacobian. This does
    not diagnose downstream clipping, contact feasibility, or causal failure.
    """
    if type(expected_episodes) is not int or not 1 <= expected_episodes <= 4096:
        raise ValueError("bounded complete episode count required")
    g = np.asarray(gates)
    z = np.asarray(latent_actions)
    if (
        g.dtype != np.float64
        or z.dtype != np.float64
        or g.shape != (expected_episodes, 270)
        or z.shape != (expected_episodes, 270, 12)
        or not np.isfinite(g).all()
        or not np.isfinite(z).all()
        or np.any(g < 0)
        or np.any(g > 1)
        or np.max(np.abs(z)) > 1e6
    ):
        raise ValueError("complete finite typed gates and bounded latent actions required")
    # Existing complete order/outcome/reward checks; no success-only filtering.
    credit = sampling_credit_inventory(
        records, np.zeros(g.shape, dtype=np.float64), expected_episodes=expected_episodes
    )
    derivative = 1 - np.tanh(z) ** 2

    def describe(mask: Any) -> dict[str, Any]:
        selected_g, selected_z, selected_d = g[mask], z[mask], derivative[mask]
        frames = selected_g.size
        coordinates = selected_z.size
        return dict(
            frame_samples=frames,
            latent_coordinates=coordinates,
            zero_gate_frames=int(np.sum(selected_g == 0)),
            gate_below_0_01_frames=int(np.sum(selected_g < 0.01)),
            gate_below_0_1_frames=int(np.sum(selected_g < 0.1)),
            unit_gate_frames=int(np.sum(selected_g == 1)),
            minimum_gate=float(selected_g.min()) if frames else None,
            maximum_gate=float(selected_g.max()) if frames else None,
            mean_gate=float(selected_g.mean()) if frames else None,
            latent_abs_above_2_coordinates=int(np.sum(np.abs(selected_z) > 2)),
            minimum_latent_tanh_derivative=float(selected_d.min()) if coordinates else None,
            mean_latent_tanh_derivative=float(selected_d.mean()) if coordinates else None,
            latent_tanh_derivative_below_0_01_coordinates=int(np.sum(selected_d < 0.01)),
        )

    frames = np.arange(30, 300)
    masks = {
        name: np.zeros(g.shape, dtype=bool)
        for name in ("before_first_contact", "at_or_after_first_contact", "episode_without_contact")
    }
    for i, record in enumerate(records):
        first = record["outcome"]["first_contact_frame"]
        if first is None:
            masks["episode_without_contact"][i] = True
        else:
            masks["before_first_contact"][i] = frames < first
            masks["at_or_after_first_contact"][i] = frames >= first
    if not np.all(sum(mask.astype(int) for mask in masks.values()) == 1):
        raise ValueError("contact-time partition must cover every retained training frame")
    return dict(
        schema="soccer.rsi.offline_sampling_action_opportunity.v1",
        completed_episodes=expected_episodes,
        frame_samples=expected_episodes * 270,
        all_success_and_failure_rows_included=True,
        aggregate=describe(np.ones(g.shape, dtype=bool)),
        overlapping_outcome_strata={
            name: dict(episode_groups=row["episode_groups"], **describe(row["episode_groups"]))
            for name, row in credit["overlapping_outcome_strata"].items()
        },
        contact_time_strata={name: describe(mask) for name, mask in masks.items()},
        tanh_measurement_applies_to_saved_latent_actions_only=True,
        downstream_projection_sensitivity_measured=False,
        causal_failure_explanation_proven=False,
        new_optimizer_steps=0,
        new_physical_executions=0,
        private_fresh_accessed=False,
        runtime_selection_authorized=False,
        training_authorized=False,
        promotion_authorized=False,
        hardware_authorized=False,
        qualification="OFFLINE_ACTION_OPPORTUNITY_NOT_POLICY_GAIN",
    )
