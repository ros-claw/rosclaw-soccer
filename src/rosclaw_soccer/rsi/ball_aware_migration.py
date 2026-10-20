"""Optional Torch: explicit zero-influence 168->177 actor and 295->304 critic.

No optimizer state is migrated. A new input ABI needs a fresh optimizer and
independent physics qualification, not an inherited policy certificate.
"""

import torch


def expand_ball_aware_state(
    state: dict[str, torch.Tensor], *, part: str
) -> dict[str, torch.Tensor]:
    if (
        part not in ("actor", "critic")
        or not state
        or any(
            not isinstance(v, torch.Tensor) or not torch.isfinite(v).all() for v in state.values()
        )
    ):
        raise ValueError("complete finite actual actor or critic state required")
    key, width = ("mlp.mlp.0.weight", 168) if part == "actor" else ("mlp.0.weight", 295)
    if key not in state or state[key].shape != (512, width):
        raise ValueError("exact declared parent architecture required")
    result = {k: v.detach().clone() for k, v in state.items()}
    result[key] = torch.cat((result[key], state[key].new_zeros((512, 9))), dim=1)
    if part == "critic":
        for name, fill in (("_mean", 0), ("_var", 1), ("_std", 1)):
            key = "obs_normalizer." + name
            if key not in result or result[key].shape != (1, 295):
                raise ValueError("exact parent critic normalizer required")
            result[key] = torch.cat((result[key], result[key].new_full((1, 9), fill)), dim=1)
    elif result.get("mlp.prior_mean", torch.empty(0)).shape != (1, 98) or result.get(
        "mlp.prior_denominator", torch.empty(0)
    ).shape != (1, 98):
        raise ValueError("unchanged original actor normalization required")
    return result
