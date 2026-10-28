"""Small offline SIM demonstration fit, not receiving-skill qualification."""

import torch


def fit_receive_demonstrations(policy, observation, target, rehearsal):
    """Fit actual teacher actions with a fixed parent-output retention penalty."""
    if (
        observation.ndim != 2
        or observation.shape[1] != 177
        or target.shape != (len(observation), 29)
        or rehearsal.shape != (1200, 177)
        or len(observation) < 100
        or not all(torch.isfinite(x).all() for x in (observation, target, rehearsal))
    ):
        raise ValueError("finite complete native teacher and rehearsal samples required")
    optimizer = torch.optim.Adam(policy.actor.parameters(), lr=1e-4)
    with torch.no_grad():
        before = float((policy.actor(observation) - target).square().mean())
    final_retention = 0.0
    for step in range(200):
        indices = (torch.arange(256, device=observation.device) + step * 256) % 1200
        prediction = policy.actor(observation)
        with torch.no_grad():
            old = policy.baseline(rehearsal[indices])
        retention = (policy.actor(rehearsal[indices]) - old).square().mean()
        loss = (prediction - target).square().mean() + retention
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(policy.actor.parameters(), 1.0)
        if not torch.isfinite(loss) or not torch.isfinite(norm):
            raise ValueError("finite imitation loss and gradient required")
        optimizer.step()
        final_retention = float(retention.detach())
    with torch.no_grad():
        after = float((policy.actor(observation) - target).square().mean())
    if not after < before:
        raise ValueError("actual selected demonstration fit must improve")
    return dict(
        updates=200,
        teacher_rows=len(observation),
        teacher_mse_before=before,
        teacher_mse_after=after,
        final_rehearsal_mse=final_retention,
        ppo_optimizer_reused=False,
    )
