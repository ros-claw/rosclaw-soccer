"""Opt-in value recalibration for a changed reward, before any policy plasticity.

Research hypothesis, NOT qualified by import or unit tests. Must demonstrate
bit-exact actor retention during actual on-policy warmup before use in PPO.
"""

from rsl_rl.algorithms import PPO

from scripts.rsi_mjlab_prior_rl import RehearsalPPO


class FootOnlyValueWarmupPPO(RehearsalPPO):
    def __init__(self, *args, value_warmup_iterations=16, **kwargs):
        if type(value_warmup_iterations) is not int or value_warmup_iterations != 16:
            raise ValueError("predeclared sixteen on-policy critic recalibration updates required")
        super().__init__(*args, **kwargs)
        if self.rnd is not None or self.is_multi_gpu:
            raise ValueError("initial diagnostic excludes RND and multi-GPU")
        self.value_warmup_iterations = value_warmup_iterations
        self.value_warmup_completed = 0

    def update(self):
        if self.value_warmup_completed >= self.value_warmup_iterations:
            return super().update()
        parameters = tuple(self.actor.parameters())
        flags = tuple(p.requires_grad for p in parameters)
        schedule = self.schedule
        try:
            for parameter in parameters:
                parameter.requires_grad_(False)
                # A retained gradient from a previous caller is not permission
                # for Adam to mutate the supposedly frozen actor.
                parameter.grad = None
            self.schedule = "fixed"  # Zero actor KL must not inflate the learning rate.
            # Deliberately no imitation/rehearsal optimizer step in this phase.
            losses = PPO.update(self)
            self.value_warmup_completed += 1
            losses["value_warmup_completed"] = float(self.value_warmup_completed)
            return losses
        finally:
            self.schedule = schedule
            for parameter, flag in zip(parameters, flags, strict=True):
                parameter.requires_grad_(flag)

    def save(self):
        state = super().save()
        state["foot_only_value_warmup"] = dict(
            required=self.value_warmup_iterations, completed=self.value_warmup_completed
        )
        return state

    def load(self, *args, **kwargs):
        raise ValueError("warmup resume is not qualified; transfer network weights explicitly")
